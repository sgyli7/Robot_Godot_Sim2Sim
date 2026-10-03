#!/usr/bin/env python3
"""Serve the frozen, full five-graph static profile on IPv4 loopback only.

Requests contain RGB pixels and measured joints, never task-object truth. This
service grants no physical qualification. Decoder tests can import this module
without importing ONNX Runtime or loading model weights.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from unitree_g1_policy_prepare import PROFILES, sha256


MAX_REQUEST_BYTES = 2_097_152
MAX_REPLY_BYTES = 1_048_576
MAX_FLOAT32 = float(np.finfo(np.float32).max)
GROUP_WIDTHS = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3}
STAMP_KEYS = {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"}
FULL_GRAPH_NAMES = {"preprocess_video", "preprocess_state", "backbone", "action_head", "decode_action"}
EXPORT_PATH = "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(payload: bytes):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)


def exact_keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} fields do not match the fixed protocol")


def decode_request(body: dict):
    exact_keys(body, {"schema", "profile", "sequence_id", "observation", "camera_rgb_b64", "state_groups"}, "Request")
    if body["schema"] != "unitree_g1_static_observation_v2" or body["profile"] != "static_apple":
        raise ValueError("Only the fixed static_apple profile is supported")
    stamp = body["observation"]
    exact_keys(stamp, STAMP_KEYS, "Observation stamp")
    for value in [body["sequence_id"], *stamp.values()]:
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError("Stamp and sequence fields must be unsigned 64-bit integers")
    if type(body["camera_rgb_b64"]) is not str or len(body["camera_rgb_b64"]) != 640 * 480 * 4:
        raise ValueError("Camera must encode exactly 640x480 RGB bytes")
    raw = base64.b64decode(body["camera_rgb_b64"], validate=True)
    if len(raw) != 640 * 480 * 3:
        raise ValueError("Camera must have exactly 640x480 RGB pixels")
    exact_keys(body["state_groups"], GROUP_WIDTHS, "Measured state")
    observation = {"ego_view": np.frombuffer(raw, dtype=np.uint8).reshape(1, 480, 640, 3).astype(np.float32)}
    for name, width in GROUP_WIDTHS.items():
        values = body["state_groups"][name]
        if type(values) is not list or len(values) != width:
            raise ValueError(f"Invalid measured joint array: {name}")
        for value in values:
            if type(value) not in (int, float) or abs(value) > MAX_FLOAT32 or not math.isfinite(value):
                raise ValueError(f"Invalid measured joint value: {name}")
        observation[name] = np.asarray(values, dtype=np.float32).reshape(1, width)
    return observation


def action_chunk(body: dict, outputs: dict) -> dict:
    exact_keys(outputs, {*GROUP_WIDTHS, "base_height_command", "navigate_command"}, "Model output")
    widths = {**GROUP_WIDTHS, "base_height_command": 1, "navigate_command": 3}
    for name, width in widths.items():
        value = outputs[name]
        if type(value) is not np.ndarray or value.shape != (1, 40, width) or value.dtype != np.float32:
            raise ValueError(f"Wrong model output shape/type: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite model output: {name}")
    frames = []
    for index in range(40):
        frame = {name: outputs[name][0, index].tolist() for name in GROUP_WIDTHS}
        frame["base_height_m"] = float(outputs["base_height_command"][0, index, 0])
        frame["navigate_mps_rps"] = outputs["navigate_command"][0, index].tolist()
        frames.append(frame)
    return {"profile": "static_apple", "observation": dict(body["observation"]),
            "sequence_id": body["sequence_id"], "model_revision": PROFILES["static_apple"]["revision"],
            "action_period_ns": 20_000_000, "frames": frames}


def verify_receipt(path: Path) -> Path:
    receipt = parse_json(path.read_bytes())
    if receipt["schema"] != "unitree_g1_policy_files_v1" or receipt["profile"] != "static_apple" or receipt["revision"] != PROFILES["static_apple"]["revision"]:
        raise ValueError("Wrong policy receipt")
    if receipt["local_files_verified"] is not True or receipt["artifact_contract"]["status"] != "interface_matched":
        raise ValueError("Policy files/interface have not been verified")
    root = Path(receipt["model_root"]).resolve()
    entries = receipt["files"]
    if type(entries) is not list or not entries:
        raise ValueError("Empty policy file receipt")
    paths = set()
    for entry in entries:
        relative = entry["path"]
        if type(relative) is not str or Path(relative).is_absolute() or relative in paths:
            raise ValueError("Invalid or duplicate policy file path")
        paths.add(relative)
        item = (root / relative).resolve()
        if not item.is_relative_to(root) or not item.is_file() or sha256(item) != entry["sha256"]:
            raise ValueError("Policy bytes have changed since the verification receipt")
    required = {f"{EXPORT_PATH}/graph.yaml", *(f"{EXPORT_PATH}/{name}.onnx" for name in FULL_GRAPH_NAMES)}
    if not required.issubset(paths):
        raise ValueError("Full five-graph export is missing from the verified receipt")
    # The receipt must include external tensor files too; graph hash validation
    # alone covers the small .onnx protobuf, not its multi-GB adjacent data.
    for item in (root / EXPORT_PATH).glob("*.onnx.data"):
        if str(item.relative_to(root)) not in paths:
            raise ValueError("External ONNX tensor data is absent from the verified receipt")
    return root


def require_full_policy(policy):
    if set(policy.sessions) != FULL_GRAPH_NAMES:
        raise ValueError("Serving requires all five graphs; small-only fallback is forbidden")
    if policy.provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
        raise ValueError("Unrecognized execution provider")


def write_capture(directory: Path, body, observation, outputs, result, timings, elapsed, seed_offset=0):
    # mkdir(exist_ok=False) reserves this request identity atomically. Partial
    # captures stay as failure evidence; a repeated identity never overwrites it.
    name = f"e{body['observation']['episode_id']}_f{body['observation']['frame_id']}_q{body['sequence_id']}"
    sample = directory / name
    sample.mkdir(exist_ok=False)
    with (sample / "observation.npz").open("xb") as stream:
        np.savez_compressed(stream, **observation)
    with (sample / "actions.npz").open("xb") as stream:
        np.savez_compressed(stream, **outputs)
    evidence = {"stamp": result["observation"], "sequence_id": result["sequence_id"],
                "profile": result["profile"], "model_revision": result["model_revision"],
                "sampling_seed": seed_offset + body["sequence_id"], "seed_offset": seed_offset,
                "stage_seconds": timings, "inference_seconds": elapsed,
                "observation_sha256": sha256(sample / "observation.npz"),
                "actions_sha256": sha256(sample / "actions.npz"),
                "input_kind": "supplied_rgb_and_measured_joint_request",
                "full_forward_verified": True, "source_rollout_verified": False,
                "bevy_rollout_verified": False, "task_qualified": False}
    with (sample / "receipt.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")


class BoundedPolicyServer(ThreadingHTTPServer):
    """Reject excess connections before creating an unbounded handler backlog."""

    daemon_threads = True
    block_on_close = False
    request_queue_size = 2

    def __init__(self, address, request_handler):
        if address[0] != "127.0.0.1":
            raise ValueError("Only IPv4 loopback is supported")
        self.handler_slots = threading.BoundedSemaphore(2)
        super().__init__(address, request_handler)

    def process_request(self, request, client_address):
        if not self.handler_slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.handler_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.handler_slots.release()


def handler(policy, capture_dir: Path | None, seed_offset: int = 0):
    require_full_policy(policy)
    if type(seed_offset) is not int or not 0 <= seed_offset < 2**32:
        raise ValueError("Seed offset must be an unsigned 32-bit integer")
    lock = threading.Lock()
    status_lock = threading.Lock()
    status = {"successful_inferences": 0, "failed_inferences": 0, "last_inference_seconds": None}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def reply(self, status_code, content):
            payload = json.dumps(content, allow_nan=False).encode()
            if len(payload) > MAX_REPLY_BYTES:
                raise ValueError("Reply exceeds protocol bound")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "unknown path"})
                return
            with status_lock:
                snapshot = dict(status)
            self.reply(200, {"profile": "static_apple", "revision": PROFILES["static_apple"]["revision"],
                             "provider": policy.provider, "loaded_graphs": sorted(policy.sessions),
                             "instruction": PROFILES["static_apple"]["fixed_instruction"],
                             "sampling_scheme": "seed_offset_plus_sequence_id", "seed_offset": seed_offset,
                             "source_rollout_verified": False, "bevy_rollout_verified": False,
                             "task_qualified": False, **snapshot})

        def do_POST(self):
            if self.path != "/infer":
                self.reply(404, {"error": "unknown path"})
                return
            if not lock.acquire(blocking=False):
                self.reply(409, {"error": "one inference is already in flight"})
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() or self.headers.get("Transfer-Encoding"):
                    self.reply(400, {"error": "one Content-Length and no Transfer-Encoding are required"})
                    return
                size = int(lengths[0])
                if not 0 < size <= MAX_REQUEST_BYTES:
                    self.reply(413, {"error": "request size is outside the bounded protocol"})
                    return
                if self.headers.get("Content-Type") != "application/json":
                    self.reply(415, {"error": "Content-Type must be application/json"})
                    return
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ValueError("Incomplete request body")
                body = parse_json(payload)
                observation = decode_request(body)
                started = time.perf_counter()
                outputs, timings = policy.infer(observation, seed=seed_offset + body["sequence_id"])
                elapsed = time.perf_counter() - started
                result = action_chunk(body, outputs)
                if type(timings) is not dict or set(timings) != FULL_GRAPH_NAMES or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in timings.values()):
                    raise ValueError("Full five-stage timing receipt is invalid")
                if capture_dir:
                    write_capture(capture_dir, body, observation, outputs, result, timings, elapsed, seed_offset)
                with status_lock:
                    status["successful_inferences"] += 1
                    status["last_inference_seconds"] = elapsed
                self.reply(200, result)
            except FileExistsError:
                self.reply(409, {"error": "capture identity already exists; previous evidence preserved"})
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                with status_lock:
                    status["failed_inferences"] += 1
                self.reply(422, {"error": str(error)[:200]})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                with status_lock:
                    status["failed_inferences"] += 1
                self.reply(500, {"error": type(error).__name__})
                print(json.dumps({"event": "inference_failed", "error": type(error).__name__}), flush=True)
            finally:
                lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Fixed per-run offset added to request sequence; default preserves existing sampling")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if not 0 <= args.seed_offset < 2**32:
        parser.error("seed-offset must be 0..4294967295")
    root = verify_receipt(args.receipt)
    from unitree_g1_static_onnx import StaticAppleOnnx
    policy = StaticAppleOnnx(root, args.device, small_only=False)
    require_full_policy(policy)
    if args.capture_dir:
        args.capture_dir.mkdir(parents=True, exist_ok=True)
    server = BoundedPolicyServer(("127.0.0.1", args.port), handler(policy, args.capture_dir, args.seed_offset))
    print(json.dumps({"event": "ready", "address": f"127.0.0.1:{args.port}",
                      "profile": "static_apple", "revision": PROFILES["static_apple"]["revision"],
                      "provider": policy.provider, "task_qualified": False}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
