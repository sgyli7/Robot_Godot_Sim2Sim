#!/usr/bin/env python3
"""Serve the frozen gn1_6 mobile-box profile on IPv4 loopback only.

The initial adapter always uses the published brown-box-to-blue-bin instruction.
It accepts RGB and measured joints, not object truth or arbitrary language. Torch
and policy imports happen only when the real model owner is explicitly started.
Protocol fixtures can import this module without loading any model/runtime.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

from unitree_g1_policy_prepare import PROFILES, GR00T_REVISION, sha256
from unitree_g1_mobile_forward import validate_source, verify_receipt
from unitree_g1_static_server import (
    BoundedPolicyServer, MAX_REQUEST_BYTES, MAX_REPLY_BYTES, MAX_FLOAT32,
    STAMP_KEYS, exact_keys, parse_json,
)
from http.server import BaseHTTPRequestHandler


PROFILE = PROFILES["mobile_box"]
GROUP_WIDTHS = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3}
INSTRUCTION_SCOPE = "fixed_published_brown_box_to_blue_bin_task_adapter"


def decode_request(body):
    exact_keys(body, {"schema", "profile", "sequence_id", "observation", "camera_rgb_b64", "state_groups"}, "Request")
    if body["schema"] != "mobile_observation_v1" or body["profile"] != "mobile_box":
        raise ValueError("Only the fixed mobile_box gn1_6 adapter is supported")
    exact_keys(body["observation"], STAMP_KEYS, "Observation stamp")
    for value in [body["sequence_id"], *body["observation"].values()]:
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError("Stamp and sequence fields must be unsigned 64-bit integers")
    if type(body["camera_rgb_b64"]) is not str or len(body["camera_rgb_b64"]) != 640 * 480 * 4:
        raise ValueError("Camera must encode exactly 640x480 RGB bytes")
    raw = base64.b64decode(body["camera_rgb_b64"], validate=True)
    if len(raw) != 640 * 480 * 3:
        raise ValueError("Camera must have exactly 640x480 RGB pixels")
    exact_keys(body["state_groups"], GROUP_WIDTHS, "Measured state")
    state = {}
    for name, width in GROUP_WIDTHS.items():
        values = body["state_groups"][name]
        if type(values) is not list or len(values) != width:
            raise ValueError(f"Invalid measured joint array: {name}")
        if any(type(value) not in (int, float) or abs(value) > MAX_FLOAT32 or not math.isfinite(value) for value in values):
            raise ValueError(f"Invalid measured joint value: {name}")
        state[name] = np.asarray(values, dtype=np.float32).reshape(1, 1, width)
    return {"video": {"ego_view": np.frombuffer(raw, dtype=np.uint8).reshape(1, 1, 480, 640, 3)},
            "state": state,
            "language": {"annotation.human.task_description": [[PROFILE["reference_instruction"]]]}}


def action_chunk(body, outputs):
    widths = {**GROUP_WIDTHS, "base_height_command": 1, "navigate_command": 3}
    exact_keys(outputs, widths, "Model output")
    for name, width in widths.items():
        value = outputs[name]
        if type(value) is not np.ndarray or value.shape != (1, 50, width) or value.dtype != np.float32:
            raise ValueError(f"Wrong model output shape/type: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite model output: {name}")
    frames = []
    for index in range(50):
        frame = {name: outputs[name][0, index].tolist() for name in GROUP_WIDTHS}
        frame["base_height_m"] = float(outputs["base_height_command"][0, index, 0])
        frame["navigate_mps_rps"] = outputs["navigate_command"][0, index].tolist()
        frames.append(frame)
    return {"profile": "mobile_box", "observation": dict(body["observation"]),
            "sequence_id": body["sequence_id"], "model_revision": PROFILE["revision"],
            "action_period_ns": 20_000_000, "frames": frames}


class MobileBoxPolicy:
    """One explicitly started original GR00T owner; no static/synthetic fallback."""

    def __init__(self, source: Path, model_root: Path, receipt: Path, seed: int, runtime_env: Path):
        source = source.resolve()
        model_root = model_root.resolve()
        runtime_env = runtime_env.resolve()
        if Path(sys.prefix).resolve() != runtime_env:
            raise ValueError("Start this owner with the explicitly selected isolated runtime environment")
        verified = parse_json(receipt.read_bytes())
        if verified["schema"] != "unitree_g1_policy_files_v1" or verified["profile"] != "mobile_box" or verified["revision"] != PROFILE["revision"] or verified["local_files_verified"] is not True or Path(verified["model_root"]).resolve() != model_root:
            raise ValueError("Wrong mobile policy receipt/model root")
        validate_source(source)
        verify_receipt(model_root, receipt)
        # Resolve only cached, frozen processor resources; no runtime networking.
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        if not Path(torch.__file__).resolve().is_relative_to(runtime_env):
            raise ValueError("Torch import escaped the explicitly selected runtime environment")
        module = Path(sys.modules[Gr00tPolicy.__module__].__file__).resolve()
        if not module.is_relative_to(source):
            raise ValueError("GR00T import did not resolve to the frozen source")
        if not torch.cuda.is_available():
            raise RuntimeError("The scheduled CUDA environment is required; no CPU fallback")
        torch.manual_seed(seed)
        self._torch = torch
        self._lock = threading.Lock()
        self._policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, str(model_root), device="cuda:0", strict=True)
        modalities = self._policy.get_modality_config()
        if modalities["state"].modality_keys != list(GROUP_WIDTHS) or modalities["action"].modality_keys != [*GROUP_WIDTHS, "base_height_command", "navigate_command"] or modalities["action"].delta_indices != list(range(50)):
            raise ValueError("Loaded policy modalities differ from frozen gn1_6 groups")
        if self._policy.language_key != "annotation.human.task_description":
            raise ValueError("Loaded policy language key differs from the frozen contract")
        self.metadata = {"profile": "mobile_box", "revision": PROFILE["revision"],
                         "gr00t_source_revision": GR00T_REVISION, "gr00t_source_root": str(source),
                         "gr00t_policy_source_sha256": sha256(module),
                         "model_root": str(model_root), "model_receipt_sha256": sha256(receipt),
                         "runtime_env": str(runtime_env), "python_executable": sys.executable,
                         "python_executable_sha256": sha256(Path(sys.executable)),
                         "server_source_sha256": sha256(Path(__file__)),
                         "torch_version": torch.__version__, "device": str(self._policy.model.device),
                         "inference_backend": "official_gr00t_n1d6_pytorch", "model_loaded": True,
                         "initial_seed": seed, "instruction_scope": INSTRUCTION_SCOPE,
                         "fixed_instruction": PROFILE["reference_instruction"]}
        # Save the exact post-construction RNG state. Restoring this between
        # benchmark episodes is equivalent to a fresh owner with this seed,
        # without loading a second copy of the model or changing its contract.
        self._initial_cpu_rng = torch.get_rng_state().clone()
        self._initial_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        self.metadata["post_load_rng_sha256"] = hashlib.sha256(
            self._initial_cpu_rng.numpy().tobytes()
            + b"".join(state.cpu().numpy().tobytes() for state in self._initial_cuda_rng)
        ).hexdigest()

    def rewind_initial_rng(self):
        """Between episodes only; handler lock excludes active inference."""
        with self._lock:
            if self._policy is None:
                raise RuntimeError("Mobile policy owner has been closed")
            self._torch.set_rng_state(self._initial_cpu_rng.clone())
            self._torch.cuda.set_rng_state_all([state.clone() for state in self._initial_cuda_rng])

    def infer(self, observation):
        with self._lock:
            if self._policy is None:
                raise RuntimeError("Mobile policy owner has been closed")
            actions, _ = self._policy.get_action(observation)
            return actions

    def close(self):
        # This service owns the model; the simulation's HTTP worker never waits
        # for this cleanup. Serialize cleanup with the one actual inference.
        with self._lock:
            self._policy = None
            self.metadata["model_loaded"] = False
            gc.collect()
            self._torch.cuda.empty_cache()


def write_capture(directory, body, observation, outputs, result, elapsed, metadata):
    name = f"e{body['observation']['episode_id']}_f{body['observation']['frame_id']}_q{body['sequence_id']}"
    sample = directory / name
    sample.mkdir(exist_ok=False)
    values = {"ego_view": observation["video"]["ego_view"], **observation["state"]}
    with (sample / "observation.npz").open("xb") as stream:
        np.savez_compressed(stream, **values)
    with (sample / "actions.npz").open("xb") as stream:
        np.savez_compressed(stream, **outputs)
    evidence = {"stamp": result["observation"], "sequence_id": result["sequence_id"],
                "profile": result["profile"], "model_revision": result["model_revision"],
                "policy_identity": dict(metadata), "inference_seconds": elapsed,
                "observation_sha256": sha256(sample / "observation.npz"),
                "actions_sha256": sha256(sample / "actions.npz"),
                "input_kind": "supplied_rgb_and_measured_joint_request",
                "full_forward_verified": metadata["model_loaded"],
                "source_rollout_verified": False, "bevy_rollout_verified": False, "task_qualified": False}
    with (sample / "receipt.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")


def handler(policy, capture_dir, allow_episode_seed_reset=False):
    lock = threading.Lock()
    status_lock = threading.Lock()
    status = {"successful_inferences": 0, "failed_inferences": 0, "last_inference_seconds": None}
    episode = {"current_episode_id": None, "episode_seed_resets": 0}
    admitted_episodes = set()

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
                snapshot = {**status, **episode,
                            "episode_seed_reset_enabled": allow_episode_seed_reset}
            self.reply(200, {**policy.metadata, **snapshot, "source_rollout_verified": False,
                             "bevy_rollout_verified": False, "task_qualified": False})

        def do_POST(self):
            if self.path != "/infer" and not (allow_episode_seed_reset and self.path == "/begin_episode"):
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
                if self.path == "/begin_episode":
                    exact_keys(body, {"schema", "episode_id", "initial_seed"}, "Episode seed reset")
                    if (body["schema"] != "mobile_episode_seed_reset_v1"
                            or type(body["episode_id"]) is not int
                            or not 0 <= body["episode_id"] < 2**64
                            or type(body["initial_seed"]) is not int
                            or body["initial_seed"] != policy.metadata["initial_seed"]):
                        raise ValueError("Episode reset must retain the owner's original seed")
                    if body["episode_id"] in admitted_episodes or len(admitted_episodes) >= 10000:
                        self.reply(409, {"error": "episode identity repeated or owner episode bound reached"})
                        return
                    policy.rewind_initial_rng()
                    admitted_episodes.add(body["episode_id"])
                    with status_lock:
                        episode["current_episode_id"] = body["episode_id"]
                        episode["episode_seed_resets"] += 1
                        snapshot = {**status, **episode}
                    self.reply(200, {**snapshot, "initial_seed": policy.metadata["initial_seed"],
                                     "post_load_rng_sha256": policy.metadata["post_load_rng_sha256"]})
                    return
                observation = decode_request(body)
                if allow_episode_seed_reset and body["observation"]["episode_id"] != episode["current_episode_id"]:
                    raise ValueError("Inference is not bound to the currently admitted benchmark episode")
                started = time.perf_counter()
                outputs = policy.infer(observation)
                elapsed = time.perf_counter() - started
                result = action_chunk(body, outputs)
                if capture_dir:
                    write_capture(capture_dir, body, observation, outputs, result, elapsed, policy.metadata)
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
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--episode-seed-reset", action="store_true",
                        help="Opt-in benchmark gate: restore post-load RNG once per unique episode")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.seed < 2**64:
        parser.error("port or seed outside valid range")
    policy = MobileBoxPolicy(args.gr00t_source, args.model_root, args.receipt, args.seed, args.runtime_env)
    server = None
    try:
        if args.capture_dir:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
        server = BoundedPolicyServer(("127.0.0.1", args.port), handler(policy, args.capture_dir, args.episode_seed_reset))
        print(json.dumps({"event": "ready", "address": f"127.0.0.1:{args.port}",
                          **policy.metadata, "task_qualified": False}), flush=True)
        server.serve_forever()
    finally:
        if server:
            server.server_close()
        policy.close()
        print(json.dumps({"event": "model_owner_closed", "profile": "mobile_box"}), flush=True)


if __name__ == "__main__":
    main()
