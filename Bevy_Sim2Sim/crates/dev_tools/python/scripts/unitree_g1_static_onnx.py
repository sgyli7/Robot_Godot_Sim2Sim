#!/usr/bin/env python3
"""Run the pinned static-apple ONNX graph on RGB + measured joint observations.

Input NPZ keys are ego_view [1,480,640,3] and left_arm/right_arm/left_hand/
right_hand [1,7], waist [1,3], all float32. Hand ordering is exactly graph.yaml's
preprocess_state interface, not decode_action's interface. --smoke substitutes
zero inputs and is only a numerical smoke test, never task-performance evidence.
Requires numpy, PyYAML and onnxruntime (or a compatible onnxruntime-gpu).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import yaml

from unitree_g1_policy_prepare import PROFILES


EXPORT_PATH = "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2"
DTYPES = {"float32": np.float32, "int64": np.int64, "int32": np.int32}


def validate_tensor(value, spec):
    if value.shape != tuple(spec["shape"]) or value.dtype != DTYPES[spec["dtype"]]:
        raise ValueError(f"Tensor shape/type mismatch: {spec['name']}: {value.shape}/{value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite tensor: {spec['name']}")


class StaticAppleOnnx:
    """Fixed-language export. Preprocessing and normalization stay inside the graph."""

    def __init__(self, model_root: Path, device: str, small_only: bool = False):
        self.root = model_root / EXPORT_PATH
        self.graph = yaml.safe_load((self.root / "graph.yaml").read_text())
        provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[device]
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Required provider {provider} unavailable: {ort.get_available_providers()}")
        if device == "cuda":
            # Load this isolated environment's NVIDIA wheels, without changing the
            # host loader configuration or falling back to system CUDA libraries.
            ort.preload_dlls(directory="")
        self.provider = provider
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 2
        options.enable_mem_pattern = True
        self.sessions = {}
        names = ["preprocess_video", "preprocess_state", "decode_action"] if small_only else list(self.graph["models"])
        for name in names:
            spec = self.graph["models"][name]
            path = self.root / spec["parameters"]["model_path"]
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != spec["parameters"]["sha256sum"]:
                raise ValueError(f"Graph checksum mismatch: {name}")
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
            if self.sessions[name].get_providers()[0] != provider:
                raise RuntimeError(f"Provider fallback at {name}")

    def stage(self, name: str, inputs: dict) -> tuple[dict, float]:
        spec = self.graph["models"][name]
        if set(inputs) != {x["name"] for x in spec["inputs"]}:
            raise ValueError(f"Wrong input keys at {name}")
        for tensor in spec["inputs"]:
            validate_tensor(inputs[tensor["name"]], tensor)
        started = time.perf_counter()
        result = self.sessions[name].run(None, inputs)
        elapsed = time.perf_counter() - started
        outputs = dict(zip((x.name for x in self.sessions[name].get_outputs()), result, strict=True))
        for tensor in spec["outputs"]:
            validate_tensor(outputs[tensor["name"]], tensor)
        return outputs, elapsed

    def infer(self, observation: dict, seed: int) -> tuple[dict, dict]:
        expected = {"ego_view", "left_arm", "right_arm", "left_hand", "right_hand", "waist"}
        if set(observation) != expected:
            raise ValueError(f"Observation keys must be exactly {sorted(expected)}")
        if observation["ego_view"].min() < 0 or observation["ego_view"].max() > 255:
            raise ValueError("RGB image is outside 0..255")
        video, tv = self.stage("preprocess_video", {"ego_view": observation["ego_view"]})
        state, ts = self.stage("preprocess_state", {key: value for key, value in observation.items() if key != "ego_view"})
        if "backbone" not in self.sessions:
            action, td = self.stage("decode_action", {"normalized_action": np.zeros((1, 40, 132), dtype=np.float32)})
            return action, {"preprocess_video": tv, "preprocess_state": ts, "decode_action": td}
        backbone, tb = self.stage("backbone", {
            "vl_input_input_ids": video["input_ids"],
            "vl_input_attention_mask": video["attention_mask"],
            "vl_input_pixel_values": video["pixel_values"],
        })
        predicted, ta = self.stage("action_head", {
            "backbone_outputs_backbone_features": backbone["converted_outputs_backbone_features"],
            "backbone_outputs_backbone_attention_mask": backbone["converted_outputs_backbone_attention_mask"],
            "backbone_outputs_image_mask": backbone["converted_outputs_image_mask"],
            "action_inputs_state": state["state"],
            "action_inputs_embodiment_id": video["embodiment_id"],
            "initial_noise": np.random.default_rng(seed).standard_normal((1, 40, 132), dtype=np.float32),
        })
        decoded, td = self.stage("decode_action", {"normalized_action": predicted["output1_action_pred"]})
        return decoded, {"preprocess_video": tv, "preprocess_state": ts, "backbone": tb,
                         "action_head": ta, "decode_action": td}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--small-only", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.input:
        with np.load(args.input, allow_pickle=False) as data:
            observation = dict(data)
    else:
        observation = {"ego_view": np.zeros((1, 480, 640, 3), dtype=np.float32),
                       **{k: np.zeros((1, 7), dtype=np.float32) for k in ["left_arm", "right_arm", "left_hand", "right_hand"]},
                       "waist": np.zeros((1, 3), dtype=np.float32)}
    started = time.perf_counter()
    policy = StaticAppleOnnx(args.model_root, args.device, args.small_only)
    load_seconds = time.perf_counter() - started
    actions, timings = policy.infer(observation, args.seed)
    receipt = {
        "profile": "static_apple", "revision": PROFILES["static_apple"]["revision"],
        "provider": policy.provider, "onnxruntime_version": ort.__version__,
        "input_kind": "synthetic_zero_smoke" if args.smoke else "supplied_observation",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
        "full_forward_verified": not args.small_only,
        "source_rollout_verified": False, "bevy_rollout_verified": False,
        "load_seconds": load_seconds, "stage_seconds": timings,
        "action_shapes": {key: list(value.shape) for key, value in actions.items()},
        "all_finite": all(np.isfinite(value).all().item() for value in actions.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **actions)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
