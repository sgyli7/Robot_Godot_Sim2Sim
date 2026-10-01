#!/usr/bin/env python3
"""Validate the pinned gn1_6 processor or run its official full policy forward.

CPU processor mode loads no policy weights. Full mode loads the real policy on
CUDA and must be scheduled with the GPU owner. Neither mode qualifies a task.
Input NPZ contains ego_view uint8 [1,480,640,3], four arm/hand groups float32
[1,7], and waist float32 [1,3], in the official processor's joint-group order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from unitree_g1_policy_prepare import PROFILES, GR00T_REVISION, artifact_contract


def validate_source(path: Path):
    rev = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if rev != GR00T_REVISION:
        raise ValueError(f"Wrong GR00T source: {rev}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("Pinned GR00T source has tracked modifications")
    sys.path.insert(0, str(path))


def verify_receipt(model_root: Path, receipt_path: Path):
    receipt = json.loads(receipt_path.read_text())
    if receipt["revision"] != PROFILES["mobile_box"]["revision"] or not receipt["local_files_verified"]:
        raise ValueError("A verified gn1_6 artifact receipt is required")
    for item in receipt["files"]:
        name = item["path"]
        path = model_root / name
        if not path.resolve().is_relative_to(model_root.resolve()):
            raise ValueError("Receipt path escapes model directory")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Artifact changed: {name}")
    artifact_contract("mobile_box", model_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["processor", "full"], default="processor")
    parser.add_argument("--seed", type=int, default=1)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", type=Path)
    inp.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    validate_source(args.gr00t_source)
    verify_receipt(args.model_root, args.receipt)
    # All processor resources are vendored in the frozen source checkout.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    if args.stage == "processor":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    torch.manual_seed(args.seed)
    dimensions = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3}
    if args.smoke:
        obs = {"ego_view": np.zeros((1, 480, 640, 3), np.uint8),
               **{key: np.zeros((1, dim), np.float32) for key, dim in dimensions.items()}}
    else:
        with np.load(args.input, allow_pickle=False) as data:
            obs = dict(data)
    if set(obs) != {*dimensions, "ego_view"}:
        raise ValueError("Input observation keys differ from the frozen profile")
    if obs["ego_view"].shape != (1, 480, 640, 3) or obs["ego_view"].dtype != np.uint8:
        raise ValueError("Input RGB must be uint8 [1,480,640,3]")
    for key, dim in dimensions.items():
        if obs[key].shape != (1, dim) or obs[key].dtype != np.float32 or not np.isfinite(obs[key]).all():
            raise ValueError(f"Invalid measured state: {key}")
    state = {key: obs[key] for key in dimensions}
    text = PROFILES["mobile_box"]["reference_instruction"]
    started = time.perf_counter()
    if args.stage == "processor":
        processor = Gr00tN1d6Processor.from_pretrained(args.model_root)
        processor.eval()
        data = VLAStepData(images={"ego_view": obs["ego_view"]}, states=state,
                           actions={}, text=text, embodiment=EmbodimentTag.NEW_EMBODIMENT)
        processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": data}])
        collated = processor.collator([processed])
        actions = processor.decode_action(np.zeros((1, 50, 128), np.float32), EmbodimentTag.NEW_EMBODIMENT,
                                          {key: value[None] for key, value in state.items()})
    else:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        if not torch.cuda.is_available():
            raise RuntimeError("Full gn1_6 forward requires the scheduled CUDA environment")
        policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, str(args.model_root), device="cuda:0", strict=True)
        actions, _ = policy.get_action({"video": {"ego_view": obs["ego_view"][None]},
                                       "state": {key: value[None] for key, value in state.items()},
                                       "language": {policy.language_key: [[text]]}})
        collated = None
    # Match Gr00tPolicy's published final cast after official denormalization.
    actions = {key: value.astype(np.float32) for key, value in actions.items()}
    expected_actions = {**dimensions, "base_height_command": 1, "navigate_command": 3}
    if set(actions) != set(expected_actions):
        raise ValueError("Wrong decoded action groups")
    for key, dim in expected_actions.items():
        if actions[key].shape != (1, 50, dim) or not np.isfinite(actions[key]).all():
            raise ValueError(f"Invalid decoded action: {key}")

    def shapes(value):
        if hasattr(value, "shape"):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict) or hasattr(value, "items"):
            return {key: shapes(v) for key, v in value.items()}
        if isinstance(value, list):
            return [shapes(v) for v in value]
        return str(type(value))

    result = {"profile": "mobile_box", "revision": PROFILES["mobile_box"]["revision"],
              "source_revision": GR00T_REVISION, "stage": args.stage,
              "input_kind": "synthetic_zero_smoke" if args.smoke else "supplied_observation",
              "full_forward_verified": args.stage == "full", "source_rollout_verified": False,
              "bevy_rollout_verified": False, "elapsed_seconds": time.perf_counter() - started,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "action_shapes": shapes(actions), "all_finite": True}
    if collated is not None:
        result["processor_inputs"] = shapes(collated)
    if args.stage == "full":
        result["torch_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **actions)
    result["actions_sha256"] = hashlib.sha256(args.output.with_suffix(".npz").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
