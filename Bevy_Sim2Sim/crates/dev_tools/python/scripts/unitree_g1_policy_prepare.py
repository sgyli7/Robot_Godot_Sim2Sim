#!/usr/bin/env python3
"""Download and verify the pinned Arena task policies without starting inference.

Use a separate model cache; downloaded upstream material is never copied into Git.
Requires huggingface_hub. The recorded hashes describe real local bytes, not merely
the presence of a Hugging Face cache entry. This command grants no task capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import struct


ARENA_REVISION = "7d75c95934c51a0318c957a8831e862ca43c53b5"
GR00T_REVISION = "e29d8fc50b0e4745120ae3fb72447986fe638aa6"
PROFILES = {
    "static_apple": {
        "repo_id": "nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace",
        "revision": "7f78bebf1a90131e7304beacfcd47eb27bad16ab",
        "format": "onnx",
        "arena_revision": "8b4a3a47fc53de23e8205089d71109a2e2348acd",
        "body_backend": "agile",
        "action_horizon": 40,
        "action_period_ns": 20_000_000,
        "fixed_instruction": "move the apple to the plate",
        "patterns": [
            "README.md", "LICENCE", "config.json", "processor_config.json",
            "statistics.json", "embodiment_id.json", "experiment_cfg/*",
            "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/*.onnx",
            "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/*.onnx.data",
            "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/graph.yaml",
            "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/README.md",
        ],
    },
    "mobile_box": {
        "repo_id": "nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation",
        "revision": "dfe74af855007f26093f362cd2d7a2f404b64b93",
        "upstream_branch": "gn1_6",
        "format": "safetensors",
        "arena_revision": ARENA_REVISION,
        "body_backend": "homie_v2",
        "action_horizon": 50,
        "action_period_ns": 20_000_000,
        "reference_instruction": (
            "Pick up the brown box from the shelf, and place it into the blue bin "
            "on the table located at the right of the shelf."
        ),
        "patterns": [
            "README.md", "config.json", "processor_config.json", "statistics.json",
            "embodiment_id.json", "experiment_cfg/*",
            "model-*.safetensors", "model.safetensors.index.json",
        ],
    },
}
CONTRACT_PATHS = [
    "pyproject.toml",
    "docs/pages/example_workflows/locomanipulation/step_5_evaluation.rst",
    "isaaclab_arena/embodiments/g1/g1.py",
    "isaaclab_arena/environments/isaaclab_arena_manager_based_env_cfg.py",
    "isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py",
    "isaaclab_arena_gr00t/embodiments/g1/modality.json",
    "isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml",
    "isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml",
    "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml",
    "isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py",
    "isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action_cfg.py",
    "isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/configs.py",
    "isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_homie_v2.yaml",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_files(info, patterns):
    from fnmatch import fnmatchcase

    return sorted(
        (entry for entry in info.siblings if any(fnmatchcase(entry.rfilename, p) for p in patterns)),
        key=lambda entry: entry.rfilename,
    )


def artifact_contract(profile_name: str, root: Path) -> dict:
    """Check the published interface itself, independently from downloaded size."""
    config = json.loads((root / "config.json").read_text())
    if profile_name == "static_apple":
        import yaml

        graph = yaml.safe_load((root / "exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/graph.yaml").read_text())
        nodes = graph["models"]
        outputs = nodes["decode_action"]["outputs"]
        if [out["shape"] for out in outputs] != [[1, 40, n] for n in [7, 7, 7, 7, 3, 1, 3]]:
            raise ValueError("Static export decoded action interface mismatch")
        return {"status": "interface_matched", "decoded_action_width": 35,
                "action_horizon": 40, "state_inputs": nodes["preprocess_state"]["inputs"],
                "decoded_outputs": outputs, "camera_input": nodes["preprocess_video"]["inputs"]}
    processor = json.loads((root / "processor_config.json").read_text())
    if config.get("model_type") != "Gr00tN1d6" or config.get("action_horizon") != 50:
        raise ValueError("Mobile checkpoint is not the documented gn1_6 50-step profile; main is incompatible")
    statistics = json.loads((root / "statistics.json").read_text())
    for name in ["embodiment_id.json", "statistics.json", "processor_config.json"]:
        if not (root / name).is_file():
            raise ValueError(f"Missing mobile processor dependency: {name}")
    settings = processor["processor_kwargs"]
    modalities = settings["modality_configs"]["new_embodiment"]
    action = modalities["action"]
    expected_names = ["left_arm", "right_arm", "left_hand", "right_hand", "waist",
                      "base_height_command", "navigate_command"]
    if action["modality_keys"] != expected_names or action["delta_indices"] != list(range(50)):
        raise ValueError("Mobile decoded action groups/horizon mismatch")
    if [entry["rep"] for entry in action["action_configs"]] != ["ABSOLUTE"] * 7:
        raise ValueError("Mobile actions are not the expected absolute representation")
    if [len(statistics["new_embodiment"]["action"][key]["mean"]) for key in expected_names] != [7, 7, 7, 7, 3, 1, 3]:
        raise ValueError("Mobile statistics do not match 35 decoded dimensions")
    shapes = {}
    for path in root.glob("model-*.safetensors"):
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 16 * 1024 * 1024:
                raise ValueError("Unexpectedly large safetensors header")
            header = json.loads(stream.read(header_size))
        for key, value in header.items():
            if key in ["action_head.action_decoder.layer2.W", "action_head.action_encoder.W1.W"]:
                shapes[key] = value["shape"]
    if shapes.get("action_head.action_decoder.layer2.W") != [32, 1024, 128]:
        raise ValueError("Mobile decoder weight tensor shape mismatch")
    return {"status": "interface_matched", "model_type": config["model_type"],
            "action_horizon": config["action_horizon"], "max_action_dim": config["max_action_dim"],
            "decoded_action_width": 35, "modalities": modalities, "weight_tensor_shapes": shapes,
            "normalization": {key: settings[key] for key in ["use_percentiles", "clip_outliers", "apply_sincos_state_encoding", "use_relative_action"]}}


def source_contract(source: Path, profile_name: str) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PROFILES[profile_name]["arena_revision"]:
        raise ValueError(f"Arena revision mismatch: {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("Arena source has tracked modifications; keep adaptations separate")
    paths = CONTRACT_PATHS.copy()
    if profile_name == "static_apple":
        paths.remove("isaaclab_arena/environments/isaaclab_arena_manager_based_env_cfg.py")
        paths.extend([
            "isaaclab_arena/environments/isaaclab_arena_manager_based_env.py",
            "isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_agile.yaml",
            "isaaclab_arena_environments/galileo_g1_static_pick_and_place_environment.py",
            "isaaclab_arena_environments/mdp/galileo_g1_static_pick_and_place/robot_configs.py",
            "isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml",
            "docs/pages/example_workflows/static_apple/step_4_evaluation.rst",
        ])
    return {
        "revision": revision,
        "files": {name: sha256(source / name) for name in paths},
        "source_physics_period_ns": 5_000_000,
        "source_control_period_ns": 20_000_000,
    }


def prepare(profile_name: str, args) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    profile = PROFILES[profile_name]
    info = HfApi().model_info(
        profile["repo_id"], revision=profile["revision"], files_metadata=True
    )
    if info.sha != profile["revision"]:
        raise ValueError("Hugging Face did not resolve the requested immutable revision")
    entries = selected_files(info, profile["patterns"])
    root = args.models_root / profile_name / profile["revision"]
    receipt = {
        "schema": "unitree_g1_policy_files_v1", "profile": profile_name,
        **{k: v for k, v in profile.items() if k != "patterns"},
        "model_root": str(root.resolve()), "gated": info.gated,
        "license_scope": "non_commercial_research",
        "local_files_verified": False, "inference_verified": False,
        "source_rollout_verified": False, "bevy_rollout_verified": False,
        "files": [],
    }
    if not entries:
        raise ValueError("No files matched the pinned profile")
    print(json.dumps({"event": "profile", "profile": profile_name,
                      "bytes": sum(e.size or 0 for e in entries)}), flush=True)
    if args.download:
        snapshot_download(
            repo_id=profile["repo_id"], revision=profile["revision"],
            local_dir=root, allow_patterns=profile["patterns"], max_workers=2,
        )
    for entry in entries:
        path = root / entry.rfilename
        record = {"path": entry.rfilename, "bytes": entry.size,
                  "upstream_lfs_sha256": entry.lfs.sha256 if entry.lfs else None,
                  "upstream_git_blob": entry.blob_id}
        if args.download or args.verify:
            if not path.is_file() or path.stat().st_size != entry.size:
                raise ValueError(f"Missing or wrong-size model file: {path}")
            record["sha256"] = sha256(path)
            if entry.lfs and record["sha256"] != entry.lfs.sha256:
                raise ValueError(f"Model hash mismatch: {path}")
            if not entry.lfs:
                content = path.read_bytes()
                blob_hash = hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest()
                if blob_hash != entry.blob_id:
                    raise ValueError(f"Metadata Git blob mismatch: {path}")
        receipt["files"].append(record)
    receipt["local_files_verified"] = bool(args.download or args.verify)
    if receipt["local_files_verified"]:
        receipt["artifact_contract"] = artifact_contract(profile_name, root)
    source = args.static_arena_source if profile_name == "static_apple" else args.arena_source
    if source:
        receipt["arena_source"] = source_contract(source, profile_name)
    if args.gr00t_source:
        revision = subprocess.check_output(["git", "-C", str(args.gr00t_source), "rev-parse", "HEAD"], text=True).strip()
        if revision != GR00T_REVISION:
            raise ValueError("GR00T source revision mismatch")
        receipt["mobile_gr00t_source"] = {
            "revision": revision,
            "files": {name: sha256(args.gr00t_source / name) for name in [
                "gr00t/policy/gr00t_policy.py", "gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py",
                "gr00t/model/gr00t_n1d6/gr00t_n1d6.py", "gr00t/model/modules/eagle_backbone.py",
            ]},
        }
    args.receipts.mkdir(parents=True, exist_ok=True)
    target = args.receipts / f"{profile_name}_files.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({"event": "receipt", "path": str(target),
                      "local_files_verified": receipt["local_files_verified"]}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=[*PROFILES, "all"])
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--arena-source", type=Path)
    parser.add_argument("--static-arena-source", type=Path)
    parser.add_argument("--gr00t-source", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    names = list(PROFILES) if args.profile == "all" else [args.profile]
    try:
        for name in names:
            prepare(name, args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
