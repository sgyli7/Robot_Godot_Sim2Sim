#!/usr/bin/env python3
"""Read-only audit of one native station T1 startup/live-policy capture.

Checks actual pixels, sampling identity, original actions, execution order and
physical counts. Placement truth is consumed only after execution, never used
as model input or control. This grants no Qwen or continuous-time qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(case: Path, seed_offset: int, legacy_sequence_seed: bool = False):
    native = case / "native"
    rows = [json.loads(line) for line in (native / "owner_steps.jsonl").read_text().splitlines()]
    receipt = json.loads((native / "capture_receipt.json").read_text())
    execution = json.loads((case / "execution.json").read_text())
    placement = json.loads((case / "placement_geometry_audit.json").read_text())
    config = json.loads((case / "config.json").read_text())
    if legacy_sequence_seed:
        commit = execution["code_commit"]
        if seed_offset != 0 or not re.fullmatch("[0-9a-f]{40}", commit):
            raise ValueError("Legacy sequence sampling requires the exact source commit and offset0")
        source = subprocess.check_output(["git", "show", commit +
            ":Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_static_server.py"],
            cwd=Path(__file__).resolve().parents[5], text=True)
        if 'policy.infer(observation, seed=body["sequence_id"])' not in source:
            raise ValueError("Legacy seed identity is not supported by the pinned executed source")
    if digest(case / "config.json") != execution["config_sha256"]:
        raise ValueError("Case configuration changed after execution")
    if not config.get("static_startup") or config["policy"]["max_calls"] != 8:
        raise ValueError("Audit requires the frozen60+320Tick static station profile")
    episode = config["runner"]["body"]["static_agile"]["episode_id"]
    sequences = {}
    contact_ticks = off_shelf_ticks = 0
    for tick, row in enumerate(rows, 1):
        body = row["body"]["static_agile"]
        step = body["step_configuration"]
        if (row["episode_id"] != episode or row["owner_episode_integrations"] != tick
                or body["integration_count"] != tick or step["physics_hz"] != 50
                or np.float32(step["dt"]) != np.float32(0.02)
                or step["num_internal_pgs_iterations"] != 16
                or step["num_solver_iterations"] != 1 or step["max_ccd_substeps"] != 1
                or step["additional_solver_iterations_max"] != 0):
            raise ValueError("Episode/clock/source integration contract changed")
        if body["task_objects"]["world_counts"] != {
                "bodies": 57, "colliders": 2608, "impulse_joints": 0,
                "multibody_joint_handles": 52, "entity_mappings": 0}:
            raise ValueError("Native task topology changed")
        if body["native_static_environment"]["collider_count"] != 2553:
            raise ValueError("Scientific station geometry is incomplete")
        if tick <= 60:
            startup = row["execution"]["startup"]
            if startup["ticks"] != tick or startup["original_vla_output"]:
                raise ValueError("Startup was omitted or mislabelled as VLA")
        else:
            applied = row["execution"]["original_vla"]
            sequence = 1 + (tick - 61) // 40
            if (applied["sequence_id"] != sequence or applied["admitted_chunks"] != sequence
                    or applied["frame_index"] != (tick - 61) % 40):
                raise ValueError("Original chunk/frame execution was rebased or skipped")
            sequences.setdefault(sequence, []).append(applied)
        apple = body["task_objects"]["objects"][0]
        contact_ticks += any(c["other_robot_body_index"] is not None and c["normal_impulse_n_s"] > 0
                             for c in apple["last_solve_contacts"])
        off_shelf_ticks += not apple["source_shelf_contact"]
    images = []
    captures = sorted((case / "policy_captures").iterdir())
    if len(captures) != 8:
        raise ValueError("Expected exactly eight fresh original policy captures")
    for sample in captures:
        model_receipt = json.loads((sample / "receipt.json").read_text())
        sequence = model_receipt["sequence_id"]
        stamp = json.loads((native / f"live_stamp_{sequence:04}.json").read_text())
        reply = json.loads((native / f"live_reply_{sequence:04}.json").read_text())
        source_tick = 60 + (sequence - 1) * 40
        if (stamp["source_ticks"] != [source_tick, source_tick]
                or stamp["sim_time_ns"] != source_tick * 20_000_000
                or reply["observation"] != model_receipt["stamp"]
                or reply["observation"] != sequences[sequence][0]["observation"]
                or model_receipt.get("sampling_seed", sequence if legacy_sequence_seed else None) != seed_offset + sequence
                or model_receipt.get("seed_offset", 0 if legacy_sequence_seed else None) != seed_offset):
            raise ValueError("Actual image stamp or sampling identity changed")
        with np.load(sample / "observation.npz", allow_pickle=False) as observation:
            pixels = np.asarray(Image.open(native / f"live_ego_{sequence:04}.png").convert("RGB"))
            if not np.array_equal(observation["ego_view"][0], pixels):
                raise ValueError("Model did not receive the captured RGB pixels")
        with np.load(sample / "actions.npz", allow_pickle=False) as actions:
            widths = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3,
                      "base_height_command": 1, "navigate_command": 3}
            for group in widths:
                values = [([frame["base_height_m"]] if group == "base_height_command"
                           else frame["navigate_mps_rps"] if group == "navigate_command"
                           else frame[group]) for frame in reply["frames"]]
                if not np.array_equal(actions[group][0], np.asarray(values, dtype=np.float32)):
                    raise ValueError("Original decoded model actions changed before submission")
        images.append({"sequence": sequence, "actual_source_tick": source_tick,
                       "sampling_seed": seed_offset + sequence,
                       "actual_rgb_sha256": digest(native / f"live_ego_{sequence:04}.png")})
    if (len(rows) != 380 or receipt["actual_integrations"] != 380
            or receipt["actual_model_successes"] != 380 or receipt["owner_step_records"] != 380
            or not receipt["owner_step_trace_complete"] or receipt["owner_step_trace_dropped"] != 0
            or receipt["live_policy_inference_calls"] != 8 or receipt["live_policy_successes"] != 8
            or not receipt["static_startup"]["ready"]):
        raise ValueError("Startup, body inference, VLA or evidence counts are incomplete")
    lighting = receipt["native_station_illumination"]
    light = lighting["directional_lights"][0]
    if (not lighting["read_actual_ecs_lights"] or lighting["added_directional_lights"] != 0
            or light["shadow_maps_enabled"]
            or np.float32(lighting["ambient_brightness"]) != np.float32(2328.2639821816138)
            or np.float32(light["illuminance"]) != np.float32(981.3631918604952)):
        raise ValueError("Frozen lighting did not affect the actual station lights")
    if placement["input_sha256"]["trace"] != digest(native / "owner_steps.jsonl"):
        raise ValueError("Placement auditor used a different physical trace")
    return {"schema": "g1_native_static_capture_audit_v1", "actual_integrations": 380,
            "startup_integrations": 60, "original_task_integrations": 320, "fresh_vla_calls": 8,
            "seed_offset": seed_offset, "images": images, "same_native_station_world": True,
            "legacy_sequence_seed_verified_from_pinned_source": legacy_sequence_seed,
            "strict_placement_passed": placement["diagnostic_release_window_passed"],
            "max_continuous_placement_seconds": placement["max_continuous_placement_seconds"],
            "minimum_margin_m": placement["minimum_margin_during_ready_samples_m"],
            "standing_all_ticks": all(s["standing"] for s in placement["samples"]),
            "apple_robot_positive_contact_ticks": contact_ticks, "apple_no_shelf_contact_ticks": off_shelf_ticks,
            "active_sim_wall_ratio": 7.6 / receipt["active_wall_seconds"],
            "continuous_boundary_sim_wall_ratio": 7.58 / receipt["continuous_boundary_wall_seconds"],
            "control_deadlines_missed": receipt["control_deadlines_missed"], "pending_ticks": receipt["pending_ticks"],
            "pauses_for_camera_and_policy": receipt["pauses_for_camera_and_policy"],
            "qwen_target_selection": False, "full_task_qualified": False,
            "trace_sha256": digest(native / "owner_steps.jsonl"), "code_commit": execution["code_commit"],
            "binary_sha256": execution["binary_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-sequence-seed", action="store_true",
                        help="Read older offset0 receipts only after verifying exact committed sequence-seed source")
    args = parser.parse_args()
    result = audit(args.case, args.seed_offset, args.legacy_sequence_seed)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
