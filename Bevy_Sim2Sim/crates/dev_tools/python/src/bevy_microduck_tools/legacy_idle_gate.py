"""Independently reject unstable original alpha_stand diagnostic rollouts.

The historical Unity scenario threshold is useful for spotting a broken Idle,
but passing it does not qualify the Bevy plant, BAM, or a 60/60 skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HORIZON_TICKS = 240
MIN_UPRIGHT_DOT = 0.5
MAX_FINAL_DRIFT_M = 0.2


def _root(frame: dict[str, Any], expected_step: int) -> tuple[list[float], list[float]]:
    if frame.get("global_step") != expected_step:
        raise ValueError(f"pose frame {expected_step} has a different global_step")
    poses = [pose for pose in frame.get("poses", []) if pose.get("source_body_id") == 1]
    if len(poses) != 1:
        raise ValueError(f"pose frame {expected_step} has no unique root body")
    position = poses[0]["translation"]
    rotation = poses[0]["rotation_xyzw"]
    if len(position) != 3 or len(rotation) != 4:
        raise ValueError(f"pose frame {expected_step} has invalid root dimensions")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (*position, *rotation)):
        raise ValueError(f"pose frame {expected_step} has non-finite root values")
    norm = math.sqrt(sum(value * value for value in rotation))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"pose frame {expected_step} has a non-unit root rotation")
    return position, [value / norm for value in rotation]


def evaluate(report: dict[str, Any], report_sha256: str) -> dict[str, Any]:
    if report.get("schema") != "station_robot_60hz_diagnostic_v1":
        raise ValueError("input is not a station robot diagnostic report")
    if report.get("legacy_model_name") != "alpha_stand":
        raise ValueError("input is not the original alpha_stand policy")
    if report.get("physics_hz") != 60 or report.get("policy_hz") != 60:
        raise ValueError("input is not a 60/60 diagnostic")
    frames = report.get("pose_frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("input has no pose frames")

    first_position, _ = _root(frames[0], 0)
    min_upright = 1.0
    min_upright_step = 0
    first_below_threshold = None
    final_position = first_position
    for step, frame in enumerate(frames):
        final_position, rotation = _root(frame, step)
        x, _, z, _ = rotation
        upright = 1.0 - 2.0 * (x * x + z * z)
        if upright < min_upright:
            min_upright = upright
            min_upright_step = step
        if first_below_threshold is None and upright < MIN_UPRIGHT_DOT:
            first_below_threshold = step

    drift = math.hypot(final_position[0] - first_position[0],
                       final_position[2] - first_position[2])
    complete = (
        len(frames) == HORIZON_TICKS + 1
        and report.get("integration_count") == HORIZON_TICKS
        and report.get("policy_inference_count") == HORIZON_TICKS
        and report.get("passed") is True
        and report.get("final_clock", {}).get("global_step") == HORIZON_TICKS
    )
    checks = {
        "complete_4_seconds": complete,
        "minimum_upright_dot_at_least_0_5": min_upright >= MIN_UPRIGHT_DOT,
        "final_horizontal_drift_at_most_0_2_m": drift <= MAX_FINAL_DRIFT_M,
    }
    return {
        "schema": "legacy_alpha_stand_behavior_gate_v1",
        "source_report_sha256": report_sha256,
        "model_file_sha256": report.get("model_file_sha256"),
        "policy_model_sha256": report.get("policy_model_sha256"),
        "scope": "historical_alpha_stand_threshold_only_not_BAM_or_skill_qualification",
        "checks": checks,
        "behavior_passed": all(checks.values()),
        "skill_qualified": False,
        "minimum_upright_dot": min_upright,
        "minimum_upright_step": min_upright_step,
        "first_upright_below_0_5_step": first_below_threshold,
        "final_horizontal_drift_m": drift,
        "evaluated_pose_frames": len(frames),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="SHA-bound original alpha_stand diagnostic JSON")
    parser.add_argument("output", type=Path, help="new verdict JSON path")
    args = parser.parse_args(argv)
    raw = args.report.read_bytes()
    result = evaluate(json.loads(raw), hashlib.sha256(raw).hexdigest())
    if args.output.exists():
        parser.error("output already exists; keep the previous verdict immutable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"behavior_passed": result["behavior_passed"],
                      "verdict": str(args.output)}, sort_keys=True))
    return 0 if result["behavior_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
