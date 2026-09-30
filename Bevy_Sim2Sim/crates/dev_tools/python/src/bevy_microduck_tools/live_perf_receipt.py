"""CPU-only reading of a station_robot_live_preview_v3 report.

No window, GPU, physics step, or policy runs here. A source ``passed`` value
of true only means the live wiring ledger closed. This receipt always writes
``performance_qualified`` and ``skill_qualified`` as false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "station_robot_live_preview_v3"
RECEIPT_SCHEMA = "station_robot_live_perf_receipt_v2"
TICK_FIELDS = (
    "policy_inference_count",
    "integration_count",
    "completed_publishes",
    "last_displayed_step",
)
RAPIER_STAGE_NS = (
    "step_ns",
    "broad_phase_ns",
    "final_broad_phase_ns",
    "narrow_phase_ns",
    "islands_ns",
    "constraints_ns",
    "solver_ns",
    "ccd_ns",
    "user_changes_ns",
)
RAPIER_EXTRA_STAGE_NS = ("ccd_toi_ns", "update_ns", "collision_detection_ns")


def _distribution(samples: list[int]) -> dict[str, Any]:
    ordered = sorted(samples)
    if not ordered:
        return {"count": 0, "p50_ns": None, "p95_ns": None, "p99_ns": None, "max_ns": None}
    return {
        "count": len(ordered),
        **{f"p{p}_ns": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
           for p in (50, 95, 99)},
        "max_ns": ordered[-1],
    }


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _timing(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return {
        "count": _require_int(value.get("count"), f"{label}.count"),
        "sum_ns": _require_int(value.get("sum_ns"), f"{label}.sum_ns"),
        "max_ns": _require_int(value.get("max_ns"), f"{label}.max_ns"),
    }


def _require_sequence(rows: object, label: str, key: str, expected: range) -> None:
    if not isinstance(rows, list) or len(rows) != len(expected):
        observed = len(rows) if isinstance(rows, list) else "invalid"
        raise ValueError(f"{label} length {observed} must be {len(expected)}")
    for index, step in enumerate(expected):
        row = rows[index]
        if not isinstance(row, dict) or type(row.get(key)) is not int or row[key] != step:
            raise ValueError(f"{label} {key} is not contiguous at index {index}")


def _display_fps(frame_start: dict[str, Any]) -> float | None:
    # frame_start_to_start.count * 1e9 / sum_ns. A zero sum is not a frame rate.
    if frame_start["sum_ns"] == 0:
        return None
    return frame_start["count"] * 1_000_000_000 / frame_start["sum_ns"]


def _advance_wall(summary: dict[str, Any]) -> dict[str, Any]:
    count = summary["count"]
    mean_ns = None if count == 0 else summary["sum_ns"] / count
    return {"count": count, "mean_ns": mean_ns, "max_ns": summary["max_ns"]}


def _deadline_lag(summary: dict[str, Any], missed_ticks: int) -> dict[str, Any]:
    if missed_ticks > summary["count"]:
        raise ValueError("deadline_missed_ticks exceeds measured ticks")
    return {
        "count": summary["count"],
        "max_ns": summary["max_ns"],
        "over_zero_count": missed_ticks,
    }


def _rapier(profile: dict[str, Any], ticks: int) -> dict[str, Any]:
    compiled = _require_bool(profile.get("rapier_profiler_compiled"), "rapier_profiler_compiled")
    rows = profile.get("rapier_counters_per_tick")
    if not isinstance(rows, list):
        raise ValueError("rapier_counters_per_tick must be an array")
    if not compiled:
        return {"status": "unavailable", "reason": "rapier_profiler_not_compiled"}
    if len(rows) != ticks:
        raise ValueError(
            f"rapier counter row count {len(rows)} must equal completed ticks {ticks}"
        )
    stage_names = list(RAPIER_STAGE_NS)
    if any(any(name in row for name in RAPIER_EXTRA_STAGE_NS)
           for row in rows if isinstance(row, dict)):
        stage_names.extend(RAPIER_EXTRA_STAGE_NS)
    stages = {name: {"sum_ns": 0, "max_ns": 0} for name in stage_names}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rapier counter row {index} must be an object")
        for name in stage_names:
            sample = _require_int(row.get(name), f"rapier {name} at row {index}")
            stages[name]["sum_ns"] += sample
            stages[name]["max_ns"] = max(stages[name]["max_ns"], sample)
    return {"status": "compiled", "row_count": len(rows), "stages": stages}


def _sample_metrics(report: dict[str, Any], profile: dict[str, Any], ticks: int,
                    frames: dict[str, Any], advance: dict[str, Any],
                    lag: dict[str, Any], missed_ticks: int) -> dict[str, Any]:
    result = {"frame_intervals": None, "worker_tick_wall": None}
    if "frame_samples" in profile:
        rows = profile["frame_samples"]
        count = _require_int(report.get("display_frames"), "display_frames", minimum=1)
        _require_sequence(rows, "frame_samples", "display_frame", range(count))
        intervals = []
        previous_start = None
        for row in rows:
            start = _require_int(row.get("start_since_run_ns"), "frame start")
            interval = row.get("previous_interval_ns")
            if previous_start is None:
                if interval is not None:
                    raise ValueError("first frame cannot have a previous interval")
            else:
                interval = _require_int(interval, "frame interval", minimum=1)
                if start - previous_start != interval:
                    raise ValueError("frame interval disagrees with frame start timestamps")
                intervals.append(interval)
            previous_start = start
        if (len(intervals), sum(intervals), max(intervals, default=0)) != (
                frames["count"], frames["sum_ns"], frames["max_ns"]):
            raise ValueError("frame samples disagree with frame timing summary")
        result["frame_intervals"] = _distribution(intervals) | {
            "over_20ms": sum(v > 20_000_000 for v in intervals),
            "over_33_33ms": sum(v > 33_333_334 for v in intervals),
        }
    if "worker_tick_samples" in profile:
        rows = profile["worker_tick_samples"]
        _require_sequence(rows, "worker_tick_samples", "global_step", range(1, ticks + 1))
        wall = [_require_int(row.get("advance_cpu_ns"), "worker tick wall time") for row in rows]
        lags = [_require_int(row.get("deadline_lag_ns"), "worker tick deadline lag") for row in rows]
        if (len(wall), sum(wall), max(wall, default=0)) != (
                advance["count"], advance["sum_ns"], advance["max_ns"]):
            raise ValueError("worker tick wall samples disagree with timing summary")
        if (len(lags), sum(lags), max(lags, default=0), sum(v > 0 for v in lags)) != (
                lag["count"], lag["sum_ns"], lag["max_ns"], missed_ticks):
            raise ValueError("worker tick deadline samples disagree with timing summary")
        result["worker_tick_wall"] = _distribution(wall)
    return result


def _profile_metrics(report: dict[str, Any], ticks: int, missed_ticks: int,
                     max_lag_ns: int) -> dict[str, Any]:
    if "live_profile" not in report:
        return {
            "display_fps": None,
            "advance_frame_wall": None,
            "worker_deadline_lag": None,
            "rapier_counters": {"status": "unavailable", "reason": "live_profile_absent"},
            "frame_intervals": None,
            "worker_tick_wall": None,
        }
    profile = report["live_profile"]
    if not isinstance(profile, dict):
        raise ValueError("live_profile must be an object")
    frames = _timing(profile.get("frame_start_to_start"), "frame_start_to_start")
    advance = _timing(profile.get("advance_frame_cpu"), "advance_frame_cpu")
    lag = _timing(profile.get("worker_deadline_lag"), "worker_deadline_lag")
    if lag["count"] != ticks or lag["max_ns"] != max_lag_ns:
        raise ValueError("profile deadline lag disagrees with worker receipt")
    if _require_int(profile.get("deadline_missed_ticks"), "deadline_missed_ticks") != missed_ticks:
        raise ValueError("profile deadline misses disagree with worker receipt")
    samples = _sample_metrics(report, profile, ticks, frames, advance, lag, missed_ticks)
    return {
        "display_fps": _display_fps(frames),
        "advance_frame_wall": _advance_wall(advance),
        "worker_deadline_lag": _deadline_lag(lag, missed_ticks),
        "rapier_counters": _rapier(profile, ticks),
        **samples,
    }


def analyze(report: dict[str, Any], *, source_sha256: str | None = None) -> dict[str, Any]:
    """Validate one v3 live report and return a non-qualifying performance receipt."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if report.get("schema") != SOURCE_SCHEMA:
        raise ValueError("input is not a station_robot_live_preview_v3 report")
    if source_sha256 is not None and (
            not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")

    source_passed = _require_bool(report.get("passed"), "passed")
    if not source_passed:
        raise ValueError("source report did not pass its wiring ledger")
    if _require_bool(report.get("performance_qualified"), "performance_qualified"):
        raise ValueError("diagnostic source cannot claim performance qualification")
    if _require_bool(report.get("skill_qualified"), "skill_qualified"):
        raise ValueError("diagnostic source cannot claim skill qualification")
    minimum = _require_int(report.get("minimum_ticks"), "minimum_ticks", minimum=1)
    actual = _require_int(report.get("actual_completed_ticks"), "actual_completed_ticks")
    if actual != minimum:
        raise ValueError(
            f"actual_completed_ticks {actual} must equal minimum_ticks {minimum}"
        )
    for field in TICK_FIELDS:
        value = _require_int(report.get(field), field)
        if value != actual:
            raise ValueError(f"{field} {value} must equal completed ticks {actual}")
    if _require_bool(report.get("worker_finished"), "worker_finished") is not True:
        raise ValueError("worker_finished must be true")
    worker_wall_ns = _require_int(report.get("worker_wall_ns"), "worker_wall_ns", minimum=1)
    missed_ticks = _require_int(report.get("worker_deadline_missed_ticks"), "worker_deadline_missed_ticks")
    max_lag_ns = _require_int(report.get("worker_max_deadline_lag_ns"), "worker_max_deadline_lag_ns")
    if missed_ticks > actual or (missed_ticks == 0 and max_lag_ns != 0):
        raise ValueError("worker deadline counts and maximum lag disagree")
    deadline_met = _require_bool(report.get("worker_60hz_deadline_met"), "worker_60hz_deadline_met")
    if deadline_met and (missed_ticks != 0 or worker_wall_ns > (actual * 1_000_000_000 + 59) // 60):
        raise ValueError("worker_60hz_deadline_met conflicts with timings")

    # The initial pose is global_step 0, so the pose ledger is one longer than the ticks.
    _require_sequence(report.get("pose_frames"), "pose_frames", "global_step", range(actual + 1))
    _require_sequence(report.get("trace"), "trace", "tick_before", range(actual))
    clock = report.get("final_clock")
    if not isinstance(clock, dict) or type(clock.get("global_step")) is not int or clock["global_step"] != actual:
        raise ValueError("final_clock global_step must equal completed ticks")
    pending_ticks = _require_int(report.get("pending_ticks"), "pending_ticks")
    if pending_ticks != 0 or clock.get("pending_ticks") != pending_ticks:
        raise ValueError("completed worker has pending fixed-step debt")

    metrics = _profile_metrics(report, actual, missed_ticks, max_lag_ns)
    # passed=true is the wiring ledger only. Never copy it into either qualification.
    return {
        "schema": RECEIPT_SCHEMA,
        "source_report_sha256": source_sha256,
        "source_schema": SOURCE_SCHEMA,
        "source_passed": source_passed,
        "passed_means": "wiring_only",
        "ticks": actual,
        "worker_finished": True,
        "worker_wall_ns": worker_wall_ns,
        "worker_deadline_missed_ticks": missed_ticks,
        "worker_max_deadline_lag_ns": max_lag_ns,
        "worker_60hz_deadline_met": deadline_met,
        "physics_throughput_hz": actual * 1_000_000_000 / worker_wall_ns,
        "display_fps": metrics["display_fps"],
        "worker_advance_timing_semantics": "wall_time_including_descheduling; legacy_source_field_names_contain_cpu",
        "advance_frame_wall": metrics["advance_frame_wall"],
        "worker_deadline_lag": metrics["worker_deadline_lag"],
        "rapier_counters": metrics["rapier_counters"],
        "frame_intervals": metrics["frame_intervals"],
        "worker_tick_wall": metrics["worker_tick_wall"],
        "performance_qualified": False,
        "skill_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="station_robot_live_preview_v3 JSON report")
    parser.add_argument(
        "output", nargs="?", type=Path,
        help="new receipt JSON path; an existing path is left untouched",
    )
    args = parser.parse_args(argv)
    try:
        raw = args.report.read_bytes()
    except OSError as error:
        print(f"cannot read report: {error}", file=sys.stderr)
        return 1
    try:
        document = json.loads(raw)
        receipt = analyze(document, source_sha256=hashlib.sha256(raw).hexdigest())
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
        return 0
    if args.output.exists():
        print("output already exists; refusing to overwrite", file=sys.stderr)
        return 1
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(text)
    except FileExistsError:
        print("output already exists; refusing to overwrite", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"cannot write receipt: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
