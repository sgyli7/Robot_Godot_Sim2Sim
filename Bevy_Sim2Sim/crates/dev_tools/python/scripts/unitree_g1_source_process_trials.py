#!/usr/bin/env python3
"""Prepare or explicitly run unchanged Homie standing in separate processes.

``prepare`` is CPU-only and never imports the simulator. ``run`` requires a
separately coordinated GPU window. Each trial invokes the frozen original
standing harness with repeats=1; physics, initial state, and reset are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def command(plan: dict, index: int) -> list[str]:
    return [plan["python"], plan["harness"], "--arena-source", plan["arena_source"],
            "--lab-source", plan["lab_source"], "--homie-assets", plan["homie_assets"],
            "--output", str(Path(plan["output_dir"]) / f"trial_{index:02d}.json"),
            "--run-source", "--ticks", str(plan["ticks"]), "--repeats", "1",
            "--seed", str(plan["seed"])]


def prepare(args) -> None:
    # Refuse reuse: previous traces must never be overwritten by a new attempt.
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.harness.resolve()
    frozen = output / "frozen_source_stand.py"
    with frozen.open("xb") as stream:
        stream.write(source.read_bytes())
    frozen.chmod(0o444)
    plan = {
        "kind": "homie_fresh_process_standing_plan", "schema_version": 1,
        "gpu_started": False, "source_stand_verified": False,
        "python": str(args.python.resolve()), "harness": str(frozen),
        "harness_sha256": digest(frozen), "original_harness": str(source),
        "arena_source": str(args.arena_source.resolve()),
        "lab_source": str(args.lab_source.resolve()),
        "homie_assets": str(args.homie_assets.resolve()), "output_dir": str(output),
        "trials": args.trials, "ticks": args.ticks, "seed": args.seed,
        "repeats_per_process": 1, "timeout_seconds_per_process": args.timeout_seconds,
        "process_environment": {
            "LD_PRELOAD": "/lib/aarch64-linux-gnu/libgomp.so.1",
            "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
            "ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE": "1", "PYTHONUNBUFFERED": "1",
        },
        "comparison": "Fresh processes versus existing same-process repeats; no dynamics or reset changes",
    }
    plan["commands"] = [command(plan, i) for i in range(args.trials)]
    write_new(output / "plan.json", plan)
    print(json.dumps({"plan": str(output / "plan.json"), "harness_sha256": plan["harness_sha256"],
                      "trials": args.trials, "gpu_started": False}), flush=True)


def stop_owned_group(process: subprocess.Popen) -> None:
    """Only signal the fresh session created for this child process."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Kit can exit its parent before a worker. This group contains only the
    # subprocess session we created, never the desktop or another agent's job.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(args) -> None:
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("kind") != "homie_fresh_process_standing_plan" or plan.get("schema_version") != 1:
        raise ValueError("Unsupported process-trial plan")
    if digest(Path(plan["harness"])) != plan["harness_sha256"]:
        raise ValueError("Frozen harness identity changed")
    if plan["repeats_per_process"] != 1 or plan["commands"] != [command(plan, i) for i in range(plan["trials"])]:
        raise ValueError("Plan does not match separate-process trial contract")
    output = Path(plan["output_dir"])
    # An exclusive claim protects all outputs, including interrupted attempts.
    write_new(output / "execution_claim.json", {"plan_sha256": digest(plan_path), "pid": os.getpid()})
    if any((output / f"trial_{i:02d}{suffix}").exists()
           for i in range(plan["trials"]) for suffix in [".json", ".jsonl", ".log"]):
        raise FileExistsError("A trial output already exists; prepare a new directory")
    environment = os.environ.copy()
    environment.update(plan["process_environment"])
    results = []
    for i in range(plan["trials"]):
        log_path = output / f"trial_{i:02d}.log"
        started = time.monotonic()
        result = {"trial": i, "source_stand_verified": False}
        with log_path.open("x") as log:
            process = subprocess.Popen(command(plan, i), cwd=plan["arena_source"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["pid"] = process.pid
            print(f"EVENT fresh-process trial={i} pid={process.pid} started", flush=True)
            try:
                result["exit_code"] = process.wait(timeout=plan["timeout_seconds_per_process"])
            except subprocess.TimeoutExpired:
                result["timeout"] = True
                stop_owned_group(process)
                result["exit_code"] = process.returncode
            except BaseException:
                stop_owned_group(process)
                raise
            finally:
                stop_owned_group(process)
        result["seconds"] = time.monotonic() - started
        receipt_path = output / f"trial_{i:02d}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            result["source_stand_verified"] = bool(
                result["exit_code"] == 0 and receipt.get("source_stand_verified")
                and len(receipt.get("rollouts", [])) == 1
                and receipt["rollouts"][0].get("completed_ticks") == plan["ticks"])
        result["files_sha256"] = {suffix: digest(output / f"trial_{i:02d}{suffix}")
                                  for suffix in [".json", ".jsonl", ".log"]
                                  if (output / f"trial_{i:02d}{suffix}").exists()}
        write_new(output / f"trial_{i:02d}_result.json", result)
        results.append(result)
        print(f"EVENT fresh-process trial={i} exit={result['exit_code']} passed={result['source_stand_verified']}", flush=True)
    passed = all(result["source_stand_verified"] for result in results)
    write_new(output / "results.json", {"plan_sha256": digest(plan_path), "trials": results,
                                       "fresh_process_stand_verified": passed,
                                       "same_process_reset_verified": False, "bevy_qualified": False})
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare", help="CPU-only plan creation; no simulator execution")
    for name in ["python", "harness", "arena-source", "lab-source", "homie-assets", "output-dir"]:
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--trials", type=int, default=2)
    prep.add_argument("--ticks", type=int, default=150)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--timeout-seconds", type=int, default=360)
    execute = modes.add_parser("run", help="Explicitly run a prepared plan in a coordinated GPU window")
    execute.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.trials < 1 or args.ticks < 1 or args.timeout_seconds < 1:
            parser.error("Trial count, ticks, and timeout must be positive")
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
