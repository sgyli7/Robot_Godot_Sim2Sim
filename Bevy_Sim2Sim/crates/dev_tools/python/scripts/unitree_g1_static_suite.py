#!/usr/bin/env python3
"""Run a frozen5position x2seed native station T1 profile suite once.

This intermediate suite uses explicit camera/model pauses. It does not qualify
Qwen task selection or continuous real-time execution. Input manifest and tool
hashes are checked before creating a physical owner; failures are retained.
Requires the local numpy/PIL audit runtime and the isolated T1 model runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from unitree_g1_static_capture_audit import audit

REVISION = "7f78bebf1a90131e7304beacfcd47eb27bad16ab"
SCRIPTS = Path(__file__).resolve().parent


class SuiteInterrupted(BaseException):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_port_available(port=5557):
    # Match ThreadingHTTPServer's reuse policy. A completed seed group can leave
    # accepted connections in TIME_WAIT; that is not an active model owner.
    # An active listener still causes bind to fail and is never stopped here.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def retained_cases(directory, expected_summary_sha, manifest, binary_sha):
    """Re-audit a completed seed0 group without repeating physical execution."""
    summary_path = directory / "summary.json"
    parent_path = directory / "frozen_manifest.json"
    if digest(summary_path) != expected_summary_sha:
        raise ValueError("Retained partial summary changed")
    parent = json.loads(parent_path.read_text())
    old = json.loads(summary_path.read_text())
    if (digest(parent_path) != manifest.get("parent_manifest_sha256")
            or old["manifest_sha256"] != digest(parent_path)
            or old["complete"] or old["binary_sha256"] != binary_sha):
        raise ValueError("Continuation does not identify the incomplete original run")
    for key in ("cases", "acceptance", "profile", "revision", "seed_offsets",
                "physics_hz", "integrations_per_tick", "startup_ticks", "task_ticks",
                "max_vla_calls", "model_receipt_sha256", "pauses_for_camera_and_policy"):
        if parent[key] != manifest[key]:
            raise ValueError(f"Continuation changed frozen case or acceptance: {key}")
    for name, checksum in parent["tools_sha256"].items():
        if name != "unitree_g1_static_suite.py" and manifest["tools_sha256"].get(name) != checksum:
            raise ValueError("Continuation changed model or audit implementation")
    results = old["cases"]
    if (len(results) != 5 or {(c["position_id"], c["seed_offset"]) for c in results}
            != {(p, 0) for p in range(5)}):
        raise ValueError("Only a completed five-case seed0 group can be resumed")
    lifecycle = json.loads((directory / "seed_offset_0/model_lifecycle.json").read_text())
    if (not lifecycle["owned_model_closed"] or lifecycle["initial_health"]["successful_inferences"] != 0
            or lifecycle["final_health"]["successful_inferences"] != 40
            or lifecycle["final_health"]["failed_inferences"] != 0):
        raise ValueError("Retained model owner or actual inference counter is incomplete")
    for result in results:
        case = directory / f"position_{result['position_id']}_seed_0"
        spec = next(c for c in manifest["cases"]
                    if (c["position_id"], c["seed_offset"]) == (result["position_id"], 0))
        execution = json.loads((case / "execution.json").read_text())
        if (json.loads((case / "config.json").read_text()) != spec["configuration"]
                or execution["code_commit"] != old["code_commit"]
                or execution["binary_sha256"] != binary_sha
                or execution["frozen_manifest_sha256"] != old["manifest_sha256"]
                or execution["result"] != result
                or not result["evidence_verified"] or result["actual_integrations"] != 380
                or result["fresh_vla_calls"] != 8 or not result["model_counter_verified"]):
            raise ValueError("Retained case identity or actual execution is incomplete")
        verified = audit(case, 0)
        for key in ("strict_placement_passed", "trace_sha256", "standing_all_ticks",
                    "max_continuous_placement_seconds", "minimum_margin_m"):
            if verified[key] != result[key]:
                raise ValueError(f"Retained case failed independent re-audit: {key}")
    return results, {"directory": str(directory), "summary_sha256": expected_summary_sha,
                     "manifest_sha256": digest(parent_path), "code_commit": old["code_commit"],
                     "reexecuted_cases": 0, "actual_integrations": 1900, "fresh_vla_calls": 40}


def run(args):
    if digest(args.manifest) != args.manifest_sha256:
        raise ValueError("Frozen suite manifest checksum changed")
    manifest = json.loads(args.manifest.read_text())
    if (manifest["schema"] != "g1_native_station_static_profile_suite_v1"
            or manifest["profile"] != "static_apple" or manifest["revision"] != REVISION
            or manifest["physics_hz"] != 50 or manifest["integrations_per_tick"] != 1
            or manifest["startup_ticks"] != 60 or manifest["task_ticks"] != 320
            or manifest["max_vla_calls"] != 8 or manifest["seed_offsets"] != [0, 42]
            or manifest["full_task_qualified"] or not manifest["pauses_for_camera_and_policy"]):
        raise ValueError("Suite changed the frozen matched profile or overstated qualification")
    rules = {"required_successes": 8, "cases": 10, "linear_speed_max_m_s": 0.02,
             "angular_speed_max_rad_s": 0.1, "continuous_placement_seconds": 2.0,
             "all_apple_collision_vertices_inside_actual_plate": True,
             "released_from_robot": True, "positive_actual_plate_support": True,
             "standing_without_fall": True, "application_timeout_seconds": 75}
    if manifest["acceptance"] != rules:
        raise ValueError("Placement thresholds or timeout changed after freezing")
    for name, expected in manifest["tools_sha256"].items():
        if name not in {"unitree_g1_static_suite.py", "unitree_g1_static_server.py",
                        "unitree_g1_static_capture_audit.py", "unitree_g1_static_placement_audit.py"}:
            raise ValueError("Unknown executable in the frozen tool set")
        if digest(SCRIPTS / name) != expected:
            raise ValueError(f"Executed tool changed after freezing: {name}")
    build = json.loads(args.build_receipt.read_text())
    binary = args.binary.resolve()
    if build["exit_code"] != 0 or digest(binary) != build["binary_sha256"]:
        raise ValueError("App binary does not match successful build receipt")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.workspace, text=True).strip()
    if (head != build["head"] or subprocess.check_output(["git", "status", "--porcelain"], cwd=args.workspace, text=True)):
        raise ValueError("Build/source identity changed or integration checkout is dirty")
    if digest(args.model_receipt) != manifest["model_receipt_sha256"]:
        raise ValueError("Original five-graph model receipt changed")
    runtime = subprocess.run([str(args.model_python), "-c",
        "import sys,json,numpy,onnxruntime;print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'numpy':numpy.__version__,'onnxruntime':onnxruntime.__version__,'providers':onnxruntime.get_available_providers()}))"],
        capture_output=True, text=True, timeout=30, check=True)
    runtime_identity = json.loads(runtime.stdout)
    if (Path(runtime_identity["prefix"]).resolve() != args.model_python.parent.parent.resolve()
            or "CUDAExecutionProvider" not in runtime_identity["providers"]):
        raise ValueError("Model launcher does not select its isolated CUDA environment")
    cases = manifest["cases"]
    expected_pairs = {(position, seed) for position in range(5) for seed in [0, 42]}
    if len(cases) != 10 or {(c["position_id"], c["seed_offset"]) for c in cases} != expected_pairs:
        raise ValueError("Suite must contain exactly5fixed positions x2distinct seeds")
    episodes = set()
    for case in cases:
        config = case["configuration"]
        body = config["runner"]["body"]["static_agile"]
        episode = body["episode_id"]
        if episode in episodes:
            raise ValueError("Cases repeat a physical episode identity")
        episodes.add(episode)
        if (not config["static_startup"] or not config.get("station")
                or config.get("background_visual") or config["policy"]["max_calls"] != 8
                or config["policy"]["endpoint"] != "http://127.0.0.1:5557/infer"
                or config["diagnostic_constraint_sweeps"] != 16
                or not config["predictive_limit_diagnostic"]):
            raise ValueError("Case changed its unique native station/body/VLA owner")
    retained, origin = [], None
    if args.resume_output is not None:
        if not args.resume_summary_sha256:
            raise ValueError("Continuation requires the pinned partial summary checksum")
        retained, origin = retained_cases(args.resume_output, args.resume_summary_sha256,
                                           manifest, digest(binary))
    elif args.resume_summary_sha256:
        raise ValueError("Partial summary checksum requires its retained output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, args.output / "frozen_manifest.json")
    shutil.copy2(args.build_receipt, args.output / "build.json")
    save(args.output / "model_runtime.json", runtime_identity)
    summary = {"schema": "g1_native_static_profile_suite_result_v1", "complete": False,
               "manifest_sha256": args.manifest_sha256, "code_commit": head,
               "binary_sha256": digest(binary), "cases": retained,
               "successes": sum(int(c["strict_placement_passed"]) for c in retained),
               "expected_cases": 10, "required_successes": 8,
               "actual_integrations": sum(c["actual_integrations"] for c in retained),
               "fresh_vla_calls": sum(c["fresh_vla_calls"] for c in retained),
               "retained_run": origin, "new_actual_integrations": 0, "new_fresh_vla_calls": 0,
               "pauses_for_camera_and_policy": True, "qwen_target_selection": False,
               "continuous_real_time_qualified": False, "full_task_qualified": False}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def health():
        with opener.open("http://127.0.0.1:5557/health", timeout=2) as response:
            return json.load(response)
    def interrupted(signum, frame):
        raise SuiteInterrupted(f"owned suite interrupted by signal {signum}")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    try:
        for seed in [0, 42]:
            if retained and seed == 0:
                continue
            group = args.output / f"seed_offset_{seed}"
            group.mkdir()
            shared_captures = group / "policy_captures"
            command = [str(args.model_python), "-u", str(SCRIPTS / "unitree_g1_static_server.py"),
                       "--receipt", str(args.model_receipt), "--device", "cuda", "--port", "5557",
                       "--seed-offset", str(seed), "--capture-dir", str(shared_captures)]
            lifecycle = {"command": command, "seed_offset": seed, "owned_model_closed": False}
            server = None
            with (group / "model.log").open("x") as log:
                try:
                    check_port_available()
                    server = subprocess.Popen(command, cwd=args.workspace, stdout=log, stderr=subprocess.STDOUT)
                    lifecycle["owned_model_pid"] = server.pid
                    deadline = time.monotonic() + 90
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError("Owned N1.7 model exited before readiness")
                        try:
                            initial = health()
                            if (initial["revision"] != REVISION or len(initial["loaded_graphs"]) != 5
                                    or initial["seed_offset"] != seed or initial["successful_inferences"] != 0):
                                raise ValueError("Original model/seed/counter identity mismatch")
                            lifecycle["initial_health"] = initial
                            break
                        except (OSError, urllib.error.URLError):
                            if time.monotonic() > deadline:
                                raise TimeoutError("Original model load exceeded90seconds")
                            time.sleep(0.2)
                    for spec in [c for c in cases if c["seed_offset"] == seed]:
                        case = args.output / f"position_{spec['position_id']}_seed_{seed}"
                        case.mkdir()
                        config = case / "config.json"
                        save(config, spec["configuration"])
                        episode = spec["configuration"]["runner"]["body"]["static_agile"]["episode_id"]
                        before = health()
                        app = [str(binary), "--scene", "g1_camera_diagnostic", "--robot", "g1",
                               "--g1-config", str(config), "--g1-ticks", "380", "--output", str(case / "native")]
                        result = {"position_id": spec["position_id"], "seed_offset": seed, "episode_id": episode,
                                  "strict_placement_passed": False, "evidence_verified": False,
                                  "actual_integrations": 0, "fresh_vla_calls": 0}
                        execution = {"code_commit": head, "working_tree_dirty": False,
                                     "binary_sha256": build["binary_sha256"], "config_sha256": digest(config),
                                     "model_command": command, "app_command": app,
                                     "frozen_manifest_sha256": args.manifest_sha256,
                                     "case": result.copy(), "formal_full_task_qualified": False}
                        save(case / "execution.json", execution)
                        case_start = time.monotonic()
                        try:
                            print(f"EVENT native_T1_case_started position={spec['position_id']} seed_offset={seed} episode={episode}", flush=True)
                            with (case / "app.log").open("x") as app_log:
                                child = subprocess.Popen(app, cwd=args.workspace, stdout=app_log, stderr=subprocess.STDOUT)
                                execution["owned_app_pid"] = child.pid
                                try:
                                    child.wait(timeout=90)
                                finally:
                                    if child.poll() is None:
                                        child.terminate()
                                        try:
                                            child.wait(timeout=10)
                                        except subprocess.TimeoutExpired:
                                            child.kill()
                                            child.wait(timeout=5)
                            execution["app_exit_code"] = child.returncode
                            after = health()
                            result["fresh_vla_calls"] = after["successful_inferences"] - before["successful_inferences"]
                            native_receipt = case / "native/capture_receipt.json"
                            if native_receipt.exists():
                                result["actual_integrations"] = json.loads(native_receipt.read_text())["actual_integrations"]
                            captures = case / "policy_captures"
                            captures.mkdir()
                            for sample in shared_captures.glob(f"e{episode}_f*_q*"):
                                shutil.copytree(sample, captures / sample.name)
                            if child.returncode != 0:
                                raise RuntimeError("Native app failed; actual partial counts and log retained")
                            if result["fresh_vla_calls"] != 8:
                                raise ValueError("Original inference count does not match frozen eight-call budget")
                            geometry = spec["configuration"]["runner"]["body"]["static_agile"]["task_objects"]["definition"]
                            with (case / "placement_audit.log").open("x") as audit_log:
                                check = subprocess.run([sys.executable, str(SCRIPTS / "unitree_g1_static_placement_audit.py"),
                                    "--definition", geometry, "--trace", str(case / "native/owner_steps.jsonl"),
                                    "--output", str(case / "placement_geometry_audit.json")],
                                    stdout=audit_log, stderr=subprocess.STDOUT, timeout=30)
                            if check.returncode not in [0, 1]:
                                raise RuntimeError("Independent placement auditor could not complete")
                            verified = audit(case, seed)
                            save(case / "independent_capture_audit.json", verified)
                            result.update({key: verified[key] for key in ["strict_placement_passed",
                                "max_continuous_placement_seconds", "minimum_margin_m", "standing_all_ticks",
                                "active_sim_wall_ratio", "continuous_boundary_sim_wall_ratio",
                                "control_deadlines_missed", "pending_ticks", "trace_sha256"]})
                            result["evidence_verified"] = True
                        except Exception as error:
                            result["error"] = repr(error)
                        except SuiteInterrupted as error:
                            result["error"] = repr(error)
                            raise
                        finally:
                            try:
                                final_health = health()
                                result["fresh_vla_calls"] = final_health["successful_inferences"] - before["successful_inferences"]
                                result["failed_vla_attempts"] = final_health["failed_inferences"] - before["failed_inferences"]
                                result["model_counter_verified"] = True
                            except Exception as error:
                                result["model_counter_verified"] = False
                                result["model_counter_error"] = repr(error)
                            native_receipt = case / "native/capture_receipt.json"
                            result["actual_integration_count_complete"] = native_receipt.exists()
                            if native_receipt.exists():
                                result["actual_integrations"] = json.loads(native_receipt.read_text())["actual_integrations"]
                            else:
                                trace = case / "native/owner_steps.jsonl"
                                if trace.exists():
                                    recorded = [json.loads(line) for line in trace.read_text().splitlines()]
                                    if recorded:
                                        result["actual_integrations"] = recorded[-1]["owner_episode_integrations"]
                                        result["count_is_recorded_lower_bound"] = True
                            result["wall_seconds"] = time.monotonic() - case_start
                            execution["result"] = result
                            save(case / "execution.json", execution)
                            summary["cases"].append(result)
                            summary["successes"] += int(result["strict_placement_passed"] and result["evidence_verified"])
                            summary["actual_integrations"] += result["actual_integrations"]
                            summary["fresh_vla_calls"] += result["fresh_vla_calls"]
                            summary["new_actual_integrations"] += result["actual_integrations"]
                            summary["new_fresh_vla_calls"] += result["fresh_vla_calls"]
                            save(args.output / "summary.json", summary)
                        print(f"EVENT native_T1_case_completed position={spec['position_id']} seed_offset={seed} passed={result['strict_placement_passed']} verified={result['evidence_verified']} ticks={result['actual_integrations']} vla={result['fresh_vla_calls']}", flush=True)
                finally:
                    if server is not None:
                        try:
                            lifecycle["final_health"] = health()
                        except Exception as error:
                            lifecycle["final_health_error"] = repr(error)
                        if server.poll() is None:
                            server.terminate()
                        try:
                            server.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=5)
                        lifecycle.update(owned_model_closed=True, model_exit_code=server.returncode)
                    save(group / "model_lifecycle.json", lifecycle)
        summary["complete"] = len(summary["cases"]) == 10
        summary["profile_score_threshold_passed"] = summary["complete"] and summary["successes"] >= 8
    except BaseException as error:
        summary["error"] = repr(error)
    finally:
        summary["wall_seconds"] = time.monotonic() - start
        save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-output", type=Path)
    parser.add_argument("--resume-summary-sha256")
    args = parser.parse_args()
    # Resolving a venv's bin/python symlink selects the base interpreter and
    # loses its site-packages. Preserve that launcher path deliberately.
    args.model_python = args.model_python.absolute()
    for name in ["manifest", "binary", "build_receipt", "model_receipt", "workspace", "output"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.resume_output is not None:
        args.resume_output = args.resume_output.resolve()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
