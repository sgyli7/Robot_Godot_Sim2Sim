#!/usr/bin/env python3
"""Run one frozen five-position x two-seed native station T2 suite.

No case retries, threshold changes, model warmups or synthetic actions. Owns
only its child N1.6/app processes; the prestarted owned Qwen service is pinned.
Camera/model pauses are disclosed, so this score alone cannot qualify full 1x.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from unitree_g1_mobile_capture_audit import REVISION, audit, digest

SCRIPTS = Path(__file__).resolve().parent
RULES = {"cases": 10, "required_successes": 8, "linear_speed_max_m_s": .02,
         "angular_speed_max_rad_s": .1, "final_continuous_placement_seconds": 2.,
         "source_target_distance_min_m": 1.8, "box_displacement_min_m": 1.8,
         "robot_displacement_min_m": .8, "all_box_vertices_inside_bin_floor_prism": True,
         "upward_actual_bin_support": True, "released_from_robot": True,
         "all_postgrasp_pre_release_hand_only_support": True, "standing_without_fall": True,
         "no_nonfoot_fixed_world_support": True, "three_fresh_local_qwen_decisions": True,
         "fresh_after_thumb_release_rgb": True, "application_timeout_seconds": 120,
         "outer_timeout_seconds": 150, "maximum_native_integrations": 3300,
         "maximum_fresh_vla_calls": 4, "maximum_qwen_calls": 3}
TOOLS = {"unitree_g1_mobile_suite.py", "unitree_g1_mobile_capture_audit.py",
         "unitree_g1_mobile_placement_audit.py", "unitree_g1_static_placement_audit.py",
         "unitree_g1_joint_kinematics_audit.py", "unitree_g1_mobile_server.py",
         "unitree_g1_mobile_forward.py", "unitree_g1_policy_prepare.py", "unitree_g1_static_server.py"}


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate(manifest, scripts=SCRIPTS):
    if (manifest["schema"] != "g1_native_station_mobile_profile_suite_v1"
            or manifest["revision"] != REVISION or manifest["physics_hz"] != 50
            or manifest["integrations_per_tick"] != 1 or manifest["seeds"] != [0, 42]
            or manifest["acceptance"] != RULES or manifest["full_task_qualified"]
            or not manifest["pauses_for_camera_and_policy"]):
        raise ValueError("Frozen profile, seeds, rules or qualification changed")
    if set(manifest["tools_sha256"]) != TOOLS:
        raise ValueError("Incomplete frozen tool identities")
    for name, checksum in manifest["tools_sha256"].items():
        if digest(scripts / name) != checksum:
            raise ValueError(f"Frozen executable changed: {name}")
    cases = manifest["cases"]
    if (len(cases) != 10 or {(c["position_id"], c["seed"]) for c in cases}
            != {(position, seed) for position in range(5) for seed in [0, 42]}):
        raise ValueError("Exactly five positions x two seeds required")
    if [(c["seed"], c["position_id"]) for c in cases] != [(s, p) for s in [0, 42] for p in range(5)]:
        raise ValueError("Frozen case order changed")
    episodes = []
    positions = {}
    normalized = []
    for case in cases:
        config = json.loads(json.dumps(case["configuration"]))
        body = config["runner"]["body"]["mobile_homie_v2"]
        episodes.append(body.pop("episode_id"))
        placement = body["task_objects"].pop("placements")
        key = case["position_id"]
        if key in positions and positions[key] != placement:
            raise ValueError("A position pair changed between seeds")
        positions[key] = placement
        if (config["policy"] != {"endpoint": "http://127.0.0.1:5558/infer", "max_calls": 4, "timeout_ms": 20000}
                or config["diagnostic_qwen_dispatch"]["scope"] != "scientific_station_mobile_from_instruction_v1"
                or config["diagnostic_qwen_dispatch"]["connection"] != {
                    "endpoint": "http://127.0.0.1:8002/v1", "model": "qwen3.8-27b-fp8",
                    "timeout_ms": 20000, "max_output_tokens": 128}
                or body["task_objects"]["source_t2_background"]["selection"] != "station_task_fixtures"
                or not config.get("station") or config.get("background_visual", {}).get("sha256")
                != "69129a51ae7fd53ddf9a00cce7170bc780e4d0f4fceff5c720cbe359db7eee9e"):
            raise ValueError("Case changed its native station/local matched model owner")
        normalized.append(config)
    if len(set(episodes)) != 10 or any(c != normalized[0] for c in normalized):
        raise ValueError("Repeated episode or non-placement runtime parameter changed")
    if len({json.dumps(p, sort_keys=True) for p in positions.values()}) != 5:
        raise ValueError("Positions must be five distinct initial pairs")


def close_owned(child, recording=False):
    if child is None:
        return None
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT if recording else signal.SIGTERM)
        try:
            child.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    else:
        child.wait()
    return child.returncode


def record_owned_window(child, directory, ffmpeg):
    deadline = time.monotonic() + 20
    window = None
    tree_retries = 0
    while window is None:
        if child.poll() is not None:
            return None, {"reason": "owned app exited before its window mapped"}
        if time.monotonic() > deadline:
            raise TimeoutError("Owned app window mapping20s")
        # The root tree includes foreign windows. A window closing during the
        # recursive query produces BadWindow even when our app is healthy.
        # Retry this read only, within the original bound, without restarting
        # the app, its episode, inference or any physical step.
        try:
            tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True, timeout=2)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            tree_retries += 1
            time.sleep(.05)
            continue
        for wid in re.findall(r'(0x[0-9a-f]+) "G1 科学站任务验证"', tree):
            prop = subprocess.run(["xprop", "-id", wid, "_NET_WM_PID"], capture_output=True, text=True, timeout=2).stdout
            if re.search(r'=\s*' + str(child.pid) + r'\s*$', prop):
                window = wid
                break
        if time.monotonic() > deadline:
            raise TimeoutError("Owned app window mapping20s")
        time.sleep(.05)
    info = subprocess.check_output(["xwininfo", "-id", window], text=True, timeout=2)
    if ("IsViewable" not in info or int(re.search(r"Width:\s+(\d+)", info).group(1)) != 1920
            or int(re.search(r"Height:\s+(\d+)", info).group(1)) != 1080):
        raise ValueError("Owned app window changed baseline resolution")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", "x11grab", "-window_id", str(int(window, 16)),
               "-draw_mouse", "0", "-framerate", "25", "-video_size", "1920x1080", "-i", os.environ["DISPLAY"],
               "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-threads", "2",
               "-pix_fmt", "yuv420p", "-t", "150", str(directory / "mobile_live_full.mp4")]
    with (directory / "ffmpeg.log").open("x") as log:
        recorder = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return recorder, {"command": command, "window_id": window, "window_pid_matches_owned_app": True,
                      "owned_recorder_pid": recorder.pid, "read_only_window_tree_retries": tree_retries}


def run(args):
    if digest(args.manifest) != args.manifest_sha256:
        raise ValueError("Frozen manifest hash changed before physical execution")
    manifest = json.loads(args.manifest.read_text())
    validate(manifest)
    if digest(args.binary) != manifest["binary_sha256"]:
        raise ValueError("Native app binary changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.workspace, text=True).strip()
    if (head != manifest["suite_source_commit"] or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=args.workspace, text=True)):
        raise ValueError("Suite source is dirty or does not match frozen commit")
    for path, checksum in manifest["external_files_sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"External source/profile/asset changed: {path}")
    model = manifest["model"]
    if digest(model["receipt"]) != manifest["model_receipt_sha256"]:
        raise ValueError("Published N1.6 weight receipt changed")
    qwen = json.loads(subprocess.check_output(["docker", "inspect", manifest["qwen"]["container"]], text=True))[0]
    if (not qwen["State"]["Running"] or qwen["Image"] != manifest["qwen"]["image_id"]
            or qwen["Config"]["Labels"].get("local.qwen.owner") != "local-qwen-g1-validation"):
        raise ValueError("Owned Qwen service identity does not match frozen local profile")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health():
        with opener.open("http://127.0.0.1:5558/health", timeout=2) as response:
            return json.load(response)

    def metrics():
        with opener.open("http://127.0.0.1:8002/metrics", timeout=3) as response:
            return response.read().decode()

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.manifest, args.output / "frozen_manifest.json")
    save(args.output / "qwen_identity.json", qwen)
    (args.output / "qwen_metrics_before.prom").write_text(metrics())
    summary = {"schema": "g1_native_station_mobile_profile_suite_result_v1", "complete": False,
               "manifest_sha256": args.manifest_sha256, "suite_source_commit": head,
               "app_source_commit": manifest["app_source_commit"], "binary_sha256": manifest["binary_sha256"],
               "cases": [], "successes": 0, "expected_cases": 10, "required_successes": 8,
               "actual_integrations": 0, "fresh_vla_calls": 0, "qwen_http_results": 0,
               "pauses_for_camera_and_policy": True, "continuous_real_time_qualified": False,
               "full_task_qualified": False, "frozen_profile_8of10_passed": False}
    server = child = recorder = None
    current = execution = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned suite interrupted by signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for seed in manifest["seeds"]:
            group = args.output / f"seed_{seed}"
            group.mkdir()
            captures = group / "policy_captures"
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", 5558))
            command = [model["python"], "-u", str(SCRIPTS / "unitree_g1_mobile_server.py"),
                       "--gr00t-source", model["source"], "--model-root", model["root"],
                       "--receipt", model["receipt"], "--runtime-env", model["runtime"],
                       "--port", "5558", "--seed", str(seed), "--capture-dir", str(captures),
                       "--episode-seed-reset"]
            lifecycle = {"seed": seed, "command": command, "owned_model_closed": False}
            with (group / "model.log").open("x") as log:
                try:
                    print(f"EVENT loading single original N1.6 owner seed={seed}; no inference warmup", flush=True)
                    server = subprocess.Popen(command, cwd=args.workspace, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    lifecycle["owned_model_pid"] = server.pid
                    deadline = time.monotonic() + 90
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError("Owned N1.6 exited before readiness")
                        try:
                            initial = health()
                            break
                        except (OSError, urllib.error.URLError):
                            if time.monotonic() > deadline:
                                raise TimeoutError("N1.6 owner readiness90s")
                            time.sleep(.2)
                    if (initial["revision"] != REVISION or initial["initial_seed"] != seed
                            or initial["successful_inferences"] != 0 or initial["failed_inferences"] != 0
                            or not initial["episode_seed_reset_enabled"] or initial["episode_seed_resets"] != 0):
                        raise ValueError("Model revision/seed/counter gate mismatch")
                    lifecycle["initial_health"] = initial
                    save(group / "model_lifecycle.json", lifecycle)
                    for spec in [c for c in manifest["cases"] if c["seed"] == seed]:
                        current = args.output / f"position_{spec['position_id']}_seed_{seed}"
                        current.mkdir()
                        save(current / "config.json", spec["configuration"])
                        episode = spec["configuration"]["runner"]["body"]["mobile_homie_v2"]["episode_id"]
                        before = health()
                        body = json.dumps({"schema": "mobile_episode_seed_reset_v1", "episode_id": episode, "initial_seed": seed}).encode()
                        request = urllib.request.Request("http://127.0.0.1:5558/begin_episode", data=body, headers={"Content-Type": "application/json"})
                        with opener.open(request, timeout=3) as response:
                            reset = json.load(response)
                        if (reset["successful_inferences"] != before["successful_inferences"]
                                or reset["post_load_rng_sha256"] != initial["post_load_rng_sha256"]):
                            raise ValueError("Episode reset changed model counter or initial RNG identity")
                        app = [str(args.binary), "--scene", "g1_station_mobile_release_diagnostic", "--robot", "g1",
                               "--g1-config", str(current / "config.json"), "--g1-ticks", "3300", "--output", str(current / "native")]
                        execution = {"position_id": spec["position_id"], "seed": seed, "episode_id": episode,
                                     "manifest_sha256": args.manifest_sha256, "binary_sha256": manifest["binary_sha256"],
                                     "initial_health": before, "seed_reset": reset, "command": app,
                                     "app_reaped": False, "retries": 0, "warmup_inferences": 0}
                        save(current / "execution.json", execution)
                        (current / "qwen_metrics_before.prom").write_text(metrics())
                        print(f"EVENT frozen T2 position={spec['position_id']} seed={seed} episode={episode} start; no retries", flush=True)
                        with (current / "app.log").open("x") as app_log:
                            started = time.monotonic()
                            child = subprocess.Popen(app, cwd=args.workspace, stdout=app_log, stderr=subprocess.STDOUT, start_new_session=True)
                            execution["owned_app_pid"] = child.pid
                            try:
                                recorder, video = record_owned_window(child, current, manifest["ffmpeg"])
                                execution["recording"] = video
                                child.wait(timeout=max(.1, RULES["outer_timeout_seconds"] - (time.monotonic() - started)))
                            except subprocess.TimeoutExpired:
                                execution["outer_timeout"] = True
                            finally:
                                execution["recorder_exit_code"] = close_owned(recorder, recording=True)
                                execution["recorder_reaped"] = True
                                recorder = None
                                execution["app_exit_code"] = close_owned(child)
                                execution["app_reaped"] = True
                                child = None
                                execution["wall_seconds"] = time.monotonic() - started
                        after = health()
                        execution["final_health"] = after
                        calls = after["successful_inferences"] - before["successful_inferences"]
                        (current / "qwen_metrics_after.prom").write_text(metrics())
                        local = list(captures.glob(f"e{episode}_f*_q*"))
                        for path in local:
                            shutil.copytree(path, current / "policy_captures" / path.name)
                        try:
                            result = audit(current, seed)
                            result["model_counter_verified"] = (calls == result["fresh_vla_calls"] == len(local)
                                and after["failed_inferences"] == before["failed_inferences"]
                                and after["current_episode_id"] == episode)
                            result["strict_task_passed"] &= result["model_counter_verified"] and execution["app_exit_code"] == 0
                        except Exception as error:
                            result = {"evidence_verified": False, "strict_task_passed": False,
                                      "audit_error": repr(error), "actual_integrations": 0, "fresh_vla_calls": calls}
                            receipt_path = current / "native/capture_receipt.json"
                            if receipt_path.exists():
                                partial = json.loads(receipt_path.read_text())
                                result["actual_integrations"] = partial["actual_integrations"]
                                result["failure_reason"] = partial["failure_reason"]
                                result["qwen_http_results"] = partial.get("mobile_assist_handoff", {}).get("qwen_postgrasp_transport", {}).get("http_results", 0)
                        result.update(position_id=spec["position_id"], seed=seed, episode_id=episode)
                        video_path = current / "mobile_live_full.mp4"
                        if video_path.exists():
                            execution["video_sha256"] = digest(video_path)
                        execution["result"] = result
                        save(current / "execution.json", execution)
                        save(current / "independent_route_audit.json", result)
                        summary["cases"].append(result)
                        summary["actual_integrations"] += result["actual_integrations"]
                        summary["fresh_vla_calls"] += calls
                        summary["qwen_http_results"] += result.get("qwen_http_results", 0)
                        summary["successes"] += int(result["strict_task_passed"])
                        save(args.output / "summary.json", summary)
                        print(f"EVENT case ended strict_pass={result['strict_task_passed']} ticks={result['actual_integrations']} N16={calls} score={summary['successes']}/{len(summary['cases'])}", flush=True)
                    lifecycle["final_health"] = health()
                finally:
                    if server is not None and server.poll() is None:
                        try:
                            lifecycle["final_health"] = health()
                        except Exception as error:
                            lifecycle["final_health_error"] = repr(error)
                    lifecycle["exit_code"] = close_owned(server)
                    lifecycle["owned_model_closed"] = True
                    server = None
                    save(group / "model_lifecycle.json", lifecycle)
        summary["complete"] = len(summary["cases"]) == 10
        summary["frozen_profile_8of10_passed"] = (summary["complete"] and summary["successes"] >= 8
            and all(c["evidence_verified"] for c in summary["cases"]))
    finally:
        close_owned(recorder, recording=True)
        close_owned(child)
        close_owned(server)
        if current is not None and execution is not None:
            execution["app_reaped"] = True
            save(current / "execution.json", execution)
        summary["owned_child_processes_reaped"] = True
        try:
            (args.output / "qwen_metrics_after.prom").write_text(metrics())
        finally:
            save(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "binary", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
