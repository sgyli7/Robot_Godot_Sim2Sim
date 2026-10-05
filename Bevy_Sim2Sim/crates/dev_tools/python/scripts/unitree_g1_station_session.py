#!/usr/bin/env python3
"""Own one local model and one finite Chinese fixed-task development window.

No weights are installed, no service configuration is changed, and no task
qualification is granted. Closing the window or Ctrl+C reaps owned processes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time
import urllib.error
import urllib.request

WORKSPACE = Path(__file__).resolve().parents[4]
REVISIONS = {"static": "7f78bebf1a90131e7304beacfcd47eb27bad16ab",
             "mobile": "dfe74af855007f26093f362cd2d7a2f404b64b93"}
DEFAULT_CACHE = Path("/home/ethan/models/unitree_g1")
QWEN_SERVICE = Path("/home/ethan/LocalServices/Local_Qwen/qwen38-27b-fp8-dgx-spark/g1-service.sh")
SESSION_LOCK = Path("/tmp/sai-g1-station-session.lock")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def argument_path(name, path):
    # CPython locates pyvenv.cfg from the invoked executable path. Resolving
    # bin/python's symlink would silently leave the isolated model environment.
    return path.absolute() if name == "policy_python" else path.resolve()


def prepare_configuration(value, profile, port, maximum_episodes, smoke, inflight_reset=False,
                          with_local_qwen=False):
    """Build a new configuration before Rust loads it; never edit a live asset."""
    value = json.loads(json.dumps(value))
    body = "static_agile" if profile == "static" else "mobile_homie_v2"
    if set(value.get("runner", {}).get("body", {})) != {body}:
        raise ValueError("The chosen task requires its distinct original body configuration")
    if (not 2 <= maximum_episodes <= 10 or value.get("task_lab") is not None
            or value.get("local_model_startup") is not None):
        raise ValueError("Use 2..=10 episodes and an externally prepared fixed task configuration")
    if inflight_reset and (profile != "mobile" or not smoke):
        raise ValueError("In-flight Qwen reset is a separate mobile smoke test")
    policy = value.get("policy", {})
    if policy.get("max_calls") != (1 if profile == "static" else 4):
        raise ValueError("Task action horizons/call budgets must retain their original profile")
    if profile == "mobile":
        dispatch = value.get("diagnostic_qwen_dispatch", {})
        if (dispatch.get("scope") != "scientific_station_mobile_from_instruction_v1"
                or dispatch.get("connection", {}).get("endpoint") != "http://127.0.0.1:8002/v1"):
            raise ValueError("Mobile UI requires its existing local Qwen fixed-profile admission")
    elif not (value.get("static_startup") and value.get("static_grasp_fixed_camera_pair")
              and value.get("static_pair_program")):
        raise ValueError("Static UI requires the disclosed observed-grasp/fixed-pair place route")
    if profile == "static":
        if value.get("diagnostic_qwen_dispatch") is not None:
            raise ValueError("Static Qwen configuration is prepared only by --with-local-qwen")
        if with_local_qwen:
            value["diagnostic_qwen_dispatch"] = {
                "scope": "scientific_station_static_from_instruction_v1",
                "connection": {"endpoint": "http://127.0.0.1:8002/v1",
                               "model": "qwen3.8-27b-fp8", "timeout_ms": 20000,
                               "max_output_tokens": 128},
                "instruction": "把苹果放到盘子里"}
    value["policy"]["endpoint"] = f"http://127.0.0.1:{port}/infer"
    value["station_task_controls"] = {"maximum_episodes": maximum_episodes,
                                      "smoke_stop_reset": smoke,
                                      "smoke_initial_inflight_reset": inflight_reset}
    return value


def close_owned(process, recording=False):
    if process is None:
        return None
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT if recording else signal.SIGTERM)
        except ProcessLookupError:
            pass  # The owned child may exit between poll and signal.
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return process.wait()


def check_unused_port(port):
    # Our just-closed server can leave TIME_WAIT sockets. SO_REUSEADDR permits
    # that restart while still refusing a live listener; SO_REUSEPORT is absent.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def acquire_session_lock(path=SESSION_LOCK):
    """One resource owner across profiles/worktrees; reject peers immediately.

    Keep the lock inode after closing: unlinking a held lock would allow two
    processes to acquire different files with the same path.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise ValueError("G1 session lock is not an owned regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another G1 station session owns the models; close it first") from error
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"owner_pid": os.getpid(), "scope": "g1_station_session"}).encode())
        return fd
    except BaseException:
        os.close(fd)
        raise


def inspect_qwen():
    value = json.loads(subprocess.check_output(
        ["docker", "inspect", "qwen38-27b-fp8-g1"], text=True))[0]
    if value["Config"]["Labels"].get("local.qwen.owner") != "local-qwen-g1-validation":
        raise ValueError("Existing Qwen container belongs to another owner")
    return value


def cleanup_session(receipt, model, app, recorder, qwen_start, started_qwen, service):
    """Attempt every owned cleanup even if Docker is unavailable or stop fails."""
    errors = []
    for label, process, recording in [
            ("recorder_exit_code", recorder, True),
            ("application_cleanup_exit_code", app, False),
            ("qwen_start_cleanup_exit_code", qwen_start, False)]:
        try:
            receipt[label] = close_owned(process, recording)
        except Exception as error:
            errors.append(f"{label}: {error!r}")
    if started_qwen:
        try:
            current = inspect_qwen()  # Refuses a changed/foreign owner.
            if current["State"]["Running"]:
                initial = receipt.get("qwen_identity")
                if initial is not None and (current["Id"] != initial["Id"]
                        or current["State"]["StartedAt"] != initial["State"]["StartedAt"]):
                    raise ValueError("Qwen container identity/start changed; refusing to stop another session")
                subprocess.run([str(service), "stop"], check=True, timeout=45)
            stopped = inspect_qwen()
            receipt["qwen_stopped_identity"] = stopped
            if stopped["State"]["Running"]:
                raise RuntimeError("Owned Qwen container remains running")
        except Exception as error:
            errors.append(f"owned_qwen_stop: {error!r}")
    try:
        receipt["model_exit_code"] = close_owned(model)
    except Exception as error:
        errors.append(f"model_exit_code: {error!r}")
    receipt["cleanup_errors"] = errors
    receipt["all_owned_handles_reaped"] = not errors
    return errors


def run(args):
    config = prepare_configuration(json.loads(args.config.read_text()), args.profile,
                                   args.policy_port, args.maximum_episodes, args.smoke_stop_reset,
                                   args.smoke_reset_during_qwen, args.with_local_qwen)
    use_qwen = args.profile == "mobile" or args.with_local_qwen
    episode = args.episode_id if args.episode_id is not None else time.time_ns() // 1_000_000
    if not 0 < episode < (1 << 64) - args.maximum_episodes:
        raise ValueError("A positive episode identity with room for all resets is required")
    body = "static_agile" if args.profile == "static" else "mobile_homie_v2"
    config["runner"]["body"][body]["episode_id"] = episode
    if not args.binary.is_file() or not 1 <= args.session_timeout <= 7200:
        raise ValueError("An existing compiled app and a 1..=7200 second session budget are required")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "config.json", config)
    receipt = {"schema": "g1_owned_fixed_task_window_session_v1", "profile": args.profile,
               "source_config_sha256": digest(args.config), "configuration_sha256": digest(args.output / "config.json"),
               "binary_sha256": digest(args.binary), "maximum_episodes": args.maximum_episodes,
               "session_timeout_seconds": args.session_timeout, "task_qualified": False,
               "initial_episode_id": episode,
               "prepared_only": args.prepare_only, "all_owned_handles_reaped": False,
               "N16_episode_rng_reset_enabled": False if args.profile == "mobile" else None}
    receipt["smoke_reset_during_actual_qwen"] = args.smoke_reset_during_qwen
    receipt["local_qwen_initial_profile_gate"] = use_qwen
    save(args.output / "session_receipt.json", receipt)
    if args.prepare_only:
        receipt["all_owned_handles_reaped"] = True
        save(args.output / "session_receipt.json", receipt)
        print("EVENT prepared immutable session configuration; no processes or inference", flush=True)
        return 0
    model = app = recorder = qwen_start = None
    session_lock = None
    started_qwen = False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health():
        with opener.open(f"http://127.0.0.1:{args.policy_port}/health", timeout=2) as response:
            return json.load(response)

    try:
        session_lock = acquire_session_lock()
        receipt["resource_owner_lock"] = str(SESSION_LOCK)
        check_unused_port(args.policy_port)
        if use_qwen and inspect_qwen()["State"]["Running"]:
            raise ValueError("The owned G1 Qwen service is already managed by another active session")
        script = WORKSPACE / "crates/dev_tools/python/scripts" / (
            "unitree_g1_static_server.py" if args.profile == "static" else "unitree_g1_mobile_server.py")
        command = [str(args.policy_python), "-u", str(script), "--receipt", str(args.model_receipt),
                   "--port", str(args.policy_port), "--capture-dir", str(args.output / "policy_captures")]
        if args.profile == "static":
            command += ["--device", "cuda", "--seed-offset", str(args.seed)]
        else:
            # UI worlds require fresh observation/action identities, not a frozen
            # benchmark RNG restart. A single owner remains resident across reset.
            command += ["--gr00t-source", str(args.gr00t_source), "--model-root", str(args.model_root),
                        "--runtime-env", str(args.runtime_env), "--seed", str(args.seed)]
        receipt["model_command"] = command
        with (args.output / "model.log").open("x") as log:
            model = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt["owned_model_pid"] = model.pid
        save(args.output / "session_receipt.json", receipt)
        deadline = time.monotonic() + 90
        print("EVENT loading one original local task model before own Qwen", flush=True)
        while True:
            if model.poll() is not None:
                raise RuntimeError("Owned task model exited before readiness; see model.log")
            try:
                actual = health()
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Original task model startup exceeded 90 seconds")
                time.sleep(.2)
        if actual.get("revision") != REVISIONS[args.profile] or actual["successful_inferences"] != 0:
            raise ValueError("Wrong model revision or reused inference owner")
        receipt["initial_model_health"] = actual
        if use_qwen:
            started_qwen = True
            with (args.output / "qwen_start.log").open("x") as log:
                qwen_start = subprocess.Popen([str(args.qwen_service), "start"], stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            if qwen_start.wait(timeout=360) != 0:
                raise RuntimeError("Owned local Qwen startup failed")
            qwen_start = None
            receipt["qwen_identity"] = inspect_qwen()
        command = [str(args.binary), "--scene", "g1_static_observed_place_diagnostic" if args.profile == "static"
                   else "g1_station_mobile_release_diagnostic", "--robot", "g1", "--g1-config",
                   str(args.output / "config.json"), "--g1-ticks", "1100" if args.profile == "static" else "3300",
                   "--output", str(args.output / "native")]
        receipt["application_command"] = command
        with (args.output / "app.log").open("x") as log:
            app = subprocess.Popen(command, cwd=WORKSPACE, env=os.environ.copy(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        receipt["owned_app_pid"] = app.pid
        if args.record:
            from unitree_g1_mobile_suite import record_owned_window
            recorder, receipt["recording"] = record_owned_window(app, args.output, str(args.ffmpeg))
            receipt["recording_maximum_seconds"] = 150
        save(args.output / "session_receipt.json", receipt)
        print("EVENT Chinese task window ready; Start, Stop, Reset; close window or Ctrl+C to finish", flush=True)
        receipt["application_exit_code"] = app.wait(timeout=args.session_timeout)
        receipt["final_model_health"] = health()
        return receipt["application_exit_code"]
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        errors = cleanup_session(receipt, model, app, recorder, qwen_start,
                                 started_qwen, args.qwen_service)
        if session_lock is not None:
            os.close(session_lock)
            receipt["resource_owner_lock_released"] = True
        save(args.output / "session_receipt.json", receipt)
        print("EVENT owned session cleanup verified" if not errors else
              "EVENT owned session cleanup failed; see session_receipt.json", flush=True)
        if errors:
            raise RuntimeError("Owned session cleanup failed: " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["static", "mobile"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-episodes", type=int, default=4)
    parser.add_argument("--smoke-stop-reset", action="store_true")
    parser.add_argument("--smoke-reset-during-qwen", action="store_true",
                        help="Reset at zero Tick while the real initial Qwen request is in flight")
    parser.add_argument("--with-local-qwen", action="store_true",
                        help="Admit the static original task from real initial RGB through local Qwen")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/home/ethan/.local/bin/ffmpeg"))
    parser.add_argument("--session-timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=int,
                        help="Fresh initial identity; defaults to current Unix milliseconds")
    parser.add_argument("--policy-port", type=int)
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--gr00t-source", type=Path, default=Path("/home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaac_gr00t_n16"))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_CACHE / "mobile_box" / REVISIONS["mobile"])
    parser.add_argument("--runtime-env", type=Path, default=DEFAULT_CACHE / "envs/policy_gr00t_n16_cu13")
    parser.add_argument("--qwen-service", type=Path, default=QWEN_SERVICE)
    args = parser.parse_args()
    args.smoke_stop_reset |= args.smoke_reset_during_qwen
    args.policy_port = args.policy_port or (5557 if args.profile == "static" else 5558)
    args.policy_python = args.policy_python or DEFAULT_CACHE / "envs" / (
        "policy_onnx_cu13" if args.profile == "static" else "policy_gr00t_n16_cu13") / "bin/python"
    args.model_receipt = args.model_receipt or Path("/home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy") / (
        "static_apple_files.json" if args.profile == "static" else "mobile_box_files.json")
    for name in ["config", "binary", "output", "policy_python", "model_receipt", "gr00t_source", "model_root", "runtime_env", "qwen_service", "ffmpeg"]:
        setattr(args, name, argument_path(name, getattr(args, name)))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
