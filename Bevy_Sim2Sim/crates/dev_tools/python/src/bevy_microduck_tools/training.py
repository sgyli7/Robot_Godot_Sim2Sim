"""A measured, isolated training/resume entry using the actual upstream runner."""

from __future__ import annotations

import json
import hashlib
import re
import multiprocessing
import os
import time
from pathlib import Path

from .serialization import sha256_file, write_json
from .workflow import Rejection, TrainingBudget, admit
from .authorization import LearningRequest
from .source_runtime_identity import verify_candidate_runtime


MAX_SMOKE_SECONDS = 180
MAX_MESSAGE_BYTES = 4096


def _atomic_worker_result(path: Path, result: dict) -> str:
    """Publish complete result bytes before sending their small IPC locator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return sha256_file(path)


def _send_result_locator(connection, path: Path, result: dict) -> None:
    digest = _atomic_worker_result(path, result)
    message = json.dumps({"status": result["status"], "result_path": str(path.resolve()),
                          "result_sha256": digest}).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise ValueError("Worker result locator exceeds the bounded IPC protocol")
    connection.send_bytes(message)


def _worker(connection, result_path, source_root, output, skill, substeps, seed, device, iterations, checkpoint, binding):
    path = Path(result_path)
    try:
        from .cli import _audit_authorized_learning
        result = _audit_authorized_learning(Path(source_root), Path(output), skill, binding=binding,
                       substeps=substeps, seed=seed, device=device, iterations=iterations, checkpoint=Path(checkpoint))
        _send_result_locator(connection, path, {"status": "completed_structural_smoke", "result": result})
    except BaseException as error:
        # Preserve an incomplete serialization file as diagnostics, never success.
        if not path.exists() and not path.with_suffix(path.suffix + ".partial").exists():
            _send_result_locator(connection, path, {"status": "failed", "error_type": type(error).__name__,
                                                    "error": str(error)})
        raise
    finally:
        connection.close()


def _restore_worker(connection, result_path, source_root, output, origin_path, seed, device):
    path = Path(result_path)
    try:
        from .cli import audit
        origin = json.loads(Path(origin_path).read_text())
        result = audit(Path(source_root), Path(output), origin["skill"], steps=None,
                       substeps=origin["source_substeps"], seed=seed, device=device, video=True,
                       learning_iterations=0, resume=Path(origin["checkpoint"]["path"]),
                       resume_origin=Path(origin_path))
        _send_result_locator(connection, path, {"status": "completed_zero_update_restore", "result": result})
    except BaseException as error:
        if not path.exists() and not path.with_suffix(path.suffix + ".partial").exists():
            _send_result_locator(connection, path, {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        connection.close()


def _run_child(target, arguments: tuple, result_path: Path, wall_seconds: float) -> dict:
    """Read bounded IPC while waiting, then validate the atomic file and exit."""
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sending, str(result_path.resolve()), *arguments))
    deadline = time.monotonic() + wall_seconds
    message = None
    message_bytes = 0
    pipe_open = True
    failure = None
    try:
        process.start()
        sending.close()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = {"status": "budget_exhausted", "reason": "Measured per-run hard deadline expired; child killed"}
                break
            if pipe_open and message is None and receiving.poll(min(.05, remaining)):
                try:
                    raw = receiving.recv_bytes(MAX_MESSAGE_BYTES)
                    message_bytes = len(raw)
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError("Worker locator is not an object")
                except EOFError:
                    pipe_open = False
                except (OSError, ValueError) as error:
                    failure = {"status": "failed", "reason": f"Invalid bounded worker IPC: {error}"}
                    break
            process.join(timeout=min(.05, max(0, deadline - time.monotonic())))
            if not process.is_alive():
                if message is None and pipe_open and receiving.poll(0):
                    try:
                        raw = receiving.recv_bytes(MAX_MESSAGE_BYTES)
                        message_bytes = len(raw)
                        message = json.loads(raw)
                    except (EOFError, OSError, ValueError) as error:
                        failure = {"status": "failed", "reason": f"Child exited without valid worker IPC: {error}"}
                break
        if failure is not None:
            return failure
        if not isinstance(message, dict):
            return {"status": "failed", "reason": f"Child exited {process.exitcode} without a result locator"}
        if set(message) != {"status", "result_path", "result_sha256"}:
            return {"status": "failed", "reason": "Worker result locator fields do not match protocol"}
        if message["result_path"] != str(result_path.resolve()):
            return {"status": "failed", "reason": "Worker returned an unexpected result path"}
        if not result_path.is_file():
            return {"status": "failed", "reason": "Child atomic result file missing (partial file is not accepted)"}
        if sha256_file(result_path) != message["result_sha256"]:
            return {"status": "failed", "reason": "Child atomic result file hash differs from locator"}
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError) as error:
            return {"status": "failed", "reason": f"Invalid complete child result JSON: {error}"}
        if not isinstance(result, dict) or result.get("status") != message["status"]:
            return {"status": "failed", "reason": "Child result status differs from locator"}
        if result["status"] not in {"completed_structural_smoke", "completed_zero_update_restore", "failed"}:
            return {"status": "failed", "reason": "Child result has an unknown status"}
        if result["status"] != "failed" and process.exitcode != 0:
            return {"status": "failed", "reason": f"Child claimed success but exited {process.exitcode}"}
        result.update(atomic_worker_result={"path": str(result_path.resolve()), "sha256": message["result_sha256"]},
                      ipc_message_bytes=message_bytes, child_exitcode=process.exitcode)
        return result
    except Exception as error:
        return {"status": "failed", "reason": f"Child launch/wait failed: {type(error).__name__}: {error}"}
    finally:
        if process.pid is not None and process.is_alive():
            process.kill()
            process.join(timeout=1)
        receiving.close()
        sending.close()


def train_bounded(candidate_path: Path, evidence_path: Path, review_path: Path, *,
                  source_root: Path, output: Path, ledger: Path, seed: int,
                  iterations: int = 1, wall_seconds: float = 180, device: str = "cuda:0", gpus: int = 1) -> dict:
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    candidate_file_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    evidence = json.loads(evidence_path.read_text())
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    request = LearningRequest(iterations, wall_seconds, seed, gpus, str(ledger.resolve()))
    admit(candidate, "learning", evidence, review, request=request)
    verify_candidate_runtime(candidate, source_root)
    if gpus != 1 or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise Rejection("Current actual runner supports exactly one explicitly indexed CUDA GPU")
    if not 1 <= iterations <= 2:
        raise Rejection("First training batch permits only one or two actual upstream PPO updates")
    if not 0 < wall_seconds <= MAX_SMOKE_SECONDS:
        raise Rejection("Structural training smoke requires a hard deadline of at most 180 seconds")
    result_path = output.parent / f"{output.name}_worker_result.json"
    if (output.exists() and any(output.iterdir())) or result_path.exists() or result_path.with_suffix(result_path.suffix + ".partial").exists():
        raise Rejection("Use unique output and worker-result paths for each training child")
    adoption = json.loads(Path(candidate["artifacts"]["adoption"]["path"]).read_text())
    skill = adoption["skill"]
    budget = TrainingBudget(ledger)
    run_id = budget.reserve_authorized(skill, candidate["candidate_id"], review, review_sha256, request)
    binding = {"ledger_path": str(ledger.resolve()), "run_id": run_id, "candidate_id": candidate["candidate_id"],
               "review_sha256": review_sha256, "request": request.record(),
               "candidate_path": str(candidate_path.resolve()), "candidate_file_sha256": candidate_file_sha256}
    started = time.monotonic()
    status = "failed"
    try:
        result = _run_child(_worker, (str(source_root), str(output), skill,
                            candidate["contract"]["source_substeps"], seed, device,
                            iterations, candidate["artifacts"]["checkpoint"]["path"], binding), result_path, wall_seconds)
        status = result["status"]
        elapsed = time.monotonic() - started
        result.update(parent_candidate_id=candidate["candidate_id"], run_id=run_id,
                      measured_gpu_seconds=elapsed * gpus, training_seed=seed,
                      requested_ppo_iterations=iterations, authorization_id=review["authorization_id"],
                      review_sha256=review_sha256, learning_request=request.record(),
                      skill_qualified=False, purpose="structural short training/resume/export smoke")
        write_json(output.parent / f"{output.name}_training_result.json", result)
        return result
    except BaseException:
        status = "failed"
        raise
    finally:
        # Account actual launch, wait, teardown and output, including failures.
        budget.finish(run_id, time.monotonic() - started, status)


def restore_bounded(origin_path: Path, *, source_root: Path, output: Path, ledger: Path,
                    seed: int, wall_seconds: float = 180, device: str = "cuda:0") -> dict:
    """Run no learner at all; retain previous measured update provenance."""
    from .provenance import restored_origin
    input_origin = json.loads(origin_path.read_text())
    origin = restored_origin(origin_path, Path(input_origin["checkpoint"]["path"]))
    if not 0 < wall_seconds <= MAX_SMOKE_SECONDS:
        raise Rejection("Restored natural inference requires a hard deadline of at most 180 seconds")
    result_path = output.parent / f"{output.name}_worker_result.json"
    if (output.exists() and any(output.iterdir())) or result_path.exists() or result_path.with_suffix(result_path.suffix + ".partial").exists():
        raise Rejection("Use unique paths for each zero-update restore")
    budget = TrainingBudget(ledger)
    run_id = budget.reserve(origin["skill"], origin["parent_candidate_id"], seed, wall_seconds)
    started = time.monotonic()
    status = "failed"
    try:
        result = _run_child(_restore_worker, (str(source_root), str(output), str(origin_path.resolve()), seed, device), result_path, wall_seconds)
        status = result["status"]
        result.update(parent_candidate_id=origin["parent_candidate_id"], originating_learning_run_id=origin["originating_run_id"],
                      origin_input_sha256=sha256_file(origin_path), run_id=run_id,
                      measured_gpu_seconds=time.monotonic()-started, training_seed=seed,
                      ppo_iterations_this_invocation=0, known_prior_ppo_iterations=origin["known_prior_ppo_iterations"],
                      skill_qualified=False, purpose="fresh-environment zero-update restored checkpoint natural inference validation")
        write_json(output.parent / f"{output.name}_restore_result.json", result)
        return result
    finally:
        budget.finish(run_id, time.monotonic()-started, status)
