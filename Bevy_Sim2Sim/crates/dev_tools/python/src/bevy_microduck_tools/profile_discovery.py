"""Historical zero-update source profile accounting and CPU diagnostics.

The public discovery entry and the old private source worker are disabled.
CPU fixtures can exercise authorization and observation without creating a
source environment. A discovery receipt never grants learning or native
completeness.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .authorization import Rejection
from .profile_identity import MODE, PROFILE, VerifiedDeclaredInputs, private_cache_environment, require_zero_token, validate_inputs, verify_declared_inputs
from .serialization import identity, sha256_file, write_json
from .workflow import TrainingBudget

REVIEW_SCHEMA = "microduck_root_discovery_review_v2"
REQUEST_SCHEMA = "microduck_zero_update_discovery_request_v2"


class DiscoveryDeadlineExceeded(Rejection):
    pass


@contextmanager
def _cpu_hard_deadline(seconds: float):
    """Interrupt a stuck CPU preflight before it can reserve a GPU."""
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def timed_out(_signal, _frame):
        raise DiscoveryDeadlineExceeded(f"CPU phase exceeded bounded {seconds}-second deadline")
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class DiscoveryRequest:
    input_id: str
    nonce: str
    mode: str
    seed: int
    wall_seconds: float
    gpus: int
    budget_ledger_path: str

    def __post_init__(self):
        if (not isinstance(self.input_id, str) or len(self.input_id) != 64 or
                not isinstance(self.nonce, str) or self.mode != MODE or
                type(self.seed) is not int or self.seed != PROFILE["seed"] or
                type(self.wall_seconds) not in (int, float) or not math.isfinite(self.wall_seconds) or
                not 0 < self.wall_seconds <= 180 or type(self.gpus) is not int or self.gpus != 1 or
                not Path(self.budget_ledger_path).is_absolute()):
            raise Rejection("Discovery request must be exact zero-update, one-GPU, bounded input")
        try:
            if str(uuid.UUID(self.nonce)) != self.nonce:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise Rejection("Discovery nonce is not a canonical UUID") from None
        object.__setattr__(self, "budget_ledger_path", str(Path(self.budget_ledger_path).resolve()))

    def record(self) -> dict:
        return {"schema": REQUEST_SCHEMA, "input_id": self.input_id, "nonce": self.nonce,
                "mode": self.mode, "seed": self.seed, "wall_seconds": self.wall_seconds,
                "gpus": self.gpus, "budget_ledger_path": self.budget_ledger_path}


def authorize_discovery(review: dict, request: DiscoveryRequest, *, now: float | None = None) -> None:
    fields = {"schema", "authorization_id", "reviewer_role", "decision", "reviewed_at", "expires_at",
              "input_id", "nonce", "authorized_mode", "learning_allowed", "max_runs",
              "max_wall_seconds", "seed", "gpus", "budget_ledger_path"}
    if not isinstance(review, dict) or set(review) != fields or review["schema"] != REVIEW_SCHEMA:
        raise Rejection("Exact root zero-update discovery review required")
    try:
        if str(uuid.UUID(review["authorization_id"])) != review["authorization_id"]:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise Rejection("Discovery authorization ID must be a canonical UUID") from None
    instant = time.time() if now is None else now
    if (review["reviewer_role"] != "root_gpt" or review["decision"] != "approve" or
            type(review["reviewed_at"]) not in (int, float) or type(review["expires_at"]) not in (int, float) or
            not review["reviewed_at"] <= instant <= review["expires_at"] or
            review["input_id"] != request.input_id or review["nonce"] != request.nonce or
            review["authorized_mode"] != MODE or review["learning_allowed"] is not False or
            type(review["max_runs"]) is not int or review["max_runs"] != 1 or
            type(review["max_wall_seconds"]) not in (int, float) or
            not 0 < request.wall_seconds <= review["max_wall_seconds"] <= 180 or
            review["seed"] != request.seed or review["gpus"] != request.gpus or
            review["budget_ledger_path"] != request.budget_ledger_path):
        raise Rejection("Discovery review does not authorize this exact bounded zero-update request")


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Rejection(f"Invalid {label} digest")


def _binding(binding: dict) -> DiscoveryRequest:
    fields = {"ledger_path", "run_id", "input_id", "nonce", "mode", "manifest_path", "manifest_sha256",
              "review_sha256", "request"}
    if not isinstance(binding, dict) or set(binding) != fields or binding["mode"] != MODE:
        raise Rejection("Incomplete private zero-update discovery binding")
    for key in ("run_id", "input_id", "manifest_sha256", "review_sha256"):
        _digest(binding[key], key)
    if not Path(binding["ledger_path"]).is_absolute() or not Path(binding["manifest_path"]).is_absolute():
        raise Rejection("Private discovery binding requires absolute paths")
    record = binding["request"]
    if not isinstance(record, dict) or set(record) != {"schema", "input_id", "nonce", "mode", "seed",
                                                      "wall_seconds", "gpus", "budget_ledger_path"} or record["schema"] != REQUEST_SCHEMA:
        raise Rejection("Private discovery request record changed")
    request = DiscoveryRequest(**{key: value for key, value in record.items() if key != "schema"})
    if (request.record() != record or request.input_id != binding["input_id"] or
            request.nonce != binding["nonce"] or request.budget_ledger_path != binding["ledger_path"]):
        raise Rejection("Private discovery request differs from consumed run")
    return request


def reserve_discovery(manifest_path: Path, review_path: Path, ledger_path: Path, *,
                      parent_token: VerifiedDeclaredInputs, wall_seconds: float = 180) -> dict:
    """Caller performs full parent byte preflight before invoking this function."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_inputs(manifest)
    require_zero_token(parent_token, manifest, manifest_path=manifest_path)
    if parent_token.runtime_verified is not True or parent_token.verification_phase != "parent":
        raise Rejection("Discovery reservation requires actual full v9 installed-byte parent preflight")
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    request = DiscoveryRequest(manifest["input_id"], manifest["nonce"], MODE,
                               PROFILE["seed"], wall_seconds, 1, str(ledger_path.resolve()))
    authorize_discovery(review, request)
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    budget = TrainingBudget(ledger_path)
    with budget.locked() as ledger:
        authorize_discovery(review, request)
        if sha256_file(review_path) != review_sha or sha256_file(manifest_path) != manifest_sha:
            raise Rejection("Discovery review or profile input changed before atomic consumption")
        if any(row.get("discovery_authorization", {}).get("authorization_id") == review["authorization_id"] or
               row.get("authorization", {}).get("authorization_id") == review["authorization_id"]
               for row in ledger["runs"]):
            raise Rejection("Single-use discovery authorization already consumed")
        if any(row.get("discovery_authorization", {}).get("nonce") == request.nonce for row in ledger["runs"]):
            raise Rejection("Discovery nonce already consumed by another run")
        run_id = budget._append_reservation(ledger, PROFILE["skill"], request.input_id,
                                            request.seed, request.wall_seconds, request.gpus)
        row = next(item for item in ledger["runs"] if item["run_id"] == run_id)
        row["identity_kind"] = "source_profile_input_id"
        row["discovery_authorization"] = {"authorization_id": review["authorization_id"],
                  "review_sha256": review_sha, "input_id": request.input_id, "nonce": request.nonce,
                  "mode": MODE, "request": request.record(), "manifest_sha256": manifest_sha}
    return {"ledger_path": str(ledger_path.resolve()), "run_id": run_id, "input_id": request.input_id,
            "nonce": request.nonce, "mode": MODE, "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha, "review_sha256": review_sha, "request": request.record()}


def claim_worker(binding: dict) -> dict:
    """Single PID can claim consumed discovery run; not a learning worker claim."""
    request = _binding(binding)
    if sha256_file(Path(binding["manifest_path"])) != binding["manifest_sha256"]:
        raise Rejection("Private discovery manifest changed")
    manifest = json.loads(Path(binding["manifest_path"]).read_text())
    validate_inputs(manifest)
    if manifest["input_id"] != binding["input_id"] or manifest["nonce"] != binding["nonce"]:
        raise Rejection("Private discovery input identity mismatch")
    with TrainingBudget(Path(binding["ledger_path"])).locked() as ledger:
        rows = [row for row in ledger["runs"] if row["run_id"] == binding["run_id"]]
        if len(rows) != 1 or rows[0]["state"] != "reserved":
            raise Rejection("Private discovery has no single open consumed run")
        row = rows[0]
        auth = row.get("discovery_authorization", {})
        if (auth.get("review_sha256") != binding["review_sha256"] or auth.get("input_id") != binding["input_id"] or
                auth.get("nonce") != binding["nonce"] or auth.get("mode") != MODE or
                auth.get("request") != request.record() or auth.get("manifest_sha256") != binding["manifest_sha256"] or
                "discovery_claim" in row or time.time() > row["started_at"] + request.wall_seconds):
            raise Rejection("Private discovery consumption or single claim invalid")
        row["discovery_claim"] = {"pid": os.getpid(), "claimed_at": time.time(),
                                   "manifest_sha256": binding["manifest_sha256"], "nonce": binding["nonce"]}
    return manifest


def require_claimed_worker(binding: dict, token: VerifiedDeclaredInputs, manifest: dict) -> None:
    _binding(binding)
    require_zero_token(token, manifest, manifest_path=Path(binding["manifest_path"]))
    if token.runtime_verified is not True or token.verification_phase != "child":
        raise Rejection("Private worker lacks actual full installed-byte verification")
    with TrainingBudget(Path(binding["ledger_path"])).locked(commit=False) as ledger:
        rows = [row for row in ledger["runs"] if row["run_id"] == binding["run_id"]]
        if len(rows) != 1 or rows[0]["state"] != "reserved" or rows[0].get("discovery_claim") != {
                "pid": os.getpid(), "claimed_at": rows[0].get("discovery_claim", {}).get("claimed_at"),
                "manifest_sha256": binding["manifest_sha256"], "nonce": binding["nonce"]}:
            raise Rejection("Zero-update science flow lacks same-process consumed discovery claim")


@dataclass(frozen=True)
class ParentPreflight:
    token: VerifiedDeclaredInputs
    elapsed_seconds: float
    file_count: int

    def record(self) -> dict:
        return {"input_id": self.token.input_id, "nonce": self.token.nonce, "mode": self.token.mode,
                "native_profile_complete": False, "parent_preflight_cpu_seconds": self.elapsed_seconds,
                "verified_file_count": self.file_count}


def preflight_parent(manifest_path: Path, *, hard_seconds: float = 120, require_runtime: bool = True) -> ParentPreflight:
    """Bounded CPU parent check; no reservation, environment or GPU on failure."""
    if hard_seconds != 120:
        raise Rejection("Parent discovery byte preflight is fixed at 120 seconds")
    started = time.monotonic()
    with _cpu_hard_deadline(120):
        manifest = json.loads(manifest_path.read_text())
        token = verify_declared_inputs(manifest, manifest_path=manifest_path, require_runtime=require_runtime)
    elapsed = time.monotonic() - started
    if elapsed > hard_seconds:
        raise Rejection("Parent CPU preflight exceeded 120 seconds before GPU reservation")
    return ParentPreflight(token, elapsed, len(manifest["declared_files"]))


def finalize_profile(manifest: dict, observer: dict, *, phase_journal: list[dict],
                     source_result: dict | None, worker_status: str) -> dict:
    """Preserve unknown origins and unexecuted optimizer modes in a scoped receipt."""
    validate_inputs(manifest)
    declared = {row["path"]: row for row in manifest["declared_files"]}
    used = observer.get("posthoc_actual_use", [])
    unknown = observer.get("unknown_inputs", [])
    failures = []
    if worker_status != "completed": failures.append("source_worker_not_completed")
    if not observer.get("trace_complete"): failures.append("observer_trace_incomplete")
    if observer.get("unresolved"): failures.append("unresolved_file_use")
    if unknown: failures.append("unknown_actual_input_requires_root_review")
    if (source_result is None or source_result.get("stage") != "source_rollout" or
            not source_result.get("capture", {}).get("natural_task_terminal") or
            not source_result.get("evaluation", {}).get("complete_first_source_episode")):
        failures.append("natural_source_terminal_unproven")
    phases = {item.get("phase") for item in phase_journal}
    required = {"source_compile", "source_compile_completed", "env_initialization", "graph_creation",
                "runner_restore", "deterministic_inference", "natural_terminal_reset",
                "save_resume_export", "video_closed", "post_execution_declared_inputs_verified"}
    if not required <= phases: failures.append("phase_coverage_incomplete")
    # JIT recipes are not considered complete merely because an unrecognized
    # generated artifact exists. First run remains provisional pending review.
    payload = {"schema": "microduck_source_profile_dependency_receipt_v2",
               "input_id": manifest["input_id"], "nonce": manifest["nonce"],
               "mode": MODE, "profile": manifest["profile"],
               "installed_cpu_receipt_sha256": manifest["runtime_receipt"]["sha256"],
               "actual_use": used, "observer": observer, "phase_journal": phase_journal,
               "source_result": source_result, "declared_actual_bytes": len(declared),
               "unknown_inputs": unknown, "failures": failures,
               "scope": {"executed": sorted(phases & required),
                         "not_executed": manifest["not_executed"],
                         "ppo_updates_this_run": 0,
                         "source_force_components": "actual qfrc_bias/constraint/own_friction/actuator and dof IDs are in raw trajectory only when capture succeeds"},
               "native_profile_complete": False, "learning_allowed": False,
               "qualification": "provisional finite-profile posthoc dependency observation; root acceptance pending"}
    payload["identity"] = identity(payload)
    return payload


def prepare_source_inputs(*, source_root: Path, runtime_receipt: Path, origin_path: Path,
                          output_root: Path, wheel: Path, manifest_path: Path) -> dict:
    """CPU-only preparation; no MuJoCo model, environment, forward or GPU call."""
    from .profile_identity import make_inputs
    from .source_adapter import load_source, make_skill_cfg
    if manifest_path.exists():
        raise Rejection("Profile input manifest paths are immutable")
    started = time.monotonic()
    with _cpu_hard_deadline(120):
        source = load_source(source_root)
        if source["commit"] != "5946fd9cdbc58956424420153e51975af3b30d77":
            raise Rejection("Pinned official MicroDuck source revision changed")
        _, adoption = make_skill_cfg(source, PROFILE["skill"], substeps=1, seed=PROFILE["seed"])
        if adoption.get("family") != PROFILE["model_family"]:
            raise Rejection("Actual adopted robot family differs from fixed discovery profile")
        tool_files = sorted(Path(__file__).parent.glob("*.py"))
        executables = [wheel.resolve()]
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise Rejection("Real source video requires an installed ffmpeg executable")
        executables.append(Path(ffmpeg))
        manifest = make_inputs(source_root=source_root, runtime_receipt=runtime_receipt,
            origin_path=origin_path, output_root=output_root,
            source_files=[source_root / name for name in source["files"]],
            additional_files=[*tool_files, *executables], source_commit=source["commit"],
            adoption_sha256=adoption["sha256"])
        write_json(manifest_path, manifest)
    elapsed = time.monotonic() - started
    if elapsed > 120:
        raise Rejection("CPU profile preparation exceeded the 120-second preparation bound")
    return {"input_id": manifest["input_id"], "nonce": manifest["nonce"],
            "declared_files": len(manifest["declared_files"]), "prepare_cpu_seconds": elapsed,
            "native_profile_complete": False, "learning_allowed": False}


def _worker(binding_path: Path, result_path: Path, source_output: Path, phase_path: Path) -> int:
    """Reject the retired v2 source child before reading or writing any path."""
    raise Rejection("Private v2 source worker disabled until a new live scope and budget protocol exists")


def _profile_candidate(source_output: Path, manifest_path: Path, review_path: Path,
                       receipt_path: Path, observer_dir: Path, *, actual_result: dict) -> dict:
    from .workflow import bind_candidate, bind_evidence, admit
    provisional_path = source_output / "candidate.json"
    provisional = json.loads(provisional_path.read_text())
    raw_artifacts = {key: Path(value["path"]) for key, value in provisional["artifacts"].items()}
    raw_artifacts.update(source_profile_input=manifest_path, source_profile_review=review_path,
                         source_profile_receipt=receipt_path,
                         source_profile_observer=observer_dir / "observer_report.json",
                         source_profile_phase=source_output.parent / "phase_journal.jsonl")
    for index, path in enumerate(sorted((observer_dir / "trace").glob("trace.[0-9]*"))):
        raw_artifacts[f"source_profile_trace_{index}"] = path
    candidate = bind_candidate(raw_artifacts, provisional["contract"])
    write_json(source_output.parent / "candidate_profile.json", candidate)
    evidence = []
    for stage, report, coverage in (
        ("adoption", {"source_commit": json.loads((source_output / "source_inventory.json").read_text())["commit"]},
          ["actual_registry", "source_runtime_inputs"]),
        ("source_compile", {"compiled_sha256": sha256_file(source_output / "compiled_robot.json")},
          ["full_body", "bam_parameters", "native_hulls"])):
        admit(candidate, stage, evidence)
        report.update(candidate_id=candidate["candidate_id"], status="passed", stage=stage,
                      skill_qualified=False, review_complete=False, learning_allowed=False)
        path = source_output.parent / f"profile_{stage}_evidence.json"
        write_json(path, report)
        evidence.append(bind_evidence(candidate, stage, path, coverage))
    # The actual natural first episode is retained, but dependency provenance
    # is still provisional and root has not reviewed video/phase. Do not issue
    # a passing source_rollout stage from this discovery tool.
    write_json(source_output.parent / "profile_source_rollout_evidence.json",
               {"candidate_id": candidate["candidate_id"], "stage": "source_rollout",
                "status": "diagnostic_only", "capture": actual_result["capture"],
                "independent_evaluation": actual_result["evaluation"],
                "profile_dependency_receipt_sha256": sha256_file(receipt_path),
                "reason": "dependency source scope and video await independent root review",
                "learning_allowed": False, "skill_qualified": False})
    write_json(source_output.parent / "profile_evidence.json", evidence)
    return {"candidate_id": candidate["candidate_id"], "profile_evidence": str((source_output.parent / "profile_evidence.json").resolve())}


def _finish_discovery_budget(ledger_path: Path, run_id: str, *, started: float,
                             deadline: float, status: str) -> tuple[str, float]:
    """Close one run once; never persist a deadline-uncertain completion.

    An in-lock clock check cannot prove when the later replace/fsync becomes
    durable. Until an external completion protocol exists, every discovery
    closure is charged and closed as budget_exhausted. Even a requested failure
    may cross the deadline while committing, so it cannot claim a shorter
    measured budget or an on-time durable failure.
    """
    if status not in {"failed", "budget_exhausted", "captured_provisional_dependencies",
                      "completed_zero_update_discovery"}:
        raise Rejection("Unknown zero-update discovery closure status")
    if not 0 < started < deadline or not all(math.isfinite(value) for value in (started, deadline)):
        raise Rejection("Invalid monotonic discovery deadline")
    budget = TrainingBudget(ledger_path)
    with budget.locked() as ledger:
        rows = [row for row in ledger["runs"] if row["run_id"] == run_id]
        if (len(rows) != 1 or rows[0]["state"] != "reserved" or
                rows[0].get("identity_kind") != "source_profile_input_id"):
            raise Rejection("Discovery closure needs one exact open consumed run")
        row = rows[0]
        decision_at = time.monotonic()
        elapsed_at_decision = max(0.0, decision_at - started)
        crossed = decision_at >= deadline
        positive = status in {"captured_provisional_dependencies", "completed_zero_update_discovery"}
        closed_state = "budget_exhausted"
        reason = ("v13_deadline_crossed_before_commit" if crossed else
                  "v13_completion_commit_unproven" if positive else
                  "v13_noncompletion_commit_deadline_unproven")
        # A later fsync can extend wall time. Keep the observed lower bound
        # distinct from the full reserved budget charged for uncertainty.
        charged_seconds = max(elapsed_at_decision, row["wall_seconds"])
        row.update(state=closed_state, measured_gpu_seconds=elapsed_at_decision * row["gpus"],
                   ended_at=time.time(), deadline_correction=reason,
                   requested_discovery_status=status,
                   closure_elapsed_lower_bound_seconds=elapsed_at_decision,
                   reserved_wall_seconds_charged=row["wall_seconds"],
                   charged_gpu_seconds=charged_seconds * row["gpus"],
                   budget_accounting_basis="conservative_reserved_charge; measured_gpu_seconds is precommit lower bound")
    return closed_state, time.monotonic() - started


def discover_source(manifest_path: Path, review_path: Path, *, ledger_path: Path,
                    output_root: Path, wall_seconds: float = 180) -> dict:
    """No scientific entry until v3 binds finite negatives and external scope."""
    raise Rejection("v12 discovery requires a new scope-bound v3 manifest and review; CPU parser alone cannot start source")


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("source", "runtime-receipt", "origin", "output-root", "wheel", "manifest"):
        prep.add_argument("--" + key, type=Path, required=True)
    run = sub.add_parser("discover")
    for key in ("manifest", "review", "ledger", "output-root"):
        run.add_argument("--" + key, type=Path, required=True)
    run.add_argument("--wall-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_source_inputs(source_root=args.source, runtime_receipt=args.runtime_receipt,
                    origin_path=args.origin, output_root=args.output_root, wheel=args.wheel, manifest_path=args.manifest)
    elif args.command == "discover":
        result = discover_source(args.manifest, args.review, ledger_path=args.ledger,
                                 output_root=args.output_root, wall_seconds=args.wall_seconds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
