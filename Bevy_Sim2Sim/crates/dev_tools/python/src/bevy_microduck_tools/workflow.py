"""Fail-closed identity, stage, review, and measured training-budget gates."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from dataclasses import dataclass
import time
from contextlib import contextmanager
from pathlib import Path

from .serialization import identity, sha256_file
from .authorization import LearningRequest, Rejection, STAGES, authorize


def _learning_binding_request(binding):
    required = {"ledger_path", "run_id", "candidate_id", "review_sha256", "request", "candidate_path", "candidate_file_sha256"}
    if not isinstance(binding, dict) or set(binding) != required or not isinstance(binding["request"], dict):
        raise Rejection("Private learning requires a complete consumed parent run binding")
    try:
        request = LearningRequest(**binding["request"])
    except TypeError:
        raise Rejection("Invalid private learning request binding") from None
    for key in ("ledger_path", "candidate_path"):
        if not isinstance(binding[key], str) or not Path(binding[key]).is_absolute():
            raise Rejection("Private learning binding paths must be absolute")
    for key in ("run_id", "candidate_id", "review_sha256", "candidate_file_sha256"):
        digest = binding[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Rejection("Private learning binding requires exact SHA256 identities")
    if str(Path(binding["ledger_path"]).resolve()) != request.budget_ledger_path:
        raise Rejection("Private learning ledger binding mismatch")
    return request


def bind_candidate(artifacts: dict[str, Path], contract: dict) -> dict:
    required = {"source_inventory", "adoption", "compiled_robot", "bam_parameters", "checkpoint", "onnx", "evaluator"}
    if not required <= artifacts.keys():
        raise Rejection(f"Missing identity artifacts: {sorted(required - artifacts.keys())}")
    payload = {"schema": "microduck_candidate_v1", "contract": contract,
               "artifacts": {key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                             for key, path in artifacts.items()}}
    return {"candidate_id": identity(payload), **payload}


def verify_candidate(candidate: dict) -> None:
    payload = {key: value for key, value in candidate.items() if key != "candidate_id"}
    if identity(payload) != candidate["candidate_id"]:
        raise Rejection("Candidate manifest changed; downstream qualifications invalid")
    checkpoint = Path(candidate["artifacts"]["checkpoint"]["path"])
    if (checkpoint.parent / "superseded.json").exists():
        raise Rejection("Candidate explicitly superseded; old videos and reviews cannot authorize promotion")
    for key, record in candidate["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise Rejection(f"Candidate artifact changed or missing: {key}")


def bind_evidence(candidate: dict, stage: str, path: Path, coverage: list[str]) -> dict:
    if stage not in STAGES or not coverage:
        raise Rejection("Unknown stage or empty test coverage")
    verify_candidate(candidate)
    return {"candidate_id": candidate["candidate_id"], "stage": stage,
            "path": str(path.resolve()), "sha256": sha256_file(path), "coverage": coverage}


def verify_evidence(candidate: dict, evidence: dict) -> dict:
    if evidence["candidate_id"] != candidate["candidate_id"]:
        raise Rejection("Evidence belongs to a different candidate")
    if not evidence.get("coverage"):
        raise Rejection("Empty coverage cannot authorize a stage")
    path = Path(evidence["path"])
    if not path.is_file() or sha256_file(path) != evidence["sha256"]:
        raise Rejection("Evidence changed or missing")
    report = json.loads(path.read_text())
    if report.get("candidate_id") != candidate["candidate_id"] or report.get("status") != "passed":
        raise Rejection("Evidence is not a passing report for this candidate")
    if evidence["stage"] == "source_rollout":
        from .evaluation import evaluate_raw
        raw = candidate["artifacts"].get("source_trajectory")
        video = candidate["artifacts"].get("source_video")
        if raw is None or video is None:
            raise Rejection("Complete source episode and corresponding real video are missing")
        evaluation = evaluate_raw(Path(raw["path"]))
        if not evaluation["complete_first_source_episode"]:
            raise Rejection("Bounded diagnostic horizon cannot satisfy the complete first source episode gate")
        if not report.get("capture", {}).get("natural_task_terminal"):
            raise Rejection("Source report did not preserve a natural task terminal")
    return report


def verify_review(candidate: dict, review: dict, evidence: list[dict], *, stage: str, request: LearningRequest | None = None, now: float | None = None) -> None:
    now = time.time() if now is None else now
    verify_candidate(candidate)
    if review.get("candidate_id") != candidate["candidate_id"]:
        raise Rejection("Review identity mismatch")
    if review.get("reviewer_role") != "root_gpt" or review.get("decision") != "approve":
        raise Rejection("Root GPT approval record required")
    if not (review["reviewed_at"] <= now <= review["expires_at"]):
        raise Rejection("Future or expired review")
    if review.get("evidence_identity") != identity(evidence):
        raise Rejection("Evidence changed since review")
    if not review.get("observations") or not review.get("anomalies_checked"):
        raise Rejection("Review lacks timestamped observations or anomaly coverage")
    for note in review["observations"]:
        if not {"time_seconds", "data_basis", "method", "disposition"} <= note.keys():
            raise Rejection("Incomplete temporal review observation")
    if not review.get("video_artifacts"):
        raise Rejection("Real corresponding video is required for promotion review")
    authorize(review, stage, request)
    for artifact in review["video_artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or not path.stat().st_size or sha256_file(path) != artifact["sha256"]:
            raise Rejection("Review video changed, empty, or missing")


def admit(candidate: dict, stage: str, evidence: list[dict], review: dict | None = None, *,
          request: LearningRequest | None = None) -> None:
    """Complete skill scores are required only at final target/release stages."""
    verify_candidate(candidate)
    if stage not in STAGES:
        raise Rejection(f"Unknown stage {stage}")
    preceding = set(STAGES[:STAGES.index(stage)])
    reports = {}
    for record in evidence:
        reports[record["stage"]] = verify_evidence(candidate, record)
    if not preceding <= reports.keys():
        raise Rejection(f"Missing predecessor evidence: {sorted(preceding - reports.keys())}")
    promotion = stage in {"learning", "selection", "export", "target_validation", "gpt_review", "release"}
    if promotion and review is None:
        raise Rejection("Root GPT temporal review required for candidate promotion")
    if review is not None:
        verify_review(candidate, review, evidence, stage=stage, request=request)
    if stage == "release":
        final = reports["target_validation"]
        expected = set(candidate["contract"]["required_skills"])
        if set(final.get("skill_results", {})) != expected:
            raise Rejection("Incomplete final skill coverage")
        for skill, result in final["skill_results"].items():
            if (result.get("nominal_success", 0) < .95 or result.get("held_out_n", 0) < 200 or
                result.get("held_out_success", 0) < .90 or result.get("paired_ci95_lower", -1) < -.05 or
                result.get("serious_safety_events", 1) != 0 or result.get("independent_training_seeds", 0) < 3):
                raise Rejection(f"Final skill gate failed: {skill}")
        if not final.get("regression_passed") or not final.get("true_target_single_step_60"):
            raise Rejection("Behavior regression or true target 60Hz gate failed")


class TrainingBudget:
    """Persist reservations and measured GPU seconds; concurrent runs share a ledger."""

    RUN_SECONDS = 2 * 3600
    SKILL_GPU_SECONDS = 12 * 3600

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _validate_ledger(data):
        if not isinstance(data, dict) or set(data) != {"runs"} or not isinstance(data["runs"], list):
            raise Rejection("Corrupt budget ledger structure")
        ids = set()
        for row in data["runs"]:
            required = {"skill", "candidate_id", "seed", "wall_seconds", "gpus", "reserved_gpu_seconds", "started_at", "state", "run_id"}
            if not isinstance(row, dict) or not required <= row.keys():
                raise Rejection("Corrupt budget ledger run")
            if not all(isinstance(row[key], str) and row[key] for key in ("skill", "candidate_id", "state", "run_id")):
                raise Rejection("Corrupt budget ledger run identity")
            if row["state"] not in {"reserved", "failed", "budget_exhausted", "completed_structural_smoke", "completed_zero_update_restore", "completed_zero_update_discovery", "captured_provisional_dependencies"}:
                raise Rejection("Unknown budget ledger run state")
            if row.get("identity_kind") == "source_profile_input_id" and row["state"] in {
                    "completed_zero_update_discovery", "captured_provisional_dependencies"}:
                raise Rejection("Discovery ledger cannot claim completion through legacy closure")
            if row["state"] == "reserved" and "measured_gpu_seconds" in row:
                raise Rejection("Open reservation falsely claims measured budget")
            if row["state"] != "reserved" and not {"measured_gpu_seconds", "ended_at"} <= row.keys():
                raise Rejection("Closed reservation lacks actual measured budget")
            if row["run_id"] in ids:
                raise Rejection("Duplicate budget ledger run identity")
            ids.add(row["run_id"])
            if type(row["gpus"]) is not int or row["gpus"] < 1 or type(row["seed"]) is not int:
                raise Rejection("Corrupt budget ledger GPU count or seed")
            for key in ("wall_seconds", "reserved_gpu_seconds", "started_at", "measured_gpu_seconds", "ended_at",
                        "reserved_wall_seconds_charged", "charged_gpu_seconds", "closure_elapsed_lower_bound_seconds"):
                if key in row and (type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0):
                    raise Rejection("Corrupt budget ledger numeric value")
            if row["wall_seconds"] <= 0 or row["reserved_gpu_seconds"] != row["wall_seconds"] * row["gpus"]:
                raise Rejection("Corrupt budget ledger reservation")
            if "charged_gpu_seconds" in row:
                fields = {"reserved_wall_seconds_charged", "closure_elapsed_lower_bound_seconds",
                          "requested_discovery_status", "budget_accounting_basis", "deadline_correction"}
                if (row.get("identity_kind") != "source_profile_input_id" or
                        row["state"] != "budget_exhausted" or not fields <= row.keys() or
                        row["reserved_wall_seconds_charged"] != row["wall_seconds"] or
                        row["charged_gpu_seconds"] < max(row["reserved_gpu_seconds"], row["measured_gpu_seconds"]) or
                        row["measured_gpu_seconds"] != row["closure_elapsed_lower_bound_seconds"] * row["gpus"]):
                    raise Rejection("Corrupt conservative discovery budget charge")
            if "worker_claim" in row:
                claim = row["worker_claim"]
                if ("authorization" not in row or not isinstance(claim, dict) or
                        set(claim) != {"pid", "claimed_at", "candidate_file_sha256"} or
                        type(claim["pid"]) is not int or claim["pid"] < 1 or
                        type(claim["claimed_at"]) not in (int, float) or not math.isfinite(claim["claimed_at"]) or
                        claim["claimed_at"] < row["started_at"]):
                    raise Rejection("Corrupt learning worker claim")
                digest = claim["candidate_file_sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise Rejection("Corrupt worker manifest identity")
            if "authorization" in row:
                auth = row["authorization"]
                if (not isinstance(auth, dict) or set(auth) != {"authorization_id", "review_sha256", "candidate_id", "request"}
                        or auth["candidate_id"] != row["candidate_id"]):
                    raise Rejection("Corrupt authorization ledger binding")
                # Validate immutable UUID and consumed request without inventing a review.
                import uuid
                try:
                    if str(uuid.UUID(auth["authorization_id"])) != auth["authorization_id"]:
                        raise ValueError()
                    digest = auth["review_sha256"]
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError()
                    request = LearningRequest(**auth["request"])
                except (ValueError, TypeError, AttributeError):
                    raise Rejection("Corrupt consumed authorization") from None
                if (request.seed != row["seed"] or request.gpus != row["gpus"] or request.wall_seconds != row["wall_seconds"]):
                    raise Rejection("Consumed request differs from reserved budget")
            if "discovery_authorization" in row:
                from .profile_discovery import DiscoveryRequest, MODE, REQUEST_SCHEMA
                auth = row["discovery_authorization"]
                required_auth = {"authorization_id", "review_sha256", "input_id", "nonce", "mode", "request", "manifest_sha256"}
                if (not isinstance(auth, dict) or set(auth) != required_auth or row.get("identity_kind") != "source_profile_input_id" or
                        auth["input_id"] != row["candidate_id"] or auth["mode"] != MODE or "authorization" in row):
                    raise Rejection("Corrupt zero-update discovery ledger binding")
                try:
                    import uuid
                    if str(uuid.UUID(auth["authorization_id"])) != auth["authorization_id"]:
                        raise ValueError()
                    for key in ("review_sha256", "input_id", "manifest_sha256"):
                        value = auth[key]
                        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                            raise ValueError()
                    record = auth["request"]
                    if not isinstance(record, dict) or record.get("schema") != REQUEST_SCHEMA:
                        raise ValueError()
                    request = DiscoveryRequest(**{key: value for key, value in record.items() if key != "schema"})
                except (ValueError, TypeError, AttributeError, KeyError) as error:
                    raise Rejection(f"Corrupt consumed discovery request: {error}") from error
                if (request.record() != auth["request"] or request.input_id != auth["input_id"] or
                        request.nonce != auth["nonce"] or request.seed != row["seed"] or
                        request.gpus != row["gpus"] or request.wall_seconds != row["wall_seconds"]):
                    raise Rejection("Consumed discovery request differs from reserved budget")
                if "discovery_claim" in row:
                    claim = row["discovery_claim"]
                    if (not isinstance(claim, dict) or set(claim) != {"pid", "claimed_at", "manifest_sha256", "nonce"} or
                            type(claim["pid"]) is not int or claim["pid"] < 1 or
                            type(claim["claimed_at"]) not in (int, float) or claim["claimed_at"] < row["started_at"] or
                            claim["manifest_sha256"] != auth["manifest_sha256"] or claim["nonce"] != auth["nonce"]):
                        raise Rejection("Corrupt discovery worker claim")
            elif "discovery_claim" in row or "identity_kind" in row:
                raise Rejection("Orphan discovery claim or identity kind")

    def _atomic_commit(self, data):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                             prefix=self.path.name + ".", suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def locked(self, *, commit=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text()) if self.path.exists() else {"runs": []}
            except (OSError, ValueError) as error:
                raise Rejection(f"Corrupt or unreadable budget ledger: {error}") from error
            self._validate_ledger(data)
            if any(row.get("authorization", {}).get("request", {}).get("budget_ledger_path", str(self.path.resolve())) != str(self.path.resolve()) for row in data["runs"]):
                raise Rejection("Consumed authorization belongs to a different ledger path")
            if any(row.get("discovery_authorization", {}).get("request", {}).get("budget_ledger_path", str(self.path.resolve())) != str(self.path.resolve()) for row in data["runs"]):
                raise Rejection("Consumed discovery authorization belongs to a different ledger path")
            yield data
            self._validate_ledger(data)
            if commit:
                self._atomic_commit(data)

    def _append_reservation(self, ledger, skill, candidate_id, seed, wall_seconds, gpus, authorization=None):
        if (type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or
                not 0 < wall_seconds <= self.RUN_SECONDS or type(gpus) is not int or gpus < 1):
            raise Rejection("Run budget exceeds two hours or has invalid GPU count")
        if type(seed) is not int or not 1000000 <= seed < 2000000:
            raise Rejection("Training seed must belong to the training partition")
        used = sum(row.get("charged_gpu_seconds", row.get("measured_gpu_seconds", row["reserved_gpu_seconds"]))
                   for row in ledger["runs"] if row["skill"] == skill)
        reserved = wall_seconds * gpus
        if used + reserved > self.SKILL_GPU_SECONDS:
            raise Rejection("Skill twelve GPU-hour budget exhausted")
        row = {"skill": skill, "candidate_id": candidate_id, "seed": seed,
               "wall_seconds": wall_seconds, "gpus": gpus, "reserved_gpu_seconds": reserved,
               "started_at": time.time(), "state": "reserved"}
        if authorization is not None:
            row["authorization"] = authorization
        row["run_id"] = identity(row)
        ledger["runs"].append(row)
        return row["run_id"]

    def reserve(self, skill: str, candidate_id: str, seed: int, wall_seconds: float, *, gpus: int = 1) -> str:
        """Budget accounting only (e.g. zero-update inference); grants no learning capability."""
        with self.locked() as ledger:
            return self._append_reservation(ledger, skill, candidate_id, seed, wall_seconds, gpus)

    def reserve_authorized(self, skill: str, candidate_id: str, review: dict,
                           review_sha256: str, request: LearningRequest) -> str:
        """Consume review permission and budget atomically before any worker starts."""
        authorize(review, "learning", request)
        if review.get("candidate_id") != candidate_id or str(self.path.resolve()) != request.budget_ledger_path:
            raise Rejection("Reservation candidate or ledger differs from authorized request")
        if (not isinstance(review_sha256, str) or len(review_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in review_sha256)):
            raise Rejection("Exact review SHA256 required for authorization consumption")
        with self.locked() as ledger:
            now = time.time()
            if (review.get("reviewer_role") != "root_gpt" or review.get("decision") != "approve" or
                    type(review.get("reviewed_at")) not in (int, float) or type(review.get("expires_at")) not in (int, float) or
                    not review["reviewed_at"] <= now <= review["expires_at"]):
                raise Rejection("Learning reservation requires current root approval at atomic consumption")
            prior = [row["authorization"] for row in ledger["runs"]
                     if row.get("authorization", {}).get("authorization_id") == review["authorization_id"]]
            if any(row["review_sha256"] != review_sha256 or row["candidate_id"] != candidate_id for row in prior):
                raise Rejection("Authorization ID already bound to different review bytes or candidate")
            limit = review["learning_limit"]
            if len(prior) >= limit["max_runs"] or sum(row["request"]["iterations"] for row in prior) + request.iterations > limit["iterations"]:
                raise Rejection("Review learning authorization already consumed or cumulative iteration limit exhausted")
            binding = {"authorization_id": review["authorization_id"], "review_sha256": review_sha256,
                       "candidate_id": candidate_id, "request": request.record()}
            return self._append_reservation(ledger, skill, candidate_id, request.seed, request.wall_seconds, request.gpus, binding)

    def claim_learning_worker(self, binding: dict) -> None:
        """Exactly one private worker may start against a consumed parent run."""
        request = _learning_binding_request(binding)
        if str(self.path.resolve()) != request.budget_ledger_path or binding["ledger_path"] != request.budget_ledger_path:
            raise Rejection("Private learning worker ledger binding mismatch")
        candidate_path = Path(binding["candidate_path"])
        if sha256_file(candidate_path) != binding["candidate_file_sha256"]:
            raise Rejection("Private learning worker parent manifest bytes changed")
        candidate = json.loads(candidate_path.read_text())
        verify_candidate(candidate)
        if candidate["candidate_id"] != binding["candidate_id"]:
            raise Rejection("Private learning worker candidate identity mismatch")
        with self.locked() as ledger:
            rows = [row for row in ledger["runs"] if row["run_id"] == binding["run_id"]]
            if len(rows) != 1:
                raise Rejection("Private learning worker has no unique consumed parent run")
            row = rows[0]
            expected = {"authorization_id": row.get("authorization", {}).get("authorization_id"),
                        "review_sha256": binding["review_sha256"], "candidate_id": binding["candidate_id"],
                        "request": request.record()}
            if row.get("authorization") != expected or row["state"] != "reserved":
                raise Rejection("Private learning worker run is not an open authorized consumption")
            if "worker_claim" in row:
                raise Rejection("Consumed learning run already claimed by a worker")
            if time.time() > row["started_at"] + request.wall_seconds:
                raise Rejection("Private learning worker reservation deadline elapsed")
            row["worker_claim"] = {"pid": os.getpid(), "claimed_at": time.time(),
                                   "candidate_file_sha256": binding["candidate_file_sha256"]}

    def finish(self, run_id: str, elapsed_seconds: float, status: str) -> None:
        if type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise Rejection("Invalid measured elapsed time")
        with self.locked() as ledger:
            rows = [row for row in ledger["runs"] if row["run_id"] == run_id]
            if len(rows) != 1:
                raise Rejection("Budget run identity missing or ambiguous")
            row = rows[0]
            if row["state"] != "reserved":
                raise Rejection("Run already closed")
            if row.get("identity_kind") == "source_profile_input_id":
                raise Rejection("Discovery run requires the single-transaction fail-closed closure")
            row.update(measured_gpu_seconds=elapsed_seconds * row["gpus"], state=status, ended_at=time.time())


def verify_claimed_learning(binding: dict | None, *, iterations: int, seed: int, checkpoint: Path | None) -> None:
    """Read the actual consumed/claimed run before a private scientific entry."""
    request = _learning_binding_request(binding)
    if type(iterations) is not int or iterations != request.iterations or seed != request.seed:
        raise Rejection("Private learning arguments differ from the consumed request")
    candidate_path = Path(binding["candidate_path"])
    if sha256_file(candidate_path) != binding["candidate_file_sha256"]:
        raise Rejection("Private learning parent manifest changed")
    candidate = json.loads(candidate_path.read_text())
    verify_candidate(candidate)
    if (candidate["candidate_id"] != binding["candidate_id"] or checkpoint is None or
            checkpoint.resolve() != Path(candidate["artifacts"]["checkpoint"]["path"]).resolve()):
        raise Rejection("Private learning checkpoint or candidate binding mismatch")
    with TrainingBudget(Path(binding["ledger_path"])).locked(commit=False) as ledger:
        rows = [row for row in ledger["runs"] if row["run_id"] == binding["run_id"]]
        if len(rows) != 1:
            raise Rejection("Private learning consumed run missing or ambiguous")
        row = rows[0]
        auth = row.get("authorization", {})
        if (row["state"] != "reserved" or auth.get("review_sha256") != binding["review_sha256"] or
                auth.get("candidate_id") != binding["candidate_id"] or auth.get("request") != request.record() or
                row.get("worker_claim", {}).get("pid") != os.getpid() or
                row.get("worker_claim", {}).get("candidate_file_sha256") != binding["candidate_file_sha256"]):
            raise Rejection("Private learning has no matching active worker claim")


@dataclass(frozen=True)
class VerifiedRuntimeIdentity:
    """Same-process private flow token; not a lock against concurrent environment writes."""
    candidate_id: str
    run_id: str
    source_root: str
    process_id: int
    runtime_identity: str
    receipt_sha256: str


def verify_claimed_runtime(binding: dict, source_root: Path) -> VerifiedRuntimeIdentity:
    """One full byte check before the private compile/environment/runner flow."""
    from .source_runtime_identity import verify_candidate_runtime
    _learning_binding_request(binding)
    candidate = json.loads(Path(binding["candidate_path"]).read_text())
    if sha256_file(Path(binding["candidate_path"])) != binding["candidate_file_sha256"]:
        raise Rejection("Runtime preflight parent manifest changed")
    result=verify_candidate_runtime(candidate, source_root)
    return VerifiedRuntimeIdentity(candidate["candidate_id"],binding["run_id"],str(source_root.resolve()),
                                   os.getpid(),result["identity"],candidate["artifacts"]["source_runtime_identity"]["sha256"])


def require_verified_runtime(token, binding: dict, source_root: Path) -> None:
    if (not isinstance(token,VerifiedRuntimeIdentity) or token.process_id!=os.getpid() or
            token.candidate_id!=binding["candidate_id"] or token.run_id!=binding["run_id"] or
            token.source_root!=str(source_root.resolve())):
        raise Rejection("Private science flow lacks matching same-process verified runtime identity")
    candidate=json.loads(Path(binding["candidate_path"]).read_text())
    if sha256_file(Path(binding["candidate_path"]))!=binding["candidate_file_sha256"]:
        raise Rejection("Verified runtime parent manifest changed within private science flow")
    artifact=candidate["artifacts"].get("source_runtime_identity")
    if artifact is None or artifact["sha256"]!=token.receipt_sha256 or sha256_file(Path(artifact["path"]))!=token.receipt_sha256:
        raise Rejection("Verified runtime receipt changed within private science flow")
