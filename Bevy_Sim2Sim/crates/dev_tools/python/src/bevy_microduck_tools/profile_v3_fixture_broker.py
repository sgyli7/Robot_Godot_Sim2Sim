"""CPU-only single-use admission fixture; never grants a source capability.

The SQLite store is deliberately separate from TrainingBudget. Its directory is
owned by the caller, so this module cannot prove a production trust boundary
against a hostile process with the same UID. No source worker imports this API.
"""
from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .authorization import Rejection
from .profile_v3_cpu import _absolute, _read_document, review_request_v3_cpu, validate_inputs_v3
from .serialization import identity

STORE_SCHEMA = "microduck_v3_cpu_fixture_store_v1"
SCOPE_SCHEMA = "microduck_v3_cpu_fake_scope_v1"
REVIEW_SCHEMA = "microduck_v3_cpu_fixture_claim_review_v1"
RECEIPT_SCHEMA = "microduck_v3_cpu_fixture_claim_receipt_v1"
DATABASE = "admission.sqlite3"
RESERVED_GPU_SECONDS = 180
SOURCE_EXECUTION_ALLOWED = False


def _fields(value: object, required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != required:
        raise Rejection(f"{label} has unknown or missing fields")
    return value


def _uuid(value: object, label: str) -> str:
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise Rejection(f"{label} must be a canonical UUID") from None
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise Rejection(f"CPU fixture document is not canonical JSON: {error}") from error


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev, right.st_ino, stat.S_IFMT(right.st_mode))


@contextmanager
def _state_directory(path: Path):
    canonical = _absolute(path, "CPU fixture state directory")
    try:
        descriptor = os.open(canonical, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise Rejection(f"CPU fixture state directory cannot open: {error}") from error
    try:
        state = os.fstat(descriptor)
        if (not stat.S_ISDIR(state.st_mode) or state.st_uid != os.getuid() or
                state.st_mode & 0o077):
            raise Rejection("CPU fixture state directory must be caller-owned mode 0700")
        _check_namespace(Path(canonical), descriptor, database=False)
        yield Path(canonical), descriptor
    finally:
        os.close(descriptor)


def _check_namespace(path: Path, directory_fd: int, *, database: bool,
                     expected_database: os.stat_result | None = None) -> None:
    """Detect observed path pivots; this is not same-UID adversarial isolation."""
    try:
        if not _same_inode(os.fstat(directory_fd), os.stat(path, follow_symlinks=False)):
            raise Rejection("CPU fixture state directory pathname changed")
        if database:
            anchored = os.stat(DATABASE, dir_fd=directory_fd, follow_symlinks=False)
            public = os.stat(path / DATABASE, follow_symlinks=False)
            if (not stat.S_ISREG(anchored.st_mode) or not _same_inode(anchored, public) or
                    (expected_database is not None and not _same_inode(anchored, expected_database))):
                raise Rejection("CPU fixture database pathname changed")
    except OSError as error:
        raise Rejection(f"CPU fixture state path is missing or changed: {error}") from error


def _connect(directory_fd: int, *, create: bool) -> sqlite3.Connection:
    # /proc/self/fd anchors DB and journal lookups to the already opened parent.
    mode = "rwc" if create else "rw"
    uri = f"file:/proc/self/fd/{directory_fd}/{DATABASE}?mode={mode}"
    connection = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("PRAGMA synchronous=EXTRA")
    if create:
        if connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
            raise Rejection("CPU fixture database needs rollback journal mode")
    elif connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
        raise Rejection("CPU fixture database journal mode changed")
    return connection


def initialize_cpu_fixture_store(state_root: Path, *, budget_cap_gpu_seconds: int) -> str:
    """Create an isolated test store; this never opens a real TrainingBudget."""
    if type(budget_cap_gpu_seconds) is not int or budget_cap_gpu_seconds < RESERVED_GPU_SECONDS:
        raise Rejection("CPU fixture budget cap must cover one conservative reservation")
    canonical = _absolute(state_root, "CPU fixture state directory")
    try:
        Path(canonical).mkdir(mode=0o700)
    except OSError as error:
        raise Rejection(f"CPU fixture state directory must be new: {error}") from error
    with _state_directory(Path(canonical)) as (path, directory_fd):
        try:
            connection = _connect(directory_fd, create=True)
            try:
                store_id = str(uuid.uuid4())
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK (id = 1), "
                                   "schema TEXT NOT NULL, store_id TEXT NOT NULL, "
                                   "budget_cap_gpu_seconds INTEGER NOT NULL)")
                connection.execute("CREATE TABLE claims ("
                                   "claim_id TEXT PRIMARY KEY, authorization_id TEXT NOT NULL UNIQUE, "
                                   "nonce TEXT NOT NULL UNIQUE, input_id TEXT NOT NULL UNIQUE, "
                                   "scope_id TEXT NOT NULL UNIQUE, request_id TEXT NOT NULL, "
                                   "v3_manifest_sha256 TEXT NOT NULL, review_sha256 TEXT NOT NULL, "
                                   "scope_sha256 TEXT NOT NULL, reserved_gpu_seconds INTEGER NOT NULL, "
                                   "state TEXT NOT NULL CHECK (state = 'consumed_cpu_fixture'), "
                                   "consumed_at_unix_ns INTEGER NOT NULL, expires_at_unix_ns INTEGER NOT NULL)")
                connection.execute("INSERT INTO metadata VALUES (1, ?, ?, ?)",
                                   (STORE_SCHEMA, store_id, budget_cap_gpu_seconds))
                connection.commit()
            finally:
                connection.close()
            os.fsync(directory_fd)
            _check_namespace(path, directory_fd, database=True)
            return store_id
        except sqlite3.Error as error:
            raise Rejection(f"CPU fixture store initialization failed: {error}") from error


def make_fake_scope_v3_cpu(v3_manifest_path: Path, *, fake_leaf_uuid: str) -> dict:
    """Describe a fake CPU barrier. This is never a systemd attestation."""
    _uuid(fake_leaf_uuid, "fake scope leaf")
    manifest, _ = _read_document(v3_manifest_path, "v3 CPU fixture manifest")
    validate_inputs_v3(manifest)
    plan, _ = _read_document(manifest["scope_plan"]["path"], "v3 CPU fixture scope plan")
    value = {"schema": SCOPE_SCHEMA, "kind": "fake_cpu_barrier_only",
             "input_id": manifest["input_id"], "nonce": manifest["nonce"],
             "scope_plan_id": plan["scope_plan_id"], "boot_id": plan["boot_id"],
             "unit_name": plan["unit_name"], "parent_cgroup": plan["expected_parent_cgroup"],
             "fake_leaf_uuid": fake_leaf_uuid, "started_monotonic_ns": time.monotonic_ns(),
             "source_execution_allowed": False, "learning_allowed": False}
    value["scope_id"] = identity(value)
    return value


def _validate_fake_scope(scope: dict, manifest: dict) -> None:
    _fields(scope, {"schema", "kind", "input_id", "nonce", "scope_plan_id", "boot_id",
                    "unit_name", "parent_cgroup", "fake_leaf_uuid", "started_monotonic_ns",
                    "source_execution_allowed", "learning_allowed", "scope_id"}, "fake CPU scope")
    plan, _ = _read_document(manifest["scope_plan"]["path"], "v3 CPU fixture scope plan")
    _uuid(scope["fake_leaf_uuid"], "fake scope leaf")
    now = time.monotonic_ns()
    if (scope["schema"] != SCOPE_SCHEMA or scope["kind"] != "fake_cpu_barrier_only" or
            scope["input_id"] != manifest["input_id"] or scope["nonce"] != manifest["nonce"] or
            scope["scope_plan_id"] != plan["scope_plan_id"] or scope["boot_id"] != plan["boot_id"] or
            scope["unit_name"] != plan["unit_name"] or
            scope["parent_cgroup"] != plan["expected_parent_cgroup"] or
            type(scope["started_monotonic_ns"]) is not int or
            not 0 < scope["started_monotonic_ns"] <= now or
            now - scope["started_monotonic_ns"] >= RESERVED_GPU_SECONDS * 1_000_000_000 or
            scope["source_execution_allowed"] is not False or scope["learning_allowed"] is not False or
            identity({key: item for key, item in scope.items() if key != "scope_id"}) != scope["scope_id"]):
        raise Rejection("Fake CPU scope identity or deadline changed")


def _check_review(review: dict, review_sha: str, request: dict, scope: dict, scope_sha: str,
                  manifest: dict, state_root: Path) -> None:
    _fields(review, {"schema", "authorization_id", "reviewer_role", "decision",
                     "reviewed_at_unix_ns", "expires_at_unix_ns", "request",
                     "fake_scope_path", "fake_scope_sha256", "fake_scope_id", "state_root", "store_id",
                     "max_runs", "max_wall_seconds", "gpus", "reserved_gpu_seconds",
                     "single_use_scope", "source_execution_allowed", "learning_allowed"},
            "CPU fixture claim review")
    _uuid(review["authorization_id"], "CPU fixture authorization ID")
    _uuid(review["store_id"], "CPU fixture store ID")
    negative, _ = _read_document(manifest["negative_envelope"]["path"], "v3 negative envelope")
    now = time.time_ns()
    if (review["schema"] != REVIEW_SCHEMA or review["reviewer_role"] != "root_gpt" or
            review["decision"] != "approve_cpu_fixture_claim_only" or
            type(review["reviewed_at_unix_ns"]) is not int or
            type(review["expires_at_unix_ns"]) is not int or
            not negative["sealed_at_unix_ns"] <= review["reviewed_at_unix_ns"] <= now < review["expires_at_unix_ns"] or
            _canonical_json(review["request"]) != _canonical_json(request) or
            review["fake_scope_path"] != _absolute(review["fake_scope_path"], "fake scope path") or
            review["fake_scope_sha256"] != scope_sha or review["fake_scope_id"] != scope["scope_id"] or
            review["state_root"] != str(state_root) or
            type(review["max_runs"]) is not int or review["max_runs"] != 1 or
            type(review["max_wall_seconds"]) is not int or review["max_wall_seconds"] != 180 or
            type(review["gpus"]) is not int or review["gpus"] != 1 or
            type(review["reserved_gpu_seconds"]) is not int or
            review["reserved_gpu_seconds"] != RESERVED_GPU_SECONDS or
            review["single_use_scope"] != "cooperative_cpu_fixture_only" or
            review["source_execution_allowed"] is not False or review["learning_allowed"] is not False):
        raise Rejection("CPU fixture review is expired, drifted or could claim science capability")
    if len(review_sha) != 64:
        raise Rejection("CPU fixture review bytes have no SHA-256")


def _current_inputs(v3_manifest_path: Path, fake_scope_path: Path, review_path: Path,
                    state_root: Path) -> tuple[dict, str, dict, str, dict, str, dict]:
    request = review_request_v3_cpu(v3_manifest_path)
    manifest, manifest_sha = _read_document(v3_manifest_path, "v3 CPU fixture manifest")
    if request["v3_manifest_file_sha256"] != manifest_sha or request["input_id"] != manifest["input_id"]:
        raise Rejection("v3 manifest changed between request and fixture admission")
    scope, scope_sha = _read_document(fake_scope_path, "fake CPU scope")
    _validate_fake_scope(scope, manifest)
    review, review_sha = _read_document(review_path, "CPU fixture claim review")
    _check_review(review, review_sha, request, scope, scope_sha, manifest, state_root)
    if review["fake_scope_path"] != _absolute(fake_scope_path, "fake scope path"):
        raise Rejection("CPU fixture review names another fake scope path")
    return manifest, manifest_sha, scope, scope_sha, review, review_sha, request


def _commit(connection: sqlite3.Connection) -> None:
    """A small seam for deterministic CPU fault injection; never bypassed live."""
    connection.commit()


def claim_cpu_fixture(v3_manifest_path: Path, fake_scope_path: Path, review_path: Path,
                      state_root: Path) -> dict:
    """Consume once in one scratch DB; return no barrier or source capability.

    A durable row may remain when post-commit expiry/drift rejects. That row is
    burned: it is never refunded and cannot be used to launch source.
    """
    state_root = Path(_absolute(state_root, "CPU fixture state directory"))
    first = _current_inputs(v3_manifest_path, fake_scope_path, review_path, state_root)
    manifest, manifest_sha, scope, scope_sha, review, review_sha, request = first
    with _state_directory(state_root) as (path, directory_fd):
        try:
            expected_database = os.stat(DATABASE, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise Rejection(f"CPU fixture database is missing: {error}") from error
        _check_namespace(path, directory_fd, database=True, expected_database=expected_database)
        try:
            connection = _connect(directory_fd, create=False)
            committed = False
            try:
                _check_namespace(path, directory_fd, database=True, expected_database=expected_database)
                connection.execute("BEGIN IMMEDIATE")
                second = _current_inputs(v3_manifest_path, fake_scope_path, review_path, state_root)
                if (second[1], second[3], second[5], second[6]) != (manifest_sha, scope_sha, review_sha, request):
                    raise Rejection("CPU fixture input, scope or review bytes changed before consumption")
                _check_namespace(path, directory_fd, database=True, expected_database=expected_database)
                metadata = connection.execute("SELECT schema, store_id, budget_cap_gpu_seconds FROM metadata WHERE id=1").fetchone()
                if metadata is None or metadata[0] != STORE_SCHEMA or metadata[1] != review["store_id"]:
                    raise Rejection("Unknown CPU fixture budget store")
                spent = connection.execute("SELECT COALESCE(SUM(reserved_gpu_seconds), 0) FROM claims").fetchone()[0]
                if spent + RESERVED_GPU_SECONDS > metadata[2]:
                    raise Rejection("CPU fixture conservative budget exhausted")
                claim_id = identity({"authorization_id": review["authorization_id"],
                                     "review_sha256": review_sha, "input_id": manifest["input_id"],
                                     "nonce": manifest["nonce"], "scope_id": scope["scope_id"],
                                     "request_id": request["request_id"], "state_root": str(state_root),
                                     "store_id": review["store_id"]})
                connection.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (claim_id, review["authorization_id"], manifest["nonce"],
                                    manifest["input_id"], scope["scope_id"], request["request_id"],
                                    manifest_sha, review_sha, scope_sha, RESERVED_GPU_SECONDS,
                                    "consumed_cpu_fixture", time.time_ns(), review["expires_at_unix_ns"]))
                _check_namespace(path, directory_fd, database=True, expected_database=expected_database)
                _commit(connection)
                committed = True
            except BaseException:
                if not committed:
                    connection.rollback()
                raise
            finally:
                connection.close()
            os.fsync(directory_fd)
            _check_namespace(path, directory_fd, database=True, expected_database=expected_database)
            check = _connect(directory_fd, create=False)
            try:
                row = check.execute("SELECT claim_id, state FROM claims WHERE authorization_id=?",
                                    (review["authorization_id"],)).fetchone()
            finally:
                check.close()
            if row != (claim_id, "consumed_cpu_fixture"):
                raise Rejection("CPU fixture consumption not visible after durable commit")
            third = _current_inputs(v3_manifest_path, fake_scope_path, review_path, state_root)
            if (third[1], third[3], third[5], third[6]) != (manifest_sha, scope_sha, review_sha, request):
                raise Rejection("CPU fixture input, scope or review bytes changed after consumption")
            if time.monotonic_ns() - scope["started_monotonic_ns"] >= RESERVED_GPU_SECONDS * 1_000_000_000:
                raise Rejection("Fake CPU scope resource deadline passed after consumption")
            return {"schema": RECEIPT_SCHEMA, "claim_id": claim_id,
                    "authorization_id": review["authorization_id"],
                    "input_id": manifest["input_id"], "scope_id": scope["scope_id"],
                    "store_id": review["store_id"],
                    "reserved_gpu_seconds": RESERVED_GPU_SECONDS,
                    "cpu_fixture_claim_recorded": True,
                    "source_execution_allowed": False, "learning_allowed": False,
                    "qualification": "stable_store_cpu_fixture_only; no barrier or production authorization"}
        except sqlite3.Error as error:
            raise Rejection(f"CPU fixture transaction failed or was already consumed: {error}") from error


def inspect_cpu_fixture_store(state_root: Path) -> dict:
    """Read-only fixture accounting for tests; cannot authorize any worker."""
    with _state_directory(state_root) as (path, directory_fd):
        _check_namespace(path, directory_fd, database=True)
        try:
            connection = _connect(directory_fd, create=False)
            try:
                metadata = connection.execute("SELECT schema, store_id, budget_cap_gpu_seconds FROM metadata WHERE id=1").fetchone()
                claims = connection.execute("SELECT claim_id, authorization_id, nonce, input_id, scope_id, "
                                            "reserved_gpu_seconds, state FROM claims ORDER BY claim_id").fetchall()
            finally:
                connection.close()
            if metadata is None or metadata[0] != STORE_SCHEMA:
                raise Rejection("Unknown CPU fixture budget store")
            return {"schema": STORE_SCHEMA, "store_id": metadata[1],
                    "budget_cap_gpu_seconds": metadata[2],
                    "charged_gpu_seconds": sum(row[5] for row in claims),
                    "claims": [dict(zip(("claim_id", "authorization_id", "nonce", "input_id",
                                         "scope_id", "reserved_gpu_seconds", "state"), row)) for row in claims]}
        except sqlite3.Error as error:
            raise Rejection(f"Cannot inspect CPU fixture store: {error}") from error
