"""CPU-only v3 identity and read-only review validation.

No function in this module creates a source environment, a systemd scope, a
GPU reservation, review claim, or a learning authorization. The public
discovery entry is still closed. A future live protocol needs its own atomic
single-use review, scope and budget binding.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import uuid
from pathlib import Path

from .authorization import Rejection
from .negative_lookup import SPEC_SCHEMA, seal_contract, verify_contract
from .profile_identity import MODE, PROFILE, validate_inputs
from .serialization import identity

NEGATIVE_SCHEMA = "microduck_v3_presealed_negative_envelope_v1"
SCOPE_SCHEMA = "microduck_v3_scope_plan_cpu_v1"
INPUT_SCHEMA = "microduck_source_profile_inputs_v3_cpu_v2"
TEMPLATE_SCHEMA = "microduck_v3_cpu_review_request_template_v2"
REQUEST_SCHEMA = "microduck_v3_cpu_review_request_v2"
REVIEW_SCHEMA = "microduck_v3_cpu_identity_review_v2"
SOURCE_EXECUTION_ALLOWED = False
MAX_REFERENCE_BYTES = 128 * 1024 * 1024
READ_BLOCK_BYTES = 1024 * 1024


def _fields(value: object, required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != required:
        raise Rejection(f"{label} has unknown or missing fields")
    return value


def _digest(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise Rejection(f"{label} must be a lowercase SHA-256 digest")
    return value


def _uuid(value: object, label: str) -> str:
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError, TypeError):
        raise Rejection(f"{label} must be a canonical UUID") from None
    return value


def _absolute(value: Path | str, label: str) -> str:
    if not isinstance(value, (str, Path)):
        raise Rejection(f"{label} must be an absolute canonical path")
    path = Path(value)
    try:
        canonical = str(path.resolve())
    except (OSError, RuntimeError) as error:
        raise Rejection(f"{label} cannot resolve: {error}") from error
    if not path.is_absolute() or str(path) != canonical:
        raise Rejection(f"{label} must be an absolute canonical path")
    return str(path)


def _pairs_unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise Rejection(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _unknown_constant(value: str):
    raise Rejection(f"Non-JSON numeric constant: {value}")


def _regular_file(path: Path | str, label: str, *, retain_bytes: bool) -> tuple[bytes | None, str]:
    """Never read a pipe/device; hash one bounded regular inode by descriptor."""
    canonical = _absolute(path, label)
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise Rejection(f"{label} is not an accessible regular file: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_REFERENCE_BYTES:
            raise Rejection(f"{label} must be a bounded regular file")
        digest = hashlib.sha256()
        chunks = [] if retain_bytes else None
        length = 0
        while True:
            block = os.read(descriptor, READ_BLOCK_BYTES)
            if not block:
                break
            length += len(block)
            if length > MAX_REFERENCE_BYTES:
                raise Rejection(f"{label} exceeds the bounded byte limit")
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
        current = os.stat(canonical, follow_symlinks=False)
        fingerprint = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                                     value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if fingerprint(before) != fingerprint(after) or fingerprint(before) != fingerprint(current) or length != after.st_size:
            raise Rejection(f"{label} changed while its actual bytes were read")
        return (b"".join(chunks) if chunks is not None else None), digest.hexdigest()
    except OSError as error:
        raise Rejection(f"Cannot read actual {label} regular bytes: {error}") from error
    finally:
        os.close(descriptor)


def _read_document(path: Path | str, label: str) -> tuple[dict, str]:
    raw, digest = _regular_file(path, label, retain_bytes=True)
    assert raw is not None
    try:
        value = json.loads(raw, object_pairs_hook=_pairs_unique,
                           parse_constant=_unknown_constant)
    except (UnicodeError, ValueError) as error:
        raise Rejection(f"{label} is missing or invalid: {error}") from error
    if not isinstance(value, dict):
        raise Rejection(f"{label} must contain one JSON object")
    return value, digest


def _v2(path: Path | str) -> tuple[dict, str]:
    manifest, digest = _read_document(path, "v2 source input manifest")
    validate_inputs(manifest)
    return manifest, digest


def _boot_id() -> str:
    try:
        return _uuid(Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "boot ID")
    except OSError as error:
        raise Rejection(f"Cannot bind local boot ID: {error}") from error


def _actual_sha(path: Path, label: str) -> str:
    _, digest = _regular_file(path, label, retain_bytes=False)
    return digest


def _unit_name(nonce: str) -> str:
    return f"bevy-profile-v3-{uuid.UUID(nonce).hex}.scope"


def make_negative_envelope_v3(v2_manifest_path: Path, rule_spec: dict) -> dict:
    """Re-seal the current filesystem; an old v12 contract is not an input."""
    manifest, manifest_sha = _v2(v2_manifest_path)
    _fields(rule_spec, {"schema", "rules"}, "v3 finite rule specification")
    if rule_spec["schema"] != SPEC_SCHEMA:
        raise Rejection("v3 requires a finite rule specification, not an old sealed contract")
    inner = seal_contract(rule_spec)
    value = {"schema": NEGATIVE_SCHEMA, "v2_manifest_sha256": manifest_sha,
             "v2_input_id": manifest["input_id"], "nonce": manifest["nonce"],
             "output_root": manifest["output_root"], "sealed_at_unix_ns": time.time_ns(),
             "seal_kind": "fresh_cpu_reseal_before_review", "sealed_contract": inner,
             "sealed_contract_id": inner["contract_id"],
             "source_execution_allowed": SOURCE_EXECUTION_ALLOWED}
    value["envelope_id"] = identity(value)
    return value


def _validate_negative_envelope(value: dict, manifest: dict, manifest_sha: str) -> None:
    _fields(value, {"schema", "v2_manifest_sha256", "v2_input_id", "nonce", "output_root",
                    "sealed_at_unix_ns", "seal_kind", "sealed_contract", "sealed_contract_id",
                    "source_execution_allowed", "envelope_id"}, "v3 negative envelope")
    if (value["schema"] != NEGATIVE_SCHEMA or value["seal_kind"] != "fresh_cpu_reseal_before_review" or
            value["source_execution_allowed"] is not False or
            value["v2_manifest_sha256"] != manifest_sha or
            value["v2_input_id"] != manifest["input_id"] or value["nonce"] != manifest["nonce"] or
            value["output_root"] != manifest["output_root"] or
            type(value["sealed_at_unix_ns"]) is not int or value["sealed_at_unix_ns"] <= 0 or
            value["sealed_at_unix_ns"] > time.time_ns() or
            not isinstance(value["sealed_contract"], dict) or
            value["sealed_contract_id"] != value["sealed_contract"].get("contract_id") or
            identity({key: item for key, item in value.items() if key != "envelope_id"}) != value["envelope_id"]):
        raise Rejection("v3 negative envelope changed or is not bound to this v2 input")
    _digest(value["sealed_contract_id"], "sealed contract identity")
    verify_contract(value["sealed_contract"])


def make_scope_plan_v3(v2_manifest_path: Path, *, parent_cgroup: str,
                       supervisor_path: Path) -> dict:
    """Describe an exact future scope without starting a unit or worker."""
    manifest, manifest_sha = _v2(v2_manifest_path)
    parent = _parent_cgroup(parent_cgroup)
    executable = _absolute(supervisor_path, "supervisor path")
    if not Path(executable).is_file():
        raise Rejection("Scope supervisor source is not an actual file")
    value = {"schema": SCOPE_SCHEMA, "v2_manifest_sha256": manifest_sha,
             "nonce": manifest["nonce"], "output_root": manifest["output_root"],
             "unit_name": _unit_name(manifest["nonce"]), "expected_parent_cgroup": parent,
             "runtime_max_seconds": 180, "kill_mode": "control-group",
             "send_sigkill": True, "pre_science_barrier": True,
             "supervisor_path": executable, "supervisor_sha256": _actual_sha(Path(executable), "supervisor"),
             "boot_id": _boot_id(), "scope_launch_enabled": False}
    value["scope_plan_id"] = identity(value)
    return value


def _parent_cgroup(value: object) -> str:
    if (not isinstance(value, str) or not value.startswith("/user.slice/") or
            not value.endswith("/app.slice") or "//" in value or "\x00" in value or
            any(part in {"", ".", ".."} for part in value.split("/")[1:])):
        raise Rejection("Scope plan needs one exact user app.slice parent")
    return value


def _validate_scope_plan(value: dict, manifest: dict, manifest_sha: str) -> None:
    _fields(value, {"schema", "v2_manifest_sha256", "nonce", "output_root", "unit_name",
                    "expected_parent_cgroup", "runtime_max_seconds", "kill_mode",
                    "send_sigkill", "pre_science_barrier", "supervisor_path",
                    "supervisor_sha256", "boot_id", "scope_launch_enabled", "scope_plan_id"},
            "v3 scope plan")
    executable = _absolute(value["supervisor_path"], "scope supervisor path")
    if (value["schema"] != SCOPE_SCHEMA or value["v2_manifest_sha256"] != manifest_sha or
            value["nonce"] != manifest["nonce"] or value["output_root"] != manifest["output_root"] or
            value["unit_name"] != _unit_name(manifest["nonce"]) or
            value["runtime_max_seconds"] != 180 or type(value["runtime_max_seconds"]) is not int or
            value["kill_mode"] != "control-group" or value["send_sigkill"] is not True or
            value["pre_science_barrier"] is not True or value["scope_launch_enabled"] is not False or
            value["boot_id"] != _boot_id() or
            value["supervisor_sha256"] != _actual_sha(Path(executable), "supervisor") or
            identity({key: item for key, item in value.items() if key != "scope_plan_id"}) != value["scope_plan_id"]):
        raise Rejection("v3 scope plan changed or cannot authorize even a CPU identity fixture")
    _parent_cgroup(value["expected_parent_cgroup"])
    _digest(value["supervisor_sha256"], "supervisor SHA")


def _reference(path: Path | str, file_sha: str, item_id: str) -> dict:
    return {"path": _absolute(path, "v3 referenced file"),
            "file_sha256": _digest(file_sha, "referenced file SHA"),
            "identity": _digest(item_id, "referenced content identity")}


def _template(manifest: dict) -> dict:
    return {"schema": TEMPLATE_SCHEMA, "mode": MODE, "nonce": manifest["nonce"],
            "output_root": manifest["output_root"], "seed": PROFILE["seed"],
            "gpus": 1, "max_wall_seconds": 180, "single_use_enforced": False,
            "learning_allowed": False, "source_execution_allowed": False}


def make_inputs_v3(v2_manifest_path: Path, negative_envelope_path: Path,
                   scope_plan_path: Path) -> dict:
    manifest, manifest_sha = _v2(v2_manifest_path)
    negative, negative_sha = _read_document(negative_envelope_path, "v3 negative envelope")
    _validate_negative_envelope(negative, manifest, manifest_sha)
    scope, scope_sha = _read_document(scope_plan_path, "v3 scope plan")
    _validate_scope_plan(scope, manifest, manifest_sha)
    value = {"schema": INPUT_SCHEMA, "mode": MODE, "nonce": manifest["nonce"],
             "output_root": manifest["output_root"],
             "v2_source": _reference(v2_manifest_path, manifest_sha, manifest["input_id"]),
             "negative_envelope": _reference(negative_envelope_path, negative_sha, negative["envelope_id"]),
             "scope_plan": _reference(scope_plan_path, scope_sha, scope["scope_plan_id"]),
             "review_request_template": _template(manifest),
             "qualification": "read_only_cpu_identity; single-use and source execution not implemented",
             "source_execution_allowed": False, "learning_allowed": False}
    value["input_id"] = identity(value)
    return value


def validate_inputs_v3(value: dict) -> None:
    _fields(value, {"schema", "mode", "nonce", "output_root", "v2_source",
                    "negative_envelope", "scope_plan", "review_request_template",
                    "qualification", "source_execution_allowed", "learning_allowed", "input_id"},
            "v3 CPU source input")
    if (value["schema"] != INPUT_SCHEMA or value["mode"] != MODE or
            value["qualification"] != "read_only_cpu_identity; single-use and source execution not implemented" or
            value["source_execution_allowed"] is not False or value["learning_allowed"] is not False or
            identity({key: item for key, item in value.items() if key != "input_id"}) != value["input_id"]):
        raise Rejection("v3 input identity or CPU-only scope changed")
    _digest(value["input_id"], "v3 input identity")
    _uuid(value["nonce"], "v3 nonce")
    _absolute(value["output_root"], "v3 output root")
    for name in ("v2_source", "negative_envelope", "scope_plan"):
        _fields(value[name], {"path", "file_sha256", "identity"}, f"v3 {name} reference")
        _absolute(value[name]["path"], f"v3 {name} path")
        _digest(value[name]["file_sha256"], f"v3 {name} file SHA")
        _digest(value[name]["identity"], f"v3 {name} identity")
    manifest, manifest_sha = _v2(value["v2_source"]["path"])
    if (manifest_sha != value["v2_source"]["file_sha256"] or
            manifest["input_id"] != value["v2_source"]["identity"] or
            manifest["nonce"] != value["nonce"] or manifest["output_root"] != value["output_root"]):
        raise Rejection("Referenced v2 source manifest bytes or identity changed")
    negative, negative_sha = _read_document(value["negative_envelope"]["path"], "v3 negative envelope")
    _validate_negative_envelope(negative, manifest, manifest_sha)
    if (negative_sha != value["negative_envelope"]["file_sha256"] or
            negative["envelope_id"] != value["negative_envelope"]["identity"]):
        raise Rejection("Referenced v3 negative envelope bytes changed")
    scope, scope_sha = _read_document(value["scope_plan"]["path"], "v3 scope plan")
    _validate_scope_plan(scope, manifest, manifest_sha)
    if (scope_sha != value["scope_plan"]["file_sha256"] or
            scope["scope_plan_id"] != value["scope_plan"]["identity"]):
        raise Rejection("Referenced v3 scope plan bytes changed")
    _fields(value["review_request_template"], {"schema", "mode", "nonce", "output_root",
                "seed", "gpus", "max_wall_seconds", "single_use_enforced",
                "learning_allowed", "source_execution_allowed"}, "v3 review request template")
    if value["review_request_template"] != _template(manifest):
        raise Rejection("v3 review request template changed or contains an identity cycle")


def review_request_v3_cpu(v3_manifest_path: Path) -> dict:
    manifest, manifest_sha = _read_document(v3_manifest_path, "v3 source input manifest")
    validate_inputs_v3(manifest)
    value = {"schema": REQUEST_SCHEMA, "input_id": manifest["input_id"],
             "v3_manifest_file_sha256": manifest_sha,
             "v2_manifest_file_sha256": manifest["v2_source"]["file_sha256"],
             "negative_envelope_file_sha256": manifest["negative_envelope"]["file_sha256"],
             "scope_plan_file_sha256": manifest["scope_plan"]["file_sha256"],
             "template": manifest["review_request_template"]}
    value["request_id"] = identity(value)
    return value


def _validate_cpu_identity_review(review: dict, request: dict, sealed_at_unix_ns: int,
                                  *, now: float | None = None) -> None:
    _fields(review, {"schema", "review_id", "reviewer_role", "decision",
                     "reviewed_at", "expires_at", "request", "single_use_enforced",
                     "learning_allowed", "source_execution_allowed"}, "v3 CPU identity review")
    _uuid(review["review_id"], "v3 CPU identity review ID")
    instant = time.time() if now is None else now
    if (review["schema"] != REVIEW_SCHEMA or review["reviewer_role"] != "root_gpt" or
            review["decision"] != "acknowledge_cpu_identity_only" or
            type(review["reviewed_at"]) not in (int, float) or
            type(review["expires_at"]) not in (int, float) or
            not all(math.isfinite(item) for item in (review["reviewed_at"], review["expires_at"], instant)) or
            not sealed_at_unix_ns / 1e9 <= review["reviewed_at"] <= instant <= review["expires_at"] or
            review["request"] != request or review["single_use_enforced"] is not False or
            review["learning_allowed"] is not False or review["source_execution_allowed"] is not False):
        raise Rejection("v3 CPU identity review does not match this exact read-only request")


def verify_review_v3_cpu(v3_manifest_path: Path, review_path: Path) -> None:
    """Validate an exact, current CPU identity review without consuming it.

    Passing this read-only check does not authorize a source run or establish
    single-use semantics. A future live protocol requires a separate trusted
    atomic review, scope and budget transaction.
    """
    request = review_request_v3_cpu(v3_manifest_path)
    review, review_sha = _read_document(review_path, "v3 CPU identity review")
    manifest, _ = _read_document(v3_manifest_path, "v3 source input manifest")
    negative, _ = _read_document(manifest["negative_envelope"]["path"], "v3 negative envelope")
    _validate_cpu_identity_review(review, request, negative["sealed_at_unix_ns"])
    if (review_request_v3_cpu(v3_manifest_path) != request or
            _read_document(review_path, "v3 CPU identity review")[1] != review_sha):
        raise Rejection("v3 identity or review changed during read-only verification")
