"""Read-only v3 CPU identity and review validation; no source or GPU."""
from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from bevy_microduck_tools import profile_v3_cpu
from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.negative_lookup import SPEC_SCHEMA, seal_contract
from bevy_microduck_tools.profile_discovery import discover_source
from bevy_microduck_tools.profile_v3_cpu import (
    INPUT_SCHEMA, REVIEW_SCHEMA, SCOPE_SCHEMA, make_inputs_v3,
    make_negative_envelope_v3, make_scope_plan_v3, review_request_v3_cpu,
    validate_inputs_v3, verify_review_v3_cpu)
from bevy_microduck_tools.serialization import identity, write_json
from test_profile_v10 import _fixture


PARENT_CGROUP = "/user.slice/user-1000.slice/user@1000.service/app.slice"


def _prepared(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _, v2_path, _, old_review_path, _ = _fixture(tmp_path)
    # The v12 witness inventories every ancestor directory. Keep its search
    # directory outside the scratch output tree, whose files change below.
    missing = Path("/etc") / f"bevy-v3-{uuid.uuid4().hex}.missing"
    specification = {"schema": SPEC_SCHEMA, "rules": [{
        "rule_id": "loader_candidate", "path": str(missing), "syscall": "openat",
        "errno": "ENOENT", "phase": "before_first_science_phase", "dirfd": "AT_FDCWD",
        "qualifier": "O_RDONLY", "max_count": 2}]}
    negative_path = tmp_path / "negative_v3.json"
    envelope = make_negative_envelope_v3(v2_path, specification)
    write_json(negative_path, envelope)
    supervisor = tmp_path / "future_supervisor.py"
    supervisor.write_text("# CPU-only identity fixture; never executed\n")
    scope_path = tmp_path / "scope_v3.json"
    scope = make_scope_plan_v3(v2_path, parent_cgroup=PARENT_CGROUP,
                               supervisor_path=supervisor)
    write_json(scope_path, scope)
    v3_path = tmp_path / "inputs_v3.json"
    v3 = make_inputs_v3(v2_path, negative_path, scope_path)
    write_json(v3_path, v3)
    request = review_request_v3_cpu(v3_path)
    review = {"schema": REVIEW_SCHEMA, "review_id": str(uuid.uuid4()),
              "reviewer_role": "root_gpt", "decision": "acknowledge_cpu_identity_only",
              "reviewed_at": time.time(), "expires_at": time.time() + 90,
              "request": request, "single_use_enforced": False,
              "learning_allowed": False, "source_execution_allowed": False}
    review_path = tmp_path / "review_v3.json"
    write_json(review_path, review)
    return locals()


def test_v3_identity_is_acyclic_exact_and_review_has_no_claim(tmp_path):
    item = _prepared(tmp_path)
    v3 = item["v3"]
    request = item["request"]
    validate_inputs_v3(v3)
    assert v3["schema"] == INPUT_SCHEMA
    assert v3["input_id"] == identity({key: value for key, value in v3.items() if key != "input_id"})
    assert "input_id" not in v3["review_request_template"]
    assert "scratch_claim_ledger_path" not in v3["review_request_template"]
    assert v3["review_request_template"]["single_use_enforced"] is False
    assert request["input_id"] == v3["input_id"]
    assert request["template"] == v3["review_request_template"]
    assert request["request_id"] == identity({key: value for key, value in request.items()
                                               if key != "request_id"})
    assert item["scope"]["schema"] == SCOPE_SCHEMA
    assert item["scope"]["unit_name"].endswith(uuid.UUID(v3["nonce"]).hex + ".scope")
    assert item["scope"]["scope_launch_enabled"] is False
    assert v3["source_execution_allowed"] is False and v3["learning_allowed"] is False
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tmp_path.rglob("*") if path.is_file()}
    verify_review_v3_cpu(item["v3_path"], item["review_path"])
    verify_review_v3_cpu(item["v3_path"], item["review_path"])
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert not hasattr(profile_v3_cpu, "claim_review_v3_cpu_fixture")
    assert not hasattr(profile_v3_cpu, "_atomic_claim_write")


@pytest.mark.parametrize("changed", ["v2_path", "negative_path", "scope_path"])
def test_referenced_actual_json_bytes_drift_rejects_even_same_semantics(tmp_path, changed):
    item = _prepared(tmp_path)
    target = item[changed]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(Rejection):
        review_request_v3_cpu(item["v3_path"])


def test_scope_properties_supervisor_bytes_and_negative_witness_are_not_ignored(tmp_path, monkeypatch):
    item = _prepared(tmp_path)
    original = item["scope_path"].read_bytes()
    mutated = copy.deepcopy(item["scope"])
    mutated["runtime_max_seconds"] = 181
    mutated["scope_plan_id"] = identity({key: value for key, value in mutated.items()
                                         if key != "scope_plan_id"})
    write_json(item["scope_path"], mutated)
    with pytest.raises(Rejection):
        review_request_v3_cpu(item["v3_path"])
    item["scope_path"].write_bytes(original)
    item["supervisor"].write_text("# byte drift\n")
    with pytest.raises(Rejection):
        review_request_v3_cpu(item["v3_path"])
    item["supervisor"].write_text("# CPU-only identity fixture; never executed\n")
    from bevy_microduck_tools import negative_lookup
    original_witness = negative_lookup._witness

    def changed_witness(path, errno):
        value = original_witness(path, errno)
        return {**value, "first_missing": "/different/path"}

    monkeypatch.setattr(negative_lookup, "_witness", changed_witness)
    with pytest.raises(Rejection):
        review_request_v3_cpu(item["v3_path"])


def test_new_envelope_and_scope_reject_cross_nonce_output_or_v2_bytes(tmp_path):
    item = _prepared(tmp_path)
    for field, changed in (("nonce", str(uuid.uuid4())),
                           ("output_root", item["v3"]["output_root"] + "/wrong"),
                           ("v2_manifest_sha256", "0" * 64)):
        envelope = copy.deepcopy(item["envelope"])
        envelope[field] = changed
        envelope["envelope_id"] = identity({key: value for key, value in envelope.items()
                                            if key != "envelope_id"})
        write_json(item["negative_path"], envelope)
        with pytest.raises(Rejection):
            make_inputs_v3(item["v2_path"], item["negative_path"], item["scope_path"])
    write_json(item["negative_path"], item["envelope"])
    scope = copy.deepcopy(item["scope"])
    scope["output_root"] += "/wrong"
    scope["scope_plan_id"] = identity({key: value for key, value in scope.items()
                                       if key != "scope_plan_id"})
    write_json(item["scope_path"], scope)
    with pytest.raises(Rejection):
        make_inputs_v3(item["v2_path"], item["negative_path"], item["scope_path"])


def test_old_reviews_and_raw_historical_contract_cannot_impersonate_v3(tmp_path):
    item = _prepared(tmp_path)
    with pytest.raises(Rejection):
        verify_review_v3_cpu(item["v3_path"], item["old_review_path"])
    with pytest.raises(Rejection):
        review_request_v3_cpu(item["v2_path"])
    old_claim_review = copy.deepcopy(item["review"])
    old_claim_review["schema"] = "microduck_v3_cpu_fixture_review_v1"
    write_json(item["review_path"], old_claim_review)
    with pytest.raises(Rejection):
        verify_review_v3_cpu(item["v3_path"], item["review_path"])
    old_contract = seal_contract(item["specification"])
    write_json(item["negative_path"], old_contract)
    with pytest.raises(Rejection, match="envelope"):
        make_inputs_v3(item["v2_path"], item["negative_path"], item["scope_path"])


def test_missing_fields_identity_cycle_and_duplicate_json_keys_reject(tmp_path):
    item = _prepared(tmp_path)
    for change in ("extra_top", "missing_top", "template_cycle", "wrong_nonce",
                   "wrong_output_root", "single_use_claim"):
        value = copy.deepcopy(item["v3"])
        if change == "extra_top":
            value["review_request"] = item["request"]
        elif change == "missing_top":
            del value["scope_plan"]
        elif change == "template_cycle":
            value["review_request_template"]["input_id"] = value["input_id"]
        elif change == "wrong_nonce":
            value["nonce"] = str(uuid.uuid4())
        elif change == "single_use_claim":
            value["review_request_template"]["single_use_enforced"] = True
        else:
            value["output_root"] += "/other"
        value["input_id"] = identity({key: entry for key, entry in value.items()
                                      if key != "input_id"})
        write_json(item["v3_path"], value)
        with pytest.raises(Rejection):
            review_request_v3_cpu(item["v3_path"])
    item["v3_path"].write_text('{"schema":"one","schema":"two"}\n')
    with pytest.raises(Rejection, match="Duplicate JSON field"):
        review_request_v3_cpu(item["v3_path"])


def test_review_scope_expiry_request_or_claim_language_rejects(tmp_path):
    item = _prepared(tmp_path)
    for field, changed in (("schema", "microduck_root_discovery_review_v2"),
                           ("decision", "approve_cpu_fixture_claim"),
                           ("learning_allowed", True),
                           ("source_execution_allowed", True),
                           ("single_use_enforced", True),
                           ("expires_at", 0)):
        review = copy.deepcopy(item["review"])
        review[field] = changed
        write_json(item["review_path"], review)
        with pytest.raises(Rejection):
            verify_review_v3_cpu(item["v3_path"], item["review_path"])
    review = copy.deepcopy(item["review"])
    review["request"]["scope_plan_file_sha256"] = "0" * 64
    write_json(item["review_path"], review)
    with pytest.raises(Rejection):
        verify_review_v3_cpu(item["v3_path"], item["review_path"])
    review = copy.deepcopy(item["review"])
    review["max_claims"] = 1
    write_json(item["review_path"], review)
    with pytest.raises(Rejection):
        verify_review_v3_cpu(item["v3_path"], item["review_path"])


def test_review_byte_drift_during_validation_rejects(tmp_path, monkeypatch):
    item = _prepared(tmp_path)
    original = profile_v3_cpu._validate_cpu_identity_review

    def mutate_after_validating(review, request, sealed_at_unix_ns, **kwargs):
        original(review, request, sealed_at_unix_ns, **kwargs)
        item["review_path"].write_bytes(item["review_path"].read_bytes() + b" ")

    monkeypatch.setattr(profile_v3_cpu, "_validate_cpu_identity_review", mutate_after_validating)
    with pytest.raises(Rejection, match="changed during"):
        verify_review_v3_cpu(item["v3_path"], item["review_path"])


def test_fifo_references_reject_without_blocking(tmp_path):
    item = _prepared(tmp_path)
    fifo = tmp_path / "substituted.json"
    os.mkfifo(fifo)
    script = """\
import sys
from pathlib import Path
from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_v3_cpu import _actual_sha, _read_document
for operation in (_read_document, _actual_sha):
    try:
        operation(Path(sys.argv[1]), 'FIFO reference')
    except Rejection:
        continue
    raise AssertionError('FIFO reference was accepted')
"""
    completed = subprocess.run([sys.executable, "-c", script, str(fifo)],
                               capture_output=True, text=True, timeout=3, check=False)
    assert completed.returncode == 0, completed.stderr
    item["supervisor"].unlink()
    os.mkfifo(item["supervisor"])
    script = """\
import sys
from pathlib import Path
from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_v3_cpu import review_request_v3_cpu
try:
    review_request_v3_cpu(Path(sys.argv[1]))
except Rejection:
    sys.exit(0)
raise AssertionError('FIFO supervisor was accepted')
"""
    completed = subprocess.run([sys.executable, "-c", script, str(item["v3_path"])],
                               capture_output=True, text=True, timeout=3, check=False)
    assert completed.returncode == 0, completed.stderr


def test_oversized_reference_rejects_before_read(tmp_path):
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(profile_v3_cpu.MAX_REFERENCE_BYTES + 1)
    with pytest.raises(Rejection, match="bounded regular file"):
        profile_v3_cpu._read_document(oversized, "oversized reference")
    with pytest.raises(Rejection, match="bounded regular file"):
        profile_v3_cpu._actual_sha(oversized, "oversized reference")


def test_public_discover_remains_closed_before_ledger_or_source(tmp_path):
    ledger = tmp_path / "unused_real_budget_path.json"
    with pytest.raises(Rejection, match="requires a new scope-bound"):
        discover_source(tmp_path / "missing_v3.json", tmp_path / "missing_review.json",
                        ledger_path=ledger, output_root=tmp_path / "missing_output")
    assert not ledger.exists() and not (tmp_path / "missing_output").exists()
