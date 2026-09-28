"""Scratch CPU fake-scope consumption; never source, GPU, or real budget."""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

import bevy_microduck_tools
from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools import profile_v3_fixture_broker as broker
from bevy_microduck_tools.profile_v3_cpu import _read_document, review_request_v3_cpu
from bevy_microduck_tools.serialization import identity, write_json
from test_profile_v3_cpu import _prepared


def _fixture(tmp_path: Path, *, state_root: Path | None = None, cap: int = 360) -> dict:
    item = _prepared(tmp_path / "inputs")
    state_root = tmp_path / "broker" if state_root is None else state_root
    if not state_root.exists():
        store_id = broker.initialize_cpu_fixture_store(state_root, budget_cap_gpu_seconds=cap)
    else:
        store_id = broker.inspect_cpu_fixture_store(state_root)["store_id"]
    scope_path = tmp_path / "fake_scope.json"
    scope = broker.make_fake_scope_v3_cpu(item["v3_path"], fake_leaf_uuid=str(uuid.uuid4()))
    write_json(scope_path, scope)
    _, scope_sha = _read_document(scope_path, "CPU fixture scope")
    review_path = tmp_path / "claim_review.json"
    review = {"schema": broker.REVIEW_SCHEMA, "authorization_id": str(uuid.uuid4()),
              "reviewer_role": "root_gpt", "decision": "approve_cpu_fixture_claim_only",
              "reviewed_at_unix_ns": time.time_ns(),
              "expires_at_unix_ns": time.time_ns() + 60_000_000_000,
              "request": review_request_v3_cpu(item["v3_path"]),
              "fake_scope_path": str(scope_path.resolve()), "fake_scope_sha256": scope_sha,
              "fake_scope_id": scope["scope_id"], "state_root": str(state_root.resolve()),
              "store_id": store_id,
              "max_runs": 1, "max_wall_seconds": 180, "gpus": 1,
              "reserved_gpu_seconds": 180, "single_use_scope": "cooperative_cpu_fixture_only",
              "source_execution_allowed": False, "learning_allowed": False}
    write_json(review_path, review)
    return {**item, "state_root": state_root, "scope_path": scope_path,
            "scope": scope, "review_path": review_path, "review": review}


def _claim(item: dict) -> dict:
    return broker.claim_cpu_fixture(item["v3_path"], item["scope_path"],
                                    item["review_path"], item["state_root"])


def _subprocess_environment() -> dict:
    environment = dict(os.environ)
    package_parent = str(Path(bevy_microduck_tools.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (package_parent, environment.get("PYTHONPATH", "")) if part)
    return environment


def test_one_durable_consumption_records_only_cpu_fixture_claim(tmp_path):
    item = _fixture(tmp_path)
    receipt = _claim(item)
    assert receipt["cpu_fixture_claim_recorded"] is True
    assert receipt["source_execution_allowed"] is False
    assert receipt["learning_allowed"] is False
    assert "barrier_token" not in receipt and "release_allowed" not in receipt
    store = broker.inspect_cpu_fixture_store(item["state_root"])
    assert store["charged_gpu_seconds"] == 180 and len(store["claims"]) == 1
    assert store["claims"][0]["claim_id"] == receipt["claim_id"]
    with pytest.raises(Rejection):
        _claim(item)
    assert broker.inspect_cpu_fixture_store(item["state_root"]) == store
    assert not (item["v2_path"].parent / "ledger.json").exists()


def test_two_processes_contend_one_review_and_restart_cannot_reclaim(tmp_path):
    item = _fixture(tmp_path)
    script = """\
import sys
from pathlib import Path
from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_v3_fixture_broker import claim_cpu_fixture
try:
    claim_cpu_fixture(*(Path(value) for value in sys.argv[1:]))
except Rejection:
    print('REJECTED')
else:
    print('CONSUMED')
"""
    paths = [str(item[name]) for name in ("v3_path", "scope_path", "review_path", "state_root")]
    command = [sys.executable, "-c", script, *paths]
    children = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, env=_subprocess_environment()) for _ in range(2)]
    outcomes = []
    for child in children:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stderr
        outcomes.append(stdout.strip())
    assert sorted(outcomes) == ["CONSUMED", "REJECTED"]
    restarted = subprocess.run(command, capture_output=True, text=True, timeout=10,
                               env=_subprocess_environment())
    assert restarted.returncode == 0 and restarted.stdout.strip() == "REJECTED"
    store = broker.inspect_cpu_fixture_store(item["state_root"])
    assert len(store["claims"]) == 1 and store["charged_gpu_seconds"] == 180


def test_cross_expiry_durable_row_is_burned_without_barrier_release(tmp_path, monkeypatch):
    item = _fixture(tmp_path)
    review = copy.deepcopy(item["review"])
    review["expires_at_unix_ns"] = time.time_ns() + 3_000_000_000
    write_json(item["review_path"], review)
    original = broker._commit

    def commit_then_cross_expiry(connection):
        original(connection)
        delay = max(0.0, (review["expires_at_unix_ns"] - time.time_ns()) / 1e9 + 0.05)
        time.sleep(delay)

    monkeypatch.setattr(broker, "_commit", commit_then_cross_expiry)
    with pytest.raises(Rejection, match="expired"):
        _claim(item)
    store = broker.inspect_cpu_fixture_store(item["state_root"])
    assert len(store["claims"]) == 1 and store["charged_gpu_seconds"] == 180
    with pytest.raises(Rejection):
        _claim(item)


def test_new_review_uuid_cannot_reuse_input_or_nonce(tmp_path):
    item = _fixture(tmp_path)
    _claim(item)
    review = copy.deepcopy(item["review"])
    review["authorization_id"] = str(uuid.uuid4())
    write_json(item["review_path"], review)
    with pytest.raises(Rejection):
        _claim(item)
    assert len(broker.inspect_cpu_fixture_store(item["state_root"])["claims"]) == 1


def test_embedded_request_json_types_must_match_exactly(tmp_path):
    item = _fixture(tmp_path)
    for field, changed in (("gpus", True), ("max_wall_seconds", 180.0)):
        review = copy.deepcopy(item["review"])
        review["request"]["template"][field] = changed
        write_json(item["review_path"], review)
        with pytest.raises(Rejection):
            _claim(item)
    assert broker.inspect_cpu_fixture_store(item["state_root"])["claims"] == []


def test_recreated_store_at_same_path_has_a_new_identity(tmp_path):
    item = _fixture(tmp_path)
    original_store_id = item["review"]["store_id"]
    old_store = tmp_path / "old-store"
    item["state_root"].rename(old_store)
    new_store_id = broker.initialize_cpu_fixture_store(item["state_root"], budget_cap_gpu_seconds=360)
    assert new_store_id != original_store_id
    with pytest.raises(Rejection, match="Unknown CPU fixture budget store"):
        _claim(item)
    assert broker.inspect_cpu_fixture_store(item["state_root"])["claims"] == []


@pytest.mark.parametrize("changed", ["review", "manifest", "scope"])
def test_bytes_changed_between_preflight_and_transaction_do_not_consume(tmp_path, monkeypatch, changed):
    item = _fixture(tmp_path)
    original = broker._connect
    path = {"review": item["review_path"], "manifest": item["v3_path"],
            "scope": item["scope_path"]}[changed]
    injected = False

    def drift(directory_fd, *, create):
        nonlocal injected
        if not create and not injected:
            injected = True
            path.write_bytes(path.read_bytes() + b" ")
        return original(directory_fd, create=create)

    monkeypatch.setattr(broker, "_connect", drift)
    with pytest.raises(Rejection):
        _claim(item)
    assert broker.inspect_cpu_fixture_store(item["state_root"])["claims"] == []


def test_observed_parent_pivot_refuses_and_never_writes_to_new_path(tmp_path, monkeypatch):
    item = _fixture(tmp_path / "first")
    outside = tmp_path / "outside"
    broker.initialize_cpu_fixture_store(outside, budget_cap_gpu_seconds=360)
    moved = tmp_path / "moved-original"
    original = broker._connect
    injected = False

    def pivot(directory_fd, *, create):
        nonlocal injected
        if not create and not injected:
            injected = True
            item["state_root"].rename(moved)
            item["state_root"].symlink_to(outside, target_is_directory=True)
        return original(directory_fd, create=create)

    monkeypatch.setattr(broker, "_connect", pivot)
    with pytest.raises(Rejection, match="pathname changed"):
        _claim(item)
    assert broker.inspect_cpu_fixture_store(moved)["claims"] == []
    assert broker.inspect_cpu_fixture_store(outside)["claims"] == []


def test_observed_database_inode_replacement_refuses(tmp_path, monkeypatch):
    item = _fixture(tmp_path)
    original = broker._connect
    injected = False

    def replace_db(directory_fd, *, create):
        nonlocal injected
        if not create and not injected:
            injected = True
            database = item["state_root"] / broker.DATABASE
            database.rename(item["state_root"] / "old-admission.sqlite3")
            shutil.copyfile(item["state_root"] / "old-admission.sqlite3", database)
        return original(directory_fd, create=create)

    monkeypatch.setattr(broker, "_connect", replace_db)
    with pytest.raises(Rejection, match="database pathname changed"):
        _claim(item)
    assert broker.inspect_cpu_fixture_store(item["state_root"])["claims"] == []


def test_old_review_fake_scope_drift_and_budget_refuse(tmp_path):
    item = _fixture(tmp_path / "first", cap=180)
    with pytest.raises(Rejection):
        broker.claim_cpu_fixture(item["v3_path"], item["scope_path"],
                                 item["old_review_path"], item["state_root"])
    scope = copy.deepcopy(item["scope"])
    scope["unit_name"] = "wrong.scope"
    scope["scope_id"] = identity({key: value for key, value in scope.items() if key != "scope_id"})
    write_json(item["scope_path"], scope)
    with pytest.raises(Rejection):
        _claim(item)
    write_json(item["scope_path"], item["scope"])
    first = _claim(item)
    assert first["source_execution_allowed"] is False
    second = _fixture(tmp_path / "second", state_root=item["state_root"])
    with pytest.raises(Rejection, match="budget exhausted"):
        _claim(second)
    assert broker.inspect_cpu_fixture_store(item["state_root"])["charged_gpu_seconds"] == 180
