"""Synthetic deadline/commit tests. No source worker, GPU or real ledger."""
from __future__ import annotations

import dataclasses
import json
import os
import stat
import threading
import time
from types import SimpleNamespace

import pytest

from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_discovery import (
    _finish_discovery_budget, preflight_parent, reserve_discovery)
from bevy_microduck_tools import workflow, profile_discovery
from bevy_microduck_tools.workflow import TrainingBudget
from test_profile_v10 import _fixture


COMPLETED = "completed_zero_update_discovery"


def _state(path):
    return json.loads(path.read_text())["runs"][0]["state"]


def _reserved_discovery(tmp_path):
    _, manifest_path, _, review_path, ledger_path = _fixture(tmp_path)
    token = dataclasses.replace(preflight_parent(manifest_path, require_runtime=False).token,
                                runtime_verified=True)
    binding = reserve_discovery(manifest_path, review_path, ledger_path,
                                parent_token=token, wall_seconds=.2)
    return ledger_path, binding["run_id"]


@pytest.mark.parametrize("pause_stage", ["before_commit", "before_replace", "after_replace_before_dir_fsync"])
@pytest.mark.parametrize("requested_status", [COMPLETED, "captured_provisional_dependencies", "failed"])
def test_delayed_atomic_commit_across_deadline_never_exposes_completed(tmp_path, monkeypatch,
                                                                       pause_stage, requested_status):
    path, run_id = _reserved_discovery(tmp_path)
    original = TrainingBudget._atomic_commit
    entered = threading.Event()
    release = threading.Event()
    observed = []
    failures = []
    result = []
    commit_attempts = []
    paused_at = []
    # Only the closure's monotonic clock is synthetic. Event timeouts and real
    # replace/fsync still run normally, irrespective of host startup latency.
    clock = [100.0]
    monkeypatch.setattr(profile_discovery, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=time.time))

    def pause_before_durable_step(message):
        paused_at.append(clock[0])
        entered.set()
        if not release.wait(2):
            raise TimeoutError(message)

    def checked_commit(self, data):
        # The exact prospective durable bytes are already fail-closed before
        # we force this commit across the deadline.
        assert data["runs"][0]["state"] == "budget_exhausted"
        commit_attempts.append(data["runs"][0]["state"])
        if pause_stage == "before_commit":
            pause_before_durable_step("synthetic commit release was not delivered")
        original(self, data)

    monkeypatch.setattr(TrainingBudget, "_atomic_commit", checked_commit)
    if pause_stage == "before_replace":
        original_replace = workflow.os.replace

        def delayed_replace(source, destination):
            pause_before_durable_step("synthetic replace release was not delivered")
            return original_replace(source, destination)

        monkeypatch.setattr(workflow.os, "replace", delayed_replace)
    elif pause_stage == "after_replace_before_dir_fsync":
        original_fsync = workflow.os.fsync

        def delayed_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                pause_before_durable_step("synthetic directory fsync release was not delivered")
            return original_fsync(fd)

        monkeypatch.setattr(workflow.os, "fsync", delayed_fsync)
    started = clock[0]
    deadline = started + .25

    def close():
        try:
            result.append(_finish_discovery_budget(path, run_id, started=started,
                          deadline=deadline, status=requested_status))
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(2)
        observed.append(_state(path))
        # Advance across the deadline while the selected durable step is still
        # blocked. Release occurs only after both before/after observations.
        clock[0] = deadline + .025
        observed.append(_state(path))
    finally:
        release.set()
        worker.join(timeout=2)
    assert not worker.is_alive() and not failures
    assert commit_attempts == ["budget_exhausted"]
    assert len(paused_at) == 1 and paused_at[0] < deadline
    observed.append(_state(path))
    assert set(observed) <= {"reserved", "budget_exhausted"}
    assert observed[-1] == "budget_exhausted"
    assert ("reserved" in observed) == (pause_stage != "after_replace_before_dir_fsync")
    assert result[0][0] == "budget_exhausted"
    assert result[0][1] >= deadline - started
    assert _state(path) != COMPLETED
    with TrainingBudget(path).locked(commit=False) as ledger:
        row = ledger["runs"][0]
        assert row["deadline_correction"] == ("v13_completion_commit_unproven" if
            requested_status != "failed" else "v13_noncompletion_commit_deadline_unproven")
        assert row["requested_discovery_status"] == requested_status
        assert row["budget_accounting_basis"].startswith("conservative_reserved_charge")
        assert row["reserved_wall_seconds_charged"] == .2
        assert row["charged_gpu_seconds"] >= row["reserved_gpu_seconds"]
        assert row["measured_gpu_seconds"] == row["closure_elapsed_lower_bound_seconds"]


@pytest.mark.parametrize("error_stage", ["before_replace", "after_replace_before_dir_fsync"])
def test_commit_uncertainty_never_leaves_completed(tmp_path, monkeypatch, error_stage):
    path, run_id = _reserved_discovery(tmp_path)
    if error_stage == "before_replace":
        def fail_replace(source, destination):
            raise OSError("synthetic durable-commit uncertainty")
        monkeypatch.setattr(workflow.os, "replace", fail_replace)
    else:
        original_fsync = workflow.os.fsync

        def fail_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("synthetic durable-commit uncertainty")
            return original_fsync(fd)

        monkeypatch.setattr(workflow.os, "fsync", fail_directory_fsync)
    started = time.monotonic()
    with pytest.raises(OSError, match="synthetic durable-commit uncertainty"):
        _finish_discovery_budget(path, run_id, started=started, deadline=started + .2,
                                 status=COMPLETED)
    assert _state(path) == ("budget_exhausted" if error_stage == "after_replace_before_dir_fsync" else "reserved")
    assert _state(path) != COMPLETED


def test_legacy_finish_cannot_close_discovery_but_other_budget_flow_still_works(tmp_path):
    fixture_root = tmp_path / "discovery"
    fixture_root.mkdir()
    manifest, manifest_path, _, review_path, discovery_ledger = _fixture(fixture_root)
    token = dataclasses.replace(preflight_parent(manifest_path, require_runtime=False).token,
                                runtime_verified=True)
    binding = reserve_discovery(manifest_path, review_path, discovery_ledger,
                                parent_token=token, wall_seconds=.2)
    with pytest.raises(Rejection, match="single-transaction"):
        TrainingBudget(discovery_ledger).finish(binding["run_id"], .01, COMPLETED)
    assert _state(discovery_ledger) == "reserved"
    started = time.monotonic()
    _finish_discovery_budget(discovery_ledger, binding["run_id"], started=started,
                             deadline=started + .2, status=COMPLETED)
    assert _state(discovery_ledger) == "budget_exhausted"
    assert manifest["input_id"] == binding["input_id"]

    ordinary = tmp_path / "ordinary.json"
    normal_budget = TrainingBudget(ordinary)
    ordinary_id = normal_budget.reserve("standing", "ordinary_fixture", 1000001, .2)
    normal_budget.finish(ordinary_id, .01, "completed_structural_smoke")
    assert _state(ordinary) == "completed_structural_smoke"
    with normal_budget.locked(commit=False) as ledger:
        assert ledger["runs"][0]["measured_gpu_seconds"] == .01


def test_discovery_closure_rejects_ordinary_budget_run(tmp_path):
    path = tmp_path / "ordinary.json"
    run_id = TrainingBudget(path).reserve("standing", "ordinary_fixture", 1000001, .2)
    started = time.monotonic()
    with pytest.raises(Rejection, match="exact open consumed run"):
        _finish_discovery_budget(path, run_id, started=started, deadline=started + .2,
                                 status=COMPLETED)
    assert _state(path) == "reserved"
