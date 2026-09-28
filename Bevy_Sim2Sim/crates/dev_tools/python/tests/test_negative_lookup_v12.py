"""v12 CPU-only negative-lookup and scratch-budget fixtures; no science worker."""
from __future__ import annotations

import copy
import concurrent.futures
import dataclasses
import json
import os
import time
from pathlib import Path

import pytest

from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.negative_lookup import (
    SPEC_SCHEMA, parse_failed_call, seal_contract, verify_contract)
from bevy_microduck_tools.profile_observer import parse_trace
from bevy_microduck_tools.profile_discovery import (
    _finish_discovery_budget, discover_source, preflight_parent, reserve_discovery)
from bevy_microduck_tools.workflow import TrainingBudget
from test_profile_v10 import _fixture


def _rule(path: Path | str, *, syscall="openat", qualifier="O_RDONLY",
          errno="ENOENT", phase="before_first_science_phase", max_count=1, rule_id="one") -> dict:
    return {"rule_id": rule_id, "path": str(path), "syscall": syscall,
            "errno": errno, "phase": phase, "dirfd": "AT_FDCWD",
            "qualifier": qualifier, "max_count": max_count}


def _seal(*rows: dict) -> dict:
    return seal_contract({"schema": SPEC_SCHEMA, "rules": list(rows)})


def _trace(tmp_path: Path, *calls: str) -> Path:
    folder = tmp_path / "trace"
    folder.mkdir(exist_ok=True)
    for index, call in enumerate(calls, 123):
        (folder / f"trace.{index}").write_text(
            f"1.000000 {call}\n1.900000 +++ exited with 0 +++\n")
    return folder


def _open(path: str, flag="O_RDONLY", errno="ENOENT") -> str:
    return f'openat(AT_FDCWD<{os.getcwd()}>, {json.dumps(path)}, {flag}) = -1 {errno} (failed)'


def test_four_presealed_readonly_failed_shapes_and_raw_receipts(tmp_path):
    trace = _trace(tmp_path)
    missing = tmp_path / "missing.so"
    newline = tmp_path / "python3.12\n._pth"
    plain = tmp_path / "plain"
    plain.write_bytes(b"plain")
    rules = [
        _rule(missing, syscall="openat", qualifier="O_RDONLY|O_CLOEXEC", rule_id="open"),
        _rule(newline, syscall="newfstatat", qualifier="0", rule_id="stat"),
        _rule(missing, syscall="faccessat", qualifier="R_OK", rule_id="access"),
        _rule(plain, syscall="readlinkat", qualifier="4096", errno="EINVAL", rule_id="link"),
    ]
    contract = _seal(*rules)
    cwd = os.getcwd()
    calls = [
        _open(str(missing), "O_RDONLY|O_CLOEXEC"),
        f'newfstatat(AT_FDCWD<{cwd}>, {json.dumps(str(newline))}, 0x1234, 0) = -1 ENOENT (failed)',
        f'faccessat(AT_FDCWD<{cwd}>, {json.dumps(str(missing))}, R_OK) = -1 ENOENT (failed)',
        f'readlinkat(AT_FDCWD<{cwd}>, {json.dumps(str(plain))}, 0x1234, 4096) = -1 EINVAL (failed)',
    ]
    for index, call in enumerate(calls, 123):
        (trace / f"trace.{index}").write_text(
            f"1.000000 {call}\n1.900000 +++ exited with 0 +++\n")
    result = parse_trace(trace, command_exitcode=0, negative_contract=contract)
    assert result["trace_complete"] is True, result["unresolved"]
    assert len(result["failed_candidates"]) == 4
    assert {event["rule_id"] for event in result["negative_lookup_events"]} == {
        "open", "stat", "access", "link"}
    assert all(event["trace_sha256"] and event["line_number"] == 1 and
               event["contract_id"] == contract["contract_id"] and
               event["raw_call"] in calls for event in result["negative_lookup_events"])


def test_original_dotdot_resolves_after_symlink_not_lexically(tmp_path):
    trace = _trace(tmp_path)
    real = tmp_path / "real"
    deep = real / "deep"
    deep.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to("real/deep", target_is_directory=True)
    raw = str(tmp_path / "link") + "/../missing"
    lexical = tmp_path / "missing"
    lexical.write_bytes(b"would fool lexical normpath")
    contract = _seal(_rule(raw))
    witness = contract["rules"][0]["witness"]
    assert witness["first_missing"] == str(real / "missing")
    assert witness["first_missing"] != str(lexical)
    (trace / "trace.123").write_text(
        f"1.000000 {_open(raw)}\n1.900000 +++ exited with 0 +++\n")
    parsed = parse_trace(trace, command_exitcode=0, negative_contract=contract)
    assert parsed["trace_complete"] is True, parsed["unresolved"]
    assert parsed["negative_lookup_events"][0]["path"] == raw


def test_readlink_einval_follows_ancestor_link_but_not_final_link(tmp_path):
    _trace(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    plain = target / "plain"
    plain.write_bytes(b"x")
    ancestor = tmp_path / "alias"
    ancestor.symlink_to("target", target_is_directory=True)
    contract = _seal(_rule(ancestor / "plain", syscall="readlinkat",
                                qualifier="1023", errno="EINVAL"))
    assert contract["rules"][0]["witness"]["resolved_path"] == str(plain)
    final_link = target / "final-link"
    final_link.symlink_to("plain")
    with pytest.raises(Rejection, match="actually a symlink"):
        _seal(_rule(final_link, syscall="readlinkat", qualifier="1023", errno="EINVAL"))


@pytest.mark.parametrize("changed_call", [
    lambda path: _open(path, "O_RDWR"),
    lambda path: _open(path, "O_RDONLY|O_CREAT"),
    lambda path: _open(path, errno="EACCES"),
    lambda path: f'openat(3</tmp>, {json.dumps(path)}, O_RDONLY) = -1 ENOENT (failed)',
    lambda path: 'openat(AT_FDCWD, "relative.so", O_RDONLY) = -1 ENOENT (failed)',
    lambda path: f'openat(AT_FDCWD, {json.dumps(path + "extra")}, O_RDONLY) = -1 ENOENT (failed)',
])
def test_unapproved_args_errno_dirfd_relative_and_extra_path_are_diagnostic(tmp_path, changed_call):
    trace = _trace(tmp_path)
    missing = tmp_path / "missing.so"
    contract = _seal(_rule(missing))
    (trace / "trace.123").write_text(
        f"1.000000 {changed_call(str(missing))}\n1.900000 +++ exited with 0 +++\n")
    result = parse_trace(trace, command_exitcode=0, negative_contract=contract)
    assert result["trace_complete"] is False
    assert result["unresolved"]
    assert result["negative_lookup_events"] == []


def test_extra_occurrence_or_wrong_phase_is_diagnostic(tmp_path):
    trace = _trace(tmp_path)
    missing = tmp_path / "missing.so"
    contract = _seal(_rule(missing))
    call = _open(str(missing))
    (trace / "trace.123").write_text(
        f"1.000000 {call}\n1.100000 {call}\n1.900000 +++ exited with 0 +++\n")
    exceeded = parse_trace(trace, command_exitcode=0, negative_contract=contract)
    assert exceeded["trace_complete"] is False
    assert len(exceeded["negative_lookup_events"]) == 1
    assert any("count" in row["reason"] for row in exceeded["unresolved"])
    (trace / "trace.123").write_text(f"1.000000 {call}\n1.900000 +++ exited with 0 +++\n")
    wrong_phase = parse_trace(trace, command_exitcode=0, negative_contract=contract,
                              phases=[{"wall_seconds": .5, "phase": "source_compile"}])
    assert wrong_phase["trace_complete"] is False
    assert any("no unique" in row["reason"] for row in wrong_phase["unresolved"])


def test_invalid_readlink_buffer_or_unknown_stat_flag_is_rejected(tmp_path):
    plain = tmp_path / "plain"
    plain.write_bytes(b"plain")
    missing = tmp_path / "missing"
    for call in (
        f'readlinkat(AT_FDCWD, {json.dumps(str(plain))}, 0x1234, 0) = -1 EINVAL (failed)',
        f'readlinkat(AT_FDCWD, {json.dumps(str(plain))}, NULL, 4096) = -1 EINVAL (failed)',
        f'newfstatat(AT_FDCWD, {json.dumps(str(missing))}, 0x1234, AT_EMPTY_PATH) = -1 ENOENT (failed)',
    ):
        with pytest.raises(Rejection):
            parse_failed_call(call)


def test_witness_file_or_ancestor_change_and_contract_tamper_reject(tmp_path):
    trace = _trace(tmp_path)
    absent = tmp_path / "nested" / "missing"
    (tmp_path / "nested").mkdir()
    contract = _seal(_rule(absent))
    malformed = copy.deepcopy(contract)
    malformed["rules"][0]["witness"]["first_missing"] = "/fake"
    with pytest.raises(Rejection, match="identity"):
        verify_contract(malformed)
    absent.write_bytes(b"appeared after sealing")
    with pytest.raises(Rejection, match="witness|pathname"):
        verify_contract(contract)
    absent.unlink()
    new_entry = tmp_path / "nested" / "other"
    new_entry.write_bytes(b"changed directory membership")
    with pytest.raises(Rejection, match="witness"):
        verify_contract(contract)


def test_no_unbounded_glob_private_cache_or_truncated_path(tmp_path):
    for path in ("/tmp/*", "/proc/self/exe", "relative", "/tmp/../dev/null"):
        with pytest.raises(Rejection):
            _seal(_rule(path))
    with pytest.raises(Rejection):
        parse_failed_call('openat(AT_FDCWD, "/tmp/missing"..., O_RDONLY) = -1 ENOENT (failed)')


def test_v12_public_science_entry_refuses_before_ledger_or_process(tmp_path):
    ledger = tmp_path / "real_science_must_not_exist.json"
    output = tmp_path / "science_must_not_exist"
    with pytest.raises(Rejection, match="scope-bound v3"):
        discover_source(tmp_path / "missing_manifest", tmp_path / "missing_review",
                        ledger_path=ledger, output_root=output)
    assert not ledger.exists() and not output.exists()


def test_scratch_review_two_reservations_compete_and_timeout_never_completes(tmp_path):
    manifest, manifest_path, _, review_path, ledger_path = _fixture(tmp_path)
    token = dataclasses.replace(preflight_parent(manifest_path, require_runtime=False).token,
                                runtime_verified=True)

    def reserve():
        try:
            return reserve_discovery(manifest_path, review_path, ledger_path,
                                     parent_token=token, wall_seconds=1)
        except Rejection as error:
            return str(error)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: reserve(), range(2)))
    granted = [item for item in outcomes if isinstance(item, dict)]
    refused = [item for item in outcomes if isinstance(item, str)]
    assert len(granted) == len(refused) == 1
    assert "already consumed" in refused[0]
    started = time.monotonic() - 2
    status, measured = _finish_discovery_budget(ledger_path, granted[0]["run_id"],
        started=started, deadline=started + 1, status="completed_zero_update_discovery")
    assert status == "budget_exhausted" and measured > 1
    with TrainingBudget(ledger_path).locked(commit=False) as ledger:
        rows = ledger["runs"]
        assert len(rows) == 1 and rows[0]["state"] == "budget_exhausted"
        assert rows[0]["discovery_authorization"]["input_id"] == manifest["input_id"]
    assert "already consumed" in reserve()
