"""v11 fail-closed CPU fixtures; no source environment, MuJoCo, CUDA or PPO."""
from __future__ import annotations

import os
import dataclasses
import subprocess
import time
from pathlib import Path

import pytest

from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_discovery import (
    REVIEW_SCHEMA, DiscoveryRequest, _finish_discovery_budget, authorize_discovery,
    preflight_parent, reserve_discovery)
from bevy_microduck_tools.profile_identity import (
    SELECTED_ENV, _execution_env_hashes, verify_declared_inputs)
from bevy_microduck_tools.profile_observer import observe_command, parse_trace
from bevy_microduck_tools.workflow import TrainingBudget
from test_profile_v10 import _fixture


def _stat_result(path: Path) -> str:
    value = path.stat()
    return (f"{{st_dev=makedev(0x{os.major(value.st_dev):x}, 0x{os.minor(value.st_dev):x}), "
            f"st_ino={value.st_ino}, st_mode=S_IFREG|0644, st_size={value.st_size}}}")


def _synthetic_trace(tmp_path: Path, lines: list[str]) -> Path:
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "trace.123").write_text("\n".join(lines + ["1.900000 +++ exited with 0 +++"]) + "\n")
    return trace


def test_openat2_fstat_stat_and_readlink_have_explained_actual_use(tmp_path):
    data = tmp_path / "input.bin"
    data.write_bytes(b"source bytes")
    link = tmp_path / "input-link"
    link.symlink_to(data.name)
    stat = _stat_result(data)
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 openat2(AT_FDCWD<{tmp_path}>, "{data.name}", '
        f'{{flags=O_RDONLY|O_CLOEXEC}}, 24) = 3<{data}>',
        f"1.100000 fstat(3<{data}>, {stat}) = 0",
        f'1.200000 newfstatat(AT_FDCWD<{tmp_path}>, "{data.name}", {stat}, 0) = 0',
        f'1.300000 readlinkat(AT_FDCWD<{tmp_path}>, "{link.name}", "{data.name}", 4096) = {len(data.name)}',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is True
    assert not result["failed_candidates"] and not result["unresolved"]
    assert result["syscall_inventory"]["openat2"] == 1
    assert len(result["successful_metadata"]) == 3
    assert {row["path"] for row in result["posthoc_actual_use"]} == {str(data), str(link)}
    used = next(row for row in result["posthoc_actual_use"] if row["path"] == str(data))
    assert used["actual_opened_devices"] and used["observed_metadata_devices"]


def test_actual_cpu_openat2_stat_readlink_and_failed_loader_candidates(tmp_path):
    data = tmp_path / "payload"
    data.write_bytes(b"actual CPU bytes")
    link = tmp_path / "payload-link"
    link.symlink_to(data.name)
    source = tmp_path / "probe.c"
    source.write_text(
        "#define _GNU_SOURCE\n"
        "#include <fcntl.h>\n#include <linux/openat2.h>\n#include <sys/stat.h>\n"
        "#include <sys/syscall.h>\n#include <unistd.h>\n"
        "int main(int argc,char**argv){if(argc!=3)return 2;"
        "struct open_how h={.flags=O_RDONLY};"
        "int fd=syscall(SYS_openat2,AT_FDCWD,argv[1],&h,sizeof(h));if(fd<0)return 3;"
        "struct stat st;if(fstat(fd,&st))return 4;close(fd);"
        "char buf[256];if(readlink(argv[2],buf,sizeof(buf))<0)return 5;"
        "if(stat(argv[1],&st))return 6;if(access(argv[1],R_OK))return 7;return 0;}\n")
    binary = tmp_path / "probe"
    subprocess.run(["cc", str(source), "-o", str(binary)], check=True)
    result = observe_command([str(binary), str(data), str(link)], tmp_path / "observed",
                             wall_seconds=8)
    assert result["status"] == "completed"
    assert result["trace_complete"] is False  # Loader ENOENT has no v11 negative contract.
    assert any(row["syscall"] == "openat2" and row["opened_path"] == str(data)
               for row in result["open_selections"])
    metadata_calls = {row["syscall"] for row in result["successful_metadata"]}
    assert {"fstat", "newfstatat"} <= metadata_calls
    assert "readlink" in metadata_calls or "readlinkat" in metadata_calls
    assert any(row["errno"] == "ENOENT" for row in result["failed_candidates"])
    assert any(row["path"] == str(data) for row in result["posthoc_actual_use"])


def test_failed_candidate_and_unknown_syscall_are_explicit_diagnostic(tmp_path):
    data = tmp_path / "input.bin"
    data.write_bytes(b"bytes")
    missing = tmp_path / "absent-choice.so"
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 openat(AT_FDCWD<{tmp_path}>, "{missing.name}", O_RDONLY) = -1 ENOENT (No such file or directory)',
        f'1.100000 frobulate(AT_FDCWD<{tmp_path}>, "{data.name}") = 0',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert result["failed_candidates"] == [{
        "pid": 123, "phase": "before_first_science_phase", "syscall": "openat",
        "path": str(missing), "errno": "ENOENT", "wall_seconds": 1.0}]
    assert any(row.get("reason") == "unknown traced syscall" and row.get("syscall") == "frobulate"
               for row in result["unresolved"])


def test_successful_access_and_open_without_observed_inode_fail_closed(tmp_path):
    data = tmp_path / "input.bin"
    data.write_bytes(b"bytes")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 faccessat(AT_FDCWD<{tmp_path}>, "{data.name}", R_OK) = 0',
        f'1.100000 openat2(AT_FDCWD<{tmp_path}>, "{data.name}", {{flags=O_RDONLY}}, 24) = 3<{data}>',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    reasons = [row["reason"] for row in result["unresolved"]]
    assert any("metadata lookup lacks observed inode" in reason for reason in reasons)
    assert any("opened device/inode absent" in reason for reason in reasons)


@pytest.mark.parametrize("mapping", [
    "mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, 3, 0) = 0x1234",
    "mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, 3</tmp/payload>, 0) = ?",
])
def test_unresolved_file_mapping_never_completes(tmp_path, mapping):
    trace = _synthetic_trace(tmp_path, ["1.000000 " + mapping])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any("file mapping lacks resolved FD or successful result" in row["reason"]
               for row in result["unresolved"])


def test_missing_child_trace_never_completes(tmp_path):
    trace = _synthetic_trace(tmp_path, ["1.000000 clone(child_stack=NULL, flags=SIGCHLD) = 456"])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any(row.get("pid") == 456 and "child trace" in row["reason"]
               for row in result["unresolved"])


def test_proc_fd_alias_to_regular_file_never_counts_as_platform_only(tmp_path):
    data = tmp_path / "payload"
    data.write_bytes(b"payload")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 openat(AT_FDCWD, "/proc/self/fd/9", O_RDONLY) = 4<{data}>',
        f'1.100000 newfstatat(AT_FDCWD, "/proc/self/fd/9", {_stat_result(data)}, 0) = 0',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    reasons = [row.get("reason", "") for row in result["unresolved"]]
    assert any("platform fd alias selected ordinary file" in reason for reason in reasons)
    assert any("platform metadata or fd alias" in reason for reason in reasons)


def test_proc_exe_metadata_cannot_be_silently_classified_as_platform(tmp_path):
    data = tmp_path / "payload"
    data.write_bytes(b"payload")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 newfstatat(AT_FDCWD, "/proc/self/exe", {_stat_result(data)}, 0) = 0',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any("platform metadata" in row["reason"] for row in result["unresolved"])


def test_each_read_open_fd_needs_its_own_fstat(tmp_path):
    data = tmp_path / "payload"
    data.write_bytes(b"payload")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 openat(AT_FDCWD<{tmp_path}>, "payload", O_RDONLY) = 3<{data}>',
        f'1.100000 fstat(3<{data}>, {_stat_result(data)}) = 0',
        f'1.200000 openat(AT_FDCWD<{tmp_path}>, "payload", O_RDONLY) = 4<{data}>',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any(row.get("fd") == 4 and "own observed" in row["reason"]
               for row in result["unresolved"])


def test_open_flags_do_not_read_pathname_text(tmp_path):
    data = tmp_path / "O_WRONLY"
    data.write_bytes(b"payload")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 openat(AT_FDCWD<{tmp_path}>, "O_WRONLY", O_RDONLY) = 3<{data}>',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any("own observed" in row["reason"] for row in result["unresolved"])


def test_mmap_flag_is_not_inferred_from_filename(tmp_path):
    data = tmp_path / "MAP_ANONYMOUS"
    data.write_bytes(b"payload")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, 3<{data}>, 0) = 0x1234',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any("mapped FD lacks traced open identity" in row["reason"] for row in result["unresolved"])


def test_two_path_syscall_requires_both_actual_paths(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"a")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 rename("{source}", 0x1234) = 0',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any("cannot resolve exact pathname" in row["reason"] for row in result["unresolved"])


def test_successful_stat_size_changed_is_diagnostic(tmp_path):
    data = tmp_path / "payload"
    data.write_bytes(b"payload")
    wrong = _stat_result(data).replace("st_size=7", "st_size=70")
    trace = _synthetic_trace(tmp_path, [
        f'1.000000 newfstatat(AT_FDCWD<{tmp_path}>, "payload", {wrong}, 0) = 0',
    ])
    result = parse_trace(trace, command_exitcode=0)
    assert result["trace_complete"] is False
    assert any("metadata size/type changed" in row["reason"] for row in result["unresolved"])


def test_fifo_hash_is_rejected_without_blocking(tmp_path):
    from bevy_microduck_tools.profile_observer import _real_file, _checked_sha256_file
    fifo = tmp_path / "replaced-by-fifo"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(Rejection, match="regular|nonregular"):
        _real_file(str(fifo), deadline=started + 1)
    with pytest.raises(Rejection, match="regular|nonregular"):
        _checked_sha256_file(fifo, started + 1)
    assert time.monotonic() - started < .5


@pytest.mark.parametrize("key", ["HOME", "TMPDIR", "XDG_CACHE_HOME", "CUBLAS_WORKSPACE_CONFIG",
                                 "PYTHONHASHSEED", "PYTHONSAFEPATH", "PYTHONNOUSERSITE"])
def test_execution_environment_binds_cache_and_numeric_keys(tmp_path, monkeypatch, key):
    manifest, path, _, _, _ = _fixture(tmp_path)
    assert key in SELECTED_ENV
    assert key in _execution_env_hashes({key: "a"})
    monkeypatch.setenv(key, "different-value")
    with pytest.raises(Rejection, match="resolution|environment"):
        verify_declared_inputs(manifest, manifest_path=path, require_runtime=False)


def test_v10_review_schema_cannot_authorize_v11(tmp_path):
    manifest, _, review, _, ledger = _fixture(tmp_path)
    request = DiscoveryRequest(manifest["input_id"], manifest["nonce"], "zero_update_discovery",
                               1000001, 180, 1, str(ledger))
    assert REVIEW_SCHEMA.endswith("_v2")
    with pytest.raises(Rejection, match="Exact root"):
        authorize_discovery({**review, "schema": "microduck_root_discovery_review_v1"}, request)


def test_postprocess_overrun_never_claims_complete(tmp_path, monkeypatch):
    from bevy_microduck_tools import profile_observer
    original = profile_observer.parse_trace

    def delayed(*args, **kwargs):
        time.sleep(.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(profile_observer, "parse_trace", delayed)
    result = observe_command(["/bin/true"], tmp_path / "observed", wall_seconds=.5)
    assert result["status"] == "budget_exhausted"
    assert result["trace_complete"] is False
    assert (tmp_path / "observed" / "observer_report.json").is_file()


def test_v13_positive_discovery_closes_once_without_completed_state(tmp_path):
    _, manifest_path, _, review_path, ledger = _fixture(tmp_path)
    token = dataclasses.replace(preflight_parent(manifest_path, require_runtime=False).token,
                                runtime_verified=True)
    binding = reserve_discovery(manifest_path, review_path, ledger,
                                parent_token=token, wall_seconds=.2)
    run_id = binding["run_id"]
    started = time.monotonic()
    state, measured = _finish_discovery_budget(ledger, run_id, started=started,
                                                deadline=started + .2,
                                                status="completed_zero_update_discovery")
    assert state == "budget_exhausted" and measured >= 0
    with TrainingBudget(ledger).locked(commit=False) as stored:
        assert stored["runs"][0]["state"] == "budget_exhausted"
        assert stored["runs"][0]["deadline_correction"] == "v13_completion_commit_unproven"
        assert stored["runs"][0]["requested_discovery_status"] == "completed_zero_update_discovery"


def test_v13_preexisting_deadline_exhaustion_closes_once(tmp_path):
    _, manifest_path, _, review_path, ledger = _fixture(tmp_path)
    token = dataclasses.replace(preflight_parent(manifest_path, require_runtime=False).token,
                                runtime_verified=True)
    binding = reserve_discovery(manifest_path, review_path, ledger,
                                parent_token=token, wall_seconds=.2)
    run_id = binding["run_id"]
    started = time.monotonic() - .1
    state, measured = _finish_discovery_budget(ledger, run_id, started=started,
                                                deadline=started + .04, status="failed")
    assert state == "budget_exhausted" and measured >= .1
    with TrainingBudget(ledger).locked(commit=False) as stored:
        row = stored["runs"][0]
        assert row["deadline_correction"] == "v13_deadline_crossed_before_commit"
        assert row["charged_gpu_seconds"] >= .2
        assert row["reserved_wall_seconds_charged"] == .2
        assert row["measured_gpu_seconds"] == row["closure_elapsed_lower_bound_seconds"]
