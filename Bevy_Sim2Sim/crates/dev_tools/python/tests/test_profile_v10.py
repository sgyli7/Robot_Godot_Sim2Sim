"""Actual CPU observer/authorization fixtures; no robot environment or GPU."""
from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import uuid
from pathlib import Path

import pytest

from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_discovery import (MODE, REVIEW_SCHEMA, DiscoveryRequest,
    _finish_discovery_budget, authorize_discovery, claim_worker, finalize_profile,
    preflight_parent, reserve_discovery, require_claimed_worker)
from bevy_microduck_tools.profile_identity import (PROFILE, RESTORED_CHECKPOINT_SHA256, SELECTED_ENV, SCHEMA, _execution_env_hashes, _regular_catalog,
    private_cache_environment,
    require_zero_token, verify_declared_inputs)
from bevy_microduck_tools.profile_observer import ProducerJournal, observe_command, parse_trace
from bevy_microduck_tools.serialization import identity, sha256_file, write_json
from bevy_microduck_tools.workflow import TrainingBudget


def _fixture(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("science=0\n")
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"native_profile_complete":false}\n')
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture only; not a policy")
    wheel = tmp_path / "fixture.whl"
    wheel.write_bytes(b"fixture bytes; not a distributable wheel")
    manifest_path = tmp_path / "manifest.json"
    private_root = tmp_path / "source_run" / "private_cache"
    child_env = dict(os.environ)
    child_env.update(WARP_CACHE_PATH=str(private_root / "warp"), CUDA_CACHE_PATH=str(private_root / "cuda"))
    receipt = {"schema": SCHEMA, "mode": MODE, "nonce": str(uuid.uuid4()),
        "source_root": str(tmp_path), "source_commit": "fixture-only",
        "profile": dict(PROFILE), "adoption_sha256": "a" * 64,
        "runtime_receipt": {"path": str(runtime), "sha256": sha256_file(runtime),
                            "identity": "b" * 64, "native_profile_complete": False},
        "checkpoint_origin": {"path": str(source), "sha256": sha256_file(source),
                              "checkpoint_path": str(checkpoint), "checkpoint_sha256": RESTORED_CHECKPOINT_SHA256,
                              "known_prior_ppo_iterations": 1},
        "python_executable": str(Path(sys.executable).absolute()), "python_version": sys.version,
        "stdlib_root": sysconfig.get_path("stdlib"),
        "adopted_source_files": [str(source)],
        "tool_python_files": [str(source)],
        "tool_wheel": str(wheel),
        "resolution": {key: os.environ.get(key) for key in SELECTED_ENV},
        "execution_environment_sha256": _execution_env_hashes(),
        "child_resolution": {key: child_env.get(key) for key in SELECTED_ENV},
        "child_execution_environment_sha256": _execution_env_hashes(child_env),
        "private_cache_root": str(private_root),
        "cache_policy": {"warp": "empty_run_private_cache", "cuda": "empty_run_private_cache",
                         "adopted_seed_files": [], "cold_compile_may_exceed_180s": True},
        "declared_files": _regular_catalog([source, runtime, checkpoint, wheel, Path(sys.executable)]),
        "output_root": str(tmp_path / "source_run"), "parent_preflight_max_seconds": 120,
        "source_child_hard_seconds": 180,
        "covered_modes_requested": ["source_compile", "env_graph", "deterministic_inference",
                                    "natural_terminal_reset", "save_resume_export", "real_video"],
        "not_executed": ["PPO_stochastic_collection", "PPO_backward", "Adam_step", "learn_reset"],
        "qualification": "provisional input identity, zero updates; requires actual-use observer and root review"}
    receipt["input_id"] = identity(receipt)
    write_json(manifest_path, receipt)
    ledger = tmp_path / "ledger.json"
    review = {"schema": REVIEW_SCHEMA, "authorization_id": str(uuid.uuid4()),
              "reviewer_role": "root_gpt", "decision": "approve", "reviewed_at": time.time() - 1,
              "expires_at": time.time() + 90, "input_id": receipt["input_id"], "nonce": receipt["nonce"],
              "authorized_mode": MODE, "learning_allowed": False, "max_runs": 1,
              "max_wall_seconds": 180, "seed": PROFILE["seed"], "gpus": 1,
              "budget_ledger_path": str(ledger.resolve())}
    review_path = tmp_path / "review.json"
    write_json(review_path, review)
    return receipt, manifest_path, review, review_path, ledger


def test_cpu_byte_token_and_exact_mode(tmp_path):
    manifest, path, review, review_path, ledger = _fixture(tmp_path)
    result = preflight_parent(path, require_runtime=False)
    assert result.record()["native_profile_complete"] is False
    token = verify_declared_inputs(manifest, manifest_path=path, require_runtime=False)
    require_zero_token(token, manifest, manifest_path=path)
    for wrong in (dataclasses.replace(token, process_id=token.process_id + 1),
                  dataclasses.replace(token, nonce=str(uuid.uuid4())),
                  dataclasses.replace(token, mode="learning"),
                  dataclasses.replace(token, native_profile_complete=True)):
        with pytest.raises(Rejection): require_zero_token(wrong, manifest, manifest_path=path)
    source = tmp_path / "source.py"
    source.write_text("science=1\n")
    with pytest.raises(Rejection, match="changed"):
        preflight_parent(path, require_runtime=False)


def test_parent_child_cache_resolution_is_distinct_and_bound(tmp_path, monkeypatch):
    manifest, path, _, _, _ = _fixture(tmp_path)
    parent = verify_declared_inputs(manifest, manifest_path=path, require_runtime=False)
    assert parent.verification_phase == "parent"
    child_env = private_cache_environment(manifest)
    with monkeypatch.context() as patch:
        patch.setenv("WARP_CACHE_PATH", child_env["WARP_CACHE_PATH"])
        patch.setenv("CUDA_CACHE_PATH", child_env["CUDA_CACHE_PATH"])
        with pytest.raises(Rejection, match="resolution"):
            verify_declared_inputs(manifest, manifest_path=path, require_runtime=False)
        child = verify_declared_inputs(manifest, manifest_path=path, require_runtime=False, for_child=True)
        assert child.verification_phase == "child"


def test_single_use_authorization_and_claim(tmp_path):
    manifest, path, review, review_path, ledger = _fixture(tmp_path)
    preflight = preflight_parent(path, require_runtime=False)
    with pytest.raises(Rejection, match="actual full v9"):
        reserve_discovery(path, review_path, ledger, parent_token=preflight.token)
    # Exercise only atomic ledger mechanics with a synthetic full-check claim;
    # this is not an actual scientific environment or a production token.
    fixture_token = dataclasses.replace(preflight.token, runtime_verified=True)
    with pytest.raises(Rejection):
        reserve_discovery(path, review_path, ledger,
                          parent_token=dataclasses.replace(fixture_token, native_profile_complete=True))
    assert not ledger.exists()
    def attempt():
        try:
            return reserve_discovery(path, review_path, ledger, parent_token=fixture_token)
        except Rejection as error:
            return str(error)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    bindings = [result for result in results if isinstance(result, dict)]
    assert len(bindings) == 1
    assert any("already consumed" in result for result in results if isinstance(result, str))
    consumed = bindings[0]
    with pytest.raises(Rejection): claim_worker({**consumed, "nonce": str(uuid.uuid4())})
    with pytest.raises(Rejection): claim_worker({**consumed, "manifest_sha256": "0" * 64})
    actual = claim_worker(consumed)
    token = dataclasses.replace(verify_declared_inputs(actual, manifest_path=path, require_runtime=False),
                                runtime_verified=True, verification_phase="child")
    require_claimed_worker(consumed, token, actual)
    with pytest.raises(Rejection, match="claim"):
        claim_worker(consumed)
    with TrainingBudget(ledger).locked(commit=False) as rows:
        assert len(rows["runs"]) == 1
        assert rows["runs"][0]["identity_kind"] == "source_profile_input_id"
    started = time.monotonic()
    closed, _ = _finish_discovery_budget(ledger, consumed["run_id"], started=started,
                                         deadline=started + .2, status="failed")
    assert closed == "budget_exhausted"
    with pytest.raises(Rejection, match="already consumed"):
        reserve_discovery(path, review_path, ledger, parent_token=fixture_token)


def test_review_scope_and_corrupt_ledger_refuse(tmp_path):
    manifest, path, review, review_path, ledger = _fixture(tmp_path)
    request = DiscoveryRequest(manifest["input_id"], manifest["nonce"], MODE, 1000001, 180, 1, str(ledger))
    authorize_discovery(review, request)
    for edit in ({"learning_allowed": True}, {"authorized_mode": "learning"},
                 {"nonce": str(uuid.uuid4())}, {"max_runs": 2},
                 {"max_wall_seconds": 30}, {"gpus": 2}):
        wrong = {**review, **edit}
        with pytest.raises(Rejection): authorize_discovery(wrong, request)
    ledger.write_text('{"runs":[{"broken":true}]}')
    preflight = preflight_parent(path, require_runtime=False)
    with pytest.raises(Rejection, match="Corrupt"):
        reserve_discovery(path, review_path, ledger,
                          parent_token=dataclasses.replace(preflight.token, runtime_verified=True))


def test_actual_transient_dlopen_file_and_exec_child(tmp_path):
    c = tmp_path / "short.c"
    c.write_text("int answer(void){return 42;}\n")
    library = tmp_path / "libshort.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(c), "-o", str(library)], check=True)
    loader_source = tmp_path / "short_loader.c"
    loader_source.write_text("#include <dlfcn.h>\nint main(int argc,char**argv){"
                             "void*h=dlopen(argv[1],RTLD_NOW);if(!h)return 2;"
                             "int(*f)(void)=dlsym(h,\"answer\");int v=f();dlclose(h);return v==42?0:3;}\n")
    loader = tmp_path / "short_loader"
    subprocess.run(["cc", str(loader_source), "-ldl", "-o", str(loader)], check=True)
    data = tmp_path / "actual_input.txt"
    data.write_text("real file bytes\n")
    script = tmp_path / "actual_use.py"
    script.write_text("import subprocess,sys\n"
                      f"subprocess.run([{str(loader)!r},{str(library)!r}],check=True)\n"
                      f"assert open({str(data)!r}).read().strip()=='real file bytes'\n"
                      "subprocess.run([sys.executable,'-c','import sys;print(123)'],check=True)\n")
    output = observe_command([sys.executable, str(script)], tmp_path / "observed", wall_seconds=8,
        declared={row["path"]: row for row in _regular_catalog([script, data, library, loader, Path(sys.executable)])})
    assert output["status"] == "completed" and not output["trace_complete"]
    assert output["failed_candidates"] and output["successful_metadata"]
    assert any(row["errno"] == "ENOENT" for row in output["failed_candidates"])
    assert len(output["trace_files"]) >= 2
    lib = next(row for row in output["posthoc_actual_use"] if row["resolved_path"] == str(library.resolve()))
    assert lib["actual_opened_devices"] and lib["sha256"] == sha256_file(library)
    assert any(item["path"] == str(data) for item in output["posthoc_actual_use"])
    assert len(output["exec_events"]) >= 2
    assert output["unknown_inputs"]  # system loader + stdlib remain provisional
    trace_file = Path(output["trace_files"][0]["path"])
    lines = trace_file.read_text().splitlines()
    trace_file.write_text("\n".join(lines[:-1]) + "\n")
    partial = parse_trace(trace_file.parent, command_exitcode=0)
    assert partial["trace_complete"] is False


def test_actual_child_phase_and_generated_journal_join(tmp_path):
    phase = tmp_path / "phases.jsonl"
    producers = tmp_path / "producers.json"
    blob = tmp_path / "new_module.cubin"
    script = tmp_path / "journal_child.py"
    script.write_text("import json,time,pathlib\n"
                      "from bevy_microduck_tools.profile_observer import ProducerJournal\n"
                      f"phase=pathlib.Path({str(phase)!r})\n"
                      "phase.write_text(json.dumps({'phase':'fixture_actual_use','wall_seconds':time.time()})+'\\n')\n"
                      f"blob=pathlib.Path({str(blob)!r});blob.write_bytes(b'generated fixture')\n"
                      "journal=ProducerJournal();journal.record_generated(blob,producer='fixture',recipe={'input':'actual'})\n"
                      "journal.before_load(blob,expected_producer='fixture')\n"
                      "assert blob.read_bytes()==b'generated fixture'\n"
                      f"pathlib.Path({str(producers)!r}).write_text(json.dumps({{'schema':'microduck_profile_producer_journal_v1','generated':journal.produced,'loaded':journal.loaded,'scope':'CPU fixture'}}))\n")
    report = observe_command([sys.executable, str(script)], tmp_path / "observed", wall_seconds=8,
                             phase_path=phase, generated_path=producers)
    assert report["status"] == "completed"
    assert report["trace_complete"] is False
    assert report["failed_candidates"] and report["successful_metadata"]
    used = next(row for row in report["posthoc_actual_use"] if row["path"] == str(blob))
    assert used["classification"] == "generated_input"
    assert "fixture_actual_use" in used["phases"]
    assert report["producer_journal"]["sha256"] == sha256_file(producers)


def test_missing_trace_and_process_group_deadline(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(Rejection, match="Missing syscall trace"):
        parse_trace(tmp_path / "empty", command_exitcode=0)
    marker = tmp_path / "child_pid"
    script = tmp_path / "tree.py"
    script.write_text("import subprocess,time\n"
                      "p=subprocess.Popen(['sleep','60'],start_new_session=True)\n"
                      f"open({str(marker)!r},'w').write(str(p.pid))\n"
                      "time.sleep(60)\n")
    result = observe_command([sys.executable, str(script)], tmp_path / "timed", wall_seconds=.4)
    assert result["status"] == "budget_exhausted"
    assert result["trace_complete"] is False
    if marker.exists():
        pid = int(marker.read_text())
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            state = "gone"
        assert state in {"gone", "Z", "X"}
    assert result["teardown"]["pidfd_killed_tracees"]


def test_generated_recipe_tamper_missing_and_unknown(tmp_path):
    blob = tmp_path / "module.ptx"
    blob.write_bytes(b"ptx fixture one")
    journal = ProducerJournal()
    with pytest.raises(Rejection, match="missing"):
        journal.before_load(blob)
    journal.record_generated(blob, producer="fixture-compiler", recipe={"source_sha256": "a" * 64,
                                                                     "target_arch": "sm_121"})
    journal.before_load(blob, expected_producer="fixture-compiler")
    with pytest.raises(Rejection, match="changed"):
        journal.before_load(blob, expected_producer="unknown-compiler")
    blob.write_bytes(b"tampered blob")
    with pytest.raises(Rejection, match="changed"):
        journal.before_load(blob, expected_producer="fixture-compiler")
    blob.write_bytes(b"ptx fixture one")
    journal.produced[str(blob.resolve())]["recipe"]["target_arch"] = "sm_80"
    with pytest.raises(Rejection, match="changed"):
        journal.before_load(blob, expected_producer="fixture-compiler")


def test_actual_jit_cache_migration_and_same_bytes_new_inode_rejected(tmp_path):
    temporary = tmp_path / "temporary.cubin"
    temporary.write_bytes(b"actual generated bytes")
    final = tmp_path / "cache" / "module.cubin"
    final.parent.mkdir()
    journal = ProducerJournal()
    journal.record_generated(temporary, producer="fixture-warp-compiler", recipe={"source_sha256": "a" * 64})
    temporary.rename(final)
    moved = journal.transfer(temporary, final, operation="fixture-rename")
    assert moved["sha256"] == sha256_file(final)
    journal.before_load(final, expected_producer="warp.cache_migration")
    replacement = tmp_path / "same_bytes.cubin"
    replacement.write_bytes(final.read_bytes())
    os.replace(replacement, final)
    with pytest.raises(Rejection, match="changed"):
        journal.before_load(final, expected_producer="warp.cache_migration")


def test_profile_receipt_never_promotes_missing_phases_or_unknown_inputs(tmp_path):
    manifest, _, _, _, _ = _fixture(tmp_path)
    observer = {"trace_complete": False, "posthoc_actual_use": [],
                "unknown_inputs": ["/usr/lib/unknown.so"], "unresolved": [{"reason": "partial trace"}]}
    receipt = finalize_profile(manifest, observer, phase_journal=[], source_result=None, worker_status="failed")
    assert receipt["native_profile_complete"] is False
    assert receipt["learning_allowed"] is False
    assert set(receipt["scope"]["not_executed"]) == set(manifest["not_executed"])
    assert {"source_worker_not_completed", "observer_trace_incomplete", "unresolved_file_use",
            "unknown_actual_input_requires_root_review", "natural_source_terminal_unproven",
            "phase_coverage_incomplete"} <= set(receipt["failures"])
