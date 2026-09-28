"""Real spawned-process transport tests; no policy or training is executed."""

import json
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from bevy_microduck_tools.serialization import sha256_file
from bevy_microduck_tools.training import _atomic_worker_result, _run_child, _send_result_locator, train_bounded


def process_probe(connection, result_path, mode):
    path = Path(result_path)
    try:
        if mode == "large":
            _send_result_locator(connection, path, {"status": "completed_structural_smoke", "test_only": True,
                                  "result": {"video_blob": "x" * (4 * 1024 * 1024), "policy_or_training_executed": False}})
        elif mode == "zero_restore":
            _send_result_locator(connection, path, {"status": "completed_zero_update_restore", "test_only": True,
                                  "ppo_iterations_this_invocation": 0})
        elif mode == "partial":
            path.with_suffix(path.suffix + ".partial").write_text('{"status":')
            connection.send_bytes(json.dumps({"status": "completed_structural_smoke", "result_path": str(path.resolve()),
                                               "result_sha256": "0" * 64}).encode())
        elif mode == "bad_json":
            path.write_text('{"status":')
            connection.send_bytes(json.dumps({"status": "completed_structural_smoke", "result_path": str(path.resolve()),
                                               "result_sha256": sha256_file(path)}).encode())
        elif mode == "bad_hash":
            _atomic_worker_result(path, {"status": "completed_structural_smoke", "test_only": True})
            connection.send_bytes(json.dumps({"status": "completed_structural_smoke", "result_path": str(path.resolve()),
                                               "result_sha256": "0" * 64}).encode())
        elif mode == "reported_error":
            _send_result_locator(connection, path, {"status": "failed", "error_type": "ExpectedTestError", "test_only": True})
        elif mode == "crash":
            raise RuntimeError("Expected test-only child crash")
        elif mode == "success_then_crash":
            _send_result_locator(connection, path, {"status": "completed_structural_smoke", "test_only": True})
            raise RuntimeError("Expected test-only post-send crash")
        elif mode == "oversized":
            connection.send_bytes(b"x" * 8192)
        elif mode == "timeout":
            connection.close()
            time.sleep(10)
    finally:
        connection.close()


def budget_timeout_probe(connection, result_path, *unused):
    process_probe(connection, result_path, "timeout")


class TrainingExecutorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_probe(self, mode, seconds=5):
        return _run_child(process_probe, (mode,), self.root / f"{mode}.json", seconds)

    def test_four_megabyte_result_is_atomic_file_with_small_ipc(self):
        result = self.run_probe("large")
        self.assertEqual(result["status"], "completed_structural_smoke")
        self.assertEqual(len(result["result"]["video_blob"]), 4 * 1024 * 1024)
        self.assertLess(result["ipc_message_bytes"], 4096)
        path = Path(result["atomic_worker_result"]["path"])
        self.assertEqual(sha256_file(path), result["atomic_worker_result"]["sha256"])
        self.assertFalse(path.with_suffix(path.suffix + ".partial").exists())
        print(f"TEST_ONLY transport: {path.stat().st_size} result bytes, {result['ipc_message_bytes']} IPC bytes; no policy/training")

    def test_zero_update_restore_transport_preserves_distinct_status(self):
        result = self.run_probe("zero_restore")
        self.assertEqual(result["status"], "completed_zero_update_restore")
        self.assertEqual(result["ppo_iterations_this_invocation"], 0)

    def test_partial_missing_corrupt_and_mismatched_result_are_rejected(self):
        for mode, reason in (("partial", "missing"), ("bad_json", "JSON"), ("bad_hash", "hash")):
            with self.subTest(mode=mode):
                result = self.run_probe(mode)
                self.assertEqual(result["status"], "failed")
                self.assertIn(reason, result["reason"])

    def test_reported_error_crash_and_nonzero_success_exit_are_rejected(self):
        self.assertEqual(self.run_probe("reported_error")["error_type"], "ExpectedTestError")
        self.assertEqual(self.run_probe("crash")["status"], "failed")
        result = self.run_probe("success_then_crash")
        self.assertEqual(result["status"], "failed")
        self.assertIn("claimed success", result["reason"])

    def test_oversized_ipc_does_not_block_parent(self):
        result = self.run_probe("oversized")
        self.assertEqual(result["status"], "failed")
        self.assertIn("IPC", result["reason"])

    def test_live_process_is_killed_at_deadline_and_budget_is_measured(self):
        # Admission/runtime identity and the scientific worker are replaced here.
        # Real spawn, pipe/exit handling, kill, and budget ledger are exercised.
        paths = {name: self.root / f"{name}.json" for name in ("candidate", "evidence", "review", "adoption")}
        paths["adoption"].write_text(json.dumps({"skill": "test_only_standing"}))
        paths["candidate"].write_text(json.dumps({"candidate_id": "test_only_transport", "contract": {"source_substeps": 1},
                    "artifacts": {"adoption": {"path": str(paths['adoption'])}, "checkpoint": {"path": "not_loaded_in_test"}}}))
        paths["evidence"].write_text('[]')
        paths["review"].write_text(json.dumps({"test_only":True,"schema":"microduck_root_review_v2",
                "authorization_id":str(uuid.uuid4()),"candidate_id":"test_only_transport",
                "reviewer_role":"root_gpt","decision":"approve","reviewed_at":time.time()-1,"expires_at":time.time()+10,
                "authorized_stages":["learning"],"learning_allowed":True,
                "learning_limit":{"iterations":1,"max_wall_seconds":.3,"seed":1000001,"gpus":1,
                                  "max_runs":1,"budget_ledger_path":str((self.root/"ledger.json").resolve())}}))
        ledger = self.root / "ledger.json"
        started = time.monotonic()
        with patch("bevy_microduck_tools.training.admit"), patch("bevy_microduck_tools.training.verify_candidate_runtime"), patch("bevy_microduck_tools.training._worker", budget_timeout_probe):
            result = train_bounded(paths["candidate"], paths["evidence"], paths["review"], source_root=self.root,
                                   output=self.root / "timeout_run", ledger=ledger, seed=1000001, wall_seconds=.3)
        elapsed = time.monotonic() - started
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertLess(elapsed, 2)
        run = json.loads(ledger.read_text())["runs"][0]
        self.assertEqual(run["state"], "budget_exhausted")
        self.assertGreaterEqual(run["measured_gpu_seconds"], .3)
        self.assertLess(run["measured_gpu_seconds"], 2)
        self.assertNotEqual(run["measured_gpu_seconds"], run["reserved_gpu_seconds"])


if __name__ == "__main__":
    unittest.main()
