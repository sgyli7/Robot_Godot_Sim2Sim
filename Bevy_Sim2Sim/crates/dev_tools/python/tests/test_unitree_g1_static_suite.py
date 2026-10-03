"""CPU ownership/continuation checks; no task-performance evidence."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unitree_g1_static_suite as suite


class ContinuationTests(unittest.TestCase):
    def test_port_probe_rejects_live_listener_and_accepts_its_closed_connections(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaises(OSError):
                suite.check_port_available(port)
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = listener.accept()
                accepted.close()
                self.assertEqual(client.recv(1), b"")
        suite.check_port_available(port)

    def fixture(self, directory):
        parent = {key: 1 for key in ("acceptance", "profile", "revision", "seed_offsets",
            "physics_hz", "integrations_per_tick", "startup_ticks", "task_ticks",
            "max_vla_calls", "model_receipt_sha256", "pauses_for_camera_and_policy")}
        parent["cases"] = [{"position_id": p, "seed_offset": s,
                            "configuration": {"episode": p + 5 * s}}
                           for s in (0, 42) for p in range(5)]
        parent["tools_sha256"] = {"unitree_g1_static_suite.py": "old", "model.py": "same"}
        suite.save(directory / "frozen_manifest.json", parent)
        parent_sha = suite.digest(directory / "frozen_manifest.json")
        old = {"complete": False, "binary_sha256": "binary", "code_commit": "old-head",
               "manifest_sha256": parent_sha, "cases": []}
        for p in range(5):
            result = {"position_id": p, "seed_offset": 0, "strict_placement_passed": p == 0,
                "trace_sha256": str(p), "standing_all_ticks": True,
                "max_continuous_placement_seconds": 2.0 if p == 0 else 0.0,
                "minimum_margin_m": None, "evidence_verified": True,
                "actual_integrations": 380, "fresh_vla_calls": 8, "model_counter_verified": True}
            case = directory / f"position_{p}_seed_0"
            case.mkdir()
            suite.save(case / "config.json", parent["cases"][p]["configuration"])
            suite.save(case / "execution.json", {"code_commit": "old-head",
                "binary_sha256": "binary", "frozen_manifest_sha256": parent_sha, "result": result})
            old["cases"].append(result)
        suite.save(directory / "summary.json", old)
        (directory / "seed_offset_0").mkdir()
        suite.save(directory / "seed_offset_0/model_lifecycle.json", {
            "owned_model_closed": True, "initial_health": {"successful_inferences": 0},
            "final_health": {"successful_inferences": 40, "failed_inferences": 0}})
        child = json.loads(json.dumps(parent))
        child["parent_manifest_sha256"] = parent_sha
        child["tools_sha256"]["unitree_g1_static_suite.py"] = "new"
        return child, old

    def test_retained_failures_are_reaudited_without_reexecution(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            child, old = self.fixture(directory)
            with patch.object(suite, "audit", side_effect=lambda case, seed:
                    old["cases"][int(case.name.split("_")[1])]) as audit:
                results, origin = suite.retained_cases(directory,
                    suite.digest(directory / "summary.json"), child, "binary")
            self.assertEqual(results, old["cases"])
            self.assertEqual(audit.call_count, 5)
            self.assertEqual(sum(c["strict_placement_passed"] for c in results), 1)
            self.assertEqual(origin["reexecuted_cases"], 0)
            self.assertEqual(origin["code_commit"], "old-head")

    def test_continuation_rejects_changed_thresholds_cases_models_and_summary(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            child, _ = self.fixture(directory)
            checksum = suite.digest(directory / "summary.json")
            for key in ("acceptance", "cases", "model_receipt_sha256"):
                changed = {**child, key: "changed"}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    suite.retained_cases(directory, checksum, changed, "binary")
            with self.assertRaises(ValueError):
                suite.retained_cases(directory, "wrong-summary", child, "binary")


if __name__ == "__main__":
    unittest.main()
