"""Adversarial suite/route fixtures only, no simulated robot successes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unitree_g1_mobile_suite as suite
from unitree_g1_mobile_capture_audit import digest, route_evidence


def manifest(directory):
    for name in suite.TOOLS:
        (directory / name).write_text("fixture executable identity\n")
    cases = []
    for seed in [0, 42]:
        for position in range(5):
            cfg = {"runner": {"body": {"mobile_homie_v2": {
                "episode_id": seed * 5 + position,
                "task_objects": {"placements": [{"kind": "t2_box", "root_pose": {"position": [position, 0, 1]}}],
                                 "source_t2_background": {"selection": "station_task_fixtures"}}}}},
                "policy": {"endpoint": "http://127.0.0.1:5558/infer", "max_calls": 4, "timeout_ms": 20000},
                "station": {"fixture": "protocol_fixture"},
                "background_visual": {"sha256": "69129a51ae7fd53ddf9a00cce7170bc780e4d0f4fceff5c720cbe359db7eee9e"},
                "diagnostic_qwen_dispatch": {"scope": "scientific_station_mobile_from_instruction_v1",
                    "connection": {"endpoint": "http://127.0.0.1:8002/v1", "model": "qwen3.8-27b-fp8",
                                   "timeout_ms": 20000, "max_output_tokens": 128}}}
            cases.append({"position_id": position, "seed": seed, "configuration": cfg})
    return {"schema": "g1_native_station_mobile_profile_suite_v1", "revision": suite.REVISION,
            "physics_hz": 50, "integrations_per_tick": 1, "seeds": [0, 42],
            "acceptance": copy.deepcopy(suite.RULES), "full_task_qualified": False,
            "pauses_for_camera_and_policy": True, "cases": cases,
            "tools_sha256": {name: suite.digest(directory / name) for name in suite.TOOLS}}


def route_fixture():
    definition = {"bodies": [{"name": "left_ankle_roll_link"}, {"name": "left_hand_thumb_link"}]}
    rows = []
    for tick in range(1, 203):
        body = {"integration_count": tick, "torque_update_count": tick,
                "inference": {"inference_count": tick}, "step_configuration": {
                    "physics_hz": 50, "dt": .02, "num_solver_iterations": 1,
                    "max_ccd_substeps": 1, "num_internal_pgs_iterations": 4,
                    "additional_solver_iterations_max": 0},
                "native_static_environment": {"collider_count": 2553, "original_broad_floor_removed": True},
                "native_station_contacts": [], "root_upright_cosine": 1., "root_position_source": [tick*.01, 0, .75],
                "task_objects": {"objects": [{"kind": "t2_box", "position_source": [tick*.01, 0, 1.],
                    "last_solve_contacts": [{"active_solver_normal_impulse_n_s": .02, "other_robot_body_index": 1}]}]}}
        rows.append({"body": {"mobile_homie_v2": body}, "owner_episode_integrations": tick,
                     "execution": {"original_vla" if tick <= 200 else "classical_carry": {}}})
    return definition, rows


class MobileSuiteTests(unittest.TestCase):
    def partial_case(self, directory):
        definition, rows = route_fixture()
        (directory / "native").mkdir()
        body = directory / "robot.json"
        objects = directory / "objects.json"
        body.write_text(json.dumps(definition))
        objects.write_text("{}")
        cfg = {"runner": {"body": {"mobile_homie_v2": {
            "episode_id": 17, "definition": str(body), "definition_sha256": digest(body),
            "task_objects": {"definition": str(objects), "definition_sha256": digest(objects)}}}}}
        (directory / "config.json").write_text(json.dumps(cfg))
        for row in rows:
            row["episode_id"] = 17
        trace = directory / "native/owner_steps.jsonl"
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows, trace

    def test_timeout_retains_verified_trace_count_without_fabricating_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, trace = self.partial_case(directory)
            result = suite.failed_case_evidence(directory, 4, FileNotFoundError("capture_receipt.json"))
            self.assertEqual(result["actual_integrations"], len(rows))
            self.assertTrue(result["partial_trace_verified"])
            self.assertFalse(result["evidence_verified"])
            self.assertFalse(result["strict_task_passed"])
            self.assertTrue(result["application_receipt_missing"])
            self.assertEqual(result["trace_sha256"], digest(trace))
            self.assertEqual(result["fresh_vla_calls"], 4)
            self.assertNotIn("qwen_http_results", result)
            self.assertFalse((directory / "native/capture_receipt.json").exists())

    def test_invalid_or_missing_partial_trace_is_unknown_instead_of_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, trace = self.partial_case(directory)
            for mutation in ["duplicate", "foreign_episode", "substep", "truncated", "missing"]:
                changed = copy.deepcopy(rows)
                if mutation == "duplicate": changed[-1]["owner_episode_integrations"] -= 1
                if mutation == "foreign_episode": changed[-1]["episode_id"] = 18
                if mutation == "substep": changed[-1]["body"]["mobile_homie_v2"]["step_configuration"]["physics_hz"] = 200
                trace.write_text("".join(json.dumps(row) + "\n" for row in changed))
                if mutation == "truncated": trace.write_text(trace.read_text() + '{"episode_id":')
                if mutation == "missing": trace.unlink()
                result = suite.failed_case_evidence(directory, 4, RuntimeError("strict audit failed"))
                self.assertIsNone(result["actual_integrations"], mutation)
                self.assertFalse(result["partial_trace_verified"], mutation)
                self.assertFalse(result["strict_task_passed"], mutation)
                self.assertIn("partial_trace_error", result)

    def test_foreign_definitions_and_unverified_receipt_counts_cannot_override_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, _ = self.partial_case(directory)
            receipt = directory / "native/capture_receipt.json"
            receipt.write_text(json.dumps({"actual_integrations": 999, "failure_reason": "timeout"}))
            result = suite.failed_case_evidence(directory, 4, RuntimeError("mismatched count"))
            self.assertEqual(result["actual_integrations"], len(rows))
            self.assertFalse(result["application_receipt_missing"])
            self.assertFalse(result["evidence_verified"])
            (directory / "robot.json").write_text("{}")
            result = suite.failed_case_evidence(directory, 4, RuntimeError("changed definition"))
            self.assertIsNone(result["actual_integrations"])
            self.assertFalse(result["partial_trace_verified"])
            self.assertIn("unbound", result["partial_trace_error"])

    def test_foreign_window_closing_retries_only_the_read_without_restarting_app(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        recorder = Mock(pid=456)
        tree = '0x123456 "G1 科学站任务验证"'
        info = 'IsViewable\nWidth: 1920\nHeight: 1080\n'
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(suite.subprocess, "check_output", side_effect=[
                    subprocess.CalledProcessError(1, ["xwininfo", "-root", "-tree"]), tree, info]) as query, \
                patch.object(suite.subprocess, "run", return_value=Mock(stdout='_NET_WM_PID = 123\n')), \
                patch.object(suite.subprocess, "Popen", return_value=recorder) as spawn, \
                patch.object(suite.time, "sleep"), patch.dict(suite.os.environ, {"DISPLAY": ":fixture"}):
            actual, receipt = suite.record_owned_window(child, Path(temporary), "fixture_ffmpeg")
            self.assertIs(actual, recorder)
            self.assertEqual(receipt["read_only_window_tree_retries"], 1)
            self.assertEqual(query.call_count, 3)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(spawn.call_args.args[0][0], "fixture_ffmpeg")
            self.assertTrue(receipt["window_pid_matches_owned_app"])

    def test_threshold_seed_order_duplicate_and_tool_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frozen = manifest(directory)
            suite.validate(frozen, directory)
            mutations = []
            c = copy.deepcopy(frozen); c["acceptance"]["required_successes"] = 7; mutations.append(c)
            c = copy.deepcopy(frozen); c["cases"] = list(reversed(c["cases"])); mutations.append(c)
            c = copy.deepcopy(frozen); c["cases"][1]["configuration"]["runner"]["body"]["mobile_homie_v2"]["episode_id"] = 0; mutations.append(c)
            c = copy.deepcopy(frozen); c["cases"][5]["configuration"]["runner"]["body"]["mobile_homie_v2"]["task_objects"]["placements"][0]["root_pose"]["position"][0] = .01; mutations.append(c)
            c = copy.deepcopy(frozen); c["cases"][0]["configuration"]["policy"]["endpoint"] = "https://remote.invalid/infer"; mutations.append(c)
            for changed in mutations:
                with self.assertRaises(ValueError):
                    suite.validate(changed, directory)
            (directory / "unitree_g1_mobile_server.py").write_text("changed after freeze")
            with self.assertRaises(ValueError):
                suite.validate(frozen, directory)

    def test_background_support_and_stale_contact_cannot_prove_held_walk(self):
        definition, rows = route_fixture()
        self.assertTrue(route_evidence(definition, rows)["all_postgrasp_pre_release_hand_only_support"])
        for contact in [{"active_solver_normal_impulse_n_s": 0., "other_robot_body_index": 1},
                        {"active_solver_normal_impulse_n_s": .02, "other_robot_body_index": None}]:
            changed = copy.deepcopy(rows)
            changed[-1]["body"]["mobile_homie_v2"]["task_objects"]["objects"][0]["last_solve_contacts"] = [contact]
            self.assertFalse(route_evidence(definition, changed)["all_postgrasp_pre_release_hand_only_support"])

    def test_midroute_fall_and_fixed_world_brace_are_remembered(self):
        definition, rows = route_fixture()
        rows[50]["body"]["mobile_homie_v2"]["root_upright_cosine"] = .9
        rows[80]["body"]["mobile_homie_v2"]["native_station_contacts"] = [{
            "other_body_fixed": True, "active_solver_normal_impulse_n_s": .1, "robot_body_index": 1}]
        result = route_evidence(definition, rows)
        self.assertFalse(result["standing_all_ticks"])
        self.assertEqual(result["nonfoot_positive_fixed_world_contact_ticks"], [81])

    def test_repeated_integrations_and_hidden_substeps_are_rejected(self):
        definition, rows = route_fixture()
        changed = copy.deepcopy(rows)
        changed[50]["body"]["mobile_homie_v2"]["torque_update_count"] = 50
        with self.assertRaises(ValueError):
            route_evidence(definition, changed)
        rows[50]["body"]["mobile_homie_v2"]["step_configuration"]["max_ccd_substeps"] = 2
        with self.assertRaises(ValueError):
            route_evidence(definition, rows)


if __name__ == "__main__":
    unittest.main()
