"""Session isolation and owned cleanup tests; no simulated task successes."""
import copy
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unitree_g1_station_session as session


def configuration(profile):
    value = {"runner": {"body": {"static_agile" if profile == "static" else
                               "mobile_homie_v2": {"episode_id": 17}}},
             "policy": {"endpoint": "original", "max_calls": 1 if profile == "static" else 4}}
    if profile == "static":
        value.update(static_startup={"source": "original"},
                     static_grasp_fixed_camera_pair={"source": "actual_rgb"},
                     static_pair_program={"source": "disclosed_classical"})
    else:
        value["diagnostic_qwen_dispatch"] = {
            "scope": "scientific_station_mobile_from_instruction_v1",
            "connection": {"endpoint": "http://127.0.0.1:8002/v1"}}
    return value


class StationSessionTests(unittest.TestCase):
    def test_one_process_cannot_take_resources_while_a_peer_holds_the_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.lock"
            descriptor = session.acquire_session_lock(path)
            try:
                probe = subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0,sys.argv[1]); "
                    "import unitree_g1_station_session as s; s.acquire_session_lock(sys.argv[2])",
                    str(Path(session.__file__).parent), str(path)], capture_output=True, text=True)
                self.assertNotEqual(probe.returncode, 0)
                self.assertIn("Another G1 station session", probe.stderr)
            finally:
                os.close(descriptor)
            descriptor = session.acquire_session_lock(path)
            os.close(descriptor)
            link = Path(temporary) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                session.acquire_session_lock(link)

    def test_restart_with_same_label_cannot_be_stopped_as_the_original_session(self):
        receipt = {"qwen_identity": {"Id": "ours", "State": {"StartedAt": "first"}}}
        changed = {"Id": "ours", "State": {"Running": True, "StartedAt": "another-start"}}
        with patch.object(session, "inspect_qwen", return_value=changed), \
                patch.object(session, "close_owned", return_value=0) as close, \
                patch.object(session.subprocess, "run") as stop:
            errors = session.cleanup_session(receipt, object(), None, None, None, True, Path("service"))
        stop.assert_not_called()
        self.assertTrue(errors)
        self.assertFalse(receipt["all_owned_handles_reaped"])
        self.assertEqual(close.call_count, 4)

    def test_static_qwen_gate_cannot_modify_the_original_body_or_action_contract(self):
        original = configuration("static")
        snapshot = copy.deepcopy(original)
        prepared = session.prepare_configuration(original, "static", 5557, 2, True,
                                                  with_local_qwen=True)
        self.assertEqual(original, snapshot)
        self.assertEqual(prepared["runner"], snapshot["runner"])
        self.assertEqual(prepared["policy"]["max_calls"], 1)
        gate = prepared["diagnostic_qwen_dispatch"]
        self.assertEqual(gate["scope"], "scientific_station_static_from_instruction_v1")
        self.assertEqual(gate["connection"]["max_output_tokens"], 128)
        self.assertEqual(gate["connection"]["endpoint"], "http://127.0.0.1:8002/v1")
        with self.assertRaises(ValueError):
            session.prepare_configuration(prepared, "static", 5557, 2, True)

    def test_model_python_keeps_its_actual_virtual_environment_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "model_env"
            (environment / "bin").mkdir(parents=True)
            executable = environment / "bin/python"
            executable.symlink_to(sys.executable)
            (environment / "pyvenv.cfg").write_text(
                f"home = {Path(sys.executable).resolve().parent}\ninclude-system-site-packages = true\n")
            command = session.argument_path("policy_python", executable)
            actual = subprocess.check_output([str(command), "-c", "import sys; print(sys.prefix)"], text=True).strip()
            self.assertEqual(Path(actual), environment)

    def test_preparation_preserves_input_and_distinct_original_contract(self):
        for profile in ["static", "mobile"]:
            original = configuration(profile)
            snapshot = copy.deepcopy(original)
            prepared = session.prepare_configuration(original, profile, 5559, 4, False)
            self.assertEqual(original, snapshot)
            self.assertEqual(prepared["runner"], snapshot["runner"])
            self.assertEqual(prepared["policy"]["max_calls"], snapshot["policy"]["max_calls"])
            self.assertEqual(prepared["policy"]["endpoint"], "http://127.0.0.1:5559/infer")
            with self.assertRaises(ValueError):
                session.prepare_configuration(original, "mobile" if profile == "static" else "static", 5559, 4, False)

    def test_rejects_task_mixing_changed_horizons_and_unbounded_episodes(self):
        for field, replacement in [("task_lab", {}), ("local_model_startup", {}),
                                   ("policy", {"max_calls": 5})]:
            value = configuration("mobile")
            value[field] = replacement
            with self.assertRaises(ValueError):
                session.prepare_configuration(value, "mobile", 5558, 4, False)
        for maximum in [0, 1, 11]:
            with self.assertRaises(ValueError):
                session.prepare_configuration(configuration("static"), "static", 5557, maximum, False)

    def test_probe_does_not_take_over_a_live_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaises(OSError):
                session.check_unused_port(port)
        session.check_unused_port(port)

    def test_inflight_reset_requires_the_separate_mobile_smoke_route(self):
        for profile, smoke in [("static", True), ("mobile", False)]:
            with self.assertRaises(ValueError):
                session.prepare_configuration(configuration(profile), profile, 5558, 2, smoke, True)
        value = session.prepare_configuration(configuration("mobile"), "mobile", 5558, 2, True, True)
        self.assertTrue(value["station_task_controls"]["smoke_initial_inflight_reset"])

    def test_docker_failure_still_reaps_model_and_never_claims_cleanup(self):
        model, app, recorder, starter = [object() for _ in range(4)]
        receipt = {}
        with patch.object(session, "close_owned", return_value=0) as close, \
                patch.object(session, "inspect_qwen", side_effect=OSError("Docker unavailable")), \
                patch.object(session.subprocess, "run") as stop:
            errors = session.cleanup_session(receipt, model, app, recorder, starter, True, Path("owner-service"))
        self.assertEqual([call.args[0] for call in close.call_args_list], [recorder, app, starter, model])
        stop.assert_not_called()
        self.assertTrue(errors)
        self.assertFalse(receipt["all_owned_handles_reaped"])

    def test_unstarted_qwen_is_never_inspected_or_stopped(self):
        receipt = {}
        with patch.object(session, "close_owned", return_value=0), \
                patch.object(session, "inspect_qwen") as inspect, \
                patch.object(session.subprocess, "run") as stop:
            self.assertEqual(session.cleanup_session(receipt, None, None, None, None, False, Path("service")), [])
        inspect.assert_not_called()
        stop.assert_not_called()
        self.assertTrue(receipt["all_owned_handles_reaped"])

    def test_child_exit_signal_race_is_reaped(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.return_value = 0
        with patch.object(session.os, "killpg", side_effect=ProcessLookupError):
            self.assertEqual(session.close_owned(child), 0)
        child.wait.assert_called()


if __name__ == "__main__":
    unittest.main()
