"""Named-order/HTTP fixtures only; no Torch, policy weights or physical evidence."""

import base64
from contextlib import closing
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unitree_g1_mobile_server as protocol


def request():
    index = 1
    groups = {}
    for name, width in protocol.GROUP_WIDTHS.items():
        groups[name] = list(range(index, index + width))
        index += width
    return {"schema": "mobile_observation_v1", "profile": "mobile_box", "sequence_id": 7,
            "observation": {"episode_id": 2, "frame_id": 5, "sim_time_ns": 40_000_000, "captured_at_unix_ms": 100},
            "camera_rgb_b64": base64.b64encode(bytes([17]) * (640 * 480 * 3)).decode(),
            "state_groups": groups}


def outputs():
    result = {}
    index = 1
    for name, width in {**protocol.GROUP_WIDTHS, "base_height_command": 1, "navigate_command": 3}.items():
        result[name] = np.broadcast_to(np.arange(index, index + width, dtype=np.float32), (1, 50, width)).copy()
        index += width
    return result


class FixturePolicy:
    def __init__(self, block=False):
        self.metadata = {"profile": "mobile_box", "revision": protocol.PROFILE["revision"],
                         "inference_backend": "protocol_fixture", "model_loaded": False,
                         "instruction_scope": protocol.INSTRUCTION_SCOPE,
                         "fixed_instruction": protocol.PROFILE["reference_instruction"]}
        self.metadata.update(initial_seed=42, post_load_rng_sha256="fixture_not_model_rng")
        self.rewinds = 0
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.observation = None

    def infer(self, observation):
        self.calls += 1
        self.observation = observation
        self.entered.set()
        if self.block and not self.release.wait(2):
            raise RuntimeError("protocol fixture gate timeout")
        return outputs()

    def rewind_initial_rng(self):
        self.rewinds += 1


class DecoderTests(unittest.TestCase):
    def test_distinct_nonzero_left_right_finger_and_waist_values_keep_order(self):
        observation = protocol.decode_request(request())
        self.assertEqual(observation["video"]["ego_view"].shape, (1, 1, 480, 640, 3))
        self.assertEqual(observation["video"]["ego_view"].dtype, np.uint8)
        self.assertTrue((observation["video"]["ego_view"] == 17).all())
        state = observation["state"]
        self.assertEqual(state["left_hand"].tolist(), [[[15., 16., 17., 18., 19., 20., 21.]]])
        self.assertEqual(state["right_hand"].tolist(), [[[22., 23., 24., 25., 26., 27., 28.]]])
        self.assertEqual(state["waist"].tolist(), [[[29., 30., 31.]]])
        self.assertEqual(np.concatenate([state[name] for name in protocol.GROUP_WIDTHS], axis=-1).flatten().tolist(), list(range(1, 32)))

    def test_only_fixed_mobile_schema_and_no_language_field_are_accepted(self):
        for update in [{"schema": "unitree_g1_static_observation_v1"}, {"profile": "static_apple"},
                       {"language_instruction": "pick an unqualified cup"}, {"instruction": "arbitrary"}]:
            body = request()
            body.update(update)
            with self.assertRaises(ValueError):
                protocol.decode_request(body)
        observation = protocol.decode_request(request())
        self.assertEqual(observation["language"], {"annotation.human.task_description": [[protocol.PROFILE["reference_instruction"]]]})

    def test_stamp_and_joint_types_shapes_and_finite_are_strict(self):
        for value in [True, -1, 2**64, "7", 7.0]:
            body = request()
            body["sequence_id"] = value
            with self.assertRaises(ValueError):
                protocol.decode_request(body)
        for value in [True, "0.1", None, float("nan"), float("inf"), 1e100]:
            body = request()
            body["state_groups"]["right_hand"][0] = value
            with self.assertRaises(ValueError):
                protocol.decode_request(body)
        body = request()
        body["state_groups"]["waist"] = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            protocol.decode_request(body)
        body = request()
        body["observation"]["extra"] = 0
        with self.assertRaises(ValueError):
            protocol.decode_request(body)

    def test_rgb_exact_bytes_and_duplicate_json_are_required(self):
        for value in [None, "", "?" * (640 * 480 * 4), request()["camera_rgb_b64"][:-4]]:
            body = request()
            body["camera_rgb_b64"] = value
            with self.assertRaises((ValueError, TypeError)):
                protocol.decode_request(body)
        with self.assertRaises(ValueError):
            protocol.parse_json(b'{"sequence_id":1,"sequence_id":2}')

    def test_decoded_50_frame_order_revision_and_tail_are_preserved(self):
        body = request()
        chunk = protocol.action_chunk(body, outputs())
        self.assertEqual(len(chunk["frames"]), 50)
        self.assertEqual(chunk["profile"], "mobile_box")
        self.assertEqual(chunk["model_revision"], protocol.PROFILE["revision"])
        self.assertEqual(chunk["action_period_ns"], 20_000_000)
        self.assertEqual(chunk["observation"], body["observation"])
        self.assertEqual(chunk["sequence_id"], 7)
        frame = chunk["frames"][0]
        self.assertEqual(frame["left_hand"], list(range(15, 22)))
        self.assertEqual(frame["right_hand"], list(range(22, 29)))
        self.assertEqual(frame["waist"], [29, 30, 31])
        self.assertEqual(frame["base_height_m"], 32)
        self.assertEqual(frame["navigate_mps_rps"], [33, 34, 35])

    def test_static_horizon_output_dtype_and_nonfinite_are_rejected(self):
        for value in [np.zeros((1, 40, 7), np.float32), np.zeros((1, 50, 7), np.float64),
                      np.full((1, 50, 7), np.inf, np.float32), [[1] * 7] * 50]:
            result = outputs()
            result["left_hand"] = value
            with self.assertRaises(ValueError):
                protocol.action_chunk(request(), result)

    def test_evidence_is_exclusive_and_fixed_scope_is_explicit(self):
        policy = FixturePolicy()
        body = request()
        observation = protocol.decode_request(body)
        actions = outputs()
        chunk = protocol.action_chunk(body, actions)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol.write_capture(directory, body, observation, actions, chunk, 0.01, policy.metadata)
            sample = directory / "e2_f5_q7"
            before = {path.name: path.read_bytes() for path in sample.iterdir()}
            with self.assertRaises(FileExistsError):
                protocol.write_capture(directory, body, observation, actions, chunk, 0.02, policy.metadata)
            self.assertEqual(before, {path.name: path.read_bytes() for path in sample.iterdir()})
            receipt = json.loads((sample / "receipt.json").read_text())
            self.assertFalse(receipt["full_forward_verified"])
            self.assertFalse(receipt["task_qualified"])
            self.assertEqual(receipt["policy_identity"]["instruction_scope"], protocol.INSTRUCTION_SCOPE)

    def test_protocol_import_does_not_import_torch_onnxruntime_or_groot(self):
        scripts = str(Path(protocol.__file__).parent)
        code = "import sys;sys.path.insert(0,sys.argv[1]);import unitree_g1_mobile_server;assert 'torch' not in sys.modules;assert 'onnxruntime' not in sys.modules;assert not any(name=='gr00t' or name.startswith('gr00t.') for name in sys.modules)"
        subprocess.run([sys.executable, "-c", code, scripts], check=True, timeout=5)

    def test_wrong_runtime_environment_fails_before_any_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                protocol.MobileBoxPolicy(root / "source", root / "model", root / "receipt.json", 1, root / "wrong_env")

    def test_wrong_receipt_identity_fails_before_source_or_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "receipt.json"
            value = {"schema": "unitree_g1_policy_files_v1", "profile": "static_apple",
                     "revision": protocol.PROFILE["revision"], "local_files_verified": True,
                     "model_root": str(root / "model")}
            receipt.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                protocol.MobileBoxPolicy(root / "source", root / "model", receipt, 1, Path(sys.prefix))
            value["profile"] = "mobile_box"
            value["local_files_verified"] = 1
            receipt.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                protocol.MobileBoxPolicy(root / "source", root / "model", receipt, 1, Path(sys.prefix))


class HttpFixtureTests(unittest.TestCase):
    def setUp(self):
        self.policy = FixturePolicy(block=True)
        self.server = protocol.BoundedPolicyServer(("127.0.0.1", 0), protocol.handler(self.policy, None))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.policy.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())

    def connection(self):
        return http.client.HTTPConnection(*self.server.server_address, timeout=2)

    def wait_idle_handlers(self):
        for _ in range(2):
            self.assertTrue(self.server.handler_slots.acquire(timeout=2))
        for _ in range(2):
            self.server.handler_slots.release()

    def call(self, body):
        with closing(self.connection()) as connection:
            connection.request("POST", "/infer", json.dumps(body), {"Content-Type": "application/json"})
            reply = connection.getresponse()
            return reply.status, json.loads(reply.read())

    def reset_episode(self, episode_id=2, seed=42, **extra):
        body = {"schema": "mobile_episode_seed_reset_v1", "episode_id": episode_id,
                "initial_seed": seed, **extra}
        with closing(self.connection()) as connection:
            connection.request("POST", "/begin_episode", json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())

    def enable_seed_gate(self):
        self.server.RequestHandlerClass = protocol.handler(self.policy, None, True)

    def test_episode_reset_is_opt_in_and_cannot_change_seed_or_reuse_identity(self):
        self.assertEqual(self.reset_episode()[0], 404)
        self.enable_seed_gate()
        for seed in [0, True, "42"]:
            self.assertEqual(self.reset_episode(seed=seed)[0], 422)
            self.wait_idle_handlers()
        self.assertEqual(self.reset_episode(invisible_object_position=[0, 0, 0])[0], 422)
        self.wait_idle_handlers()
        self.assertEqual(self.policy.rewinds, 0)
        status, result = self.reset_episode()
        self.assertEqual(status, 200)
        self.assertEqual(result["successful_inferences"], 0)
        self.assertEqual(result["episode_seed_resets"], 1)
        self.wait_idle_handlers()
        self.assertEqual(self.reset_episode()[0], 409)
        self.assertEqual(self.policy.rewinds, 1)

    def test_old_episode_and_inflight_reset_are_rejected_without_changing_rng(self):
        self.enable_seed_gate()
        self.assertEqual(self.call(request())[0], 422)
        self.wait_idle_handlers()
        self.assertEqual(self.reset_episode()[0], 200)
        self.wait_idle_handlers()
        completed = []
        runner = threading.Thread(target=lambda: completed.append(self.call(request())))
        runner.start()
        try:
            self.assertTrue(self.policy.entered.wait(2))
            self.assertEqual(self.reset_episode(3)[0], 409)
            self.assertEqual(self.policy.rewinds, 1)
        finally:
            self.policy.release.set()
            runner.join(2)
        self.wait_idle_handlers()
        self.assertEqual(completed[0][0], 200)
        status, result = self.reset_episode(3)
        self.assertEqual(status, 200)
        self.assertEqual(result["successful_inferences"], 1)
        self.wait_idle_handlers()
        self.assertEqual(self.call(request())[0], 422)
        self.assertEqual(self.policy.calls, 1)

    def test_single_inference_fixed_health_and_actual_counter(self):
        completed = []
        runner = threading.Thread(target=lambda: completed.append(self.call(request())))
        runner.start()
        try:
            self.assertTrue(self.policy.entered.wait(2))
            self.assertEqual(self.call(request())[0], 409)
            self.assertTrue(self.server.handler_slots.acquire(timeout=2))
            self.server.handler_slots.release()
            with closing(self.connection()) as connection:
                connection.request("GET", "/health")
                health = json.loads(connection.getresponse().read())
            self.assertEqual(health["successful_inferences"], 0)
            self.assertEqual(health["fixed_instruction"], protocol.PROFILE["reference_instruction"])
            self.assertFalse(health["model_loaded"])
            self.assertFalse(health["task_qualified"])
            self.policy.release.set()
            runner.join(2)
            self.assertEqual(completed[0][0], 200)
            self.assertEqual(self.policy.calls, 1)
        finally:
            self.policy.release.set()
            runner.join(2)

    def test_wrong_profile_or_instruction_injection_never_enters_model(self):
        for update in [{"profile": "static_apple"}, {"language_instruction": "arbitrary"}]:
            body = request()
            body.update(update)
            self.assertEqual(self.call(body)[0], 422)
            self.wait_idle_handlers()
        self.assertEqual(self.policy.calls, 0)

    def test_http_size_type_and_transfer_encoding_are_bounded(self):
        for headers, expected in [({"Content-Length": "2097153"}, 413),
                                  ({"Content-Length": "2", "Content-Type": "text/plain"}, 415),
                                  ({"Content-Length": "2", "Transfer-Encoding": "chunked"}, 400)]:
            with closing(self.connection()) as connection:
                connection.request("POST", "/infer", "{}", headers)
                reply = connection.getresponse()
                self.assertEqual(reply.status, expected)
                reply.read()
            self.wait_idle_handlers()
        self.assertEqual(self.policy.calls, 0)


if __name__ == "__main__":
    unittest.main()
