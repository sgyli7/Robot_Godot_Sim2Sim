"""CPU protocol fixtures only; no model weights or task-performance evidence."""

import base64
from contextlib import closing
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unitree_g1_static_server as protocol


def request():
    return {"schema": "unitree_g1_static_observation_v1", "profile": "static_apple", "sequence_id": 7,
            "observation": {"episode_id": 2, "frame_id": 5, "sim_time_ns": 40_000_000, "captured_at_unix_ms": 100},
            "camera_rgb_b64": base64.b64encode(bytes([17]) * (640 * 480 * 3)).decode(),
            "state_groups": {name: [0.1] * width for name, width in protocol.GROUP_WIDTHS.items()}}


def outputs():
    widths = {**protocol.GROUP_WIDTHS, "base_height_command": 1, "navigate_command": 3}
    return {name: np.full((1, 40, width), 0.5, dtype=np.float32) for name, width in widths.items()}


class FixturePolicy:
    provider = "CPUExecutionProvider"

    def __init__(self, block=False):
        self.sessions = {name: object() for name in protocol.FULL_GRAPH_NAMES}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.calls = 0
        self.inputs = None

    def infer(self, observation, seed):
        self.calls += 1
        self.inputs = observation
        self.entered.set()
        if self.block and not self.release.wait(2):
            raise RuntimeError("fixture inference gate timed out")
        return outputs(), {name: 0.001 for name in protocol.FULL_GRAPH_NAMES}


class DecoderTests(unittest.TestCase):
    def test_real_byte_layout_and_joint_groups_are_preserved(self):
        observation = protocol.decode_request(request())
        self.assertEqual(observation["ego_view"].shape, (1, 480, 640, 3))
        self.assertEqual(observation["ego_view"].dtype, np.float32)
        self.assertTrue((observation["ego_view"] == 17).all())
        self.assertEqual(observation["left_arm"].shape, (1, 7))
        self.assertAlmostEqual(float(observation["left_arm"][0, 0]), 0.1)

    def test_request_and_stamp_are_exact_objects(self):
        for change in [lambda x: x.update(extra=1), lambda x: x.update(profile="mobile_box"),
                       lambda x: x.update(observation=[]), lambda x: x["observation"].update(extra=1)]:
            body = request()
            change(body)
            with self.assertRaises(ValueError):
                protocol.decode_request(body)
        for value in [True, 1.0, "1", -1, 2**64]:
            body = request()
            body["observation"]["frame_id"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.decode_request(body)

    def test_joint_strings_bools_nonfinite_and_wrong_shape_are_rejected(self):
        for value in [True, "0.1", None, float("nan"), float("inf"), 1e100]:
            body = request()
            body["state_groups"]["left_arm"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.decode_request(body)
        for values in [[0.1] * 6, [[0.1] * 7], 0.1]:
            body = request()
            body["state_groups"]["left_arm"] = values
            with self.assertRaises(ValueError):
                protocol.decode_request(body)

    def test_rgb_requires_exact_base64_bytes(self):
        for value in [None, "", request()["camera_rgb_b64"][:-4], "?" * (640 * 480 * 4)]:
            body = request()
            body["camera_rgb_b64"] = value
            with self.assertRaises((ValueError, TypeError)):
                protocol.decode_request(body)

    def test_json_duplicate_fields_and_nonfinite_constants_are_rejected(self):
        for payload in [b'{"sequence_id":1,"sequence_id":2}', b'{"value":NaN}', b'{"value":Infinity}']:
            with self.assertRaises(ValueError):
                protocol.parse_json(payload)

    def test_output_shape_dtype_and_finite_are_checked_before_encoding(self):
        valid = protocol.action_chunk(request(), outputs())
        self.assertEqual(len(valid["frames"]), 40)
        self.assertEqual(valid["action_period_ns"], 20_000_000)
        self.assertEqual(valid["observation"], request()["observation"])
        self.assertEqual(valid["sequence_id"], 7)
        self.assertEqual(valid["model_revision"], protocol.PROFILES["static_apple"]["revision"])
        for value in [np.zeros((1, 39, 7), np.float32), np.zeros((1, 40, 7), np.float64),
                      np.full((1, 40, 7), np.nan, np.float32), [[0.0] * 7] * 40]:
            result = outputs()
            result["left_arm"] = value
            with self.assertRaises(ValueError):
                protocol.action_chunk(request(), result)
        result = outputs()
        result["extra"] = np.zeros(1)
        with self.assertRaises(ValueError):
            protocol.action_chunk(request(), result)

    def test_small_only_policy_cannot_be_served(self):
        policy = FixturePolicy()
        del policy.sessions["backbone"]
        with self.assertRaises(ValueError):
            protocol.handler(policy, None)

    def test_evidence_write_is_exclusive_and_includes_identity_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            body = request()
            observation = protocol.decode_request(body)
            result = outputs()
            chunk = protocol.action_chunk(body, result)
            timings = {name: 0.001 for name in protocol.FULL_GRAPH_NAMES}
            protocol.write_capture(directory, body, observation, result, chunk, timings, 0.01)
            sample = directory / "e2_f5_q7"
            before = {path.name: path.read_bytes() for path in sample.iterdir()}
            with self.assertRaises(FileExistsError):
                protocol.write_capture(directory, body, observation, result, chunk, timings, 0.02)
            self.assertEqual(before, {path.name: path.read_bytes() for path in sample.iterdir()})
            receipt = json.loads((sample / "receipt.json").read_text())
            self.assertEqual(receipt["sequence_id"], 7)
            self.assertEqual(receipt["observation_sha256"], protocol.sha256(sample / "observation.npz"))
            self.assertFalse(receipt["task_qualified"])

    def test_receipt_guards_full_graph_hashes_and_external_data_without_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export = directory / protocol.EXPORT_PATH
            export.mkdir(parents=True)
            for name in ["graph.yaml", *(f"{name}.onnx" for name in protocol.FULL_GRAPH_NAMES), "backbone.onnx.data"]:
                (export / name).write_bytes(b"CPU protocol fixture, never a model")
            receipt = {"schema": "unitree_g1_policy_files_v1", "profile": "static_apple",
                       "revision": protocol.PROFILES["static_apple"]["revision"],
                       "local_files_verified": True, "artifact_contract": {"status": "interface_matched"},
                       "model_root": str(directory),
                       "files": [{"path": str(path.relative_to(directory)), "sha256": protocol.sha256(path)}
                                 for path in export.iterdir()]}
            path = directory / "receipt.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(protocol.verify_receipt(path), directory.resolve())
            changes = [lambda value: value.update(revision="unfrozen"),
                       lambda value: value.update(local_files_verified=1),
                       lambda value: value["files"].append(dict(value["files"][0])),
                       lambda value: value["files"].pop(),
                       lambda value: value["files"][0].update(path="../outside.onnx"),
                       lambda value: value["files"][0].update(sha256="0" * 64)]
            for change in changes:
                value = json.loads(json.dumps(receipt))
                change(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    protocol.verify_receipt(path)


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

    def call(self, body):
        with closing(self.connection()) as connection:
            connection.request("POST", "/infer", json.dumps(body), {"Content-Type": "application/json"})
            reply = connection.getresponse()
            return reply.status, json.loads(reply.read())

    def test_single_inference_and_health_reports_actual_counter(self):
        completed = []
        runner = threading.Thread(target=lambda: completed.append(self.call(request())))
        runner.start()
        try:
            self.assertTrue(self.policy.entered.wait(2))
            status, _ = self.call(request())
            self.assertEqual(status, 409)
            # The 409 handler releases its slot after writing the response.
            # Wait for that explicit gate before opening the health connection.
            self.assertTrue(self.server.handler_slots.acquire(timeout=2))
            self.server.handler_slots.release()
            with closing(self.connection()) as connection:
                connection.request("GET", "/health")
                reply = connection.getresponse()
                health = json.loads(reply.read())
            self.assertEqual(health["successful_inferences"], 0)
            self.assertFalse(health["task_qualified"])
            self.policy.release.set()
            runner.join(2)
            self.assertFalse(runner.is_alive())
            self.assertEqual(completed[0][0], 200)
            self.assertEqual(self.policy.calls, 1)
            with closing(self.connection()) as connection:
                connection.request("GET", "/health")
                reply = connection.getresponse()
                health = json.loads(reply.read())
            self.assertEqual(health["successful_inferences"], 1)
        finally:
            self.policy.release.set()
            runner.join(2)

    def test_malformed_input_does_not_enter_inference(self):
        body = request()
        body["state_groups"]["waist"][0] = True
        status, error = self.call(body)
        self.assertEqual(status, 422)
        self.assertIn("error", error)
        self.assertEqual(self.policy.calls, 0)

    def test_request_size_content_type_and_transfer_encoding_are_bounded(self):
        for headers, expected in [({"Content-Length": "2097153"}, 413),
                                  ({"Content-Length": "2", "Content-Type": "text/plain"}, 415),
                                  ({"Content-Length": "2", "Transfer-Encoding": "chunked"}, 400)]:
            with closing(self.connection()) as connection:
                connection.request("POST", "/infer", "{}", headers)
                reply = connection.getresponse()
                self.assertEqual(reply.status, expected)
                reply.read()
            # Reading the reply can precede the handler's finally/slot release.
            # Close both explicit handler gates before the next sequential probe.
            for _ in range(2):
                self.assertTrue(self.server.handler_slots.acquire(timeout=2))
            for _ in range(2):
                self.server.handler_slots.release()
        self.assertEqual(self.policy.calls, 0)


if __name__ == "__main__":
    unittest.main()
