"""The diagnostic execution receipt must not hide a failed Idle behavior."""

import copy
import unittest

from bevy_microduck_tools.legacy_idle_gate import evaluate


def report(rotations: list[list[float]], final_x: float = 0.0) -> dict:
    frames = []
    for step, rotation in enumerate(rotations):
        frames.append({"global_step": step, "poses": [{"source_body_id": 1,
                       "translation": [final_x if step == len(rotations)-1 else 0.0, .125, 0.0],
                       "rotation_xyzw": rotation}]})
    return {"schema": "station_robot_60hz_diagnostic_v1", "legacy_model_name": "alpha_stand",
            "physics_hz": 60, "policy_hz": 60, "integration_count": 240,
            "policy_inference_count": 240, "passed": True,
            "final_clock": {"global_step": 240}, "pose_frames": frames}


class LegacyIdleGateTests(unittest.TestCase):
    def test_stable_full_horizon_passes_historical_threshold_only(self):
        result = evaluate(report([[0, 0, 0, 1]] * 241), "source-sha")
        self.assertTrue(result["behavior_passed"])
        self.assertFalse(result["skill_qualified"])

    def test_fall_fails_even_when_execution_report_passes(self):
        frames = [[0, 0, 0, 1]] * 114 + [[0.8, 0, 0, 0.6]] * 127
        result = evaluate(report(frames), "source-sha")
        self.assertFalse(result["behavior_passed"])
        self.assertEqual(result["first_upright_below_0_5_step"], 114)

    def test_incomplete_horizon_and_large_drift_fail(self):
        incomplete = report([[0, 0, 0, 1]] * 64)
        self.assertFalse(evaluate(incomplete, "source-sha")["behavior_passed"])
        self.assertFalse(evaluate(report([[0, 0, 0, 1]] * 241, .21), "source-sha")["behavior_passed"])

    def test_changed_pose_step_is_rejected(self):
        broken = copy.deepcopy(report([[0, 0, 0, 1]] * 241))
        broken["pose_frames"][19]["global_step"] = 20
        with self.assertRaisesRegex(ValueError, "different global_step"):
            evaluate(broken, "source-sha")


if __name__ == "__main__":
    unittest.main()
