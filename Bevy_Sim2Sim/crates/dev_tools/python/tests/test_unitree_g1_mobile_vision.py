"""Finite visual planning regressions; fixtures do not prove physical success."""
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unitree_g1_mobile_vision as vision


def fine_proposal(box_x, box_y=0., half_width=.12):
    # Minimise the failed whole-object interval to its two horizontal axes.
    box = [[x, y, 0.] for x, y in itertools.product(
        [-half_width, half_width], [-half_width, half_width])]
    floor = [[x, y, 0.] for x, y in itertools.product([-.2, .2], [-.2, .2])]
    geometry = {"objects": [
        {"kind": "t2_box", "convex_parts": [{"points": box}]},
        {"kind": "t2_bin", "convex_parts": [None, None, {"points": floor}]},
    ]}
    bin_pose, box_pose = np.eye(4), np.eye(4)
    box_pose[:2, 3] = [box_x, box_y]
    detections = [{"marker_id": marker, "root_from_marker": pose.tolist()}
                  for marker, pose in [(21, bin_pose), (22, box_pose)]]
    observation = {"stamp": {"episode_id": 1, "frame_id": 1,
                             "sim_time_ns": 20_000_000, "captured_at_unix_ms": 1},
                   "measured_joints": {"root_rotation_wxyz": [1., 0., 0., 0.]}}
    with patch.object(vision, "public_json", return_value=geometry):
        return vision.fine_from_visible_markers(
            detections, observation, Path("public_geometry_fixture"),
            {21: np.eye(4), 22: np.eye(4)}, 0.)


class FinePlanningTests(unittest.TestCase):
    def test_far_containment_is_not_erased_by_a_single_step_horizon(self):
        proposal = fine_proposal(-.79)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["state"], "advance")
        np.testing.assert_allclose(proposal["physical_containment_distance_interval_m"], [.73, .85])
        self.assertEqual(proposal["selected_physical_distance_m"], .15)
        self.assertEqual(proposal["goal"]["relative_distance_m"], .1)
        self.assertFalse(proposal["step_is_final"])
        self.assertTrue(proposal["new_image_required_after_step"])

    def test_clipping_cannot_turn_a_wide_actual_interval_into_a_narrow_one(self):
        proposal = fine_proposal(-.723)
        self.assertEqual(proposal["state"], "advance")
        np.testing.assert_allclose(proposal["physical_containment_distance_interval_m"], [.663, .783])

    def test_actual_narrow_interval_and_lateral_miss_still_cannot_advance(self):
        self.assertEqual(fine_proposal(-.3, half_width=.17)["state"], "blocked")
        self.assertIsNone(fine_proposal(-.4, box_y=.1))

    def test_five_existing_steps_cannot_plan_unbounded_travel(self):
        self.assertIsNone(fine_proposal(-.9))

    def test_aligned_state_does_not_admit_another_step(self):
        proposal = fine_proposal(0.)
        self.assertEqual(proposal["state"], "aligned")
        self.assertIsNone(proposal["goal"])
        self.assertIsNone(proposal["selected_physical_distance_m"])


if __name__ == "__main__":
    unittest.main()
