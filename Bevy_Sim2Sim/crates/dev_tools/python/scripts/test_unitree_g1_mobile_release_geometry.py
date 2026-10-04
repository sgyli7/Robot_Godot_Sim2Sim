"""Convex release clearance at the existing 35cm/10mm contract boundary."""
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    import scipy.optimize
    import unitree_g1_mobile_vision as vision
except ImportError:
    np = None


@unittest.skipIf(np is None, "native vision NumPy/SciPy/OpenCV environment required")
class ReleaseGeometryTests(unittest.TestCase):
    def release(self, hand_center, box_angle=0.):
        frames = [np.eye(4) for _ in range(53)]
        frames[28][:3, 3] = [.12, 0., 0.]
        frames[45][:3, 3] = [-.12, 0., 0.]
        bodies = [{"name": "unused"} for _ in range(53)]
        bodies[28]["name"] = "left_hand_palm_link"
        bodies[45]["name"] = "right_hand_palm_link"
        small = np.asarray(list(itertools.product([-.005, .005], repeat=3)))
        center = np.asarray(hand_center)
        collisions = [{"body": 28, "shape": "convex_hull",
                       "points": (small + center - frames[28][:3, 3]).tolist()},
                      {"body": 45, "shape": "convex_hull",
                       "points": (small - center - frames[45][:3, 3]).tolist()}]
        box = np.asarray(list(itertools.product([-.1, .1], repeat=3)))
        pose = np.eye(4)
        c, s = np.cos(box_angle), np.sin(box_angle)
        pose[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        observation = {"measured_joints": {"positions": [0.] * 43,
                                           "root_rotation_wxyz": [1., 0., 0., 0.]}}
        with patch.object(vision, "original_self_body_frames", return_value=frames):
            return vision.horizontal_release_clearance(
                {"bodies": bodies, "collisions": collisions}, observation, pose, box)

    def test_rotated_box_empty_rectangle_corner_does_not_require_extra_opening(self):
        result = self.release([.07, .13, 0.], np.pi / 4)
        self.assertTrue(result["hand_clearance_admitted"])
        self.assertLessEqual(result["required_measured_palm_gap_m"], .35)
        self.assertEqual(result["minimum_hand_clearance_m"], .01)

    def test_five_mm_separation_still_rejects_at_existing_maximum_gap(self):
        result = self.release([.055, 0., 0.])
        self.assertFalse(result["hand_clearance_admitted"])
        self.assertGreater(result["required_measured_palm_gap_m"], .35)

    def test_contact_at_maximum_opening_cannot_pass(self):
        result = self.release([.04, 0., 0.])
        self.assertFalse(result["hand_clearance_admitted"])

    def test_optimizer_candidate_cannot_inflate_actual_vertex_support_clearance(self):
        # A claimed successful but infeasible closest point is not a geometric
        # certificate. Full original vertex support must still reject 5mm.
        with patch("scipy.optimize.minimize", return_value=SimpleNamespace(
                success=True, x=np.array([-99., .3, .4]))):
            self.assertFalse(self.release([.055, 0., 0.])["hand_clearance_admitted"])

    def test_original_limits_and_prediction_scope_are_retained(self):
        result = self.release([.1, 0., 0.])
        self.assertTrue(result["hand_clearance_admitted"])
        self.assertEqual(result["maximum_commanded_palm_gap_m"], .35)
        self.assertEqual(result["opening_profile"], "gravity_horizontal_original_fingers")
        self.assertTrue(result["prediction_is_not_physical_detachment"])


if __name__ == "__main__":
    unittest.main()
