import itertools
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from unitree_g1_mobile_hand_geometry import separation
from unitree_g1_mobile_vision import localize


class OriginalHullSeparation(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray(list(itertools.product([-.03, .03], repeat=3)))

    def test_disjoint_complete_hulls_have_a_checked_support_plane(self):
        pose = np.eye(4)
        pose[0, 3] = .14
        result = separation(self.points, pose, 1.)
        self.assertAlmostEqual(result["linf_distance_m"], .01, places=10)
        self.assertAlmostEqual(result["certified_separation_m"], .01, places=10)
        self.assertEqual(result["complete_original_vertex_count"], 8)
        self.assertFalse(result["physical_contact_force_measured"])

    def test_a_single_original_vertex_can_change_the_whole_hull_verdict(self):
        pose = np.eye(4)
        pose[0, 3] = .14
        with_original_vertex = np.vstack([self.points, [-.06, 0., 0.]])
        result = separation(with_original_vertex, pose, 1.)
        self.assertEqual(result["linf_distance_m"], 0.)
        self.assertEqual(result["complete_original_vertex_count"], 9)

    def test_overlap_is_geometry_and_does_not_measure_force(self):
        result = separation(self.points, np.eye(4), 1.)
        self.assertEqual(result["linf_distance_m"], 0.)
        self.assertEqual(result["certified_separation_m"], 0.)
        self.assertFalse(result["physical_contact_force_measured"])

    def test_invalid_pose_vertices_or_expired_solve_never_return_a_candidate(self):
        for pose in [np.diag([-1., 1., 1., 1.]), np.ones((4, 4)), np.eye(3)]:
            with self.assertRaises(ValueError):
                separation(self.points, pose, 1.)
        for points, remaining in [(self.points*np.nan, 1.), (self.points, 0.), (self.points, float("nan")), (self.points, float("inf"))]:
            with self.assertRaises(ValueError):
                separation(points, np.eye(4), remaining)

    @unittest.skipUnless(os.environ.get("G1_GRASP_RECORDING_FIXTURE"), "requires actual recorded RGB fixture")
    def test_actual_post_lift_box_only_pair_can_measure_hands_without_bin_navigation(self):
        case = Path(os.environ["G1_GRASP_RECORDING_FIXTURE"])
        config = json.loads((case/"config.json").read_text())
        vision = config["mobile_scan"]["vision"]
        pair = case/"native/station_pickup_after"
        found = localize(pair/"ego.png", pair/"observation.json",
                         Path(config["runner"]["body"]["mobile_homie_v2"]["definition"]),
                         geometry_path=Path(vision["task_geometry"]["path"]),
                         fiducial_path=Path(vision["fiducial_calibration"]["path"]),
                         secondary_image_path=pair/"secondary/ego.png",
                         secondary_observation_path=pair/"secondary/observation.json",
                         held_box_feedback=True, held_contact_geometry=True)
        report = found["held_contact_geometry"]
        self.assertEqual(len(report["bodies"]), 16)
        self.assertFalse(report["holding_proven"])
        self.assertFalse(report["owner_execution_admitted"])
        self.assertIsNone(found["navigation_proposal"])
        self.assertEqual({b["body_index"] for b in report["bodies"]},
                         set(range(28, 36)) | set(range(45, 53)))
        self.assertEqual(next(b for b in report["bodies"] if b["body_index"] == 28)["complete_original_vertex_count"], 21392)

    def test_report_cannot_be_requested_without_current_paired_feedback(self):
        with self.assertRaisesRegex(ValueError, "current paired held feedback"):
            localize(None, None, None, held_contact_geometry=True)


if __name__ == "__main__":
    unittest.main()
