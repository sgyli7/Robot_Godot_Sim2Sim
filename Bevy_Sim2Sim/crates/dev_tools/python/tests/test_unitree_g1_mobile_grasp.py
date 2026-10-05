import itertools
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from unitree_g1_mobile_grasp import minimum_common_insertion
from unitree_g1_mobile_vision import localize


def hull(low, high):
    points = np.asarray(list(itertools.product(*zip(low, high))))
    return ConvexHull(points).equations


class OriginalThumbInsertionTests(unittest.TestCase):
    def test_one_translation_must_reach_both_independent_thumbs(self):
        planes = [hull([-.13, -.03, -.01], [-.115, -.01, .01]),
                  hull([-.145, .01, -.01], [-.135, .03, .01])]
        shift, witnesses = minimum_common_insertion(planes, [1., 0., 0.])
        self.assertAlmostEqual(shift, .035)
        # Independently check both original hand half spaces and box bounds.
        for hand, point in zip(planes, np.asarray(witnesses)):
            self.assertLessEqual(float((hand[:, :3] @ (point-[shift, 0., 0.])+hand[:, 3]).max()), 1e-8)
            self.assertLessEqual(float(np.abs(point).max()), .1+1e-8)

    def test_unreachable_second_thumb_rejects_clipping(self):
        planes = [hull([-.13, -.03, -.01], [-.115, -.01, .01]),
                  hull([-.17, .01, -.01], [-.16, .03, .01])]
        self.assertIsNone(minimum_common_insertion(planes, [1., 0., 0.]))

    def test_foreign_nonfinite_or_nonunit_axis_is_rejected(self):
        planes = [hull([-.13, -.03, -.01], [-.115, -.01, .01])]*2
        for direction in [[float("nan"), 0., 0.], [2., 0., 0.], [0., 0.]]:
            with self.assertRaises(ValueError):
                minimum_common_insertion(planes, direction)

    def test_closed_recovery_cannot_use_unpaired_or_navigation_ingress(self):
        with self.assertRaisesRegex(ValueError, "closed regrasp requires current paired"):
            localize(Path("a"), Path("b"), Path("d"), closed_regrasp=True)

    def test_closed_recovery_cannot_relabel_an_open_or_lifted_frame(self):
        from unitree_g1_mobile_grasp import propose
        for tick in [200, 250, 350, 450, 501, 801, 1001]:
            observation = {"stamp": {"episode_id":7,"frame_id":9,"sim_time_ns":tick*20_000_000,"captured_at_unix_ms":123}}
            with self.assertRaisesRegex(ValueError, "closed300/650/1000Tick"):
                propose({}, observation, None, {}, closed_regrasp=True)
        observation = {"stamp": {"episode_id":7,"frame_id":9,"sim_time_ns":550*20_000_000,"captured_at_unix_ms":123}}
        with self.assertRaisesRegex(ValueError, "pre-lift200..300Tick"):
            propose({}, observation, None, {})

    def test_opened_forecast_cannot_consume_foreign_or_relabelled_closed_self(self):
        from unitree_g1_mobile_grasp import propose
        current = {"stamp": {"episode_id":7,"frame_id":20,"sim_time_ns":450*20_000_000,"captured_at_unix_ms":456}}
        for episode, tick, frame, wall in [(8,300,10,123),(7,350,10,123),(7,300,20,123),(7,300,10,456)]:
            reference={"stamp":{"episode_id":episode,"frame_id":frame,"sim_time_ns":tick*20_000_000,"captured_at_unix_ms":wall}}
            with self.assertRaisesRegex(ValueError,"own actual closed300/650"):
                propose({},current,None,{},closed_reference=reference)
        reference={"stamp":{"episode_id":7,"frame_id":10,"sim_time_ns":300*20_000_000,"captured_at_unix_ms":123}}
        before = json.dumps(current,sort_keys=True)
        with self.assertRaisesRegex(ValueError,"nonrigid"):
            propose({},current,None,{},closed_reference=reference)
        self.assertEqual(json.dumps(current,sort_keys=True),before)

    def test_lift_feedback_cannot_use_unpaired_or_forecast_authority(self):
        for kwargs in [{}, {"held_contact_geometry":True,"held_box_feedback":True,"closed_regrasp":True}]:
            with self.assertRaisesRegex(ValueError,"lift feedback requires explicit current paired"):
                localize(Path("a"),Path("b"),Path("d"),lift_feedback=True,**kwargs)

    def test_unpaired_or_memory_grasp_mode_fails_before_loading_files(self):
        for kwargs in [{}, {"secondary_image_path": Path("b"), "secondary_observation_path": Path("c"),
                            "geometry_path": Path("g"), "memory_path": Path("m")}]:
            with self.assertRaisesRegex(ValueError, "grasp centering requires current paired"):
                localize(Path("a"), Path("a"), Path("d"), grasp_centering=True, **kwargs)

    @unittest.skipUnless(os.environ.get("G1_GRASP_RECORDING_FIXTURE"),
                         "actual paired pre-lift RGB fixture must be explicitly supplied")
    def test_recorded_box_only_pre_lift_needs_no_bin_navigation_board(self):
        case = Path(os.environ["G1_GRASP_RECORDING_FIXTURE"])
        config = json.loads((case/"config.json").read_text())
        body = config["runner"]["body"]["mobile_homie_v2"]
        vision = config["mobile_scan"]["vision"]
        views = case/"native/station_pickup_before"
        result = localize(views/"ego.png", views/"observation.json", Path(body["definition"]),
                          geometry_path=Path(vision["task_geometry"]["path"]),
                          fiducial_path=Path(vision["fiducial_calibration"]["path"]),
                          secondary_image_path=views/"secondary/ego.png",
                          secondary_observation_path=views/"secondary/observation.json", grasp_centering=True)
        original = json.loads((views/"localization.json").read_text())
        self.assertEqual(result["same_tick_box_pair_pose"], original["same_tick_box_pair_pose"])
        proposal = result["grasp_centering_proposal"]
        self.assertGreater(proposal["common_forward_translation_m"], .041)
        self.assertLess(proposal["common_forward_translation_m"], .042)
        self.assertFalse(proposal["owner_execution_admitted"])
        self.assertFalse(proposal["physical_contact_forces_proven"])
        self.assertIsNone(result["navigation_proposal"])
        self.assertIsNone(result.get("clearance_proposal"))
        later = case/"native/station_pickup_after"
        with self.assertRaisesRegex(ValueError, "pre-lift200..300Tick"):
            localize(later/"ego.png", later/"observation.json", Path(body["definition"]),
                     geometry_path=Path(vision["task_geometry"]["path"]),
                     fiducial_path=Path(vision["fiducial_calibration"]["path"]),
                     secondary_image_path=later/"secondary/ego.png",
                     secondary_observation_path=later/"secondary/observation.json", grasp_centering=True)


if __name__ == "__main__":
    unittest.main()
