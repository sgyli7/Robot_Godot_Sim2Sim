"""Geometry and episode isolation tests; synthetic pixels are not task evidence."""
import importlib.util
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if importlib.util.find_spec("cv2"):
    import cv2
    import unitree_g1_mobile_vision as vision
else:
    vision = None


@unittest.skipIf(vision is None, "optional actual-RGB OpenCV runtime absent")
class MobileBoxPairTests(unittest.TestCase):
    def fixture(self):
        k = np.array([[458.1245526, 0., 320.], [0., 458.1245526, 240.], [0., 0., 1.]])
        cameras = [np.eye(4), np.eye(4)]
        cameras[1][:3, 3] = [0., .08, 0.]
        cameras[1][:3, :3] = cv2.Rodrigues(np.array([np.deg2rad(8), 0., 0.]))[0]
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([np.pi+.04, .02, .01]))[0]
        pose[:3, 3] = [0., .04, .46]
        points = np.array([[-.03,.03,0], [.03,.03,0], [.03,-.03,0], [-.03,-.03,0]])
        pixels = []
        for camera in cameras:
            m = np.linalg.inv(camera) @ pose
            pixels.append(cv2.projectPoints(points, cv2.Rodrigues(m[:3, :3])[0],
                                             m[:3, 3], k, None)[0].reshape(4, 2))
        return k, cameras, pose, pixels

    def test_joint_views_resolve_frontal_pose_without_an_upright_prior(self):
        k, cameras, pose, pixels = self.fixture()
        result = vision.fit_current_box_camera_pair(pixels, k, cameras, .06)
        np.testing.assert_allclose(result["root_from_marker"], pose, atol=1e-7, rtol=0)
        self.assertFalse(result["upright_prior_used"])
        self.assertLessEqual(max(result["candidate_fit_evaluations"]), 50)

    def test_correlated_subpixel_error_still_keeps_the_pose_within_placement_scale(self):
        k, cameras, pose, pixels = self.fixture()
        noise = np.array([[-.1,-.8], [.1,-.9], [-.2,-.6], [-.5,-.7]])
        result = vision.fit_current_box_camera_pair([p+noise for p in pixels], k, cameras, .06)
        found = np.asarray(result["root_from_marker"])
        self.assertLess(np.linalg.norm(found[:3, 3]-pose[:3, 3]), .01)
        angle = np.degrees(np.arccos(np.clip((np.trace(found[:3, :3]@pose[:3, :3].T)-1)/2, -1, 1)))
        self.assertLess(angle, 3.)

    def test_wrong_baseline_or_inconsistent_second_image_is_rejected(self):
        k, cameras, _, pixels = self.fixture()
        changed = [p.copy() for p in pixels]
        changed[1][0] += [10., -10.]
        with self.assertRaises(ValueError):
            vision.fit_current_box_camera_pair(changed, k, cameras, .06)
        cameras[1][:3, 3] = [0., .15, 0.]
        with self.assertRaises(ValueError):
            vision.fit_current_box_camera_pair(pixels, k, cameras, .06)

    def test_pair_rejects_reset_stale_frame_one_sensor_ulp_and_foreign_fields(self):
        first = {"schema": "g1_mobile_marker_observation_v1", "stamp": {
            "episode_id": 7, "frame_id": 11, "sim_time_ns": 20000000, "captured_at_unix_ms": 10000},
            "camera": {}, "camera_mount_profile": "auxiliary_grip_overview",
            "measured_joints": {"positions": [0.]*43, "velocities": [0.]*43,
                                "root_rotation_wxyz": [1., 0., 0., 0.],
                                "root_angular_velocity_body": [0.]*3, "root_velocity_source": [0.]*3}}
        second = copy.deepcopy(first)
        second["camera_mount_profile"] = "auxiliary_bin_placement"
        second["stamp"].update(frame_id=12, captured_at_unix_ms=10020)
        vision.validate_mobile_box_pair_observations(first, second)
        for key, value in [("episode_id", 8), ("frame_id", 11), ("sim_time_ns", 40000000),
                           ("captured_at_unix_ms", 12001)]:
            changed = copy.deepcopy(second)
            changed["stamp"][key] = value
            with self.assertRaises(ValueError):
                vision.validate_mobile_box_pair_observations(first, changed)
        changed = copy.deepcopy(second)
        changed["measured_joints"]["positions"][0] = float(np.nextafter(np.float32(0.), np.float32(1.)))
        with self.assertRaises(ValueError):
            vision.validate_mobile_box_pair_observations(first, changed)
        changed = copy.deepcopy(second)
        changed["box_pose"] = [0., 0., 0.]
        with self.assertRaises(ValueError):
            vision.validate_mobile_box_pair_observations(first, changed)


if __name__ == "__main__":
    unittest.main()
