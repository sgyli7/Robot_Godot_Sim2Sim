"""Public RGB board tests; synthetic geometry cannot qualify a physical task."""
import importlib.util
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
class MobileBinBoardTests(unittest.TestCase):
    def test_public_mount_cannot_be_replaced_by_a_runtime_object_pose(self):
        markers = [dict(marker_id=identity, png_path=f"print_{identity}.png", png_sha256="0"*64,
                        center_source_m=[x, .18, .45], rotation_wxyz=[2**-.5, 2**-.5, 0., 0.])
                   for identity, x in [(23, -.25), (24, .25)]]
        vision.validate_public_bin_board(markers)
        markers[0]["center_source_m"][2] += .001
        with self.assertRaises(ValueError):
            vision.validate_public_bin_board(markers)
        markers[0]["center_source_m"][2] -= .001
        markers[0]["actual_object_pose"] = np.eye(4).tolist()
        with self.assertRaises(ValueError):
            vision.validate_public_bin_board(markers)

    def fixture(self):
        k = np.array([[458.1245526, 0., 320.], [0., 458.1245526, 240.], [0., 0., 1.]])
        rotation = cv2.Rodrigues(np.array([1.62, .03, .1]))[0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, [0., .2, 1.25]
        mounts = {}
        pixels = {}
        for marker, x, z, size in [(21, 0., .6, .16), (23, -.25, .45, .12),
                                    (24, .25, .45, .12)]:
            mount = np.eye(4)
            mount[:3, :3] = cv2.Rodrigues(np.array([np.pi / 2, 0., 0.]))[0]
            mount[:3, 3] = [x, .18, z]
            mounts[marker] = mount
            h = size / 2
            points = np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])
            points = points @ mount[:3, :3].T + mount[:3, 3]
            pixels[marker] = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0],
                                              pose[:3, 3], k, None)[0].reshape(4, 2)
        return k, pose, mounts, pixels

    def test_recovers_actual_tilt_without_an_upright_prior(self):
        k, pose, mounts, pixels = self.fixture()
        result = vision.fit_public_bin_board(pixels, k, np.eye(4), mounts[21])
        recovered = np.asarray(result["root_from_marker"]) @ np.linalg.inv(mounts[21])
        np.testing.assert_allclose(recovered, pose, atol=1e-7, rtol=0)
        self.assertFalse(result["world_or_contact_truth_input"])
        self.assertEqual(result["marker_ids"], [21, 23, 24])

    def test_bounded_subpixel_noise_keeps_bin_origin_within_placement_margin(self):
        k, pose, mounts, pixels = self.fixture()
        noise = np.array([[-.23, -.69], [-.87, -.17], [-.31, -.44], [-.24, -.99]])
        pixels = {marker: p + noise for marker, p in pixels.items()}
        result = vision.fit_public_bin_board(pixels, k, np.eye(4), mounts[21])
        recovered = np.asarray(result["root_from_marker"]) @ np.linalg.inv(mounts[21])
        self.assertLess(np.linalg.norm(recovered[:3, 3] - pose[:3, 3]), .02)
        self.assertLess(max(result["per_marker_reprojection_rms_px"]), 1.)

    def test_missing_duplicate_or_inconsistent_board_remains_unadmitted(self):
        k, _, mounts, pixels = self.fixture()
        for modification in ["missing", "extra_identity", "inconsistent", "nonfinite"]:
            changed = {marker: p.copy() for marker, p in pixels.items()}
            if modification == "missing":
                changed.pop(23)
            elif modification == "extra_identity":
                changed[25] = changed[23].copy()
            elif modification == "inconsistent":
                changed[23][0] += [5., -5.]
            else:
                changed[24][0, 0] = np.nan
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                vision.fit_public_bin_board(changed, k, np.eye(4), mounts[21])


if __name__ == "__main__":
    unittest.main()
