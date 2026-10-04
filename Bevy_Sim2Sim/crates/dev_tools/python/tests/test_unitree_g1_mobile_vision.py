"""Finite visual planning regressions; fixtures do not prove physical success."""
import itertools
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if importlib.util.find_spec("cv2") is not None:
    import unitree_g1_mobile_vision as vision
else:
    vision = None


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


def thumb_proposal(right_inward=.065, joint_limit=1., floor_margin=.03, left_path_safe=True):
    """Minimal full-hull pattern: fixed left pose cannot clear the right thumb."""
    pose = {"position": [0., 0., 0.], "rotation_wxyz": [1., 0., 0., 0.]}
    bodies = [{"name": f"fixture_link_{i}"} for i in range(53)]
    joints, collisions = [], []
    for side, palm, sign, slots in [("left", 28, 1, [26, 27]), ("right", 45, -1, [40, 41])]:
        bodies[palm]["name"] = f"{side}_hand_palm_link"
        joints.append({"name": f"{side}_fixture_mount", "parent": 0, "child": palm,
                       "kind": "fixed", "frame_parent": {**pose, "position": [sign*.12, 0., 0.]},
                       "frame_child": pose})
        for offset, label in enumerate(["palm", "index_0", "index_1", "middle_0", "middle_1",
                                         "thumb_0", "thumb_1", "thumb_2"]):
            body = palm+offset
            bodies[body]["name"] = f"{side}_hand_{label}_link"
            if offset:
                parent = palm if offset <= 5 else body-1
                kind = "revolute" if offset in [5, 6] else "fixed"
                name = vision.JOINT_NAMES[slots[offset-5]] if kind == "revolute" else f"{side}_{label}_fixture"
                joints.append({"name": name, "parent": parent, "child": body,
                               "kind": kind, "axis": "Y" if offset == 5 else "Z",
                               "limits": [-joint_limit, joint_limit],
                               "frame_parent": pose, "frame_child": pose})
            center = ([right_inward, .06, .06] if side == "right" and offset in [6, 7]
                      else [.03, .03, -.03] if side == "left" and offset in [6, 7] and left_path_safe
                      else [sign*.03, 0., 0.])
            points = (np.array(list(itertools.product([-.005, .005], repeat=3))) + center).tolist()
            collisions.append({"body": body, "shape": "convex_hull", "points": points})
    definition = {"bodies": bodies, "joints": joints, "collisions": collisions}
    geometry = {"objects": [{"kind": "t2_box", "convex_parts": [{
        "points": list(itertools.product([-.1, .1], repeat=3))}]}]}
    observation = {"stamp": {"episode_id": 1, "frame_id": 2,
                             "sim_time_ns": 20_000_000, "captured_at_unix_ms": 1},
                   "measured_joints": {"positions": [0.]*43, "root_rotation_wxyz": [1., 0., 0., 0.]}}
    placement = {"release_admitted": False, "hand_clearance_admitted": False,
                 "minimum_signed_floor_margin_m": floor_margin, "estimated_drop_height_m": .2,
                 "visible_bin_upward_cosine": 1., "self_root_speed_m_s": 0.}
    with patch.object(vision, "public_json", return_value=geometry):
        return vision.thumb_preparation_from_visible_markers(
            [{"marker_id": 22, "root_from_marker": np.eye(4).tolist()}], observation,
            Path("public_geometry_fixture"), {22: np.eye(4)}, definition, placement)


@unittest.skipUnless(vision is not None, "visual tests require the pinned G1 OpenCV runtime")
class FinePlanningTests(unittest.TestCase):
    def test_held_feedback_cannot_request_memory_unpaired_or_other_modes(self):
        for kwargs in [{}, {"secondary_image_path": Path("secondary")},
                       {"secondary_image_path": Path("secondary"), "secondary_observation_path": Path("state"), "box_view_only": True},
                       {"secondary_image_path": Path("secondary"), "secondary_observation_path": Path("state"), "memory_path": Path("memory")},
                       {"secondary_image_path": Path("secondary"), "secondary_observation_path": Path("state"), "placement_view_only": True}]:
            with self.assertRaisesRegex(ValueError, "held-box feedback requires"):
                vision.localize(Path("absent_rgb"), Path("absent_self"), Path("definition"),
                                held_box_feedback=True, **kwargs)

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


@unittest.skipUnless(vision is not None and importlib.util.find_spec("scipy") is not None,
                     "thumb geometry tests require the pinned G1 OpenCV/SciPy runtime")
class ThumbPlanningTests(unittest.TestCase):
    def test_right_thumb_obstruction_requires_both_hands_not_more_left_motion(self):
        proposal = thumb_proposal()
        self.assertTrue(proposal["preparation_admitted"])
        targets = proposal["preparation_goal"]["bounded_thumb_targets_rad"]
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(abs(q) <= .25 for q in targets))
        self.assertTrue(any(abs(q) > .001 for q in targets[2:]))
        self.assertGreaterEqual(proposal["candidate_clearance"][
            "minimum_certified_separation_at_maximum_gap_m"], .016)
        self.assertTrue(proposal["fresh_release_observation_required"])
        self.assertFalse(proposal["release_authorized"])

    def test_existing_admitted_left_only_pose_keeps_its_original_goal(self):
        proposal = thumb_proposal(right_inward=.03)
        self.assertTrue(proposal["preparation_admitted"])
        self.assertEqual(proposal["schema"], "g1_visible_station_thumb_preparation_v1")
        self.assertEqual(set(proposal["preparation_goal"]), {"observation"})

    def test_insufficient_joint_range_and_unreachable_geometry_cannot_authorize(self):
        for values in [dict(right_inward=.18), dict(joint_limit=.01)]:
            proposal = thumb_proposal(**values)
            self.assertFalse(proposal["preparation_admitted"])
            self.assertIsNone(proposal["preparation_goal"])

    def test_non_hand_placement_rejection_never_enters_finger_planning(self):
        self.assertIsNone(thumb_proposal(floor_margin=.019))

    def test_clear_final_pose_cannot_admit_an_inward_preparation_path(self):
        # This geometry has a feasible 18 mm final certificate, but the legacy
        # left target moves another hull inward by 1.56 mm during preparation.
        proposal = thumb_proposal(left_path_safe=False)
        self.assertFalse(proposal["preparation_admitted"])
        self.assertIsNone(proposal["preparation_goal"])


if __name__ == "__main__":
    unittest.main()
