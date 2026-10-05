"""Current RGB/self-state proposal for original G1 pre-squeeze grasp depth.

Rigid geometric contact onset does not certify contact forces, the approach
path, or a physical grasp. The native owner must separately validate and execute
a reacquisition path before accepting this opt-in proposal.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

MAXIMUM_INSERTION_M = .05


def minimum_common_insertion(planes, direction):
    """One scalar shift and two independent complete-hull witness points."""
    direction = np.asarray(direction, dtype=float)
    if (len(planes) != 2 or direction.shape != (3,) or not np.isfinite(direction).all()
            or abs(np.linalg.norm(direction)-1.) > 1e-7):
        raise ValueError("grasp insertion lacks two original thumbs or a unit axis")
    matrices, bounds = [], []
    for side, values in enumerate(planes):
        values = np.asarray(values, dtype=float)
        if (values.ndim != 2 or values.shape[1] != 4 or len(values) < 4
                or not np.isfinite(values).all()):
            raise ValueError("original thumb hull planes are malformed")
        matrix = np.zeros((len(values)+6, 7))
        offset = side*3
        matrix[:len(values), offset:offset+3] = values[:, :3]
        matrix[-6:, offset:offset+3] = np.vstack([np.eye(3), -np.eye(3)])
        matrix[-6:, 6] = np.r_[direction, -direction]
        matrices.append(matrix)
        bounds.append(np.r_[-values[:, 3], np.full(6, .1)])
    matrix, bound = np.vstack(matrices), np.concatenate(bounds)
    solved = linprog([0.]*6+[1.], A_ub=matrix, b_ub=bound,
                     bounds=[(None, None)]*6+[(0., MAXIMUM_INSERTION_M)], method="highs",
                     options={"time_limit": 1.})
    if solved.status == 2:
        return None
    if (not solved.success or not np.isfinite(solved.x).all()
            or float((matrix @ solved.x-bound).max()) > 2e-7
            or not 0. <= solved.x[6] <= MAXIMUM_INSERTION_M):
        raise ValueError("finite original-thumb insertion solve failed validation")
    shift = float(solved.x[6])
    witnesses = (solved.x[:6].reshape(2, 3)+shift*direction).tolist()
    return shift, witnesses


def propose(definition, observation, root_from_box, geometry, *, closed_regrasp=False):
    """Use measured self FK and a current paired-RGB pose, never world truth."""
    from unitree_g1_mobile_vision import (DEFINITION_SHA256, TASK_GEOMETRY_SHA256,
                                         original_self_body_frames, rotation)

    stamp = observation["stamp"]
    if (set(stamp) != {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"}
            or any(type(stamp[k]) is not int or stamp[k] <= 0 for k in stamp)
            or stamp["sim_time_ns"] % 20_000_000
            or (stamp["sim_time_ns"]//20_000_000 not in (300, 550, 800) if closed_regrasp
                else not 200 <= stamp["sim_time_ns"]//20_000_000 <= 300)):
        raise ValueError("grasp depth requires current closed300/550/800Tick RGB" if closed_regrasp
                         else "grasp depth requires a current pre-lift200..300Tick observation")
    pose = np.asarray(root_from_box, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.array_equal(pose[3], [0., 0., 0., 1.])
            or np.max(np.abs(pose[:3, :3].T @ pose[:3, :3]-np.eye(3))) > 1e-7
            or abs(np.linalg.det(pose[:3, :3])-1.) > 1e-7):
        raise ValueError("grasp proposal has a nonrigid or reflected visible box pose")
    box = next(o for o in geometry["objects"] if o["kind"] == "t2_box")
    points = np.concatenate([np.asarray(p["points"]) for p in box["convex_parts"]])
    if (points.shape != (8, 3) or not np.isfinite(points).all()
            or not np.all(np.abs(points) == .1) or len(np.unique(points, axis=0)) != 8
            or not np.array_equal(points.min(0), [-.1]*3)
            or not np.array_equal(points.max(0), [.1]*3)):
        raise ValueError("grasp proposal requires the original20cm convex box")
    state = observation["measured_joints"]
    frames = original_self_body_frames(definition, state["positions"])
    up = rotation(state["root_rotation_wxyz"]).T @ np.array([0., 0., 1.])
    left, right = frames[28][:3, 0], frames[45][:3, 0]
    if left @ right < .8:
        raise ValueError("original palms do not define one shared forward insertion")
    forward = left+right
    forward -= up*float(forward @ up)/float(up @ up)
    if np.linalg.norm(forward) < .5:
        raise ValueError("original shared forward axis is vertical or degenerate")
    forward /= np.linalg.norm(forward)
    box_from_root = np.linalg.inv(pose)
    planes = []
    for index, name in [(33, "left_hand_thumb_0_link"), (50, "right_hand_thumb_0_link")]:
        if definition["bodies"][index]["name"] != name:
            raise ValueError("original thumb identity changed")
        candidates = [c for c in definition["collisions"] if c["body"] == index]
        if len(candidates) != 1 or candidates[0]["shape"] != "convex_hull":
            raise ValueError("original thumb convex geometry changed")
        thumb_points = np.asarray(candidates[0]["points"], dtype=float)
        if thumb_points.shape != (94, 3) or not np.isfinite(thumb_points).all():
            raise ValueError("original94-vertex thumb geometry is absent or nonfinite")
        hull = ConvexHull(thumb_points)
        relative = box_from_root @ frames[index]
        normals = hull.equations[:, :3] @ relative[:3, :3].T
        planes.append(np.column_stack([normals, hull.equations[:, 3]-normals @ relative[:3, 3]]))
    proposed = minimum_common_insertion(planes, pose[:3, :3].T @ forward)
    result = {"schema": "g1_current_rgb_original_hand_grasp_centering_v1", "observation": stamp,
              "planner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "robot_definition_sha256": DEFINITION_SHA256, "task_geometry_sha256": TASK_GEOMETRY_SHA256,
              "source": "current_paired_RGB_original_self_FK_and_public_thumb_hulls",
              "uses_measured_hands_not_commanded_targets": True,
              "maximum_common_translation_m": MAXIMUM_INSERTION_M,
              "geometry_candidate_found": proposed is not None,
              "world_or_contact_truth_input": False, "physical_contact_forces_proven": False,
              "approach_path_proven": False, "owner_execution_admitted": False, "task_qualified": False}
    if closed_regrasp:
        result["closed_regrasp_only"] = True
    result["requires_owner_reacquisition_if_grip_already_closed"] = True
    if proposed is not None:
        shift, witnesses = proposed
        result.update(common_forward_translation_m=shift,
                      common_offset_root_source_m=(forward*shift).tolist(),
                      predicted_thumb_contact_witnesses_box_m=witnesses,
                      preserves_hypothetical_palm_gap_rotation_and_finger_targets=True)
    return result
