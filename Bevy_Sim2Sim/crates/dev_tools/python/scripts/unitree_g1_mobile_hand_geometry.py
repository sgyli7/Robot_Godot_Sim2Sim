"""Original hand hull separation from visible pose and self FK, never forces.

This finite report does not establish holding or authorize a controller action.
LP witnesses are checked against complete original vertices, not a fitted mesh.
"""
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def separation(points, box_from_hand, remaining_seconds):
    points = np.asarray(points, dtype=float)
    pose = np.asarray(box_from_hand, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 65536
            or not np.isfinite(points).all() or pose.shape != (4, 4)
            or not np.isfinite(pose).all() or not np.isfinite(remaining_seconds) or remaining_seconds <= 0.
            or not np.array_equal(pose[3], [0., 0., 0., 1.])
            or np.max(np.abs(pose[:3, :3].T @ pose[:3, :3]-np.eye(3))) > 1e-7
            or abs(np.linalg.det(pose[:3, :3])-1.) > 1e-7):
        raise ValueError("finite original-hand geometry inputs are invalid")
    deadline = time.perf_counter()+remaining_seconds
    hull = ConvexHull(points)
    normals = hull.equations[:, :3] @ pose[:3, :3].T
    offsets = hull.equations[:, 3]-normals @ pose[:3, 3]
    matrix = np.zeros((len(normals)+6, 7))
    matrix[:len(normals), :3] = normals
    matrix[-6:, :3] = np.vstack([np.eye(3), -np.eye(3)])
    matrix[-6:, 3:6] = -matrix[-6:, :3]
    matrix[-6:, 6] = -1.
    bound = np.r_[-offsets, np.zeros(6)]
    remaining = deadline-time.perf_counter()
    if remaining <= 0.:
        raise ValueError("complete-hull geometry preprocessing exhausted its budget")
    solved = linprog([0.]*6+[1.], A_ub=matrix, b_ub=bound,
                     bounds=[(None, None)]*3+[(-.1, .1)]*3+[(0., None)],
                     method="highs", options={"time_limit": remaining})
    if (not solved.success or not np.isfinite(solved.x).all()
            or float((matrix @ solved.x-bound).max()) > 2e-7):
        raise ValueError("finite complete-hull separation solve failed")
    transformed = points @ pose[:3, :3].T+pose[:3, 3]
    dual = solved.ineqlin.marginals[-6:]
    normal = dual[:3]-dual[3:]
    norm = np.linalg.norm(normal)
    gap = 0.
    if norm > 1e-10:
        normal /= norm
        hand_projection = transformed @ normal
        box_radius = float(np.abs(normal).sum())*.1
        gap = max(0., float(hand_projection.min()-box_radius),
                  float(-box_radius-hand_projection.max()))
    if solved.fun > 2e-7 and gap < 1e-8:
        raise ValueError("separation lacks an all-original-vertex plane certificate")
    if time.perf_counter() > deadline:
        raise ValueError("complete-hull separation exhausted its finite budget")
    return {"linf_distance_m": float(solved.fun), "certified_separation_m": gap,
            "complete_original_vertex_count": len(points),
            "plane_normal_box": None if norm <= 1e-10 else normal.tolist(),
            "hand_witness_box_m": solved.x[:3].tolist(),
            "box_witness_box_m": solved.x[3:6].tolist(),
            "physical_contact_force_measured": False}


def measure(definition, observation, root_from_box, geometry):
    from unitree_g1_mobile_vision import (DEFINITION_SHA256, TASK_GEOMETRY_SHA256,
                                         original_self_body_frames, transform)

    stamp = observation["stamp"]
    if (set(stamp) != {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"}
            or any(type(stamp[k]) is not int or stamp[k] <= 0 for k in stamp)
            or stamp["sim_time_ns"] % 20_000_000):
        raise ValueError("original hand geometry lacks a current camera boundary")
    box = next(o for o in geometry["objects"] if o["kind"] == "t2_box")
    vertices = np.concatenate([np.asarray(p["points"]) for p in box["convex_parts"]])
    if (vertices.shape != (8, 3) or len(np.unique(vertices, axis=0)) != 8
            or not np.all(np.abs(vertices) == .1)):
        raise ValueError("hand geometry requires the original20cm convex box")
    pose = np.asarray(root_from_box, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or np.linalg.norm(pose[:3, 3]) > 8.
            or not np.array_equal(pose[3], [0., 0., 0., 1.])
            or np.max(np.abs(pose[:3, :3].T @ pose[:3, :3]-np.eye(3))) > 1e-7
            or abs(np.linalg.det(pose[:3, :3])-1.) > 1e-7):
        raise ValueError("visible box pose is nonrigid, reflected or outside bounds")
    frames = original_self_body_frames(definition, observation["measured_joints"]["positions"])
    box_from_root = np.linalg.inv(pose)
    deadline = time.perf_counter()+1.
    bodies = []
    for index in list(range(28, 36))+list(range(45, 53)):
        collision = [c for c in definition["collisions"] if c["body"] == index]
        if len(collision) != 1:
            raise ValueError("original hand collision identity changed")
        collision = collision[0]
        if collision["shape"] == "convex_hull":
            points = np.asarray(collision["points"])
        elif collision["shape"] == "box" and index in (34, 51):
            points = np.asarray(list(itertools.product(*[[-x, x] for x in collision["half_extents"]])))
            local = transform(collision["local_pose"])
            points = points @ local[:3, :3].T+local[:3, 3]
        else:
            raise ValueError("original hand collision shape changed")
        result = separation(points, box_from_root @ frames[index],
                            deadline-time.perf_counter())
        result.update(body_index=index, body_name=definition["bodies"][index]["name"],
                      side="left" if index < 40 else "right")
        bodies.append(result)
    if time.perf_counter() > deadline:
        raise ValueError("complete hand geometry exhausted its total solve budget")
    return {"schema": "g1_current_rgb_original_hand_separation_v1", "observation": stamp,
            "robot_definition_sha256": DEFINITION_SHA256,
            "task_geometry_sha256": TASK_GEOMETRY_SHA256,
            "geometry_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": "current_paired_RGB_original_self_FK_and_complete_hand_hulls",
            "bodies": bodies, "world_or_contact_truth_input": False,
            "rigid_point_20mm_guard_changed": False, "holding_proven": False,
            "physical_contact_forces_proven": False, "owner_execution_admitted": False,
            "task_qualified": False}
