"""Disclosed printed-marker localization from actual RGB and G1 self state.

No task-object/world pose, contact, saved action or learned policy is accepted.
This finite development tool proposes navigation; it never actuates a world.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np


JOINT_NAMES = tuple(
    [f"{side}_{name}_joint" for side in ("left", "right")
     for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{name}_joint" for name in ("yaw", "roll", "pitch")]
    + [f"{side}_{name}_joint" for side in ("left", "right")
       for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw", "hand_index_0",
                    "hand_index_1", "hand_middle_0", "hand_middle_1",
                    "hand_thumb_0", "hand_thumb_1", "hand_thumb_2")]
)
DEFINITION_SHA256 = "571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd"
MARKER_SIZES = {21: 0.16, 22: 0.10}
TASK_GEOMETRY_SHA256 = "19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0"
BIN_BOARD_PROFILE = "auxiliary_bin_board_targets"


def validate_public_bin_board(markers):
    """Fixed calibration only; object pose is always measured from current RGB."""
    if not isinstance(markers, list) or len(markers) != 2:
        raise ValueError("public bin board requires exactly its two extra printed labels")
    for marker, identity, x in zip(markers, [23, 24], [-.25, .25]):
        if (set(marker) != {"marker_id", "png_path", "png_sha256", "center_source_m", "rotation_wxyz"}
                or marker["marker_id"] != identity
                or marker["center_source_m"] != [x, .18, .45]
                or not np.allclose(marker["rotation_wxyz"], [2**-.5, 2**-.5, 0., 0.], atol=1e-7, rtol=0)
                or not isinstance(marker["png_path"], str) or not marker["png_path"]
                or not isinstance(marker["png_sha256"], str) or len(marker["png_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in marker["png_sha256"])):
            raise ValueError("public bin board identity, dimensions or rigid mount changed")


def fit_public_bin_board(pixels_by_id, intrinsics, camera_in_root, bin_marker_mount):
    """One bounded joint twelve-corner PnP; no gravity/upright/world prior."""
    if set(pixels_by_id) != {21, 23, 24}:
        raise ValueError("all three distinct current bin board labels are required")
    original_mount = transform({"position": [0., .18, .60],
                                "rotation_wxyz": [2**-.5, 2**-.5, 0., 0.]})
    if not np.allclose(bin_marker_mount, original_mount, atol=1e-7, rtol=0):
        raise ValueError("bin board cannot use a foreign rigid attachment")
    points, pixels, edges = [], [], []
    for marker, x, z, size in [(21, 0., .6, .16), (23, -.25, .45, .12), (24, .25, .45, .12)]:
        measured = np.asarray(pixels_by_id[marker], dtype=np.float64)
        if measured.shape != (4, 2) or not np.isfinite(measured).all():
            raise ValueError("invalid current board corners")
        edge = float(np.linalg.norm(measured-np.roll(measured, -1, axis=0), axis=1).min())
        if edge < 8 or (measured < 0).any() or (measured >= [640., 480.]).any():
            raise ValueError("board label lies outside its unchanged image/edge gate")
        h = size / 2
        local = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]])
        points.append(local @ original_mount[:3, :3].T + [x, .18, z])
        pixels.append(measured)
        edges.append(edge)
    points, pixels = np.concatenate(points), np.concatenate(pixels)
    ok, rvec, tvec = cv2.solvePnP(points, pixels, intrinsics, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("current public board pose solve failed")
    rvec, tvec = cv2.solvePnPRefineLM(points, pixels, intrinsics, None, rvec, tvec,
                                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    projected = cv2.projectPoints(points, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
    errors = np.sqrt(np.mean(np.sum((projected-pixels).reshape(3, 4, 2)**2, axis=2), axis=1))
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    normal = pose[:3, :3] @ original_mount[:3, 2]
    if (not np.isfinite(pose).all() or not np.isfinite(errors).all() or errors.max() > 1.
            or camera_points[:, 2].min() <= .1 or camera_points[:, 2].max() >= 5.
            or float(normal @ camera_points.mean(axis=0)) >= 0):
        raise ValueError("public bin board failed unchanged residual, depth or front-face gate")
    return {"schema": "g1_actual_rgb_public_bin_board_pose_v1", "marker_ids": [21, 23, 24],
            "corners_px": pixels.reshape(3, 4, 2).tolist(),
            "per_marker_reprojection_rms_px": errors.tolist(), "minimum_edges_px": edges,
            "solver": "SQPNP_then_bounded30_iteration_LM", "upright_prior_used": False,
            "root_from_marker": (camera_in_root @ pose @ original_mount).tolist(),
            "world_or_contact_truth_input": False, "task_qualified": False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_current_box_camera_pair(pixels_by_view, intrinsics, cameras_in_root, marker_size):
    """Fit both planar candidates to two current, fixed-baseline RGB views.

    Only measured corners, pinhole calibration and self FK enter the solve.
    Each candidate gets at most fifty evaluations; image residual chooses it.
    """
    from scipy.optimize import least_squares

    pixels = np.asarray(pixels_by_view, dtype=np.float64)
    cameras = np.asarray(cameras_in_root, dtype=np.float64)
    if (pixels.shape != (2, 4, 2) or cameras.shape != (2, 4, 4)
            or not np.isfinite(pixels).all() or not np.isfinite(cameras).all()
            or not np.isfinite(intrinsics).all() or marker_size != .06
            or (pixels < 0).any() or (pixels >= [640., 480.]).any()
            or abs(np.linalg.norm(cameras[0, :3, 3]-cameras[1, :3, 3])-.08) > 1e-7):
        raise ValueError("box pair requires current in-frame corners and the published8cm baseline")
    for camera in cameras:
        if (not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(camera[:3, :3].T@camera[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or abs(np.linalg.det(camera[:3, :3])-1) > 1e-6):
            raise ValueError("invalid self FK camera transform")
    edges = np.linalg.norm(pixels-np.roll(pixels, -1, axis=1), axis=2).min(axis=1)
    if edges.min() < 8:
        raise ValueError("box pair fails the unchanged8px marker edge gate")
    h = marker_size / 2
    points = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]])
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(points, pixels[0], intrinsics, None,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not 1 <= len(rvecs) <= 2:
        raise ValueError("no finite current primary IPPE candidates")
    inverse_cameras = np.linalg.inv(cameras)

    def pose(parameters):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = cv2.Rodrigues(parameters[:3])[0], parameters[3:]
        return m

    def residual(parameters):
        projected = []
        root_pose = pose(parameters)
        for camera in inverse_cameras:
            m = camera @ root_pose
            projected.append(cv2.projectPoints(points, cv2.Rodrigues(m[:3, :3])[0],
                                                m[:3, 3], intrinsics, None)[0].reshape(4, 2))
        return (np.asarray(projected)-pixels).ravel()

    candidates, evaluations = [], []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        seed = np.eye(4)
        seed[:3, :3], seed[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        seed = cameras[0] @ seed
        parameters = np.r_[cv2.Rodrigues(seed[:3, :3])[0].ravel(), seed[:3, 3]]
        fit = least_squares(residual, parameters, max_nfev=50,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        evaluations.append(int(fit.nfev))
        root_pose = pose(fit.x)
        errors = np.sqrt(np.mean(np.sum(residual(fit.x).reshape(2, 4, 2)**2, axis=2), axis=1))
        valid = bool(fit.success and np.isfinite(root_pose).all() and np.isfinite(errors).all()
                     and errors.max() <= 1.)
        for camera in inverse_cameras:
            m = camera @ root_pose
            camera_points = points @ m[:3, :3].T + m[:3, 3]
            valid = valid and bool(camera_points[:, 2].min() > .1
                                   and camera_points[:, 2].max() < 5.
                                   and float(m[:3, 2] @ camera_points.mean(axis=0)) < 0)
        if valid:
            candidates.append((float(fit.fun@fit.fun), index, root_pose, errors))
    if not candidates:
        raise ValueError("box pair failed unchanged residual, depth or front-face gate")
    _, index, root_pose, errors = min(candidates, key=lambda item: item[0])
    return {"schema": "g1_actual_rgb_same_tick_box_pair_pose_v1", "marker_id": 22,
            "corners_px": pixels.tolist(), "minimum_edges_px": edges.tolist(),
            "per_view_reprojection_rms_px": errors.tolist(), "root_from_marker": root_pose.tolist(),
            "camera_baseline_m": float(np.linalg.norm(cameras[0, :3, 3]-cameras[1, :3, 3])),
            "candidate_fit_evaluations": evaluations, "selected_candidate": index,
            "solver": "two_IPPE_seeds_bounded50_joint_camera_fit", "upright_prior_used": False,
            "world_or_contact_truth_input": False, "task_qualified": False}


def validate_mobile_box_pair_observations(first, second):
    """No tolerance on native f32 sensors, episode or simulation time."""
    allowed = {"schema", "stamp", "camera", "measured_joints", "camera_mount_profile"}
    profiles = [{"auxiliary_grip_overview", "auxiliary_bin_placement"},
                {"auxiliary_pregrasp_overview", "auxiliary_bin_placement"}]
    keys = {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"}
    if (set(first) != allowed or set(second) != allowed
            or first["schema"] != "g1_mobile_marker_observation_v1"
            or second["schema"] != first["schema"] or first["camera"] != second["camera"]
            or {first["camera_mount_profile"], second["camera_mount_profile"]} not in profiles
            or set(first["stamp"]) != keys or set(second["stamp"]) != keys):
        raise ValueError("box pair requires only its two published camera/self schemas")
    a, b = first["stamp"], second["stamp"]
    if (any(type(v) is not int or v <= 0 for stamp in (a, b) for v in stamp.values())
            or a["episode_id"] != b["episode_id"] or a["sim_time_ns"] != b["sim_time_ns"]
            or a["sim_time_ns"] % 20_000_000 or a["frame_id"] >= b["frame_id"]
            or not 0 < b["captured_at_unix_ms"]-a["captured_at_unix_ms"] <= 2000):
        raise ValueError("box pair changed episode/Tick or exceeded its2s capture budget")
    lengths = {"positions": 43, "velocities": 43, "root_rotation_wxyz": 4,
               "root_angular_velocity_body": 3, "root_velocity_source": 3}
    for observation in (first, second):
        if set(observation["measured_joints"]) != set(lengths):
            raise ValueError("box pair has foreign self fields")
    for key, length in lengths.items():
        a = np.asarray(first["measured_joints"][key], dtype=np.float32)
        b = np.asarray(second["measured_joints"][key], dtype=np.float32)
        if (a.shape != (length,) or b.shape != a.shape or not np.isfinite(a).all()
                or not np.isfinite(b).all() or not np.array_equal(a.view(np.uint32), b.view(np.uint32))):
            raise ValueError("box pair changed native self-sensor bits")


class PinnedPublicVisionAssets:
    """One preparation-time copy of public calibration, never observations.

    Only these three explicitly supplied files can use this cache. Their hashes
    bind the in-memory data; localize still reads every actual RGB/self frame.
    The localizer uses these records only as calibration and never mutates them.
    """
    def __init__(self, definition, geometry, fiducials):
        self._files = {}
        for path, expected, limit in ((definition, DEFINITION_SHA256, 64*1024*1024),
                                      (geometry, TASK_GEOMETRY_SHA256, 1024*1024),
                                      (fiducials, None, 16*1024)):
            path = Path(path).resolve(strict=True)
            if not path.is_file() or path.stat().st_size > limit:
                raise ValueError("public calibration exceeds its preparation byte budget")
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if expected is not None and digest != expected:
                raise ValueError("public calibration identity mismatch")
            self._files[str(path)] = (digest, json.loads(raw))
        if len(self._files) != 3:
            raise ValueError("public calibration paths must be distinct")

    def _bound(self, path):
        try:
            return self._files[str(Path(path).resolve(strict=True))]
        except KeyError as error:
            raise ValueError("unknown path cannot enter public calibration cache") from error


def public_json(path, assets=None):
    return json.loads(Path(path).read_text()) if assets is None else assets._bound(path)[1]


def public_sha(path, assets=None):
    return sha(path) if assets is None else assets._bound(path)[0]


def rotation(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(q @ q - 1) > 2e-5:
        raise ValueError("invalid original quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def transform(pose):
    result = np.eye(4)
    result[:3, :3] = rotation(pose["rotation_wxyz"])
    result[:3, 3] = pose["position"]
    return result


def original_self_body_frames(definition, positions):
    """Original FK in root coordinates, using named measured self joints only."""
    if len(positions) != 43 or not np.isfinite(positions).all():
        raise ValueError("vision requires43original measured joint positions")
    slots = dict(zip(JOINT_NAMES, positions))
    frames = {0: np.eye(4)}
    pending = list(definition["joints"])
    while pending:
        ready = next((j for j in pending if j["parent"] in frames), None)
        if ready is None:
            raise ValueError("disconnected original G1 kinematic tree")
        pending.remove(ready)
        motion = np.eye(4)
        if ready["kind"] == "revolute":
            axis = np.eye(3)["XYZ".index(ready["axis"])]
            motion[:3, :3] = cv2.Rodrigues(axis * slots[ready["name"]])[0]
        frames[ready["child"]] = (frames[ready["parent"]] @ transform(ready["frame_parent"])
                                  @ motion @ np.linalg.inv(transform(ready["frame_child"])))
    return frames


def root_from_camera(definition, positions, camera_profile="arena_ego"):
    """Original FK in root coordinates, followed by the published ROS camera mount."""
    frames = original_self_body_frames(definition, positions)
    mount_q = np.array([0.32651, -0.62721, 0.62721, -0.32651])
    mount_q /= np.linalg.norm(mount_q)
    mount = transform({"position": [0.04485, 0., 0.35325], "rotation_wxyz": mount_q})
    if camera_profile == "auxiliary_grip_overview":
        mount[:3, 3] += np.array([0., 0., .15])
        mount[:3, :3] = mount[:3, :3] @ cv2.Rodrigues(np.array([np.pi/12, 0., 0.]))[0]
    elif camera_profile == "auxiliary_pregrasp_overview":
        mount[:3, 3] += np.array([0., 0., .15])
        mount[:3, :3] = mount[:3, :3] @ cv2.Rodrigues(np.array([math.radians(7), 0., 0.]))[0]
    elif camera_profile == "auxiliary_bin_placement":
        mount[:3, 3] += np.array([0., 0., .23])
        mount[:3, :3] = mount[:3, :3] @ cv2.Rodrigues(np.array([math.radians(7), 0., 0.]))[0]
    elif camera_profile != "arena_ego":
        raise ValueError("unknown fixed head camera mount")
    return frames[19] @ mount


def propagate_target_memory(memory_path, observation):
    """Fresh static-target assumption, propagated using original self motion."""
    if Path(memory_path).stat().st_size > 128 * 1024:
        raise ValueError("target memory exceeds byte budget")
    memory = json.loads(Path(memory_path).read_text())
    keys = {"schema", "origin_observation", "current_observation", "origin_image_sha256",
            "origin_input_sha256", "root_from_target_marker", "origin_root_rotation_wxyz",
            "self_velocity_displacement_source_m", "static_target_assumption",
            "world_or_contact_truth_input"}
    if set(memory) != keys or memory["schema"] != "g1_mobile_static_target_memory_v1" or memory["static_target_assumption"] is not True or memory["world_or_contact_truth_input"] is not False:
        raise ValueError("foreign target-memory fields/source")
    now, old = observation["stamp"], memory["origin_observation"]
    if memory["current_observation"] != now or set(old) != set(now) or old["episode_id"] != now["episode_id"] or old["frame_id"] >= now["frame_id"] or old["sim_time_ns"] % 20_000_000:
        raise ValueError("foreign/old target-memory camera identity")
    age_ns = now["sim_time_ns"] - old["sim_time_ns"]
    wall_age_ms = now["captured_at_unix_ms"] - old["captured_at_unix_ms"]
    if not 0 < age_ns <= 8_000_000_000 or not 0 < wall_age_ms <= 12_000:
        raise ValueError("static visual target memory expired")
    for key in ("origin_image_sha256", "origin_input_sha256"):
        h = memory[key]
        if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("target memory has no original image/input identity")
    previous = np.asarray(memory["root_from_target_marker"], dtype=np.float64)
    displacement = np.asarray(memory["self_velocity_displacement_source_m"], dtype=np.float64)
    if previous.shape != (4, 4) or not np.isfinite(previous).all() or not np.allclose(previous[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or not np.allclose(previous[:3, :3].T @ previous[:3, :3], np.eye(3), atol=1e-6, rtol=0) or abs(np.linalg.det(previous[:3, :3])-1) > 1e-6 or displacement.shape != (3,) or not np.isfinite(displacement).all() or np.linalg.norm(displacement) > .25:
        raise ValueError("invalid bounded self-motion/visual target transform")
    origin_rotation = rotation(memory["origin_root_rotation_wxyz"])
    current_rotation = rotation(observation["measured_joints"]["root_rotation_wxyz"])
    result = np.eye(4)
    result[:3, :3] = current_rotation.T @ origin_rotation @ previous[:3, :3]
    result[:3, 3] = current_rotation.T @ (origin_rotation @ previous[:3, 3] - displacement)
    return {"schema": "g1_propagated_static_visual_target_v1", "observation": now,
            "origin_observation": old, "target_identity": 21,
            "origin_image_sha256": memory["origin_image_sha256"],
            "origin_input_sha256": memory["origin_input_sha256"], "memory_sha256": sha(memory_path),
            "age_sim_ns": age_ns, "age_wall_ms": wall_age_ms,
            "static_target_assumption": True, "world_or_contact_truth_input": False,
            "root_from_marker": result.tolist(), "task_qualified": False}


def fixed_marker_layout(path, camera_profile, public_assets=None):
    sizes = MARKER_SIZES.copy()
    mounts = {21: transform({"position": [.008351, .0113635, .0045], "rotation_wxyz": [1., 0., 0., 0.]}),
              22: transform({"position": [.1005, 0., 0.], "rotation_wxyz": [2**-.5, 0., 2**-.5, 0.]})}
    digest = None
    layout = "original_arena"
    if path is not None:
        if Path(path).stat().st_size > 16*1024:
            raise ValueError("public fiducial document exceeds byte budget")
        doc = public_json(path, public_assets)
        fields = {"schema", "dictionary", "png_paths", "png_sha256"}
        if set(doc) not in (fields, fields | {"layout_profile"}, fields | {"layout_profile", "mobile_bin_board_markers"}) or doc["schema"] != "g1_task_fiducials_v1" or doc["dictionary"] != "DICT_4X4_50":
            raise ValueError("foreign fixed public marker layout")
        layout = doc.get("layout_profile", "original_arena")
        if layout == BIN_BOARD_PROFILE:
            validate_public_bin_board(doc.get("mobile_bin_board_markers"))
        elif "mobile_bin_board_markers" in doc:
            raise ValueError("bin board calibration cannot enter a different marker profile")
        digest = public_sha(path, public_assets)
    if layout in ("auxiliary_grip_targets", BIN_BOARD_PROFILE):
        if camera_profile not in ("auxiliary_grip_overview", "auxiliary_bin_placement", "auxiliary_pregrasp_overview"):
            raise ValueError("auxiliary labels require their explicitly declared sensor")
        sizes[22] = .06
        mounts[21] = transform({"position": [0., .18, .60], "rotation_wxyz": [2**-.5, 2**-.5, 0., 0.]})
        mounts[22] = transform({"position": [.1005, 0., -.04], "rotation_wxyz": [2**-.5, 0., 2**-.5, 0.]})
    elif layout != "original_arena" or camera_profile != "arena_ego":
        raise ValueError("unbound auxiliary sensor or unsupported marker layout")
    return sizes, mounts, layout, digest


def clearance_from_visible_markers(detections, observation, geometry_path, marker_mounts=None, public_assets=None):
    """Public immutable geometry and actual RGB poses, never world coordinates."""
    if Path(geometry_path).stat().st_size > 1024 * 1024 or public_sha(geometry_path, public_assets) != TASK_GEOMETRY_SHA256:
        raise ValueError("original public task geometry identity mismatch")
    objects = public_json(geometry_path, public_assets)["objects"]
    by_id = {d["marker_id"]: d for d in detections}
    if set(by_id) != {21, 22}:
        raise ValueError("clearance requires both current visible markers")
    root_rotation = rotation(observation["measured_joints"]["root_rotation_wxyz"])
    bounds = {}
    for kind, marker_id, position, mount_rotation in (
        ("t2_bin", 21, [0.008351, 0.0113635, 0.0045], np.eye(3)),
        ("t2_box", 22, [0.1005, 0., 0.], rotation([2**-.5, 0., 2**-.5, 0.])),
    ):
        marker = by_id[marker_id]
        if marker["minimum_edge_px"] < 8 or marker["reprojection_rms_px"] > 1:
            raise ValueError("clearance marker quality failed")
        pose = np.asarray(marker["root_from_marker"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("nonfinite marker transform")
        mount = np.eye(4)
        mount[:3, :3] = mount_rotation
        mount[:3, 3] = position
        root_from_object = pose @ np.linalg.inv(mount if marker_mounts is None else marker_mounts[marker_id])
        vertices = np.concatenate([np.asarray(part["points"], dtype=np.float64)
                                   for part in next(o for o in objects if o["kind"] == kind)["convex_parts"]])
        gravity_points = (vertices @ root_from_object[:3, :3].T + root_from_object[:3, 3]) @ root_rotation.T
        bounds[kind] = {"minimum": gravity_points.min(axis=0).tolist(),
                        "maximum": gravity_points.max(axis=0).tolist()}
    rise = float(bounds["t2_bin"]["maximum"][2] + .1 - bounds["t2_box"]["minimum"][2])
    duration = max(50, math.ceil(rise / .001))
    admitted = .01 <= rise <= .18 and duration <= 200
    return {"schema": "g1_visible_marker_clearance_v1", "observation": observation["stamp"],
            "source_geometry_sha256": TASK_GEOMETRY_SHA256,
            "source": "actual_rgb_marker_pose_and_public_original_collision_geometry",
            "world_or_contact_truth_input": False, "task_qualified": False,
            "bounds_relative_to_robot_in_gravity_frame": bounds,
            "minimum_desired_rim_clearance_m": .1, "required_raise_m": rise,
            "raise_goal": {"observation": observation["stamp"], "distance_m": rise,
                           "duration_ticks": duration} if admitted else None}


def horizontal_release_clearance(definition, observation, root_from_box, box_vertices, *, _plane_output=None):
    """Certify separation of original hand hulls from the visible convex box.

    This estimates a rigid horizontal opening from measured self FK and current
    RGB. It uses no world/contact state and cannot prove physical detachment.
    """
    from scipy.optimize import LinearConstraint, minimize
    from scipy.spatial import ConvexHull

    frames = original_self_body_frames(definition, observation["measured_joints"]["positions"])
    delta = frames[28][:3, 3] - frames[45][:3, 3]
    up = rotation(observation["measured_joints"]["root_rotation_wxyz"]).T @ np.array([0., 0., 1.])
    vertical = float(delta @ up)
    horizontal = delta - vertical * up
    span = float(np.linalg.norm(horizontal))
    if span < 1e-4:
        raise ValueError("horizontal release lacks a lateral palm direction")
    direction = horizontal / span
    basis = np.column_stack([direction, np.cross(direction, up), up])
    box = (box_vertices @ root_from_box[:3, :3].T + root_from_box[:3, 3]) @ basis
    if abs(vertical) >= .35:
        raise ValueError("measured palm vertical offset exceeds the release envelope")
    maximum_shift = .5 * (np.sqrt(.35**2 - vertical**2) - span)
    if maximum_shift < 0.:
        raise ValueError("measured palm span already exceeds the release envelope")
    shifts = []
    certificates = []
    for collider in definition["collisions"]:
        index = collider["body"]
        name = definition["bodies"][index]["name"]
        if not name.startswith(("left_hand_", "right_hand_")):
            continue
        if collider["shape"] == "convex_hull":
            points = np.asarray(collider["points"], dtype=np.float64)
        elif collider["shape"] == "box":
            points = np.asarray(list(itertools.product(*[[-x, x] for x in collider["half_extents"]])))
            local = transform(collider["local_pose"])
            points = points @ local[:3, :3].T + local[:3, 3]
        else:
            raise ValueError("unsupported original hand collision shape")
        local_points = points
        points = (points @ frames[index][:3, :3].T + frames[index][:3, 3]) @ basis
        left = name.startswith("left_")
        outward = np.array([1. if left else -1., 0., 0.])
        target = maximum_shift * outward
        differences = (box[:, None, :] - points[None, :, :]).reshape(-1, 3)
        planes = ConvexHull(differences).equations
        solved = minimize(lambda p: .5 * np.dot(p-target, p-target), differences.mean(axis=0),
                          jac=lambda p: p-target,
                          constraints=LinearConstraint(planes[:, :3], -np.inf, -planes[:, 3]),
                          method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        if not solved.success:
            raise ValueError("original hand release clearance solve failed")
        nearest_delta = target-solved.x
        norm = float(np.linalg.norm(nearest_delta))
        normal = nearest_delta/norm if norm > 1e-9 else outward
        if float(normal @ outward) <= 1e-6:
            normal = outward
        # The optimizer proposes a direction only. Max support over ALL original
        # vertex differences proves the plane for both complete convex hulls,
        # independently of optimizer accuracy or its candidate feasibility.
        support = float((differences @ normal).max())
        projection = float(normal @ outward)
        certificate = float(target @ normal-support)
        required_shift = max(0., (.010000001 + support)/projection)
        if not np.isfinite([certificate, required_shift]).all():
            raise ValueError("nonfinite original hand separation certificate")
        shifts.append(required_shift)
        certificates.append(certificate)
        if _plane_output is not None:
            # A fixed support plane remains a complete-hull certificate when
            # fingers rotate. The bounded planner never trusts optimizer fit.
            _plane_output.append((index, local_points, basis @ normal,
                                  float(target @ normal) - float((box @ normal).max())))
    if not shifts:
        raise ValueError("visible box has no original hand collision geometry for release")
    required = float(np.hypot(span + 2 * max(shifts), vertical))
    return {"opening_profile": "gravity_horizontal_original_fingers",
            "robot_collision_geometry_sha256": DEFINITION_SHA256,
            "minimum_hand_clearance_m": .01,
            "required_measured_palm_gap_m": required,
            "maximum_commanded_palm_gap_m": .35,
            "hand_clearance_admitted": required <= .35,
            "clearance_method": "original_convex_vertex_support_plane_v1",
            "minimum_certified_separation_at_maximum_gap_m": min(certificates),
            "certified_hand_colliders": len(certificates),
            "prediction_is_not_physical_detachment": True}


def bounded_thumb_candidate(definition, observation, root_from_box, vertices, planes, legacy_targets):
    """One four-joint solve; certify all hulls and the declared 50 Tick path."""
    from scipy.optimize import minimize

    slots = np.array([26, 27, 40, 41])
    initial = np.asarray(observation["measured_joints"]["positions"], dtype=float)
    if np.any(np.abs(initial[slots]) > .02):
        return None
    by_name = {j["name"]: j for j in definition["joints"]}
    bounds = []
    for slot in slots:
        limits = by_name[JOINT_NAMES[slot]]["limits"]
        lower, upper = max(-.25, limits[0]), min(.25, limits[1])
        if not np.isfinite([lower, upper]).all() or lower > upper:
            return None
        bounds.append((lower, upper))
    seed = np.clip(np.asarray([*legacy_targets, 0., 0.]),
                   np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])

    def certificates(targets):
        positions = initial.copy()
        positions[slots] = targets
        frames = original_self_body_frames(definition, positions)
        return np.asarray([constant + float(((points @ frames[body][:3, :3].T
                                             + frames[body][:3, 3]) @ normal).min())
                           for body, points, normal, constant in planes])

    solved = minimize(lambda q: .5*np.dot(q-seed, q-seed), seed, jac=lambda q: q-seed,
                      method="SLSQP", bounds=bounds,
                      constraints={"type": "ineq", "fun": lambda q: certificates(q)-.018},
                      options={"ftol": 1e-12, "maxiter": 30})
    targets = np.asarray(solved.x, dtype=np.float32).astype(float)
    if (not solved.success or not np.isfinite(targets).all()
            or any(not low <= q <= high for q, (low, high) in zip(targets, bounds))
            or certificates(targets).min() < .016):
        return None
    initial_certificates = certificates(initial[slots])
    maximum_inward = max(float((initial_certificates - certificates(
        initial[slots] + (targets-initial[slots])*(tick/50.))).max()) for tick in range(51))
    if not np.isfinite(maximum_inward) or maximum_inward > .001:
        return None
    candidate = {**observation, "measured_joints": {**observation["measured_joints"],
                 "positions": initial.tolist()}}
    for slot, target in zip(slots, targets):
        candidate["measured_joints"]["positions"][slot] = float(target)
    clearance = horizontal_release_clearance(definition, candidate, root_from_box, vertices)
    if (not clearance["hand_clearance_admitted"]
            or clearance["minimum_certified_separation_at_maximum_gap_m"] < .016):
        return None
    path = {"schema": "g1_hypothetical_thumb_support_plane_path_v1", "samples": 51,
            "maximum_additional_inward_projection_m": maximum_inward,
            "maximum_allowed_additional_inward_projection_m": .001,
            "all_original_hand_hulls": len(planes), "actual_contact_safety_proven": False}
    return targets.tolist(), clearance, path


def placement_from_visible_markers(detections, observation, geometry_path, mounts, definition, public_assets=None):
    """Conservative release admission from current RGB and public geometry only."""
    objects = {o["kind"]: o for o in public_json(geometry_path, public_assets)["objects"]}
    poses = {d["marker_id"]: np.asarray(d["root_from_marker"]) @ np.linalg.inv(mounts[d["marker_id"]]) for d in detections}
    if set(poses) != {21, 22}:
        raise ValueError("placement requires both current visible markers")
    bin_from_box = np.linalg.inv(poses[21]) @ poses[22]
    vertices = np.concatenate([np.asarray(p["points"]) for p in objects["t2_box"]["convex_parts"]])
    points = vertices @ bin_from_box[:3, :3].T + bin_from_box[:3, 3]
    floor = np.asarray(objects["t2_bin"]["convex_parts"][2]["points"])
    # The pinned original part2 has a rectangular flat-floor footprint.
    hull = cv2.convexHull(floor[:, :2].astype(np.float32)).reshape(-1, 2)
    lower, upper = floor[:, :2].min(axis=0), floor[:, :2].max(axis=0)
    if len(hull) != 4 or not all(np.allclose(p, [lower[0] if p[0] < 0 else upper[0], lower[1] if p[1] < 0 else upper[1]], atol=1e-7) for p in hull):
        raise ValueError("public original floor is not the fixed rectangle")
    margin = float(min((points[:, :2]-lower).min(), (upper-points[:, :2]).min()))
    drop = float(points[:, 2].min()-floor[:, 2].max())
    upright = float((rotation(observation["measured_joints"]["root_rotation_wxyz"]) @ poses[21][:3, :3])[2, 2])
    root_speed = float(np.linalg.norm(observation["measured_joints"]["root_velocity_source"]))
    opening = horizontal_release_clearance(definition, observation, poses[22], vertices)
    admitted = margin >= .02 and .05 <= drop <= .4 and upright >= .98 and root_speed <= .05 and opening["hand_clearance_admitted"]
    return {"schema": "g1_visible_marker_release_admission_v2", "observation": observation["stamp"],
            "source_geometry_sha256": TASK_GEOMETRY_SHA256, "world_or_contact_truth_input": False,
            "source": "actual_two_marker_rgb_and_public_original_collision_vertices",
            "minimum_signed_floor_margin_m": margin, "minimum_required_floor_margin_m": .02,
            "estimated_drop_height_m": drop, "visible_bin_upward_cosine": upright,
            "self_root_speed_m_s": root_speed, "release_admitted": admitted, "task_qualified": False,
            "release_goal": {"observation": observation["stamp"], "target_palm_gap_m": .35,
                             "duration_ticks": 100} if admitted else None, **opening}


def thumb_preparation_from_visible_markers(detections, observation, geometry_path, mounts, definition, placement, public_assets=None):
    """Propose one disclosed finger pose; it cannot authorize physical release.

    The candidate FK is labelled hypothetical. After the finite original-motor
    preparation, another actual RGB/self observation must pass normal release
    admission. The current measured pose remains rejected here.
    """
    if (placement["release_admitted"] or placement["hand_clearance_admitted"]
            or placement["minimum_signed_floor_margin_m"] < .02
            or not .05 <= placement["estimated_drop_height_m"] <= .4
            or placement["visible_bin_upward_cosine"] < .98
            or placement["self_root_speed_m_s"] > .05):
        return None
    positions = observation["measured_joints"]["positions"]
    if any(abs(positions[i]) > .02 for i in (26, 27)):
        return None
    targets = np.asarray([-.04068526, -.18428603], dtype=np.float32).astype(float).tolist()
    candidate = {**observation, "measured_joints": {**observation["measured_joints"], "positions": list(positions)}}
    candidate["measured_joints"]["positions"][26:28] = targets
    objects = {o["kind"]: o for o in public_json(geometry_path, public_assets)["objects"]}
    vertices = np.concatenate([np.asarray(p["points"]) for p in objects["t2_box"]["convex_parts"]])
    box_pose = next(np.asarray(d["root_from_marker"]) @ np.linalg.inv(mounts[22]) for d in detections if d["marker_id"] == 22)
    planes = []
    clearance = horizontal_release_clearance(definition, candidate, box_pose, vertices, _plane_output=planes)
    admitted = clearance["hand_clearance_admitted"] and clearance["minimum_certified_separation_at_maximum_gap_m"] >= .016
    result = {"schema": "g1_visible_station_thumb_preparation_v1", "observation": observation["stamp"],
            "robot_collision_geometry_sha256": DEFINITION_SHA256, "source_geometry_sha256": TASK_GEOMETRY_SHA256,
            "world_or_contact_truth_input": False, "candidate_is_not_measured_state": True,
            "fresh_release_observation_required": True, "release_authorized": False,
            "target_left_thumb_angles_rad": targets, "measured_initial_left_thumb_angles_rad": positions[26:28],
            "ramp_ticks": 50, "settling_ticks": 50, "maximum_tracking_error_rad": .02,
            "minimum_candidate_clearance_m": .016, "candidate_clearance": clearance,
            "preparation_admitted": admitted,
            "preparation_goal": {"observation": observation["stamp"]} if admitted else None}
    if admitted:
        return result
    bounded = bounded_thumb_candidate(definition, observation, box_pose, vertices, planes, targets)
    if bounded is not None:
        selected, clearance, path = bounded
        result.update(schema="g1_visible_station_thumb_preparation_v2",
                      target_left_thumb_angles_rad=selected[:2],
                      target_right_thumb_angles_rad=selected[2:],
                      measured_initial_right_thumb_angles_rad=positions[40:42],
                      maximum_absolute_thumb_angle_rad=.25, planning_clearance_m=.018,
                      planning_method="fixed_support_planes_bounded_slsqp_v1",
                      candidate_clearance=clearance, hypothetical_preparation_path=path,
                      preparation_admitted=True,
                      preparation_goal={"observation": observation["stamp"],
                                        "bounded_thumb_targets_rad": selected})
    return result


def fine_from_visible_markers(detections, observation, geometry_path, mounts, heading, public_assets=None):
    """One short observed approach step, then another image; no world state."""
    objects = {o["kind"]: o for o in public_json(geometry_path, public_assets)["objects"]}
    poses = {d["marker_id"]: np.asarray(d["root_from_marker"]) @ np.linalg.inv(mounts[d["marker_id"]]) for d in detections}
    if set(poses) != {21, 22}:
        raise ValueError("fine approach requires both current visible markers")
    bin_from_box = np.linalg.inv(poses[21]) @ poses[22]
    vertices = np.concatenate([np.asarray(p["points"]) for p in objects["t2_box"]["convex_parts"]])
    points = vertices @ bin_from_box[:3, :3].T + bin_from_box[:3, 3]
    floor = np.asarray(objects["t2_bin"]["convex_parts"][2]["points"])
    lower, upper = floor[:, :2].min(axis=0)+.02, floor[:, :2].max(axis=0)-.02
    direction = poses[21][:3, :3].T @ rotation(observation["measured_joints"]["root_rotation_wxyz"]).T @ np.array([math.cos(heading), math.sin(heading), 0.])
    current_margin = float(min((points[:, :2]-(lower-.02)).min(), ((upper+.02)-points[:, :2]).min()))
    # Compute the complete geometric interval before applying the existing
    # five-command travel budget. Clipping hi first can erase a feasible target
    # or turn a wide interval into a falsely blocked narrow one. Each command
    # still advances only .15 m and requires another current image.
    lo, hi = 0., math.inf
    for axis in (0, 1):
        for coordinate in points[:, axis]:
            if abs(direction[axis]) < 1e-8:
                if not lower[axis] <= coordinate <= upper[axis]:
                    return None
            else:
                ends = sorted([(lower[axis]-coordinate)/direction[axis], (upper[axis]-coordinate)/direction[axis]])
                lo, hi = max(lo, ends[0]), min(hi, ends[1])
    if hi < lo or not math.isfinite(hi) or lo > 5 * .15:
        return None
    aligned = current_margin >= .02
    blocked = not aligned and (hi-lo < .04 or hi < .15)
    physical_distance = None if aligned or blocked else .15
    shifted = points[:, :2] if physical_distance is None else points[:, :2]+physical_distance*direction[:2]
    margin = float(min((shifted-(lower-.02)).min(), ((upper+.02)-shifted).min()))
    return {"schema": "g1_visible_marker_fine_carry_v2", "observation": observation["stamp"],
            "source_geometry_sha256": TASK_GEOMETRY_SHA256, "world_or_contact_truth_input": False,
            "state": "aligned" if aligned else "blocked" if blocked else "advance",
            "current_floor_margin_m": current_margin, "maximum_physical_step_m": .15,
            "physical_containment_distance_interval_m": [lo, hi], "selected_physical_distance_m": physical_distance,
            "existing_navigation_stop_margin_m": .05, "predicted_floor_margin_m": margin,
            "step_is_final": bool(physical_distance is not None and lo <= physical_distance <= hi),
            "new_image_required_after_step": True, "task_qualified": False,
            "goal": None if physical_distance is None else {"observation": observation["stamp"],
            "heading_yaw_source_rad": heading, "relative_distance_m": .1}}


def localize(image_path, observation_path, definition_path, geometry_path=None, memory_path=None, box_view_only=False, fiducial_path=None, placement_view_only=False, *, public_assets=None, secondary_image_path=None, secondary_observation_path=None, held_box_feedback=False, grasp_centering=False, held_contact_geometry=False):
    if held_contact_geometry and not held_box_feedback:
        raise ValueError("hand separation requires explicitly requested current paired held feedback")
    if grasp_centering and (secondary_image_path is None or secondary_observation_path is None
            or geometry_path is None or memory_path is not None or box_view_only or placement_view_only
            or held_box_feedback):
        raise ValueError("grasp centering requires current paired RGB and original public geometry")
    if held_box_feedback and (secondary_image_path is None or secondary_observation_path is None
            or geometry_path is None or memory_path is not None or box_view_only or placement_view_only):
        raise ValueError("held-box feedback requires current paired RGB and normal public-geometry mode")
    if box_view_only and (geometry_path is not None or memory_path is not None):
        raise ValueError("current box view cannot request navigation geometry or target memory")
    if Path(image_path).stat().st_size > 16 * 1024 * 1024 or Path(observation_path).stat().st_size > 128 * 1024:
        raise ValueError("vision input exceeds finite image/self-state byte budget")
    observation = json.loads(Path(observation_path).read_text())
    allowed = {"schema", "stamp", "camera", "measured_joints"}
    if set(observation) not in (allowed, allowed | {"camera_mount_profile"}) or observation["schema"] != "g1_mobile_marker_observation_v1":
        raise ValueError("vision accepts only whitelisted RGB calibration and self sensors")
    stamp = observation["stamp"]
    if set(stamp) != {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"} or any(stamp[k] <= 0 for k in stamp) or stamp["sim_time_ns"] % 20_000_000:
        raise ValueError("invalid current camera identity/time")
    state = observation["measured_joints"]
    if set(state) != {"positions", "velocities", "root_rotation_wxyz", "root_angular_velocity_body", "root_velocity_source"}:
        raise ValueError("foreign self-state fields")
    if len(state["velocities"]) != 43 or not np.isfinite(state["velocities"]).all():
        raise ValueError("invalid original self velocities")
    for key in ("root_angular_velocity_body", "root_velocity_source"):
        if len(state[key]) != 3 or not np.isfinite(state[key]).all():
            raise ValueError("invalid original self motion sensor")
    if public_sha(definition_path, public_assets) != DEFINITION_SHA256:
        raise ValueError("original robot definition identity mismatch")
    definition = public_json(definition_path, public_assets)
    camera_profile = observation.get("camera_mount_profile", "arena_ego")
    if camera_profile == "auxiliary_pregrasp_overview" and not (held_contact_geometry or grasp_centering):
        raise ValueError("pregrasp camera requires its explicit geometry diagnostic")
    marker_sizes, marker_mounts, layout_profile, fiducial_hash = fixed_marker_layout(fiducial_path, camera_profile, public_assets)
    if placement_view_only and (box_view_only or memory_path is not None or geometry_path is None or layout_profile not in ("auxiliary_grip_targets", BIN_BOARD_PROFILE)):
        raise ValueError("placement view requires its bound auxiliary labels/public geometry and no other mode")
    camera_in_root = root_from_camera(definition, state["positions"], camera_profile)
    root_rotation = rotation(state["root_rotation_wxyz"])
    c = observation["camera"]
    if set(c) != {"fx", "fy", "cx", "cy", "vertical_fov_radians", "near_m", "far_m"}:
        raise ValueError("foreign camera calibration fields")
    if not all(abs(c[k]-v) < 1e-6 for k, v in {"fx":458.1245526,"fy":458.1245526,"cx":320.,"cy":240.}.items()):
        raise ValueError("original pinhole contract mismatch")
    intrinsics = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.]])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError("actual ego PNG requires640x480RGB")
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
    corners, ids, rejected = detector.detectMarkers(image)
    box_pair = None
    if secondary_image_path is not None or secondary_observation_path is not None:
        if (secondary_image_path is None or secondary_observation_path is None
                or memory_path is not None or (geometry_path is None and not box_view_only)
                or layout_profile not in ("auxiliary_grip_targets", BIN_BOARD_PROFILE)):
            raise ValueError("box pair requires its explicit auxiliary geometry mode")
        parent = Path(observation_path).resolve(strict=True).parent
        for path, expected, limit in ((secondary_image_path, parent/"secondary"/"ego.png", 16*1024*1024),
                                      (secondary_observation_path, parent/"secondary"/"observation.json", 128*1024)):
            path = Path(path)
            if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
                    or path.resolve(strict=True) != expected or path.stat().st_size > limit):
                raise ValueError("box pair secondary input is outside its owned finite capture directory")
        secondary = json.loads(Path(secondary_observation_path).read_text())
        validate_mobile_box_pair_observations(observation, secondary)
        other_image = cv2.imread(str(secondary_image_path), cv2.IMREAD_COLOR)
        if other_image is None or other_image.shape != (480, 640, 3):
            raise ValueError("box pair secondary image requires640x480RGB")
        other_corners, other_ids, _ = detector.detectMarkers(other_image)
        pixels = []
        for cs, identities in ((corners, ids), (other_corners, other_ids)):
            found = [c.reshape(4, 2).astype(np.float64) for c, identity in
                     zip(cs, [] if identities is None else identities.flatten()) if int(identity) == 22]
            if len(found) != 1:
                raise ValueError("box pair requires exactly one current22label in each image")
            pixels.append(found[0])
        other_camera = root_from_camera(definition, secondary["measured_joints"]["positions"],
                                        secondary["camera_mount_profile"])
        box_pair = fit_current_box_camera_pair(pixels, intrinsics, [camera_in_root, other_camera], marker_sizes[22])
        box_pair.update({"observations": [stamp, secondary["stamp"]],
                         "image_sha256": [sha(image_path), sha(secondary_image_path)],
                         "input_sha256": [sha(observation_path), sha(secondary_observation_path)],
                         "camera_mount_profiles": [camera_profile, secondary["camera_mount_profile"]]})
    board = None
    if layout_profile == BIN_BOARD_PROFILE and not box_view_only and not grasp_centering and not held_contact_geometry:
        board_pixels = {}
        for marker_corners, marker_id in zip(corners, [] if ids is None else ids.flatten()):
            marker_id = int(marker_id)
            if marker_id in (21, 23, 24):
                if marker_id in board_pixels:
                    raise ValueError("duplicate identity in current bin board RGB")
                board_pixels[marker_id] = marker_corners.reshape(4, 2).astype(np.float64)
        board = fit_public_bin_board(board_pixels, intrinsics, camera_in_root, marker_mounts[21])
    detections = []
    for marker_corners, marker_id in zip(corners, [] if ids is None else ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in marker_sizes:
            continue
        pixels = marker_corners.reshape(4, 2).astype(np.float64)
        shortest_edge = float(np.linalg.norm(pixels-np.roll(pixels, -1, axis=0), axis=1).min())
        h = marker_sizes[marker_id] / 2
        object_points = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]])
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_points, pixels, intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        if ok:
            for rvec, tvec in zip(rvecs, tvecs):
                r = cv2.Rodrigues(rvec)[0]
                t = tvec.reshape(3)
                projected = cv2.projectPoints(object_points, rvec, tvec, intrinsics, None)[0].reshape(4, 2)
                error = float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))
                if 0.1 < t[2] < 5. and float(r[:, 2] @ t) < 0:
                    camera_from_marker = np.eye(4)
                    camera_from_marker[:3, :3] = r
                    camera_from_marker[:3, 3] = t
                    candidates.append((error, camera_from_marker))
        if not candidates or shortest_edge < 8:
            continue
        error, camera_from_marker = min(candidates, key=lambda item: item[0])
        if error > 1.0:
            continue
        marker_in_root = camera_in_root @ camera_from_marker
        if box_pair is not None and marker_id == 22:
            marker_in_root = np.asarray(box_pair["root_from_marker"])
            error = box_pair["per_view_reprojection_rms_px"][0]
        if board is not None and marker_id == 21:
            marker_in_root = np.asarray(board["root_from_marker"])
            error = board["per_marker_reprojection_rms_px"][0]
        detections.append({"marker_id": marker_id, "kind": "target_bin" if marker_id == 21 else "carried_box",
                           "corners_px": pixels.tolist(), "minimum_edge_px": shortest_edge,
                           "reprojection_rms_px": error, "root_from_marker": marker_in_root.tolist(),
                           "root_position_m": marker_in_root[:3, 3].tolist()})
    if len({d["marker_id"] for d in detections}) != len(detections):
        raise ValueError("duplicate identity in actual marker image")
    proposal = None
    fine_proposal = None
    by_id = {d["marker_id"]: np.array(d["root_from_marker"]) for d in detections}
    memory_estimate = propagate_target_memory(memory_path, observation) if memory_path is not None else None
    used_memory = memory_estimate is not None and 21 not in by_id
    if memory_estimate is not None:
        if 22 not in by_id:
            raise ValueError("target memory cannot replace a current carried-box image")
        if 21 in by_id:
            if np.linalg.norm(by_id[21][:3, 3] - np.asarray(memory_estimate["root_from_marker"])[:3, 3]) > .03:
                raise ValueError("current visible target moved outside static-memory tolerance")
        else:
            by_id[21] = np.asarray(memory_estimate["root_from_marker"])
    if not box_view_only and not placement_view_only and not grasp_centering and not held_contact_geometry and 21 in by_id and 22 in by_id:
        if layout_profile == "original_arena":
            bin_p = by_id[21][:3, 3]
            box_p = (by_id[22] @ np.array([0., 0., -0.1005, 1.]))[:3]
        else:
            # Recover object coordinates from the published attachment frames.
            # The target point remains the original bin floor marker center.
            bin_p = (by_id[21] @ np.linalg.inv(marker_mounts[21]) @ np.array([.008351, .0113635, .0045, 1.]))[:3]
            box_p = (by_id[22] @ np.linalg.inv(marker_mounts[22]))[:3, 3]
        # Rotate relative vectors into the gravity/yaw-aligned source plane.
        bin_world = root_rotation @ bin_p
        box_world = root_rotation @ box_p
        heading = float(np.arctan2(bin_world[1], bin_world[0]))
        distance = float(np.linalg.norm(bin_world[:2])-np.linalg.norm(box_world[:2]))
        if camera_profile in ("auxiliary_grip_overview", "auxiliary_bin_placement") and geometry_path is not None and memory_path is None:
            fine_proposal = fine_from_visible_markers(detections, observation, geometry_path, marker_mounts, heading, public_assets)
        if not (0.1 <= distance <= 2.5):
            # Whole-box current containment admits no walking command. The
            # original navigation minimum remains in force for all movement.
            if fine_proposal is None or fine_proposal["state"] != "aligned":
                raise ValueError("visual approach lies outside bounded navigation envelope")
        else:
            proposal = {"observation": stamp, "heading_yaw_source_rad": heading,
                        "relative_distance_m": distance, "source": "actual_box_rgb_and_static_target_memory_self_velocity" if used_memory else "actual_rgb_marker_pnp_and_original_self_fk",
                        "target_identity": 21, "task_qualified": False, "automatically_executed": False}
    result = {"schema":"g1_mobile_actual_marker_localization_v1", "observation":stamp,
            "image_sha256":sha(image_path), "input_sha256":sha(observation_path),
            "robot_definition_sha256":DEFINITION_SHA256, "opencv_version":cv2.__version__,
            "actual_rgb_only_object_measurement":True, "world_or_contact_truth_input":False,
            "camera_in_root_source":camera_in_root.tolist(), "detections":detections,
            "original_self_root_rotation_wxyz":state["root_rotation_wxyz"],
            "rejected_marker_candidates":len(rejected), "navigation_proposal":proposal,
            "target_bin_detected":any(d["marker_id"] == 21 for d in detections), "carried_box_detected":22 in by_id,
            "task_qualified":False}
    if box_view_only or camera_profile in ("auxiliary_grip_overview", "auxiliary_bin_placement", "auxiliary_pregrasp_overview"):
        if box_view_only:
            result["box_view_only"] = True
        if 22 in by_id:
            frames = original_self_body_frames(definition, state["positions"])
            midpoint = (frames[28][:3, 3] + frames[45][:3, 3]) * .5
            center = (by_id[22] @ np.linalg.inv(marker_mounts[22]))[:3, 3]
            result["current_box_palm_center_distance_m"] = float(np.linalg.norm(center-midpoint))
            if (box_view_only or held_box_feedback) and box_pair is not None:
                # Finite pickup feedback uses only these measured RGB/self
                # vectors. It never receives shelf/contact/world body poses.
                left_relative = np.linalg.inv(frames[28]) @ np.r_[center, 1.]
                if box_view_only:
                    result["current_box_gravity_center_m"] = (root_rotation @ center).tolist()
                    result["current_midpalm_gravity_center_m"] = (root_rotation @ midpoint).tolist()
                    result["current_box_relative_left_palm_m"] = left_relative[:3].tolist()
                if held_box_feedback:
                    result["held_box_feedback"] = {
                        "schema": "g1_current_paired_rgb_relative_grip_v1", "observation": stamp,
                        "current_box_relative_left_palm_m": left_relative[:3].tolist(),
                        "world_or_contact_truth_input": False, "task_qualified": False}
                    if held_contact_geometry:
                        from unitree_g1_mobile_hand_geometry import measure
                        result["held_contact_geometry"] = measure(
                            definition, observation, by_id[22] @ np.linalg.inv(marker_mounts[22]),
                            public_json(geometry_path, public_assets))
    if fiducial_path is not None:
        result["fiducial_calibration_sha256"] = fiducial_hash
        result["marker_layout_profile"] = layout_profile
        result["camera_mount_profile"] = camera_profile
    if board is not None:
        result["public_bin_board_pose"] = board
    if box_pair is not None:
        result["same_tick_box_pair_pose"] = box_pair
    if grasp_centering:
        if box_pair is None or 22 not in by_id:
            raise ValueError("grasp centering lost the current paired visible box")
        from unitree_g1_mobile_grasp import propose
        result["grasp_centering_proposal"] = propose(
            definition, observation, by_id[22] @ np.linalg.inv(marker_mounts[22]),
            public_json(geometry_path, public_assets))
    if memory_estimate is not None:
        result["target_memory_estimate"] = memory_estimate
        result["target_memory_used"] = used_memory
    if geometry_path is not None and not used_memory and not grasp_centering and not held_contact_geometry:
        result["clearance_proposal"] = clearance_from_visible_markers(detections, observation, geometry_path, marker_mounts, public_assets)
        if camera_profile in ("auxiliary_grip_overview", "auxiliary_bin_placement") and not placement_view_only:
            result["fine_approach_proposal"] = fine_proposal
    if placement_view_only:
        result["placement_view_only"] = True
        result["release_proposal"] = placement_from_visible_markers(detections, observation, geometry_path, marker_mounts, definition, public_assets)
        if camera_profile == "auxiliary_bin_placement":
            result["thumb_preparation_proposal"] = thumb_preparation_from_visible_markers(
                detections, observation, geometry_path, marker_mounts, definition,
                result["release_proposal"], public_assets)
    if held_contact_geometry and "held_contact_geometry" not in result:
        raise ValueError("requested hand geometry lost the current paired visible box")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--definition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--target-memory", type=Path)
    parser.add_argument("--box-view-only", action="store_true")
    parser.add_argument("--fiducials", type=Path)
    parser.add_argument("--placement-view-only", action="store_true")
    parser.add_argument("--secondary-image", type=Path)
    parser.add_argument("--secondary-observation", type=Path)
    parser.add_argument("--held-box-feedback", action="store_true")
    parser.add_argument("--grasp-centering", action="store_true")
    parser.add_argument("--held-contact-geometry", action="store_true")
    args = parser.parse_args()
    result = localize(args.image, args.observation, args.definition, args.geometry, args.target_memory, args.box_view_only, args.fiducials, args.placement_view_only,
                      secondary_image_path=args.secondary_image, secondary_observation_path=args.secondary_observation,
                      held_box_feedback=args.held_box_feedback, grasp_centering=args.grasp_centering,
                      held_contact_geometry=args.held_contact_geometry)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("target_bin_detected", "carried_box_detected", "navigation_proposal", "task_qualified")}))


if __name__ == "__main__":
    main()
