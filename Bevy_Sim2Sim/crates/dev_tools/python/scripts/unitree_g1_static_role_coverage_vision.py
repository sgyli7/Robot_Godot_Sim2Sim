#!/usr/bin/env python3
"""Separate disclosed role coverage: paired pattern or one decoded Arena apple.

No world/contact/pose truth enters this program. The original two-view method
remains unchanged. A single-view dictionary result is explicitly identified,
never described as stereo or full-pattern registration.
"""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from unitree_g1_static_pair_vision import localize_fixed_pair
from unitree_g1_static_vision import PinnedPublicStaticVisionAssets, localize, observation


def decoded_arena_apple(folder, value, assets):
    image = cv2.imread(str(folder / 'ego.png'), cv2.IMREAD_COLOR)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
    corners, ids, _ = detector.detectMarkers(image)
    entries = [quad.reshape(4, 2).astype(np.float64)
               for quad, marker in zip(corners, [] if ids is None else ids.ravel()) if marker == 31]
    if len(entries) != 1:
        raise ValueError('Single Arena apple requires one actual dictionary-decoded ID31')
    pixels = entries[0]
    edge = float(np.linalg.norm(pixels - np.roll(pixels, -1, axis=0), axis=1).min())
    if edge < 8.:
        raise ValueError('Single Arena apple failed unchanged eight-pixel gate')
    _, size, mount, mount_rotation = assets.layout[31]
    half = size / 2
    points = np.array([[-half, half, 0.], [half, half, 0.],
                       [half, -half, 0.], [-half, -half, 0.]])
    intrinsics = np.array([[458.1245526, 0., 320.], [0., 458.1245526, 240.], [0., 0., 1.]])
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        points, pixels, intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    for rv, tv in zip(rotations if ok else [], translations if ok else []):
        rotation = cv2.Rodrigues(rv)[0]
        translation = tv.ravel()
        depths = (points @ rotation.T + translation)[:, 2]
        projected = cv2.projectPoints(points, rv, tv, intrinsics, None)[0].reshape(4, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
        if (np.isfinite(rms) and rms <= 1. and depths.min() > .1
                and depths.max() < 5. and float(rotation[:, 2] @ translation) < 0):
            camera_marker = np.eye(4)
            camera_marker[:3, :3], camera_marker[:3, 3] = rotation, translation
            object_marker = np.eye(4)
            object_marker[:3, :3], object_marker[:3, 3] = mount_rotation, mount
            pose = (assets.root_camera(value['measured_joints']['positions'], 'arena_ego')
                    @ camera_marker @ np.linalg.inv(object_marker))
            candidates.append((rms, pose))
    if not candidates:
        raise ValueError('Single Arena apple failed unchanged depth/front-face/reprojection gate')
    rms, pose = min(candidates, key=lambda entry: entry[0])
    return dict(object_kind='t1_apple', marker_id=31, root_from_object=pose.tolist(),
                reprojection_rms_px=rms, per_view_reprojection_rms_px=[rms],
                minimum_edge_px=edge, optimizer_evaluations=0, observed_view_indices=[1],
                pixel_method='standard_dictionary_subpixel_corner_pnp_v1')


def localize_role_coverage(first, second, definition, fiducials, fiducial_sha256):
    folders = [Path(first), Path(second)]
    values = [observation(folder / 'observation.json') for folder in folders]
    # The original pair function performs the exact same-Tick/self/profile,
    # public asset and baseline validation. Do not catch other failures.
    try:
        result = localize_fixed_pair(first, second, definition, fiducials, fiducial_sha256)
    except ValueError as error:
        if str(error) != 't1_apple common physical marker missing from actual pair':
            raise
        a, b = [value['stamp'] for value in values]
        if (a['episode_id'] != b['episode_id'] or a['sim_time_ns'] != b['sim_time_ns']
                or a['frame_id'] >= b['frame_id']
                or not 0 < b['captured_at_unix_ms'] - a['captured_at_unix_ms'] <= 2000
                or [value['camera']['mount_profile'] for value in values]
                != ['static_placement_overview', 'arena_ego']):
            raise ValueError('Role coverage pair changed native identity or published mounts')
        for name in values[0]['measured_joints']:
            if not np.array_equal(np.asarray(values[0]['measured_joints'][name], np.float32),
                                  np.asarray(values[1]['measured_joints'][name], np.float32)):
                raise ValueError('Role coverage pair changed original self sensors')
        assets = PinnedPublicStaticVisionAssets(definition, fiducials, fiducial_sha256)
        views = [localize(folder / 'ego.png', folder / 'observation.json', definition,
                          fiducials, fiducial_sha256, public_assets=assets) for folder in folders]
        cameras = [assets.root_camera(value['measured_joints']['positions'], value['camera']['mount_profile'])
                   for value in values]
        baseline = float(np.linalg.norm(cameras[0][:3, 3] - cameras[1][:3, 3]))
        if abs(baseline - .15) > 1e-7:
            raise ValueError('Role coverage changed published fifteen-centimeter baseline')
        # Reuse the exact original plate-only fitting arithmetic by creating
        # no synthetic apple observation: fit only the actual plate pixels.
        from scipy.optimize import least_squares
        plates = [next((d for d in view['detections'] if d['object_kind'] == 't1_plate'
                        and d['marker_id'] == 32), None) for view in views]
        if any(plate is None for plate in plates):
            raise ValueError('Role coverage requires actual plate32 in both views')
        _, size, mount, mount_rotation = assets.layout[32]
        half = size / 2
        points = np.array([[-half, half, 0.], [half, half, 0.],
                           [half, -half, 0.], [-half, -half, 0.]]) @ mount_rotation.T + mount
        pixels = [np.asarray(plate['corners_px']) for plate in plates]
        inverse_cameras = [np.linalg.inv(camera) for camera in cameras]
        initial = np.asarray(plates[0]['root_from_object'])
        parameters = np.r_[cv2.Rodrigues(initial[:3, :3])[0].ravel(), initial[:3, 3]]
        def project(p, index):
            root_points = points @ cv2.Rodrigues(p[:3])[0].T + p[3:]
            camera_points = root_points @ inverse_cameras[index][:3, :3].T + inverse_cameras[index][:3, 3]
            return camera_points[:, :2] / camera_points[:, 2:] * [458.1245526, 458.1245526] + [320., 240.], camera_points
        def residual(p):
            return np.concatenate([(project(p, i)[0] - pixels[i]).ravel() for i in range(2)])
        fit = least_squares(residual, parameters, max_nfev=100)
        errors = [float(np.sqrt(np.mean(np.sum((project(fit.x, i)[0] - pixels[i]) ** 2, axis=1)))) for i in range(2)]
        if not fit.success or not np.isfinite(fit.x).all() or max(errors) > 1.:
            raise ValueError('Role coverage plate failed unchanged one-pixel gate')
        rotation = cv2.Rodrigues(fit.x[:3])[0]
        for index in range(2):
            camera_points = project(fit.x, index)[1]
            normal = inverse_cameras[index][:3, :3] @ rotation @ mount_rotation[:, 2]
            if camera_points[:, 2].min() <= .1 or camera_points[:, 2].max() >= 5. or float(normal @ camera_points.mean(axis=0)) >= 0:
                raise ValueError('Role coverage plate failed original depth/front-face gate')
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, fit.x[3:]
        plate = dict(object_kind='t1_plate', marker_id=32, root_from_object=pose.tolist(),
                     reprojection_rms_px=float(np.sqrt(np.mean(np.sum(fit.fun.reshape(-1, 2) ** 2, axis=1)))),
                     per_view_reprojection_rms_px=errors, minimum_edge_px=min(p['minimum_edge_px'] for p in plates),
                     optimizer_evaluations=fit.nfev, observed_view_indices=[0, 1],
                     pixel_method='public_fixed_pair_joint_pattern_v1')
        result = dict(observations=[value['stamp'] for value in values],
                      image_sha256=[view['image_sha256'] for view in views],
                      input_sha256=[view['input_sha256'] for view in views],
                      definition_sha256=views[0]['definition_sha256'], fiducial_sha256=fiducial_sha256,
                      camera_profiles=['static_placement_overview', 'arena_ego'], public_camera_baseline_m=baseline,
                      detections=[decoded_arena_apple(folders[1], values[1], assets), plate],
                      world_or_contact_truth_input=False, actuation_proposed=False, task_qualified=False)
    else:
        for detection in result['detections']:
            detection.update(observed_view_indices=[0, 1], pixel_method='public_fixed_pair_joint_pattern_v1')
    result.update(schema='g1_static_actual_fixed_pair_role_coverage_localization_v2',
                  source='public_fixed_pair_explicit_per_role_view_coverage_v2')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['first', 'second', 'definition', 'fiducials', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fiducial-sha256', required=True)
    args = parser.parse_args()
    if args.first.resolve().parent != args.second.resolve().parent or args.output.exists():
        raise ValueError('Role coverage requires owned same-episode input and fresh output')
    for folder in [args.first, args.second]:
        if any((folder / name).is_symlink() for name in ['ego.png', 'observation.json']):
            raise ValueError('Role coverage inputs must be regular owned files')
    result = localize_role_coverage(args.first, args.second, args.definition, args.fiducials, args.fiducial_sha256)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(result, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
