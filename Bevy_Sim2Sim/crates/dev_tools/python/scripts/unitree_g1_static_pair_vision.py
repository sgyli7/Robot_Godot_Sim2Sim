#!/usr/bin/env python3
"""Disclosed traditional same-Tick fixed-camera RGB geometry; no world inputs."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from unitree_g1_static_vision import (
    PinnedPublicStaticVisionAssets, localize, observation, sha,
)


def localize_fixed_pair(first, second, definition, fiducials, fiducial_sha256):
    folders = [Path(first), Path(second)]
    values = [observation(folder / 'observation.json') for folder in folders]
    a, b = [value['stamp'] for value in values]
    if (a['episode_id'] != b['episode_id'] or a['sim_time_ns'] != b['sim_time_ns']
            or a['frame_id'] >= b['frame_id']
            or not 0 < b['captured_at_unix_ms'] - a['captured_at_unix_ms'] <= 2000
            or [value['camera']['mount_profile'] for value in values]
            != ['static_placement_overview', 'arena_ego']):
        raise ValueError('Fixed pair changed episode, Tick, frame or public mount')
    for name in values[0]['measured_joints']:
        if not np.array_equal(np.asarray(values[0]['measured_joints'][name], np.float32),
                              np.asarray(values[1]['measured_joints'][name], np.float32)):
            raise ValueError('Fixed pair changed original self measurements')
    assets = PinnedPublicStaticVisionAssets(definition, fiducials, fiducial_sha256)
    views = [localize(folder / 'ego.png', folder / 'observation.json', definition,
                      fiducials, fiducial_sha256, public_assets=assets) for folder in folders]
    cameras = [assets.root_camera(value['measured_joints']['positions'], value['camera']['mount_profile'])
               for value in values]
    baseline = float(np.linalg.norm(cameras[0][:3, 3] - cameras[1][:3, 3]))
    if abs(baseline - .15) > 1e-7:
        raise ValueError('Fixed pair changed the published 15cm baseline')
    inverse_cameras = [np.linalg.inv(camera) for camera in cameras]
    detections = []
    for kind, marker in [('t1_apple', 31), ('t1_plate', 32)]:
        entries = [next((entry for entry in view['detections']
                         if entry['object_kind'] == kind and entry['marker_id'] == marker), None)
                   for view in views]
        if any(entry is None for entry in entries):
            raise ValueError(f'{kind} common physical marker missing from actual pair')
        _, size, mount, mount_rotation = assets.layout[marker]
        half = size / 2
        points = np.array([[-half, half, 0.], [half, half, 0.],
                           [half, -half, 0.], [-half, -half, 0.]]) @ mount_rotation.T + mount
        pixels = [np.asarray(entry['corners_px']) for entry in entries]
        initial = np.asarray(entries[0]['root_from_object'])
        initial_parameters = np.r_[cv2.Rodrigues(initial[:3, :3])[0].ravel(), initial[:3, 3]]

        def project(parameters, index):
            root_points = points @ cv2.Rodrigues(parameters[:3])[0].T + parameters[3:]
            camera_points = (root_points @ inverse_cameras[index][:3, :3].T
                             + inverse_cameras[index][:3, 3])
            return (camera_points[:, :2] / camera_points[:, 2:] * [458.1245526, 458.1245526]
                    + [320., 240.]), camera_points

        def residual(parameters):
            return np.concatenate([(project(parameters, index)[0] - pixels[index]).ravel()
                                   for index in range(2)])

        fit = least_squares(residual, initial_parameters, max_nfev=100)
        per_view_rms = [float(np.sqrt(np.mean(np.sum((project(fit.x, index)[0] - pixels[index]) ** 2, axis=1))))
                        for index in range(2)]
        if not fit.success or not np.isfinite(fit.x).all() or max(per_view_rms) > 1.:
            raise ValueError('Fixed pair failed its unchanged one-pixel reprojection gate')
        object_rotation = cv2.Rodrigues(fit.x[:3])[0]
        for index in range(2):
            camera_points = project(fit.x, index)[1]
            normal = inverse_cameras[index][:3, :3] @ object_rotation @ mount_rotation[:, 2]
            if (camera_points[:, 2].min() <= .1 or camera_points[:, 2].max() >= 5.
                    or float(normal @ camera_points.mean(axis=0)) >= 0):
                raise ValueError('Fixed pair failed depth or printed-face orientation admission')
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = object_rotation, fit.x[3:]
        detections.append(dict(object_kind=kind, marker_id=marker, root_from_object=pose.tolist(),
                               reprojection_rms_px=float(np.sqrt(np.mean(np.sum(fit.fun.reshape(-1, 2) ** 2, axis=1)))),
                               per_view_reprojection_rms_px=per_view_rms,
                               minimum_edge_px=min(entry['minimum_edge_px'] for entry in entries),
                               optimizer_evaluations=fit.nfev))
    return dict(schema='g1_static_actual_fixed_pair_rgb_localization_v1',
                source='public_existing_fixed_camera_pair_joint_pattern_v1',
                observations=[value['stamp'] for value in values],
                image_sha256=[view['image_sha256'] for view in views],
                input_sha256=[view['input_sha256'] for view in views],
                definition_sha256=views[0]['definition_sha256'], fiducial_sha256=fiducial_sha256,
                camera_profiles=['static_placement_overview', 'arena_ego'],
                public_camera_baseline_m=baseline, detections=detections,
                world_or_contact_truth_input=False, actuation_proposed=False, task_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['first', 'second', 'definition', 'fiducials', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fiducial-sha256', required=True)
    args = parser.parse_args()
    if args.first.resolve().parent != args.second.resolve().parent or args.output.exists():
        raise ValueError('Fixed pair requires owned same-episode inputs and fresh output')
    for folder in [args.first, args.second]:
        if any((folder / name).is_symlink() for name in ['ego.png', 'observation.json']):
            raise ValueError('Fixed pair inputs must be regular owned files')
    result = localize_fixed_pair(args.first, args.second, args.definition,
                                 args.fiducials, args.fiducial_sha256)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(result, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
