#!/usr/bin/env python3
"""Conservative independent geometry/release audit of original SDK evidence.

Source200/50Hz results cannot qualify native50Hz Rapier. Release requires
positive same-Tick AABB separation from every exported original robot collider;
overlapping bounds are inconclusive, even if the contact report has zero force.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from unitree_g1_joint_kinematics_audit import rotation
from unitree_g1_static_placement_audit import DEFINITION_SHA, digest, footprint

ROBOT_SHA = '571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd'


def sdk_rotation(pose):
    return rotation([pose[6], *pose[3:6]])


def collision_bounds(collision, pose):
    body_rotation = sdk_rotation(pose)
    body_position = np.asarray(pose[:3])
    shape = collision['shape']
    if shape == 'convex_hull':
        points = np.asarray(collision['points']) @ body_rotation.T + body_position
        return points.min(axis=0), points.max(axis=0)
    local = collision['local_pose']
    center = np.asarray(local['position']) @ body_rotation.T + body_position
    orientation = body_rotation @ rotation(local['rotation_wxyz'])
    if shape == 'sphere':
        extent = np.full(3, collision['radius'])
    elif shape == 'capsule':
        axis = {'X': 0, 'Y': 1, 'Z': 2}[collision['axis']]
        extent = np.abs(orientation[:, axis]) * collision['half_height'] + collision['radius']
    elif shape == 'box':
        extent = np.abs(orientation) @ collision['half_extents']
    else:
        raise ValueError('Unsupported original source collision shape')
    return center - extent, center + extent


def aabb_gap(left, right):
    distances = np.maximum(np.maximum(right[0] - left[1], left[0] - right[1]), 0.)
    return float(np.linalg.norm(distances))


def evaluate(definition, robot, receipt, rows):
    metadata = receipt['placement_contact_view']
    if (not receipt['success_auto_reset_suppressed_for_audit_only']
            or not receipt['original_drop_and_timeout_terms_unchanged']
            or receipt['source_physics_hz'] != 200 or receipt['source_control_hz'] != 50
            or metadata['schema_writes'] != 0 or metadata['filter_count'] != 54):
        raise ValueError('Foreign source audit contract')
    names = metadata['robot_body_names']
    indices = [names.index(body['name']) for body in robot['bodies']]
    objects = {o['kind']: o for o in definition['objects']}
    apple_vertices = np.asarray([v for p in objects['t1_apple']['convex_parts'] for v in p['points']])
    polygon = footprint([v for p in objects['t1_plate']['convex_parts'] for v in p['points']])
    edges = np.roll(polygon, -1, axis=0) - polygon
    inward = np.column_stack((-edges[:, 1], edges[:, 0]))
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    samples = []
    previous = -1
    suffix = maximum = 0
    fell = False
    for row in rows:
        tick = row['control_tick']
        if tick != previous + 1 or row['physics_steps_since_reset'] != tick * 4:
            raise ValueError('Source trace reset, gap or extra integration')
        previous = tick
        truth = row['acceptance_truth_only']
        apple = truth['apple_01_objaverse_robolab']
        plate = truth['clay_plates_hot3d_robolab']
        contacts = row['acceptance_contacts_only']
        poses = row['acceptance_robot_body_link_poses_w']
        matrix = np.asarray(contacts['apple_filter_force_matrix_world_n'])
        if matrix.shape != (54, 3) or len(poses) != 53 or contacts['source_solve_dt_s'] != .005:
            raise ValueError('Incomplete original contact/self-body evidence')
        vertices_world = apple_vertices @ sdk_rotation(apple['pose']).T + apple['pose'][:3]
        vertices_plate = (vertices_world - plate['pose'][:3]) @ sdk_rotation(plate['pose'])
        margins = vertices_plate[:, :2] @ inward.T - np.sum(polygon * inward, axis=1)
        apple_bounds = vertices_world.min(axis=0), vertices_world.max(axis=0)
        gaps = [aabb_gap(apple_bounds, collision_bounds(c, poses[indices[c['body']]]))
                for c in robot['collisions']]
        if not np.isfinite([*margins.ravel(), *gaps, *matrix.ravel()]).all():
            raise ValueError('Nonfinite geometry/contact evidence')
        contact_points = contacts['contact_points']
        reconstructed = np.zeros_like(matrix)
        for point in contact_points:
            # SDK normal scalars are signed. The product with the reported
            # normal, not a positive-only scalar test, is the body force.
            reconstructed[point['filter_index']] += (
                point['normal_force_n'] * np.asarray(point['normal_world']))
        force_error = float(np.max(np.abs(reconstructed - matrix)))
        if not np.isfinite(force_error) or force_error > 1e-4:
            raise ValueError('Detailed signed contact forces disagree with measured pair matrix')
        robot_force = float(np.linalg.norm(matrix[1:], axis=1).sum())
        supported = bool(matrix[0, 2] > 0 and any(
            p['filter_index'] == 0 and abs(p['normal_force_n']) > 0 for p in contact_points))
        touching_robot_report = any(p['filter_index'] > 0 and (
            p['separation_m'] <= 0 or abs(p['normal_force_n']) > 0) for p in contact_points)
        separated = min(gaps) > 0 and robot_force == 0 and not touching_robot_report
        linear = float(np.linalg.norm(apple['velocity'][:3]))
        angular = float(np.linalg.norm(apple['velocity'][3:]))
        standing = row['upright'] > .95 and row['root_link_pose_w'][2] + .795 >= .35
        fell |= not standing
        ready = tick > 0 and margins.min() >= 0 and supported and separated and (
            linear < .02 and angular < .1 and standing and not fell)
        suffix = suffix + 1 if ready else 0
        maximum = max(maximum, suffix)
        samples.append({'tick': tick, 'minimum_footprint_margin_m': float(margins.min()),
            'positive_plate_upward_force_n': float(matrix[0, 2]), 'plate_support': supported,
            'robot_contact_force_norm_sum_n': robot_force,
            'signed_contact_pair_reconstruction_max_error_n': force_error,
            'minimum_robot_aabb_gap_m': min(gaps), 'separation_proved': separated,
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular,
            'standing': standing, 'ready': bool(ready),
            'stable_seconds': max(suffix - 1, 0) / 50.})
    if not samples:
        raise ValueError('No actual source samples')
    return {'schema': 'g1_original_source_conservative_placement_audit_v1',
            'source_diagnostic_placement_passed': maximum >= 101,
            'max_stable_seconds': max(maximum - 1, 0) / 50.,
            'completed_source_controls': receipt['completed_control_ticks'],
            'actual_source_physics_integrations': receipt['measured_physics_steps_since_reset'],
            'source_samples': len(samples), 'fresh_original_vla_calls': receipt['policy_calls'],
            'apple_collision_vertices': len(apple_vertices), 'robot_collision_shapes': len(robot['collisions']),
            'release_rule': 'strictly positive same-Tick exported-shape AABB separation from all robot colliders and no touching/nonzero signed contact report; overlap is inconclusive',
            'samples': samples, 'native_task_qualified': False, 'formal_task_qualified': False,
            'scope': 'original SDK200/50Hz independent audit only; no model/command input'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('definition', 'robot', 'receipt', 'trace', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if digest(args.definition) != DEFINITION_SHA or digest(args.robot) != ROBOT_SHA:
        parser.error('Original geometry identity changed')
    result = evaluate(json.loads(args.definition.read_text()), json.loads(args.robot.read_text()),
        json.loads(args.receipt.read_text()), [json.loads(x) for x in args.trace.read_text().splitlines()])
    result['input_sha256'] = {name: digest(getattr(args, name)) for name in ('definition', 'robot', 'receipt', 'trace')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))
    if not result['source_diagnostic_placement_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
