#!/usr/bin/env python3
"""Compare read-only SDK world Jacobians/FK with a native zero-Tick assembly.

This is an independent mapping diagnostic. SDK truth and measured poses are
never policy inputs, and a native assembly initialized at measured q is not a
task rollout or pose replay. The command fails on a missing/nonunit axis,
changed counter, joint order error or body-pose discrepancy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from unitree_g1_source_expert import JOINT_NAMES

DEFINITION_SHA = '571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd'
BASIS = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
AXES = dict(zip(('X', 'Y', 'Z'), np.eye(3)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float)
    if not np.isfinite(wxyz).all() or abs(np.dot(wxyz, wxyz) - 1) > 2e-6:
        raise ValueError('Nonunit/nonfinite boundary quaternion')
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def transform(position, wxyz):
    out = np.eye(4)
    out[:3, :3] = rotation(wxyz)
    out[:3, 3] = position
    return out


def source_pose(pose):
    return transform(pose['position'], pose['rotation_wxyz'])


def sdk_pose(pose):
    # Pinned Lab Warp body_link_pose_w is [position, quaternion xyzw].
    return transform(pose[:3], [pose[6], *pose[3:6]])


def native_pose(body, translation):
    q = body['rotation_xyzw']
    out = np.eye(4)
    out[:3, :3] = BASIS.T @ rotation([q[3], *q[:3]]) @ BASIS
    out[:3, 3] = BASIS.T @ np.asarray(body['translation']) - translation
    return out


def audit(definition, sdk, stamp, translation):
    if (sdk['schema'] != 'g1_source_joint_kinematics_v1'
            or sdk['before_sdk_counters'] != sdk['after_sdk_counters']
            or sdk['extra_integrations'] != 0 or sdk['coordinate_writes'] != 0
            or sdk['source_physics_steps_since_reset'] != sdk['control_tick'] * 4):
        raise ValueError('Unverified read-only SDK boundary')
    state = stamp['native_state']
    frame = state['body_frame']
    if frame['source_tick'] != 0 or frame['sim_time'] != 0:
        raise ValueError('Native input must be a zero-integration assembly')
    if (len(definition['bodies']) != 53 or len(definition['joints']) != 52
            or len(sdk['body_names']) != 53 or len(set(sdk['body_names'])) != 53
            or len(sdk['joint_names']) != 43 or len(set(sdk['joint_names'])) != 43
            or len(frame['bodies']) != 53):
        raise ValueError('Unexpected G1 topology or duplicate names')
    jac = np.asarray(sdk['world_jacobian_linear_then_angular'])
    if jac.shape != (53, 6, 49) or not np.isfinite(jac).all():
        raise ValueError('Missing/nonfinite floating-base SDK Jacobian')
    q = dict(zip(sdk['joint_names'], sdk['joint_positions'], strict=True))
    if set(q) != set(JOINT_NAMES):
        raise ValueError('SDK joint identities do not match canonical native contract')
    native_q = np.asarray(state['measured_joints']['positions'])
    if native_q.shape != (43,) or not np.isfinite(native_q).all():
        raise ValueError('Native 43-joint measured state is absent')
    q_error = float(np.abs(native_q - np.asarray([q[name] for name in JOINT_NAMES])).max())
    sdk_bodies = {name: sdk_pose(pose) for name, pose in zip(
        sdk['body_names'], sdk['body_link_pose_w'], strict=True)}
    native_bodies = {definition['bodies'][body['body']]['name']: native_pose(body, translation)
                     for body in frame['bodies']}
    fk = {0: sdk_bodies['pelvis']}
    remaining = list(definition['joints'])
    while remaining:
        progress = False
        for joint in remaining[:]:
            if joint['parent'] not in fk:
                continue
            angle = q[joint['name']] if joint['kind'] == 'revolute' else 0.
            motion = np.eye(4)
            axis = AXES[joint['axis']]
            motion[:3, :3] = rotation([np.cos(angle/2), *(axis*np.sin(angle/2))])
            fk[joint['child']] = (fk[joint['parent']] @ source_pose(joint['frame_parent'])
                                  @ motion @ np.linalg.inv(source_pose(joint['frame_child'])))
            remaining.remove(joint)
            progress = True
        if not progress:
            raise ValueError('Disconnected/cyclic source joint tree')
    bodies = []
    for index, body in enumerate(definition['bodies']):
        name = body['name']
        source = sdk_bodies[name]
        native = native_bodies[name]
        bodies.append({'name': name,
            'source_fk_position_error_m': float(np.linalg.norm(fk[index][:3, 3] - source[:3, 3])),
            'source_fk_rotation_matrix_max_error': float(np.abs(fk[index][:3, :3] - source[:3, :3]).max()),
            'native_position_error_m': float(np.linalg.norm(native[:3, 3] - source[:3, 3])),
            'native_rotation_matrix_max_error': float(np.abs(native[:3, :3] - source[:3, :3]).max())})
    joints = []
    for joint in definition['joints']:
        if joint['kind'] != 'revolute':
            continue
        parent = definition['bodies'][joint['parent']]['name']
        child = definition['bodies'][joint['child']]['name']
        axis = rotation(joint['frame_parent']['rotation_wxyz']) @ AXES[joint['axis']]
        column = 6 + sdk['joint_names'].index(joint['name'])
        measured_axis = jac[sdk['body_names'].index(child), 3:6, column]
        expected_source = sdk_bodies[parent][:3, :3] @ axis
        expected_native = native_bodies[parent][:3, :3] @ axis
        joints.append({'name': joint['name'], 'source_q_rad': q[joint['name']],
            'sdk_world_angular_axis': measured_axis.tolist(),
            'sdk_axis_norm': float(np.linalg.norm(measured_axis)),
            'source_axis_error': float(np.linalg.norm(measured_axis - expected_source)),
            'native_axis_error': float(np.linalg.norm(measured_axis - expected_native)),
            'native_axis_dot': float(np.dot(measured_axis, expected_native))})
    metrics = {key: max(item[key] for item in bodies) for key in (
        'source_fk_position_error_m', 'source_fk_rotation_matrix_max_error',
        'native_position_error_m', 'native_rotation_matrix_max_error')}
    metrics.update({key: max(item[key] for item in joints)
                    for key in ('source_axis_error', 'native_axis_error')})
    metrics['sdk_axis_max_norm_error'] = max(abs(item['sdk_axis_norm'] - 1.) for item in joints)
    metrics['native_same_q_max_error_rad'] = q_error
    passed = (len(joints) == 43 and all(value < 1e-4 for value in metrics.values()))
    return {'schema': 'g1_joint_kinematics_comparison_v1', 'passed': passed,
        'task_qualified': False, 'source_control_tick': sdk['control_tick'],
        'native_physics_integrations': 0, 'native_policy_inferences': 0,
        'source_diagnostic_extra_integrations': 0,
        'environment_translation_source': translation.tolist(),
        'metrics': metrics, 'joints': joints, 'bodies': bodies,
        'scope': 'same-q mapping diagnostic; initialization at measured q is not task performance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('definition', 'sdk', 'native', 'native_receipt', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--translation', type=float, nargs=3, default=[0., 0., .795])
    args = parser.parse_args()
    if digest(args.definition) != DEFINITION_SHA:
        parser.error('Frozen source definition changed')
    receipt = json.loads(args.native_receipt.read_text())
    if not receipt['capture_succeeded'] or any(receipt[key] != 0 for key in (
            'actual_integrations', 'actual_torque_updates', 'actual_model_attempts',
            'live_policy_inference_calls')):
        parser.error('Native capture must independently report zero integration/inference')
    result = audit(json.loads(args.definition.read_text()), json.loads(args.sdk.read_text()),
                   json.loads(args.native.read_text()), np.asarray(args.translation))
    result['input_sha256'] = {name: digest(getattr(args, name)) for name in (
        'definition', 'sdk', 'native', 'native_receipt')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'passed': result['passed'], 'metrics': result['metrics']}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
