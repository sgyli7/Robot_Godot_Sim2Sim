#!/usr/bin/env python3
"""Bounded causal renderer comparison, never a native task rollout.

Original Isaac is the sole task-physics owner. This private renderer receives
its measured poses, initializes a fresh native zero-Tick assembly, checks all
53 body frames and both prop frames, then returns actual Bevy RGB. The separate
policy bridge receives only that image and measured named self-state. External
pose initialization is recorded explicitly and cannot qualify native control.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from PIL import Image

from unitree_g1_joint_kinematics_audit import BASIS, DEFINITION_SHA, audit, rotation, sdk_pose
from unitree_g1_source_expert import JOINT_NAMES

TRANSLATION = np.array([0., 0., .795])
PROP_NAMES = {'t1_apple': 'apple_01_objaverse_robolab',
              't1_plate': 'clay_plates_hot3d_robolab'}
MATCHED_PROPS_SHA = '19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0'
ROBOT_VISUAL_SHA = '98711070da898c75089b66ab35c799900cacb1e2980c537f9fcc1da8c8e9297d'
BACKGROUND_SHA = '46f8d43cb8d0998daace4ae792dd80d39342993a8d0267fd68722b917c513a2a'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(connection, length):
    value = bytearray()
    while len(value) < length:
        part = connection.recv(length - len(value))
        if not part:
            raise ValueError('Source renderer request disconnected')
        value.extend(part)
    return bytes(value)


def initialized_pose(pose):
    frame = sdk_pose(pose)  # Also validates finite/unit SDK XYZW quaternion.
    return {'position': (frame[:3, 3] + TRANSLATION).tolist(),
            'rotation_wxyz': [pose[6], *pose[3:6]]}


def request_configuration(template, request):
    if (set(request) != {'schema', 'episode_id', 'control_tick',
                        'environment_translation_source', 'joint_kinematics', 'task_objects'}
            or request['schema'] != 'g1_source_renderer_pose_v1'
            or request['environment_translation_source'] != TRANSLATION.tolist()
            or type(request['episode_id']) is not int or request['episode_id'] < 1
            or type(request['control_tick']) is not int
            or request['control_tick'] not in range(0, 281, 40)
            or set(request['task_objects']) != set(PROP_NAMES.values())):
        raise ValueError('Unexpected renderer request identity/scene')
    sdk = request['joint_kinematics']
    if (sdk['control_tick'] != request['control_tick']
            or sdk['before_sdk_counters'] != sdk['after_sdk_counters']
            or sdk['extra_integrations'] != 0 or sdk['coordinate_writes'] != 0
            or sdk['source_physics_steps_since_reset'] != sdk['control_tick'] * 4
            or len(sdk['joint_names']) != 43 or set(sdk['joint_names']) != set(JOINT_NAMES)
            or len(sdk['body_names']) != 53 or len(set(sdk['body_names'])) != 53):
        raise ValueError('Unverified SDK measured-pose boundary')
    q = dict(zip(sdk['joint_names'], sdk['joint_positions'], strict=True))
    canonical = [q[name] for name in JOINT_NAMES]
    if not np.isfinite(canonical).all():
        raise ValueError('Nonfinite external joint measurements')
    config = copy.deepcopy(template)
    body = config['runner']
    body['episode_id'] = request['episode_id']
    body['default_positions'] = canonical
    body['root_pose'] = initialized_pose(sdk['body_link_pose_w'][sdk['body_names'].index('pelvis')])
    objects = body['task_objects']
    objects.pop('source_t1_shelf', None)
    objects['placements'] = [{'kind': kind, 'root_pose': initialized_pose(
        request['task_objects'][name]['pose'])} for kind, name in PROP_NAMES.items()]
    config['render_only_environment_translation'] = TRANSLATION.tolist()
    config.pop('predictive_limit_diagnostic', None)
    config.pop('diagnostic_constraint_sweeps', None)
    return config


def render(args, template, definition, request, directory):
    directory.mkdir()
    (directory / 'request.json').write_text(json.dumps(request, allow_nan=False))
    config = request_configuration(template, request)
    config_path = directory / 'config.json'
    config_path.write_text(json.dumps(config, allow_nan=False))
    command = [str(args.binary), '--scene', 'g1_camera_diagnostic', '--robot', 'g1',
               '--g1-config', str(config_path), '--g1-ticks', '0', '--output', str(directory / 'native')]
    started = time.monotonic()
    with (directory / 'app.log').open('x') as log:
        completed = subprocess.run(command, cwd=args.workdir, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=70)
    if completed.returncode != 0:
        raise ValueError(f'Zero-Tick native renderer exited {completed.returncode}')
    native = directory / 'native'
    receipt_path = native / 'capture_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if (not receipt['capture_succeeded'] or not receipt['external_pose_render_only']
            or receipt['task_qualified'] or any(receipt[key] != 0 for key in (
                'requested_ticks', 'actual_integrations', 'actual_torque_updates',
                'actual_model_attempts', 'actual_model_successes', 'live_policy_inference_calls'))):
        raise ValueError('Native renderer advanced physics/controller or omitted diagnostic scope')
    stamp = json.loads((native / 'ego_stamp.json').read_text())
    correspondence = audit(definition, request['joint_kinematics'], stamp, TRANSLATION)
    if not correspondence['passed'] or stamp['episode_id'] != request['episode_id']:
        raise ValueError('Native initialized robot does not match measured SDK body frames')
    frame = receipt['last_task_object_frame']
    if frame['source_tick'] != 0 or frame['sim_time'] != 0 or len(frame['objects']) != 2:
        raise ValueError('Unexpected rendered object frame')
    props = []
    for obj in frame['objects']:
        source = sdk_pose(request['task_objects'][PROP_NAMES[obj['kind']]]['pose'])
        native_position = np.asarray(obj['position_source']) - TRANSLATION
        quat = obj['rotation_engine_xyzw']
        native_rotation = BASIS.T @ rotation([quat[3], *quat[:3]]) @ BASIS
        metrics = {'kind': obj['kind'],
                   'position_error_m': float(np.linalg.norm(source[:3, 3] - native_position)),
                   'rotation_matrix_max_error': float(np.abs(source[:3, :3] - native_rotation).max())}
        if metrics['position_error_m'] > 1e-4 or metrics['rotation_matrix_max_error'] > 1e-4:
            raise ValueError('Rendered prop does not match source SDK pose')
        props.append(metrics)
    correspondence['props'] = props
    (directory / 'pose_correspondence.json').write_text(json.dumps(correspondence, allow_nan=False, indent=2))
    with Image.open(native / 'ego_640x480.png') as image:
        if image.size != (640, 480):
            raise ValueError('Native camera resolution changed')
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).tobytes()
    result = {'schema': 'g1_source_renderer_rgb_v1', 'episode_id': request['episode_id'],
        'control_tick': request['control_tick'], 'pose_correspondence_passed': True,
        'native_integrations': 0, 'native_controller_updates': 0, 'native_policy_calls': 0,
        'captured_at_unix_ms': stamp['captured_at_unix_ms'],
        'rgb_sha256': hashlib.sha256(rgb).hexdigest(),
        'native_receipt_sha256': digest(receipt_path),
        'zero_tick_render_seconds': time.monotonic() - started,
        'scope': 'external source-pose initialized render; source physics only; no native task qualification'}
    (directory / 'reply_receipt.json').write_text(json.dumps(result, indent=2))
    result['camera_rgb_b64'] = base64.b64encode(rgb).decode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('socket', 'output', 'template', 'binary', 'workdir'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, default=8)
    args = parser.parse_args()
    args.socket = args.socket.resolve()
    if args.socket.exists() or not 1 <= args.max_calls <= 8:
        parser.error('Use a fresh private socket and 1..8 renders')
    template = json.loads(args.template.read_text())
    if ('policy' in template or 'body' in template['runner']
            or template['runner']['definition_sha256'] != DEFINITION_SHA
            or digest(template['runner']['definition']) != DEFINITION_SHA
            or template['runner']['task_objects']['definition_sha256'] != MATCHED_PROPS_SHA
            or digest(template['runner']['task_objects']['definition']) != MATCHED_PROPS_SHA
            or template['visual_sha256'] != ROBOT_VISUAL_SHA
            or template['background_visual']['sha256'] != BACKGROUND_SHA
            or template['exposure_ev100'] != 11.7
            or template['directional_shadow_maps'] is not False):
        parser.error('Use the frozen static renderer candidate and matched 6.0 props')
    definition = json.loads(Path(template['runner']['definition']).read_text())
    args.output.mkdir()
    summary = {'schema': 'g1_source_renderer_bridge_v1', 'task_qualified': False,
        'harness_sha256': digest(__file__), 'binary_sha256': digest(args.binary),
        'template_sha256': digest(args.template), 'renders': [], 'closed': False,
        'scope': 'finite causal image-domain comparison; external-pose initialization is explicit'}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            os.chdir(args.socket.parent)
            server.bind(args.socket.name)
            args.socket.chmod(0o666)
            server.listen(1)
            server.settimeout(180)
            print(json.dumps({'event': 'renderer_ready', 'socket': str(args.socket)}), flush=True)
            identity = None
            for index in range(args.max_calls):
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(90)
                    size = int.from_bytes(exact(connection, 4), 'big')
                    if not 0 < size <= 2_097_152:
                        raise ValueError('Renderer request exceeds finite bound')
                    request = json.loads(exact(connection, size))
                    if request == {'stop': True}:
                        stopped = b'{"stopped":true}'
                        connection.sendall(len(stopped).to_bytes(4, 'big') + stopped)
                        break
                    try:
                        if request['control_tick'] != index * 40 or (
                                identity is not None and identity != request['episode_id']):
                            raise ValueError('Old/out-of-order renderer episode or frame')
                        identity = request['episode_id']
                        reply = render(args, template, definition, request,
                                       args.output / f'tick_{request["control_tick"]:04}')
                        summary['renders'].append({key: value for key, value in reply.items()
                                                   if key != 'camera_rgb_b64'})
                        print(json.dumps({'event': 'rendered', 'tick': request['control_tick'],
                            'seconds': reply['zero_tick_render_seconds']}), flush=True)
                    except Exception as error:
                        summary['error'] = repr(error)
                        reply = {'error': repr(error)}
                    data = json.dumps(reply, allow_nan=False).encode()
                    if len(data) > 2_097_152:
                        raise ValueError('Renderer response exceeds finite bound')
                    connection.sendall(len(data).to_bytes(4, 'big') + data)
                    if 'error' in reply:
                        break
    finally:
        if args.socket.exists():
            args.socket.unlink()
        summary['closed'] = True
        (args.output / 'receipt.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
