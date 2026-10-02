#!/usr/bin/env python3
"""Prepare/validate one frozen expert action sequence for SOURCE diagnostics.

This is an open-loop diagnostic, never an autonomous task score. The original
Arena replay policy sends only arms/hands, navigation and base height to WBC;
recorded lower-body actions and torso RPY are deliberately not replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DATASET_REVISION = '37ba80a99486a4c308477854f54b4e82e77777dc'
DATASET_ID = 'nvidia/Arena-G1-Static-PickNPlace-Task'
FROZEN_FILES = {
    'lerobot/meta/info.json': 'a38b8ebcce655fddc8b6ca0086af3afaf86b912d4cebe1a68689fc6f89d19b32',
    'lerobot/meta/modality.json': '36dcaef30bb34a4c0337775932d83ea7b79b36f19e3a715239bab6852a09d9cc',
    'lerobot/data/chunk-000/episode_000000.parquet': 'f61c7214a33717cf475f67f540c9149f67f38a3483064d6f9778f579187afcf1',
}
GROUPS = {'left_arm': (15, 22), 'left_hand': (22, 29),
          'right_arm': (29, 36), 'right_hand': (36, 43)}
ARM_SUFFIXES = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                'wrist_roll', 'wrist_pitch', 'wrist_yaw')
HAND_SUFFIXES = ('index_0', 'index_1', 'middle_0', 'middle_1', 'thumb_0', 'thumb_1', 'thumb_2')
JOINT_NAMES = [f'{side}_{suffix}_joint' for side in ('left', 'right')
               for suffix in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
JOINT_NAMES += ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
for _side in ('left', 'right'):
    JOINT_NAMES += [f'{_side}_{suffix}_joint' for suffix in ARM_SUFFIXES]
    JOINT_NAMES += [f'{_side}_hand_{suffix}_joint' for suffix in HAND_SUFFIXES]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vector(value, length, label):
    if not isinstance(value, list) or len(value) != length or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f'Invalid finite {label} vector')
    return value


def validate_sequence(value):
    if value.get('schema') != 'g1_original_source_expert_v1':
        raise ValueError('Wrong expert diagnostic schema')
    if value.get('dataset_id') != DATASET_ID or value.get('dataset_revision') != DATASET_REVISION:
        raise ValueError('Expert dataset identity changed')
    if value.get('input_sha256') != FROZEN_FILES or value.get('joint_names') != JOINT_NAMES:
        raise ValueError('Expert bytes or joint order changed')
    if value.get('action_period_ns') != 20_000_000 or value.get('episode_index') != 0:
        raise ValueError('Expert episode/timing changed')
    if value.get('after_sequence') != 'hold_last_frame_until_original_termination':
        raise ValueError('Unspecified end-of-sequence behavior')
    vector(value.get('initial_joint_positions'), 43, 'initial state')
    frames = value.get('frames')
    if not isinstance(frames, list) or len(frames) != 154:
        raise ValueError('Frozen expert sequence must contain 154 frames')
    expected = set(GROUPS) | {'waist', 'navigate_mps_rps', 'base_height_m'}
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != expected:
            raise ValueError('Expert frame fields changed')
        for group in GROUPS:
            vector(frame[group], 7, group)
        if vector(frame['waist'], 3, 'waist') != [0, 0, 0]:
            raise ValueError('Original replay leaves waist to WBC')
        vector(frame['navigate_mps_rps'], 3, 'navigation')
        vector([frame['base_height_m']], 1, 'height')
        if not .3 <= frame['base_height_m'] <= 1.0:
            raise ValueError('Expert height outside original diagnostic bound')
    return value


def load_sequence(path, expected_sha256):
    if not expected_sha256 or len(expected_sha256) != 64 or Path(path).stat().st_size > 262_144:
        raise ValueError('Expert sequence requires a SHA-256 identity and bounded file')
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Expert sequence SHA-256 mismatch')
    return validate_sequence(json.loads(data))


def prepare(cache):
    # Imported only in the host preparation tool, not the source container.
    import pyarrow.parquet as parquet
    for relative, expected in FROZEN_FILES.items():
        if sha256(cache / relative) != expected:
            raise ValueError(f'Frozen expert input changed: {relative}')
    info = json.loads((cache / 'lerobot/meta/info.json').read_text())
    modality = json.loads((cache / 'lerobot/meta/modality.json').read_text())
    if any(info['features'][key]['names'] != JOINT_NAMES for key in ('action', 'observation.state')):
        raise ValueError('Published expert joint names do not match canonical physical order')
    for group, (start, end) in GROUPS.items():
        if modality['action'][group] != {'start': start, 'end': end}:
            raise ValueError(f'Published expert action slice changed: {group}')
    rows = parquet.read_table(cache / 'lerobot/data/chunk-000/episode_000000.parquet').to_pylist()
    frames = []
    for index, row in enumerate(rows):
        if row['frame_index'] != index or row['episode_index'] != 0 or abs(row['timestamp'] - index * .02) > 1e-8:
            raise ValueError('Expert episode has a gap, another episode or different timing')
        action = vector(row['action'], 43, 'recorded action')
        frame = {group: action[start:end] for group, (start, end) in GROUPS.items()}
        frame.update(waist=[0, 0, 0], navigate_mps_rps=row['teleop.navigate_command'],
                     base_height_m=vector(row['teleop.base_height_command'], 1, 'recorded height')[0])
        frames.append(frame)
    return validate_sequence({
        'schema': 'g1_original_source_expert_v1', 'dataset_id': DATASET_ID,
        'dataset_revision': DATASET_REVISION, 'episode_index': 0, 'input_sha256': FROZEN_FILES,
        'joint_names': JOINT_NAMES, 'action_period_ns': 20_000_000,
        'initial_joint_positions': rows[0]['observation.state'], 'frames': frames,
        'after_sequence': 'hold_last_frame_until_original_termination',
        'published_final_reward': rows[-1]['next.reward'], 'published_final_done': rows[-1]['next.done'],
        'qualified': False, 'scope': 'source-only expert diagnostic; no perception; WBC retains lower body',
        'object_initial_pose_available': False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.cache)
    with args.output.open('x') as stream:
        json.dump(result, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({'path': str(args.output), 'sha256': sha256(args.output),
                      'frames': len(result['frames']), 'qualified': False}))


if __name__ == '__main__':
    main()
