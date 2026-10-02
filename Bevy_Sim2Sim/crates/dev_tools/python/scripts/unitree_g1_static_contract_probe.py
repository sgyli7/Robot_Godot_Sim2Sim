#!/usr/bin/env python3
"""Numerically audit static ONNX normalization against frozen training statistics.

Compares both declared graph input labels and original Arena named hand order.
Runs only the small preprocessing/decode graphs on CPU, never task inference or
physics. No model bytes or graph metadata are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from unitree_g1_static_onnx import StaticAppleOnnx


GROUPS = ('left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--source-joints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_groups = yaml.safe_load(args.source_joints.read_text())['joints']
    stats_path = args.model_root / 'statistics.json'
    stats = json.loads(stats_path.read_text())['new_embodiment']
    owner = StaticAppleOnnx(args.model_root, 'cpu', small_only=True)
    specs = {s['name']: s for s in owner.graph['models']['preprocess_state']['inputs']}
    random = np.random.default_rng(42)
    numerical = []
    label_errors = []
    midpoint = None
    for sample in range(33):
        desired = {key: np.zeros(len(source_groups[key])) if sample == 0 else
            random.uniform(-.8, .8, len(source_groups[key])) for key in GROUPS}
        training_inputs = {}
        label_inputs = {}
        for key in GROUPS:
            low = np.array(stats['state'][key]['q01'])
            high = np.array(stats['state'][key]['q99'])
            value = (low + (desired[key] + 1) * (high - low) / 2).astype(np.float32)
            training_inputs[key] = value.reshape(1, -1)
            label_inputs[key] = np.array([[value[source_groups[key].index(name)]
                for name in specs[key]['element_names'][0]]], dtype=np.float32)
        expected = np.concatenate(list(desired.values()))
        normalized, _ = owner.stage('preprocess_state', training_inputs)
        declared, _ = owner.stage('preprocess_state', label_inputs)
        actual = normalized['state'][0, 0, :31]
        numerical.append(float(np.abs(actual - expected).max()))
        label_errors.append(float(np.abs(declared['state'][0, 0, :31] - expected).max()))
        if sample == 0:
            midpoint = {'training_order': actual.tolist(), 'declared_labels': declared['state'][0, 0, :31].tolist()}
    decoded, _ = owner.stage('decode_action', {'normalized_action': np.zeros((1, 40, 132), np.float32)})
    decode_errors = {key: float(np.abs(value[0, 0] - (
        np.array(stats['action'][key]['q01']) + np.array(stats['action'][key]['q99'])) / 2).max())
        for key, value in decoded.items()}
    receipt = {'schema': 'g1_static_numerical_contract_probe_v1', 'qualified': False,
        'full_policy_inference': False, 'physics_steps': 0, 'provider': owner.provider,
        'samples': 33, 'seed': 42, 'source_joint_groups_sha256': digest(args.source_joints),
        'statistics_sha256': digest(stats_path), 'graph_yaml_sha256': digest(owner.root / 'graph.yaml'),
        'preprocess_state_sha256': digest(owner.root / 'preprocess_state.onnx'),
        'decode_action_sha256': digest(owner.root / 'decode_action.onnx'),
        'normalization_tolerance': 2e-5, 'training_order_max_error': max(numerical),
        'declared_labels_max_error': max(label_errors), 'zero_decode_errors': decode_errors,
        'midpoint': midpoint, 'source_order': {key: source_groups[key] for key in GROUPS},
        'declared_input_labels': {key: specs[key]['element_names'][0] for key in GROUPS},
        'effective_contract': 'unitree_g1_static_observation_v2; canonical index(2),middle(2),thumb(3) hands',
        'passed': max(numerical) <= 2e-5 and max(decode_errors.values()) <= 2e-5}
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items()
        if key not in ('midpoint', 'source_order', 'declared_input_labels')}, indent=2))
    if not receipt['passed']:
        raise ValueError('Numerical contract probe failed; inspect receipt before task rollout')


if __name__ == '__main__':
    main()
