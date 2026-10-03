#!/usr/bin/env python3
"""Independent T1 geometry/release audit of an actual native physical trace.

The disclosed diagnostic target is the vertical prism over the original
plate's convex outer XY footprint, in its actual moving body frame. Every
vertex of every original apple collision hull must lie inside that footprint.
Plate support, hand separation, standing, speed and two continuous seconds
are required separately. This is not the frozen ten-episode autonomous task
acceptance suite and never supplies an observation or control command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from unitree_g1_joint_kinematics_audit import BASIS, rotation

DEFINITION_SHA = '19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def footprint(points):
    values = sorted(set(map(tuple, np.asarray(points)[:, :2])))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    chains = []
    for ordered in (values, values[::-1]):
        chain = []
        for point in ordered:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    result = np.asarray(chains[0] + chains[1])
    if len(result) < 3:
        raise ValueError('Plate footprint is degenerate')
    return result


def body_rotation(sample):
    x, y, z, w = sample['rotation_engine_xyzw']
    return BASIS.T @ rotation([w, x, y, z]) @ BASIS


def active_normal_impulse(contact):
    """Current solver-point evidence; legacy cache totals cannot prove support."""
    impulse = contact.get('active_solver_normal_impulse_n_s')
    if impulse is None:
        return None
    if not np.isfinite(impulse) or impulse < 0.:
        raise ValueError('Invalid active solver contact impulse')
    return float(impulse)


def robot_contact_blocks_release(contact):
    """Positive separated zero-force speculative pairs are not physical touch.

    Any touching/penetrating surface, nonzero supporting force, or unresolved
    distance remains a release blocker. No contact-distance tolerance is added.
    """
    if contact['other_robot_body_index'] is None:
        return False
    if 'geometric_distance_after_step_m' not in contact:
        raise ValueError('Release audit requires same-Tick post-integration shape distances')
    distance = contact['geometric_distance_after_step_m']
    impulse = active_normal_impulse(contact)
    if impulse is None:
        return True
    if distance is None:
        return True
    if not np.isfinite(distance):
        raise ValueError('Invalid independent contact distance')
    return distance <= 0. or impulse > 0.


def evaluate(definition, rows):
    objects = {o['kind']: o for o in definition['objects']}
    apple_points = np.asarray([v for p in objects['t1_apple']['convex_parts'] for v in p['points']])
    polygon = footprint([v for p in objects['t1_plate']['convex_parts'] for v in p['points']])
    edges = np.roll(polygon, -1, axis=0) - polygon
    inward = np.column_stack((-edges[:, 1], edges[:, 0]))
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    previous_tick = 0
    episode = None
    suffix = 0
    maximum = 0
    candidate_suffix = 0
    candidate_maximum = 0
    minimum_margin = float('inf')
    samples = []
    fell = False
    for row in rows:
        body = row['body']
        if 'static_agile' in body:
            body = body['static_agile']
        tick = body['integration_count']
        frame = body['task_objects']
        if (tick != previous_tick + 1 or frame['source_tick'] != tick
                or abs(frame['sim_time'] - tick * .02) > 1e-6
                or np.float32(body['step_configuration']['dt']) != np.float32(.02)
                or body['step_configuration']['physics_hz'] != 50
                or frame['definition_sha256'] != DEFINITION_SHA):
            raise ValueError('Trace gap, time/physics mismatch or foreign object definition')
        if episode is None:
            episode = frame['episode_id']
        if frame['episode_id'] != episode:
            raise ValueError('Mixed/reset episode in release window')
        pair = {o['kind']: o for o in frame['objects']}
        apple, plate = pair['t1_apple'], pair['t1_plate']
        # Row vectors: source apple-local -> world -> moving plate-local.
        offset = np.asarray(apple['position_source']) - plate['position_source']
        points = (apple_points @ body_rotation(apple).T + offset) @ body_rotation(plate)
        margins = (points[:, :2] @ inward.T) - np.sum(polygon * inward, axis=1)
        margin = float(margins.min())
        if not np.isfinite(points).all() or not np.isfinite(margins).all():
            raise ValueError('Nonfinite object geometry/pose')
        inside = margin >= 0.
        contact = apple['last_solve_contacts']
        no_candidate_pair = not any(c['other_robot_body_index'] is not None for c in contact)
        released = not any(robot_contact_blocks_release(c) for c in contact)
        supported = any(c['other_task_kind'] == 't1_plate'
                        and (active_normal_impulse(c) or 0.) > 0. for c in contact)
        linear = float(np.linalg.norm(apple['linear_velocity_source']))
        angular = float(np.linalg.norm(apple['angular_velocity_source']))
        standing = body['root_upright_cosine'] > .95 and body['root_position_source'][2] >= .35
        fell |= not standing
        if not np.isfinite([linear, angular, body['root_upright_cosine']]).all():
            raise ValueError('Nonfinite independent velocity/standing state')
        ready = inside and released and supported and linear < .02 and angular < .1 and standing and not fell
        suffix = suffix + 1 if ready else 0
        maximum = max(maximum, suffix)
        candidate_suffix = candidate_suffix + 1 if ready and no_candidate_pair else 0
        candidate_maximum = max(candidate_maximum, candidate_suffix)
        if ready:
            minimum_margin = min(minimum_margin, margin)
        samples.append({'tick': tick, 'phase': row.get('phase', 'saved_source_action'),
            'all_collision_vertices_inside_plate_footprint': inside,
            'minimum_signed_footprint_margin_m': margin,
            'no_robot_contact_pair': no_candidate_pair,
            'no_actual_robot_contact': released, 'positive_plate_support_impulse': supported,
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular,
            'standing': standing, 'ready': ready,
            'continuous_window_seconds': max(suffix - 1, 0) / 50.})
        previous_tick = tick
    if not samples:
        raise ValueError('Empty physical trace')
    return {'schema': 'g1_independent_static_placement_audit_v4',
        'diagnostic_release_window_passed': maximum >= 101,
        'autonomous_task_qualified': False, 'formal_ten_episode_acceptance_passed': False,
        'target_rule': 'vertical prism over convex outer footprint of frozen original plate collision geometry; physical plate support required separately',
        'target_footprint_plate_local_xy': polygon.tolist(), 'apple_collision_vertices': len(apple_points),
        'target_rule_frozen_for_formal_suite': False,
        'samples': samples, 'actual_physics_samples': len(samples),
        'max_continuous_placement_seconds': max(maximum - 1, 0) / 50.,
        'candidate_pair_only_max_continuous_seconds': max(candidate_maximum - 1, 0) / 50.,
        'final_continuous_placement_seconds': max(suffix - 1, 0) / 50.,
        'contact_evidence_rule': 'current solver-point identities required; legacy all-point cache totals cannot prove support; unavailable active impulse blocks robot release',
        'release_rule': 'no robot pair with unresolved/nonpositive same-Tick post-integration shape distance or positive active solver normal impulse; cached solver distances never establish release',
        'minimum_margin_during_ready_samples_m': minimum_margin if minimum_margin != float('inf') else None,
        'scope': 'read-only diagnostic truth; no model input, control, pose write or autonomous qualification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('definition', 'trace', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.definition) != DEFINITION_SHA:
        parser.error('Frozen matched T1 collision definition changed')
    result = evaluate(json.loads(args.definition.read_text()),
                      [json.loads(line) for line in args.trace.read_text().splitlines()])
    result['input_sha256'] = {name: digest(getattr(args, name)) for name in ('definition', 'trace')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({key: result[key] for key in (
        'diagnostic_release_window_passed', 'max_continuous_placement_seconds',
        'minimum_margin_during_ready_samples_m', 'autonomous_task_qualified')}))
    if not result['diagnostic_release_window_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
