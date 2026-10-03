#!/usr/bin/env python3
"""Read-only native T2 placement audit; never supplies decisions or controls.

The diagnostic target is the upward prism over convex part2 of the frozen
original blue bin: its interior flat floor, excluding the outer rim. The prism
moves with the bin. Every box collision vertex must lie inside its XY footprint
and above the floor's lowest local Z. Upward normal support from that bin,
separation from all robot links, standing, strict speed limits and a contiguous
two-second span are required independently. This rule is not a formal ten-run
benchmark or a qualification of arbitrary natural-language tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from unitree_g1_static_placement_audit import (
    DEFINITION_SHA, active_normal_impulse, body_rotation, digest, footprint,
    robot_contact_blocks_release,
)

FLOOR_PART = 2


def evaluate(definition, rows):
    objects = {o['kind']: o for o in definition['objects']}
    box_points = np.asarray([v for p in objects['t2_box']['convex_parts'] for v in p['points']])
    floor_points = np.asarray(objects['t2_bin']['convex_parts'][FLOOR_PART]['points'])
    polygon = footprint(floor_points)
    floor_bottom = float(floor_points[:, 2].min())
    edges = np.roll(polygon, -1, axis=0) - polygon
    inward = np.column_stack((-edges[:, 1], edges[:, 0]))
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    previous = 0
    episode = None
    suffix = maximum = 0
    fell = False
    samples = []
    for row in rows:
        body = row['body']['mobile_homie_v2']
        frame = body['task_objects']
        tick = body['integration_count']
        if (tick != previous + 1 or frame['source_tick'] != tick
                or abs(frame['sim_time'] - tick*.02) > 1e-6
                or body['step_configuration']['physics_hz'] != 50
                or np.float32(body['step_configuration']['dt']) != np.float32(.02)
                or frame['definition_sha256'] != DEFINITION_SHA):
            raise ValueError('Trace gap, time/physics mismatch or foreign task definition')
        if episode is None:
            episode = frame['episode_id']
        if episode != frame['episode_id']:
            raise ValueError('Mixed/reset episode in placement window')
        pair = {o['kind']: o for o in frame['objects']}
        box, bin_object = pair['t2_box'], pair['t2_bin']
        offset = np.asarray(box['position_source']) - bin_object['position_source']
        points = (box_points @ body_rotation(box).T + offset) @ body_rotation(bin_object)
        margins = points[:, :2] @ inward.T - np.sum(polygon*inward, axis=1)
        if not np.isfinite(points).all() or not np.isfinite(margins).all():
            raise ValueError('Nonfinite collision geometry/pose')
        margin = float(margins.min())
        inside = margin >= 0. and float(points[:, 2].min()) >= floor_bottom
        contacts = box['last_solve_contacts']
        released = not any(robot_contact_blocks_release(c) for c in contacts)
        support_z = 0.
        for contact in contacts:
            if contact['other_task_kind'] == 't2_bin' and (active_normal_impulse(contact) or 0.) > 0.:
                # Earlier traces lack direction; do not infer upward support.
                vector = np.asarray(contact.get('active_solver_normal_impulse_on_object_source'))
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError('Missing/nonfinite bin support direction')
                support_z += float(vector[2])
        supported = support_z > 0.
        linear = float(np.linalg.norm(box['linear_velocity_source']))
        angular = float(np.linalg.norm(box['angular_velocity_source']))
        standing = body['root_upright_cosine'] > .95 and body['root_position_source'][2] >= .35
        fell |= not standing
        if not np.isfinite([linear, angular, body['root_upright_cosine']]).all():
            raise ValueError('Nonfinite independent velocity/standing state')
        ready = inside and released and supported and standing and not fell and linear < .02 and angular < .1
        suffix = suffix + 1 if ready else 0
        maximum = max(maximum, suffix)
        samples.append({'tick': tick, 'all_box_vertices_inside_bin_floor_prism': inside,
            'minimum_signed_floor_margin_m': margin,
            'lowest_box_vertex_bin_local_z_m': float(points[:, 2].min()),
            'no_actual_robot_contact': released, 'upward_bin_normal_impulse_n_s': support_z,
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular,
            'standing': standing, 'episode_fall_remembered': fell, 'ready': ready,
            'continuous_window_seconds': max(suffix-1, 0)/50.})
        previous = tick
    if not samples:
        raise ValueError('Empty physical trace')
    return {'schema': 'g1_independent_mobile_placement_audit_v2',
        'diagnostic_release_window_passed': maximum >= 101,
        'autonomous_task_qualified': False, 'formal_ten_episode_acceptance_passed': False,
        'target_rule_frozen_for_formal_suite': False,
        'contact_evidence_rule': 'current solver-point identities required; legacy all-point cache totals cannot prove support; unavailable active impulse blocks robot release',
        'target_rule': 'upward prism over frozen original bin interior flat floor convex part2; every box vertex inside; bin normal impulse upward in source world Z',
        'target_floor_bin_local_xy': polygon.tolist(), 'target_floor_lowest_local_z_m': floor_bottom,
        'box_collision_vertices': len(box_points),
        'max_continuous_placement_seconds': max(maximum-1, 0)/50.,
        'final_continuous_placement_seconds': max(suffix-1, 0)/50.,
        'actual_physics_samples': len(samples), 'samples': samples,
        'scope': 'independent read-only truth; no model input, controller command, pose write, replay or qualification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('definition', 'trace', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.definition) != DEFINITION_SHA:
        parser.error('Frozen original task collision definition changed')
    result = evaluate(json.loads(args.definition.read_text()),
        [json.loads(line) for line in args.trace.read_text().splitlines()])
    result['input_sha256'] = {name: digest(getattr(args, name)) for name in ('definition', 'trace')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({key: result[key] for key in ('diagnostic_release_window_passed',
        'max_continuous_placement_seconds', 'final_continuous_placement_seconds', 'autonomous_task_qualified')}))
    if not result['diagnostic_release_window_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
