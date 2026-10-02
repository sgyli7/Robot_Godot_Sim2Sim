"""Synthetic adversarial checks of the independent T2 judge, not robot trials."""
import copy
import unittest

import numpy as np
from unitree_g1_mobile_placement_audit import DEFINITION_SHA, evaluate


def cube_points(half):
    return [[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)]


def fixture(count=101):
    definition = {'objects': [
        {'kind': 't2_box', 'convex_parts': [{'points': cube_points(.1)}]},
        {'kind': 't2_bin', 'convex_parts': [{}, {}, {'points': [[x, y, z]
            for x in (-1., 1.) for y in (-.5, .5) for z in (0., .01)]}]}]}
    box = {'kind': 't2_box', 'position_source': [0., 0., .11],
        'rotation_engine_xyzw': [0., 0., 0., 1.],
        'linear_velocity_source': [0., 0., 0.], 'angular_velocity_source': [0., 0., 0.],
        'last_solve_contacts': [{'other_task_kind': 't2_bin', 'other_robot_body_index': None,
            'normal_impulse_n_s': .01962, 'normal_impulse_on_object_source': [0., 0., .01962]}]}
    bin_object = {'kind': 't2_bin', 'position_source': [0., 0., 0.],
        'rotation_engine_xyzw': [0., 0., 0., 1.]}
    rows = []
    for tick in range(1, count+1):
        rows.append({'body': {'mobile_homie_v2': {'integration_count': tick,
            'step_configuration': {'physics_hz': 50, 'dt': .02},
            'root_upright_cosine': 1., 'root_position_source': [0., 0., .75],
            'task_objects': {'definition_sha256': DEFINITION_SHA, 'source_tick': tick,
                'sim_time': tick*.02, 'episode_id': 42,
                'objects': [copy.deepcopy(box), copy.deepcopy(bin_object)]}}}})
    return definition, rows


def body(row):
    return row['body']['mobile_homie_v2']


class MobilePlacementTests(unittest.TestCase):
    def test_two_second_span_needs_101_samples(self):
        d, rows = fixture()
        self.assertFalse(evaluate(d, rows[:100])['diagnostic_release_window_passed'])
        result = evaluate(d, rows)
        self.assertTrue(result['diagnostic_release_window_passed'])
        self.assertFalse(result['autonomous_task_qualified'])

    def test_prism_moves_and_rotates_with_actual_bin(self):
        d, rows = fixture()
        for row in rows:
            for obj in body(row)['task_objects']['objects']:
                obj['position_source'] = np.asarray(obj['position_source']) + [3., -2., .4]
                obj['position_source'] = obj['position_source'].tolist()
                # Source yaw90 maps to engine Y rotation90.
                obj['rotation_engine_xyzw'] = [0., -np.sqrt(.5), 0., np.sqrt(.5)]
        self.assertTrue(evaluate(d, rows)['diagnostic_release_window_passed'])

    def test_entire_box_must_fit_interior_floor(self):
        d, rows = fixture()
        for row in rows:
            body(row)['task_objects']['objects'][0]['position_source'][0] = .95
        self.assertFalse(evaluate(d, rows)['diagnostic_release_window_passed'])

    def test_side_contact_cannot_supply_upward_support(self):
        d, rows = fixture()
        for row in rows:
            body(row)['task_objects']['objects'][0]['last_solve_contacts'][0][
                'normal_impulse_on_object_source'] = [.01962, 0., 0.]
        self.assertFalse(evaluate(d, rows)['diagnostic_release_window_passed'])

    def test_unknown_or_touching_robot_distance_blocks_release(self):
        for distance in (None, 0., .001):
            d, rows = fixture()
            for row in rows:
                body(row)['task_objects']['objects'][0]['last_solve_contacts'].append({
                    'other_task_kind': None, 'other_robot_body_index': 28,
                    'geometric_distance_after_step_m': distance, 'normal_impulse_n_s': 0.})
            self.assertEqual(evaluate(d, rows)['diagnostic_release_window_passed'], distance == .001)

    def test_fall_remains_failure_after_standing_recovers(self):
        d, rows = fixture(202)
        body(rows[0])['root_upright_cosine'] = .7
        self.assertFalse(evaluate(d, rows)['diagnostic_release_window_passed'])

    def test_speed_threshold_is_strict(self):
        d, rows = fixture()
        for row in rows:
            body(row)['task_objects']['objects'][0]['linear_velocity_source'] = [.02, 0., 0.]
        self.assertFalse(evaluate(d, rows)['diagnostic_release_window_passed'])

    def test_gap_and_reset_do_not_join_windows(self):
        d, rows = fixture()
        for changed in (rows[:50]+rows[51:], copy.deepcopy(rows)):
            if len(changed) == len(rows):
                body(changed[50])['task_objects']['episode_id'] = 43
            with self.assertRaises(ValueError):
                evaluate(d, changed)


if __name__ == '__main__':
    unittest.main()
