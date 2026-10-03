"""Guard SDK quaternion/bounds evidence; synthetic tests do not qualify a robot."""
import unittest
import json
import numpy as np
from unitree_g1_t1_source_placement import aabb_gap, collision_bounds, sdk_rotation, evaluate


class SourcePlacementGeometry(unittest.TestCase):
    def test_sdk_xyzw_quaternion_and_offset_are_not_native_wxyz(self):
        pose = [1., 2., 3., 0., 0., 1., 0.]
        np.testing.assert_allclose(sdk_rotation(pose) @ [1., 0., 0.], [-1., 0., 0.], atol=1e-12)
        collision = {'shape': 'sphere', 'radius': .1,
            'local_pose': {'position': [.5, 0., 0.], 'rotation_wxyz': [1., 0., 0., 0.]}}
        lower, upper = collision_bounds(collision, pose)
        np.testing.assert_allclose(lower, [.4, 1.9, 2.9])
        np.testing.assert_allclose(upper, [.6, 2.1, 3.1])

    def test_bounds_overlap_and_touch_do_not_prove_release(self):
        apple = (np.zeros(3), np.ones(3))
        for lower in ([.2, .2, .2], [1., 0., 0.]):
            self.assertEqual(aabb_gap(apple, (np.array(lower), np.array(lower) + 1)), 0.)
        self.assertAlmostEqual(aabb_gap(apple, (np.array([1.01, 0., 0.]), np.array([2., 1., 1.]))), .01)

    def test_capsule_axis_and_box_rotation_bound_actual_surface(self):
        pose = [0., 0., 0., 0., 0., 0., 1.]
        local = {'position': [0., 0., 0.], 'rotation_wxyz': [2**-.5, 0., 2**-.5, 0.]}
        capsule = {'shape': 'capsule', 'axis': 'Z', 'radius': .1,
                   'half_height': .4, 'local_pose': local}
        np.testing.assert_allclose(collision_bounds(capsule, pose)[1], [.5, .1, .1], atol=1e-12)
        box = {'shape': 'box', 'half_extents': [.1, .2, .3], 'local_pose': local}
        np.testing.assert_allclose(collision_bounds(box, pose)[1], [.3, .2, .1], atol=1e-12)

    def test_positive_measured_support_serializes_and_short_window_stays_failed(self):
        shape = {'shape': 'sphere', 'radius': .1, 'body': 0,
            'local_pose': {'position': [0., 0., 0.], 'rotation_wxyz': [1., 0., 0., 0.]}}
        definition = {'objects': [
            {'kind': 't1_apple', 'convex_parts': [{'points': [[0., 0., 0.]]}]},
            {'kind': 't1_plate', 'convex_parts': [{'points': [[-.1, -.1, 0.],
                [.1, -.1, 0.], [.1, .1, 0.], [-.1, .1, 0.]]}]}]}
        robot = {'bodies': [{'name': 'source_body'}], 'collisions': [shape]}
        receipt = {'placement_contact_view': {'robot_body_names': ['source_body'],
            'filter_count': 54, 'schema_writes': 0},
            'success_auto_reset_suppressed_for_audit_only': True,
            'original_drop_and_timeout_terms_unchanged': True,
            'source_physics_hz': 200, 'source_control_hz': 50,
            'completed_control_ticks': 1, 'measured_physics_steps_since_reset': 4, 'policy_calls': 1}
        matrix = np.zeros((54, 3)); matrix[0, 2] = 1.
        row = {'control_tick': 0, 'physics_steps_since_reset': 0,
            'root_link_pose_w': [0., 0., 0., 0., 0., 0., 1.], 'upright': 1.,
            'acceptance_truth_only': {name: {'pose': [0., 0., 0., 0., 0., 0., 1.],
                'velocity': [0.]*6} for name in ('apple_01_objaverse_robolab', 'clay_plates_hot3d_robolab')},
            'acceptance_robot_body_link_poses_w': [[3., 3., 3., 0., 0., 0., 1.]]*53,
            'acceptance_contacts_only': {'apple_filter_force_matrix_world_n': matrix.tolist(),
                'source_solve_dt_s': .005, 'contact_points': [{'filter_index': 0,
                    'normal_force_n': -1., 'normal_world': [0., 0., -1.], 'separation_m': 0.}]}}
        result = json.loads(json.dumps(evaluate(definition, robot, receipt, [row]), allow_nan=False))
        self.assertTrue(result['samples'][0]['plate_support'])
        self.assertFalse(result['source_diagnostic_placement_passed'])


if __name__ == '__main__':
    unittest.main()
