"""Guard SDK quaternion/bounds evidence; synthetic tests do not qualify a robot."""
import unittest
import numpy as np
from unitree_g1_t1_source_placement import aabb_gap, collision_bounds, sdk_rotation


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


if __name__ == '__main__':
    unittest.main()
