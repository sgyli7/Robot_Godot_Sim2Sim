"""Release-criterion regressions; these are not physical task qualifications."""
import unittest

from unitree_g1_static_placement_audit import robot_contact_blocks_release


class ReleaseDistanceTests(unittest.TestCase):
    def sample(self, geometric=.016, impulse=0., cached=.0196):
        return {'other_robot_body_index': 35, 'normal_impulse_n_s': impulse,
                'min_solver_distance_m': cached,
                'geometric_distance_after_step_m': geometric}

    def test_speculative_pair_does_not_mean_hand_touch(self):
        self.assertFalse(robot_contact_blocks_release(self.sample()))

    def test_current_touch_overrides_positive_cached_distance(self):
        self.assertTrue(robot_contact_blocks_release(self.sample(geometric=0.)))

    def test_separation_requires_zero_force(self):
        self.assertTrue(robot_contact_blocks_release(self.sample(impulse=.001)))

    def test_unsupported_geometry_is_not_release_evidence(self):
        self.assertTrue(robot_contact_blocks_release(self.sample(geometric=None)))

    def test_cached_only_trace_cannot_certify_current_separation(self):
        sample = self.sample()
        del sample['geometric_distance_after_step_m']
        with self.assertRaises(ValueError):
            robot_contact_blocks_release(sample)

    def test_nonfinite_truth_is_rejected(self):
        with self.assertRaises(ValueError):
            robot_contact_blocks_release(self.sample(geometric=float('nan')))
        with self.assertRaises(ValueError):
            robot_contact_blocks_release(self.sample(impulse=float('nan')))


if __name__ == '__main__':
    unittest.main()
