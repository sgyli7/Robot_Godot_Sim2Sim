"""Original mobile contract admission guards; no simulator or model is loaded."""
import copy
import unittest

from unitree_g1_t2_source_task import GROUP_NAMES, PROFILE, validate_reply


class MobileSourceGuards(unittest.TestCase):
    def setUp(self):
        self.request = {'profile': 'mobile_box', 'sequence_id': 2,
                        'observation': {'episode_id': 8, 'frame_id': 50,
                                        'sim_time_ns': 1_000_000_000, 'captured_at_unix_ms': 20}}
        frame = {key: [0.0] * len(names) for key, names in GROUP_NAMES.items()}
        frame.update(base_height_m=.75, navigate_mps_rps=[0.0, 0.0, 0.0])
        self.reply = {**copy.deepcopy(self.request), 'model_revision': PROFILE['revision'],
                      'action_period_ns': 20_000_000, 'frames': [copy.deepcopy(frame) for _ in range(50)]}

    def test_retains_original_fifty_frame_identity(self):
        self.assertIs(validate_reply(self.request, self.reply), self.reply['frames'])

    def test_static_profile_horizon_and_foreign_episode_are_rejected(self):
        for field, value in [('profile', 'static_apple'), ('frames', self.reply['frames'][:40]),
                             ('model_revision', '7f78bebf1a90131e7304beacfcd47eb27bad16ab'),
                             ('action_period_ns', 5_000_000), ('sequence_id', 3)]:
            changed = copy.deepcopy(self.reply); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reply(self.request, changed)
        changed = copy.deepcopy(self.reply); changed['observation']['episode_id'] += 1
        with self.assertRaises(ValueError):
            validate_reply(self.request, changed)

    def test_rebased_stamp_and_boolean_identity_are_rejected(self):
        for field, value in [('frame_id', 51), ('sim_time_ns', 1_020_000_000), ('episode_id', True)]:
            changed = copy.deepcopy(self.reply); changed['observation'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reply(self.request, changed)

    def test_malformed_action_or_truth_payload_is_rejected(self):
        for group, value in [('left_arm', [0.0] * 6), ('left_hand', [float('nan')] * 7),
                             ('navigate_mps_rps', [True, 0.0, 0.0]), ('base_height_m', float('inf')),
                             ('object_pose', [1.0, 2.0, 3.0])]:
            changed = copy.deepcopy(self.reply); changed['frames'][0][group] = value
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_reply(self.request, changed)


if __name__ == '__main__':
    unittest.main()
