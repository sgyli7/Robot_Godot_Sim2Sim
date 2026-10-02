"""Validation guards for original demonstration diagnostic input."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from unitree_g1_source_expert import DATASET_ID, DATASET_REVISION, FROZEN_FILES, JOINT_NAMES, load_sequence


class ExpertSequenceTests(unittest.TestCase):
    def setUp(self):
        self.value = {'schema': 'g1_original_source_expert_v1', 'dataset_id': DATASET_ID,
            'dataset_revision': DATASET_REVISION, 'episode_index': 0, 'input_sha256': FROZEN_FILES,
            'joint_names': JOINT_NAMES, 'action_period_ns': 20_000_000,
            'initial_joint_positions': [0.] * 43,
            'after_sequence': 'hold_last_frame_until_original_termination',
            'frames': [{'left_arm': [float(x) for x in range(7)], 'left_hand': [0.] * 7,
                'right_arm': [0.] * 7, 'right_hand': [0.] * 7, 'waist': [0.] * 3,
                'navigate_mps_rps': [0.] * 3, 'base_height_m': .72} for _ in range(154)]}

    def read(self, value, wrong_hash=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expert.json'
            path.write_text(json.dumps(value))
            expected = '0' * 64 if wrong_hash else hashlib.sha256(path.read_bytes()).hexdigest()
            return load_sequence(path, expected)

    def test_identity_and_named_hand_order_survive_loading(self):
        loaded = self.read(self.value)
        self.assertEqual(loaded['frames'][0]['left_arm'], list(range(7)))
        self.assertEqual(loaded['joint_names'][22], 'left_hand_index_0_joint')

    def test_hash_joint_order_and_time_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            self.read(self.value, wrong_hash=True)
        for field, changed in [('joint_names', JOINT_NAMES[::-1]), ('action_period_ns', 5_000_000),
                               ('dataset_revision', 'unfrozen'), ('after_sequence', 'repeat')]:
            value = copy.deepcopy(self.value); value[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(value)

    def test_nonfinite_wrong_shapes_and_recorded_waist_are_rejected(self):
        for field, changed in [('left_hand', [0.] * 6), ('right_arm', [float('nan')] * 7),
                               ('waist', [1., 0., 0.]), ('base_height_m', True)]:
            value = copy.deepcopy(self.value); value['frames'][0][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(value)


if __name__ == '__main__':
    unittest.main()
