"""Original mobile contract admission guards; no simulator or model is loaded."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unitree_g1_t2_source_task import GROUP_NAMES, PROFILE, SOURCE_PROFILES, main, validate_reply, verify_source_receipt


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


class MobileSourceIdentityGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.roots = {name: base / name for name in ('arena', 'lab')}
        self.profile = 'release_0_2_1'
        revisions = SOURCE_PROFILES[self.profile]
        self.receipt = base / 'receipt.json'
        self.data = {'schema': 'g1_t2_source_tree_v1', 'source_profile': self.profile,
                     'commits': {'arena': revisions[0], 'lab': revisions[1]}, 'files': []}
        for name, root in self.roots.items():
            root.mkdir()
            file = root / 'module.py'; file.write_text(name)
            self.data['files'].append({'root': name, 'path': 'module.py',
                                      'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})

    def verify(self):
        self.receipt.write_text(json.dumps(self.data))
        sha = hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        return verify_source_receipt(self.receipt, sha, self.roots, self.profile)

    def test_mounted_source_bytes_must_match_even_with_unavailable_git(self):
        self.assertEqual(self.verify(), 2)
        (self.roots['lab'] / 'module.py').write_text('different source')
        with self.assertRaises(ValueError):
            self.verify()

    def test_other_revision_or_omitted_root_is_rejected(self):
        original = copy.deepcopy(self.data)
        self.data['commits']['arena'] = SOURCE_PROFILES['development_0_3'][0]
        with self.assertRaises(ValueError):
            self.verify()
        self.data = original; self.data['files'] = self.data['files'][:1]
        with self.assertRaises(ValueError):
            self.verify()

    def test_external_symlink_or_traversal_cannot_supply_source(self):
        outside = Path(self.temporary.name) / 'external.py'; outside.write_text('arena')
        module = self.roots['arena'] / 'module.py'; module.unlink(); module.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.verify()
        self.data['files'][0]['path'] = '../external.py'
        with self.assertRaises(ValueError):
            self.verify()

    def test_duplicate_paths_and_changed_receipt_are_rejected(self):
        self.data['files'].append(copy.deepcopy(self.data['files'][0]))
        with self.assertRaises(ValueError):
            self.verify()
        with self.assertRaises(ValueError):
            verify_source_receipt(self.receipt, '0'*64, self.roots, self.profile)


class RenderAuditAdmission(unittest.TestCase):
    def test_light_probe_requires_exclusive_zero_step_render_audit(self):
        base = ['source_query', '--arena-source', '/unused', '--lab-source', '/unused',
                '--homie-assets', '/unused', '--task-assets', '/unused',
                '--output', '/unused', '--episode-id', '5', '--run-source',
                '--source-profile', 'release_0_2_1', '--scene-light-causal-probe']
        for extra in ([], ['--scene-render-audit', '--policy-port', '5558'],
                      ['--scene-render-audit', '--ticks', '100', '--policy-port', '5558']):
            with self.subTest(extra=extra), patch('sys.argv', base + extra), \
                    patch('unitree_g1_t2_source_task.source_check') as source, \
                    patch('sys.stderr'), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            source.assert_not_called()

    def test_render_query_cannot_run_with_model_motion_or_other_query(self):
        base = ['source_query', '--arena-source', '/unused', '--lab-source', '/unused',
                '--homie-assets', '/unused', '--task-assets', '/unused',
                '--output', '/unused', '--episode-id', '5', '--run-source',
                '--source-profile', 'release_0_2_1', '--scene-render-audit']
        for extra in (['--policy-port', '5558'], ['--ticks', '100', '--policy-port', '5558'],
                      ['--contact-settings-audit'], ['--background-owner-audit']):
            with self.subTest(extra=extra), patch('sys.argv', base + extra), \
                    patch('unitree_g1_t2_source_task.source_check') as source, \
                    patch('sys.stderr'), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            source.assert_not_called()


if __name__ == '__main__':
    unittest.main()
