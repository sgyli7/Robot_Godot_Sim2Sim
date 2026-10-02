"""Boundary guards only; no fixture qualifies an original or native robot."""
import unittest

from unitree_g1_t1_source_task import take_prefetched_chunk


class SourceSchedulingGuards(unittest.TestCase):
    def test_replacement_preserves_original_frames_and_acquisition_identity(self):
        frames = [{'original_frame': i} for i in range(40)]
        pending = {'frames': frames, 'sequence': 2, 'observed_tick': 10}
        actual, sequence, observed = take_prefetched_chunk(pending, 40, 40)
        self.assertIs(actual, frames)
        self.assertEqual([frame['original_frame'] for frame in actual], list(range(40)))
        self.assertEqual((sequence, observed), (2, 10))
        self.assertEqual(pending['observed_tick'], 10)

    def test_cannot_skip_predecessor_or_apply_early_late_or_rebased(self):
        pending = {'frames': list(range(40)), 'sequence': 3, 'observed_tick': 50}
        for tick, used in [(50, 10), (79, 39), (80, 39), (80, 41), (81, 40), (120, 40)]:
            with self.subTest(tick=tick, used=used), self.assertRaises(ValueError):
                take_prefetched_chunk(pending, tick, used)
        _, sequence, observed = take_prefetched_chunk(pending, 80, 40)
        self.assertEqual((sequence, observed), (3, 50))

    def test_missing_or_partial_replacement_fails(self):
        for pending in [None, {'frames': list(range(39)), 'sequence': 2, 'observed_tick': 10}]:
            with self.subTest(pending=pending), self.assertRaises(ValueError):
                take_prefetched_chunk(pending, 40, 40)


if __name__ == '__main__':
    unittest.main()
