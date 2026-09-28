"""Exercise the actual installed timestamp adapter on CPU Torch batch tensors."""

import unittest
from types import SimpleNamespace

try:
    import torch
    from mjlab.utils.buffers.delay_buffer import DelayBuffer
except ImportError:
    torch = None

from bevy_microduck_tools.source_adapter import delay_plan
from bevy_microduck_tools.timing import install_delays


@unittest.skipIf(torch is None, "Requires the adopted upstream scientific environment")
class SourceTimingTests(unittest.TestCase):
    def setUp(self):
        home = torch.linspace(-.4, .4, 14).repeat(2, 1)
        data = SimpleNamespace(default_joint_pos=home, encoder_bias=torch.zeros_like(home))
        actuator = SimpleNamespace(target_ids=torch.arange(14), _delay_buffer=DelayBuffer(batch_size=2))
        self.env = SimpleNamespace(_sim_step_counter=0, physics_dt=1/60,
                                   observation_manager=SimpleNamespace(_group_obs_term_delay_buffer={}),
                                   scene={"robot": SimpleNamespace(data=data, actuators=[actuator])})
        plan = {"physics_dt": 1/60, "observation": {}, "motor": [delay_plan(3, 6, .005, 1/60)]}
        install_delays(self.env, plan)
        self.buffer = actuator._delay_buffer
        self.home = home

    def test_reset_warmup_does_not_store_or_sample_framework_zero(self):
        self.buffer.reset()
        for _ in range(3):
            self.buffer.append(torch.zeros((2, 14)))
            self.assertTrue(torch.equal(self.buffer.compute(), self.home))
        self.assertTrue(torch.equal(self.buffer.append_count, torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.equal(self.buffer._step_count, torch.zeros(2, dtype=torch.long)))
        self.assertIsNone(self.buffer._history)

    def test_valid_real_zero_target_is_stored_and_served(self):
        self.buffer.reset()
        for tick in range(1, 4):
            self.env._sim_step_counter = tick
            self.buffer.append(torch.zeros((2, 14)))
            result = self.buffer.compute()
        self.assertTrue(torch.equal(result, torch.zeros((2, 14))))
        self.assertTrue(torch.equal(self.buffer.append_count, torch.full((2,), 3)))
        self.assertTrue(torch.all(self.buffer._timestamps[:, :3] >= 0))

    def test_repeat_read_and_partial_reset_preserve_other_world(self):
        for tick in range(1, 4):
            self.env._sim_step_counter = tick
            self.buffer.append(torch.full((2, 14), float(tick)))
            self.buffer.compute()
        other_times = self.buffer._timestamps[1].clone()
        other_history = self.buffer._history[1].clone()
        other_count = self.buffer.append_count[1].clone()
        other_output = self.buffer._cached[1].clone()
        rng = torch.random.get_rng_state().clone()
        self.buffer.append(torch.full((2, 14), 999.0))
        self.buffer.compute()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.buffer.reset(torch.tensor([0]))
        self.buffer.append(torch.zeros((2, 14)))
        warmup = self.buffer.compute()
        self.assertTrue(torch.equal(warmup[0], self.home[0]))
        self.assertTrue(torch.equal(warmup[1], other_output))
        self.assertTrue(torch.equal(other_times, self.buffer._timestamps[1]))
        self.assertTrue(torch.equal(other_history, self.buffer._history[1]))
        self.assertTrue(torch.equal(other_count, self.buffer.append_count[1]))
        self.env._sim_step_counter = 4
        self.buffer.append(torch.zeros((2, 14)))
        self.buffer.compute()
        self.assertEqual(float(self.buffer._timestamps[0, 0]), 0.0)
        self.assertEqual(int(self.buffer.append_count[0]), 1)

    def test_ppo_inference_collection_then_normal_reset(self):
        with torch.inference_mode():
            self.env._sim_step_counter = 1
            self.buffer.append(torch.ones((2, 14)))
            self.buffer.compute()
        self.buffer.reset()
        self.assertEqual(int(self.buffer.append_count.sum()), 0)


if __name__ == "__main__":
    unittest.main()
