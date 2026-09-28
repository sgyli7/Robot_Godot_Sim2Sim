"""Production BAM/reward reset functions on small CPU tensors, with no physics."""
import unittest
from types import SimpleNamespace

try:
    import torch
    import mjlab_microduck.tasks  # Populate the official task registry before BAM import.
    from bam.mjlab import BamActuator
    from mjlab_microduck.tasks import mdp
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires the adopted scientific environment')
class ResetModeTests(unittest.TestCase):
    def fixture(self):
        feedback=torch.arange(1,29,dtype=torch.float32).reshape(2,14)/100
        scene={'robot':SimpleNamespace(data=SimpleNamespace(actuator_force=feedback,joint_vel=feedback+1))}
        env=SimpleNamespace(scene=scene,num_envs=2,device='cpu')
        actuator=object.__new__(BamActuator)
        actuator._delay_buffer=None
        with torch.inference_mode():
            mdp.joint_torque_rate_l2(env)
            # This exact named BAM history field is manufactured on CPU;
            # BAM compute/physics are not claimed to have executed in this fixture.
            actuator._prev_motor_torque=feedback.clone()
        return env,actuator,feedback

    def test_original_resets_reject_outside_inference_mode(self):
        for operation in ('BAM','reward'):
            env,actuator,feedback=self.fixture()
            self.assertTrue(torch.is_inference(env._prev_actuator_forces))
            self.assertTrue(torch.is_inference(actuator._prev_motor_torque))
            with self.subTest(operation=operation),self.assertRaisesRegex(RuntimeError,'InferenceMode'):
                if operation=='BAM':actuator.reset(torch.tensor([0]))
                else:mdp.reset_action_history(env,torch.tensor([0]))
            self.assertTrue(torch.equal(env.scene['robot'].data.actuator_force,feedback))

    def test_matching_mode_partial_and_full_reset_preserve_feedback(self):
        env,actuator,feedback=self.fixture()
        second=actuator._prev_motor_torque[1].clone()
        with torch.inference_mode():
            actuator.reset(torch.tensor([0]))
            mdp.reset_action_history(env,torch.tensor([0]))
        self.assertTrue(torch.equal(actuator._prev_motor_torque[0],torch.zeros(14)))
        self.assertTrue(torch.equal(actuator._prev_motor_torque[1],second))
        self.assertTrue(torch.equal(env._prev_actuator_forces,feedback))
        self.assertTrue(torch.equal(env.scene['robot'].data.actuator_force,feedback))
        with torch.inference_mode():
            actuator.reset()
            mdp.reset_action_history(env,torch.tensor([0,1]))
        self.assertTrue(torch.equal(actuator._prev_motor_torque,torch.zeros_like(feedback)))
        self.assertTrue(torch.equal(env._prev_actuator_forces,feedback))
        self.assertTrue(torch.is_inference(env._prev_actuator_forces))
        print('TEST_ONLY actual BAM/reset_action_history CPU functions: same-mode legal; native feedback unchanged; zero compute/integration/learning')


if __name__=='__main__':unittest.main()
