"""Recovery goal semantics and real native reset/measurement boundaries."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.goal_recovery import goal_scores, make_goal_recovery_cfg
from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv


def test_goal_does_not_award_composite_for_low_vertical_or_high_horizontal_body():
    height = torch.tensor([.332, .166, .332, .332])
    quaternion = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.],
        [2**-.5, 0., 2**-.5, 0.], [1., 0., 0., 0.]])
    pose = torch.zeros((4, 18))
    pose[3, 9] = 1.2
    scores = goal_scores(height, quaternion, pose, torch.zeros_like(pose), target_height=.332)
    assert scores["standing_composite"][0] == 1
    assert scores["standing_composite"][1] < .13
    assert scores["standing_composite"][2] < .002
    assert scores["standing_composite"][3] < .48
    assert scores["upright_sharp"][1] == 0
    assert scores["height_l1"][1] < 0 and scores["leg_pose_l1"][3] < 0


def test_height_goal_is_invariant_to_robot_length_units():
    quaternion = torch.tensor([[1., 0., 0., 0.], [2**-.5, 2**-.5, 0., 0.]])
    pose = torch.zeros((2, 18))
    original = goal_scores(torch.tensor([.115, .06]), quaternion, pose, pose,
        target_height=.115)
    scaled = goal_scores(torch.tensor([.115, .06])*3, quaternion, pose, pose,
        target_height=.115*3)
    for name in original:
        torch.testing.assert_close(original[name], scaled[name])


def test_native_com_critic_and_selective_cold_reset_preserve_running_worlds(tmp_path, monkeypatch):
    root = os.environ.get("GOOSE_GOAL_EXPERIMENT")
    if not root:
        pytest.skip("Requires the named frozen Goose goal-task inputs")
    root = Path(root)
    monkeypatch.chdir(tmp_path)
    poses = np.load(root/"births.npz")["qpos"]
    cfg = make_goal_recovery_cfg(root/"candidate/robot.xml", root/"candidate/contract.json",
        poses=poses, num_envs=4, frontier_step=0)
    assert cfg.sim.mujoco.timestep == .02 and cfg.decimation == 1 and not cfg.auto_reset
    assert set(cfg.terminations) == {"time_out", "contact_domain_failure"}
    env = GooseDevelopmentEnv(cfg, "cpu")
    try:
        observations, _ = env.reset()
        assert observations["actor"].shape == (4, 65)
        assert observations["critic"].shape == (4, 69)
        torso_height = env.scene["robot"].data.root_com_pos_w[:, 2]-env.scene.env_origins[:, 2]
        torch.testing.assert_close(observations["critic"][:, -1], torso_height.float())
        assert bool((torso_height-env.sim.data.qpos[:, 2]).abs().max() > .1)
        before = env.sim.data.time.clone()
        observations, rewards, failed, _, _ = env.step(torch.zeros((4, 18)))
        assert torch.isfinite(rewards).all() and not failed.any()
        torch.testing.assert_close(env.sim.data.time-before, torch.full_like(before, .02), atol=1e-6, rtol=0)
        term = env.action_manager.get_term("goose")
        state = env.sim.data.qpos[1:].clone()
        velocity = env.sim.data.qvel[1:].clone()
        target = term.drive.target[1:].clone()
        clock = env.sim.data.time[1:].clone()
        integrations = term.drive.completed_ticks[1:].clone()
        env.reset(env_ids=torch.tensor([0]))
        for actual, expected in ((env.sim.data.qpos[1:], state), (env.sim.data.qvel[1:], velocity),
                (term.drive.target[1:], target), (env.sim.data.time[1:], clock),
                (term.drive.completed_ticks[1:], integrations)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert int(term.drive.completed_ticks[0]) == 0
        assert env.contact_adapter is None
    finally:
        env.close()
