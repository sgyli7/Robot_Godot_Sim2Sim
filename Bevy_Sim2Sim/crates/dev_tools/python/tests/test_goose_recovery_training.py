"""Actual native get-up reset boundaries, contact and stable-hold measurements."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv
from bevy_microduck_tools.goose.recovery_training import make_recovery_cfg


def test_fallen_births_keep_real_contact_and_selective_reset_does_not_integrate(
        tmp_path, monkeypatch):
    root = os.environ.get("GOOSE_RECOVERY_EXPERIMENT")
    if not root:
        pytest.skip("Frozen named rigid recovery inputs required")
    monkeypatch.chdir(tmp_path)
    root = Path(root)
    library = root / "recovery_domain/reset_states.npz"
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    cfg = make_recovery_cfg(root / "candidate/robot.xml", root / "candidate/contract.json",
        library, digest, num_envs=16, stage=2)
    assert set(cfg.terminations) == {"time_out"}
    assert cfg.decimation == 1 and cfg.sim.mujoco.timestep == .02
    env = GooseDevelopmentEnv(cfg, "cpu")
    try:
        observations, _ = env.reset()
        assert observations["actor"].shape == (16, 65)
        assert observations["critic"].shape == (16, 69)
        poses = torch.from_numpy(np.load(library)["qpos"]).to(env.sim.data.qpos)
        actual = env.sim.data.qpos.clone()
        actual[:, :3] -= env.scene.env_origins
        torch.testing.assert_close(actual, poses[env._goose_recovery_cases], atol=2e-6, rtol=0)
        assert bool((actual[:, 3] < .9).any())
        for _ in range(20):
            observations, reward, failed, timed_out, _ = env.step(torch.zeros((16, 18)))
            assert torch.isfinite(reward).all()
            assert not failed.any() and not timed_out.any()
        torch.testing.assert_close(env.sim.data.time.clone(), torch.full_like(env.sim.data.time, .4),
            atol=1e-6, rtol=0)
        remaining = torch.arange(1, 16)
        state = env.sim.data.qpos[remaining].clone()
        clock = env.sim.data.time[remaining].clone()
        drive = env.action_manager.get_term("goose").drive
        ticks = drive.completed_ticks[remaining].clone()
        env.reset(env_ids=torch.tensor([0]))
        torch.testing.assert_close(env.sim.data.qpos[remaining], state, atol=0, rtol=0)
        torch.testing.assert_close(env.sim.data.time[remaining], clock, atol=0, rtol=0)
        torch.testing.assert_close(drive.completed_ticks[remaining], ticks, atol=0, rtol=0)
        assert int(drive.completed_ticks[0]) == 0
        assert env.contact_adapter is None
        assert int(env._goose_recovery_stable_ticks.max()) <= 20
    finally:
        env.close()


def test_dense_stability_requires_raised_upright_and_rewards_reduced_motion(monkeypatch):
    from bevy_microduck_tools.goose import recovery_training as course
    up = torch.tensor([.64, .99, .99, .99])
    height = torch.tensor([1., .6, 1., 1.])
    velocity = torch.tensor([[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.], [0., 0., 0.]])
    angular = torch.zeros((4, 3))
    monkeypatch.setattr(course, "recovery_state", lambda env: (up, height, velocity, angular))
    reward = course.recovery_dense_stand_reward(None, 1.)
    assert reward[0] == reward[1] == 0.
    assert 0. < reward[2] < reward[3] <= 1.
    assert course.recovery_stable(None, 1.).tolist() == [False, False, False, True]


def test_near_stand_reference_leaves_folded_getup_actions_and_hold_metric_unchanged(monkeypatch):
    from types import SimpleNamespace
    from bevy_microduck_tools.goose import recovery_training as course

    up = torch.tensor([.89, .99, .99, .99])
    height = torch.tensor([1., .84, 1., 1.])
    velocity = torch.zeros((4, 3))
    angular = torch.zeros((4, 3))
    actions = torch.full((4, 18), .2)
    actions[3] = 0.
    original = actions.clone()
    env = SimpleNamespace(action_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(drive=SimpleNamespace(actions=actions))))
    monkeypatch.setattr(course, "recovery_state", lambda _: (up, height, velocity, angular))

    before = course.recovery_stable(env, 1.).clone()
    cost = course.near_stand_target_cost(env, 1.)
    torch.testing.assert_close(cost, torch.tensor([0., 0., .04, 0.]))
    assert torch.equal(actions, original)
    assert torch.equal(course.recovery_stable(env, 1.), before)


def test_progressive_birth_preserves_other_worlds_and_counts_real_hold(tmp_path, monkeypatch):
    from bevy_microduck_tools.goose.recovery_training import make_progressive_recovery_cfg

    root = os.environ.get("GOOSE_PROGRESSIVE_EXPERIMENT")
    if not root:
        pytest.skip("Frozen progressive recovery inputs required")
    root = Path(root)
    monkeypatch.chdir(tmp_path)
    protocol = __import__("json").loads((root / "protocol.json").read_text())
    cfg = make_progressive_recovery_cfg(root / "candidate/robot.xml",
        root / "candidate/contract.json", root / "parent_library/reset_states.npz",
        protocol["parent_library_sha256"], root / "progressive_births/reset_states.npz",
        protocol["progressive_sha256"], num_envs=4, contact_failure_depth_m=.05)
    env = GooseDevelopmentEnv(cfg, "cpu")
    try:
        observations, _ = env.reset()
        assert observations["actor"].shape == (4, 65)
        assert observations["critic"].shape == (4, 69)
        for _ in range(16):
            env.step(torch.zeros((4, 18)))
        assert env._goose_recovery_max_hold.max() <= 16
        pose = env.sim.data.qpos[1:].clone()
        clock = env.sim.data.time[1:].clone()
        ticks = env.action_manager.get_term("goose").drive.completed_ticks[1:].clone()
        env.reset(env_ids=torch.tensor([0]))
        assert torch.equal(pose, env.sim.data.qpos[1:])
        assert torch.equal(clock, env.sim.data.time[1:])
        assert torch.equal(ticks, env.action_manager.get_term("goose").drive.completed_ticks[1:])
        assert env._goose_recovery_max_hold[0] == 0
        assert env._goose_recovery_success_tick[0] == -1
    finally:
        env.close()
