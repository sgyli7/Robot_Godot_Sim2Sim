from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.motion_tracking import (
    CompletedMotionState, FilteredAngularVelocityTracking,
    FilteredPlanarVelocityTracking, with_microduck_motion_tracking)


def make_env():
    body = SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        root_link_ang_vel_b=torch.zeros(2, 3))
    velocity = SimpleNamespace(data=torch.zeros(2, 3))
    commands = torch.zeros(2, 3)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02,
        common_step_counter=1, scene={"robot": SimpleNamespace(data=body), "com": velocity},
        command_manager=SimpleNamespace(get_command=lambda name: commands))
    return env, body, velocity, commands


def test_two_rewards_advance_once_and_leave_raw_motion_unchanged():
    env, body, velocity, commands = make_env()
    params = dict(command_name="velocity", std=.3, tracking_tau_s=.2,
        velocity_sensor_name="com")
    cfg = SimpleNamespace(params=params)
    planar = FilteredPlanarVelocityTracking(cfg, env)
    angular = FilteredAngularVelocityTracking(cfg, env)
    assert planar.state is angular.state
    planar(env, **params)
    env.common_step_counter += 1
    velocity.data[:] = torch.tensor([1., 2., 3.])
    body.root_link_ang_vel_b[:] = torch.tensor([.4, .5, .6])
    before = velocity.data.clone(), body.root_link_ang_vel_b.clone(), commands.clone()
    linear_result = planar(env, **params)
    filtered = planar.state.mean.clone()
    angular_result = angular(env, **params)
    assert torch.equal(filtered, planar.state.mean)
    assert torch.allclose(filtered[0], torch.tensor([1., 2., 3., .4, .5, .6])*planar.state.alpha)
    assert torch.allclose(linear_result, torch.exp(-filtered[:, :2].square().sum(1)/.3**2))
    assert torch.allclose(angular_result, torch.exp(-filtered[:, 3:].square().sum(1)/.3**2))
    assert all(torch.equal(x, y) for x, y in zip(before,
        (velocity.data, body.root_link_ang_vel_b, commands)))


def test_partial_reset_discards_only_that_worlds_old_reward_memory():
    env, _, velocity, _ = make_env()
    state = CompletedMotionState(env, .2, "com")
    velocity.data[:, 0] = torch.tensor([1., 2.])
    state.update(env)
    state.reset(torch.tensor([0]))
    env.common_step_counter += 1
    velocity.data[:, 0] = torch.tensor([3., 4.])
    actual = state.update(env)
    assert actual[0, 0] == 3.
    assert actual[1, 0] == pytest.approx(2.+2.*state.alpha)


def test_com_velocity_rotates_into_body_frame_and_requires_50hz():
    env, body, velocity, _ = make_env()
    body.root_link_quat_w[:] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    velocity.data[:, 0] = 1.
    state = CompletedMotionState(env, .2, "com")
    actual = state.update(env)
    assert torch.allclose(actual[:, :3], torch.tensor([[0., -1., 0.]]).repeat(2, 1), atol=2e-7)
    env.step_dt = .01
    with pytest.raises(ValueError, match="50 Hz"):
        CompletedMotionState(env, .2, "com")


def test_opt_in_preserves_old_reward_profile_and_all_other_configuration():
    linear = SimpleNamespace(func=object(), weight=5.,
        params={"command_name": "velocity", "std": .15, "sensor_name": "com"})
    angular = SimpleNamespace(func=object(), weight=2.,
        params={"command_name": "velocity", "std": .3})
    cfg = SimpleNamespace(rewards={"track_linear_velocity": linear,
        "track_angular_velocity": angular, "upright": "same_penalty"},
        commands={"three_axes": [-.15, .3, .1, .6]}, observations=[65, 69],
        physics=[.02, 1], action_contract=[18])
    result = with_microduck_motion_tracking(cfg)
    assert cfg.rewards["track_linear_velocity"] is linear
    assert cfg.rewards["track_linear_velocity"].params["sensor_name"] == "com"
    assert result.rewards["track_linear_velocity"].func is FilteredPlanarVelocityTracking
    assert result.rewards["track_angular_velocity"].func is FilteredAngularVelocityTracking
    assert result.rewards["track_linear_velocity"].weight == 5.
    assert result.rewards["upright"] == cfg.rewards["upright"]
    for field in ("commands", "observations", "physics", "action_contract"):
        assert getattr(result, field) == getattr(cfg, field)
