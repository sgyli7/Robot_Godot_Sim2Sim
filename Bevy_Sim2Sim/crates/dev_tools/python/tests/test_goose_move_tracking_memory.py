from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from bevy_microduck_tools.goose.motion_tracking import (
    CompletedMotionState, FilteredAngularVelocityTracking,
    FilteredPlanarVelocityTracking)
from bevy_microduck_tools.goose.move_tracking_memory import (
    motion_tracking_memory_observation, with_move_tracking_memory_critic)


def actor_signal(env):
    return torch.zeros(env.num_envs, 65)


def make_cfg():
    params = dict(tracking_tau_s=.2, velocity_sensor_name="com")
    return SimpleNamespace(observations={
        "actor": ObservationGroupCfg(terms={"base": ObservationTermCfg(func=actor_signal)}),
        "critic": ObservationGroupCfg(terms={"base": ObservationTermCfg(func=actor_signal)})},
        rewards={"track_linear_velocity": SimpleNamespace(
            func=FilteredPlanarVelocityTracking, weight=5., params=dict(params)),
            "track_angular_velocity": SimpleNamespace(
                func=FilteredAngularVelocityTracking, weight=2., params=dict(params))},
        physical_contract=[50, 200, 18], commands=[.4, .7, .3])


def test_read_is_detached_from_filter_update_and_partial_reset_discards_stale_values():
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02)
    state = CompletedMotionState(env, .2, "com")
    env._goose_motion_tracking_state = state
    state.mean[:] = torch.arange(12).reshape(2, 6)
    state.ready[:] = True
    before = state.mean.clone(), state.ready.clone(), state.last_step
    observation = motion_tracking_memory_observation(env)
    assert torch.equal(observation[:, :6], state.mean)
    assert torch.equal(observation[:, 6], torch.ones(2))
    observation[:] = -100.
    assert torch.equal(state.mean, before[0]) and torch.equal(state.ready, before[1])
    assert state.last_step == before[2]
    state.reset(torch.tensor([0]))
    observation = motion_tracking_memory_observation(env)
    assert observation[0].count_nonzero() == 0
    assert torch.equal(observation[1, :6], before[0][1]) and observation[1, 6] == 1.


def test_uninitialized_reward_manager_and_corrupted_memory_are_distinguished():
    env = SimpleNamespace(num_envs=2, device="cpu")
    assert torch.equal(motion_tracking_memory_observation(env), torch.zeros(2, 7))
    env._goose_motion_tracking_state = SimpleNamespace(mean=torch.zeros(2, 5), ready=torch.zeros(2))
    with pytest.raises(ValueError, match="six EMA"):
        motion_tracking_memory_observation(env)


def test_opt_in_preserves_actor_rewards_control_and_default_profile():
    cfg = make_cfg()
    result = with_move_tracking_memory_critic(cfg)
    assert "motion_tracking_memory" not in cfg.observations["critic"].terms
    assert result.observations["actor"] == cfg.observations["actor"]
    assert result.rewards == cfg.rewards
    assert result.physical_contract == cfg.physical_contract and result.commands == cfg.commands
    assert result.observations["critic"].terms["motion_tracking_memory"].func is motion_tracking_memory_observation
    with pytest.raises(ValueError, match="twice"):
        with_move_tracking_memory_critic(result)
    cfg.rewards["track_linear_velocity"].func = actor_signal
    with pytest.raises(ValueError, match="filtered"):
        with_move_tracking_memory_critic(cfg)


def test_same_actor_state_with_distinct_reward_histories_gets_distinct_critic_state():
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02)
    state = CompletedMotionState(env, .2, "com")
    env._goose_motion_tracking_state = state
    state.ready[:] = True
    state.mean[1, 5] = .3
    actor = actor_signal(env)
    critic = torch.cat((actor, motion_tracking_memory_observation(env)), dim=-1)
    assert torch.equal(actor[0], actor[1])
    assert not torch.equal(critic[0], critic[1])
    assert critic.shape == (2, 72)
