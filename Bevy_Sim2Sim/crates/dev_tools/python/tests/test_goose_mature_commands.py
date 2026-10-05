"""Retain a trained gait when activating formerly unused goal inputs."""
from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.mature_training import initialize_unused_lateral_yaw


def make_transfer_case():
    def model(size):
        mean = torch.zeros((1, size))
        variance = torch.ones((1, size))
        mean[:, 6] = .04
        variance[:, 7:9] = 0.
        return SimpleNamespace(mlp=[torch.nn.Linear(size, 18)],
            obs_normalizer=SimpleNamespace(_mean=mean, _var=variance,
                _std=variance.sqrt()))
    actor, critic = model(65), model(69)
    state = {}
    for net in (actor, critic):
        moment = torch.ones_like(net.mlp[0].weight)
        moment[:, 7:9] = 0.
        state[net.mlp[0].weight] = {"exp_avg": moment.clone(),
            "exp_avg_sq": moment.clone(), "step": torch.tensor(12.)}
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic,
        optimizer=SimpleNamespace(state=state)))
    commands = SimpleNamespace(rel_standing_envs=.25, ranges=SimpleNamespace(
        lin_vel_y=(-.05, .05), ang_vel_z=(-.3, .3)))
    return runner, commands


def output(model, raw):
    n = model.obs_normalizer
    return model.mlp[0]((raw-n._mean)/(n._std+.01))


def test_unused_inputs_keep_forward_policy_and_optimizer_history():
    runner, commands = make_transfer_case()
    snapshots = []
    for net in (runner.alg.actor, runner.alg.critic):
        obs = torch.randn((32, net.mlp[0].in_features))
        obs[:, 7:9] = 0.
        moments = {k: v.clone() for k, v in runner.alg.optimizer.state[net.mlp[0].weight].items()}
        snapshots.append((net, obs, output(net, obs).detach(),
            net.mlp[0].weight[:, 6].detach().clone(), moments))
    initialize_unused_lateral_yaw(runner, commands)
    for net, obs, expected, forward, moments in snapshots:
        torch.testing.assert_close(output(net, obs), expected, atol=0, rtol=0)
        obs[:, 7:9] = torch.tensor([.05, .3])
        torch.testing.assert_close(output(net, obs), expected, atol=0, rtol=0)
        assert torch.equal(net.mlp[0].weight[:, 6], forward)
        for key, value in moments.items():
            assert torch.equal(runner.alg.optimizer.state[net.mlp[0].weight][key], value)
    with pytest.raises(ValueError, match="already have learned statistics"):
        initialize_unused_lateral_yaw(runner, commands)


def test_rejecting_trained_critic_never_partially_mutates_actor():
    runner, commands = make_transfer_case()
    runner.alg.critic.obs_normalizer._var[:, 8] = .01
    before = runner.alg.actor.mlp[0].weight.detach().clone()
    with pytest.raises(ValueError, match="already have learned statistics"):
        initialize_unused_lateral_yaw(runner, commands)
    assert torch.equal(runner.alg.actor.mlp[0].weight, before)


def test_nonfinite_command_ranges_are_rejected_before_transfer():
    runner, commands = make_transfer_case()
    commands.ranges.ang_vel_z = (-float("inf"), float("inf"))
    with pytest.raises(ValueError, match="finite nonzero"):
        initialize_unused_lateral_yaw(runner, commands)
