from types import SimpleNamespace

import pytest
import torch

from rsl_rl.modules import EmpiricalNormalization

from bevy_microduck_tools.goose.command_encoding import remap_forward_command_encoder


def make_model(size):
    normalizer = EmpiricalNormalization(size)
    normalizer.train()
    torch.manual_seed(size)
    normalizer.update(torch.randn(128, size) * .3 + .2)
    return SimpleNamespace(obs_dim=size, obs_normalizer=normalizer)


def test_calibration_preserves_parent_function_including_rsl_epsilon():
    actor, critic = make_model(65), make_model(69)
    before = []
    for model in (actor, critic):
        obs = torch.randn(40, model.obs_dim)
        old_input = obs.clone()
        old_input[:, 6] *= 2.
        before.append((obs, model.obs_normalizer(old_input).clone(),
            {k: v.clone() for k, v in model.obs_normalizer.state_dict().items()}))
    report = remap_forward_command_encoder(actor, critic, factor=2.)
    assert report["reward_command_rescaled"] is False
    for model, (obs, expected, old_state) in zip((actor, critic), before):
        torch.testing.assert_close(model.obs_normalizer(obs), expected, rtol=2e-7, atol=1e-6)
        for name in ("_mean", "_std", "_var"):
            mask = torch.arange(model.obs_dim) != 6
            assert torch.equal(model.obs_normalizer.state_dict()[name][:, mask], old_state[name][:, mask])
        frozen = {k: v.clone() for k, v in model.obs_normalizer.state_dict().items()}
        model.obs_normalizer.update(torch.randn(128, model.obs_dim))
        assert all(torch.equal(v, frozen[k]) for k, v in model.obs_normalizer.state_dict().items())


def test_failed_validation_cannot_partially_change_actor_encoder():
    actor, critic = make_model(65), make_model(69)
    critic.obs_normalizer._std[:, 6] = 0.
    saved = {k: v.clone() for k, v in actor.obs_normalizer.state_dict().items()}
    with pytest.raises(ValueError, match="positive finite"):
        remap_forward_command_encoder(actor, critic, factor=2.)
    assert all(torch.equal(v, saved[k]) for k, v in actor.obs_normalizer.state_dict().items())
    assert actor.obs_normalizer.until is None


@pytest.mark.parametrize("factor", (0., -1., float("nan"), float("inf")))
def test_invalid_factors_are_rejected(factor):
    with pytest.raises(ValueError, match="finite and positive"):
        remap_forward_command_encoder(make_model(65), make_model(69), factor=factor)


def test_same_encoder_cannot_be_rescaled_twice():
    actor, critic = make_model(65), make_model(69)
    remap_forward_command_encoder(actor, critic, factor=2.)
    with pytest.raises(ValueError, match="already calibrated"):
        remap_forward_command_encoder(actor, critic, factor=2.)
