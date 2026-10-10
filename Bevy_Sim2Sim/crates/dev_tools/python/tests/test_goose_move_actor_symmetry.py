"""Exercise the installed RSL extension/PPO with an unchanged terrain Critic."""
import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import Symmetry
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from bevy_microduck_tools.goose.native_action_units import (
    PublicActionScale, PublicUnitsGaussian,
)
from bevy_microduck_tools.goose.native_symmetry import mirror_public_observation
from bevy_microduck_tools.goose.move_actor_symmetry import (
    augment_actor_only_symmetry, install_actor_only_native_mirror_units,
    make_actor_only_mirror_cfg,
)


def observations():
    return TensorDict({"actor": torch.randn(4, 65),
                       "critic": torch.randn(4, 275)}, batch_size=[4])


def mapped_actor(obs, scale):
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 18,
                    hidden_dims=(16,), obs_normalization=True,
                    distribution_cfg=dict(class_name="GaussianDistribution",
                                          std_type="log", init_std=0.2))
    actor.distribution = PublicUnitsGaussian(actor.distribution, scale)
    actor.mlp.add_module("public_units", PublicActionScale(scale))
    return actor


def extension():
    return Symmetry(env=None, **make_actor_only_mirror_cfg())


def test_auxiliary_never_constructs_mirrored_terrain_truth_or_mutates_rollout():
    obs = observations()
    original = obs.clone()
    obs["critic"].fill_(float("nan"))
    out, _ = augment_actor_only_symmetry(env=None, obs=obs)
    assert set(out.keys()) == {"actor"}
    assert out.batch_size == torch.Size([8])
    assert torch.equal(out["actor"][:4], original["actor"])
    assert torch.equal(mirror_public_observation(out["actor"][4:]), original["actor"])
    assert torch.isnan(obs["critic"]).all()
    assert torch.equal(obs["actor"], original["actor"])
    bad = TensorDict({"actor": torch.zeros(4, 69)}, batch_size=[4])
    with pytest.raises(ValueError, match="public65"):
        augment_actor_only_symmetry(env=None, obs=bad)


def test_auxiliary_is_invariant_to_paired_public_motor_units_and_has_gradient():
    torch.manual_seed(237)
    obs = observations()
    base = mapped_actor(obs, torch.ones(18))
    native_losses = []
    for scale in (torch.ones(18), torch.tensor([
            .08, .02, .03, .04, .05, .16,
            .07, .06, .03, .02, .03, .1,
            .07, .06, .03, .02, .03, .1])):
        actor = copy.deepcopy(base)
        actor.distribution.public_action_scale.copy_(scale)
        actor.mlp[-1].public_action_scale.copy_(scale)
        ex = extension()
        install_actor_only_native_mirror_units(SimpleNamespace(symmetry=ex))
        batch = SimpleNamespace(observations=obs.clone(), actions=torch.randn(4, 18))
        actions = batch.actions.clone()
        ex.augment_batch(batch, 4)
        assert set(batch.observations.keys()) == {"actor", "critic"}
        loss = ex.compute_loss(actor, batch, 4)
        native_losses.append(loss.detach())
        loss.backward()
        assert torch.isfinite(loss) and loss > 0
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                   for p in actor.parameters())
        assert torch.equal(batch.actions, actions)
    torch.testing.assert_close(native_losses[0], native_losses[1], rtol=1e-6, atol=1e-7)


def test_real_native_ppo_update_keeps_critic275_on_original_samples():
    torch.manual_seed(238)
    obs = observations()
    actor = mapped_actor(obs, torch.ones(18))
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1,
                     hidden_dims=(16,), obs_normalization=True)
    storage = RolloutStorage("rl", 4, 2, obs, (18,))
    alg = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
              symmetry_cfg=dict(env=None, **make_actor_only_mirror_cfg()))
    install_actor_only_native_mirror_units(alg)
    with torch.no_grad():
        for _ in range(2):
            alg.act(obs)
            next_obs = observations()
            alg.process_env_step(next_obs, torch.arange(4).float(),
                                 torch.zeros(4, dtype=torch.bool), {})
            obs = next_obs
        alg.compute_returns(obs)
    stored_critic = storage.observations["critic"].clone()
    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]
    critic_inputs, actor_groups = [], []
    hook = critic.register_forward_pre_hook(
        lambda model, args: critic_inputs.append(tuple(args[0]["critic"].shape)))
    actor_hook = actor.register_forward_pre_hook(
        lambda model, args: actor_groups.append(set(args[0].keys())))
    losses = alg.update()
    hook.remove()
    actor_hook.remove()
    assert critic_inputs == [(8, 275)]
    assert actor_groups == [{"actor", "critic"}, {"actor"}]
    assert torch.equal(storage.observations["critic"], stored_critic)
    assert losses["symmetry"] > 0 and all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert any(not torch.equal(old, new) for old, new in zip(actor_before, actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(critic_before, critic.parameters()))
    assert alg.optimizer.state


def test_entry_rejects_ppo_relabeling_and_repeated_unit_conversion():
    ex = extension()
    ex.use_data_augmentation = True
    with pytest.raises(ValueError, match="separate Actor-only"):
        install_actor_only_native_mirror_units(SimpleNamespace(symmetry=ex))
    ex.use_data_augmentation = False
    install_actor_only_native_mirror_units(SimpleNamespace(symmetry=ex))
    with pytest.raises(ValueError, match="already installed"):
        install_actor_only_native_mirror_units(SimpleNamespace(symmetry=ex))
