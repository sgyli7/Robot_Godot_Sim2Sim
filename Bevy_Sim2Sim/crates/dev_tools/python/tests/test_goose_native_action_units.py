"""Check physical Gaussian/mean units with the actual native RSL model."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from bevy_microduck_tools.goose.native_action_units import (
    initialize_native_action_units, initialize_native_tracking_action_units)


@pytest.fixture(params=[(65, "log", initialize_native_action_units),
                        (113, "scalar", initialize_native_tracking_action_units)])
def recipe(request):
    return request.param


@pytest.fixture
def contract():
    root = os.environ.get("GOOSE_DISCRETE_MJLAB_EXPERIMENT")
    if root is None:
        pytest.skip("Requires the received18-axis rigid Goose contract")
    return json.loads((Path(root)/"candidate/contract.json").read_text())


def fresh_runner(recipe):
    dim, sigma_type, _ = recipe
    torch.manual_seed(149)
    obs = TensorDict({"actor": torch.randn(32, dim)}, batch_size=[32])
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 18,
        hidden_dims=(16,), obs_normalization=True, distribution_cfg={
            "class_name": "GaussianDistribution", "std_type": sigma_type, "init_std": 1.})
    critic = torch.nn.Linear(69, 1)
    optimizer = torch.optim.Adam((*actor.parameters(), *critic.parameters()))
    return SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic,
                                               optimizer=optimizer)), obs


def test_native_mean_and_sigma_map_to_real_Goose_radians(contract, recipe):
    runner, obs = fresh_runner(recipe)
    actor = runner.alg.actor
    before = actor(obs).detach().clone()
    original_parameters = {k: v.detach().clone() for k, v in actor.named_parameters()}
    parameter_ids = [id(p) for p in actor.parameters()]
    normalizer = copy.deepcopy(actor.obs_normalizer.state_dict())
    critic = copy.deepcopy(runner.alg.critic.state_dict())
    joints = contract["joints"]
    scales = before.new_tensor([j["action_scale_rad"] for j in joints])
    expected_sigma_rad = before.new_tensor([
        .25*j["torque_peak_limit_nm"]/j["kp_nm_rad"] for j in joints])
    recipe[2](runner, contract)
    # Catch the actual mismatch: sigma1 must not mean1.1/1.8 Goose radians.
    actor.distribution.update(actor(obs))
    sigma_rad = actor.distribution.std.detach()[0]*scales
    torch.testing.assert_close(sigma_rad, expected_sigma_rad, rtol=2e-7, atol=1e-8)
    torch.testing.assert_close(actor(obs)*scales, before*expected_sigma_rad,
                               rtol=2e-6, atol=1e-8)
    assert not runner.alg.optimizer.state
    # Native Adam must keep its own coordinates and original parameter objects.
    assert [id(p) for p in actor.parameters()] == parameter_ids
    for key, value in original_parameters.items():
        assert torch.equal(dict(actor.named_parameters())[key], value)
    for key, value in normalizer.items():
        assert torch.equal(actor.obs_normalizer.state_dict()[key], value)
    for key, value in critic.items():
        assert torch.equal(runner.alg.critic.state_dict()[key], value)


def test_fresh_units_reject_trained_statistics_before_any_change(contract, recipe):
    runner, obs = fresh_runner(recipe)
    actor = runner.alg.actor
    actor.update_normalization(obs)
    saved = copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError, match="fresh"):
        recipe[2](runner, contract)
    for key, value in saved.items():
        assert torch.equal(actor.state_dict()[key], value)


def test_fresh_units_reject_existing_adam_state_before_any_change(contract, recipe):
    runner, _ = fresh_runner(recipe)
    actor = runner.alg.actor
    runner.alg.optimizer.zero_grad()
    sum(p.square().sum() for p in actor.parameters()).backward()
    runner.alg.optimizer.step()
    saved = copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError, match="fresh"):
        recipe[2](runner, contract)
    for key, value in saved.items():
        assert torch.equal(actor.state_dict()[key], value)


def test_actual_native_ppo_surrogate_adam_and_kl_survive_unit_mapping(contract, recipe):
    native, obs = fresh_runner(recipe)
    native.alg.actor.double()
    native.alg.critic.double()
    obs = obs.to(dtype=torch.float64)
    actor = copy.deepcopy(native.alg.actor)
    critic = copy.deepcopy(native.alg.critic)
    mapped = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic,
        optimizer=torch.optim.Adam((*actor.parameters(), *critic.parameters()))))
    recipe[2](mapped, contract)
    scale = actor(obs).new_tensor([.25*j['torque_peak_limit_nm'] /
        (j['kp_nm_rad']*j['action_scale_rad']) for j in contract['joints']])
    base = native.alg.actor
    base.distribution.update(base(obs))
    actor.distribution.update(actor(obs))
    noise = torch.linspace(-1., 1., 32*18, dtype=torch.float64).reshape(32, 18)
    action = (base.distribution.mean+noise*base.distribution.std).detach()
    public_action = action*scale
    old = [m.distribution.log_prob(a).detach() for m, a in
        ((base, action), (actor, public_action))]
    old_params = [tuple(p.detach().clone() for p in m.distribution.params)
        for m in (base, actor)]
    advantage = torch.linspace(-1., 1., 32, dtype=torch.float64)
    for _ in range(3):
        for runner, a, old_log in ((native, action, old[0]), (mapped, public_action, old[1])):
            m = runner.alg.actor
            m.distribution.update(m(obs))
            ratio = (m.distribution.log_prob(a)-old_log).exp()
            # Actual native clipped PPO surrogate and entropy formula.
            loss = torch.maximum(-advantage*ratio,
                -advantage*ratio.clamp(.8, 1.2)).mean()-.005*m.distribution.entropy.mean()
            runner.alg.optimizer.zero_grad()
            loss.backward()
            runner.alg.optimizer.step()
        torch.testing.assert_close(actor(obs), base(obs)*scale, rtol=2e-8, atol=2e-10)
        for m in (base, actor):
            m.distribution.update(m(obs))
        torch.testing.assert_close(actor.distribution.std, base.distribution.std*scale,
            rtol=2e-8, atol=2e-10)
        kl = [m.distribution.kl_divergence(p, m.distribution.params)
            for m, p in zip((base, actor), old_params)]
        torch.testing.assert_close(kl[0], kl[1], rtol=2e-8, atol=2e-10)


def test_native_sigma_bounds_remain_in_native_coordinates(contract, recipe):
    runner, obs = fresh_runner(recipe)
    actor = runner.alg.actor
    original_bounds = actor.distribution.std_range.copy()
    recipe[2](runner, contract)
    scale = actor(obs).new_tensor([.25*j['torque_peak_limit_nm'] /
        (j['kp_nm_rad']*j['action_scale_rad']) for j in contract['joints']])
    for value, bound in ((-100., original_bounds[0]), (100., original_bounds[1])):
        with torch.no_grad():
            if recipe[1] == "log":
                actor.distribution.log_std_param.fill_(value)
            else:
                actor.distribution.std_param.fill_(0. if value < 0. else bound*10.)
        actor.distribution.update(actor(obs))
        torch.testing.assert_close(actor.distribution.std, (scale*bound).expand(32, 18),
            rtol=2e-6, atol=1e-12)


def test_actual_native_onnx_export_contains_public_unit_map(contract, tmp_path, recipe):
    from onnx.reference import ReferenceEvaluator
    runner, obs = fresh_runner(recipe)
    actor = runner.alg.actor
    recipe[2](runner, contract)
    path = tmp_path/'own_public_actor.onnx'
    torch.onnx.export(actor.as_onnx(verbose=False), obs['actor'][:2], str(path),
        input_names=['obs'], output_names=['actions'], opset_version=18, dynamo=False,
        dynamic_axes={'obs': {0: 'batch'}, 'actions': {0: 'batch'}})
    actual = ReferenceEvaluator(str(path)).run(None, {'obs': obs['actor'].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), actor(obs), rtol=2e-6, atol=1e-7)
