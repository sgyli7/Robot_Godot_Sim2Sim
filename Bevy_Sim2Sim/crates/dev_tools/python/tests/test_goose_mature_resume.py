"""Exercise real mjlab/RSL checkpoint loading without a physics world."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from bevy_microduck_tools.goose import mature_training


def make_algorithm(schedule="adaptive"):
    obs = TensorDict({"actor": torch.randn(8, 65),
        "critic": torch.randn(8, 69)}, batch_size=[8])
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 18, hidden_dims=(16,),
        obs_normalization=True, distribution_cfg={
            "class_name": "GaussianDistribution", "std_type": "log",
            "init_std": .03})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(16,),
        obs_normalization=True)
    storage = RolloutStorage("rl", 8, 2, obs, (18,))
    algorithm = PPO(actor, critic, storage, schedule=schedule)
    return algorithm, obs


def saved_case(tmp_path, schedule="adaptive"):
    torch.manual_seed(109)
    original, obs = make_algorithm(schedule)
    # One real Adam step creates moments; it does not train a robot policy.
    original.optimizer.zero_grad()
    sum(p.square().sum() for p in (*original.actor.parameters(),
        *original.critic.parameters())).backward()
    original.optimizer.step()
    for model in (original.actor, original.critic):
        model.update_normalization(obs)
    original.learning_rate = 7.59375e-5
    original.optimizer.param_groups[0]["lr"] = original.learning_rate
    checkpoint = original.save()
    checkpoint.update(iter=512, infos={"env_state": {"common_step_counter": 12288}})
    path = tmp_path / "resume.pt"
    torch.save(checkpoint, path)
    resumed, _ = make_algorithm(schedule)
    # Use the actual upstream load method, including file I/O and env state.
    # Its constructor needs a physics environment only for rollout setup.
    runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
    runner.alg = resumed
    runner.current_learning_iteration = 0
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    return original, runner, obs, path


def load(runner, path, **kwargs):
    return mature_training.load_mature_checkpoint(
        runner, path, map_location="cpu", **kwargs)


def test_adaptive_resume_restores_scheduler_without_changing_policy_or_adam(tmp_path):
    original, runner, obs, path = saved_case(tmp_path)
    infos = load(runner, path)
    assert runner.alg.learning_rate == original.learning_rate
    assert runner.alg.optimizer.param_groups[0]["lr"] == original.learning_rate
    assert runner.current_learning_iteration == 512
    assert infos["env_state"]["common_step_counter"] == 12288
    assert runner.env.unwrapped.common_step_counter == 12288
    for before, after in ((original.actor, runner.alg.actor),
            (original.critic, runner.alg.critic)):
        torch.testing.assert_close(after(obs), before(obs), rtol=0, atol=0)
        for key, value in before.state_dict().items():
            assert torch.equal(after.state_dict()[key], value)
    for before, after in zip(original.optimizer.state.values(),
            runner.alg.optimizer.state.values(), strict=True):
        for key, value in before.items():
            assert torch.equal(after[key], value)


def test_actor_transfer_keeps_requested_fresh_optimizer_and_rate(tmp_path):
    _, runner, _, path = saved_case(tmp_path)
    load(runner, path, load_cfg={"actor": True, "critic": True,
        "optimizer": False, "iteration": True})
    assert runner.alg.learning_rate == .001
    assert runner.alg.optimizer.param_groups[0]["lr"] == .001
    assert not runner.alg.optimizer.state


def test_fixed_schedule_keeps_unused_config_scalar(tmp_path):
    original, runner, _, path = saved_case(tmp_path, "fixed")
    load(runner, path)
    assert runner.alg.learning_rate == .001
    assert runner.alg.optimizer.param_groups[0]["lr"] == original.learning_rate


def test_nonfinite_restored_adaptive_rate_is_rejected(tmp_path):
    _, runner, _, path = saved_case(tmp_path)
    saved = torch.load(path, weights_only=False)
    saved["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="finite positive"):
        load(runner, path)
