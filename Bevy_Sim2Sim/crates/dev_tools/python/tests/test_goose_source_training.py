"""Real upstream PPO and manual terminal-reset integration on frozen Goose."""
import os
from pathlib import Path

import pytest
import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

from bevy_microduck_tools.goose.mjlab_baseline import build_task_proxy_reference
from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv
from bevy_microduck_tools.goose.source_training import (
    GooseRslEnv, completed_root_state, fallen, make_low_speed_cfg,
    make_runner_cfg, make_standing_cfg)


def build_environment(tmp_path, monkeypatch, config_factory):
    monkeypatch.chdir(tmp_path)
    package = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not package:
        pytest.skip("Frozen external 004 package required")
    package = Path(package)
    model, contract = build_task_proxy_reference(
        package/"robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml",
        package/"robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        package, tmp_path/"candidate")
    return GooseDevelopmentEnv(config_factory(model, contract, num_envs=2,
        episode_length_s=.06), "cpu")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    env = build_environment(tmp_path, monkeypatch, make_standing_cfg)
    yield env
    env.close()


@pytest.fixture
def low_speed_environment(tmp_path, monkeypatch):
    env = build_environment(tmp_path, monkeypatch, make_low_speed_cfg)
    yield env
    env.close()


def test_terminal_reset_preserves_timeout_and_other_world(environment):
    seen = []
    def terminal(env, ids, timeouts, observation):
        seen.append((ids.clone(), timeouts.clone(), env.sim.data.time.clone(),
                     env.action_manager.get_term("goose").drive.completed_ticks.clone()))
    wrapper = GooseRslEnv(environment, on_terminal=terminal)
    wrapper.step(torch.zeros((2,18)))
    environment.episode_length_buf[0] = 2
    observations, _, dones, extras = wrapper.step(torch.zeros((2,18)))
    assert dones.tolist() == [1,0] and extras["time_outs"].tolist() == [True,False]
    assert seen[0][0].tolist() == [0] and seen[0][3].tolist() == [2,2]
    torch.testing.assert_close(environment.sim.data.time.clone(),torch.tensor([0.,.04]),atol=2e-6,rtol=0)
    assert environment.action_manager.get_term("goose").drive.completed_ticks.tolist() == [0,2]
    assert observations["actor"].dtype == torch.float32
    assert wrapper.real_integrations == 4


def test_real_upstream_optimizer_update_and_checkpoint(environment,tmp_path):
    wrapper = GooseRslEnv(environment)
    cfg = make_runner_cfg()
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"].update(num_learning_epochs=1,num_mini_batches=2)
    runner = MjlabOnPolicyRunner(wrapper,cfg,log_dir=None,device="cpu")
    before = {k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
    runner.learn(1)
    assert wrapper.real_integrations == 8
    assert any(not torch.equal(before[k],v) for k,v in runner.alg.actor.state_dict().items())
    assert runner.alg.optimizer.state and all(torch.isfinite(v).all()
        for v in runner.alg.actor.parameters())
    checkpoint = tmp_path/"actual_ppo.pt"
    runner.save(str(checkpoint))
    # A real reload uses a fresh model, whose normalizer buffers were not
    # allocated under the rollout's torch.inference_mode context.
    restored = MjlabOnPolicyRunner(wrapper,make_runner_cfg(),log_dir=None,device="cpu")
    restored.load(str(checkpoint),map_location="cpu")
    for key,value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(restored.alg.actor.state_dict()[key],value,atol=0,rtol=0)
    assert checkpoint.is_file()


def test_fall_termination_reads_integrated_quaternion(environment):
    wrapper = GooseRslEnv(environment)
    assert not fallen(environment).any()
    term = environment.action_manager.get_term("goose")
    # Leave derived xmat unchanged: the terminal check must read new qpos.
    environment.sim.data.qpos[0,term.root_q+3:term.root_q+7] = torch.tensor([.9238795,0.,.3826834,0.])
    assert fallen(environment).tolist() == [True,False]
    assert wrapper.real_integrations == 0


def test_commands_do_not_write_root_state(low_speed_environment):
    env = low_speed_environment
    wrapper = GooseRslEnv(env)
    before = [getattr(env.sim.data, name).clone() for name in ("qpos", "qvel", "time")]
    for _ in range(10):
        env.command_manager.get_term("velocity").time_left[:] = 0
        env.command_manager.compute(dt=.02)
    after = [getattr(env.sim.data, name).clone() for name in ("qpos", "qvel", "time")]
    for expected, actual in zip(before, after):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert wrapper.real_integrations == 0
    # Native get_observations reuses its last Tick cache. Explicitly publish
    # the manually resampled command for this zero-integral test.
    env.observation_manager.compute(update_history=True)
    obs = wrapper.get_observations()["actor"]
    assert obs.shape == (2, 65)
    torch.testing.assert_close(obs[:, 6:9], env.command_manager.get_command("velocity"))


def test_reward_uses_command_that_produced_action(low_speed_environment):
    env = low_speed_environment
    wrapper = GooseRslEnv(env)
    command_term = env.command_manager.get_term("velocity")
    action_term = env.action_manager.get_term("goose")
    previous = torch.tensor([[-.04, 0., -.15]]).expand(2, -1)
    action_term.commands.copy_(previous)
    # Resample AFTER this Tick's reward, before the next Actor observation.
    command_term.time_left[:] = 0
    command_term.cfg.rel_standing_envs = 0.
    command_term.cfg.ranges.lin_vel_x = (.05, .05)
    command_term.cfg.ranges.ang_vel_z = (.1, .1)
    obs, _, _, _ = wrapper.step(torch.zeros((2, 18)))
    actual = completed_root_state(env)[2][:, :2]
    expected = torch.exp(-(actual-previous[:, :2]).square().sum(-1)/.05**2)*4.
    idx = env.reward_manager.active_terms.index("velocity")
    torch.testing.assert_close(env.reward_manager._step_reward[:, idx], expected)
    torch.testing.assert_close(obs["actor"][:, 6:9], torch.tensor([[.05, 0., .1]]).expand(2, -1))
    assert wrapper.real_integrations == 2
