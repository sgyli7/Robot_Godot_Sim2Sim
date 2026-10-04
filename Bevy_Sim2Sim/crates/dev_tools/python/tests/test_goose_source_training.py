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
    make_runner_cfg, make_standing_cfg, initialize_velocity_transfer,
    make_asymmetric_low_speed_cfg, make_forward_cfg, initialize_leg_exploration)


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


@pytest.fixture
def asymmetric_environment(tmp_path, monkeypatch):
    env = build_environment(tmp_path, monkeypatch, make_asymmetric_low_speed_cfg)
    yield env
    env.close()


@pytest.fixture
def forward_environment(tmp_path, monkeypatch):
    env = build_environment(tmp_path, monkeypatch, make_forward_cfg)
    yield env
    env.close()


def test_forward_course_keeps_native_physics_and_no_velocity_assist(forward_environment):
    env = forward_environment
    command = env.cfg.commands["velocity"]
    assert command.ranges.lin_vel_x == (.025, .055)
    assert command.ranges.lin_vel_y == (0., 0.)
    assert command.ranges.ang_vel_z == (0., 0.)
    assert command.rel_standing_envs == .1 and command.init_velocity_prob == 0
    assert env.cfg.decimation == 1 and env.cfg.sim.mujoco.timestep == .02


def test_exploration_preserves_mean_actor_and_native_resume(forward_environment):
    wrapper = GooseRslEnv(forward_environment)
    cfg = make_runner_cfg(entropy_coef=.01)
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
    runner = MjlabOnPolicyRunner(wrapper, cfg, log_dir=None, device="cpu")
    checkpoint = os.environ.get("GOOSE_MOTION_CHECKPOINT")
    if checkpoint:
        runner.load(checkpoint, strict=True, map_location="cpu")
    actor = runner.alg.actor
    inputs = wrapper.get_observations()
    before = actor(inputs).detach().clone()
    upper_std = actor.distribution.log_std_param[:6].detach().clone()
    previous_state = {key: value.clone() for key, value in
        runner.alg.optimizer.state.get(actor.distribution.log_std_param, {}).items()}
    initialize_leg_exploration(runner)
    torch.testing.assert_close(actor(inputs), before, atol=0, rtol=0)
    torch.testing.assert_close(actor.distribution.log_std_param[:6], upper_std, atol=0, rtol=0)
    torch.testing.assert_close(actor.distribution.log_std_param[6:].exp(), torch.full((12,), .12))
    state = runner.alg.optimizer.state.get(actor.distribution.log_std_param, {})
    if checkpoint:
        torch.testing.assert_close(state["step"], previous_state["step"], atol=0, rtol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[key][:6], previous_state[key][:6], atol=0, rtol=0)
        assert not state["exp_avg"][6:].any() and not state["exp_avg_sq"][6:].any()
    with pytest.raises(ValueError):
        initialize_leg_exploration(runner, std=float("nan"))
    runner.learn(1)
    assert wrapper.real_integrations == 8
    assert torch.isfinite(actor.distribution.log_std_param).all()


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


def test_privileged_motion_does_not_enter_actor(asymmetric_environment):
    env = asymmetric_environment
    wrapper = GooseRslEnv(env)
    term = env.action_manager.get_term("goose")
    command = env.command_manager.get_term("velocity")
    command.vel_command_b[:] = torch.tensor([.05,0.,0.])
    command.is_standing_env[:] = False
    # Diagnostic pair only: no training or reset velocity assistance.
    env.sim.data.qvel[1,term.root_v] = .05
    obs = env.observation_manager.compute(update_history=True)
    assert obs["actor"].shape == (2,65) and obs["critic"].shape == (2,69)
    torch.testing.assert_close(obs["actor"][0],obs["actor"][1],atol=0,rtol=0)
    torch.testing.assert_close(obs["critic"][:,:65],obs["actor"],atol=0,rtol=0)
    assert abs(float(obs["critic"][1,65]-obs["critic"][0,65])-.05)<1e-7
    assert wrapper.real_integrations == 0


def test_native_ppo_with_asymmetric_critic(asymmetric_environment):
    env = asymmetric_environment
    wrapper = GooseRslEnv(env)
    cfg = make_runner_cfg(critic_group="critic")
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"].update(num_learning_epochs=1,num_mini_batches=2)
    runner = MjlabOnPolicyRunner(wrapper,cfg,log_dir=None,device="cpu")
    checkpoint = os.environ.get("GOOSE_MOTION_CHECKPOINT") or os.environ.get("GOOSE_STANDING_CHECKPOINT")
    if checkpoint:
        # Supported upstream fine-tune: keep Actor, start value/optimizer fresh.
        runner.load(checkpoint,load_cfg={"actor":True,"critic":False,
            "optimizer":False,"iteration":False},strict=True,map_location="cpu")
        if not runner.alg.actor.obs_normalizer._var[...,6:9].any():
            initialize_velocity_transfer(runner,env.cfg.commands["velocity"],initialize_critic=False)
    before = {k:v.clone() for k,v in runner.alg.actor.named_parameters()}
    runner.learn(1)
    assert runner.alg.actor.obs_dim == 65 and runner.alg.critic.obs_dim == 69
    assert wrapper.real_integrations == 8
    assert any(not torch.equal(before[k],v) for k,v in runner.alg.actor.named_parameters())
    assert all(torch.isfinite(v).all() for v in runner.alg.critic.parameters())


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


def test_new_goal_transfer_preserves_standing_actions(low_speed_environment):
    checkpoint = os.environ.get("GOOSE_STANDING_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Actual frozen standing checkpoint required")
    env = low_speed_environment
    wrapper = GooseRslEnv(env)
    runner = MjlabOnPolicyRunner(wrapper, make_runner_cfg(), log_dir=None, device="cpu")
    runner.load(checkpoint, strict=True, map_location="cpu")
    runner.alg.eval_mode()
    observation = wrapper.get_observations()
    observation["actor"][:, 6:9] = 0
    with torch.inference_mode():
        expected = runner.alg.actor(observation).clone()
    before = {k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
    optimizer_before = {p:{k:v.clone() for k,v in values.items() if torch.is_tensor(v)}
                        for p, values in runner.alg.optimizer.state.items()}
    if not os.environ.get("GOOSE_TRANSFER_NEGATIVE_CONTROL"):
        initialize_velocity_transfer(runner, env.cfg.commands["velocity"])
    for command in ((0.,0.,0.),(.06,0.,0.),(-.04,0.,0.),(0.,0.,.15),(0.,0.,-.15)):
        observation["actor"][:, 6:9] = torch.tensor(command)
        with torch.inference_mode():
            actual = runner.alg.actor(observation)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)
    # Already learned inputs and output behavior must remain intact.
    for k, value in runner.alg.actor.state_dict().items():
        if k == "mlp.0.weight":
            torch.testing.assert_close(value[:, :6], before[k][:, :6], atol=0, rtol=0)
            torch.testing.assert_close(value[:, 9:], before[k][:, 9:], atol=0, rtol=0)
        elif k in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var"):
            torch.testing.assert_close(value[..., :6], before[k][..., :6], atol=0, rtol=0)
            torch.testing.assert_close(value[..., 9:], before[k][..., 9:], atol=0, rtol=0)
        else:
            torch.testing.assert_close(value, before[k], atol=0, rtol=0)
    for p, values in optimizer_before.items():
        for key, value in values.items():
            torch.testing.assert_close(runner.alg.optimizer.state[p][key], value, atol=0, rtol=0)
    assert wrapper.real_integrations == 0
