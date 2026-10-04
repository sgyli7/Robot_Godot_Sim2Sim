"""Goose flat-floor curriculum using native mjlab and the upstream RSL runner.

The development environment retains its manual-reset and numerical-failure
guards. This thin RSL adapter resets ordinary episode endings only after their
completed control history and terminal observation have been recorded.
"""
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl.config import (
    RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg)
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from .batch_drive import _quat_matrix
from .mjlab_env import actor_observation, make_development_env_cfg

REVISION = "goose_flat_standing_ppo_v1"
VELOCITY_REVISION = "goose_flat_low_speed_ppo_v1"
VELOCITY_TRANSFER_REVISION = "goose_standing_to_velocity_transfer_v1"


def completed_root_state(env):
    """Read integrated qpos/qvel, avoiding upstream pre-forward derived data."""
    term = env.action_manager.get_term("goose")
    d = env.sim.data
    rotation = _quat_matrix(d.qpos[:, term.root_q+3:term.root_q+7])
    position = d.qpos[:, term.root_q:term.root_q+3] - env.scene.env_origins
    velocity = d.qvel[:, term.root_v:term.root_v+3]
    local_velocity = torch.einsum("bij,bj->bi", rotation.transpose(1, 2), velocity)
    angular = d.qvel[:, term.root_v+3:term.root_v+6]
    return position, rotation, local_velocity, angular


def upright_reward(env):
    return torch.exp(-8*(1-completed_root_state(env)[1][:, 2, 2]))


def velocity_reward(env, std=.2):
    command = env.action_manager.get_term("goose").commands
    velocity = completed_root_state(env)[2]
    return torch.exp(-(velocity[:, :2]-command[:, :2]).square().sum(-1)/std**2)


def yaw_reward(env, std=.4):
    command = env.action_manager.get_term("goose").commands
    return torch.exp(-(completed_root_state(env)[3][:, 2]-command[:, 2]).square()/std**2)


def height_reward(env):
    return torch.exp(-(completed_root_state(env)[0][:, 2]-.002).square()/.0025)


def standing_drift(env):
    position = completed_root_state(env)[0]
    command = env.action_manager.get_term("goose").commands
    standing = command.abs().sum(-1) < 1e-6
    return position[:, :2].square().sum(-1)*standing


def vertical_motion(env):
    return completed_root_state(env)[2][:, 2].square()


def rocking(env):
    return completed_root_state(env)[3][:, :2].square().sum(-1)


def action_change(env):
    term = env.action_manager.get_term("goose")
    return (term.raw_action-term.drive.actions).square().mean(-1)


def positive_power(env):
    term = env.action_manager.get_term("goose")
    return (term.drive.last_tau*env.sim.data.qvel[:, term.vids]).clamp(min=0).sum(-1)


def fallen(env):
    position, rotation, _, _ = completed_root_state(env)
    # The first curriculum ends before entering the known unqualified fallen
    # contact domain. Recovery will use its own termination rules and actor.
    return (rotation[:, 2, 2] < .90) | (position[:, 2] < -.07)


def episode_timeout(env):
    return env.episode_length_buf >= env.max_episode_length


def make_standing_cfg(model_path: Path, contract_path: Path, *, num_envs=16,
                      episode_length_s=8., seed=41):
    cfg = make_development_env_cfg(model_path, contract_path, num_envs=num_envs)
    cfg.seed = seed
    cfg.episode_length_s = episode_length_s
    cfg.rewards = {
        "upright": RewardTermCfg(func=upright_reward, weight=3.),
        "velocity": RewardTermCfg(func=velocity_reward, weight=1.),
        "yaw": RewardTermCfg(func=yaw_reward, weight=.5),
        "height": RewardTermCfg(func=height_reward, weight=1.),
        "standing_drift": RewardTermCfg(func=standing_drift, weight=-20.),
        "vertical_motion": RewardTermCfg(func=vertical_motion, weight=-.2),
        "rocking": RewardTermCfg(func=rocking, weight=-.1),
        "action_change": RewardTermCfg(func=action_change, weight=-.03),
        "power": RewardTermCfg(func=positive_power, weight=-.001),
    }
    cfg.terminations = {
        "fallen": TerminationTermCfg(func=fallen),
        "time_out": TerminationTermCfg(func=episode_timeout, time_out=True),
    }
    return cfg


def commanded_observation(env):
    """Publish the native next-Tick command in the existing three slots.

    mjlab evaluates reward before resampling commands. The action term thus
    retains the command that produced the completed action until this point.
    No root velocity or pose is written to help the robot follow a command.
    """
    command = env.command_manager.get_command("velocity")
    env.action_manager.get_term("goose").commands.copy_(command)
    return actor_observation(env)


def make_low_speed_cfg(model_path: Path, contract_path: Path, *, num_envs=16,
                       episode_length_s=12., seed=42):
    """First velocity curriculum, using mjlab's existing command generator."""
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
    cfg = make_standing_cfg(model_path, contract_path, num_envs=num_envs,
                            episode_length_s=episode_length_s, seed=seed)
    cfg.commands = {"velocity": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(2., 4.),
        rel_standing_envs=.35, heading_command=False, rel_heading_envs=0.,
        rel_world_envs=0., rel_forward_envs=0., init_velocity_prob=0.,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-.04, .06), lin_vel_y=(0., 0.),
            ang_vel_z=(-.15, .15)))}
    cfg.observations["actor"].terms["base"].func = commanded_observation
    cfg.rewards["velocity"] = RewardTermCfg(
        func=velocity_reward, weight=4., params={"std": .05})
    cfg.rewards["yaw"] = RewardTermCfg(
        func=yaw_reward, weight=1., params={"std": .1})
    # A stop at a new position must not reward returning to the spawn point.
    # Zero-command velocity tracking supplies the standing/stop objective.
    del cfg.rewards["standing_drift"]
    return cfg


def initialize_velocity_transfer(runner, command_cfg):
    """Initialize only previously unused goal inputs of a standing policy.

    The standing checkpoint saw zero commands exclusively. Its random input
    columns were never trained and zero variance magnifies a new command.
    Zero those columns and seed their moments from the declared command
    distribution. All existing state inputs, output weights, exploration and
    optimizer history remain intact; PPO is free to learn goal conditioning.
    This is a one-time transfer initialization, never a runtime action override.
    """
    nonstanding = 1-command_cfg.rel_standing_envs
    ranges = command_cfg.ranges
    intervals = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
    means = [nonstanding*(low+high)/2 for low, high in intervals]
    variances = [nonstanding*((high-low)**2/12+((low+high)/2)**2)-mean**2
                 for (low, high), mean in zip(intervals, means)]
    models = (runner.alg.actor, runner.alg.critic)
    for model in models:
        if model.mlp[0].in_features != 65:
            raise ValueError("Standing transfer requires the original 65-value input")
        normalizer = model.obs_normalizer
        if (normalizer._mean[..., 6:9].abs().any()
                or normalizer._var[..., 6:9].abs().any()):
            raise ValueError("Do not erase an already trained velocity policy")
        state = runner.alg.optimizer.state.get(model.mlp[0].weight, {})
        if any(state[key][:, 6:9].abs().any()
               for key in ("exp_avg", "exp_avg_sq") if key in state):
            raise ValueError("Standing checkpoint contains learned command optimizer history")
    with torch.no_grad():
        for model in models:
            model.mlp[0].weight[:, 6:9].zero_()
            normalizer = model.obs_normalizer
            normalizer._mean[..., 6:9] = normalizer._mean.new_tensor(means)
            normalizer._var[..., 6:9] = normalizer._var.new_tensor(variances)
            normalizer._std[..., 6:9] = normalizer._var[..., 6:9].sqrt()
    return VELOCITY_TRANSFER_REVISION


class GooseRslEnv(RslRlVecEnvWrapper):
    """Upstream RSL wrapper plus explicit, recorded finite episode resets."""

    def __init__(self, env, *, on_terminal=None):
        self.on_terminal = on_terminal
        self.real_integrations = 0
        self.terminal_episodes = 0
        super().__init__(env, clip_actions=1.)

    def get_observations(self):
        return super().get_observations().to(dtype=torch.float32)

    def reset(self):
        observations, extras = super().reset()
        return observations.to(dtype=torch.float32), extras

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        self.real_integrations += self.num_envs
        # TerminationManager buffers are cleared by reset: copy before reset.
        rewards, dones = rewards.clone(), dones.clone()
        extras = dict(extras)
        if "time_outs" in extras:
            extras["time_outs"] = extras["time_outs"].clone()
        ids = dones.nonzero(as_tuple=False).flatten()
        if ids.numel():
            if self.on_terminal is not None:
                self.on_terminal(self.env, ids, extras["time_outs"], observations)
            self.terminal_episodes += ids.numel()
            observations, reset_extras = self.env.reset(env_ids=ids)
            # Preserve upstream episode reward/metric logs from the reset.
            extras["log"] = reset_extras.get("log", {})
            from tensordict import TensorDict
            observations = TensorDict(observations, batch_size=[self.num_envs])
        return observations.to(dtype=torch.float32), rewards.float(), dones, extras


def make_runner_cfg(*, seed=41):
    cfg = RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=24, max_iterations=1500,
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
        actor=RslRlModelCfg(hidden_dims=(128, 128), obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": .03,
                              "std_type": "log"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(num_learning_epochs=4, num_mini_batches=4,
            learning_rate=3e-4, entropy_coef=.001),
        logger="tensorboard", save_interval=100, upload_model=False,
        experiment_name="goose_standing", run_name=REVISION, clip_actions=1.)
    return asdict(cfg)
