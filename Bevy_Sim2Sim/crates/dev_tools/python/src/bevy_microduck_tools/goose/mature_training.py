"""Goose task profiles using the installed mjlab humanoid PPO recipe.

These profiles change learning conditions only. The frozen plant, motor
limits, 65-value Actor, native integration and manual reset boundary remain
owned by the existing Goose adapters.
"""
from dataclasses import asdict

import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from .foot_curriculum import make_rigid_motion_course_cfg, rigid_contact_limit_exceeded
from .source_training import commanded_observation, critic_motion_state

REVISION = "goose_mature_humanoid_ppo_v1"


def make_mature_runner_cfg(*, seed, recovery=False):
    """Reuse the pinned upstream humanoid optimizer, rollout and MLP recipe."""
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.seed = seed
    cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",)}
    # Log std permits positive finite per-axis initialization below. The mean
    # outputs still denote the original absolute position-target offsets.
    cfg.actor.distribution_cfg["std_type"] = "log"
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    cfg.experiment_name = "goose_recovery" if recovery else "goose_velocity"
    cfg.run_name = REVISION
    cfg.clip_actions = 1.
    return asdict(cfg)


def initialize_mature_actor(runner, *, recovery=False):
    """Fresh policy: exact zero mean and useful, declared joint exploration.

    This is initialization, not an online target override or an optimizer
    restart. It must not erase a trained policy or existing Adam moments.
    """
    if runner.alg.optimizer.state:
        raise ValueError("Mature initialization requires a fresh optimizer")
    actor = runner.alg.actor
    if actor.obs_dim != 65 or actor.distribution.log_std_param.shape != (18,):
        raise ValueError("Mature Goose initialization requires 65 to18")
    upper = [.15, .25, .3, .2, .15, .05] if recovery else [.05]*6
    leg = [.15, .25, .8, .8, .6, .2]
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.zero_()
        actor.distribution.log_std_param.copy_(
            actor.distribution.log_std_param.new_tensor(upper+leg+leg).log())


def make_mature_velocity_cfg(model_path, contract_path, *, num_envs=1024,
                             seed=83, forward_only=True):
    cfg = make_rigid_motion_course_cfg(model_path, contract_path,
        num_envs=num_envs, seed=seed, episode_length_s=12.)
    command = cfg.commands["velocity"]
    command.ranges.lin_vel_x = (.1, .3) if forward_only else (-.15, .3)
    command.ranges.lin_vel_y = (0., 0.) if forward_only else (-.1, .1)
    command.ranges.ang_vel_z = (0., 0.) if forward_only else (-.6, .6)
    command.rel_standing_envs = .2
    cfg.rewards["velocity"].params["std"] = .15
    cfg.rewards["yaw"].params["std"] = .3
    # Native humanoid tasks allow knee/hip bending more than roll/yaw.
    cfg.rewards["pose"].params["std_walking"] = {
        "neck_.*|head_.*|beak_.*": .35,
        ".*hip_yaw.*": .35, ".*hip_roll.*": .35,
        ".*hip_pitch.*": .5, ".*knee.*": .7,
        ".*ankle_pitch.*": .4, ".*ankle_roll.*": .35}
    cfg.observations["critic"] = ObservationGroupCfg(terms={
        "base": ObservationTermCfg(func=commanded_observation),
        "motion": ObservationTermCfg(func=critic_motion_state)},
        enable_corruption=False, concatenate_terms=True)
    cfg.terminations["contact_domain_failure"] = TerminationTermCfg(
        func=rigid_contact_limit_exceeded, params={"max_depth_m": .05})
    return cfg
