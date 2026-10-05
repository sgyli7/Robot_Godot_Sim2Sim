"""Goose get-up curriculum on public mjlab reset, rewards and RSL interfaces.

Frozen poses are episode births only. Native contact, actuators and one 20ms
step remain responsible for every subsequent state. A held upright pose is a
training metric; qualification also requires an independent locomotion handoff.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from mjlab.envs.mdp import action_rate_l2, is_terminated
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from .artifacts import sha256
from .foot_curriculum import rigid_collision_depth, rigid_contact_limit_exceeded
from .source_training import (
    completed_root_state, critic_motion_state, episode_timeout, make_standing_cfg,
    positive_power)
from .mjlab_env import actor_observation

REVISION = "goose_native_recovery_curriculum_v1"


def _read_library(path, digest, model_path, contract_path):
    if sha256(path) != digest:
        raise ValueError("Recovery birth library changed")
    manifest = json.loads(Path(path).with_suffix(".json").read_text())
    if (manifest["model_sha256"] != sha256(model_path)
            or manifest["contract_sha256"] != sha256(contract_path)):
        raise ValueError("Recovery poses must bind the actual frozen plant")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(path) as data:
        poses = data["qpos"].copy()
        nominal_height = float(data["nominal_com_height"])
    if (poses.shape != (13, model.nq) or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1., atol=1e-8)
            or nominal_height <= 0):
        raise ValueError("Thirteen finite declared recovery births are required")
    return poses, nominal_height, model


def reset_recovery_birth(env, env_ids, *, poses, joint_names, stage=0):
    """Use upstream entity writes solely at an explicit episode reset."""
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids)
    stage = int(stage)
    if stage not in (0, 1, 2):
        raise ValueError("Recovery stages are 30, 60 and 90 degrees")
    library = torch.as_tensor(poses, dtype=env.sim.data.qpos.dtype, device=env.device)
    # Include earlier stages and standing so get-up retains stable stand behavior.
    count = 4*(stage+1)
    selection = torch.randint(count+1, (len(ids),), device=env.device)
    selection = torch.where(selection == count, 12, selection)
    pose = library[selection].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    robot = env.scene["robot"]
    robot.write_root_link_pose_to_sim(pose[:, :7], env_ids=ids)
    robot.write_root_link_velocity_to_sim(pose.new_zeros((len(ids), 6)), env_ids=ids)
    joints, names = robot.find_joints(joint_names, preserve_order=True)
    if names != list(joint_names):
        raise ValueError("Recovery joint order changed")
    robot.write_joint_state_to_sim(pose[:, 7:], torch.zeros_like(pose[:, 7:]),
        joint_ids=joints, env_ids=ids)
    if not hasattr(env, "_goose_recovery_cases"):
        env._goose_recovery_cases = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._goose_recovery_stable_ticks = torch.zeros_like(env._goose_recovery_cases)
    env._goose_recovery_cases[ids] = selection
    env._goose_recovery_stable_ticks[ids] = 0


def recovery_state(env):
    """Refresh COM from completed qpos without integrating or solving contact."""
    with wp.ScopedDevice(env.sim.wp_device):
        mjw.kinematics(env.sim.wp_model, env.sim.wp_data)
        mjw.com_pos(env.sim.wp_model, env.sim.wp_data)
    _, rotation, velocity, angular = completed_root_state(env)
    term = env.action_manager.get_term("goose")
    height = env.sim.data.subtree_com[:, term.torso, 2]-env.scene.env_origins[:, 2]
    return rotation[:, 2, 2], height, velocity, angular


def recovery_progress(env, nominal_height):
    up, height, _, _ = recovery_state(env)
    return .5*(up+1.)+.5*(height/nominal_height).clamp(0, 1.)


def recovery_stable(env, nominal_height):
    up, height, velocity, angular = recovery_state(env)
    return ((up >= .95) & (height >= .85*nominal_height)
        & (velocity.norm(dim=-1) <= .04) & (angular.norm(dim=-1) <= .2))


def recovery_stand_reward(env, nominal_height):
    return recovery_stable(env, nominal_height).float()


def recovery_dense_stand_reward(env, nominal_height):
    """Smooth stand feedback while the stochastic policy learns stabilization.

    The hard150-Tick success metric is unchanged. A low, reclining body gets
    zero from this term; a nearly standing body receives graded feedback for
    reducing linear and angular motion instead of waiting for a rare binary
    stable sample during exploration.
    """
    up, height, velocity, angular = recovery_state(env)
    upright = ((up-.8)/.15).clamp(0., 1.)
    raised = ((height/nominal_height-.65)/.2).clamp(0., 1.)
    motion = torch.exp(-velocity.square().sum(-1)/.15**2
        - angular.square().sum(-1)/.6**2)
    return upright*raised*motion


def near_stand_target_cost(env, nominal_height):
    """Reduce unnecessary targets once the body has reached standing height.

    Folded low get-up postures remain unconstrained by this term. The
    original normalized neutral target is a training reference only.
    """
    up, height, velocity, angular = recovery_state(env)
    near = ((up >= .95) & (height >= .85*nominal_height)
        & (velocity.norm(dim=-1) <= .1) & (angular.norm(dim=-1) <= .4))
    actions = env.action_manager.get_term("goose").drive.actions
    return near*actions.square().mean(-1)


def recovery_stand_motion(env):
    up, _, velocity, angular = recovery_state(env)
    return (up >= .95)*(velocity.square().sum(-1)+.1*angular.square().sum(-1))


def record_recovery_hold(env, env_ids, *, nominal_height):
    stable = recovery_stable(env, nominal_height)
    ticks = env._goose_recovery_stable_ticks
    ticks.copy_(torch.where(stable, ticks+1, 0))


def held_recovery_ticks(env):
    return env._goose_recovery_stable_ticks.float()


def make_recovery_cfg(model_path, contract_path, library_path, library_sha256, *,
                      num_envs=128, stage=0, seed=73, episode_length_s=13.,
                      contact_failure_depth_m=None):
    """Separate get-up Actor; do not terminate ordinary body-ground contact."""
    poses, height, model = _read_library(
        Path(library_path), library_sha256, model_path, contract_path)
    cfg = make_standing_cfg(model_path, contract_path, num_envs=num_envs,
        seed=seed, episode_length_s=episode_length_s)
    joint_names = tuple(model.joint(i).name for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)
    cfg.events["recovery_birth"] = EventTermCfg(func=reset_recovery_birth,
        mode="reset", params={"poses": poses, "joint_names": joint_names, "stage": stage})
    cfg.events["recovery_hold"] = EventTermCfg(func=record_recovery_hold,
        mode="step", params={"nominal_height": height})
    cfg.rewards = {
        "recovery_progress": RewardTermCfg(func=recovery_progress,
            weight=3., params={"nominal_height": height}),
        "stable_stand": RewardTermCfg(func=recovery_stand_reward,
            weight=8., params={"nominal_height": height}),
        "stand_motion": RewardTermCfg(func=recovery_stand_motion, weight=-1.),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-.05),
        "power": RewardTermCfg(func=positive_power, weight=-.001),
    }
    cfg.terminations = {"time_out": TerminationTermCfg(
        func=episode_timeout, time_out=True)}
    if contact_failure_depth_m is not None:
        if not 0 < contact_failure_depth_m <= .05:
            raise ValueError("Finite contact-domain episode boundary must be at most50mm")
        cfg.terminations["contact_domain_failure"] = TerminationTermCfg(
            func=rigid_contact_limit_exceeded,
            params={"max_depth_m": contact_failure_depth_m})
        cfg.rewards["contact_domain_failure"] = RewardTermCfg(
            func=is_terminated, weight=-200.)
    cfg.metrics["recovery_stable_ticks"] = MetricsTermCfg(func=held_recovery_ticks)
    cfg.metrics["rigid_contact_depth_m"] = MetricsTermCfg(func=rigid_collision_depth)
    cfg.observations["critic"] = ObservationGroupCfg(terms={
        "base": ObservationTermCfg(func=actor_observation),
        "motion": ObservationTermCfg(func=critic_motion_state)},
        enable_corruption=False, concatenate_terms=True)
    return cfg


def reset_progressive_birth(env, env_ids, *, poses, joint_names, level=0):
    """Sample a named near-standing-to-fallen library at episode reset only.

    Rows are four directions per level, followed by the standing row. Earlier
    levels remain in the sampling distribution; qualification uses a separate
    fixed fallen library and cannot be earned by the easier training births.
    """
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    library = torch.as_tensor(poses, dtype=env.sim.data.qpos.dtype, device=env.device)
    if (len(library)-1) % 4 or not 0 <= level < (len(library)-1)//4:
        raise ValueError("Progressive recovery requires four directions per level")
    selection = torch.randint(4*(level+1), (len(ids),), device=env.device)
    # Most cases exercise the current frontier, retaining earlier cases and
    # standing. Curriculum decisions must use frontier cases separately.
    sample = torch.rand(len(ids), device=env.device)
    frontier = 4*level+torch.randint(4, (len(ids),), device=env.device)
    selection = torch.where(sample < .65, frontier, selection)
    selection = torch.where(sample >= .85, len(library)-1, selection)
    pose = library[selection].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    robot = env.scene["robot"]
    robot.write_root_link_pose_to_sim(pose[:, :7], env_ids=ids)
    robot.write_root_link_velocity_to_sim(pose.new_zeros((len(ids), 6)), env_ids=ids)
    joints, names = robot.find_joints(joint_names, preserve_order=True)
    if names != list(joint_names):
        raise ValueError("Progressive recovery joint order changed")
    robot.write_joint_state_to_sim(pose[:, 7:], torch.zeros_like(pose[:, 7:]),
        joint_ids=joints, env_ids=ids)
    if not hasattr(env, "_goose_recovery_cases"):
        env._goose_recovery_cases = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._goose_recovery_stable_ticks = torch.zeros_like(env._goose_recovery_cases)
    if not hasattr(env, "_goose_recovery_max_hold"):
        env._goose_recovery_max_hold = torch.zeros_like(env._goose_recovery_cases)
        env._goose_recovery_success_tick = torch.full_like(env._goose_recovery_cases, -1)
    env._goose_recovery_cases[ids] = selection
    env._goose_recovery_stable_ticks[ids] = 0
    env._goose_recovery_max_hold[ids] = 0
    env._goose_recovery_success_tick[ids] = -1


def record_progressive_hold(env, env_ids, *, nominal_height):
    record_recovery_hold(env, env_ids, nominal_height=nominal_height)
    ticks = env._goose_recovery_stable_ticks
    env._goose_recovery_max_hold.copy_(torch.maximum(env._goose_recovery_max_hold, ticks))
    success = (ticks >= 150) & (env._goose_recovery_success_tick < 0)
    env._goose_recovery_success_tick[success] = env.episode_length_buf[success]


def make_progressive_recovery_cfg(model_path, contract_path, parent_library,
        parent_sha256, progressive_library, progressive_sha256, *, level=0,
        dense_stability=False, anchored_stand=False, **kwargs):
    """Public event curriculum over a hash-bound library; same get-up task."""
    cfg = make_recovery_cfg(model_path, contract_path, parent_library,
        parent_sha256, **kwargs)
    if sha256(progressive_library) != progressive_sha256:
        raise ValueError("Progressive recovery library changed")
    metadata = json.loads(Path(progressive_library).with_suffix(".json").read_text())
    if (metadata["model_sha256"] != sha256(model_path)
            or metadata["contract_sha256"] != sha256(contract_path)):
        raise ValueError("Progressive library must bind the actual plant")
    with np.load(progressive_library) as data:
        poses = data["qpos"].copy()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (poses.ndim != 2 or poses.shape[1] != model.nq or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1., atol=1e-8)):
        raise ValueError("Progressive poses must be finite with unit quaternions")
    params = cfg.events["recovery_birth"].params
    cfg.events["recovery_birth"] = EventTermCfg(func=reset_progressive_birth,
        mode="reset", params={"poses": poses, "joint_names": params["joint_names"],
            "level": level})
    cfg.events["recovery_hold"].func = record_progressive_hold
    if dense_stability:
        cfg.rewards["stable_stand"].func = recovery_dense_stand_reward
    if anchored_stand:
        cfg.rewards["near_stand_target"] = RewardTermCfg(
            func=near_stand_target_cost, weight=-500.,
            params=cfg.rewards["stable_stand"].params.copy())
    return cfg
