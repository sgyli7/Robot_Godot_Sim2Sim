"""Goose goal-state recovery on the existing native mjlab/RSL lifecycle.

Task design follows Pollen MicroDuck's fixed-goal StandUp task (upstream
5946fd9): Gaussian/L1 height and pose, height-gated uprightness, a
multiplicative standing score, and light native motion regularizers. No MD
policy, normalization statistics, joint angles or motion clips are imported.
The tensor formulas below retain the corresponding upstream math; the
measurement is explicitly Goose's torso COM, not its virtual free-root origin.
This separate development task does not inherit either robot's qualifications.
"""
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs.mdp import action_rate_l2, joint_pos_limits
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity.mdp import rewards as native_rewards

from .artifacts import JOINT_ORDER
from .foot_curriculum import rigid_collision_depth, rigid_contact_limit_exceeded
from .mjlab_env import actor_observation, make_development_env_cfg
from .source_training import episode_timeout

REVISION = "goose_com_goal_recovery_v1"
SELF_SENSOR = "goose_recovery_self_contact"
MOMENTUM_SENSOR = "goose_recovery_angular_momentum"


def goal_scores(height, quaternion, joint_pos, joint_target, *, target_height):
    """Pinned StandUp goal formulas, with Goose names and length conversion.

    Widths .04/.015, gates .06/.115 and upward gate .125 are dimensionless
    ratios to MD's .115m task height. L1/velocity coefficients use the inverse
    length scale, preserving their relative units. Joint widths stay radians.
    Head control is an explicit fixed six-axis Goose HOME goal for this task.
    """
    scale = target_height / .115
    error = joint_pos - joint_target
    leg_error = error[:, 6:18]
    tilt = 2 * quaternion[:, 1:3].square().sum(-1)
    height_error = (height - target_height) / scale
    t = ((height / scale - .06) / (.115 - .06)).clamp(0, 1)
    return {
        "leg_pose": torch.exp(-(leg_error / .5).square()).mean(-1),
        "head_pose": torch.exp(-(error[:, :6] / .5).square()).mean(-1),
        "leg_pose_l1": -leg_error.abs().mean(-1),
        "height": torch.exp(-(height_error / .04).square()),
        "height_sharp": torch.exp(-(height_error / .015).square()),
        "height_l1": -height_error.abs(),
        "upright": 1 - tilt,
        "upright_sharp": torch.exp(-tilt / .3**2) * t.square() * (3 - 2*t),
        "standing_composite": (torch.exp(-(height_error / .04).square())
            * torch.exp(-tilt / .4**2)
            * torch.exp(-leg_error.square().mean(-1) / .4**2)),
    }


def _goal_state(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_goose_goal_joint_ids"):
        ids, names = robot.find_joints(JOINT_ORDER, preserve_order=True)
        if names != list(JOINT_ORDER):
            raise ValueError("Goal task requires the original Goose observable order")
        env._goose_goal_joint_ids = ids
    data = robot.data
    height = data.root_com_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height, data.root_link_quat_w,
        data.joint_pos[:, env._goose_goal_joint_ids],
        data.default_joint_pos[:, env._goose_goal_joint_ids],
        data.root_com_lin_vel_w)


def fixed_goal_reward(env, *, name, target_height):
    # RewardManager calls all these terms on the same upstream pre-forward
    # snapshot. Do not run FK/forward here or mix completed and stale points.
    state = _goal_state(env)
    return goal_scores(*state[:4], target_height=target_height)[name]


def upward_velocity(env, *, target_height):
    height, _, _, _, velocity = _goal_state(env)
    scale = target_height / .115
    return velocity[:, 2].clamp_min(0) / scale * (height < .125*scale)


def vertical_acceleration_cost(env, *, target_height):
    velocity = _goal_state(env)[4][:, 2]
    previous = getattr(env, "_goose_goal_previous_vz", velocity.detach())
    acceleration = (velocity-previous) / env.step_dt
    acceleration = torch.where(env.episode_length_buf <= 1, 0., acceleration)
    env._goose_goal_previous_vz = velocity.detach().clone()
    return -acceleration.abs() / (target_height / .115)


def goal_critic_motion(env):
    """Fresh private Critic69: Actor65 + true torso COM velocity3/height1."""
    height, _, _, _, velocity = _goal_state(env)
    rotation = env.scene["robot"].data.root_link_quat_w
    from .batch_drive import _quat_matrix
    local = torch.einsum("bij,bj->bi", _quat_matrix(rotation).transpose(1, 2), velocity)
    return torch.cat((local, height[:, None]), -1).float()


def reset_goal_birth(env, env_ids, *, poses, joint_names, frontier_step):
    """Standing/front fall curriculum at genuine cold resets only.

    A fixed standing goal is active throughout every episode, with no clip or
    progress clock in the Actor. Initial32 updates use20% front births; later
    resets use80%. Evaluation always starts from the original frozen front.
    """
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    front_probability = .2 if env.common_step_counter < frontier_step else .8
    selection = (torch.rand(len(ids), device=env.device) < front_probability).long()
    library = torch.as_tensor(poses, dtype=env.sim.data.qpos.dtype, device=env.device)
    pose = library[selection].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    robot = env.scene["robot"]
    joint_ids, names = robot.find_joints(joint_names, preserve_order=True)
    if names != list(joint_names):
        raise ValueError("Cold birth must bind all original tree coordinates")
    robot.write_root_link_pose_to_sim(pose[:, :7], env_ids=ids)
    robot.write_root_link_velocity_to_sim(pose.new_zeros((len(ids), 6)), env_ids=ids)
    robot.write_joint_state_to_sim(pose[:, 7:], torch.zeros_like(pose[:, 7:]),
        joint_ids=joint_ids, env_ids=ids)
    if not hasattr(env, "_goose_goal_birth"):
        env._goose_goal_birth = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._goose_goal_hold = torch.zeros_like(env._goose_goal_birth)
        env._goose_goal_max_hold = torch.zeros_like(env._goose_goal_birth)
    env._goose_goal_birth[ids] = selection
    env._goose_goal_hold[ids] = env._goose_goal_max_hold[ids] = 0


def record_goal_hold(env, env_ids, *, nominal_com_height):
    """Post-forward actual hold metric, distinct from reward-frame height."""
    del env_ids
    height, quaternion, _, _, velocity = _goal_state(env)
    del height
    term = env.action_manager.get_term("goose")
    # Whole-robot COM, including both passive linkage bodies. No live FK or
    # writes: these are the unique native forward's actual body COM positions.
    masses = torch.as_tensor(env.sim.mj_model.body_mass,
        device=env.device, dtype=env.sim.data.xipos.dtype)
    com_height = ((env.sim.data.xipos[..., 2]*masses).sum(-1) / masses.sum()
        - env.scene.env_origins[:, 2])
    up = 1-2*quaternion[:, 1:3].square().sum(-1)
    stable = ((up >= .95) & (com_height >= .85*nominal_com_height)
        & (velocity.norm(dim=-1) <= .04)
        & (env.sim.data.qvel[:, term.root_v+3:term.root_v+6].norm(dim=-1) <= .2))
    ticks = env._goose_goal_hold
    ticks.copy_(torch.where(stable, ticks+1, 0))
    env._goose_goal_max_hold.copy_(torch.maximum(env._goose_goal_max_hold, ticks))


def goal_hold_ticks(env):
    return env._goose_goal_hold.float()


def make_goal_recovery_cfg(model_path: Path, contract_path: Path, *, poses,
                           num_envs=256, seed=149, frontier_step=32*24):
    """Fixed own-robot goal, actual sensors, native reset/PPO/termination.

    The diagnostic escape boundary is50mm; independent qualification remains
    5mm. Body-ground contact and low initial height do not terminate recovery.
    No inherited Critic semantics: initialize a fresh COM-aware private value
    network, optionally import only a Goose-trained65-value Actor.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    poses = np.asarray(poses, dtype=float)
    if (poses.shape != (2, model.nq) or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=-1), 1.)):
        raise ValueError("Two declared finite stand/front cold births required")
    data = mujoco.MjData(model)
    data.qpos[:] = poses[0]
    mujoco.mj_forward(model, data)
    torso = model.body("torso").id
    height, whole_height = float(data.xipos[torso, 2]), float(data.subtree_com[torso, 2])
    if height <= 0 or whole_height <= 0:
        raise ValueError("Standing COM height must be measured on the actual model")
    cfg = make_development_env_cfg(model_path, contract_path, num_envs=num_envs)
    cfg.seed, cfg.episode_length_s = seed, 13.
    cfg.scene.sensors = (*cfg.scene.sensors,
        ContactSensorCfg(name=SELF_SENSOR,
            primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
            fields=("found",), reduce="none", num_slots=1),
        BuiltinSensorCfg(name=MOMENTUM_SENSOR, sensor_type="subtreeangmom",
            obj=ObjRef(type="body", name="torso", entity="robot")))
    weights = {"leg_pose": 2., "head_pose": .75, "leg_pose_l1": 1.25,
        "height": 1., "height_sharp": 1., "height_l1": 7.5,
        "upright": 1.5, "upright_sharp": 1.5, "standing_composite": 3.75}
    cfg.rewards = {name: RewardTermCfg(func=fixed_goal_reward, weight=weight,
        params={"name": name, "target_height": height}) for name, weight in weights.items()}
    cfg.rewards.update({
        "upward_velocity": RewardTermCfg(func=upward_velocity, weight=.75,
            params={"target_height": height}),
        "gentle_rise": RewardTermCfg(func=vertical_acceleration_cost, weight=.005,
            params={"target_height": height}),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-.1),
        "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER,
                preserve_order=True)}),
        "angular_velocity": RewardTermCfg(func=native_rewards.body_angular_velocity_penalty,
            weight=-.05, params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso",))}),
        "angular_momentum": RewardTermCfg(func=native_rewards.angular_momentum_penalty,
            weight=-.02, params={"sensor_name": "robot/"+MOMENTUM_SENSOR}),
        "self_collisions": RewardTermCfg(func=native_rewards.self_collision_cost,
            weight=-1., params={"sensor_name": SELF_SENSOR})})
    joints = tuple(model.joint(i).name for i in range(model.njnt)
        if int(model.jnt_type[i]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)))
    cfg.events["goal_birth"] = EventTermCfg(func=reset_goal_birth, mode="reset",
        params={"poses": poses, "joint_names": joints, "frontier_step": frontier_step})
    cfg.events["goal_hold"] = EventTermCfg(func=record_goal_hold, mode="step",
        params={"nominal_com_height": whole_height})
    cfg.observations["critic"] = ObservationGroupCfg(terms={
        "base": ObservationTermCfg(func=actor_observation),
        "motion": ObservationTermCfg(func=goal_critic_motion)},
        enable_corruption=False, concatenate_terms=True)
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=episode_timeout, time_out=True),
        "contact_domain_failure": TerminationTermCfg(func=rigid_contact_limit_exceeded,
            params={"max_depth_m": .05})}
    cfg.metrics["rigid_contact_depth_m"] = MetricsTermCfg(func=rigid_collision_depth)
    cfg.metrics["held_ticks"] = MetricsTermCfg(func=goal_hold_ticks)
    return cfg
