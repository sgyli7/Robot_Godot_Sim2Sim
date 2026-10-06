"""Native mjlab humanoid velocity rewards on the frozen Goose plant.

The base profile retains all upstream reward functions and coefficients.
Goose adapts names, upper-body joints and the velocity measurement point.
The named low-speed comparison additionally tests a dimensional tolerance
mapping, without changing reward functions or weights.
"""
import copy
from types import SimpleNamespace

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

from .artifacts import JOINT_ORDER
from .foot_curriculum import (
    CONTACT_SENSOR, HEIGHT_SENSOR, SITE_NAMES, add_foot_sensors,
    rigid_collision_depth, rigid_contact_limit_exceeded)
from .mjlab_env import make_development_env_cfg
from .source_training import commanded_observation, critic_motion_state

REVISION = "goose_native_velocity_resume_v1"
FORWARD_SCALE_REVISION = "goose_native_forward_tolerance_scale_v1"


def track_root_com_linear_velocity(env, **kwargs):
    """Use torso rigid-body COM, retaining upstream error math and snapshot.

    Goose's root link origin is near the ground. EntityData supplies the
    native rigid-body point transformation; this is not whole-robot COM and
    does not refresh or mutate the live physics snapshot.
    """
    name = kwargs.get("asset_cfg", SceneEntityCfg("robot")).name
    data = SimpleNamespace(
        root_link_lin_vel_b=env.scene[name].data.root_com_lin_vel_b)
    proxy = SimpleNamespace(scene={name: SimpleNamespace(data=data)},
        command_manager=env.command_manager)
    return track_linear_velocity(proxy, **kwargs)


def make_native_velocity_cfg(model_path, contract_path, *, num_envs=256,
                             seed=127, development_initial_qpos=None):
    cfg = make_development_env_cfg(model_path, contract_path,
        num_envs=num_envs, development_initial_qpos=development_initial_qpos)
    cfg.seed = seed
    cfg.episode_length_s = 12.
    cfg.commands = {"velocity": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(3., 8.),
        rel_standing_envs=.1, heading_command=False, rel_heading_envs=0.,
        rel_world_envs=0., rel_forward_envs=0., init_velocity_prob=0.,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-.15, .3), lin_vel_y=(-.1, .1), ang_vel_z=(-.6, .6)))}
    cfg.observations["actor"].terms["base"].func = commanded_observation
    cfg.observations["critic"] = ObservationGroupCfg(terms={
        "base": ObservationTermCfg(func=commanded_observation),
        "motion": ObservationTermCfg(func=critic_motion_state)},
        enable_corruption=False, concatenate_terms=True)
    cfg = add_foot_sensors(cfg, contract_path)
    cfg.scene.sensors = (*cfg.scene.sensors,
        BuiltinSensorCfg(name="root_angmom", sensor_type="subtreeangmom",
            obj=ObjRef(type="body", name="torso", entity="robot")),
        ContactSensorCfg(name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1, history_length=4))
    stock = unitree_g1_flat_env_cfg()
    cfg.rewards = copy.deepcopy(stock.rewards)
    for reward in cfg.rewards.values():
        if "command_name" in reward.params:
            reward.params["command_name"] = "velocity"
        if reward.params.get("sensor_name") == "feet_ground_contact":
            reward.params["sensor_name"] = CONTACT_SENSOR
        if reward.params.get("height_sensor_name") == "foot_height_scan":
            reward.params["height_sensor_name"] = HEIGHT_SENSOR
    for name in ("upright", "body_ang_vel"):
        cfg.rewards[name].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("torso",))
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"] = SceneEntityCfg(
            "robot", site_names=SITE_NAMES, preserve_order=True)
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=JOINT_ORDER, preserve_order=True)
    for regime in ("std_walking", "std_running"):
        source = stock.rewards["pose"].params[regime]
        adapted = {pattern: value for pattern, value in source.items()
            if any(part in pattern for part in ("hip_", "knee", "ankle_"))}
        adapted["neck_.*|head_.*|beak_.*"] = stock.rewards["pose"].params["std_standing"][".*"]
        cfg.rewards["pose"].params[regime] = adapted
    cfg.rewards["track_linear_velocity"].func = track_root_com_linear_velocity
    cfg.terminations = {name: copy.deepcopy(stock.terminations[name])
        for name in ("time_out", "fell_over")}
    cfg.terminations["contact_domain_failure"] = TerminationTermCfg(
        func=rigid_contact_limit_exceeded, params={"max_depth_m": .05})
    cfg.metrics["rigid_contact_depth_m"] = MetricsTermCfg(func=rigid_collision_depth)
    return cfg


def make_native_forward_scale_cfg(model_path, contract_path, *, num_envs=256,
                                  seed=127, development_initial_qpos=None,
                                  scale_linear_tolerance=False):
    """Paired first-forward course; only linear tolerance differs between arms.

    Preserve the upstream dimensionless ratio std / characteristic command
    speed when mapping its initial 1m/s domain to the fixed .06m/s course.
    This is an explicit hypothesis for Goose, not a source qualification.
    Both arms use the original guided course's standing fraction and command
    resampling, without guidance, new weights or a modified control contract.
    """
    cfg = make_native_velocity_cfg(model_path, contract_path, num_envs=num_envs,
        seed=seed, development_initial_qpos=development_initial_qpos)
    command = cfg.commands["velocity"]
    command.ranges.lin_vel_x = (.06, .06)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    command.rel_standing_envs = .2
    command.resampling_time_range = (6., 10.)
    if scale_linear_tolerance:
        stock = unitree_g1_flat_env_cfg()
        source_speed = max(abs(value) for value in
            stock.commands["twist"].ranges.lin_vel_x)
        if source_speed != 1.:
            raise ValueError("Frozen upstream characteristic command speed changed")
        cfg.rewards["track_linear_velocity"].params["std"] *= .06/source_speed
    return cfg
