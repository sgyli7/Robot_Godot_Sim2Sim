"""Native mjlab humanoid velocity rewards on the frozen Goose plant.

The base profile retains all upstream reward functions and coefficients.
Goose adapts names, upper-body joints and the velocity measurement point.
The named low-speed comparison additionally tests a dimensional tolerance
mapping, without changing reward functions or weights.
"""
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity
from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand, UniformVelocityCommandCfg)

from .artifacts import JOINT_ORDER
from .foot_curriculum import (
    CONTACT_SENSOR, HEIGHT_SENSOR, SITE_NAMES, add_foot_sensors,
    rigid_collision_depth, rigid_contact_limit_exceeded)
from .mjlab_env import make_development_env_cfg
from .source_training import commanded_observation, critic_motion_state

REVISION = "goose_native_velocity_resume_v1"
FORWARD_SCALE_REVISION = "goose_native_forward_tolerance_scale_v1"
WALK_RUN_REVISION = "goose_native_walk_run_commands_v2"
SOLE_WALK_RUN_REVISION = "goose_sole_walk_run_task_v1"
SOLE_NATIVE_UNITS_REVISION = "goose_sole_walk_run_native_reward_units_v2"
SOLE_HEADING_REVISION = "goose_sole_native_heading_course_v1"
SOLE_CONTACT_DOMAIN_REVISION = "goose_sole_walk_run_completed_contact_domain_v1"


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


def make_native_walk_run_cfg(model_path, contract_path, *, course_start_step,
                             num_envs=256, seed=127,
                             development_initial_qpos=None):
    """Expose standing, .4m/s walking and .7m/s running with native commands.

    The fixed pilot schedule expands command exposure, not earned capability.
    Keep the original14 reward functions/weights/tolerances and native50Hz
    drive. UniformVelocityCommand resamples commands without writing actual
    robot state; directions and stops remain in the public three slots.

    Pass the checkpoint's env_state.common_step_counter when resuming, or0
    for fresh training. Mjlab restores that clock on load: an absolute4608
    threshold otherwise skips the walking stage of an already trained actor.
    """
    if (isinstance(course_start_step, bool)
            or not isinstance(course_start_step, int) or course_start_step < 0):
        raise ValueError("Course start must match the nonnegative restored step clock")
    cfg = make_native_velocity_cfg(model_path, contract_path, num_envs=num_envs,
        seed=seed, development_initial_qpos=development_initial_qpos)
    command = cfg.commands["velocity"]
    command.ranges.lin_vel_x = (0., .4)
    command.rel_standing_envs = .2
    cfg.curriculum = {"walk_run_commands": CurriculumTermCfg(
        func=commands_vel, params={"command_name": "velocity", "velocity_stages": [
            {"step": course_start_step, "lin_vel_x": (0., .4), "lin_vel_y": (-.1, .1),
             "ang_vel_z": (-.6, .6)},
            {"step": course_start_step+192*24, "lin_vel_x": (-.15, .7), "lin_vel_y": (-.1, .1),
             "ang_vel_z": (-.6, .6)},
        ]})}
    return cfg


class GooseSoleWalkRunCommand(UniformVelocityCommand):
    """Goose task commands, with real episode standing and explicit coverage.

    The native manager owns command timing. No root/joint/velocity writer or
    predecessor action table is used. Stage changes only resample commands;
    every episode's first 50 completed physics Ticks retain a zero command.
    """
    BUCKET_NAMES = ("stop", "walk", "backward", "left", "right", "yaw_left",
        "yaw_right", "walk_turn_left", "walk_turn_right", "run",
        "run_turn_left", "run_turn_right")

    def __init__(self, cfg, env):
        if (cfg.init_velocity_prob != 0. or cfg.heading_command
                or any((cfg.rel_heading_envs, cfg.rel_world_envs, cfg.rel_forward_envs))):
            raise ValueError("Goose task commands cannot write physical state")
        super().__init__(cfg, env)
        self.sampled_commands = torch.zeros_like(self.vel_command_b)
        self.bucket_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.warm_prefix_active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_stage = 0

    @property
    def course_stage(self):
        clock = self._env.common_step_counter - self.cfg.course_start_step
        return int(clock >= 32*24) + int(clock >= 64*24)

    def _templates(self):
        stage = self.course_stage
        vx, vy, yaw = (.06, .03, .3) if stage == 0 else (.4, .1, .6)
        backward = -.06 if stage == 0 else -.15
        turn = yaw/2
        values = [[0., 0., 0.], [vx, 0., 0.], [backward, 0., 0.],
            [0., vy, 0.], [0., -vy, 0.], [0., 0., yaw], [0., 0., -yaw],
            [vx, 0., turn], [vx, 0., -turn]]
        if stage == 2:
            values.extend([[.7, 0., 0.], [.7, 0., .3], [.7, 0., -.3]])
        return torch.tensor(values, device=self.device, dtype=self.vel_command_b.dtype)

    def _resample_command(self, env_ids):
        templates = self._templates()
        ids = torch.randint(1, len(templates), (len(env_ids),), device=self.device)
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
        ids[standing] = 0
        self.bucket_ids[env_ids] = ids
        self.sampled_commands[env_ids] = templates[ids]
        self._apply_sampled_commands(env_ids)

    def _apply_sampled_commands(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.warm_prefix_active[ids] = self._env.episode_length_buf[ids] < self.cfg.standing_ticks
        self.vel_command_b[ids] = torch.where(self.warm_prefix_active[ids, None],
            torch.zeros_like(self.sampled_commands[ids]), self.sampled_commands[ids])
        self.vel_command_w[ids] = self.vel_command_b[ids]
        self.is_standing_env[ids] = (self.bucket_ids[ids] == 0) | self.warm_prefix_active[ids]
        self.is_heading_env[ids] = False
        self.is_world_env[ids] = False
        self.is_forward_env[ids] = False

    def _update_command(self, env_ids=None):
        stage = self.course_stage
        if env_ids is None and stage != self._last_stage:
            self._resample(torch.arange(self.num_envs, device=self.device))
            self._last_stage = stage
        self._apply_sampled_commands(env_ids)


@dataclass(kw_only=True)
class GooseSoleWalkRunCommandCfg(UniformVelocityCommandCfg):
    course_start_step: int = 0
    standing_ticks: int = 50

    def build(self, env):
        return GooseSoleWalkRunCommand(self, env)


def reset_sole_landing_history(env, env_ids):
    """Reset native reward memory for reset worlds, without a physics write.

    The installed native landing reward has no reset method. Its peak-height
    history belongs to one episode and must not survive a genuine cold reset.
    The reward's original call and numeric parameters remain untouched.
    """
    landing = env.reward_manager.get_term_cfg("foot_swing_height").func
    landing.peak_heights[env_ids] = 0.


def make_sole_walk_run_task_cfg(model_path, contract_path, *, course_start_step,
                              num_envs=256, seed=127,
                              development_initial_qpos=None):
    """One Goose-owned foot, posture, command and termination task.

    Uses the existing native reward functions and Goose foot-course scales.
    It is a new learning task, not unchanged G1 reward inheritance or a claim
    that a scheduled speed command is already achieved.
    """
    if (isinstance(course_start_step, bool)
            or not isinstance(course_start_step, int) or course_start_step < 0):
        raise ValueError("Course start must match the restored nonnegative clock")
    cfg = make_native_velocity_cfg(model_path, contract_path, num_envs=num_envs,
        seed=seed, development_initial_qpos=development_initial_qpos)
    cfg.commands = {"velocity": GooseSoleWalkRunCommandCfg(
        course_start_step=course_start_step, standing_ticks=50,
        entity_name="robot", resampling_time_range=(3., 8.), rel_standing_envs=.2,
        heading_command=False, rel_heading_envs=0., rel_world_envs=0.,
        rel_forward_envs=0., init_velocity_prob=0.,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.15, .7),
            lin_vel_y=(-.1, .1), ang_vel_z=(-.6, .6)))}
    cfg.curriculum = {}
    cfg.events["sole_landing_history"] = EventTermCfg(
        func=reset_sole_landing_history, mode="reset")
    cfg.rewards["air_time"].weight = 1.
    cfg.rewards["air_time"].params.update(threshold_min=.05, threshold_max=.5,
        command_threshold=.01)
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params.update(target_height=.02, command_threshold=.01)
    moving_widths = {".*hip_.*|.*knee_.*|.*ankle_.*": .35,
        "neck_.*|head_.*|beak_.*": .15}
    cfg.rewards["pose"].params.update(std_standing={".*": .15},
        std_walking=moving_widths.copy(), std_running=moving_widths.copy(),
        walking_threshold=.01, running_threshold=.55)
    return cfg


def make_sole_walk_run_native_units_cfg(model_path, contract_path, *, course_start_step,
                                       num_envs=256, seed=127,
                                       development_initial_qpos=None):
    """Keep the own-foot task, restore the native action-rate input coordinates.

    This named profile delegates the installed loss after undoing the Actor's
    fixed output unit map. Other rewards, commands, genuine reset events,
    driver and physics are identical to the preserved v1 task. Fresh native
    Actors use initialize_native_action_units. For a retained plain public
    Gaussian, this reward mapping changes the v1 learning objective even
    when its output topology, sigma and Adam are restored. An unchanged v1
    resume uses make_sole_walk_run_task_cfg and checks the actual Actor with
    require_matching_action_rate_coordinates before the first physical Tick.
    Previous policy qualifications do not transfer to this learning profile.
    """
    from .mature_training import native_physical_action_std
    from .native_action_units import native_coordinate_action_rate_l2

    cfg = make_sole_walk_run_task_cfg(model_path, contract_path,
        course_start_step=course_start_step, num_envs=num_envs, seed=seed,
        development_initial_qpos=development_initial_qpos)
    term = cfg.rewards["action_rate_l2"]
    term.func = native_coordinate_action_rate_l2
    term.params = {"public_action_scale": native_physical_action_std(
        json.loads(Path(contract_path).read_text()))}
    return cfg


def make_contact_bounded_sole_walk_run_cfg(model_path, contract_path, *,
                                         course_start_step, num_envs=256,
                                         seed=127, development_initial_qpos=None):
    """Keep the own task within the independently evaluated floor domain.

    The retained v1 profile has a 50mm numerical escape guard. This named
    profile instead ends an episode at its first completed 5mm floor
    violation. The existing private FK query reads the integrated state;
    it does not refresh live solved forces, integrate again, project the
    robot or change collision geometry. RSL receives the terminal sample
    before the existing adapter resets that world.

    Rewards, commands, Actor inputs, actions and physics remain the v1
    profile. The different termination changes the learning task; old
    policy qualifications do not transfer to this profile.
    """
    cfg = make_sole_walk_run_task_cfg(model_path, contract_path,
        course_start_step=course_start_step, num_envs=num_envs, seed=seed,
        development_initial_qpos=development_initial_qpos)
    cfg.terminations["contact_domain_failure"] = TerminationTermCfg(
        func=rigid_contact_limit_exceeded, params={"max_depth_m": .005})
    return cfg


class GooseHeadingVelocityCommand(UniformVelocityCommand):
    """Native heading commands with the existing real standing prefix.

    Retain sampled velocities while the prefix emits zero commands. Heading
    conversion, command timers and resampling remain upstream-owned. This
    term only writes command buffers, never robot position or velocity.
    """

    def __init__(self, cfg, env):
        if (cfg.init_velocity_prob != 0. or cfg.rel_world_envs != 0.
                or cfg.rel_forward_envs != 0. or cfg.standing_ticks < 0):
            raise ValueError("Goose heading course requires state-free commands")
        super().__init__(cfg, env)
        self.sampled_commands = torch.zeros_like(self.vel_command_b)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.sampled_commands[env_ids] = self.vel_command_b[env_ids]

    def _update_command(self, env_ids=None):
        self.vel_command_b.copy_(self.sampled_commands)
        super()._update_command(env_ids)
        prefix = self._env.episode_length_buf < self.cfg.standing_ticks
        self.vel_command_b[prefix] = 0.


@dataclass(kw_only=True)
class GooseHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    standing_ticks: int = 50

    def build(self, env):
        return GooseHeadingVelocityCommand(self, env)


def make_sole_heading_velocity_cfg(model_path, contract_path, *, course_start_step,
                                   num_envs=256, seed=127,
                                   development_initial_qpos=None):
    """Named command-only continuation course; no baseline replacement.

    Use native heading goals in 80% of worlds and native signed twist goals
    in the remainder. Retain the own-foot task's fourteen reward functions,
    parameters, events, observation/action interfaces and actual physics.
    Keep the course within walking commands until route and signed turning
    improve independently; this profile grants no walk/run qualification.
    """
    cfg = make_sole_walk_run_native_units_cfg(model_path, contract_path,
        course_start_step=course_start_step, num_envs=num_envs, seed=seed,
        development_initial_qpos=development_initial_qpos)
    cfg.commands = {"velocity": GooseHeadingVelocityCommandCfg(
        entity_name="robot", standing_ticks=50,
        resampling_time_range=(3., 8.), rel_standing_envs=.2,
        heading_command=True, rel_heading_envs=.8, rel_world_envs=0.,
        rel_forward_envs=0., init_velocity_prob=0.,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.15, .4),
            lin_vel_y=(-.1, .1), ang_vel_z=(-.6, .6), heading=(-math.pi, math.pi)))}
    return cfg
