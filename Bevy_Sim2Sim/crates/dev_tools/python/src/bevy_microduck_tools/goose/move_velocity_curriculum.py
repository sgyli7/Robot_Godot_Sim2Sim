"""Opt-in carrying course using mjlab's native velocity curriculum.

The native uniform sampler reads the ranges updated by ``commands_vel``.
Template samplers must be replaced because they ignore those ranges. The
existing cold standing prefix and native command timer are preserved.
"""
import copy
from dataclasses import fields

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

from .native_velocity import GooseFreeMoveCommandCfg

REVISION = "goose_move_forward_carry_curriculum_v1"


def with_forward_carry_curriculum(cfg, completed_decisions=0):
    """Fresh course: .15, .25, .3m/s after 0, 128, 256 native PPO updates.

    One PPO update contains 24 control decisions, independent of world count.
    A resumed pilot supplies its actual completed decisions; the environment
    clock is never advanced or rewritten to simulate past experience.
    """
    if not isinstance(completed_decisions, int) or completed_decisions < 0:
        raise ValueError("Completed decisions must be an actual nonnegative count")
    if abs(cfg.sim.mujoco.timestep - .005) > 1e-12 or cfg.decimation != 4:
        raise ValueError("This course requires 50Hz decisions and 200Hz physics")
    if "carry_forward_velocity" in cfg.curriculum:
        raise ValueError("Carrying curriculum must not be added twice")
    result = copy.deepcopy(cfg)
    source = result.commands["velocity"]
    names = {field.name for field in fields(GooseFreeMoveCommandCfg)}
    command = GooseFreeMoveCommandCfg(**{
        name: getattr(source, name) for name in names})
    if (command.init_velocity_prob or command.heading_command
            or command.rel_heading_envs or command.rel_world_envs
            or command.rel_forward_envs):
        raise ValueError("Carrying commands cannot modify physical state")
    command.rel_standing_envs = .25
    stages = [dict(step=step * 24 - completed_decisions,
                   lin_vel_x=(speed * .95, speed * 1.05),
                   lin_vel_y=(0., 0.), ang_vel_z=(0., 0.))
              for step, speed in ((0, .15), (128, .25), (256, .3))]
    initial = next(stage for stage in reversed(stages) if stage["step"] <= 0)
    command.ranges.lin_vel_x = initial["lin_vel_x"]
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    result.commands["velocity"] = command
    result.curriculum["carry_forward_velocity"] = CurriculumTermCfg(
        func=commands_vel,
        params=dict(command_name="velocity", velocity_stages=stages))
    return result
