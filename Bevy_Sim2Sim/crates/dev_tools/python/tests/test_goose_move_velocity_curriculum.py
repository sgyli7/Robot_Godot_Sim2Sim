from types import SimpleNamespace

import pytest

from bevy_microduck_tools.goose.move_velocity_curriculum import (
    with_forward_carry_curriculum)
from bevy_microduck_tools.goose.native_velocity import GooseFreeMoveCommandCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


def configuration():
    command = GooseFreeMoveCommandCfg(
        entity_name="robot", resampling_time_range=(3., 4.),
        heading_command=False, rel_heading_envs=0., rel_world_envs=0.,
        rel_forward_envs=0., init_velocity_prob=0.,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(.4, .4), lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)))
    # A template sampler's extra field is deliberately not retained.
    command.templates = ((0., 0., 0.), (.4, 0., 0.))
    return SimpleNamespace(commands={"velocity": command}, curriculum={},
        sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=.005)), decimation=4,
        actor_contract=(65, 18), rewards={"tracking": ("original", 1.)})


def test_native_ranges_change_at_actual_decision_boundaries_without_clock_write():
    cfg = with_forward_carry_curriculum(configuration())
    term = SimpleNamespace(cfg=cfg.commands["velocity"])
    env = SimpleNamespace(common_step_counter=0,
        command_manager=SimpleNamespace(get_term=lambda name: term))
    course = cfg.curriculum["carry_forward_velocity"]
    assert course.func is commands_vel
    for tick, speed in ((0, .15), (3071, .15), (3072, .25),
                        (6143, .25), (6144, .3)):
        env.common_step_counter = tick
        course.func(env, None, **course.params)
        assert term.cfg.ranges.lin_vel_x == (speed * .95, speed * 1.05)
        assert env.common_step_counter == tick


def test_resume_uses_actual_decisions_and_preserves_physics_actor_and_prefix():
    source = configuration()
    cfg = with_forward_carry_curriculum(source, completed_decisions=16 * 24)
    assert source.commands["velocity"].ranges.lin_vel_x == (.4, .4)
    assert source.curriculum == {}
    assert cfg.actor_contract == source.actor_contract and cfg.rewards == source.rewards
    assert vars(cfg.sim.mujoco) == vars(source.sim.mujoco) and cfg.decimation == 4
    assert cfg.commands["velocity"].standing_ticks == 50
    assert cfg.commands["velocity"].resampling_time_range == (3., 4.)
    assert cfg.commands["velocity"].init_velocity_prob == 0
    assert cfg.commands["velocity"].rel_standing_envs == .25
    assert not hasattr(cfg.commands["velocity"], "templates")
    stages = cfg.curriculum["carry_forward_velocity"].params["velocity_stages"]
    assert [stage["step"] for stage in stages] == [-384, 2688, 5760]


def test_course_rejects_wrong_clock_reapplication_and_nonactual_offset():
    for value in (-1, 1.5):
        with pytest.raises(ValueError):
            with_forward_carry_curriculum(configuration(), value)
    cfg = configuration()
    cfg.decimation = 1
    with pytest.raises(ValueError):
        with_forward_carry_curriculum(cfg)
    with pytest.raises(ValueError):
        with_forward_carry_curriculum(with_forward_carry_curriculum(configuration()))
