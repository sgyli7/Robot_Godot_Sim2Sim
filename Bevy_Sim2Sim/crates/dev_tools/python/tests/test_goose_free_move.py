"""Direct three-axis commands must survive native sampling and refresh."""
from pathlib import Path
from types import SimpleNamespace
import copy

import torch
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

from bevy_microduck_tools.goose.native_velocity import (
    GooseFreeMoveCommand, GooseFreeMoveCommandCfg, make_free_move_cfg)


def free_cfg(monkeypatch):
    import bevy_microduck_tools.goose.native_velocity as module
    baseline = unitree_g1_flat_env_cfg()
    baseline.scene.sensors = ()
    calls = []
    def existing(*args, **kwargs):
        calls.append(kwargs)
        return copy.deepcopy(baseline)
    monkeypatch.setattr(module, "make_contact_bounded_sole_walk_run_cfg", existing)
    cfg = make_free_move_cfg(Path("robot.xml"), Path("contract.json"), num_envs=64)
    assert calls[0]["num_envs"] == 64
    return cfg, baseline


def test_free_commands_reuse_native_sampler_and_keep_physics_and_observation(monkeypatch):
    cfg, baseline = free_cfg(monkeypatch)
    command = cfg.commands["velocity"]
    assert type(command) is GooseFreeMoveCommandCfg
    assert not command.heading_command
    assert command.rel_heading_envs == command.rel_world_envs == command.rel_forward_envs == 0
    assert command.init_velocity_prob == 0
    assert cfg.curriculum == {}
    assert cfg.sim == baseline.sim
    assert cfg.observations == baseline.observations
    assert cfg.actions == baseline.actions
    assert cfg.terminations == baseline.terminations
    assert cfg.rewards["track_linear_velocity"].params["std"] == .15
    assert cfg.rewards["track_angular_velocity"].params["std"] == .3
    assert baseline.rewards["track_linear_velocity"].params["std"] != .15


def test_native_body_commands_combine_translation_yaw_and_standing_without_state_writes(monkeypatch):
    cfg, _ = free_cfg(monkeypatch)
    writes = []
    robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.full((64,), 2.)),
        write_root_link_velocity_b_to_sim=lambda *a, **kw: writes.append((a, kw)))
    env = SimpleNamespace(num_envs=64, device="cpu", scene={"robot": robot}, step_dt=.02,
        episode_length_buf=torch.full((64,), 50), common_step_counter=0)
    command = cfg.commands["velocity"].build(env)
    assert type(command) is GooseFreeMoveCommand
    assert isinstance(command, UniformVelocityCommand)
    torch.manual_seed(89)
    ids = torch.arange(64)
    command._resample_command(ids)
    before = command.command.clone()
    command._update_command(ids)
    moving = ~command.is_standing_env
    assert moving.any() and command.is_standing_env.any()
    # Native body-frame refresh must not turn player's yaw into heading hold.
    assert torch.equal(command.command[moving], before[moving])
    assert (command.command[moving, :].abs() > 0).all()
    assert command.command[moving, 1].min() < 0 < command.command[moving, 1].max()
    assert command.command[moving, 2].min() < 0 < command.command[moving, 2].max()
    assert torch.count_nonzero(command.command[command.is_standing_env]) == 0
    assert not command.is_heading_env.any() and not command.is_world_env.any()
    assert writes == []
    # Prefix masking must preserve sampled goals for the completed50th Tick.
    env.episode_length_buf.fill_(49)
    command._update_command(ids)
    assert torch.count_nonzero(command.command) == 0
    env.episode_length_buf.fill_(50)
    command._update_command(ids)
    assert torch.equal(command.command[moving], before[moving])
