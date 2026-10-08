import copy
from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.game_commands import (
    GooseBalancedGameCommand, GooseBalancedGameCommandCfg,
    sample_microduck_game_commands, with_microduck_game_commands)
from bevy_microduck_tools.goose.native_velocity import GooseFreeMoveCommandCfg


def command_cfg():
    return GooseFreeMoveCommandCfg(entity_name="robot", resampling_time_range=(2., 4.),
        rel_standing_envs=.2, heading_command=False, rel_heading_envs=0.,
        rel_world_envs=0., rel_forward_envs=0., init_velocity_prob=0.,
        ranges=GooseFreeMoveCommandCfg.Ranges(lin_vel_x=(-.15, .3),
            lin_vel_y=(-.1, .1), ang_vel_z=(-.6, .6)))


def test_balanced_sampling_covers_each_axis_and_preserves_arc_proportions():
    torch.manual_seed(105)
    bucket, commands = sample_microduck_game_commands(9000, device="cpu", dtype=torch.float32)
    counts = torch.bincount(bucket, minlength=9)
    assert ((counts > 850) & (counts < 1150)).all()
    assert torch.count_nonzero(commands[bucket == 0]) == 0
    for index, axis, sign, nominal in ((1, 0, 1, .2), (2, 0, -1, .15),
            (3, 1, 1, .1), (4, 1, -1, .1), (5, 2, 1, .5), (6, 2, -1, .5)):
        values = commands[bucket == index]
        active = values[:, axis] * sign / nominal
        assert ((active >= .75) & (active <= 1.25)).all()
        assert torch.count_nonzero(values[:, [x for x in range(3) if x != axis]]) == 0
    for index, sign in ((7, 1), (8, -1)):
        values = commands[bucket == index]
        assert torch.equal(values[:, 1], torch.zeros(len(values)))
        torch.testing.assert_close(values[:, 2], values[:, 0] * (2. * sign))


def test_named_profile_preserves_old_configuration_and_native_scheduling():
    cfg = SimpleNamespace(commands={"velocity": command_cfg()}, curriculum={},
        rewards={"same": [1, 2]}, physics=[.02, 1], observations=[65, 69])
    saved = copy.deepcopy(cfg)
    result = with_microduck_game_commands(cfg)
    assert cfg == saved
    assert type(result.commands["velocity"]) is GooseBalancedGameCommandCfg
    assert result.commands["velocity"].resampling_time_range == (2., 4.)
    assert result.commands["velocity"].standing_ticks == 50
    for name in ("rewards", "physics", "observations", "curriculum"):
        assert getattr(result, name) == getattr(cfg, name)
    cfg.curriculum = {"existing_velocity_stage": object()}
    with pytest.raises(ValueError, match="silently bypass"):
        with_microduck_game_commands(cfg)


def test_reset_subset_retains_other_worlds_goals_and_never_writes_body_state():
    writes = []
    body = SimpleNamespace(data=SimpleNamespace(heading_w=torch.full((64,), 1.7)),
        write_root_link_velocity_b_to_sim=lambda *a, **kw: writes.append((a, kw)))
    env = SimpleNamespace(num_envs=64, device="cpu", step_dt=.02,
        scene={"robot": body}, episode_length_buf=torch.full((64,), 50), common_step_counter=0)
    cfg = GooseBalancedGameCommandCfg(**vars(command_cfg()))
    cfg.rel_standing_envs = 0.
    command = cfg.build(env)
    assert isinstance(command, GooseBalancedGameCommand)
    ids = torch.arange(64)
    torch.manual_seed(105)
    command._resample_command(ids)
    saved = command.command.clone()
    command._resample_command(torch.tensor([0, 1]))
    assert torch.equal(command.command[2:], saved[2:])
    actual = command.command.clone()
    goals = command.sampled_commands.clone()
    env.episode_length_buf[:2] = 49
    command._update_command(torch.tensor([0, 1]))
    assert torch.count_nonzero(command.command[:2]) == 0
    assert torch.equal(command.command[2:], actual[2:])
    assert torch.equal(command.sampled_commands, goals)
    env.episode_length_buf[:2] = 50
    command._update_command(torch.tensor([0, 1]))
    assert torch.equal(command.command, actual)
    assert not command.is_world_env.any() and not command.is_heading_env.any()
    assert writes == []
