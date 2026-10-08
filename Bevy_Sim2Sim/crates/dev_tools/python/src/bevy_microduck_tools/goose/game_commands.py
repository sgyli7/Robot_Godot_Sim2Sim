"""MicroDuck's balanced game-direction curriculum on Goose's reset contract.

The nine body-frame templates and uniform [.75, 1.25] amplitude come from
the frozen MicroDuck game workflow. Goose retains its actual standing
prefix, command history and native timing. Sampling changes goals only.
"""
import copy
from dataclasses import dataclass

import torch

from .native_velocity import GooseFreeMoveCommand, GooseFreeMoveCommandCfg

REVISION = "goose_microduck_balanced_game_commands_v1"


def sample_microduck_game_commands(count, *, device, dtype):
    """Sample the source nine directions without consulting physical state."""
    templates = torch.tensor([
        [0., 0., 0.], [.2, 0., 0.], [-.15, 0., 0.],
        [0., .1, 0.], [0., -.1, 0.], [0., 0., .5], [0., 0., -.5],
        [.15, 0., .3], [.15, 0., -.3],
    ], device=device, dtype=dtype)
    bucket = torch.randint(len(templates), (count,), device=device)
    amplitude = .75 + .5 * torch.rand(count, 1, device=device)
    return bucket, templates[bucket] * amplitude


class GooseBalancedGameCommand(GooseFreeMoveCommand):
    """Use the MD sampler while preserving Goose's genuine 50-Tick prefix."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        bucket, values = sample_microduck_game_commands(len(env_ids),
            device=self.device, dtype=self.vel_command_b.dtype)
        self.bucket_ids[env_ids] = bucket
        self.sampled_commands[env_ids] = values
        self._apply_sampled_commands(env_ids)


@dataclass(kw_only=True)
class GooseBalancedGameCommandCfg(GooseFreeMoveCommandCfg):
    def build(self, env):
        return GooseBalancedGameCommand(self, env)


def with_microduck_game_commands(cfg):
    """Opt into equal direction coverage; preserve rewards, physics and timing."""
    if type(cfg.commands["velocity"]) is not GooseFreeMoveCommandCfg:
        raise ValueError("Balanced game commands require the Goose free-move profile")
    if cfg.curriculum:
        raise ValueError("Balanced templates must not silently bypass a velocity curriculum")
    result = copy.deepcopy(cfg)
    command = GooseBalancedGameCommandCfg(**vars(result.commands["velocity"]))
    command.rel_standing_envs = 0.  # The source stop bucket owns its 1/9 share.
    command.ranges.lin_vel_x = (-.1875, .25)
    command.ranges.lin_vel_y = (-.125, .125)
    command.ranges.ang_vel_z = (-.625, .625)
    result.commands["velocity"] = command
    return result
