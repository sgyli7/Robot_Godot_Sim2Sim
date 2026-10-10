"""Opt-in current COM measurement for the existing command-stop formula.

Tracking keeps its shared MicroDuck EMA. This only replaces the stop term's
measurement, so alternating motion cannot cancel before that term sees it.
The original command gate, exponential formula, weight and tolerance remain.
"""
import copy

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from bevy_microduck_tools.goose.motion_tracking import FilteredCommandStop

REVISION = "goose_current_com_stop_measurement231_v1"


class _CurrentCOMVelocity:
    def __init__(self, sensor_name):
        self.sensor_name = sensor_name

    def update(self, env):
        data = env.scene["robot"].data
        velocity = quat_apply_inverse(
            data.root_link_quat_w, env.scene[self.sensor_name].data
        )
        return torch.cat((velocity, data.root_link_ang_vel_b), dim=-1)

    def reset(self, env_ids):
        # This measurement has no history and must not reset the tracking EMA.
        pass


class CurrentCOMCommandStop(FilteredCommandStop):
    """Reuse the existing stop evaluation with a separate current snapshot."""

    def __init__(self, cfg, env):
        if abs(env.step_dt - 0.02) > 1e-12:
            raise ValueError("Current COM stopping requires the 50 Hz control Tick")
        self.state = _CurrentCOMVelocity(cfg.params["velocity_sensor_name"])


def with_current_com_stop_measurement(cfg):
    """Return a separate candidate; default and other reward terms stay intact."""
    term = cfg.rewards.get("command_stop")
    if term is None or term.func is not FilteredCommandStop:
        raise ValueError("Requires the original filtered command-stop term")
    if term.params["velocity_sensor_name"] != "robot/whole_robot_com_velocity":
        raise ValueError("Requires the admitted whole-robot COM velocity sensor")
    result = copy.deepcopy(cfg)
    result.rewards["command_stop"].func = CurrentCOMCommandStop
    return result
