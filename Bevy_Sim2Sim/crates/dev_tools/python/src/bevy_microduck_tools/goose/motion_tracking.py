"""Opt-in transfer of MicroDuck's completed-motion reward filter.

The MicroDuck workflow uses a shared 0.2 s exponential moving average,
advanced once per completed 50 Hz Tick and initialized after each reset.
This adaptation reads Goose's whole-robot COM sensor. It changes rewards
only; raw physics, commands, observations and stability penalties remain
unchanged. It does not establish qualification of either robot's policy.
"""
import copy
import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

REVISION = "goose_microduck_completed_motion_tracking_v1"


class CompletedMotionState:
    """Share the MicroDuck filter between the two tracking rewards."""

    def __init__(self, env, tau, sensor_name):
        if not math.isfinite(tau) or tau <= 0 or abs(env.step_dt - .02) > 1e-12:
            raise ValueError("Motion tracking requires positive tau and 50 Hz")
        self.tau = tau
        self.sensor_name = sensor_name
        self.alpha = 1. - math.exp(-env.step_dt / tau)
        self.mean = torch.zeros(env.num_envs, 6, device=env.device)
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_step = -1

    def update(self, env):
        if self.last_step != env.common_step_counter:
            data = env.scene["robot"].data
            velocity = quat_apply_inverse(data.root_link_quat_w,
                env.scene[self.sensor_name].data)
            actual = torch.cat((velocity, data.root_link_ang_vel_b), -1)
            value = (1. - self.alpha) * self.mean + self.alpha * actual
            self.mean.copy_(torch.where(self.ready[:, None], value, actual))
            self.ready[:] = True
            self.last_step = env.common_step_counter
        return self.mean

    def reset(self, env_ids):
        self.ready[env_ids] = False


class _FilteredTracking:
    def __init__(self, cfg, env):
        tau = cfg.params["tracking_tau_s"]
        sensor = cfg.params["velocity_sensor_name"]
        if not hasattr(env, "_goose_motion_tracking_state"):
            env._goose_motion_tracking_state = CompletedMotionState(env, tau, sensor)
        self.state = env._goose_motion_tracking_state
        if self.state.tau != tau or self.state.sensor_name != sensor:
            raise ValueError("Tracking rewards must share one measurement filter")

    def reset(self, env_ids):
        self.state.reset(env_ids)


class FilteredPlanarVelocityTracking(_FilteredTracking):
    def __call__(self, env, command_name, std, tracking_tau_s, velocity_sensor_name):
        del tracking_tau_s, velocity_sensor_name
        desired = env.command_manager.get_command(command_name)[:, :2]
        actual = self.state.update(env)[:, :2]
        error = (actual - desired).square().sum(-1)
        return torch.exp(-error / std ** 2)


class FilteredAngularVelocityTracking(_FilteredTracking):
    def __call__(self, env, command_name, std, tracking_tau_s, velocity_sensor_name):
        del tracking_tau_s, velocity_sensor_name
        desired = env.command_manager.get_command(command_name)[:, 2]
        actual = self.state.update(env)[:, 3:6]
        error = (actual[:, 2] - desired).square() + actual[:, :2].square().sum(-1)
        return torch.exp(-error / std ** 2)


class FilteredCommandStop(_FilteredTracking):
    """Reuse MicroDuck's zero-three-axis-command stopping reward."""

    def __call__(self, env, command_name, vel_std, tracking_tau_s, velocity_sensor_name):
        del tracking_tau_s, velocity_sensor_name
        command = env.command_manager.get_command(command_name)
        enabled = command[:, :3].square().sum(-1) < 1e-10
        velocity = self.state.update(env)[:, :2]
        return enabled * torch.exp(-velocity.square().sum(-1) / vel_std ** 2)


def with_microduck_motion_tracking(cfg, *, tracking_tau_s=.2):
    """Return a named reward-only comparison; preserve the native profile."""
    result = copy.deepcopy(cfg)
    linear = result.rewards["track_linear_velocity"]
    sensor_name = linear.params["sensor_name"]
    changes = (("track_linear_velocity", FilteredPlanarVelocityTracking),
        ("track_angular_velocity", FilteredAngularVelocityTracking))
    for name, function in changes:
        term = result.rewards[name]
        term.func = function
        term.params = {"command_name": term.params["command_name"],
            "std": term.params["std"], "tracking_tau_s": tracking_tau_s,
            "velocity_sensor_name": sensor_name}
    return result


def with_microduck_stop_precision(cfg):
    """Opt into the MD precision stage's weight2, .035 m/s stop term.

    This requires the named completed-motion profile so all three rewards
    share its single filter. It does not smooth or alter player commands.
    """
    from mjlab.managers.reward_manager import RewardTermCfg

    linear = cfg.rewards["track_linear_velocity"]
    angular = cfg.rewards["track_angular_velocity"]
    if (linear.func is not FilteredPlanarVelocityTracking
            or angular.func is not FilteredAngularVelocityTracking):
        raise ValueError("Stop precision requires MicroDuck motion tracking")
    for name in ("command_name", "tracking_tau_s", "velocity_sensor_name"):
        if linear.params[name] != angular.params[name]:
            raise ValueError("Stopping and tracking must share one motion filter")
    if "command_stop" in cfg.rewards:
        raise ValueError("A command stopping reward is already configured")
    result = copy.deepcopy(cfg)
    result.rewards["command_stop"] = RewardTermCfg(func=FilteredCommandStop,
        weight=2., params={"command_name": linear.params["command_name"],
            "vel_std": .035, "tracking_tau_s": linear.params["tracking_tau_s"],
            "velocity_sensor_name": linear.params["velocity_sensor_name"]})
    return result
