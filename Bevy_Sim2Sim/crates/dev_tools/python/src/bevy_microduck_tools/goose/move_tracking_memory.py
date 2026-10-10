"""Opt-in Critic access to the existing completed-motion reward memory.

The six EMA values and their reset flag are training state, not Actor inputs.
Enabling this changes the Critic topology and requires an explicit Actor-only
transfer with a fresh Critic/optimizer, followed by independent evaluation.
"""
import copy

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

from .motion_tracking import (
    FilteredAngularVelocityTracking, FilteredPlanarVelocityTracking)

REVISION = "goose_move_reward_memory_critic_v1"


def motion_tracking_memory_observation(env):
    """Read without advancing the filter; hide stale memory after partial reset."""
    state = getattr(env, "_goose_motion_tracking_state", None)
    if state is None:
        # ObservationManager can initialize before RewardManager owns the state.
        return torch.zeros((env.num_envs, 7), device=env.device)
    if state.mean.shape != (env.num_envs, 6) or state.ready.shape != (env.num_envs,):
        raise ValueError("Reward memory must contain six EMA values per world")
    ready = state.ready[:, None]
    mean = torch.where(ready, state.mean, torch.zeros_like(state.mean))
    return torch.cat((mean, ready.to(mean.dtype)), dim=-1).float()


def with_move_tracking_memory_critic(cfg):
    """Add the existing reward state to a copied Critic profile only."""
    if "actor" not in cfg.observations or "critic" not in cfg.observations:
        raise ValueError("Reward memory requires separate Actor and Critic groups")
    linear = cfg.rewards.get("track_linear_velocity")
    angular = cfg.rewards.get("track_angular_velocity")
    if (linear is None or angular is None
            or linear.func is not FilteredPlanarVelocityTracking
            or angular.func is not FilteredAngularVelocityTracking):
        raise ValueError("Reward memory requires the existing filtered tracking profile")
    for key in ("tracking_tau_s", "velocity_sensor_name"):
        if linear.params[key] != angular.params[key]:
            raise ValueError("Tracking rewards must share one memory")
    name = "motion_tracking_memory"
    if name in cfg.observations["critic"].terms:
        raise ValueError("Critic reward memory must not be added twice")
    result = copy.deepcopy(cfg)
    result.observations["critic"].terms[name] = ObservationTermCfg(
        func=motion_tracking_memory_observation)
    return result
