"""Reference rewards for the existing native mjlab/RSL locomotion task.

The reference is a training target only. The Actor still publishes all18
original position actions; no target, pose or torque is overwritten here.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg

from .mature_training import make_mature_velocity_cfg

REVISION = "goose_reference_guided_velocity_v1"


def reference_action(env, reference_path, reference_sha256):
    """Original phase/IMU feedback expressed in the existing action units."""
    if not hasattr(env, "_goose_action_reference"):
        path = Path(reference_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference_sha256:
            raise ValueError("Frozen locomotion reference identity changed")
        with np.load(path) as archive:
            joints = archive["joint_pos"].copy()
        if joints.shape != (128, 12) or not np.isfinite(joints).all():
            raise ValueError("Reference requires128 finite original leg poses")
        env._goose_action_reference = torch.as_tensor(joints,
            device=env.device, dtype=env.sim.data.qpos.dtype)
    term = env.action_manager.get_term("goose")
    drive = term.drive
    phase = (drive.phase-drive.phase_increment) % (2*torch.pi)
    index = phase*(128/(2*torch.pi))
    lower = index.floor().long() % 128
    fraction = (index % 1)[:, None]
    reference = env._goose_action_reference
    joint_target = (1-fraction)*reference[lower]+fraction*reference[(lower+1)%128]
    target = torch.zeros_like(drive.actions)
    target[:, 6:] = (joint_target-drive.neutral[6:])/drive.scale[6:]
    # Blend the cold first second. Phase/history belong to the real drive;
    # this expression only constructs a reward target after the real Tick.
    blend = ((drive.completed_ticks-1).clamp(min=0)/50).clamp(max=1)[:, None]
    target *= blend
    from .source_training import completed_root_state
    rotation = completed_root_state(env)[1]
    gravity_y = -rotation[:, 2, 1]
    for axis in (7, 13, 11, 17):
        target[:, axis] -= .5*gravity_y/drive.scale[axis]
    target.clamp_(-1, 1)
    moving = term.commands[:, :2].norm(dim=-1) > .001
    return target*moving[:, None]


def reference_error(env, **params):
    target = reference_action(env, **params)
    return (env.action_manager.get_term("goose").drive.actions-target).square().mean(-1)


def excessive_ground_depth(env):
    from .foot_curriculum import rigid_collision_depth
    return (rigid_collision_depth(env)-.003).clamp_min(0)


def make_guided_velocity_cfg(model_path, contract_path, reference_path,
                             reference_sha256, *, num_envs=1024, seed=109):
    """A narrow first forward course, followed by independent NN evaluation."""
    cfg = make_mature_velocity_cfg(model_path, contract_path,
        num_envs=num_envs, seed=seed)
    command = cfg.commands["velocity"]
    command.ranges.lin_vel_x = (.06, .06)
    command.rel_standing_envs = .2
    command.resampling_time_range = (6., 10.)
    cfg.rewards["velocity"].weight = 10.
    cfg.rewards["velocity"].params["std"] = .05
    # A stopped posture should not compete with an executed gait reference.
    for name in ("pose", "feet_air_time", "feet_clearance"):
        cfg.rewards.pop(name, None)
    cfg.rewards["reference"] = RewardTermCfg(func=reference_error,
        weight=-500., params={"reference_path": str(reference_path),
            "reference_sha256": reference_sha256})
    cfg.rewards["ground_depth"] = RewardTermCfg(
        func=excessive_ground_depth, weight=-20.)
    cfg.rewards["action_change"].weight = -.05
    return cfg
