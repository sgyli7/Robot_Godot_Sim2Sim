"""Opt-in privileged state for a real free payload; Actor inputs stay intact.

The pose uses the robot root link frame, not its inertial COM frame. Relative
linear velocity is the derivative in that rotating frame, including its
angular transport term. This reads native entity/model state without moving
the object, updating filters or adding any physical support.
"""
import copy

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

REVISION = "goose_move_free_payload_critic_v1"


def payload_state_observation(env, robot_name="robot", payload_name="payload"):
    """Position3, quaternion4(wxyz), relative velocity3+3, actual mass1."""
    robot = env.scene[robot_name].data
    payload = env.scene[payload_name]
    data = payload.data
    offset = data.root_link_pos_w - robot.root_link_pos_w
    orientation = quat_mul(quat_conjugate(robot.root_link_quat_w),
                           data.root_link_quat_w)
    orientation = torch.where(orientation[:, :1] < 0, -orientation, orientation)
    relative_linear = (data.root_link_lin_vel_w - robot.root_link_lin_vel_w
                       - torch.cross(robot.root_link_ang_vel_w, offset, dim=-1))
    relative_angular = data.root_link_ang_vel_w - robot.root_link_ang_vel_w
    mass = env.sim.model.body_mass[:, payload.indexing.root_body_id]
    return torch.cat((
        quat_apply_inverse(robot.root_link_quat_w, offset), orientation,
        quat_apply_inverse(robot.root_link_quat_w, relative_linear),
        quat_apply_inverse(robot.root_link_quat_w, relative_angular),
        mass.reshape(env.num_envs, 1)), dim=-1).float()


def with_move_payload_critic(cfg, robot_name="robot", payload_name="payload"):
    """Copy a task and add Critic-only state; requires a fresh Critic."""
    if "actor" not in cfg.observations or "critic" not in cfg.observations:
        raise ValueError("Payload state requires separate Actor and Critic groups")
    if robot_name not in cfg.scene.entities or payload_name not in cfg.scene.entities:
        raise ValueError("Payload state requires real robot and payload entities")
    name = "free_payload_state"
    if name in cfg.observations["critic"].terms:
        raise ValueError("Payload Critic state must not be added twice")
    result = copy.deepcopy(cfg)
    result.observations["critic"].terms[name] = ObservationTermCfg(
        func=payload_state_observation,
        params=dict(robot_name=robot_name, payload_name=payload_name))
    return result
