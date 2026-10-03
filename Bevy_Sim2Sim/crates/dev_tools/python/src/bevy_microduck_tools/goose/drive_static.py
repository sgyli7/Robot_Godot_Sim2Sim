"""Check whether Goose position targets reproduce a supplied static motor effort.

This inverse is a development check, not an equilibrium, friction or contact solver. It
does not mutate the contract, prescribe a physical trajectory, or add leg
feedforward. The caller supplies the five nominal neck gravity efforts already
computed from encoders and IMU by the existing drive.
"""
from __future__ import annotations

import numpy as np

from .artifacts import DT, JOINT_ORDER


def _vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} requires {size} finite values")
    return result


def static_drive_targets(contract, q_rad, required_torque_nm,
                         nominal_neck_gravity_nm, *, incoming_target_rad=None):
    """Return bounded actions and the remaining static effort error per axis.

The scope is zero velocity, nominal drive strength and no command delay. A
mechanically feasible holding effort can still fail target limits or continuous
caps. ``feasible`` also requires that clipping the inferred action reproduces
the requested effort, within 1e-8 Nm arithmetic tolerance. Cold target slew is
reported separately and must not be confused with an immediately settled drive.
Passive joint friction and constraints can carry residual efforts; this check
alone cannot reject or qualify a complete physical stance.
    """
    if (contract.get("joint_order") != list(JOINT_ORDER)
            or any(contract.get(k) != DT for k in
                   ("physics_dt_s", "torque_dt_s", "policy_dt_s"))):
        raise ValueError("Static Goose drive check requires the 18-axis 50 Hz contract")
    joints = contract["joints"]
    if [joint["name"] for joint in joints] != list(JOINT_ORDER):
        raise ValueError("Static Goose joint entries must match the actuator order")
    q = _vector(q_rad, 18, "Static pose")
    required = _vector(required_torque_nm, 18, "Static holding torque")
    gravity = np.zeros(18)
    gravity[:5] = _vector(nominal_neck_gravity_nm, 5, "Nominal neck gravity")
    fields = {}
    for key in ("kp_nm_rad", "action_scale_rad", "q_neutral_rad",
                "continuous_design_limit_nm", "torque_peak_limit_nm", "speed_limit_rad_s"):
        fields[key] = _vector([joint[key] for joint in joints], 18, key)
    kp, scale, neutral = (fields[key] for key in
                          ("kp_nm_rad", "action_scale_rad", "q_neutral_rad"))
    continuous, peak, speed = (fields[key] for key in
                               ("continuous_design_limit_nm", "torque_peak_limit_nm", "speed_limit_rad_s"))
    limits = np.asarray([joint["range_rad"] for joint in joints], dtype=float)
    if (limits.shape != (18, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
            or np.any(kp <= 0) or np.any(scale <= 0) or np.any(speed <= 0)
            or np.any(continuous <= 0) or np.any(peak < continuous)):
        raise ValueError("Invalid static Goose drive coefficients or limits")
    incoming = neutral.copy() if incoming_target_rad is None else _vector(
        incoming_target_rad, 18, "Incoming persistent target")
    desired = q + (required - gravity) / kp
    unclipped_action = (desired - neutral) / scale
    action = np.clip(unclipped_action, -1.0, 1.0)
    bounded = np.clip(neutral + scale * action, limits[:, 0], limits[:, 1])
    settled = kp * (bounded - q) + gravity
    error = settled - required
    first_target = incoming + np.clip(bounded - incoming, -speed * DT, speed * DT)
    first_effort = np.clip(kp * (first_target - q) + gravity, -peak, peak)
    pose_valid = (q >= limits[:, 0] - 1e-12) & (q <= limits[:, 1] + 1e-12)
    continuous_valid = np.abs(required) <= continuous + 1e-12
    reproduced = np.abs(error) <= 1e-8
    feasible = pose_valid & continuous_valid & reproduced
    checks = [dict(axis=name, pose_within_limits=bool(pose_valid[i]),
                   within_continuous_cap=bool(continuous_valid[i]),
                   target_reproduces_static_effort=bool(reproduced[i]),
                   static_effort_error_nm=float(error[i]),
                   feasible=bool(feasible[i])) for i, name in enumerate(JOINT_ORDER)]
    return dict(schema="goose_static_drive_target_check_v1",
                scope="exact static motor effort, zero velocity, nominal strength, no delay; passive friction/equilibrium/contact/trajectory not checked",
                feasible=bool(feasible.all()), axis_checks=checks,
                q_rad=q.tolist(), required_torque_nm=required.tolist(),
                nominal_neck_gravity_nm=gravity[:5].tolist(),
                unclipped_action=unclipped_action.tolist(), action=action.tolist(),
                desired_target_rad=desired.tolist(), bounded_target_rad=bounded.tolist(),
                stationary_settled_torque_nm=settled.tolist(),
                incoming_target_rad=incoming.tolist(), first_update_target_rad=first_target.tolist(),
                stationary_first_update_torque_nm=first_effort.tolist(),
                minimum_slew_updates=np.ceil(np.abs(bounded - incoming) / (speed * DT)).astype(int).tolist(),
                physics_integrations=0, equilibrium_with_joint_friction_checked=False,
                physical_pose_qualified=False)
