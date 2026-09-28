"""Compare one frozen robot-only MuJoCo solve with raw Rapier force evidence.

This is a diagnostic, not a BAM external-load admission test. Run with a Python
environment that contains the source MuJoCo package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco


def checked_bytes(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mjb", type=Path)
    parser.add_argument("mjb_sha256")
    parser.add_argument("qpos", type=Path)
    parser.add_argument("qpos_sha256")
    parser.add_argument("rapier_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--qvel", type=Path)
    parser.add_argument("--qvel-sha256")
    args = parser.parse_args()

    checked_bytes(args.mjb, args.mjb_sha256)
    qpos = json.loads(checked_bytes(args.qpos, args.qpos_sha256))
    rapier_bytes = args.rapier_report.read_bytes()
    rapier = json.loads(rapier_bytes)
    if rapier["scope"] != "robot_only_rapier_one_step_raw_force_diagnostic":
        raise ValueError("unexpected Rapier report scope")
    if rapier["qpos_file_sha256"] != args.qpos_sha256:
        raise ValueError("Rapier and MuJoCo input qpos differ")
    if (args.qvel is None) != (args.qvel_sha256 is None):
        raise ValueError("qvel path and SHA256 must be provided together")
    if args.qvel is not None:
        qvel = json.loads(checked_bytes(args.qvel, args.qvel_sha256))
        if rapier["qvel_file_sha256"] != args.qvel_sha256:
            raise ValueError("Rapier and MuJoCo input qvel differ")
    else:
        qvel = None
        if rapier["qvel_file_sha256"] is not None:
            raise ValueError("Rapier used a nonzero qvel input")
    if rapier["physics_integrations"] != 1 or rapier["policy_inferences"] != 0:
        raise ValueError("Rapier report is not one uncontrolled physical step")

    model = mujoco.MjModel.from_binary_path(str(args.mjb))
    if len(qpos) != model.nq or len(rapier["rows"]) != model.nu:
        raise ValueError("source state or driven channel dimension differs")
    if qvel is not None and len(qvel) != model.nv:
        raise ValueError("source velocity dimension differs")
    if not math.isclose(model.opt.timestep, rapier["actual_step_dt_seconds"], abs_tol=1e-9):
        raise ValueError("solver time steps differ")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel if qvel is not None else 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    before_contacts = int(data.ncon)
    before_bias = data.qfrc_bias.copy()
    before_constraint = data.qfrc_constraint.copy()
    before_passive = data.qfrc_passive.copy()
    before_actuator = data.qfrc_actuator.copy()
    before_qpos = data.qpos.copy()
    before_qvel = data.qvel.copy()

    # MuJoCo's force arrays after mj_step belong to that step's solve; a new
    # mj_forward below intentionally gives a separate post-state comparison.
    mujoco.mj_step(model, data)
    step_contacts = int(data.ncon)
    post_qpos = data.qpos.copy()
    post_qvel = data.qvel.copy()
    mujoco.mj_forward(model, data)
    post_contacts = int(data.ncon)
    post_bias = data.qfrc_bias.copy()
    post_constraint = data.qfrc_constraint.copy()

    rows = []
    force_errors = []
    initial_position_errors = []
    initial_velocity_errors = []
    position_errors = []
    velocity_errors = []
    all_frictionless = True
    for index, native in enumerate(rapier["rows"]):
        source_joint = int(model.actuator_trnid[index, 0])
        source_dof = int(model.jnt_dofadr[source_joint])
        source_qpos = int(model.jnt_qposadr[source_joint])
        if (native["source_joint"], native["source_dof"]) != (source_joint, source_dof):
            raise ValueError(f"driven source identity differs at channel {index}")
        source_external = float(-before_bias[source_dof] + before_constraint[source_dof])
        source_external_post = float(-post_bias[source_dof] + post_constraint[source_dof])
        native_external = float(native["candidate_residual_nm"])
        force_error = abs(native_external - source_external)
        initial_position_error = abs(float(native["initial_position"]) - float(before_qpos[source_qpos]))
        initial_velocity_error = abs(float(native["initial_velocity"]) - float(before_qvel[source_dof]))
        position_error = abs(float(native["post_position"]) - float(post_qpos[source_qpos]))
        velocity_error = abs(float(native["post_velocity"]) - float(post_qvel[source_dof]))
        frictionloss = float(model.dof_frictionloss[source_dof])
        passive = float(before_passive[source_dof])
        all_frictionless &= frictionloss == 0.0 and passive == 0.0
        force_errors.append(force_error)
        initial_position_errors.append(initial_position_error)
        initial_velocity_errors.append(initial_velocity_error)
        position_errors.append(position_error)
        velocity_errors.append(velocity_error)
        rows.append(
            {
                "source_joint": source_joint,
                "source_dof": source_dof,
                "mujoco_initial_position": float(before_qpos[source_qpos]),
                "mujoco_initial_velocity": float(before_qvel[source_dof]),
                "rapier_initial_position": float(native["initial_position"]),
                "rapier_initial_velocity": float(native["initial_velocity"]),
                "initial_position_abs_error_rad": initial_position_error,
                "initial_velocity_abs_error_rad_s": initial_velocity_error,
                "mujoco_solve_qfrc_bias_nm": float(before_bias[source_dof]),
                "mujoco_solve_qfrc_constraint_nm": float(before_constraint[source_dof]),
                "mujoco_solve_qfrc_passive_nm": passive,
                "mujoco_solve_qfrc_actuator_nm": float(before_actuator[source_dof]),
                "mujoco_dof_frictionloss_nm": frictionloss,
                "mujoco_external_without_own_friction_nm": source_external,
                "rapier_candidate_residual_nm": native_external,
                "force_abs_error_nm": force_error,
                "post_mujoco_external_nm": source_external_post,
                "mujoco_post_position": float(post_qpos[source_qpos]),
                "mujoco_post_velocity": float(post_qvel[source_dof]),
                "rapier_post_position": float(native["post_position"]),
                "rapier_post_velocity": float(native["post_velocity"]),
                "post_position_abs_error_rad": position_error,
                "post_velocity_abs_error_rad_s": velocity_error,
            }
        )

    report = {
        "scope": "frozen_robot_only_mujoco_rapier_one_step_force_comparison",
        "mujoco_version": mujoco.__version__,
        "mjb_sha256": args.mjb_sha256,
        "qpos_sha256": args.qpos_sha256,
        "qvel_sha256": args.qvel_sha256,
        "rapier_model_sha256": rapier["model_file_sha256"],
        "rapier_report_sha256": hashlib.sha256(rapier_bytes).hexdigest(),
        "source_timestep_seconds": model.opt.timestep,
        "rapier_timestep_seconds": rapier["actual_step_dt_seconds"],
        "source_contacts_before_step": before_contacts,
        "source_contacts_at_solve": step_contacts,
        "source_contacts_after_step": post_contacts,
        "rapier_active_contact_pair_count": rapier["active_contact_pair_count"],
        "source_driven_dofs_frictionless": all_frictionless,
        "max_force_abs_error_nm": max(force_errors),
        "rms_force_error_nm": math.sqrt(sum(error * error for error in force_errors) / len(force_errors)),
        "max_initial_position_abs_error_rad": max(initial_position_errors),
        "max_initial_velocity_abs_error_rad_s": max(initial_velocity_errors),
        "max_post_position_abs_error_rad": max(position_errors),
        "max_post_velocity_abs_error_rad_s": max(velocity_errors),
        "rows": rows,
        "bam_external_load_qualified": False,
        "source_target_contact_qualified": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
