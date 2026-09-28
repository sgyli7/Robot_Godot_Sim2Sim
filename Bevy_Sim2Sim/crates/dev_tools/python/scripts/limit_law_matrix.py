"""Record the frozen MuJoCo head-roll limit law over a small source-state grid.

This is a source-side diagnostic. It neither changes the native plant nor
qualifies Rapier limits, contacts, BAM feedback, or a policy skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np


DEPTHS = (-0.001, 0.0, 0.0002, 0.0005, 0.001, 0.005, 0.02)
VELOCITIES = (0.0, 0.3, -0.3)


def checked(path: Path, expected_sha256: str) -> bytes:
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mjb", type=Path)
    parser.add_argument("mjb_sha256")
    parser.add_argument("qpos", type=Path)
    parser.add_argument("qpos_sha256")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checked(args.mjb, args.mjb_sha256)
    base_qpos = json.loads(checked(args.qpos, args.qpos_sha256))
    model = mujoco.MjModel.from_binary_path(str(args.mjb))
    if len(base_qpos) != model.nq or not math.isclose(model.opt.timestep, 1 / 60, abs_tol=1e-12):
        raise ValueError("source state or physics time step differs from the frozen contract")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        raise ValueError("source integrator is not Euler")
    if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE):
        raise ValueError("source refsafe is disabled")

    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_roll")
    if joint < 0 or not model.jnt_limited[joint]:
        raise ValueError("head_roll limit is absent")
    qpos_index = int(model.jnt_qposadr[joint])
    dof = int(model.jnt_dofadr[joint])
    upper = float(model.jnt_range[joint, 1])
    cases = []

    for depth in DEPTHS:
        for velocity in VELOCITIES:
            data = mujoco.MjData(model)
            data.qpos[:] = base_qpos
            data.qpos[qpos_index] = upper + depth
            data.qvel[:] = 0.0
            data.qvel[dof] = velocity
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            if data.ncon != 0 or data.nefc > 1:
                raise ValueError("matrix state has contact or multiple active constraints")

            row = None
            if data.nefc == 1:
                if (
                    data.efc_type[0] != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
                    or int(data.efc_id[0]) != joint
                ):
                    raise ValueError("active row is not the frozen head_roll limit")
                jacobian = np.asarray(data.efc_J[: model.nv]).copy()
                if abs(jacobian[dof] + 1.0) > 1e-12 or np.count_nonzero(jacobian) != 1:
                    raise ValueError("limit Jacobian is not the expected signed unit DOF")
                mass = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, data, mass)
                a = float(jacobian @ np.linalg.solve(mass, jacobian))
                row = {
                    "type": "head_roll_upper_limit",
                    "source_joint": joint,
                    "source_dof": dof,
                    "jacobian_sign": -1.0,
                    "position_residual_rad": float(data.efc_pos[0]),
                    "velocity_residual_rad_s": float(data.efc_vel[0]),
                    "reference_acceleration_rad_s2": float(data.efc_aref[0]),
                    "regularizer": float(data.efc_R[0]),
                    "effective_inverse_mass": a,
                    "initial_efc_force_nm": float(data.efc_force[0]),
                }

            mujoco.mj_step(model, data)
            if data.ncon != 0 or data.nefc > 1:
                raise ValueError("step solve changed the single-limit row scope")
            cases.append(
                {
                    "depth_from_upper_rad": depth,
                    "initial_joint_velocity_rad_s": velocity,
                    "row": row,
                    "solve_constraint_nm": float(data.qfrc_constraint[dof]),
                    "solve_actuator_nm": float(data.qfrc_actuator[dof]),
                    "post_joint_position_rad": float(data.qpos[qpos_index]),
                    "post_joint_velocity_rad_s": float(data.qvel[dof]),
                }
            )

    report = {
        "scope": "frozen_mujoco_head_roll_single_limit_source_matrix",
        "mujoco_version": mujoco.__version__,
        "mjb_sha256": args.mjb_sha256,
        "base_qpos_sha256": args.qpos_sha256,
        "source_timestep_seconds": float(model.opt.timestep),
        "source_joint": joint,
        "source_dof": dof,
        "upper_limit_rad": upper,
        "limit_margin_rad": float(model.jnt_margin[joint]),
        "limit_solref": model.jnt_solref[joint].tolist(),
        "limit_solimp": model.jnt_solimp[joint].tolist(),
        "cases": cases,
        "target_limit_qualified": False,
        "bam_external_load_qualified": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
