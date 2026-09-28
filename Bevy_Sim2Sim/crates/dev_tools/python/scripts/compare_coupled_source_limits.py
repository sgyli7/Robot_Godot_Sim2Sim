"""Compare two simultaneous source-style limit rows at one true 60 Hz step.

Requires MuJoCo 3.10.0 and a paired_force_probe build with the default-off
sim2sim_source_limit_probe feature. Internal PGS passes vary without changing
the one temporal step. This is a frozen robot-only diagnostic, not BAM input.
"""

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from source_limit_probe_matrix import EXPECTED, ROOT, checked, materialize_definition, sha, write_json


OUT = ROOT / ".scratch/source_limit_two_row_v1"
BIN = ROOT / "target/debug/paired_force_probe"
COMPARE = ROOT / "crates/dev_tools/python/scripts/compare_paired_force.py"
PGS_PASSES = (1, 2, 4, 8, 16)
JOINT_NAMES = ("right_hip_pitch", "right_knee")
FROZEN = {
    "leg": {
        "mjb_sha": "832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c",
        "qpos_sha": "60caab636a662633040674f5bc434ad3620af75eb9593ccc4ba1b1ce43e805d0",
        "qvel_sha": "b65778c25e5ddc78b31aadcff3227f93156ce0cb396a6f5c2ebd8615987b9d2a",
    },
    "roller": {
        "mjb_sha": "ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191",
        "qpos_sha": "cb73871ba3a0c57f4dd172102ae4c80141452b3a0e723293c65e74a4d7f43a08",
        "qvel_sha": "b071c6ce6c4da113902b1fb6822864744f59fe3291e92ff167e285f41f29e66a",
    },
}


def source_inputs(family: str, identity: dict[str, str]) -> tuple[Path, Path, Path, list[int], float]:
    mjb = ROOT / f".scratch/root_robot_assembly_mujoco_v1/{family}_native_model.mjb"
    checked(mjb, identity["mjb_sha"])
    model = mujoco.MjModel.from_binary_path(str(mjb))
    assert mujoco.mj_versionString() == "3.10.0"
    assert abs(model.opt.timestep - 1.0 / 60.0) < 1.0e-12
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    base = ROOT / f".scratch/paired_force_v1/{family}_head_roll_limit_qpos.json"
    qpos = json.loads(checked(base, EXPECTED[family]["base_qpos_sha"]))
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_roll")
    qpos[int(model.jnt_qposadr[head])] = float(model.qpos0[int(model.jnt_qposadr[head])])
    joint_ids = []
    for name in JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint >= 0 and model.jnt_limited[joint]
        qpos[int(model.jnt_qposadr[joint])] = float(model.jnt_range[joint, 1] + 0.02)
        joint_ids.append(joint)
    qvel = [0.0] * model.nv
    qpos_path = OUT / f"{family}_qpos.json"
    qvel_path = OUT / f"{family}_qvel.json"
    assert write_json(qpos_path, qpos) == identity["qpos_sha"]
    assert write_json(qvel_path, qvel) == identity["qvel_sha"]
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    assert data.ncon == 0 and data.nefc == 2
    assert set(map(int, data.efc_id[:data.nefc])) == set(joint_ids)
    assert all(data.efc_type[index] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
               for index in range(data.nefc))
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
    row_inverse_mass = jacobian @ np.linalg.solve(mass, jacobian.T)
    coupling = abs(row_inverse_mass[0, 1]) / np.sqrt(
        row_inverse_mass[0, 0] * row_inverse_mass[1, 1]
    )
    assert 0.05 < coupling < 0.1
    return mjb, qpos_path, qvel_path, joint_ids, float(coupling)


def one_case(family: str, passes: int, definition: Path, inputs: tuple) -> dict:
    mjb, qpos, qvel, joints, coupling = inputs
    identity = FROZEN[family]
    model_sha = EXPECTED[family]["definition_sha"]
    native_path = OUT / f"{family}_pgs{passes}_native.json"
    comparison_path = OUT / f"{family}_pgs{passes}_comparison.json"
    subprocess.run(
        [str(BIN), "--all-source-limits-diagnostic",
         f"--source-limit-pgs-iterations={passes}", str(definition), model_sha,
         str(qpos), identity["qpos_sha"], str(qvel), identity["qvel_sha"], str(native_path)],
        cwd=ROOT, check=True, timeout=60, capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(COMPARE), str(mjb), identity["mjb_sha"], str(qpos),
         identity["qpos_sha"], str(native_path), str(comparison_path),
         "--qvel", str(qvel), "--qvel-sha256", identity["qvel_sha"]],
        cwd=ROOT, check=True, timeout=60, capture_output=True,
    )
    native = json.loads(native_path.read_bytes())
    comparison = json.loads(comparison_path.read_bytes())
    assert native["model_file_sha256"] == model_sha
    assert native["source_limit_probe_selected"] is True
    assert native["source_limit_probe_joint"] == "all_limited"
    assert native["source_limit_internal_pgs_iterations"] == passes
    assert native["physics_integrations"] == 1
    assert abs(native["actual_step_dt_seconds"] - 1.0 / 60.0) < 1.0e-8
    assert comparison["source_contacts_before_step"] == 0
    assert comparison["source_contacts_at_solve"] == 0
    assert comparison["source_contacts_after_step"] == 0
    assert all(row["contact_normal_impulse_nms"] == 0.0
               and row["contact_tangent_impulse_nms"] == 0.0 for row in native["rows"])
    assert all(pair["normal_impulse_sum"] == 0.0 for pair in native["active_contact_pairs"])
    rows = []
    for joint, name in zip(joints, JOINT_NAMES, strict=True):
        source = next(row for row in comparison["rows"] if row["source_joint"] == joint)
        target = next(row for row in native["rows"] if row["source_joint"] == joint)
        phases = {sample["phase"]: sample for sample in target["limit_row_timing"]}
        assert set(phases) == {"after_biased_solve", "after_position_integration", "after_unbiased_solve"}
        assert all(sample["substep_id"] == 0 for sample in phases.values())
        final = phases["after_unbiased_solve"]
        assert final["impulse_bounds"][1] > 0.0
        assert abs(target["generic_joint_impulse_nms"] + final["impulse_nms"]) < 1.0e-10
        target_force = target["generic_joint_impulse_nms"] / native["actual_step_dt_seconds"]
        source_force = source["mujoco_solve_qfrc_constraint_nm"]
        rows.append({
            "source_joint": joint,
            "joint_name": name,
            "source_limit_force_nm": source_force,
            "native_limit_force_nm": target_force,
            "limit_force_abs_error_nm": abs(target_force - source_force),
            "initial_position_abs_error_rad": source["initial_position_abs_error_rad"],
            "post_position_abs_error_rad": source["post_position_abs_error_rad"],
            "post_velocity_abs_error_rad_s": source["post_velocity_abs_error_rad_s"],
            "native_velocity_after_biased_solve_rad_s": phases["after_biased_solve"]["generalized_velocity"],
            "native_velocity_after_unbiased_solve_rad_s": final["generalized_velocity"],
        })
    return {
        "family": family,
        "internal_pgs_iterations": passes,
        "temporal_substeps": 1,
        "source_row_inverse_mass_coupling_ratio": coupling,
        "native_report_sha256": sha(native_path.read_bytes()),
        "comparison_sha256": sha(comparison_path.read_bytes()),
        "source_contact_count": 0,
        "native_active_self_contact_pairs": native["active_contact_pair_count"],
        "native_contact_impulses_zero": True,
        "rows": rows,
        "max_limit_force_abs_error_nm": max(row["limit_force_abs_error_nm"] for row in rows),
        "max_post_position_abs_error_rad": max(row["post_position_abs_error_rad"] for row in rows),
        "max_post_velocity_abs_error_rad_s": max(row["post_velocity_abs_error_rad_s"] for row in rows),
        "all_driven_dofs_max_force_abs_error_nm": comparison["max_force_abs_error_nm"],
        "all_driven_dofs_max_post_position_abs_error_rad": comparison["max_post_position_abs_error_rad"],
        "all_driven_dofs_max_post_velocity_abs_error_rad_s": comparison["max_post_velocity_abs_error_rad_s"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert BIN.exists()
    cases = []
    for family, identity in FROZEN.items():
        definition = materialize_definition(family, EXPECTED[family])
        checked(definition, EXPECTED[family]["definition_sha"])
        inputs = source_inputs(family, identity)
        for passes in PGS_PASSES:
            case = one_case(family, passes, definition, inputs)
            cases.append(case)
            print(family, passes, case["max_post_position_abs_error_rad"], flush=True)
    assert len(cases) == 10
    for family in FROZEN:
        by_pass = {case["internal_pgs_iterations"]: case for case in cases if case["family"] == family}
        assert by_pass[1]["max_post_position_abs_error_rad"] > 1.0e-4
        assert by_pass[4]["max_limit_force_abs_error_nm"] < 1.0e-6
        assert by_pass[4]["max_post_position_abs_error_rad"] < 1.0e-6
        assert by_pass[4]["max_post_velocity_abs_error_rad_s"] < 1.0e-5
    result = {
        "scope": "same_state_two_coupled_source_limit_rows_single_temporal_step",
        "mujoco_version": mujoco.mj_versionString(),
        "frozen_source": FROZEN,
        "frozen_native_model_sha256": {family: EXPECTED[family]["definition_sha"] for family in FROZEN},
        "joint_names": JOINT_NAMES,
        "internal_pgs_passes": PGS_PASSES,
        "cases": cases,
        "production_solver_changed": False,
        "contact_topology_qualified": False,
        "bam_external_load_qualified": False,
    }
    path = OUT / "coupled_comparison.json"
    write_json(path, result)
    print("result", path, sha(path.read_bytes()), flush=True)


if __name__ == "__main__":
    main()
