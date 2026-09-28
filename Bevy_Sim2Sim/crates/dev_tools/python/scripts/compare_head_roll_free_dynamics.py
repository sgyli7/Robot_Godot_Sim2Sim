"""Compare 42 frozen head_roll source states with read-only native free dynamics.

Run from Bevy_Sim2Sim with MuJoCo 3.10.0 and a paired_force_probe binary
built with sim2sim_plain_mass_probe. Writes only under --output-dir. This does
not qualify the target limit solver, contact load, BAM, or a policy skill.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np


IDENTITIES = {
    "leg": {
        "mjb": "832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c",
        "definition": "eb7a276a5199c757181c60681de54a7eb3e490431929d1fb043f8fb25a964299",
        "qpos": "e16f8e7ddf1d1cf2061b4c760faa1cbf78536d1cd8e6b992ca415eef10ae21bf",
        "matrix": "c30cfd248770f4eb761672616f6417ead877337624f590f8602459475b730a9e",
    },
    "roller": {
        "mjb": "ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191",
        "definition": "a4524cb0b63a64fd88f3fff1375c8fac375d2ec88254545409ca6b6eaca768fd",
        "qpos": "df50ef2c31243ef01c6419f18238008c9a2d8d3e444073b82c61cb27b3fb7773",
        "matrix": "48b23a63c220f9713f38a022a9254d920aad67fd2411ca87119f83fd970f027d",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: {actual} != {expected}")
    return path.read_bytes()


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return sha256(path)


def frozen_impedance(depth: float, solimp: np.ndarray) -> float:
    d0, dw, width, midpoint, power = map(float, solimp)
    if not (d0 == 0.9 and dw == 0.95 and width == 0.001 and midpoint == 0.5 and power == 2.0):
        raise ValueError("source solimp differs from this frozen diagnostic formula")
    x = min(1.0, max(0.0, depth / width))
    spline = 2.0 * x * x if x <= 0.5 else 1.0 - 2.0 * (1.0 - x) ** 2
    return d0 + (dw - d0) * spline


def native_run(probe: Path, model: Path, model_sha: str, qpos: Path, qpos_sha: str,
               qvel: Path, qvel_sha: str, output: Path, plain: bool) -> tuple[Path, str]:
    flags = ["--head-roll-free-acceleration-diagnostic"]
    if plain:
        flags.insert(0, "--plain-mass-diagnostic")
    command = [str(probe), *flags, str(model), model_sha, str(qpos), qpos_sha,
               str(qvel), qvel_sha, str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"native case {output} failed ({result.returncode}): {result.stderr}")
    return output, sha256(output)


def source_cases(family: str, output: Path) -> tuple[list[dict], list[tuple]]:
    ids = IDENTITIES[family]
    mjb = Path(f".scratch/root_robot_assembly_mujoco_v1/{family}_native_model.mjb")
    definition = Path(f".scratch/root_native_hull_import_v1/{family}_compiled_with_native_hulls.json")
    base_qpos_path = Path(f".scratch/paired_force_v1/{family}_head_roll_limit_qpos.json")
    matrix_path = Path(f".scratch/paired_force_v1/{family}_limit_matrix.json")
    checked(mjb, ids["mjb"])
    checked(definition, ids["definition"])
    base_qpos = json.loads(checked(base_qpos_path, ids["qpos"]))
    matrix = json.loads(checked(matrix_path, ids["matrix"]))
    model = mujoco.MjModel.from_binary_path(str(mjb))
    if mujoco.__version__ != "3.10.0" or matrix["mujoco_version"] != mujoco.__version__:
        raise ValueError("source MuJoCo version differs from frozen 3.10.0")
    if abs(model.opt.timestep - 1.0 / 60.0) > 1e-12:
        raise ValueError("source timestep differs from 1/60 s")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        raise ValueError("source integrator is not Euler")
    if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE):
        raise ValueError("source refsafe is disabled")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_roll")
    dof = int(model.jnt_dofadr[joint])
    qpos_index = int(model.jnt_qposadr[joint])
    upper = float(model.jnt_range[joint, 1])
    if (joint != matrix["source_joint"] or dof != matrix["source_dof"]
            or upper != matrix["upper_limit_rad"] or model.jnt_margin[joint] != 0.0
            or not model.jnt_limited[joint] or len(matrix["cases"]) != 21):
        raise ValueError("source head_roll identity, limit, or matrix cases differ")
    timeconst, dampratio = map(float, model.jnt_solref[joint])
    if timeconst != 0.02 or dampratio != 1.0:
        raise ValueError("source solref differs from frozen diagnostic")
    timeconst = max(timeconst, 2.0 * float(model.opt.timestep))
    dw = float(model.jnt_solimp[joint, 1])
    entries = []
    jobs = []
    for index, case in enumerate(matrix["cases"]):
        depth = float(case["depth_from_upper_rad"])
        velocity = float(case["initial_joint_velocity_rad_s"])
        qpos = list(base_qpos)
        qpos[qpos_index] = upper + depth
        qvel = [0.0] * model.nv
        qvel[dof] = velocity
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon != 0 or data.nefc > 1 or (data.nefc == 1) != (case["row"] is not None):
            raise ValueError("source case is not the recorded no-contact single-row state")
        active = depth > 0.0
        if active != (data.nefc == 1):
            raise ValueError("source head_roll row activation differs")
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        unit = np.zeros(model.nv)
        unit[dof] = 1.0
        a = float(unit @ np.linalg.solve(mass, unit))
        a0 = -float(data.qacc_smooth[dof])
        if active:
            impedance = frozen_impedance(depth, model.jnt_solimp[joint])
            row_velocity = -velocity
            residual = -depth
            damping = 2.0 / (dw * timeconst)
            stiffness = impedance / (dw * dw * timeconst * timeconst * dampratio * dampratio)
            aref = -damping * row_velocity - stiffness * residual
            regularizer = float(model.dof_invweight0[dof]) * (1.0 - impedance) / impedance
            source_force = max(0.0, (aref - a0) / (a + regularizer))
            recorded = case["row"]
            if (abs(aref - recorded["reference_acceleration_rad_s2"]) > 1e-10
                    or abs(regularizer - recorded["regularizer"]) > 1e-10
                    or abs(a - recorded["effective_inverse_mass"]) > 1e-8
                    or abs(-source_force - case["solve_constraint_nm"]) > 1e-10):
                raise ValueError("independently reconstructed source row differs from matrix")
        else:
            aref = regularizer = source_force = 0.0
            if case["solve_constraint_nm"] != 0.0:
                raise ValueError("inactive source row has nonzero solve force")
        stem = f"{family}_{index:02d}"
        qpos_path = output / f"{stem}_qpos.json"
        qvel_path = output / f"{stem}_qvel.json"
        qpos_sha = write_json(qpos_path, qpos)
        qvel_sha = write_json(qvel_path, qvel)
        entry = {
            "family": family,
            "case_index": index,
            "depth_from_upper_rad": depth,
            "initial_joint_velocity_rad_s": velocity,
            "source_row_active": active,
            "source_free_row_acceleration_rad_s2": a0,
            "source_effective_inverse_inertia_rad_s2_per_nm": a,
            "source_reference_acceleration_rad_s2": aref,
            "source_regularizer_rad_s2_per_nm": regularizer,
            "source_reconstructed_limit_force_nm": source_force,
            "source_solved_joint_constraint_nm": case["solve_constraint_nm"],
            "source_post_joint_position_rad": case["post_joint_position_rad"],
            "source_post_joint_velocity_rad_s": case["post_joint_velocity_rad_s"],
            "qpos_sha256": qpos_sha,
            "qvel_sha256": qvel_sha,
        }
        entries.append(entry)
        for plain in (False, True):
            label = "plain" if plain else "implicit"
            report_path = output / f"{stem}_{label}_target.json"
            jobs.append((definition, ids["definition"], qpos_path, qpos_sha,
                         qvel_path, qvel_sha, report_path, plain))
    return entries, jobs


def maximum(entries: list[dict], key: str, predicate=lambda _: True) -> dict:
    eligible = [entry for entry in entries if predicate(entry)]
    worst = max(eligible, key=lambda entry: abs(entry[key]))
    return {"absolute_error": abs(worst[key]), "signed_error": worst[key],
            "family": worst["family"], "case_index": worst["case_index"],
            "depth_from_upper_rad": worst["depth_from_upper_rad"],
            "initial_joint_velocity_rad_s": worst["initial_joint_velocity_rad_s"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("target/debug/paired_force_probe"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(".scratch/paired_limit_target_v1"))
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    jobs = []
    for family in IDENTITIES:
        source, family_jobs = source_cases(family, args.output_dir)
        sources.extend(source)
        jobs.extend(family_jobs)
    report_hashes = {}
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = {
            workers.submit(native_run, probe, *job): job[6]
            for job in jobs
        }
        for future in as_completed(futures):
            path, digest = future.result()
            report_hashes[path.name] = digest
    comparisons = []
    for source in sources:
        for label in ("implicit", "plain"):
            path = args.output_dir / f"{source['family']}_{source['case_index']:02d}_{label}_target.json"
            target = json.loads(checked(path, report_hashes[path.name]))
            free = target["head_roll_free_acceleration_diagnostic"]
            if (target["physics_integrations"] != 1 or target["native_observation_epoch"] != 1
                    or target["policy_inferences"] != 0
                    or abs(target["actual_step_dt_seconds"] - 1.0 / 60.0) > 1e-8
                    or target["qpos_file_sha256"] != source["qpos_sha256"]
                    or target["qvel_file_sha256"] != source["qvel_sha256"]
                    or target["model_file_sha256"] != IDENTITIES[source["family"]]["definition"]
                    or target["selected_free_acceleration_matrix"]
                    != ("plain_mass_diagnostic" if label == "plain" else "implicit_gyro_coriolis_mass")
                    or free["source_dof"] != (14 if source["family"] == "leg" else 16)
                    or free["upper_limit_row_jacobian_sign"] != -1
                    or any(abs(row["contact_normal_impulse_nms"]) > 0.0
                           or abs(row["contact_tangent_impulse_nms"]) > 0.0
                           for row in target["rows"])
                    or any(pair["normal_impulse_sum"] != 0.0
                           for pair in target["active_contact_pairs"])):
                raise ValueError(f"native identity, epoch, mode or contact gate failed: {path}")
            a0 = float(free["pre_constraint_free_row_acceleration_rad_s2"])
            a = float(free["plain_mass_effective_inverse_inertia_rad_s2_per_nm"])
            if not math.isfinite(a0) or not math.isfinite(a) or a <= 0.0:
                raise ValueError(f"native free dynamics invalid: {path}")
            force = (max(0.0, (source["source_reference_acceleration_rad_s2"] - a0)
                         / (a + source["source_regularizer_rad_s2_per_nm"]))
                     if source["source_row_active"] else 0.0)
            comparisons.append({
                **source,
                "target_mass_mode": label,
                "target_report": path.name,
                "target_report_sha256": report_hashes[path.name],
                "target_free_row_acceleration_rad_s2": a0,
                "target_plain_mass_effective_inverse_inertia_rad_s2_per_nm": a,
                "target_predicted_source_limit_force_nm": force,
                "target_post_joint_position_rad": free["post_position_rad"],
                "target_post_joint_velocity_rad_s": free["post_velocity_rad_s"],
                "free_row_acceleration_error_rad_s2": a0 - source["source_free_row_acceleration_rad_s2"],
                "effective_inverse_inertia_error_rad_s2_per_nm": a - source["source_effective_inverse_inertia_rad_s2_per_nm"],
                "predicted_source_limit_force_error_nm": force - source["source_reconstructed_limit_force_nm"],
                "actual_target_post_position_error_rad": free["post_position_rad"] - source["source_post_joint_position_rad"],
                "actual_target_post_velocity_error_rad_s": free["post_velocity_rad_s"] - source["source_post_joint_velocity_rad_s"],
            })
    summary = {}
    for label in ("implicit", "plain"):
        selected = [entry for entry in comparisons if entry["target_mass_mode"] == label]
        summary[label] = {
            "case_count": len(selected),
            "active_source_rows": sum(entry["source_row_active"] for entry in selected),
            "max_free_row_acceleration_error_rad_s2": maximum(selected, "free_row_acceleration_error_rad_s2"),
            "max_effective_inverse_inertia_error_rad_s2_per_nm": maximum(selected, "effective_inverse_inertia_error_rad_s2_per_nm"),
            "max_predicted_source_limit_force_error_nm": maximum(selected, "predicted_source_limit_force_error_nm"),
            "max_active_actual_post_position_error_rad": maximum(selected, "actual_target_post_position_error_rad", lambda entry: entry["source_row_active"]),
            "max_active_actual_post_velocity_error_rad_s": maximum(selected, "actual_target_post_velocity_error_rad_s", lambda entry: entry["source_row_active"]),
        }
    if len(comparisons) != 84 or any(section["case_count"] != 42 for section in summary.values()):
        raise ValueError("expected 42 source states and 84 native one-step comparisons")
    report = {
        "scope": "read_only_head_roll_free_dynamics_42_source_states_84_target_steps",
        "source_mujoco_version": mujoco.__version__,
        "identities": IDENTITIES,
        "source_case_count": 42,
        "target_step_count": 84,
        "summary": summary,
        "cases": comparisons,
        "target_limit_qualified": False,
        "multi_row_or_contact_qualified": False,
        "bam_external_load_qualified": False,
    }
    output = args.output_dir / "comparison.json"
    digest = write_json(output, report)
    print(json.dumps({"output": str(output), "sha256": digest, "summary": summary},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
