"""Replay source/native head_roll soft-limit steps with real actuator torque.

Run with MuJoCo 3.10.0 after building paired_force_probe with
sim2sim_source_limit_probe. This is a SHA-bound, robot-only, one-step diagnostic;
it neither selects the production limit law nor qualifies BAM or contacts.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / ".scratch/source_limit_actuated_v1"
BIN = ROOT / "target/debug/paired_force_probe"
COMPARE = ROOT / "crates/dev_tools/python/scripts/compare_paired_force.py"
FULL_TORQUE = ROOT / ".scratch/paired_force_v1/nonzero_actuator_torques.json"
FULL_TORQUE_SHA = "1a566de67c160ceb780cb28ea627bf5d3933dac828c684ce9498f5c7b69a77f5"
FROZEN = {
    "leg": {
        "mjb": "832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c",
        "definition": "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb",
        "qpos": "4caf605435e262d30ad8a32ca9bf7f6280e85cf5c003947e937394918180c2ac",
        "qvel": "b65778c25e5ddc78b31aadcff3227f93156ce0cb396a6f5c2ebd8615987b9d2a",
        "source_joint": 9,
    },
    "roller": {
        "mjb": "ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191",
        "definition": "6f88aa286e487c031250b427595a640b9f597604c9a20dcbca6fd69c49109c40",
        "qpos": "7ff592bdda0c2d4c1376797c4119564eaf0ec481dd3aa9191eaaedf0397564b9",
        "qvel": "b071c6ce6c4da113902b1fb6822864744f59fe3291e92ff167e285f41f29e66a",
        "source_joint": 11,
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path, expected: str) -> None:
    assert sha(path) == expected, (path, sha(path), expected)


def write_json(path: Path, value: object) -> str:
    path.write_bytes((json.dumps(value, separators=(",", ":")) + "\n").encode())
    return sha(path)


def run(family: str, mode: str, torque: Path, torque_sha: str) -> dict:
    ids = FROZEN[family]
    mjb = ROOT / f".scratch/root_robot_assembly_mujoco_v1/{family}_native_model.mjb"
    definition = ROOT / f".scratch/source_limit_probe_v1/{family}_enriched_with_hulls.json"
    qpos = ROOT / f".scratch/source_limit_probe_v1/{family}_18_qpos.json"
    qvel = ROOT / f".scratch/source_limit_probe_v1/{family}_18_qvel.json"
    for path, expected in ((mjb, ids["mjb"]), (definition, ids["definition"]),
                           (qpos, ids["qpos"]), (qvel, ids["qvel"]), (torque, torque_sha)):
        checked(path, expected)
    native = OUT / f"{family}_{mode}_native.json"
    comparison = OUT / f"{family}_{mode}_comparison.json"
    subprocess.run(
        [str(BIN), "--head-roll-source-limit-diagnostic", str(definition),
         ids["definition"], str(qpos), ids["qpos"], str(qvel), ids["qvel"],
         str(torque), torque_sha, str(native)],
        cwd=ROOT, check=True, timeout=60, capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(COMPARE), str(mjb), ids["mjb"], str(qpos), ids["qpos"],
         str(native), str(comparison), "--qvel", str(qvel), "--qvel-sha256",
         ids["qvel"], "--torques", str(torque), "--torques-sha256", torque_sha],
        cwd=ROOT, check=True, timeout=60, capture_output=True,
    )
    report = json.loads(native.read_bytes())
    paired = json.loads(comparison.read_bytes())
    assert report["source_limit_probe_selected"] is True
    assert report["physics_integrations"] == 1
    assert report["actuator_torque_file_sha256"] == torque_sha
    assert report["model_file_sha256"] == ids["definition"]
    assert all(pair["normal_impulse_sum"] == 0.0 for pair in report["active_contact_pairs"])
    assert all(row["contact_normal_impulse_nms"] == 0.0
               and row["contact_tangent_impulse_nms"] == 0.0 for row in report["rows"])
    assert paired["source_contacts_before_step"] == 0
    assert paired["source_contacts_at_solve"] == 0
    assert paired["source_contacts_after_step"] == 0
    assert paired["max_source_applied_actuator_abs_error_nm"] < 1.0e-12
    assert paired["max_native_user_force_projection_abs_error_nm"] < 1.0e-6
    row = next(row for row in report["rows"] if row["source_joint"] == ids["source_joint"])
    source_row = next(row for row in paired["rows"] if row["source_joint"] == ids["source_joint"])
    trace = next(t for t in row["limit_row_timing"] if t["phase"] == "after_unbiased_solve")
    native_force = -trace["impulse_nms"] / report["actual_step_dt_seconds"]
    source_force = source_row["mujoco_solve_qfrc_constraint_nm"]
    force_error = abs(native_force - source_force)
    assert trace["impulse_bounds"][1] > 0.0
    assert force_error < 1.0e-6
    assert source_row["post_position_abs_error_rad"] < 1.0e-6
    assert source_row["post_velocity_abs_error_rad_s"] < 1.0e-5
    assert paired["max_force_abs_error_nm"] < 1.0e-6
    assert paired["max_post_velocity_abs_error_rad_s"] < 1.0e-5
    return {
        "family": family,
        "torque_mode": mode,
        "torque_sha256": torque_sha,
        "native_report_sha256": sha(native),
        "comparison_sha256": sha(comparison),
        "source_limit_force_nm": source_force,
        "native_limit_force_nm": native_force,
        "limit_force_abs_error_nm": force_error,
        "head_roll_final_position_abs_error_rad": source_row["post_position_abs_error_rad"],
        "head_roll_final_velocity_abs_error_rad_s": source_row["post_velocity_abs_error_rad_s"],
        "all_driven_dofs_max_force_abs_error_nm": paired["max_force_abs_error_nm"],
        "all_driven_dofs_max_final_velocity_abs_error_rad_s": paired["max_post_velocity_abs_error_rad_s"],
        "source_contact_count": 0,
        "native_contact_impulse_zero": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checked(FULL_TORQUE, FULL_TORQUE_SHA)
    full = json.loads(FULL_TORQUE.read_bytes())
    assert len(full) == 14 and full[8] == 0.04
    modes = {"full_vector": (FULL_TORQUE, FULL_TORQUE_SHA)}
    for label, value in (("head_positive", 0.12), ("head_negative", -0.12)):
        torques = [0.0] * 14
        torques[8] = value
        path = OUT / f"{label}_torques.json"
        modes[label] = (path, write_json(path, torques))
    cases = [run(family, mode, *paths) for family in FROZEN for mode, paths in modes.items()]
    assert len(cases) == 6
    summary = {
        "scope": "sha_bound_robot_only_single_step_actuated_source_limit_diagnostic",
        "source_version": "MuJoCo 3.10.0",
        "target_feature": "sim2sim_source_limit_probe with explicit CLI opt-in",
        "frozen": FROZEN,
        "cases": cases,
        "max_limit_force_abs_error_nm": max(c["limit_force_abs_error_nm"] for c in cases),
        "max_final_position_abs_error_rad": max(c["head_roll_final_position_abs_error_rad"] for c in cases),
        "max_final_velocity_abs_error_rad_s": max(c["head_roll_final_velocity_abs_error_rad_s"] for c in cases),
        "multi_step_qualified": False,
        "contact_qualified": False,
        "bam_external_load_qualified": False,
    }
    result = OUT / "summary.json"
    write_json(result, summary)
    print(result, sha(result))


if __name__ == "__main__":
    main()
