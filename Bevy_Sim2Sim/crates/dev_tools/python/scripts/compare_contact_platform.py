"""Compare the frozen original left-foot platform contact in MuJoCo and Rapier.

This diagnostic consumes raw, SHA-bound scratch reports from the exact one-step
and eight-step fixtures. A mismatch is the finding; no production contact law,
BAM input, or skill gate is changed by this script.
"""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / ".scratch/contact_platform_source_v1"
TARGET = ROOT / ".scratch/contact_platform_target_v1"
OUT = TARGET / "paired_contact_comparison.json"
EXPECTED_SOURCE_ONE_SHA = "af2035d28d31013237816c6e65c2ee07d6d5cf62e106c955263da42f97224bf7"
EXPECTED_SOURCE_EIGHT_SHA = "0486357a9aeb6274d3984570acc936aed6b723168efc717d4ce54e7c4de9cd7d"
EXPECTED_SOURCE_MODEL_SHA = "90ba3d2783616604a3ccb32f889343d854feb2f3317dfbcb39141bdf0362096c"
EXPECTED_NEUTRAL_QPOS_SHA = "09a9879e8647d51933ea8d0b2cbb97137167c54d666fd372b6aa1d590798b848"
EXPECTED_TARGET_MODEL_SHA = "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb"
EXPECTED_TARGET_ONE_SHA = "1520378ce7457f904434503de6ce50a3112257c86ee348d0caad56631a84d0c4"
EXPECTED_TARGET_EIGHT_SHA = "76cdd0930e32644480f13cc98b145032ae6892854723e1064fb2d30f37104b37"
EXPECTED_TARGET_FAR_SHA = "4a2f84e2a4e13e861a6ee8ed436bc9f36e6a5173b712ef27f817d51108d5a971"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path, expected_sha: str | None = None) -> dict:
    actual = sha(path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"SHA256 mismatch: {path}: {actual}")
    return json.loads(path.read_bytes())


def target_root_source_xyz(row: dict) -> list[float]:
    engine = row["post_pose"]["poses"][0]["translation"]
    return [engine[0], -engine[2], engine[1]]


def root_position_error(source_qpos: list[float], target: dict) -> dict:
    expected = source_qpos[:3]
    actual = target_root_source_xyz(target)
    signed = [a - b for a, b in zip(actual, expected, strict=True)]
    return {"source_xyz_m": expected, "target_xyz_m": actual,
            "signed_error_xyz_m": signed, "max_abs_error_m": max(map(abs, signed))}


def root_rotation_error(source_qpos: list[float], target: dict) -> float:
    w, x, y, z = source_qpos[3:7]
    expected_engine = [x, z, -y, w]
    actual = target["post_pose"]["poses"][0]["rotation_xyzw"]
    dot = sum(a * b for a, b in zip(expected_engine, actual, strict=True))
    return 2.0 * math.acos(min(1.0, abs(dot)))


def joint_errors(source_qpos: list[float], source_qvel: list[float], target: dict) -> dict:
    rows = target["actuator_rows"]
    assert len(rows) == 14
    assert sorted(row["source_dof"] for row in rows) == list(range(6, 20))
    position = [abs(row["post_position_rad"] - source_qpos[row["source_dof"] + 1])
                for row in rows]
    velocity = [abs(row["post_velocity_rad_s"] - source_qvel[row["source_dof"]])
                for row in rows]
    return {"max_position_abs_error_rad": max(position),
            "max_velocity_abs_error_rad_s": max(velocity)}


def source_contact_rows(step: dict, continuous: bool) -> list[dict]:
    if continuous:
        return step["pre_step_dynamics_retained_after_mj_step"]["contacts"]
    return step["contacts"]


def source_normal_force(contact: dict, continuous: bool) -> float:
    return (contact["normal_force_N"] if continuous else
            contact["contact_force_frame_6d_N_Nm"][0])


def source_point(contact: dict, continuous: bool) -> list[float]:
    return contact["position_world_m"] if continuous else contact["pos_world_m"]


def positive_target_pair(target: dict) -> dict:
    positive = [pair for pair in target["contact_pairs"]
                if pair["normal_impulse_sum_ns"] > 1.0e-10]
    if len(positive) != 1 or set(positive[0]["source_geom_ids_or_platform"]) != {"platform", 28}:
        raise ValueError("target positive contact is not exclusively the left foot/platform")
    if any(abs(pair["normal_impulse_sum_ns"]) > 1.0e-10
           or abs(pair["tangent_impulse_sum_abs_ns"]) > 1.0e-10
           for pair in target["contact_pairs"] if pair is not positive[0]):
        raise ValueError("target has nonzero contact impulses outside left foot/platform")
    return positive[0]


def contact_step(source: dict, target: dict, continuous: bool) -> dict:
    contacts = source_contact_rows(source, continuous)
    if not contacts or any(row["geom_names"] !=
                           ["source_left_foot_contact_platform", "left_foot_collision"]
                           for row in contacts):
        raise ValueError("source contacts are not exclusively left foot/platform")
    dt = 1.0 / 60.0
    source_normal = sum(source_normal_force(row, continuous) for row in contacts) * dt
    pair = positive_target_pair(target)
    target_normal = pair["normal_impulse_sum_ns"]
    source_centroid_x = (sum(source_point(row, continuous)[0] *
                             source_normal_force(row, continuous) for row in contacts) /
                         sum(source_normal_force(row, continuous) for row in contacts))
    target_points = [point for manifold in pair["manifolds"]
                     for point in manifold["contacts"] if point["normal_impulse_ns"] > 0.0]
    target_centroid_x = (sum(point["anchor1_engine"][0] * point["normal_impulse_ns"]
                             for point in target_points) / target_normal)
    source_qpos = (source["post_step_integrated_state"]["qpos"] if continuous
                   else source["qpos"])
    source_qvel = (source["post_step_integrated_state"]["qvel"] if continuous
                   else source["qvel"])
    source_qfrc = (source["pre_step_dynamics_retained_after_mj_step"]
                   ["qfrc_constraint_20_dof_N_Nm"] if continuous
                   else source["qfrc_constraint"])
    source_root_impulse = [value * dt for value in
                           [source_qfrc[0], source_qfrc[2], -source_qfrc[1]]]
    target_root_impulse = [a + b for a, b in zip(
        target["all_dof_contact_normal_impulse_nms"][:3],
        target["all_dof_contact_tangent_impulse_nms"][:3], strict=True)]
    return {
        "source_contact_count": len(contacts),
        "target_positive_pair_count": 1,
        "target_solver_contact_count": pair["solver_contact_count"],
        "source_normal_impulse_ns": source_normal,
        "target_normal_impulse_ns": target_normal,
        "normal_impulse_signed_error_ns": target_normal - source_normal,
        "normal_impulse_relative_error": abs(target_normal - source_normal) / source_normal,
        "source_weighted_contact_x_m": source_centroid_x,
        "target_weighted_contact_x_m": target_centroid_x,
        "weighted_contact_x_separation_m": abs(target_centroid_x - source_centroid_x),
        "source_root_contact_impulse_engine_nms": source_root_impulse,
        "target_root_contact_impulse_engine_nms": target_root_impulse,
        "root_contact_impulse_signed_error_engine_nms": [a-b for a,b in zip(
            target_root_impulse, source_root_impulse, strict=True)],
        "root_position": root_position_error(source_qpos, target),
        "root_orientation_abs_error_rad": root_rotation_error(source_qpos, target),
        "joints": joint_errors(source_qpos, source_qvel, target),
    }


def bam_candidate_rows(source_step: dict, target: dict) -> dict:
    dt = target["step_configuration"]["dt"]
    assert len(source_step["qfrc_bias"]) == len(source_step["qfrc_constraint"]) == 20
    rows = []
    for target_row in target["actuator_rows"]:
        dof = target_row["source_dof"]
        source_external = (-source_step["qfrc_bias"][dof] +
                           source_step["qfrc_constraint"][dof])
        native_candidate = (
            target_row["gravity_projection_nm"] -
            target_row["inertial_projection_nm"] +
            (target_row["generic_joint_impulse_nms"] -
             target_row["own_dry_friction_impulse_nms"] +
             target_row["contact_normal_impulse_nms"] +
             target_row["contact_tangent_impulse_nms"]) / dt)
        rows.append({"source_dof": dof, "source_joint": target_row["source_joint"],
                     "source_bam_previous_external_load_nm": source_external,
                     "native_previous_solve_candidate_nm": native_candidate,
                     "signed_error_nm": native_candidate - source_external})
    assert len(rows) == 14 and all(row["source_dof"] == index + 6
                                   for index, row in enumerate(rows))
    worst = max(rows, key=lambda row: abs(row["signed_error_nm"]))
    return {"rows": rows, "max_abs_error_nm": abs(worst["signed_error_nm"]),
            "worst_source_dof": worst["source_dof"],
            "source_dof_frictionloss_all_zero": True,
            "qualified_for_BAM": False}


def main() -> None:
    source_one_path = SOURCE / "source_contact_report.json"
    source_eight_path = SOURCE / "continuous8_source_trajectory.json"
    target_one_path = TARGET / "one_v3_report.json"
    target_eight_path = TARGET / "eight_v3_report.json"
    source_free_path = TARGET / "source_free_negative.json"
    target_free_path = TARGET / "far_v3_report.json"
    source_one = read(source_one_path, EXPECTED_SOURCE_ONE_SHA)
    source_eight = read(source_eight_path, EXPECTED_SOURCE_EIGHT_SHA)
    target_one = read(target_one_path, EXPECTED_TARGET_ONE_SHA)
    target_eight = read(target_eight_path, EXPECTED_TARGET_EIGHT_SHA)
    source_free = read(source_free_path)
    target_free = read(target_free_path, EXPECTED_TARGET_FAR_SHA)
    target_definition = read(ROOT / ".scratch/source_limit_probe_v1/leg_enriched_with_hulls.json",
                             EXPECTED_TARGET_MODEL_SHA)
    assert all(value == 0.0 for value in target_definition["fields"]["dof_frictionloss"])
    assert source_one["identity"]["passes"]
    assert source_one["identity"]["frozen_source_exact_model_array_count"] == 468
    assert source_one["outputs_sha256"]["platform_mjb"] == EXPECTED_SOURCE_MODEL_SHA
    assert source_one["outputs_sha256"]["neutral_leg_qpos_json"] == EXPECTED_NEUTRAL_QPOS_SHA
    assert source_eight["inputs_sha256"]["source_left_foot_platform_mjb"] == EXPECTED_SOURCE_MODEL_SHA
    assert source_eight["inputs_sha256"]["neutral_leg_qpos_json"] == EXPECTED_NEUTRAL_QPOS_SHA
    assert source_eight["MjData_instances"] == 1 and source_eight["mj_step_calls"] == 8
    assert source_eight["intermediate_mj_forward_calls"] == 0
    assert source_one["mujoco_version"] == source_eight["mujoco_version"] == "3.10.0"
    for sample in source_one["samples"][:2]:
        assert sample["nefc"] == 4 * sample["ncon"]
        assert all(kind == 6 for contact in sample["contacts"]
                   for kind in contact["efc_type"])
    for source_row in source_eight["steps"]:
        dynamics = source_row["pre_step_dynamics_retained_after_mj_step"]
        assert dynamics["nefc"] == 4 * dynamics["ncon"]
    assert target_one["model_sha256"] == target_eight["model_sha256"] == EXPECTED_TARGET_MODEL_SHA
    assert target_one["qpos_sha256"] == target_eight["qpos_sha256"] == EXPECTED_NEUTRAL_QPOS_SHA
    assert target_one["physics_integrations"] == 1 and target_eight["physics_integrations"] == 8
    assert target_eight["world_count"] == 1 and len(target_eight["trace"]) == 8
    assert target_one["internal_pgs_iterations"] == target_eight["internal_pgs_iterations"] == 1
    assert target_one["step_configuration"]["num_solver_iterations"] == 1
    assert target_eight["step_configuration"]["num_solver_iterations"] == 1
    assert target_one["platform"] == target_eight["platform"]
    assert target_one["platform"]["source_center"] == source_one["platform"]["center_m"]
    assert target_one["platform"]["source_half_extents"] == source_one["platform"]["half_size_m"]
    assert target_one["platform"]["sliding_friction"] == source_one["platform"]["friction"][0]
    assert source_free["ncon"] == 0 and target_free["qpos_sha256"] == EXPECTED_NEUTRAL_QPOS_SHA
    assert not any(pair["platform_pair"] for pair in target_free["contact_pairs"])
    assert max(map(abs, target_free["all_dof_contact_normal_impulse_nms"])) == 0.0
    free_error = root_position_error(source_free["qpos"], target_free)
    single = contact_step(source_one["samples"][1], target_one, False)
    bam_candidates = bam_candidate_rows(source_one["samples"][1], target_one)
    continuous = []
    for index, (source_row, target_row) in enumerate(zip(
            source_eight["steps"], target_eight["trace"], strict=True), start=1):
        assert source_row["step_index"] == index - 1
        assert target_row["step_index"] == index
        result = contact_step(source_row, target_row, True)
        result["step"] = index
        continuous.append(result)
    assert single["source_normal_impulse_ns"] == continuous[0]["source_normal_impulse_ns"]
    assert single["target_normal_impulse_ns"] == continuous[0]["target_normal_impulse_ns"]
    assert free_error["max_abs_error_m"] < 1.0e-6
    assert single["normal_impulse_relative_error"] > 0.05
    assert single["weighted_contact_x_separation_m"] > 0.01
    assert single["root_position"]["max_abs_error_m"] > 1.0e-4
    assert bam_candidates["max_abs_error_nm"] > 0.05
    assert continuous[-1]["root_position"]["max_abs_error_m"] > 0.005
    result = {
        "schema": "original_left_foot_platform_source_target_contact_comparison_v1",
        "scope": "diagnostic_blocker_not_contact_or_BAM_or_skill_qualification",
        "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            source_one_path, source_eight_path, target_one_path, target_eight_path,
            source_free_path, target_free_path)},
        "comparison_script_sha256": sha(Path(__file__)),
        "frozen_source_platform_mjb_sha256": EXPECTED_SOURCE_MODEL_SHA,
        "frozen_target_robot_definition_sha256": EXPECTED_TARGET_MODEL_SHA,
        "neutral_qpos_sha256": EXPECTED_NEUTRAL_QPOS_SHA,
        "no_platform_free_fall_root_error": free_error,
        "single_step": single,
        "single_step_bam_previous_load_candidates": bam_candidates,
        "continuous_eight_steps": continuous,
        "max_continuous_root_position_error_m": max(
            row["root_position"]["max_abs_error_m"] for row in continuous),
        "max_continuous_normal_impulse_error_ns": max(
            abs(row["normal_impulse_signed_error_ns"]) for row in continuous),
        "contact_equivalent": False,
        "bam_external_load_qualified": False,
        "skill_qualified": False,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print("comparison", OUT, sha(OUT))
    print("free root max", free_error["max_abs_error_m"])
    print("contact step 1", single["normal_impulse_relative_error"],
          single["weighted_contact_x_separation_m"],
          single["root_position"]["max_abs_error_m"])
    print("contact step 8 root max", continuous[-1]["root_position"]["max_abs_error_m"])


if __name__ == "__main__":
    main()
