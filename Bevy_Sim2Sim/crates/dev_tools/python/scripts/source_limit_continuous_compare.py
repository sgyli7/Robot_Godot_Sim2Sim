"""Frozen MuJoCo versus one same-world native eight-step limit probe.

Requires MuJoCo 3.10.0, the frozen .mjb files, and the feature-gated native
source_limit_continuous_probe binary. No SourceCollisionWorld budget is used.
"""

import json
import subprocess
from pathlib import Path

import mujoco

from source_limit_probe_matrix import EXPECTED, ROOT, checked, materialize_definition, sha, write_json


OUT = ROOT / ".scratch/source_limit_continuous_v1"
BIN = ROOT / "target/debug/source_limit_continuous_probe"
MJB_SHA = {
    "leg": "832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c",
    "roller": "ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191",
}
STEPS = 8


def source_steps(family: str, qpos: list[float]) -> list[dict]:
    mjb = ROOT / f".scratch/root_robot_assembly_mujoco_v1/{family}_native_model.mjb"
    checked(mjb, MJB_SHA[family])
    model = mujoco.MjModel.from_binary_path(str(mjb))
    assert mujoco.mj_versionString() == "3.10.0"
    assert abs(model.opt.timestep - 1.0 / 60.0) < 1.0e-12
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.iterations == 100 and model.opt.tolerance == 1.0e-8
    assert model.opt.disableflags == 0
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_roll")
    qidx = int(model.jnt_qposadr[joint])
    didx = int(model.jnt_dofadr[joint])
    assert len(qpos) == model.nq
    assert abs(qpos[qidx] - (model.jnt_range[joint, 1] + 0.02)) < 1.0e-10
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    rows = []
    for step in range(1, STEPS + 1):
        pre_q = float(data.qpos[qidx])
        pre_v = float(data.qvel[didx])
        mujoco.mj_step(model, data)
        constraint_force = float(data.qfrc_constraint[didx])
        row_active = any(
            int(data.efc_id[index]) == joint
            and int(data.efc_type[index]) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
            for index in range(data.nefc)
        )
        rows.append(
            {
                "step": step,
                "pre_position_rad": pre_q,
                "pre_velocity_rad_s": pre_v,
                "post_position_rad": float(data.qpos[qidx]),
                "post_velocity_rad_s": float(data.qvel[didx]),
                "limit_row_active": row_active,
                "limit_signed_generalized_force_nm": constraint_force,
                "limit_signed_generalized_impulse_nms": constraint_force * model.opt.timestep,
                "source_ncon": int(data.ncon),
                "source_nefc": int(data.nefc),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert BIN.exists()
    families = []
    for family, identity in EXPECTED.items():
        definition = materialize_definition(family, identity)
        model = json.loads(definition.read_bytes())
        base_qpos = ROOT / f".scratch/paired_force_v1/{family}_head_roll_limit_qpos.json"
        qpos = json.loads(checked(base_qpos, identity["base_qpos_sha"]))
        qvel = [0.0] * model["counts"]["nv"]
        qvel_file = OUT / f"{family}_zero_qvel.json"
        qvel_sha = write_json(qvel_file, qvel)
        native_file = OUT / f"{family}_native_8step.json"
        subprocess.run(
            [
                str(BIN),
                "--source-limit-continuous-diagnostic",
                str(definition),
                identity["definition_sha"],
                str(base_qpos),
                identity["base_qpos_sha"],
                str(qvel_file),
                qvel_sha,
                str(native_file),
            ],
            cwd=ROOT,
            check=True,
        )
        native_bytes = native_file.read_bytes()
        native = json.loads(native_bytes)
        assert native["model_file_sha256"] == identity["definition_sha"]
        assert native["qpos_file_sha256"] == identity["base_qpos_sha"]
        assert native["qvel_file_sha256"] == qvel_sha
        assert native["assembly_builds"] == 1
        assert native["physics_integrations"] == STEPS
        assert [row["integration_count"] for row in native["rows"]] == list(range(1, STEPS + 1))
        source = source_steps(family, qpos)
        compared = []
        for src, dst in zip(source, native["rows"], strict=True):
            assert src["step"] == dst["step"]
            compared.append(
                {
                    "step": src["step"],
                    "source": src,
                    "native": dst,
                    "row_activation_matches": src["limit_row_active"] == dst["limit_row_active"],
                    "pre_position_abs_error_rad": abs(src["pre_position_rad"] - dst["pre_position_rad"]),
                    "pre_velocity_abs_error_rad_s": abs(src["pre_velocity_rad_s"] - dst["pre_velocity_rad_s"]),
                    "post_position_abs_error_rad": abs(src["post_position_rad"] - dst["post_position_rad"]),
                    "post_velocity_abs_error_rad_s": abs(src["post_velocity_rad_s"] - dst["post_velocity_rad_s"]),
                    "signed_force_abs_error_nm": abs(
                        src["limit_signed_generalized_force_nm"]
                        - dst["limit_row_signed_generalized_force_nm"]
                    ),
                    "native_contact_impulses_zero": (
                        dst["all_dof_contact_normal_max_abs_nms"] == 0.0
                        and dst["all_dof_contact_tangent_max_abs_nms"] == 0.0
                        and all(pair["normal_impulse_sum_nms"] == 0.0 for pair in dst["active_contact_pairs"])
                    ),
                    "contact_pair_count_matches_source_ncon": (
                        dst["active_contact_pair_count"] == src["source_ncon"]
                    ),
                }
            )
        families.append(
            {
                "family": family,
                "frozen_mjb_sha256": MJB_SHA[family],
                "native_definition_sha256": identity["definition_sha"],
                "initial_qpos_sha256": identity["base_qpos_sha"],
                "zero_qvel_sha256": qvel_sha,
                "native_report_sha256": sha(native_bytes),
                "max_post_position_abs_error_rad": max(c["post_position_abs_error_rad"] for c in compared),
                "max_post_velocity_abs_error_rad_s": max(c["post_velocity_abs_error_rad_s"] for c in compared),
                "max_signed_force_abs_error_nm": max(c["signed_force_abs_error_nm"] for c in compared),
                "row_activation_mismatches": sum(not c["row_activation_matches"] for c in compared),
                "nonzero_native_contact_steps": sum(not c["native_contact_impulses_zero"] for c in compared),
                "contact_topology_mismatch_steps": sum(
                    not c["contact_pair_count_matches_source_ncon"] for c in compared
                ),
                "source_contact_steps": sum(c["source"]["source_ncon"] > 0 for c in compared),
                "steps": compared,
            }
        )
        assert families[-1]["row_activation_mismatches"] == 0
        assert families[-1]["nonzero_native_contact_steps"] == 0
        assert families[-1]["source_contact_steps"] == 0
        assert families[-1]["max_signed_force_abs_error_nm"] < 1.0e-6
        assert families[-1]["max_post_position_abs_error_rad"] < 1.0e-6
        assert families[-1]["max_post_velocity_abs_error_rad_s"] < 1.0e-5
        print(family, families[-1]["max_post_velocity_abs_error_rad_s"], flush=True)
    result = {
        "scope": "same_world_eight_step_head_roll_source_limit_comparison",
        "mujoco_version": mujoco.mj_versionString(),
        "steps_per_family": STEPS,
        "families": families,
        "bam_external_load_qualified": False,
        "source_target_equivalent": False,
    }
    path = OUT / "continuous_comparison.json"
    print("result", path, write_json(path, result), flush=True)


if __name__ == "__main__":
    main()
