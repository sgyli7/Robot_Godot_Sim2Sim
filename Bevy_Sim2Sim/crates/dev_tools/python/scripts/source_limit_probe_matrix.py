"""SHA-bound, diagnostic-only 42-case native source-limit row comparison.

Run from Bevy_Sim2Sim after building paired_force_probe with
sim2sim_source_limit_probe and sim2sim_plain_mass_probe. It combines SHA-bound
source-limit and native-hull scratch exports into temporary definitions; it
never writes production model definitions.
"""

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / ".scratch/source_limit_probe_v1"
BIN = ROOT / "target/debug/paired_force_probe"
EXPECTED = {
    "leg": {
        "enriched_source_sha": "27c8cf0b4a16a86429a0ae8b163f553fbf9649911539caaea1cd035134545054",
        "native_hull_source_sha": "eb7a276a5199c757181c60681de54a7eb3e490431929d1fb043f8fb25a964299",
        "definition_sha": "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb",
        "matrix_sha": "c30cfd248770f4eb761672616f6417ead877337624f590f8602459475b730a9e",
        "base_qpos_sha": "e16f8e7ddf1d1cf2061b4c760faa1cbf78536d1cd8e6b992ca415eef10ae21bf",
    },
    "roller": {
        "enriched_source_sha": "9cd4cbb9275e8e0c5399165f5e2bb10df0918354188065680554575006c2e533",
        "native_hull_source_sha": "a4524cb0b63a64fd88f3fff1375c8fac375d2ec88254545409ca6b6eaca768fd",
        "definition_sha": "6f88aa286e487c031250b427595a640b9f597604c9a20dcbca6fd69c49109c40",
        "matrix_sha": "48b23a63c220f9713f38a022a9254d920aad67fd2411ca87119f83fd970f027d",
        "base_qpos_sha": "df50ef2c31243ef01c6419f18238008c9a2d8d3e444073b82c61cb27b3fb7773",
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    actual = sha(data)
    assert actual == expected, (path, actual, expected)
    return data


def write_json(path: Path, value: object) -> str:
    data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    path.write_bytes(data)
    return sha(data)


def materialize_definition(family: str, identity: dict[str, str]) -> Path:
    enriched_path = ROOT / f".scratch/limit_export_probe_v1/{family}_enriched_compiled.json"
    hull_path = ROOT / f".scratch/root_native_hull_import_v1/{family}_compiled_with_native_hulls.json"
    enriched = json.loads(checked(enriched_path, identity["enriched_source_sha"]))
    hull = json.loads(checked(hull_path, identity["native_hull_source_sha"]))
    for key in (
        "bam", "counts", "family", "home", "joint_order", "missing_optional_fields",
        "names", "per_step_mutable_fields", "schema", "units",
    ):
        assert enriched[key] == hull[key], (family, key)
    for key, value in enriched["fields"].items():
        if key not in ("jnt_solref", "jnt_solimp", "jnt_margin", "dof_invweight0"):
            assert value == hull["fields"].get(key), (family, key)
    enriched["fields"]["mesh_graphadr"] = hull["fields"]["mesh_graphadr"]
    enriched["fields"]["mesh_graph"] = hull["fields"]["mesh_graph"]
    result = OUT / f"{family}_enriched_with_hulls.json"
    data = (json.dumps(enriched, indent=2) + "\n").encode()
    assert sha(data) == identity["definition_sha"], family
    result.write_bytes(data)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases_out = []
    for family, identity in EXPECTED.items():
        definition = materialize_definition(family, identity)
        matrix = ROOT / f".scratch/paired_force_v1/{family}_limit_matrix.json"
        base_qpos = ROOT / f".scratch/paired_force_v1/{family}_head_roll_limit_qpos.json"
        checked(definition, identity["definition_sha"])
        source = json.loads(checked(matrix, identity["matrix_sha"]))
        qpos_base = json.loads(checked(base_qpos, identity["base_qpos_sha"]))
        model = json.loads(definition.read_bytes())
        source_joint = source["source_joint"]
        source_dof = source["source_dof"]
        qpos_index = model["fields"]["jnt_qposadr"][source_joint]
        assert model["names"]["jnt"][source_joint] == "head_roll"
        assert source_dof == model["fields"]["jnt_dofadr"][source_joint]
        assert len(source["cases"]) == 21
        for case_id, case in enumerate(source["cases"]):
            depth = case["depth_from_upper_rad"]
            velocity = case["initial_joint_velocity_rad_s"]
            qpos = qpos_base.copy()
            qpos[qpos_index] = source["upper_limit_rad"] + depth
            qvel = [0.0] * model["counts"]["nv"]
            qvel[source_dof] = velocity
            qpos_path = OUT / f"{family}_{case_id:02d}_qpos.json"
            qvel_path = OUT / f"{family}_{case_id:02d}_qvel.json"
            report_path = OUT / f"{family}_{case_id:02d}_report.json"
            qpos_sha = write_json(qpos_path, qpos)
            qvel_sha = write_json(qvel_path, qvel)
            subprocess.run(
                [
                    str(BIN),
                    "--head-roll-source-limit-diagnostic",
                    str(definition),
                    identity["definition_sha"],
                    str(qpos_path),
                    qpos_sha,
                    str(qvel_path),
                    qvel_sha,
                    str(report_path),
                ],
                check=True,
                cwd=ROOT,
            )
            report_bytes = report_path.read_bytes()
            report = json.loads(report_bytes)
            assert report["model_file_sha256"] == identity["definition_sha"], (family, case_id)
            assert report["qpos_file_sha256"] == qpos_sha, (family, case_id)
            assert report["qvel_file_sha256"] == qvel_sha, (family, case_id)
            assert report["source_limit_probe_selected"] is True, (family, case_id)
            assert report["source_limit_probe_joint"] == "head_roll", (family, case_id)
            assert report["physics_integrations"] == 1, (family, case_id)
            assert abs(report["actual_step_dt_seconds"] - 1.0 / 60.0) < 1.0e-8
            assert all(pair["normal_impulse_sum"] == 0.0 for pair in report["active_contact_pairs"])
            assert all(
                row["contact_normal_impulse_nms"] == 0.0
                and row["contact_tangent_impulse_nms"] == 0.0
                and (row["source_joint"] == source_joint or row["generic_joint_impulse_nms"] == 0.0)
                for row in report["rows"]
            ), (family, case_id)
            row = next(r for r in report["rows"] if r["source_joint"] == source_joint)
            assert abs(row["initial_position"] - qpos[qpos_index]) < 1.0e-6
            assert abs(row["initial_velocity"] - velocity) < 1.0e-6
            traces = [r for r in row["limit_row_timing"] if r["phase"] == "after_unbiased_solve"]
            assert len(traces) == 1, (family, case_id, traces)
            trace = traces[0]
            native_active = trace["impulse_bounds"][1] > 0.0
            source_active = case["row"] is not None
            native_force = trace["impulse_nms"] / report["actual_step_dt_seconds"]
            source_force = -case["solve_constraint_nm"]
            cases_out.append(
                {
                    "family": family,
                    "case_id": case_id,
                    "depth_rad": depth,
                    "initial_velocity_rad_s": velocity,
                    "source_row_active": source_active,
                    "native_row_active": native_active,
                    "source_force_nm": source_force,
                    "native_force_nm": native_force,
                    "force_abs_error_nm": abs(native_force - source_force),
                    "source_force_positive": source_force > 0.0,
                    "native_force_positive": native_force > 0.0,
                    "source_post_position_rad": case["post_joint_position_rad"],
                    "native_post_position_rad": row["post_position"],
                    "position_abs_error_rad": abs(row["post_position"] - case["post_joint_position_rad"]),
                    "source_post_velocity_rad_s": case["post_joint_velocity_rad_s"],
                    "native_post_velocity_rad_s": row["post_velocity"],
                    "velocity_abs_error_rad_s": abs(row["post_velocity"] - case["post_joint_velocity_rad_s"]),
                    "native_final_impulse_nms": trace["impulse_nms"],
                    "native_rhs_rad_s": trace["rhs"],
                    "native_rhs_without_bias_rad_s": trace["rhs_without_bias"],
                    "native_report_sha256": sha(report_bytes),
                    "qpos_sha256": qpos_sha,
                    "qvel_sha256": qvel_sha,
                }
            )
            print(family, case_id, depth, velocity, "force_error", abs(native_force - source_force), flush=True)

    assert len(cases_out) == 42
    assert all(case["source_row_active"] == case["native_row_active"] for case in cases_out)
    assert all(case["source_force_positive"] == case["native_force_positive"] for case in cases_out)
    assert max(case["force_abs_error_nm"] for case in cases_out) < 1.0e-6
    assert max(case["position_abs_error_rad"] for case in cases_out) < 1.0e-6
    assert max(case["velocity_abs_error_rad_s"] for case in cases_out) < 1.0e-5
    summary = {
        "scope": "diagnostic_only_frozen_head_roll_single_limit_matrix",
        "feature": "sim2sim_source_limit_probe",
        "explicit_cli_opt_in": "--head-roll-source-limit-diagnostic",
        "frozen_inputs": EXPECTED,
        "cases": cases_out,
        "row_activation_mismatches": sum(c["source_row_active"] != c["native_row_active"] for c in cases_out),
        "force_sign_mismatches": sum(c["source_force_positive"] != c["native_force_positive"] for c in cases_out),
        "max_force_abs_error_nm": max(c["force_abs_error_nm"] for c in cases_out),
        "max_position_abs_error_rad": max(c["position_abs_error_rad"] for c in cases_out),
        "max_velocity_abs_error_rad_s": max(c["velocity_abs_error_rad_s"] for c in cases_out),
        "bam_external_load_qualified": False,
        "source_target_equivalent": False,
    }
    result = OUT / "matrix_result.json"
    write_json(result, summary)
    print("result", result, sha(result.read_bytes()), flush=True)


if __name__ == "__main__":
    main()
