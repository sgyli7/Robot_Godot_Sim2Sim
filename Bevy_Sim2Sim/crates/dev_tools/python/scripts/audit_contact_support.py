"""Audit which original left-foot hull features each contact solver selected.

Run with the pinned upstream MuJoCo Python environment. This is a zero-step
geometry check against the SHA-bound original platform fixture and native
Rapier report; it does not alter the source model or qualify contact dynamics.
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / ".scratch/contact_platform_source_v1"
TARGET = ROOT / ".scratch/contact_platform_target_v1"
MODEL = ROOT / ".scratch/source_limit_probe_v1/leg_enriched_with_hulls.json"
MJB = SOURCE / "source_left_foot_platform.mjb"
QPOS = SOURCE / "neutral_leg_qpos.json"
SOURCE_REPORT = SOURCE / "source_contact_report.json"
TARGET_REPORT = TARGET / "one_v3_report.json"
OUTPUT = TARGET / "support_geometry_audit.json"
EXPECTED = {
    MODEL: "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb",
    MJB: "90ba3d2783616604a3ccb32f889343d854feb2f3317dfbcb39141bdf0362096c",
    QPOS: "09a9879e8647d51933ea8d0b2cbb97137167c54d666fd372b6aa1d590798b848",
    SOURCE_REPORT: "af2035d28d31013237816c6e65c2ee07d6d5cf62e106c955263da42f97224bf7",
    TARGET_REPORT: "1520378ce7457f904434503de6ce50a3112257c86ee348d0caad56631a84d0c4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path):
    actual = sha(path)
    if actual != EXPECTED[path]:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}")
    return json.loads(path.read_bytes())


def closest_horizontal(hull_world: np.ndarray, point_world: np.ndarray) -> tuple[int, float]:
    distances = np.linalg.norm(hull_world[:, :2] - point_world[:2], axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def main() -> None:
    fields = checked(MODEL)["fields"]
    qpos = np.asarray(checked(QPOS), dtype=np.float64)
    source_report = checked(SOURCE_REPORT)
    target_report = checked(TARGET_REPORT)
    if sha(MJB) != EXPECTED[MJB]:
        raise ValueError("source platform MJB SHA256 mismatch")
    model = mujoco.MjModel.from_binary_path(str(MJB))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    if geom != 29 or fields["geom_dataid"][geom - 1] != int(model.geom_dataid[geom]):
        raise ValueError("left-foot source geom identity changed")
    for key in ("mesh_vert", "mesh_graph", "mesh_vertadr", "mesh_vertnum",
                "mesh_pos", "mesh_quat", "mesh_scale", "geom_pos", "geom_quat",
                "geom_dataid", "geom_size", "geom_type", "geom_friction"):
        source = np.asarray(getattr(model, key))[1:] if key.startswith("geom_") else np.asarray(getattr(model, key))
        target = np.asarray(fields[key])
        if not np.array_equal(source, target):
            raise ValueError(f"source/target original collision input differs: {key}")
    if (not np.array_equal(model.dof_frictionloss, np.asarray(fields["dof_frictionloss"]))
            or np.any(model.dof_frictionloss[6:20] != 0.0)):
        raise ValueError("source/target driven DOF dry-friction inputs differ or are nonzero")
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_graphadr[mesh])
    hull_count = int(model.mesh_graph[start])
    hull_ids = model.mesh_graph[start + 2 + hull_count:start + 2 + 2 * hull_count]
    mesh_vertices = model.mesh_vert[
        model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    hull_world = (mesh_vertices[hull_ids] @ data.geom_xmat[geom].reshape(3, 3).T
                  + data.geom_xpos[geom])
    source_contacts = source_report["samples"][1]["contacts"]
    pair = [row for row in target_report["contact_pairs"]
            if row["normal_impulse_sum_ns"] > 1.0e-10]
    if len(source_contacts) != 2 or len(pair) != 1 or pair[0]["source_geom_ids_or_platform"] != ["platform", 28]:
        raise ValueError("original left-foot/platform contact identity changed")
    source_rows = []
    for contact in source_contacts:
        point = np.asarray(contact["pos_world_m"], dtype=np.float64)
        index, xy_error = closest_horizontal(hull_world, point)
        if xy_error > 1.0e-8:
            raise ValueError("source contact is not aligned with original hull vertex")
        source_rows.append({"contact_world_xyz_m": point.tolist(), "hull_vertex_world_xyz_m": hull_world[index].tolist(),
                            "hull_vertex_index": index, "horizontal_error_m": xy_error,
                            "penetration_m": -contact["dist_m"]})
    target_rows = []
    for contact in pair[0]["manifolds"][0]["contacts"]:
        anchor = contact["anchor1_engine"]
        point = np.asarray([anchor[0], -anchor[2], anchor[1]], dtype=np.float64)
        index, xy_error = closest_horizontal(hull_world, point)
        target_rows.append({"platform_anchor_source_xyz_m": point.tolist(),
                            "nearest_hull_vertex_world_xyz_m": hull_world[index].tolist(),
                            "nearest_hull_vertex_index": index,
                            "horizontal_error_m": xy_error,
                            "normal_impulse_ns": contact["normal_impulse_ns"],
                            "initial_distance_m": contact["initial_distance_m"]})
    dominant = max(target_rows, key=lambda row: row["normal_impulse_ns"])
    if dominant["horizontal_error_m"] > 1.0e-6 or dominant["normal_impulse_ns"] < .05:
        raise ValueError("dominant target contact no longer lies on source hull support")
    result = {
        "schema": "original_left_foot_contact_support_geometry_audit_v1",
        "scope": "zero_step_original_hull_feature_selection_not_contact_equivalence",
        "inputs_sha256": {str(path.relative_to(ROOT)): expected for path, expected in EXPECTED.items()},
        "script_sha256": sha(Path(__file__)),
        "mujoco_version": mujoco.__version__,
        "source_target_collision_input_arrays_exact": True,
        "source_target_driven_dof_frictionloss_exact_zero": True,
        "original_mesh_id": mesh,
        "original_mesh_vertex_count": int(model.mesh_vertnum[mesh]),
        "original_native_hull_vertex_count": hull_count,
        "deepest_original_hull_vertex_source_xyz_m": hull_world[int(np.argmin(hull_world[:, 2]))].tolist(),
        "source_contact_support": source_rows,
        "target_solver_contact_support": target_rows,
        "dominant_target_support": dominant,
        "contact_feature_selection_equivalent": False,
    }
    OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("support audit", OUTPUT, sha(OUTPUT))
    print("dominant target/source support x", dominant["platform_anchor_source_xyz_m"][0],
          source_rows[0]["contact_world_xyz_m"][0])


if __name__ == "__main__":
    main()
