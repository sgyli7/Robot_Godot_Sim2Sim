"""Exact native visual data beside a frozen definition; never edits that definition.

Run with the pinned Pollen source Python environment. Only CPU MuJoCo compilation
and mj_forward are used; no environment integration, policy or learning executes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevy_microduck_tools.source_adapter import load_source, make_skill_cfg
from bevy_microduck_tools.serialization import sha256_file


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")


def xml_geoms(path):
    tree = ET.parse(path).getroot()
    defaults = {}
    def collect(node, parent="main", inherited=None):
        name = node.get("class", "main")
        attrs = dict(inherited or {})
        own = node.find("geom")
        if own is not None:
            attrs.update(own.attrib)
        defaults[name] = {"parent": parent, "geom_explicit": {} if own is None else dict(own.attrib),
                          "geom_inherited_explicit": attrs}
        for child in node.findall("default"):
            collect(child, name, attrs)
    for node in tree.findall("default"):
        collect(node)
    geoms = []
    def visit(body, inherited_class):
        childclass = body.get("childclass", inherited_class)
        for geom in body.findall("geom"):
            classname = geom.get("class", childclass)
            geoms.append({"body_name": body.get("name"), "attributes": dict(geom.attrib),
                          "resolved_default_class": classname,
                          "default_evidence": defaults.get(classname)})
        for child in body.findall("body"):
            visit(child, childclass)
    world = tree.find("worldbody")
    assert world is not None
    for body in world.findall("body"):
        visit(body, "main")
    return geoms, defaults


def export(source_root, definitions, family, poses_path, output):
    import mujoco
    import numpy as np
    source = load_source(source_root)
    cfg, adoption = make_skill_cfg(source, "roller" if family == "roller_allcollisions" else "standing")
    entity = cfg.scene.entities["robot"].build()
    entity.spec.option.timestep = cfg.sim.mujoco.timestep
    model = entity.spec.compile()
    if model.ntex != 0 or np.any(model.mat_texid >= 0) or model.nmeshtexcoord != 0:
        raise ValueError("Visual schema omits UV/textures; native texture/UV presence must not be discarded")
    identities = []
    for path in definitions:
        target = json.loads(path.read_text())
        if target["family"] != family:
            raise ValueError("Definition family differs from original native family")
        for key, count in target["counts"].items():
            if int(getattr(model, key)) != count:
                raise ValueError(f"Native count differs: {key}")
        for key, value in target["fields"].items():
            if not hasattr(model, key):
                raise ValueError(f"Native field missing: {key}")
            np.testing.assert_array_equal(np.asarray(value), np.asarray(getattr(model, key)), err_msg=key)
        if not {"mesh_graphadr", "mesh_graph"} <= target["fields"].keys():
            raise ValueError("Target definition must include actual native collision graphs")
        identities.append({"path": str(path.resolve()), "sha256": sha256_file(path),
                           "field_count": len(target["fields"]), "counts": target["counts"],
                           "all_native_fields_exact": True})
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML, MICRODUCK_ALLCOLLISIONS_ROLLERS_XML
    original_xml = MICRODUCK_ALLCOLLISIONS_ROLLERS_XML if family == "roller_allcollisions" else MICRODUCK_ALLCOLLISIONS_XML
    original_geoms, defaults = xml_geoms(original_xml)
    assert len(original_geoms) == model.ngeom
    inventory = []
    for geom_id, original in enumerate(original_geoms):
        native_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else None
        native_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        assert original["body_name"] == native_body
        assert original["attributes"].get("mesh") == mesh_name
        assert original["attributes"].get("name") == native_name
        inventory.append({"geom_id": geom_id, "native_name": native_name, "body_name": native_body,
                          "mesh_id": mesh_id, "mesh_name": mesh_name, "group": int(model.geom_group[geom_id]),
                          "rgba": model.geom_rgba[geom_id].tolist(), "matid": int(model.geom_matid[geom_id]),
                          "contype": int(model.geom_contype[geom_id]), "conaffinity": int(model.geom_conaffinity[geom_id]),
                          "source": original})
    meshes = []
    mesh_receipt = []
    for mesh_id in range(model.nmesh):
        normal_address, normal_count = int(model.mesh_normaladr[mesh_id]), int(model.mesh_normalnum[mesh_id])
        face_address, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        normals = model.mesh_normal[normal_address:normal_address+normal_count]
        face_normals = model.mesh_facenormal[face_address:face_address+face_count]
        assert len(normals) == normal_count and len(face_normals) == face_count
        assert face_count > 0 and normal_count > 0
        assert np.all(face_normals >= 0) and np.all(face_normals < normal_count)
        assert np.isfinite(normals).all()
        meshes.append({"mesh_id": mesh_id, "normals": normals.tolist(), "face_normals": face_normals.tolist()})
        mesh_receipt.append({"mesh_id": mesh_id, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
            "normal_address": normal_address, "normal_count": normal_count, "face_address": face_address,
            "face_count": face_count, "face_normal_indices_are_local": True,
            "uv_count": int(model.mesh_texcoordnum[mesh_id]),
            "normals_raw_sha256": hashlib.sha256(normals.tobytes()).hexdigest(),
            "face_normals_raw_sha256": hashlib.sha256(face_normals.tobytes()).hexdigest()})
    materials = [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i),
        "rgba": model.mat_rgba[i].tolist(),
        **{name: float(getattr(model, "mat_"+name)[i]) for name in
           ("emission", "specular", "shininess", "reflectance", "metallic", "roughness")}}
        for i in range(model.nmat)]
    visible_ids = [i for i in range(model.ngeom) if model.geom_group[i] == 2]
    assert visible_ids and all(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in visible_ids)
    # Publish no sidecar until every requested target definition and source mapping passed.
    binary = output / "native_model.mjb"
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    with binary.open("xb") as stream:
        stream.write(buffer.tobytes())
    sidecars = []
    for i, definition in enumerate(identities):
        schema = {"schema": "microduck_visual_v1", "model_file_sha256": definition["sha256"],
            "visible_geom_group": 2, "visible_geom_ids": visible_ids, "texture_count": 0,
            "geom_group": model.geom_group.tolist(), "geom_rgba": model.geom_rgba.tolist(),
            "geom_matid": model.geom_matid.tolist(), "materials": materials, "meshes": meshes}
        path = output / ("visual.json" if i == 0 else f"visual_definition_{i}.json")
        write_new(path, schema)
        sidecars.append({"path": str(path.resolve()), "sha256": sha256_file(path), "definition": definition})
    poses = json.loads(poses_path.read_text())
    assert len(poses["cases"]) == 2
    reference_path = output / "world_vertices.json.gz"
    with reference_path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            import io
            stream = io.TextIOWrapper(zipped, encoding="utf-8")
            stream.write('{"schema":"mujoco_visible_world_vertices_v1","cases":[')
            for case_index, case in enumerate(poses["cases"]):
                data = mujoco.MjData(model)
                qpos = np.asarray(case["qpos_source"], dtype=np.float64)
                assert qpos.shape == (model.nq,)
                data.qpos[:] = qpos
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(data.qpos, qpos)
                assert data.time == 0
                if case_index:
                    stream.write(',')
                name = case.get("name", case.get("label", f"case_{case_index}"))
                stream.write('{"name":'+json.dumps(name)+',"qpos_source":'+json.dumps(qpos.tolist())+',"geoms":[')
                for visible_index, geom_id in enumerate(visible_ids):
                    mesh_id = int(model.geom_dataid[geom_id])
                    vertex_address, vertex_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
                    face_address, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
                    vertices = model.mesh_vert[vertex_address:vertex_address+vertex_count].astype(np.float64)
                    faces = model.mesh_face[face_address:face_address+face_count]
                    assert np.all(faces >= 0) and np.all(faces < vertex_count)
                    rotation = data.geom_xmat[geom_id].reshape(3,3)
                    world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
                    assert np.isfinite(world_vertices).all()
                    if visible_index:
                        stream.write(',')
                    json.dump({"geom_id": geom_id, "mesh_id": mesh_id,
                        "world_position": data.geom_xpos[geom_id].tolist(), "world_rotation_rowmajor": data.geom_xmat[geom_id].tolist(),
                        "source_vertex_count": vertex_count, "source_face_count": face_count,
                        "source_faces_local": faces.tolist(), "world_vertices_in_original_mesh_vertex_order": world_vertices.tolist()},
                        stream, allow_nan=False, separators=(',',':'))
                stream.write(']}')
            stream.write(']}\n')
            stream.flush()
            stream.detach()
    receipt = {"schema": "microduck_visual_export_receipt_v1", "family": family,
        "source_commit": source["commit"], "source_root": str(source_root.resolve()),
        "source_files": source["files"], "source_adapter_sha256": sha256_file(Path(__file__).resolve().parents[1]/"src/bevy_microduck_tools/source_adapter.py"),
        "tool": {"path":str(Path(__file__).resolve()), "sha256":sha256_file(Path(__file__))},
        "native_model": {"path": str(binary.resolve()), "sha256": sha256_file(binary)},
        "mujoco_version": mujoco.__version__, "effective_adoption_sha256": adoption["sha256"],
        "source_xml": {"path": str(original_xml), "sha256":sha256_file(original_xml)},
        "sidecars": sidecars, "original_default_classes": defaults, "geoms": inventory,
        "source_class_counts": dict(Counter(item["source"]["resolved_default_class"] for item in inventory)),
        "group_counts": dict(Counter(str(value) for value in model.geom_group.tolist())),
        "material_texture_ids": model.mat_texid.tolist(), "texture_count": model.ntex,
        "native_mesh_normal_count": model.nmeshnormal, "native_mesh_uv_count": model.nmeshtexcoord,
        "mesh_semantics": mesh_receipt, "visible_geom_count": len(visible_ids),
        "world_vertex_reference": {"path": str(reference_path.resolve()), "sha256": sha256_file(reference_path),
            "pose_input": str(poses_path.resolve()), "pose_input_sha256":sha256_file(poses_path), "cases":2,
            "method":"original native compile -> fresh MjData exact input qpos and zero qvel -> mj_forward -> geom_xmat/geom_xpos applied to ALL original mesh vertices, ALL original faces retained; no truncation",
            "integration_count":0, "actor_inference_count":0, "actuator_compute_count":0,
            "frame":"SI, source Z-up; native mesh preprocessing already applied; double world-coordinate calculations",
            "root_pose_note":"qpos follows supplied original root fixture, including its free-root home height; not a new standing-height definition"}}
    write_new(output / "receipt.json", receipt)
    return {"status":"exported_exact_native_visual", "visible_geoms":len(visible_ids), "source_class_counts":receipt["source_class_counts"],
            "sidecars": sidecars, "receipt":str((output/"receipt.json").resolve()), "world_vertices":receipt["world_vertex_reference"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--definition", type=Path, action="append", required=True)
    parser.add_argument("--family", choices=("leg_allcollisions","roller_allcollisions"), required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = export(args.source, args.definition, args.family, args.poses, args.output)
    except BaseException as error:
        write_new(args.output/"failure.json", {"status":"failed", "error_type":type(error).__name__, "error":str(error),
                   "definitions":[str(path.resolve()) for path in args.definition], "no_tolerance_relaxation":True})
        raise
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
