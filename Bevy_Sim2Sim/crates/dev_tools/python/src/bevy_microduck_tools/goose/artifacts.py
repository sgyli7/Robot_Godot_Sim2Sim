"""Immutable Goose checkpoint adoption and versioned 50 Hz MuJoCo candidates.

Candidate generation is deliberately separate from qualification. In particular,
condensing a pad is a zero-pose mass reduction, not high-frequency equivalence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

DT = 0.02
CANDIDATES = ("goose_460_full50_v1", "goose_460_condensed50_v1")
JOINT_ORDER = (
    "neck_yaw", "neck_pitch", "neck_mid_pitch", "head_pitch", "head_roll", "beak_hinge",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def adopt_checkpoint(source_root: Path) -> tuple[dict, dict]:
    source_root = source_root.resolve(strict=True)
    model = source_root / "robots/Goose_V0.1/models/training_checkpoint/robot.xml"
    contract_path = source_root / "robots/Goose_V0.1/configs/training_checkpoint_contract.json"
    manifest_path = source_root / "delivery_manifest.json"
    contract = json.loads(contract_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    verified = {}
    declared = dict(manifest["files"])
    for name, digest in manifest.get("generated_files_sha256", {}).items():
        if name in declared and declared[name] != digest:
            raise ValueError(f"Conflicting manifest identity: {name}")
        declared[name] = digest
    for name, digest in declared.items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"Manifest path escapes source: {name}")
        actual = sha256(path)
        if actual != digest:
            mismatches.append(name)
        verified[name] = actual
    if mismatches:
        raise ValueError(f"Checkpoint hash mismatches: {mismatches[:10]}")
    if (contract["joint_order"] != list(JOINT_ORDER) or len(contract["bodies"]) != 33
            or (contract["observation_size"], contract["action_size"]) != (65, 18)
            or contract["source_part_count"] != 460):
        raise ValueError("Not the authoritative 460-part / 33-body / 18-axis Goose checkpoint")
    if sha256(model) != contract["model_sha256"] or sha256(contract_path) != manifest["contract_sha256"]:
        raise ValueError("Checkpoint model/contract identity mismatch")
    receipt = {
        "schema": "goose_checkpoint_adoption_v1", "source_root": str(source_root),
        "model_path": str(model), "contract_path": str(contract_path),
        "manifest_sha256": sha256(manifest_path), "model_sha256": sha256(model),
        "contract_sha256": sha256(contract_path), "verified_file_count": len(verified),
        "source_scene_sha256": contract["source_scene_sha256"],
        "source_mutated": False, "optimizer_updates": 0,
        "source_module_sha256": {"src/sai_agent/goose/stage_one_gravity.py": sha256(source_root / "src/sai_agent/goose/stage_one_gravity.py")},
    }
    return contract, receipt


def combine_rigid_properties(parts: list[tuple[float, np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mass, COM and full inertia for tensors already in one frame."""
    mass = sum(p[0] for p in parts)
    if mass <= 0:
        raise ValueError("Rigid aggregate must have positive mass")
    center = sum((m * c for m, c, _ in parts), start=np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for m, c, tensor in parts:
        displacement = c - center
        inertia += tensor + m * (np.eye(3) * displacement.dot(displacement) - np.outer(displacement, displacement))
    return mass, center, inertia


def _values(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _fullinertia(tensor: np.ndarray) -> str:
    return _values([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])


def build_candidate(source_root: Path, destination: Path, source_contract: dict, adoption: dict,
                    candidate: str) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown Goose candidate: {candidate}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Candidate destination is not empty; preserve the previous experiment")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(adoption["contract_path"], destination / "neutral_contract.json")
    source_model = Path(adoption["model_path"])
    tree = ET.parse(source_model)
    root = tree.getroot()
    root.set("model", candidate)
    root.find("option").set("timestep", str(DT))
    root.find("option").set("integrator", "implicit")
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("memory", "256M")
    # Hash-bound absolute references avoid copying gigabytes and never write the delivery.
    compiler = root.find("compiler")
    compiler.set("meshdir", ".")
    asset_hashes = {}
    for mesh in root.findall("./asset/mesh"):
        asset = (source_model.parent / mesh.get("file")).resolve(strict=True)
        relative = str(asset.relative_to(source_model.parent)) if asset.is_relative_to(source_model.parent) else "../" + str(asset.relative_to(source_model.parent.parent))
        expected = source_contract["asset_sha256"][relative]
        mesh.set("file", str(asset))
        asset_hashes[str(asset)] = expected
    contract = copy.deepcopy(source_contract)
    contract.update(schema="goose_50hz_candidate_si_v1", candidate=candidate,
                    status="M0_UNQUALIFIED", physics_dt_s=DT, torque_dt_s=DT, policy_dt_s=DT,
                    physics_steps_per_tick=1, torque_updates_per_tick=1, policy_calls_per_tick=1,
                    observation_sizes={"locomotion": 65, "recovery": 65, "pickup": 82},
                    source_checkpoint=adoption, asset_sha256=asset_hashes,
                    integrator="implicit", optimizer_updates=0, training_release=False)
    contract["action_semantics"] = source_contract["action_semantics"].replace("PD at 200Hz", "PD at 50Hz")
    contract["solver_capacity"] = {"memory_bytes": 256 * 1024 * 1024, "physical_parameters_changed": False, "reason": "original collision candidate overflowed default mj_stackAlloc at ncon=30898"}
    contract["collision_proxy_status"] = "not_checked; original 15715 geometry collision retained"
    if candidate == CANDIDATES[1]:
        _condense_pads(root, contract)
    model_path = destination / "robot.xml"
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode")
    contract["model_sha256"] = sha256(model_path)
    contract_path = destination / "contract.json"
    write_json(contract_path, contract)
    return {"candidate": candidate, "model_path": str(model_path), "contract_path": str(contract_path),
            "model_sha256": sha256(model_path), "contract_sha256": sha256(contract_path),
            "source_contract_sha256": adoption["contract_sha256"], "body_count": len(contract["bodies"]),
            "collision_geometry_count": len(contract["collision_geometries"]),
            "timing": {"physics_dt_s": DT, "torque_dt_s": DT, "policy_dt_s": DT, "substeps": 1}}


def _condense_pads(root: ET.Element, contract: dict) -> None:
    bodies = {body["name"]: body for body in contract["bodies"]}
    elements = {body.get("name"): body for body in root.iter("body")}
    condensed = []
    for side in ("right", "left"):
        parent_name = side + "_ankle_roll"
        parent = elements[parent_name]
        parent_body = bodies[parent_name]
        parts = [(parent_body["mass_kg"], np.array(parent_body["com_local_m"]), np.array(parent_body["inertia_at_com_body_kg_m2"]))]
        for patch in contract["passive_contacts"]:
            if patch["body"] != parent_name:
                continue
            element = elements[patch["name"]]
            local = np.fromstring(element.get("pos"), sep=" ")
            pad = bodies[patch["name"]]
            parts.append((pad["mass_kg"], local + np.array(pad["com_local_m"]), np.array(pad["inertia_at_com_body_kg_m2"])))
            geom = copy.deepcopy(element.find("geom"))
            geom.set("pos", _values(local))
            # Direct-format MuJoCo reference is an uncalibrated implementation of
            # published k/c, recorded explicitly; measured probes decide acceptance.
            geom.set("solref", _values([-patch["stiffness_n_m"], -patch["damping_n_s_m"]]))
            parent.append(geom)
            parent.remove(element)
            patch.update(implementation="condensed_compliant_contact", contact_solref_n_m_n_s_m=[-patch["stiffness_n_m"], -patch["damping_n_s_m"]],
                         high_frequency_equivalence=False, travel_enforcement="not_checked; penetration rejection at 1.5mm")
            condensed.append(patch["name"])
        mass, center, tensor = combine_rigid_properties(parts)
        parent_body.update(mass_kg=mass, com_local_m=center.tolist(), inertia_at_com_body_kg_m2=tensor.tolist())
        inertial = parent.find("inertial")
        inertial.attrib.clear()
        inertial.set("mass", format(mass, ".17g"))
        inertial.set("pos", _values(center))
        inertial.set("fullinertia", _fullinertia(tensor))
    contract["bodies"] = [body for body in contract["bodies"] if body["name"] not in condensed]
    contract["passive_axes"] = 2
    contract["condensation"] = {"removed_pad_bodies": condensed, "preserves": ["closed-pose total mass", "closed-pose COM", "closed-pose full inertia", "six contact boxes per foot", "support hull"],
                                "travel_m": .0015, "dynamic_equivalence": "not_checked", "high_frequency_equivalence": False,
                                "parameters_fitted_to_probe": False}


def export_rapier_plant(bundle: dict, destination: Path) -> dict:
    """Export compiled collision frames and complete rigid tensors, never visuals."""
    import mujoco
    from .native_geometry import (
        ENGINE_VERSION, EXPORT_REVISION, collision_mesh_vertices, require_supported_engine)
    require_supported_engine()
    contract_path = Path(bundle["contract_path"])
    contract = json.loads(contract_path.read_text())
    experimental = contract.get("candidate") == "goose_460_full50_be_v2"
    physical_joint_ledger = {}
    if experimental:
        if (contract.get("schema") != "goose_50hz_experimental_si_v2"
                or contract.get("numerical_metric", {}).get("revision") != "goose_joint_backward_euler_predictive_v1"
                or contract.get("training_release") is not False):
            raise ValueError("Experimental physical export requires the explicit unqualified v2 contract")
        physical_joint_ledger = {j["joint"]: j for j in contract["numerical_metric"]["joints"]}
    model = mujoco.MjModel.from_xml_path(bundle["model_path"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = []
    for entry in contract["bodies"]:
        bid = model.body(entry["name"]).id
        parent = int(model.body_parentid[bid])
        bodies.append({**{k: entry[k] for k in ("name", "mass_kg", "com_local_m", "inertia_at_com_body_kg_m2")},
                       "parent": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) if parent else None,
                       "translation_world_m": data.xpos[bid].tolist(), "rotation_world_wxyz": data.xquat[bid].tolist()})
    feedback = {j["name"]: index for index, j in enumerate(contract["joints"])}
    active = {j.get("actuation_joint", j["name"]): index for index, j in enumerate(contract["joints"])}
    joints = []
    for jid in range(model.njnt):
        kind = int(model.jnt_type[jid])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        bid = int(model.jnt_bodyid[jid])
        parent = int(model.body_parentid[bid])
        did = int(model.jnt_dofadr[jid])
        record = {"name": name, "parent": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
                  "child": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
                  "kind": "slide" if kind == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge",
                  "origin_world_m": data.xanchor[jid].tolist(), "axis_world": data.xaxis[jid].tolist(),
                  "range": model.jnt_range[jid].tolist(), "stiffness_n_m": float(model.jnt_stiffness[jid]),
                  "damping": float(model.dof_damping[did]), "armature": float(model.dof_armature[did]),
                  "frictionloss": float(model.dof_frictionloss[did]), "driven": name in active}
        if experimental:
            ledger = physical_joint_ledger[name]
            if (did != ledger["dof"] or record["armature"] != ledger["compiled_armature"]
                    or record["damping"] != ledger["compiled_damping"]
                    or record["stiffness_n_m"] != ledger["physical_stiffness"]):
                raise ValueError("Experimental numerical coefficients differ from frozen physical ledger")
            # Source compilation uses A+h*C+h²*K and C+h*K. The target native
            # solver already integrates the physical spring/damping implicitly.
            # Exporting compiled coefficients would double their numerical effect.
            record.update(armature=ledger["physical_armature"], damping=ledger["physical_damping"],
                          range=ledger["physical_range"])
        if name in active:
            record["active_axis"] = active[name]
        if name in feedback:
            record["feedback_axis"] = feedback[name]
        joints.append(record)
    patches = {p["name"]: p for p in contract["passive_contacts"]}
    collider_names = {g["name"] for g in contract["collision_geometries"]} | set(patches)
    colliders = []
    reduced_mesh_count = 0
    max_frame_error = 0.
    world_lower, world_upper = np.full(3, np.inf), np.full(3, -np.inf)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name not in collider_names:
            continue
        bid = int(model.geom_bodyid[gid])
        record = {"name": name, "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
                  "local_position_m": model.geom_pos[gid].tolist(), "local_rotation_wxyz": model.geom_quat[gid].tolist(),
                  "friction": float(model.geom_friction[gid, 0]), "friction_coefficients": model.geom_friction[gid].tolist(),
                  "contype": int(model.geom_contype[gid]), "conaffinity": int(model.geom_conaffinity[gid]),
                  "solref": model.geom_solref[gid].tolist(), "solimp": model.geom_solimp[gid].tolist()}
        if model.geom_type[gid] == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[gid])
            vertices, support = collision_mesh_vertices(model, mesh_id)
            reduced_mesh_count += support["support_vertex_count"] < support["raw_vertex_count"]
            record.update(kind="convex_mesh", vertices_local_m=vertices.tolist(),
                          native_support=support)
        elif model.geom_type[gid] == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = model.geom_size[gid]
            vertices = np.array([[x, y, z] for x in (-extent[0], extent[0]) for y in (-extent[1], extent[1]) for z in (-extent[2], extent[2])])
            record.update(kind="box", half_extents_m=extent.tolist())
        else:
            raise ValueError(f"Unsupported collision geometry {name}")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        exported_rotation = data.xmat[bid].reshape(3, 3) @ rotation.reshape(3, 3)
        exported_origin = data.xmat[bid].reshape(3, 3) @ model.geom_pos[gid] + data.xpos[bid]
        max_frame_error = max(max_frame_error, float(np.max(np.abs(exported_origin-data.geom_xpos[gid]))),
                              float(np.max(np.abs(exported_rotation-data.geom_xmat[gid].reshape(3,3)))))
        if max_frame_error > 1e-10:
            raise ValueError("Compiled mesh local/body geometry transform mismatch")
        transformed = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        world_lower = np.minimum(world_lower, transformed.min(axis=0))
        world_upper = np.maximum(world_upper, transformed.max(axis=0))
        if name in patches:
            record["contact_patch"] = name
            record["contact_parameters"] = patches[name]
        colliders.append(record)
    if len(colliders) != len(collider_names):
        raise ValueError("Compiled collision export incomplete")
    transmission = contract["beak_transmission"]
    phase, radius = transmission["closed_crank_angle_in_xz_rad"], transmission["crank_radius_m"]
    pin = np.array(transmission["jaw_axis_world_m"]) + radius * np.array([np.cos(phase), 0., np.sin(phase)])
    raw_xml = ET.parse(bundle["model_path"]).getroot()
    exclusions = [[e.get("body1"), e.get("body2")] for e in raw_xml.findall("./contact/exclude")]
    plant = {"schema": "goose_plant50_v1", "candidate_id": bundle["candidate"],
             "neutral_contract_sha256": bundle["source_contract_sha256"], "derived_contract_sha256": sha256(contract_path),
             "model_sha256": bundle["model_sha256"], "robot_mass_kg": sum(b["mass_kg"] for b in bodies), "physics_hz": 50,
             "joint_order": list(JOINT_ORDER), "bodies": bodies, "joints": joints, "colliders": colliders,
             "exclusions": exclusions, "adjacent_body_filter": "MuJoCo excludes parent-child except parent=world",
             "jaw_loop": {"parent_body": "head_roll", "rotor_body": "beak_input_rotor", "coupler_body": "beak_coupler_link", "jaw_body": "beak_hinge",
                          "output_pin_world_m": pin.tolist(), "rotation_axis_world": [0., 1., 0.]},
             "geometry_export": {"authority": "native MuJoCo collision support points + compiled geom local transforms",
                                 "revision": EXPORT_REVISION, "engine_version": ENGINE_VERSION,
                                 "reduced_mesh_count": reduced_mesh_count,
                                 "visual_meshes_included": False, "collider_count": len(colliders),
                                 "world_bounds_m": [world_lower.tolist(), world_upper.tolist()],
                                 "vertices_are_compiler_centered": True, "mesh_scale_baked_by_compiler": True,
                                 "exported_vs_compiled_frame_max_error": max_frame_error}}
    if experimental:
        plant.update(schema="goose_plant50_experimental_v2",
                     numerical_experiment={"source_method": contract["numerical_metric"]["revision"],
                                           "predictive_scalar_stops": True, "jaw_pin_basis": "coupler_axis"},
                     source_numerical_metric=contract["numerical_metric"], qualified=False)
    write_json(destination, plant)
    return {"plant_path": str(destination), "plant_sha256": sha256(destination), "collider_count": len(colliders),
            "world_bounds_m": plant["geometry_export"]["world_bounds_m"]}
