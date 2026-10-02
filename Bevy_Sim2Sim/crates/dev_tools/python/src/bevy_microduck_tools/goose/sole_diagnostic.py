"""Guided physical foot and six spring pads, real ground, one 20ms integration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import implicit_metric
from .artifacts import DT, sha256, write_json

PROGRAMS = ("load_5n_per_patch", "load_10n_per_patch", "load_20n_per_patch",
            "load_30n_per_patch", "nominal_half_weight", "release_half_weight", "impact_half_weight")
TICKS = 100


def values(items):
    return " ".join(f"{v:.17g}" for v in items)


def inertial(element, body):
    tensor = np.asarray(body["inertia_at_com_body_kg_m2"])
    ET.SubElement(element, "inertial", mass=str(body["mass_kg"]), pos=values(body["com_local_m"]),
                  fullinertia=values([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                     tensor[0, 1], tensor[0, 2], tensor[1, 2]]))


def fixture(plant, path, impact):
    bodies = {b["name"]: b for b in plant["bodies"]}
    foot = bodies["right_ankle_roll"]
    pads = [j for j in plant["joints"] if j["parent"] == foot["name"] and j["kind"] == "slide"]
    if len(pads) != 6 or foot["rotation_world_wxyz"] != [1, 0, 0, 0]:
        raise ValueError("Requires six original pads and identity foot rotation")
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 0", integrator="implicit",
                  iterations="100", tolerance="1e-12")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="ground", type="box", size="1 1 0.1", pos="0 0 -0.1",
                  friction="0 0 0", contype="1", conaffinity="2", condim="1",
                  solref="0.005 1", solimp="0.95 0.99 0.001 0.5 2")
    position = np.asarray(foot["translation_world_m"]) + [0, 0, .01 if impact else 0]
    parent = ET.SubElement(world, "body", name=foot["name"], pos=values(position))
    inertial(parent, foot)
    ET.SubElement(parent, "joint", name="foot_vertical", type="slide", axis="0 0 1",
                  limited="false", stiffness="0", damping="0", armature="0", frictionloss="0")
    for joint in pads:
        body = bodies[joint["child"]]
        geometry = next(g for g in plant["colliders"] if g["body"] == body["name"])
        if (body["rotation_world_wxyz"] != [1, 0, 0, 0] or joint["axis_world"] != [0, 0, 1]
                or geometry["local_position_m"] != [0, 0, 0]
                or geometry["local_rotation_wxyz"] != [1, 0, 0, 0]):
            raise ValueError("Changed pad transform not supported")
        element = ET.SubElement(parent, "body", name=body["name"],
                                pos=values(np.asarray(body["translation_world_m"])-foot["translation_world_m"]))
        inertial(element, body)
        ET.SubElement(element, "joint", name=joint["name"], type="slide", axis="0 0 1",
                      range=values(joint["range"]), stiffness=str(joint["stiffness_n_m"]),
                      damping=str(joint["damping"]), armature=str(joint["armature"]),
                      frictionloss=str(joint["frictionloss"]), springref="0", limited="true")
        ET.SubElement(element, "geom", name=geometry["name"], type="box", size=values(geometry["half_extents_m"]),
                      friction="0 0 0", contype="2", conaffinity="1", condim="1",
                      solref="0.005 1", solimp="0.95 0.99 0.001 0.5 2")
    physical = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    ledger = implicit_metric.prepare_xml(root, physical)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(path)), ledger, pads, bodies


def load(program, tick, mass):
    if program.startswith("load_"):
        return 6*float(program.removeprefix("load_").removesuffix("n_per_patch"))
    if program == "release_half_weight" and tick >= 50:
        return 0.
    return mass*9.81/2


def one_case(model, plant, pads, bodies, program):
    data = mujoco.MjData(model)
    if program == "impact_half_weight":
        data.qvel[0] = -.5
    rows, walls, reason = [], [], None
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll")
    patch_ids = [(j, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j["name"]),
                  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, j["child"])) for j in pads]
    for tick in range(TICKS):
        downward = load(program, tick, plant["robot_mass_kg"])
        data.qfrc_applied[:] = 0
        data.qfrc_applied[0] = -downward
        started = time.perf_counter()
        try:
            implicit_metric.native_step(model, data)
        except (RuntimeError, mujoco.FatalError) as exc:
            reason = str(exc)
            break
        walls.append((time.perf_counter()-started)*1000)
        contacts = {j["child"]: {"distance": None, "impulse": 0.} for j in pads}
        for index, contact in enumerate(data.contact):
            if ground not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == ground else contact.geom1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other]))
            entry = contacts[name]
            entry["distance"] = float(contact.dist) if entry["distance"] is None else min(entry["distance"], float(contact.dist))
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            entry["impulse"] += float(force[0])*DT
        # Refresh only post-integration geometry; do not solve or integrate again.
        mujoco.mj_kinematics(model, data)
        pad_rows = []
        for joint, jid, bid in patch_ids:
            q = float(data.qpos[model.jnt_qposadr[jid]])
            v = float(data.qvel[model.jnt_dofadr[jid]])
            geom = next(g for g in plant["colliders"] if g["body"] == joint["child"])
            bottom = float(data.xpos[bid, 2])-geom["half_extents_m"][2]
            entry = contacts[joint["child"]]
            pad_rows.append({"patch": joint["child"], "compression_m": q, "relative_velocity_m_s": v,
                             "limit_violation_m": max(0., joint["range"][0]-q, q-joint["range"][1]),
                             "post_integration_bottom_m": bottom, "pre_integration_solver_distance_m": entry["distance"],
                             "normal_impulse_n_s": entry["impulse"], "physical_spring_force_n": joint["stiffness_n_m"]*q})
        warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
        rows.append({"tick": tick, "time_s": float(data.time), "integration_count": tick+1,
                     "downward_load_n": downward, "foot_vertical_velocity_m_s": float(data.qvel[0]),
                     "foot_origin_height_m": float(data.xpos[foot, 2]), "pads": pad_rows, "warnings": warnings})
        if warnings or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason = "nonfinite_or_native_warning"
            break
    static_rows = rows[75:] if program.startswith("load_") or program == "nominal_half_weight" else []
    return {"program": program, "integrations": len(rows), "failure_reason": reason,
            "physical_mass_kg": bodies["right_ankle_roll"]["mass_kg"] + sum(bodies[j["child"]]["mass_kg"] for j in pads),
            "max_post_integration_penetration_m": max((max(0., -p["post_integration_bottom_m"]) for r in rows for p in r["pads"]), default=0.),
            "max_limit_violation_m": max((p["limit_violation_m"] for r in rows for p in r["pads"]), default=0.),
            "static_total_spring_load_relative_error": max((abs(sum(p["physical_spring_force_n"] for p in r["pads"])-r["downward_load_n"])/r["downward_load_n"] for r in static_rows), default=0.),
            "physics_p95_ms": float(np.percentile(walls, 95)), "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plant_path, output = args.plant.resolve(), args.output.resolve()
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("Preserve prior evidence; use a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    plant = json.loads(plant_path.read_text())
    if plant["candidate_id"] != "goose_460_full50_v1":
        raise ValueError("Use original physical ledger")
    started = time.perf_counter()
    cases, ledgers = [], []
    for impact in (False, True):
        path = output.parent/("impact.xml" if impact else "static.xml")
        model, ledger, pads, bodies = fixture(plant, path, impact)
        ledgers.append({"impact": impact, "model_sha256": sha256(path), "numerical_metric": ledger})
        cases.extend(one_case(model, plant, pads, bodies, name) for name in PROGRAMS
                     if (name == "impact_half_weight") == impact)
    receipt = {"schema": "goose_guided_sole_source_diagnostic_v1", "engine": "mujoco_cpu", "engine_version": mujoco.__version__,
               "plant_sha256": sha256(plant_path), "physics_dt_s": DT, "integrations_per_tick": 1,
               "substeps": 0, "coordinate_writes_after_initialization": 0, "qualified": False, "m0_passed": False,
               "optimizer_updates": 0, "inference_count": 0, "programs": PROGRAMS, "ticks_per_program": TICKS,
               "fixture_scope": "original physical right foot + six pads; vertical guide; no other foot geometry",
               "external_load": "downward generalized force on physical foot; zero gravity; no added physical mass",
               "contact_law": {"normal": "MuJoCo native compliant contact", "solref": [.005, 1],
                               "solimp": [.95, .99, .001, .5, 2], "friction": 0, "condim": 1},
               "initial_impact_gap_added_m": .01, "initial_impact_velocity_m_s": -.5,
               "cases": cases, "model_ledgers": ledgers,
               "code_sha256": {str(p): sha256(p) for p in (Path(__file__).resolve(), Path(implicit_metric.__file__).resolve())},
               "elapsed_wall_s": time.perf_counter()-started,
               "limitations": ["no whole-body qualification", "native contact laws differ", "no foot rotation, friction or gait qualification"]}
    write_json(output, receipt)
    print(f"STATUS DIAGNOSTIC_ONLY cases={len(cases)} ticks={sum(c['integrations'] for c in cases)} {output}", flush=True)


if __name__ == "__main__":
    main()
