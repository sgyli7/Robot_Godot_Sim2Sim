"""Run bounded real CPU physics probes; produce evidence, never grant M0 alone.

Usage: python -m bevy_microduck_tools.goose.preflight --source-root EXTRACT
       --output BACKUP. All generated models and traces stay outside the repo.
"""
from __future__ import annotations

import argparse
import json
import platform
import os
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import (CANDIDATES, DT, adopt_checkpoint, build_candidate,
                              export_rapier_plant, sha256, write_json)
from .runtime import GooseSourceRuntime, RUNTIME_REVISION


def warning_counts(data):
    return {mujoco.mjtWarning(index).name: int(w.number) for index, w in enumerate(data.warning) if w.number}


def _rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def global_properties(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = float(model.body_mass.sum())
    first = (model.body_mass[:, None] * data.xipos).sum(axis=0)
    inertia = np.zeros((3, 3))
    for bid in range(1, model.nbody):
        rotation = data.xmat[bid].reshape(3, 3) @ _rotation(model.body_iquat[bid])
        position = data.xipos[bid]
        inertia += rotation @ np.diag(model.body_inertia[bid]) @ rotation.T
        inertia += model.body_mass[bid] * (np.eye(3) * position.dot(position) - np.outer(position, position))
    return {"mass_kg": mass, "first_moment_kg_m": first.tolist(), "com_world_m": (first/mass).tolist(),
            "inertia_world_origin_kg_m2": inertia.tolist()}


def inertia_probe(runtime, original_properties):
    measured = global_properties(runtime.model)
    errors = {"mass_kg": abs(measured["mass_kg"] - original_properties["mass_kg"]),
              "com_m": float(np.max(np.abs(np.array(measured["com_world_m"]) - original_properties["com_world_m"]))),
              "inertia_frobenius_kg_m2": float(np.linalg.norm(np.array(measured["inertia_world_origin_kg_m2"]) - original_properties["inertia_world_origin_kg_m2"]))}
    body_checks = []
    for body in runtime.contract["bodies"]:
        bid = runtime.model.body(body["name"]).id
        rotation = _rotation(runtime.model.body_iquat[bid])
        tensor = rotation @ np.diag(runtime.model.body_inertia[bid]) @ rotation.T
        expected = np.asarray(body["inertia_at_com_body_kg_m2"])
        body_checks.append({"body": body["name"],
                            "mass_error_kg": abs(float(runtime.model.body_mass[bid])-body["mass_kg"]),
                            "com_error_m": float(np.max(np.abs(runtime.model.body_ipos[bid]-body["com_local_m"]))),
                            "tensor_relative_error": float(np.linalg.norm(tensor-expected)/np.linalg.norm(expected))})
    passed = (errors["mass_kg"] <= 1e-10 and errors["com_m"] <= 1e-10 and errors["inertia_frobenius_kg_m2"] <= 1e-7
              and all(c["mass_error_kg"] <= 1e-10 and c["com_error_m"] <= 1e-10 and c["tensor_relative_error"] <= 1e-6 for c in body_checks))
    return {"status": "passed" if passed else "failed", "reference": original_properties, "measured": measured, "errors": errors,
            "scope": "closed-pose zero-coordinate mass/COM/full inertia; not moving pad dynamics", "per_body": body_checks}


def cold_reset_probe(runtime, destination, resets=20, ticks=100):
    rows = []
    times = []
    for reset in range(resets):
        runtime.reset()
        warnings = {}
        reason = None
        max_speed = max_penetration = max_pad_compression = 0.
        initial_time = float(runtime.data.time)
        reset_started = time.perf_counter()
        for tick in range(ticks):
            started = time.perf_counter()
            try:
                _, state = runtime.step(np.zeros(18))
            except (RuntimeError, FloatingPointError, mujoco.FatalError) as exc:
                reason = str(exc)
                warnings = warning_counts(runtime.data)
                break
            times.append(time.perf_counter() - started)
            warnings = warning_counts(runtime.data)
            max_speed = max(max_speed, float(np.max(np.abs(runtime.data.qvel))))
            max_penetration = max(max_penetration, max((-float(c.dist) for c in runtime.data.contact), default=0.))
            for patch in runtime.contract["passive_contacts"]:
                if patch.get("implementation") != "condensed_compliant_contact":
                    max_pad_compression = max(max_pad_compression, float(runtime.data.qpos[int(runtime.model.joint(patch["joint"]).qposadr[0])]))
            if warnings:
                reason = "solver_warning"
                break
            if time.perf_counter() - reset_started >= 5.:
                reason = "bounded_probe_wall_budget_5s_per_reset"
                break
        rows.append({"reset": reset, "ticks_completed": runtime.physics_integrations,
                     "controller_updates": runtime.controller_updates, "elapsed_sim_s": float(runtime.data.time) - initial_time,
                     "warnings": warnings, "failure_reason": reason, "max_abs_qvel": max_speed,
                     "max_contact_penetration_m": max_penetration, "max_pad_compression_m": max_pad_compression,
                     "final_qpos": runtime.data.qpos.tolist(), "final_qvel": runtime.data.qvel.tolist()})
    write_json(destination, {"schema": "goose_cold_reset_trace_v1", "rows": rows})
    complete = all(row["ticks_completed"] == ticks and not row["failure_reason"] for row in rows)
    timing_ok = all(row["controller_updates"] == row["ticks_completed"] and
                    abs(row["elapsed_sim_s"] - row["ticks_completed"] * DT) < 1e-9 for row in rows)
    return {"status": "passed" if complete and timing_ok else "failed", "reset_count": resets,
            "requested_ticks_per_reset": ticks, "max_wall_s_per_reset": 5., "completed_integrations": sum(r["ticks_completed"] for r in rows),
            "timing_matches_one_20ms_integration": timing_ok, "trace": str(destination), "trace_sha256": sha256(destination),
            "physics_plus_controller_p95_ms": float(np.percentile(times, 95) * 1000) if times else None,
            "failure_reasons": sorted({r["failure_reason"] for r in rows if r["failure_reason"]}),
            "qualification_limits": "finite + warning checks only; energy/escape/performance need separate acceptance"}


def axis_probe(runtime):
    checks = []
    for index, spec in enumerate(runtime.contract["joints"]):
        runtime.reset()
        action = np.zeros(18)
        action[index] = .1
        try:
            _, state = runtime.step(action)
        except (RuntimeError, FloatingPointError, mujoco.FatalError) as exc:
            checks.append({"axis": index, "feedback_joint": spec["name"], "drive_joint": spec.get("actuation_joint", spec["name"]),
                           "actuator_mapping_matches": False, "warnings": {"probe_exception": str(exc)}, "time_s": float(runtime.data.time)})
            continue
        drive = runtime.model.joint(spec.get("actuation_joint", spec["name"])).id
        mapped = int(runtime.model.actuator_trnid[index, 0]) == drive
        warnings = warning_counts(runtime.data)
        checks.append({"axis": index, "feedback_joint": spec["name"], "drive_joint": spec.get("actuation_joint", spec["name"]),
                       "actuator_mapping_matches": mapped, "warnings": warnings, "time_s": state["time_s"],
                       "target_rad": float(runtime.target[index]), "torque_nm": float(runtime.last_tau[index])})
    return {"status": "passed" if all(c["actuator_mapping_matches"] and not c["warnings"] for c in checks) else "failed",
            "scope": "one positive tick per named axis; full range sweep not checked", "rows": checks}


def jaw_probe(runtime):
    runtime.reset()
    action = np.zeros(18)
    action[5] = .25
    rows = []
    for tick in range(25):
        try:
            runtime.step(action)
        except (RuntimeError, FloatingPointError, mujoco.FatalError) as exc:
            return {"status": "failed", "reason": str(exc), "rows": rows}
        jaw = float(runtime.data.qpos[int(runtime.model.joint("beak_hinge").qposadr[0])])
        rotor = float(runtime.data.qpos[int(runtime.model.joint("beak_input_rotor").qposadr[0])])
        coupler = float(runtime.data.qpos[int(runtime.model.joint("beak_coupler_link").qposadr[0])])
        warnings = warning_counts(runtime.data)
        rows.append({"tick": tick, "jaw_rad": jaw, "rotor_rad": rotor, "coupler_rad": coupler,
                     "angle_closure_error_rad": max(abs(rotor-jaw), abs(coupler+jaw)), "warnings": warnings})
        if warnings:
            break
    # Angle equality in MuJoCo is not itself proof of Rapier four-bar closure.
    return {"status": "passed" if len(rows) == 25 and all(not r["warnings"] and r["angle_closure_error_rad"] < .005 for r in rows) else "failed",
            "scope": "source dynamic equality constraint; torque to rotor, no post-reset coordinate writes", "rows": rows}


def pad_load_probe(contract, model_path):
    """Fixture measures actual source joint OR actual condensed contact, not a fitted algebraic curve."""
    xml = ET.parse(model_path).getroot()
    elements = {b.get("name"): b for b in xml.iter("body")}
    bodies = {b["name"]: b for b in contract["bodies"]}
    rows = []
    for patch in contract["passive_contacts"]:
        for force in (5., 10., 20., 30., patch["stiffness_n_m"] * patch["travel_m"]):
            for load_schedule in ("constant_step", "one_second_ramp"):
                root = ET.Element("mujoco")
                ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
                ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 0", integrator="implicit", iterations="100")
                world = ET.SubElement(root, "worldbody")
                condensed = patch.get("implementation") == "condensed_compliant_contact"
                if condensed:
                    # Axial fixture: parent foot cannot rotate; actual patch geometry
                    # and k/c/impedance are taken without fitting from the candidate.
                    ET.SubElement(world, "geom", name="fixture_floor", type="plane", size="1 1 .1", contype="1", conaffinity="2")
                    body = ET.SubElement(world, "body", name="loaded", pos="0 0 .00075")
                    parent = bodies[patch["body"]]
                    inertial_attributes = dict(elements[patch["body"]].find("inertial").attrib)
                    original = next(g for g in elements[patch["body"]].findall("geom") if g.get("name") == patch["name"])
                    offset = np.fromstring(original.get("pos"), sep=" ")
                    inertial_attributes["pos"] = " ".join(str(v) for v in np.array(parent["com_local_m"]) - offset)
                    ET.SubElement(body, "inertial", **inertial_attributes)
                    ET.SubElement(body, "joint", name="compression", type="slide", axis="0 0 -1")
                    original = next(g for g in elements[patch["body"]].findall("geom") if g.get("name") == patch["name"])
                    attributes = dict(original.attrib)
                    attributes.pop("pos", None)
                    ET.SubElement(body, "geom", **attributes)
                else:
                    original = elements[patch["name"]]
                    body = ET.SubElement(world, "body", name="loaded")
                    # Original pad inertia and spring/limit parameters, fixed parent.
                    ET.SubElement(body, "inertial", **original.find("inertial").attrib)
                    joint_attributes = dict(original.find("joint").attrib)
                    joint_attributes["name"] = "compression"
                    ET.SubElement(body, "joint", **joint_attributes)
                model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                data = mujoco.MjData(model)
                trace = []
                for tick in range(500):
                    data.qfrc_applied[0] = force * (min((tick + 1)*DT, 1.) if load_schedule == "one_second_ramp" else 1.)
                    mujoco.mj_step(model, data)
                    trace.append(float(data.qpos[0]))
                    if warning_counts(data) or not np.isfinite(data.qpos).all():
                        break
                expected = min(force / patch["stiffness_n_m"], patch["travel_m"])
                measured = trace[-1]
                absolute_error = abs(measured - expected)
                relative_error = absolute_error / expected
                status = "passed" if not warning_counts(data) and len(trace) == 500 and absolute_error <= .00005 and relative_error <= .1 and measured <= patch["travel_m"] + .00005 else "failed"
                rows.append({"patch": patch["name"], "implementation": "compliant_contact" if condensed else "original_slider_spring",
                             "force_n": force, "load_schedule": load_schedule, "expected_compression_m": expected, "measured_compression_m": measured,
                             "absolute_error_m": absolute_error, "relative_error": relative_error, "warnings": warning_counts(data),
                             "ticks_completed": len(trace), "max_compression_m": max(trace), "status": status,
                             "fixture": "axial fixed parent; no gravity; actual original spring or candidate contact parameters"})
    step_rows = [r for r in rows if r["load_schedule"] == "constant_step"]
    ramp_rows = [r for r in rows if r["load_schedule"] == "one_second_ramp"]
    return {"status": "passed" if all(r["status"] == "passed" for r in step_rows) else "failed", "rows": step_rows,
            "ramp_rows": ramp_rows, "ramp_diagnostic_status": "passed" if all(r["status"] == "passed" for r in ramp_rows) else "failed",
            "load_schedule_description": "separate abrupt loading and 1s linear ramp; 10s single-step simulation each",
            "absolute_compression_tolerance_m": .00005, "curve_relative_tolerance": .1,
            "parameters_fitted": False, "target_engine_comparison": "not_checked",
            "release_dissipation": "not_checked", "impact": "not_checked"}


def run_candidate(bundle, original_properties):
    directory = Path(bundle["model_path"]).parent
    if (directory / "source_receipt.json").exists():
        raise FileExistsError("Frozen source receipt already exists; use a fresh experiment directory")
    started = time.perf_counter()
    runtime = GooseSourceRuntime(Path(bundle["model_path"]), Path(bundle["contract_path"]), skill="recovery")
    checks = {
        "mass_com_full_inertia": inertia_probe(runtime, original_properties),
        "cold_resets_single_integrations": cold_reset_probe(runtime, directory / "cold_reset_trace.json"),
        "named_axes": axis_probe(runtime), "jaw_constraints": jaw_probe(runtime),
        "pad_load_curve": pad_load_probe(runtime.contract, bundle["model_path"]),
        "energy_growth": {"status": "not_checked"}, "penetration_escape": {"status": "not_checked"},
        "foot_release_dissipation": {"status": "not_checked"}, "foot_impact": {"status": "not_checked"},
        "collision_proxy": {"status": "not_checked"}, "ground_reach_payload": {"status": "not_checked"},
        "source_target_pad_comparison": {"status": "not_checked"}, "target_same_version": {"status": "not_checked"},
    }
    receipt = {"schema": "goose_m0_source_receipt_v2", "candidate": bundle["candidate"],
               "contract_sha256": bundle["contract_sha256"], "model_sha256": bundle["model_sha256"],
               "engine": "mujoco_cpu", "engine_version": mujoco.__version__, "python": sys.executable,
               "platform": platform.platform(), "timing": bundle["timing"], "checks": checks,
               "runtime_code_sha256": {str(path.resolve()): sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
               "runtime_revision": RUNTIME_REVISION, "native_disableflags": int(runtime.model.opt.disableflags),
               "native_autoreset_disabled": True,
               "pose_observation_phase": "post_integration; kinematics/COM refresh only",
               "contact_force_phase": "last native pre-integration constraint solve",
               "source_complete": all(c["status"] == "passed" for name, c in checks.items() if name not in ("source_target_pad_comparison", "target_same_version")),
               "qualified": False, "optimizer_updates": 0, "elapsed_wall_s": time.perf_counter() - started,
               "numerical_status": checks["cold_resets_single_integrations"]["status"]}
    write_json(directory / "source_receipt.json", receipt)
    return receipt


REQUIRED_M0_CHECKS = frozenset({
    "mass_com_full_inertia", "cold_resets_single_integrations", "named_axes", "jaw_constraints",
    "pad_load_curve", "energy_growth", "penetration_escape", "foot_release_dissipation", "foot_impact",
    "collision_proxy", "ground_reach_payload", "source_target_pad_comparison",
})


def require_m0_admission(candidate: dict, source: dict, target: dict) -> None:
    """Both independent engines and every check must pass before any optimizer."""
    for engine, receipt in (("mujoco_cpu", source), ("rapier", target)):
        if receipt.get("engine") != engine:
            raise ValueError("M0 requires independent source and target engine receipts")
        for field in ("candidate", "contract_sha256", "model_sha256"):
            if receipt.get(field) != candidate.get(field):
                raise ValueError(f"M0 receipt identity mismatch: {field}")
        if receipt.get("timing") != candidate.get("timing"):
            raise ValueError("M0 receipt timing mismatch")
        if not REQUIRED_M0_CHECKS <= receipt.get("checks", {}).keys():
            raise ValueError("M0 has missing/untested acceptance checks; optimizer forbidden")
        if any(c.get("status") != "passed" for c in receipt["checks"].values()):
            raise ValueError("M0 has failed or untested acceptance checks; optimizer forbidden")
        if receipt.get("source_complete" if engine == "mujoco_cpu" else "target_complete") is not True:
            raise ValueError("Incomplete M0 receipt; optimizer forbidden")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.source_root = args.source_root.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Experiment output is not empty; frozen evidence must not be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    # MuJoCo writes MUJOCO_LOG.TXT in cwd; warnings remain visible on stdout.
    os.chdir(args.output)
    source_contract, adoption = adopt_checkpoint(args.source_root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "adoption_receipt.json", adoption)
    original = mujoco.MjModel.from_xml_path(adoption["model_path"])
    original_properties = global_properties(original)
    summaries = []
    for candidate in CANDIDATES:
        bundle = build_candidate(args.source_root, args.output / candidate, source_contract, adoption, candidate)
        print(f"EVENT candidate_built {candidate}", flush=True)
        bundle.update(export_rapier_plant(bundle, Path(bundle["model_path"]).parent / "rapier_plant.json"))
        write_json(Path(bundle["model_path"]).parent / "bundle.json", bundle)
        print(f"EVENT target_plant_exported {candidate} {bundle['plant_path']}", flush=True)
        receipt = run_candidate(bundle, original_properties)
        summaries.append({**bundle, "source_receipt": str(Path(bundle["model_path"]).parent / "source_receipt.json"),
                          "numerical_status": receipt["numerical_status"], "source_complete": receipt["source_complete"], "qualified": False})
        print(f"EVENT source_probed {candidate} numerical={receipt['numerical_status']} source_complete={receipt['source_complete']}", flush=True)
        # The full original always runs first. Fallback is tested only if its
        # complete source qualification did not pass; no model parameter fitting.
        if receipt["source_complete"]:
            break
    report = {"schema": "goose_m0_run_v1", "adoption": adoption, "candidates": summaries,
              "qualified": False, "optimizer_updates": 0, "next_gate": "same-version complete MuJoCo + Rapier M0 acceptance"}
    write_json(args.output / "run_summary.json", report)
    print(f"STATUS M0_UNQUALIFIED {args.output / 'run_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
