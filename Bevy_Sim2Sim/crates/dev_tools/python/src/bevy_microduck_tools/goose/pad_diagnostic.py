"""Fixed-foot physical pad loads, release and stops; no collision qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import DT, sha256, write_json
from . import implicit_metric

PROGRAMS = ("step_5n", "step_10n", "step_20n", "step_30n", "release_30n",
            "upper_stop_40n", "lower_stop_40n", "unforced_velocity")
TICKS = 100


def program(name, tick, travel):
    """Frozen conditions shared by the target probe; only initialization writes."""
    initial_q = travel if name == "lower_stop_40n" else travel/2 if name == "unforced_velocity" else 0.
    initial_v = .1 if name == "unforced_velocity" else 0.
    if name.startswith("step_"):
        force = float(name.removeprefix("step_").removesuffix("n"))
    elif name == "release_30n":
        force = 30. if tick < 50 else 0.
    elif name == "upper_stop_40n":
        force = 40.
    elif name == "lower_stop_40n":
        force = -40.
    elif name == "unforced_velocity":
        force = 0.
    else:
        raise ValueError("Unknown pad program")
    return initial_q, initial_v, force


def fixture(patch, body, destination):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 0", integrator="implicit",
                  iterations="100", tolerance="1e-12")
    element = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=patch["name"])
    tensor = np.asarray(body["inertia_at_com_body_kg_m2"])
    values = [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]
    ET.SubElement(element, "inertial", mass=str(body["mass_kg"]),
                  pos=" ".join(map(str, body["com_local_m"])), fullinertia=" ".join(map(str, values)))
    ET.SubElement(element, "joint", name=patch["joint"], type="slide", axis="0 0 1",
                  range=f"0 {patch['travel_m']}", stiffness=str(patch["stiffness_n_m"]),
                  damping=str(patch["damping_n_s_m"]), armature="0", springref="0", limited="true")
    physical = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    ledger = implicit_metric.prepare_xml(root, physical)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(destination)), ledger


def one_case(model, patch, program_name):
    data = mujoco.MjData(model)
    travel, k, c, mass = (patch[key] for key in ("travel_m", "stiffness_n_m", "damping_n_s_m", "mass_kg"))
    q, v, _ = program(program_name, 0, travel)
    data.qpos[0], data.qvel[0] = q, v
    previous_energy = .5*mass*v*v+.5*k*q*q
    initial_energy = previous_energy
    rows, reason = [], None
    maximum_free_energy_gain = 0.
    for tick in range(TICKS):
        _, _, force = program(program_name, tick, travel)
        before_q, before_v = float(data.qpos[0]), float(data.qvel[0])
        data.qfrc_applied[0] = force
        try:
            implicit_metric.native_step(model, data)
        except (RuntimeError, mujoco.FatalError) as exc:
            reason = str(exc)
            break
        q, v = float(data.qpos[0]), float(data.qvel[0])
        energy = .5*mass*v*v+.5*k*q*q
        if not force:
            maximum_free_energy_gain = max(maximum_free_energy_gain, energy-previous_energy)
        previous_energy = energy
        expected_v = (mass*before_v+DT*(force-k*before_q))/(mass+DT*c+DT**2*k)
        unconstrained_q = before_q+DT*expected_v
        violation = max(0., -q, q-travel)
        warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
        rows.append({"tick": tick, "time_s": float(data.time), "load_n": force,
                     "compression_m": q, "velocity_m_s": v, "physical_energy_j": energy,
                     "native_generalized_stop_force_n": float(data.qfrc_constraint[0]),
                     "unconstrained_backward_euler_compression_m": unconstrained_q,
                     "physical_spring_force_n": -k*q, "physical_damping_force_n": -c*v,
                     "limit_violation_m": violation, "warnings": warnings})
        if warnings or not np.isfinite([q, v, data.qacc[0]]).all():
            reason = "nonfinite_or_native_warning"
            break
    static_rows = rows[-25:] if program_name.startswith("step_") else []
    compression_error = max((abs(r["compression_m"]-r["load_n"]/k) for r in static_rows), default=0.)
    curve_error = max((abs(k*r["compression_m"]-r["load_n"])/r["load_n"] for r in static_rows), default=0.)
    violation = max((r["limit_violation_m"] for r in rows), default=0.)
    be_error = max((abs(r["compression_m"]-r["unconstrained_backward_euler_compression_m"])
                    for r in rows if 1e-8 < r["unconstrained_backward_euler_compression_m"] < travel-1e-8), default=0.)
    passes = (reason is None and len(rows) == TICKS and compression_error <= .00005
              and curve_error <= .1 and violation <= 1e-8 and be_error <= 1e-8
              and maximum_free_energy_gain <= max(1e-10, initial_energy*1e-6))
    return {"patch": patch["name"], "program": program_name, "integrations": len(rows),
            "local_checks_passed": passes, "failure_reason": reason,
            "static_compression_error_m": compression_error, "static_load_curve_relative_error": curve_error,
            "max_limit_violation_m": violation, "max_unconstrained_be_error_m": be_error,
            "max_unforced_energy_increment_j": maximum_free_energy_gain, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract_path, output = args.contract.resolve(), args.output.resolve()
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("Preserve prior evidence; use a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    contract = json.loads(contract_path.read_text())
    bodies = {body["name"]: body for body in contract["bodies"]}
    cases, ledgers = [], []
    started = time.perf_counter()
    for patch in contract["passive_contacts"]:
        path = output.parent/(patch["name"]+".xml")
        model, ledger = fixture(patch, bodies[patch["name"]], path)
        ledgers.append({"patch": patch, "body": bodies[patch["name"]], "numerical_metric": ledger,
                        "model_sha256": sha256(path)})
        cases.extend(one_case(model, patch, name) for name in PROGRAMS)
    receipt = {"schema": "goose_pad_implicit_source_diagnostic_v1", "engine": "mujoco_cpu",
               "engine_version": mujoco.__version__, "source_model_sha256": contract["model_sha256"],
               "source_contract_sha256": sha256(contract_path), "physics_dt_s": DT,
               "integrations_per_tick": 1, "substeps": 0, "qpos_writes_after_initialization": 0,
               "qualified": False, "optimizer_updates": 0, "inference_count": 0,
               "programs": PROGRAMS, "ticks_per_program": TICKS, "cases": cases, "physical_ledgers": ledgers,
               "passed_cases": sum(c["local_checks_passed"] for c in cases), "case_count": len(cases),
               "code_sha256": {str(p): sha256(p) for p in (Path(__file__).resolve(), Path(implicit_metric.__file__).resolve())},
               "elapsed_wall_s": time.perf_counter()-started,
               "limits": ["fixed parent/no ground or friction contacts", "no whole-body or GPU qualification",
                          "native row coefficients require a versioned backend implementation"]}
    write_json(output, receipt)
    print(f"STATUS DIAGNOSTIC_ONLY passed={receipt['passed_cases']}/{len(cases)} {output}", flush=True)


if __name__ == "__main__":
    main()
