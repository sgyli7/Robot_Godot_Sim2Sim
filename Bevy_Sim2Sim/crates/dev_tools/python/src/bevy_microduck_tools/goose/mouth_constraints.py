"""Compare original jaw equalities with an actual native four-bar output pin.

This is a fixed-head diagnostic, without contacts or a policy. It preserves
the delivered masses, hinges, armature, friction and motion ranges. The Tick
reference candidate derives its solver coefficients from dt, rather than
fitting mechanical stiffness. It never writes coordinates after initialization.
"""
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
from . import mouth_diagnostic

MODES = ("original_mimic", "native_pin_soft", "native_pin_tick_reference",
         "native_pin_tick_reference_euler_diagnostic", "native_pin_implicit_damping_metric",
         "native_pin_implicit_metric_predictive_stop")
ANGLES = ("beak_hinge", "beak_input_rotor", "beak_coupler_link")
ANGLE_TOLERANCE_RAD = .005
PIN_TOLERANCE_M = .0001
LIMIT_TOLERANCE_RAD = .005


def apply_implicit_damping_metric(root: ET.Element):
    """Prepare a native solve with M+h*C while preserving physical C forces.

    This diagnostic treats joint viscous forces implicitly. Nonlinear native
    velocity forces retain their explicit evaluation; it is not a replacement
    for a whole-robot integrator qualification.
    """
    option = root.find("option")
    option.set("integrator", "Euler")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("eulerdamp", "disable")
    flag.set("diagexact", "enable")
    for joint in root.iter("joint"):
        armature = float(joint.get("armature", "0"))
        damping = float(joint.get("damping", "0"))
        joint.set("armature", f"{armature+DT*damping:.17g}")


def derive_fixture(original_path: Path, destination: Path, mode: str):
    if mode not in MODES:
        raise ValueError("Unknown jaw constraint mode")
    tree = ET.parse(original_path)
    root = tree.getroot()
    root.set("model", "goose_jaw_" + mode)
    option = root.find("option")
    ET.SubElement(option, "flag", autoreset="disable")
    equality = root.find("equality")
    if mode != "original_mimic":
        original = dict(equality[0].attrib)
        equality.clear()
        attributes = {"name": "actual_output_pin", "site1": "jaw_output_pin",
                      "site2": "coupler_output_pin", "solref": original["solref"],
                      "solimp": original["solimp"]}
        if mode.startswith("native_pin_tick_reference") or mode.startswith("native_pin_implicit"):
            # MuJoCo direct solref yields b=damping/dmax and k=stiffness/dmax².
            # With constant dmax=d, this gives aref=-v/dt-r/dt² exactly.
            d = .9999
            solref = f"{-d/DT**2:.17g} {-d/DT:.17g}"
            solimp = f"{d} {d} .0001 .5 2"
            attributes.update(solref=solref, solimp=solimp)
            for joint in root.iter("joint"):
                joint.set("solreflimit", solref)
                joint.set("solimplimit", solimp)
        ET.SubElement(equality, "connect", **attributes)
    if mode == "native_pin_tick_reference_euler_diagnostic":
        option.set("integrator", "Euler")
        option.find("flag").set("eulerdamp", "disable")
    if mode.startswith("native_pin_implicit"):
        # Constraint and velocity update both use M+h*C. Native Euler with
        # eulerdamp disabled only advances the already implicitly damped solve:
        # (M+h*C)*delta_v=h*(external-C*v+J.T*lambda).
        # The added diagonal is a numerical metric, not physical armature.
        apply_implicit_damping_metric(root)
        for joint in root.iter("joint"):
            if mode == "native_pin_implicit_metric_predictive_stop":
                # Activation band only. The native row reference below uses
                # the actual bound distance, so this does not shrink the range.
                lower, upper = np.fromstring(joint.get("range"), sep=" ")
                joint.set("margin", f"{(upper-lower)/2+1e-7:.17g}")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(destination))


def predictive_native_step(model, data):
    """Solve hard pin/stops in the same implicit damping metric, then advance."""
    if mujoco.__version__ != "3.10.0":
        raise RuntimeError("Native constraint-array diagnostic pinned to MuJoCo3.10.0")
    if model.opt.solver != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise RuntimeError("Predictive row diagnostic requires the native primal Newton solver")
    mujoco.mj_step1(model, data)
    rows = []
    for row in range(data.nefc):
        kind = int(data.efc_type[row])
        if kind not in (int(mujoco.mjtConstraint.mjCNSTR_EQUALITY),
                        int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)):
            continue
        # Relative regularization is fixed before this comparison. These
        # rows describe rigid pins/stops, not a measured elastic material.
        data.efc_R[row] = max(1e-12, 1e-7*data.efc_diagA[row])
        data.efc_D[row] = 1./data.efc_R[row]
        data.efc_aref[row] = -data.efc_vel[row]/DT-data.efc_pos[row]/DT**2
        rows.append(row)
    # step1 already computed the exact diagonal/Y. Calling projectConstraint
    # again with diagexact would overwrite our R/D from the XML impedance.
    # Newton uses the primal M/J/R system and needs no rebuilt dual AR matrix.
    if data.nisland:
        for row in rows:
            native_row = int(data.map_efc2iefc[row])
            if native_row >= 0:
                data.iefc_R[native_row] = data.efc_R[row]
                data.iefc_D[native_row] = data.efc_D[row]
                data.iefc_aref[native_row] = data.efc_aref[row]
    mujoco.mj_step2(model, data)


def one_case(model, initial_rad, load_n=0., load_site="jaw_grip_load_point", *,
             torque_nm=.24, ticks=100, predictive=False):
    data = mujoco.MjData(model)
    mouth_diagnostic.set_initial_pose(model, data, initial_rad)
    indices = [int(model.joint(name).qposadr[0]) for name in ANGLES]
    ids = [model.joint(name).id for name in ANGLES]
    ranges = model.jnt_range[ids]
    rows = []
    reason = None
    for tick in range(ticks):
        data.ctrl[0] = torque_nm
        data.qfrc_applied[:] = 0.
        force = np.zeros(3)
        if load_n:
            point = data.site_xpos[model.site(load_site).id].copy()
            jaw_joint = ids[0]
            tangent = np.cross(data.xaxis[jaw_joint], point-data.xanchor[jaw_joint])
            force = -load_n*tangent/np.linalg.norm(tangent)
            mujoco.mj_applyFT(model, data, force, np.zeros(3), point,
                            model.body("beak_hinge").id, data.qfrc_applied)
        applied = data.qfrc_applied.copy()
        before = float(data.time)
        try:
            if predictive:
                predictive_native_step(model, data)
            else:
                mujoco.mj_step(model, data)
        except mujoco.FatalError as error:
            reason = str(error)
            break
        if abs(data.time-before-DT) > 1e-12:
            reason = "not_one_20ms_integration"
            break
        mujoco.mj_kinematics(model, data)
        row = mouth_diagnostic.state(model, data)
        q = data.qpos[indices]
        limit_violation = float(np.maximum(np.maximum(ranges[:, 0]-q, q-ranges[:, 1]), 0.).max())
        row.update(tick=tick, input_torque_nm=torque_nm, load_force_world_n=force.tolist(),
                   external_generalized_torque_nm=applied.tolist(), limit_violation_rad=limit_violation)
        rows.append(row)
        if row["warnings"] or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason = "nonfinite_or_native_warning"
            break
    angle_error = max((max(abs(r["rotor_jaw_error_rad"]), abs(r["coupler_jaw_error_rad"]))
                       for r in rows), default=None)
    pin_error = max((r["output_pin_distance_m"] for r in rows), default=None)
    limit_error = max((r["limit_violation_rad"] for r in rows), default=None)
    passes = (reason is None and len(rows) == ticks and angle_error <= ANGLE_TOLERANCE_RAD
              and pin_error <= PIN_TOLERANCE_M and limit_error <= LIMIT_TOLERANCE_RAD)
    return {"initial_jaw_rad": initial_rad, "load_n": load_n, "load_site": load_site,
            "input_torque_nm": torque_nm, "integrations": len(rows), "failure_reason": reason,
            "max_angle_relation_error_rad": angle_error, "max_output_pin_distance_m": pin_error,
            "max_limit_violation_rad": limit_error, "local_checks_passed": passes, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args(argv)
    model_path, contract_path, output = (p.resolve() for p in (args.model, args.contract, args.output))
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("Use a fresh jaw experiment directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    contract = json.loads(contract_path.read_text())
    if sha256(model_path) != contract["model_sha256"]:
        raise ValueError("Frozen source identity mismatch")
    original = output.parent / "isolated_original.xml"
    mouth_diagnostic.make_fixture(model_path, contract, original)
    started = time.perf_counter()
    modes = []
    for mode in args.modes:
        path = output.parent / (mode + ".xml")
        model = derive_fixture(original, path, mode)
        predictive = mode == "native_pin_implicit_metric_predictive_stop"
        cases = [one_case(model, 0., predictive=predictive), one_case(model, .275, predictive=predictive)]
        for site in ("jaw_output_pin", "jaw_grip_load_point"):
            for load in (5., 10., 20.):
                cases.append(one_case(model, .275, load, site, predictive=predictive))
        # Rated-effort and closing cases also challenge the physical stops.
        cases.extend([one_case(model, .275, torque_nm=4.4, predictive=predictive),
                      one_case(model, .275, torque_nm=-4.4, predictive=predictive)])
        row = {"mode": mode, "fixture_path": str(path), "fixture_sha256": sha256(path),
               "native_equality_types": model.eq_type.tolist(), "cases": cases,
               "native_integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
               "native_disableflags": int(model.opt.disableflags),
               "native_dof_armature": model.dof_armature.tolist(),
               "native_dof_damping": model.dof_damping.tolist(),
               "local_checks_passed": all(c["local_checks_passed"] for c in cases)}
        modes.append(row)
        print(f"EVENT mode={mode} local_pass={row['local_checks_passed']}", flush=True)
    report = {"schema": "goose_jaw_native_pin_comparison_v1", "engine": "mujoco_cpu",
              "engine_version": mujoco.__version__, "source_model_sha256": sha256(model_path),
              "source_contract_sha256": sha256(contract_path),
              "code_sha256": {str(p.resolve()): sha256(p) for p in (Path(__file__), Path(mouth_diagnostic.__file__))},
              "physics_dt_s": DT, "integrations_per_tick": 1, "substeps": 0,
              "fixed_head": True, "gravity_m_s2": [0., 0., 0.], "contacts_enabled": False,
              "native_autoreset_disabled": True, "qpos_writes_after_initialization": 0,
              "frozen_local_thresholds": {"angle_relation_rad": ANGLE_TOLERANCE_RAD,
                                           "output_pin_distance_m": PIN_TOLERANCE_M,
                                           "limit_violation_rad": LIMIT_TOLERANCE_RAD},
              "tick_reference": {"aref": "-v/dt-r/dt²", "impedance": .9999,
                                  "dt_s": DT, "applied_to": "output pin and original joint stops"},
              "implicit_metric": {"formula": "M_effective=M_physical+dt*diag(joint_damping)",
                                   "native_advance": "Euler,eulerdampdisabled;implicitdampingalreadyintheconstraintsolve",
                                   "effective_armature_is_numerical": True},
              "predictive_stop": {"aref": "-v/dt-gap_to_actual_bound/dt²",
                                  "relative_regularization": 1e-7, "minimum_regularization": 1e-12,
                                  "activation_margin_is_not_physical_range": True},
              "force_phase": "pre-integration native solve", "pose_phase": "post-integration kinematics refresh",
              "modes": modes, "elapsed_wall_s": time.perf_counter()-started, "qualified": False,
              "optimizer_updates": 0, "inference_count": 0, "actuation_mode": "prescribed rotor effort; no PD or policy",
              "previous_turn_classification": "progress; commit3547481 and nativePGScomparison",
              "limits": ["fixed parent, no whole-Goose qualification", "no gripping object or collision",
                         "no target-side comparison", "solver regularization and Tick reference need GPU parity",
                         "Euler variant is an integration diagnostic, not the approved source training route"]}
    write_json(output, report)
    print(f"STATUS DIAGNOSTIC_ONLY {output}", flush=True)


if __name__ == "__main__":
    main()
