"""CPU experiment: native 3.13 effective metric with predictive scalar bounds.

Public Python bindings do not expose the native discrete advance or its final
regularization stage. A forward preview prepares that metric; a second algebraic
constraint solve applies declared bounds. Native Euler advances once, with its
additional damping disabled. This is not a Warp or Rapier qualification.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import DT, sha256
from .native_discrete_runtime import CANDIDATE as PARENT, _require_platform
from .runtime import GooseSourceRuntime

REVISION = "goose_native_discrete_predictive_limits_v1"
CANDIDATE = "goose_460_full50_discrete_limits_v4"
CANDIDATE_EXACT = "goose_460_full50_discrete_limits_exact_v5"
PREFIX = "goose_discrete_stop_"


def prepare_limits(root, physical_model):
    """Represent both original scalar endpoints as always available native rows."""
    _require_platform()
    if physical_model.opt.timestep != DT:
        raise ValueError("Predictive limits require one 20ms Tick")
    option = root.find("option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("eulerdamp", "disable")
    flag.set("autoreset", "disable")
    tendon = root.find("tendon")
    if tendon is None:
        tendon = ET.SubElement(root, "tendon")
    elements = {j.get("name"): j for j in root.findall("./worldbody//joint")}
    rows = []
    for jid in range(physical_model.njnt):
        kind = physical_model.jnt_type[jid]
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if kind not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("Only scalar joints and free roots are supported")
        if not physical_model.jnt_limited[jid]:
            continue
        name = mujoco.mj_id2name(physical_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lower, upper = map(float, physical_model.jnt_range[jid])
        width = upper-lower
        if not np.isfinite([lower, upper]).all() or width <= 0:
            raise ValueError("Original scalar interval must be finite")
        elements[name].set("limited", "false")
        far = max(1., abs(lower), abs(upper))+width+1.
        for suffix, coefficient, bound in (("upper", 1., upper), ("lower", -1., -lower)):
            stop_name = PREFIX+name+"_"+suffix
            if any(t.get("name") == stop_name for t in tendon):
                raise ValueError("Predictive limit name collision")
            fixed = ET.SubElement(tendon, "fixed", name=stop_name, limited="true",
                                  range=f"{-far:.17g} {bound:.17g}", margin=f"{width+1e-7:.17g}",
                                  frictionloss="0", stiffness="0", damping="0")
            ET.SubElement(fixed, "joint", joint=name, coef=str(coefficient))
            rows.append({"name": stop_name, "joint": name, "coefficient": coefficient,
                         "bound": bound, "physical_range": [lower, upper]})
    return rows


def build_candidate(parent_model: Path, parent_contract: Path, destination: Path, *, exact_diagonal=False):
    _require_platform()
    contract = json.loads(parent_contract.read_text())
    if contract.get("candidate") != PARENT or contract["model_sha256"] != sha256(parent_model):
        raise ValueError("Frozen native discrete v3 parent required")
    if destination.exists():
        raise FileExistsError("Preserve the frozen predictive candidate")
    physical = mujoco.MjModel.from_xml_path(str(parent_model))
    tree = ET.parse(parent_model)
    root = tree.getroot()
    candidate_id = CANDIDATE_EXACT if exact_diagonal else CANDIDATE
    root.set("model", candidate_id)
    rows = prepare_limits(root, physical)
    if exact_diagonal:
        root.find("./option/flag").set("diagexact", "enable")
    destination.mkdir(parents=True)
    model_path = destination/"robot.xml"
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode")
    contract = copy.deepcopy(contract)
    contract.update(candidate=candidate_id, model_sha256=sha256(model_path),
                    status="SOURCE_PREDICTIVE_DISCRETE_UNQUALIFIED", training_release=False)
    contract["native_discrete"].update(revision=REVISION,
        parent_model_sha256=sha256(parent_model), parent_contract_sha256=sha256(parent_contract),
        predictive_limits={"rows": rows, "exact_diagonal": bool(exact_diagonal), "regularization": "max(1e-12, 1e-7*efc_diagA)",
                           "aref": "-efc_vel/h-actual_gap/h^2", "algebraic_solves": 2,
                           "advance": "one native Euler; no second damping or state projection"})
    path = destination/"contract.json"
    path.write_text(json.dumps(contract, indent=2)+"\n")
    return model_path, path


def advance(model, data, *, predictive=True):
    """Prepare native H, correct bounds, solve, and perform one actual advance."""
    _require_platform()
    if (model.opt.timestep != DT or model.opt.integrator != int(mujoco.mjtIntegrator.mjINT_DISCRETE)
            or model.opt.solver != int(mujoco.mjtSolver.mjSOL_NEWTON)
            or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
            or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            or model.nflex or model.opt.noslip_iterations):
        raise ValueError("Unsupported native predictive metric/solver")
    # The preview includes actuation and the effective-metric regularization.
    # No integration occurs here; it does not recalculate the caller's PD torque.
    before = float(data.time)
    mujoco.mj_forward(model, data)
    preview_time = float(data.time)
    changed = 0
    if predictive:
        for row in range(data.nefc):
            if data.efc_type[row] != int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON):
                continue
            tid = int(data.efc_id[row])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid)
            if not name.startswith(PREFIX):
                continue
            regularizer = max(1e-12, 1e-7*float(data.efc_diagA[row]))
            data.efc_R[row] = regularizer
            data.efc_D[row] = 1./regularizer
            data.efc_aref[row] = -data.efc_vel[row]/DT-data.efc_pos[row]/DT**2
            if data.nisland:
                index = int(data.map_efc2iefc[row])
                if index < 0:
                    raise RuntimeError("Active predictive row absent from native island")
                data.iefc_R[index] = regularizer
                data.iefc_D[index] = 1./regularizer
                data.iefc_aref[index] = data.efc_aref[row]
            changed += 1
        mujoco.mj_fwdConstraint(model, data)
        # The preview may have populated lazy RNE for force sensors.
        # Match native step2: refresh it for the corrected constraint result.
        data.flg_rnepost = 0
        mujoco.mj_sensorAcc(model, data)
        mujoco.mj_checkAcc(model, data)
    pre_advance_time = float(data.time)
    if any(w.number for w in data.warning) or not np.isfinite(data.qacc).all():
        raise FloatingPointError("Native predictive solve warning; state retained")
    mujoco.mj_Euler(model, data)
    if preview_time != before or pre_advance_time != before or abs(data.time-before-DT) > 1e-12:
        raise RuntimeError("Not exactly one 20ms integration")
    return {"preview_constraint_solves": 1, "corrected_constraint_solves": int(predictive),
            "integrations": 1, "corrected_limit_rows": changed, "coordinate_writes": 0}


class GoosePredictiveDiscreteRuntime(GooseSourceRuntime):
    def __init__(self, model_path: Path, contract_path: Path, **kwargs):
        _require_platform()
        contract = json.loads(contract_path.read_text())
        native = contract.get("native_discrete", {})
        expected_candidate = CANDIDATE_EXACT if native.get("predictive_limits", {}).get("exact_diagonal", False) else CANDIDATE
        if (contract.get("candidate") != expected_candidate or contract.get("integrator") != "discrete"
                or native.get("engine_version") != "3.13.0" or native.get("revision") != REVISION
                or native.get("physical_parameters_changed") is not False
                or not native.get("predictive_limits", {}).get("rows")):
            raise ValueError("Explicit predictive discrete candidate required")
        super().__init__(model_path, contract_path, **kwargs)
        compiled_exact = bool(self.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT))
        if compiled_exact != bool(native["predictive_limits"].get("exact_diagonal", False)):
            raise ValueError("Compiled exact-diagonal flag disagrees with the frozen contract")
        self.advance_receipt = None

    def _integrate(self):
        self.advance_receipt = advance(self.model, self.data)
