"""Experimental single-Tick joint springs, damping and predictive native stops.

The compiled diagonal is numerical, never physical armature. This CPU-only
MuJoCo3.10 experiment is not a qualified training integrator or GPU backend.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import DT

REVISION = "goose_joint_backward_euler_predictive_v1"


def prepare_xml(root: ET.Element, physical_model) -> dict:
    """Compile M+h*C+h²*K and C+h*K, retaining a separate physical ledger.

    Euler without its second damping solve advances the already implicit
    scalar spring solve. Nonlinear velocity forces are evaluated explicitly.
    Two fixed tendons express each original scalar interval as two native
    unilateral rows. Both ends can therefore stop motion in the first Tick.
    """
    if mujoco.__version__ != "3.10.0" or physical_model.opt.timestep != DT:
        raise ValueError("Experimental metric requires MuJoCo3.10.0 and 20ms")
    option = root.find("option")
    option.set("integrator", "Euler")
    option.set("solver", "Newton")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("eulerdamp", "disable")
    flag.set("diagexact", "enable")
    flag.set("autoreset", "disable")
    elements = {j.get("name"): j for j in root.findall("./worldbody//joint")}
    tendon = root.find("tendon")
    if tendon is None:
        tendon = ET.SubElement(root, "tendon")
    existing_names = {e.get("name") for e in root.iter()}
    rows = []
    for jid in range(physical_model.njnt):
        kind = int(physical_model.jnt_type[jid])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if kind not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("Experimental metric supports scalar joints and free roots")
        name = mujoco.mj_id2name(physical_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name not in elements:
            raise ValueError("Physical joint must have an explicit named XML element")
        joint = elements[name]
        did = int(physical_model.jnt_dofadr[jid])
        armature = float(physical_model.dof_armature[did])
        damping = float(physical_model.dof_damping[did])
        stiffness = float(physical_model.jnt_stiffness[jid])
        if not np.isfinite([armature, damping, stiffness]).all() or min(armature, damping, stiffness) < 0:
            raise ValueError("Invalid physical spring/damping/armature")
        effective_armature = armature + DT*damping + DT**2*stiffness
        effective_damping = damping + DT*stiffness
        joint.set("armature", f"{effective_armature:.17g}")
        joint.set("damping", f"{effective_damping:.17g}")
        # Make inherited/default stiffness explicit too; no fitted coefficients.
        joint.set("stiffness", f"{stiffness:.17g}")
        stops = []
        if physical_model.jnt_limited[jid]:
            lower, upper = physical_model.jnt_range[jid]
            width = float(upper-lower)
            if not np.isfinite([lower, upper]).all() or width <= 0:
                raise ValueError("Predictive stop needs a finite original interval")
            joint.set("limited", "false")
            # Each far lower bound is redundant within the original interval.
            # Its distance is always greater than this tendon activation band.
            far = max(1., abs(float(lower)), abs(float(upper))) + width + 1.
            for suffix, coefficient, bound in (("upper", 1., upper), ("lower", -1., -lower)):
                stop_name = f"goose_stop_{name}_{suffix}"
                if stop_name in existing_names:
                    raise ValueError("Predictive stop name collision")
                fixed = ET.SubElement(tendon, "fixed", name=stop_name, limited="true",
                                      range=f"{-far:.17g} {bound:.17g}",
                                      margin=f"{width+1e-7:.17g}", frictionloss="0", stiffness="0", damping="0")
                ET.SubElement(fixed, "joint", joint=name, coef=str(coefficient))
                stops.append({"name": stop_name, "coefficient": coefficient, "physical_bound": float(bound)})
        rows.append({"joint": name, "dof": did, "physical_armature": armature,
                     "physical_damping": damping, "physical_stiffness": stiffness,
                     "compiled_armature": effective_armature, "compiled_damping": effective_damping,
                     "physical_range": physical_model.jnt_range[jid].tolist(), "stops": stops})
    return {"revision": REVISION, "dt_s": DT, "joints": rows,
            "effective_armature_is_numerical": True,
            "physical_step_equation": "(M+h*C+h^2*K)*v_next=M*v+h*(F-K*(q-q_rest))",
            "nonlinear_velocity_forces": "explicit; requires whole-body qualification",
            "qualification": "CPU experiment only; no GPU or M0 admission"}


def native_step(model, data, *, normal_contact_mode="native"):
    """One native solve/integration; optional declared frictionless normal experiment.

    ``predictive_rigid`` uses each contact's actual signed distance instead of
    its search margin for the one-Tick unilateral velocity bound. It is only
    for a separately declared CPU diagnostic; default v2 behavior is unchanged.
    """
    if normal_contact_mode not in ("native", "predictive_rigid"):
        raise ValueError("Unsupported normal contact experiment")
    if (mujoco.__version__ != "3.10.0" or model.opt.timestep != DT
            or model.opt.integrator != int(mujoco.mjtIntegrator.mjINT_EULER)
            or model.opt.solver != int(mujoco.mjtSolver.mjSOL_NEWTON)
            or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)):
        raise RuntimeError("Unsupported experimental metric solver/ABI")
    before = float(data.time)
    mujoco.mj_step1(model, data)
    rigid_kinds = {int(mujoco.mjtConstraint.mjCNSTR_EQUALITY),
                   int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)}
    for row in range(data.nefc):
        kind = int(data.efc_type[row])
        predictive_contact = (normal_contact_mode == "predictive_rigid"
                              and kind == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS))
        if normal_contact_mode == "predictive_rigid" and kind in (
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)):
            raise RuntimeError("Predictive normal experiment requires frictionless condim=1")
        if kind not in rigid_kinds and not predictive_contact:
            continue
        # Newton's primal M/J/R system uses the compiled implicit metric.
        # Reprojecting here would overwrite these R/D values from XML solimp.
        data.efc_R[row] = max(1e-12, 1e-7*data.efc_diagA[row])
        data.efc_D[row] = 1./data.efc_R[row]
        distance = float(data.contact[int(data.efc_id[row])].dist) if predictive_contact else data.efc_pos[row]
        data.efc_aref[row] = -data.efc_vel[row]/DT-distance/DT**2
        if data.nisland:
            index = int(data.map_efc2iefc[row])
            if index < 0:
                raise RuntimeError("Active rigid row absent from native island")
            data.iefc_R[index] = data.efc_R[row]
            data.iefc_D[index] = data.efc_D[row]
            data.iefc_aref[index] = data.efc_aref[row]
    mujoco.mj_step2(model, data)
    if abs(float(data.time)-before-DT) > 1e-12:
        raise RuntimeError("Not exactly one native 20ms integration")
