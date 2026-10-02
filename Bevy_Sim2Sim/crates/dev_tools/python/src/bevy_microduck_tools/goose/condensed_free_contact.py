"""Unqualified six-DoF condensed-foot normal foundation, one native 20ms step.

The per-point foundation approximates each original sliding patch. It preserves
its region and uses projected K/C; it does not inherit exact pad kinematics or
high-frequency equivalence. Whole-body and frictional use remain rejected.
"""
from collections import Counter

import mujoco
import numpy as np

from .artifacts import DT
from .condensed_contact import ContactMaterial

REVISION = "goose_free_condensed_contact_v2"


def free_step(model, data, materials: tuple[ContactMaterial, ...]) -> list[dict]:
    """Perform one native integration with physical projected normal rows.

    This diagnostic retains the free joint and full physical inertia. Contact
    force and moment qualification against the original moving pads is separate.
    """
    if (mujoco.__version__ != "3.10.0" or model.opt.timestep != DT
            or model.opt.solver != int(mujoco.mjtSolver.mjSOL_NEWTON)
            or model.opt.integrator != int(mujoco.mjtIntegrator.mjINT_IMPLICIT)
            or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)):
        raise ValueError("Frozen 3.10 CPU implicit/Newton 20ms ABI required")
    if (model.nv != 6 or model.nq != 7 or model.njnt != 1 or model.nbody != 2
            or model.jnt_type[0] != int(mujoco.mjtJoint.mjJNT_FREE)
            or np.any(model.dof_damping) or np.any(model.dof_armature)):
        raise ValueError("Isolated physical free foot only; no whole-body qualification")
    bindings = {m.geom_id: m for m in materials}
    if len(bindings) != len(materials) or len(materials) != 12:
        raise ValueError("Six original patches and six declared backing boxes required")
    for material in materials:
        if (not 0 <= material.geom_id < model.ngeom or model.geom_bodyid[material.geom_id] != 1
                or not material.patch or material.stiffness_n_m <= 0 or material.damping_n_s_m < 0
                or not np.isfinite([material.stiffness_n_m, material.damping_n_s_m]).all()):
            raise ValueError("Invalid original physical patch")
    before = float(data.time)
    mujoco.mj_step1(model, data)
    up = data.xmat[1].reshape(3, 3)[:, 2]
    contacts = []
    for cid, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        candidates = [bindings[int(g)] for g in (contact.geom1, contact.geom2) if int(g) in bindings]
        if (len(candidates) != 1 or contact.dim != 1
                or not np.allclose(np.abs(contact.frame[:3]), [0., 0., 1.], atol=1e-10)):
            raise ValueError("Only frictionless horizontal plane contact is supported")
        cosine = abs(float(up @ contact.frame[:3]))
        if cosine < 0.5:
            raise ValueError("Foot tilt outside declared prototype range")
        contacts.append((cid, contact, candidates[0], cosine))
    counts = Counter(m.patch for _, _, m, _ in contacts if not m.travel_stop)
    rows = []
    for cid, contact, material, cosine in contacts:
        row = int(contact.efc_address)
        if (data.efc_type[row] != int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
                or data.efc_id[row] != cid):
            raise RuntimeError("Native contact identity mismatch")
        gap, velocity = float(contact.dist), float(data.efc_vel[row])
        if material.travel_stop:
            alpha = max(1e-12, 1e-7*float(data.efc_diagA[row]))
            aref = -velocity/DT-gap/DT**2
            k, c = 0., 0.
        else:
            # Virtual work: g = cosine*q_pad, Fn = F_pad/cosine.
            k = material.stiffness_n_m/(counts[material.patch]*cosine**2)
            c = material.damping_n_s_m/(counts[material.patch]*cosine**2)
            alpha = 1./(DT*(DT*k+c))
            aref = -velocity/DT-k*gap/(DT*(DT*k+c))
        data.efc_R[row], data.efc_D[row], data.efc_aref[row] = alpha, 1./alpha, aref
        if data.nisland:
            ir = int(data.map_efc2iefc[row])
            if ir < 0:
                raise RuntimeError("Contact absent from native island")
            data.iefc_R[ir], data.iefc_D[ir], data.iefc_aref[ir] = alpha, 1./alpha, aref
        rows.append({"row": row, "contact_id": cid, "patch": material.patch,
                     "travel_stop": material.travel_stop, "gap_m": gap,
                     "normal_world": contact.frame[:3].copy().tolist(),
                     "point_world_m": contact.pos.copy().tolist(),
                     "com_before_m": data.xipos[1].copy().tolist(),
                     "axis_cosine": cosine, "point_stiffness_n_m": k,
                     "point_damping_n_s_m": c, "native_R_inverse_kg": alpha,
                     "native_aref_m_s2": aref})
    mujoco.mj_step2(model, data)
    if abs(float(data.time)-before-DT) > 1e-12:
        raise RuntimeError("Not one native 20ms integration")
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("Nonfinite free-foot state; no reset")
    return rows
