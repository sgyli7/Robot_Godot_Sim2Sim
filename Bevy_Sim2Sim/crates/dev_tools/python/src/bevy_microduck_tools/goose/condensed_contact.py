"""Physical normal-contact rows for an isolated guided condensed-foot experiment.

This is a MuJoCo3.10 CPU diagnostic, not a whole-body or GPU training runtime.
The physical mass/inertia and per-patch K/C remain outside the solver's numerical
regularization. One call performs exactly one native 20ms integration.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np

from .artifacts import DT

REVISION = "goose_guided_condensed_physical_contact_v1"


@dataclass(frozen=True)
class ContactMaterial:
    """Original patch identity and its physical normal material, or its backing stop."""

    geom_id: int
    patch: str
    stiffness_n_m: float
    damping_n_s_m: float
    travel_stop: bool = False


def guided_step(model, data, materials: tuple[ContactMaterial, ...]) -> list[dict]:
    """Backward-Euler unilateral K/C contact plus predictive hard backing rows.

    For each soft point, Kp/Cp divide the patch totals among its actual points.
    R = 1/[h(h*Kp+Cp)] and aref = -v/h - Kp*g/[h(h*Kp+Cp)]. Thus a stationary
    point's native force is Kp*(-g), in N, independent of contact mass/solimp.
    The native primal solver uses D=1/R and couples all points through M and J.
    Backing stops use the actual gap and the existing predictive-row R guard.
    """
    if (mujoco.__version__ != "3.10.0" or model.opt.timestep != DT
            or model.opt.solver != int(mujoco.mjtSolver.mjSOL_NEWTON)
            or model.opt.integrator not in (int(mujoco.mjtIntegrator.mjINT_EULER),
                                            int(mujoco.mjtIntegrator.mjINT_IMPLICIT))
            or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)):
        raise ValueError("Guided condensed contact requires the frozen 3.10 CPU ABI and 20ms")
    if (model.nv != 1 or model.njnt != 1 or model.nbody != 2
            or model.jnt_type[0] != int(mujoco.mjtJoint.mjJNT_SLIDE)
            or not np.allclose(model.jnt_axis[0], [0., 0., 1.], atol=1e-12)
            or model.jnt_stiffness[0] != 0 or model.dof_damping[0] != 0
            or model.dof_armature[0] != 0):
        raise ValueError("Isolated one-slide physical foot only; no whole-body qualification")
    by_geom = {material.geom_id: material for material in materials}
    if len(by_geom) != len(materials) or not materials:
        raise ValueError("Each contact material needs a unique original geometry")
    for material in materials:
        if (not 0 <= material.geom_id < model.ngeom
                or model.geom_bodyid[material.geom_id] != 1 or not material.patch
                or not np.isfinite([material.stiffness_n_m, material.damping_n_s_m]).all()
                or material.damping_n_s_m < 0
                or (not material.travel_stop and material.stiffness_n_m <= 0)):
            raise ValueError("Invalid original physical contact material")
    before = float(data.time)
    mujoco.mj_step1(model, data)
    contacts = []
    for cid, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        candidates = [by_geom[int(gid)] for gid in (contact.geom1, contact.geom2) if int(gid) in by_geom]
        if len(candidates) != 1 or contact.dim != 1:
            raise ValueError("Only declared patch/backing versus frictionless ground is supported")
        if not np.allclose(np.abs(contact.frame[:3]), [0., 0., 1.], atol=1e-10):
            raise ValueError("Tilted contact needs a separately verified force/moment mapping")
        contacts.append((cid, contact, candidates[0]))
    counts = Counter(material.patch for _, _, material in contacts if not material.travel_stop)
    rows = []
    for cid, contact, material in contacts:
        row = int(contact.efc_address)
        if (data.efc_type[row] != int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
                or data.efc_id[row] != cid):
            raise RuntimeError("Native contact identity/layout mismatch")
        distance = float(contact.dist)
        velocity = float(data.efc_vel[row])
        if material.travel_stop:
            regularization = max(1e-12, 1e-7 * float(data.efc_diagA[row]))
            aref = -velocity / DT - distance / DT**2
            stiffness, damping = 0., 0.
        else:
            count = counts[material.patch]
            stiffness = material.stiffness_n_m / count
            damping = material.damping_n_s_m / count
            denominator = DT * stiffness + damping
            regularization = 1. / (DT * denominator)
            aref = -velocity / DT - stiffness * distance / (DT * denominator)
        data.efc_R[row] = regularization
        data.efc_D[row] = 1. / regularization
        data.efc_aref[row] = aref
        if data.nisland:
            island_row = int(data.map_efc2iefc[row])
            if island_row < 0:
                raise RuntimeError("Contact row absent from original native island")
            data.iefc_R[island_row] = data.efc_R[row]
            data.iefc_D[island_row] = data.efc_D[row]
            data.iefc_aref[island_row] = data.efc_aref[row]
        rows.append({"contact_id": cid, "row": row, "patch": material.patch,
                     "travel_stop": material.travel_stop, "actual_signed_distance_m": distance,
                     "native_velocity_m_s": velocity, "point_stiffness_n_m": stiffness,
                     "point_damping_n_s_m": damping, "native_R_inverse_kg": regularization,
                     "native_aref_m_s2": aref})
    mujoco.mj_step2(model, data)
    if abs(float(data.time) - before - DT) > 1e-12:
        raise RuntimeError("Not one native 20ms integration")
    return rows
