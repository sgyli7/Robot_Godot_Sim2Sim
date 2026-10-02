"""Experimental shared sliding-pad condensation, one native 20ms integration.

Each region retains one compression state and its original K/C/stroke. Its
mass is already in the physical foot; no independent pad mass is reintroduced.
This CPU fixture has no friction, whole-body, target-engine or GPU admission.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .artifacts import DT

REVISION = "goose_shared_pad_condensed_contact_v4"


@dataclass(frozen=True)
class SharedPad:
    """Original region and shared scalar sliding mechanics."""

    geom_id: int
    patch: str
    stiffness_n_m: float
    damping_n_s_m: float
    travel_m: float


def _nonnegative_quadratic(matrix, linear):
    """Bounded active-set solve of a positive-definite unilateral impulse QP."""
    count = len(linear)
    impulses = np.zeros(count)
    active = np.zeros(count, dtype=bool)
    iterations = 0
    bound = 16*count+32
    velocity_tolerance = 1e-9
    while True:
        residual = matrix@impulses+linear
        candidates = np.flatnonzero(~active & (residual < -velocity_tolerance))
        if not len(candidates):
            break
        active[candidates[np.argmin(residual[candidates])]] = True
        while True:
            iterations += 1
            if iterations > bound:
                raise RuntimeError("Shared-pad impulse solve exceeded its algebraic bound")
            ids = np.flatnonzero(active)
            trial = np.zeros(count)
            trial[ids] = np.linalg.solve(matrix[np.ix_(ids, ids)], -linear[ids])
            negative = ids[trial[ids] <= 0.]
            if not len(negative):
                impulses = trial
                break
            ratio = impulses[negative]/(impulses[negative]-trial[negative])
            step = float(np.min(ratio))
            impulses += step*(trial-impulses)
            leaving = negative[ratio <= step+1e-14]
            impulses[leaving] = 0.
            active[leaving] = False
        if not np.isfinite(impulses).all():
            raise FloatingPointError("Nonfinite shared-pad impulse; no state update")
    residual = matrix@impulses+linear
    error = max(float(np.max(np.maximum(-residual, 0.), initial=0.)),
                float(np.max(np.abs(residual[impulses > 1e-12]), initial=0.)))
    if error > 2e-8 or np.any(impulses < 0.) or not np.isfinite(residual).all():
        raise RuntimeError("Shared-pad complementarity residual failed")
    return impulses, error, iterations


def solve_shared_pads(mass, free_velocity, jacobian, gaps, patch_indices, cosines,
                      pads, compression):
    """Condense one massless K/C coordinate per patch into the coupled solve.

    The virtual diagonal h*C+h²*K is numerical. Contact rows sharing a patch
    therefore share an off-diagonal response instead of independent springs.
    Both original stroke bounds participate in the same velocity solve.
    Physical generalized coordinates may have any dimension; this algebraic
    function alone does not qualify their runtime or other constraints.
    """
    mass = np.asarray(mass, dtype=float)
    free_velocity = np.asarray(free_velocity, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    patch_indices = np.asarray(patch_indices, dtype=int)
    cosines = np.asarray(cosines, dtype=float)
    compression = np.asarray(compression, dtype=float)
    n, p, contacts = len(free_velocity), len(pads), len(gaps)
    arrays = (mass, free_velocity, jacobian, gaps, cosines, compression)
    if (mass.shape != (n, n) or jacobian.shape != (contacts, n)
            or patch_indices.shape != gaps.shape or cosines.shape != gaps.shape
            or compression.shape != (p,) or p == 0
            or any(not np.isfinite(a).all() for a in arrays)
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=1e-12)
            or np.any(patch_indices < 0) or np.any(patch_indices >= p)
            or np.any(cosines <= .5) or np.any(cosines > 1.+1e-10)):
        raise ValueError("Invalid shared-pad physical system")
    np.linalg.cholesky(mass)
    k = np.array([m.stiffness_n_m for m in pads])
    c = np.array([m.damping_n_s_m for m in pads])
    travel = np.array([m.travel_m for m in pads])
    if (not np.isfinite([k, c, travel]).all() or np.any(k <= 0.) or np.any(c < 0.)
            or np.any(travel <= 0.) or np.any(compression < -2e-8)
            or np.any(compression > travel+2e-8)):
        raise ValueError("Invalid original K/C/stroke or prior compression")
    virtual_diagonal = DT*c+DT**2*k
    virtual_free_velocity = -k*compression/(c+DT*k)
    all_free = np.r_[free_velocity, virtual_free_velocity]
    # Native physical normal Jacobians and a single shared slide column.
    rows = np.zeros((contacts+2*p, n+p))
    rows[:contacts, :n] = jacobian
    rows[np.arange(contacts), n+patch_indices] = cosines
    rows[contacts:contacts+p, n:] = np.eye(p)
    rows[contacts+p:, n:] = -np.eye(p)
    all_gaps = np.r_[gaps+cosines*compression[patch_indices], compression, travel-compression]
    inverse_rows = np.empty_like(rows)
    inverse_rows[:, :n] = np.linalg.solve(mass, rows[:, :n].T).T
    inverse_rows[:, n:] = rows[:, n:]/virtual_diagonal
    response = rows@inverse_rows.T
    # Preserve finite predictive-row regularization. It is not physical K/C.
    regularization = np.maximum(1e-12, 1e-7*np.diag(response))
    system = response+np.diag(regularization)
    linear = rows@all_free+all_gaps/DT
    impulses, residual, iterations = _nonnegative_quadratic(system, linear)
    next_velocity = all_free+inverse_rows.T@impulses
    next_compression = compression+DT*next_velocity[n:]
    violation = float(np.max(np.maximum(-next_compression, next_compression-travel), initial=0.))
    if violation > 2e-8:
        raise RuntimeError("Shared-pad original stroke exceeded; no integration")
    return {"velocity": next_velocity[:n], "compression": next_compression,
            "compression_velocity": next_velocity[n:], "normal_impulses": impulses[:contacts],
            "lower_stop_impulses": impulses[contacts:contacts+p], "upper_stop_impulses": impulses[contacts+p:],
            "complementarity_residual_m_s": residual, "algebraic_iterations": iterations,
            "max_travel_violation_m": violation, "virtual_diagonal": virtual_diagonal}


def shared_step(model, data, pads, compression):
    """Six-DoF source fixture: native M/forces, shared solve, one mj_Euler call.

    mj_step1 computes native kinematics/contact rows; actuation and smooth
    acceleration precede the custom coupled constraint solve. Only qacc is
    supplied to the integrator: no direct qpos/qvel writes or post-step repair.
    The returned compression is committed only after successful integration.
    """
    required = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP) | int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    if (mujoco.__version__ != "3.10.0" or model.opt.timestep != DT
            or model.opt.integrator != int(mujoco.mjtIntegrator.mjINT_EULER)
            or model.opt.disableflags & required != required
            or model.nv != 6 or model.nq != 7 or model.njnt != 1 or model.nbody != 2
            or model.jnt_type[0] != int(mujoco.mjtJoint.mjJNT_FREE)
            or np.any(model.dof_damping) or np.any(model.dof_armature)
            or len(pads) != 6 or len({p.geom_id for p in pads}) != 6):
        raise ValueError("Frozen isolated six-DoF Euler fixture and six original shared patches required")
    bindings = {p.geom_id: i for i,p in enumerate(pads)}
    for pad in pads:
        if not 0 <= pad.geom_id < model.ngeom or model.geom_bodyid[pad.geom_id] != 1:
            raise ValueError("Shared pad must belong to the physical foot")
    time_before = float(data.time)
    mujoco.mj_step1(model, data)
    mujoco.mj_fwdActuation(model, data)
    mujoco.mj_fwdAcceleration(model, data)
    mass = np.zeros((6,6))
    mujoco.mj_fullM(model, data, mass)
    free_velocity = data.qvel+DT*data.qacc_smooth
    axis = data.xmat[1].reshape(3,3)[:,2].copy()
    contacts = []
    jacobians, gaps, indices, cosines = [], [], [], []
    for cid, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        candidates = [bindings[int(g)] for g in (contact.geom1, contact.geom2) if int(g) in bindings]
        if (len(candidates) != 1 or contact.dim != 1
                or not np.allclose(np.abs(contact.frame[:3]), [0.,0.,1.], atol=1e-10)):
            raise ValueError("Shared-pad fixture supports declared patches and frictionless horizontal ground only")
        row = int(contact.efc_address)
        if data.efc_type[row] != int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS):
            raise ValueError("Shared-pad fixture has a nonnormal constraint")
        patch = candidates[0]
        normal = contact.frame[:3].copy()
        if model.geom_bodyid[contact.geom2] != 1:
            normal *= -1.
        point = contact.pos.copy()+axis*compression[patch]
        jacp, jacr = np.zeros((3,6)), np.zeros((3,6))
        mujoco.mj_jac(model, data, jacp, jacr, point, 1)
        cosine = float(axis@normal)
        jacobians.append(normal@jacp)
        gaps.append(float(contact.dist))
        indices.append(patch)
        cosines.append(cosine)
        contacts.append({"contact_id": cid, "patch": pads[patch].patch,
                         "point_world_m": point.tolist(), "normal_world": normal.tolist(),
                         "nominal_gap_m": float(contact.dist), "axis_cosine": cosine})
    if len(contacts) != data.nefc:
        raise ValueError("Unaccounted native constraint; no integration")
    result = solve_shared_pads(mass, free_velocity, np.asarray(jacobians).reshape(-1,6),
                               gaps, indices, cosines, pads, compression)
    data.qacc[:] = (result['velocity']-data.qvel)/DT
    data.qfrc_constraint[:] = mass@(data.qacc-data.qacc_smooth)
    mujoco.mj_Euler(model, data)
    if (abs(float(data.time)-time_before-DT) > 1e-12
            or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or any(w.number for w in data.warning)):
        raise FloatingPointError("Shared-pad native integration failed; no reset")
    for contact, impulse in zip(contacts, result['normal_impulses'], strict=True):
        contact['normal_impulse_n_s'] = float(impulse)
        contact['normal_force_n'] = float(impulse/DT)
    return result, contacts
