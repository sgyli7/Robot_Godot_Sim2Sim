"""Named CPU-only tilted sole load allocation using native MuJoCo rows.

The original BE spring/damper formula and elliptic friction remain intact.
Upright quadrature retains its original count. Tilted discovery-only rows
do not dilute stiffness allocated to the current material contacts. This
is a separate unadmitted mapping, with no replacement solver or substeps.
"""
import mujoco
import numpy as np

from .artifacts import DT


def integrate_material_share(runtime):
    model, data = runtime.model, runtime.data
    cfg = runtime.contract["contact_mapping"]
    runtime.foundation_k_by_geom = {g: cfg["per_foot_stiffness_n_m"] for g in runtime.foot_geoms}
    runtime.foundation_c_by_geom = {g: cfg["per_foot_damping_n_s_m"] for g in runtime.foot_geoms}
    mujoco.mj_step1(model, data)
    runtime._planar_sole_quadrature()
    contacts = []
    for cid, contact in enumerate(data.contact):
        if runtime.ground not in contact.geom or contact.efc_address < 0:
            continue
        gid = next(int(g) for g in contact.geom if g != runtime.ground)
        if gid in runtime.foot_geoms:
            contacts.append((cid, contact, gid))
    counts = {}
    for gid in runtime.foot_geoms:
        upright = data.xmat[int(model.geom_bodyid[gid])].reshape(3, 3)[2, 2] > .7
        counts[gid] = max(1, sum(g == gid and (upright or c.dist <= 0.)
            for _, c, g in contacts))
    runtime.sole_rows = []
    for cid, contact, gid in contacts:
        row = int(contact.efc_address)
        k = runtime.foundation_k_by_geom[gid] / counts[gid]
        damping = runtime.foundation_c_by_geom[gid] / counts[gid]
        alpha = 1. / (DT * (DT * k + damping))
        reference = -data.efc_vel[row] / DT - k * float(contact.dist) / (DT * (DT * k + damping))
        data.efc_R[row] = alpha
        data.efc_D[row] = 1. / alpha
        data.efc_aref[row] = reference
        if contact.dim > 1:
            contact.mu = contact.friction[0] * np.sqrt(data.efc_R[row + 1] / alpha)
        if data.nisland:
            island_row = int(data.map_efc2iefc[row])
            data.iefc_R[island_row] = alpha
            data.iefc_D[island_row] = 1. / alpha
            data.iefc_aref[island_row] = reference
        runtime.sole_rows.append((cid, row, gid, counts[gid], float(contact.dist),
            float(data.efc_vel[row]), 1. / alpha, reference))
    mujoco.mj_step2(model, data)
