"""Diagnostic 004 contact adapter between public mjw.step1 / step2.

The native solver and single 20 ms integrator are untouched. Tangent row
units adapt the frozen CPU master cone coefficient to Warp's fixed formula.
Contact forces in those rows must be divided by tangent_scale to report SI.
Not admitted to production or training until real-state differential checks.
"""
import dataclasses
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from mujoco_warp._src import types


@wp.struct
class ContactBuffers:
    dist: wp.array[float]
    pos: wp.array[wp.vec3]
    frame: wp.array[wp.mat33]
    includemargin: wp.array[float]
    friction: wp.array[types.vec5]
    solref: wp.array[wp.vec2]
    solreffriction: wp.array[wp.vec2]
    solimp: wp.array[types.vec5]
    dim: wp.array[int]
    geom: wp.array[wp.vec2i]
    flex: wp.array[wp.vec2i]
    vert: wp.array[wp.vec2i]
    worldid: wp.array[int]
    type: wp.array[int]
    geomcollisionid: wp.array[int]


def buffers(c):
    result = ContactBuffers()
    for f in dataclasses.fields(c):
        if f.name != 'efc_address':
            setattr(result, f.name, getattr(c, f.name))
    return result


@wp.func
def copy_contact(src: ContactBuffers, dst: ContactBuffers, i: int, j: int):
    dst.dist[j] = src.dist[i]
    dst.pos[j] = src.pos[i]
    dst.frame[j] = src.frame[i]
    dst.includemargin[j] = src.includemargin[i]
    dst.friction[j] = src.friction[i]
    dst.solref[j] = src.solref[i]
    dst.solreffriction[j] = src.solreffriction[i]
    dst.solimp[j] = src.solimp[i]
    dst.dim[j] = src.dim[i]
    dst.geom[j] = src.geom[i]
    dst.flex[j] = src.flex[i]
    dst.vert[j] = src.vert[i]
    dst.worldid[j] = src.worldid[i]
    dst.type[j] = src.type[i]
    dst.geomcollisionid[j] = src.geomcollisionid[i]


@wp.kernel
def retain_contacts(src: ContactBuffers, dst: ContactBuffers,
                    count: wp.array[int], outcount: wp.array[int], capacity: int,
                    ground: int, feet: wp.array[int], bodies: wp.array[int],
                    xmat: wp.array2d[wp.mat33], errors: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    pair = src.geom[i]
    gid = int(-1)
    if pair[0] == ground:
        gid = pair[1]
    elif pair[1] == ground:
        gid = pair[0]
    world = src.worldid[i]
    for p in range(2):
        if gid == feet[p] and xmat[world, bodies[p]][2, 2] > 0.7:
            return
    j = wp.atomic_add(outcount, 0, 1)
    if j >= capacity:
        wp.atomic_add(errors, 0, 1)
        return
    copy_contact(src, dst, i, j)


@wp.kernel
def add_sole_points(dst: ContactBuffers, outcount: wp.array[int], capacity: int,
                    ground: int, feet: wp.array[int], bodies: wp.array[int],
                    corners: wp.array2d[wp.vec3], xmat: wp.array2d[wp.mat33],
                    xpos: wp.array2d[wp.vec3], geom_xpos: wp.array2d[wp.vec3],
                    geom_friction: wp.array2d[wp.vec3], geom_solref: wp.array2d[wp.vec2],
                    geom_solimp: wp.array2d[types.vec5], errors: wp.array[int]):
    world, p, corner = wp.tid()
    gid = feet[p]
    bid = bodies[p]
    if xmat[world, bid][2, 2] <= 0.7:
        return
    point = xmat[world, bid] * corners[p, corner] + xpos[world, bid]
    gap = point[2] - geom_xpos[world, ground][2]
    if gap > 0.01:
        return
    j = wp.atomic_add(outcount, 0, 1)
    if j >= capacity:
        wp.atomic_add(errors, 0, 1)
        return
    dst.dist[j] = gap
    dst.pos[j] = point - wp.vec3(0.0, 0.0, gap / 2.0)
    dst.frame[j] = wp.mat33(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    dst.includemargin[j] = 0.01
    friction = wp.max(geom_friction[world % geom_friction.shape[0], gid][0],
                      geom_friction[world % geom_friction.shape[0], ground][0])
    dst.friction[j] = types.vec5(friction, friction, 0.01, 0.002, 0.002)
    dst.solref[j] = geom_solref[world % geom_solref.shape[0], gid]
    dst.solreffriction[j] = wp.vec2(0.0, 0.0)
    dst.solimp[j] = geom_solimp[world % geom_solimp.shape[0], gid]
    dst.dim[j] = 3
    dst.geom[j] = wp.vec2i(ground, gid)
    dst.flex[j] = wp.vec2i(-1, -1)
    dst.vert[j] = wp.vec2i(-1, -1)
    dst.worldid[j] = world
    dst.type[j] = 1
    dst.geomcollisionid[j] = corner


@wp.kernel
def rebind_rows(contact: ContactBuffers, count: wp.array[int],
                address: wp.array2d[int], ground: int, feet: wp.array[int],
                impratio_invsqrt: wp.array[float], nv: int, sparse: bool,
                D: wp.array2d[float], vel: wp.array2d[float], aref: wp.array2d[float],
                J: wp.array3d[float], rowadr: wp.array2d[int], rownnz: wp.array2d[int],
                scales: wp.array[float], errors: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    scales[i] = 1.0
    pair = contact.geom[i]
    gid = int(-1)
    if pair[0] == ground:
        gid = pair[1]
    elif pair[1] == ground:
        gid = pair[0]
    if gid != feet[0] and gid != feet[1]:
        return
    world = contact.worldid[i]
    r0 = address[i, 0]
    if r0 < 0:
        return
    if contact.dim[i] != 3:
        wp.atomic_add(errors, 0, 1)
        return
    num = int(0)
    for j in range(count[0]):
        if contact.worldid[j] == world and address[j, 0] >= 0:
            pairj = contact.geom[j]
            if (pairj[0] == ground and pairj[1] == gid) or (pairj[1] == ground and pairj[0] == gid):
                num += 1
    k = 140142.1824 / float(num)
    c = 12.0 / float(num)
    alpha = 1.0 / (0.02 * (0.02 * k + c))
    aref[world, r0] = -vel[world, r0] / 0.02 - k * contact.dist[i] * alpha
    D[world, r0] = 1.0 / alpha
    r1 = address[i, 1]
    r2 = address[i, 2]
    if r1 < 0 or r2 < 0 or D[world, r1] <= 0.0:
        wp.atomic_add(errors, 0, 1)
        return
    if wp.abs(D[world, r1] - D[world, r2]) > 1e-5 * D[world, r1]:
        wp.atomic_add(errors, 0, 1)
        return
    ratio_inv = impratio_invsqrt[world % impratio_invsqrt.shape[0]]
    scale = wp.sqrt(1.0 / (ratio_inv * ratio_inv * D[world, r1] * alpha))
    scales[i] = scale
    friction = contact.friction[i]
    friction[0] *= scale
    friction[1] *= scale
    contact.friction[i] = friction
    for j in range(1, 3):
        row = address[i, j]
        D[world, row] *= scale * scale
        vel[world, row] /= scale
        aref[world, row] /= scale
        if sparse:
            start = rowadr[world, row]
            for a in range(rownnz[world, row]):
                J[world, 0, start + a] /= scale
        else:
            for a in range(nv):
                J[world, row, a] /= scale


class FrozenContactAdapter:
    def __init__(self, model, data, cpu_model, contract, *, entity_prefix=""):
        if (contract['runtime_revision'] != 'goose_task_proxy_be_contact_v1'
                or contract['contact_mapping']['method'] != 'whole_sole_native_manifold_backward_euler_v1'
                or cpu_model.opt.timestep != .02 or cpu_model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC):
            raise ValueError('Unknown source contact identity')
        if contract['contact_mapping']['per_foot_stiffness_n_m'] != 140142.1824 or contract['contact_mapping']['per_foot_damping_n_s_m'] != 12.:
            raise ValueError('Unknown source stiffness/damping')
        patches = contract['contact_mapping']['ground_contact_quadrature']
        if len(patches) != 2 or any(len(p['bottom_corners_body_m']) != 4 for p in patches):
            raise ValueError('Only the frozen two four-point soles are supported')
        self.model, self.data = model, data
        self.device = data.qpos.device
        self.ground = cpu_model.geom('ground').id
        self.feet = wp.array([cpu_model.geom(entity_prefix+p['geom']).id for p in patches], dtype=int, device=self.device)
        self.bodies = wp.array([cpu_model.body(entity_prefix+p['body']).id for p in patches], dtype=int, device=self.device)
        self.corners = wp.array(np.array([p['bottom_corners_body_m'] for p in patches]), dtype=wp.vec3, device=self.device)
        self.scratch = dataclasses.replace(data.contact, **{
            f.name: wp.zeros_like(getattr(data.contact, f.name)) for f in dataclasses.fields(data.contact)})
        self.count = wp.zeros(1, dtype=int, device=self.device)
        self.errors = wp.zeros(1, dtype=int, device=self.device)
        self.tangent_scale = wp.ones(data.naconmax, device=self.device)

    def apply(self):
        m, d = self.model, self.data
        self.count.zero_()
        self.errors.zero_()
        wp.launch(retain_contacts, d.naconmax, inputs=[buffers(d.contact), buffers(self.scratch), d.nacon,
                  self.count, d.naconmax, self.ground, self.feet, self.bodies, d.xmat, self.errors], device=self.device)
        wp.launch(add_sole_points, (d.nworld, 2, 4), inputs=[buffers(self.scratch), self.count, d.naconmax,
                  self.ground, self.feet, self.bodies, self.corners, d.xmat, d.xpos, d.geom_xpos,
                  m.geom_friction, m.geom_solref, m.geom_solimp, self.errors], device=self.device)
        # Scratch avoids aliasing the input manifold while contacts are compacted.
        for field in dataclasses.fields(d.contact):
            if field.name != 'efc_address':
                wp.copy(getattr(d.contact, field.name), getattr(self.scratch, field.name))
        wp.copy(d.nacon, self.count)
        mjw.make_constraint(m, d)
        wp.launch(rebind_rows, d.naconmax, inputs=[buffers(d.contact), d.nacon, d.contact.efc_address,
                  self.ground, self.feet, m.opt.impratio_invsqrt, m.nv, m.is_sparse,
                  d.efc.D, d.efc.vel, d.efc.aref, d.efc.J, d.efc.J_rowadr, d.efc.J_rownnz,
                  self.tangent_scale, self.errors], device=self.device)

    def assert_valid(self):
        if self.errors.numpy()[0] or self.data.nacon.numpy()[0] > self.data.naconmax or np.any(self.data.nefc.numpy() > self.data.njmax):
            raise RuntimeError('Frozen contact adapter unsupported state or capacity overflow')
