"""Convert Mink1.3 contact velocity bounds to its QP displacement units."""
import importlib.metadata
import itertools
import math

import mujoco
import numpy as np

from mink.limits.collision_avoidance_limit import CollisionAvoidanceLimit
from mink.limits.limit import Constraint

REVISION = 'goose_mink_planning_contact_corners_v2'


def box_support_height(half_size, rotation_world):
    """Minimum box center height over a horizontal plane for this rotation.

    This is a planning goal, not a physical pose write or collision inflation.
    Linear center-height interpolation while rotating a long sole can place
    its corners below the plane even when its endpoints are both valid.
    """
    size = np.asarray(half_size, dtype=float)
    rotation = np.asarray(rotation_world, dtype=float)
    if (size.shape != (3,) or rotation.shape != (3,3)
            or not np.isfinite(size).all() or not np.isfinite(rotation).all()
            or np.any(size <= 0.)
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8,rtol=0.)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-8,rtol=0.)):
        raise ValueError('Finite positive box sizes and a proper rotation required')
    return float(np.abs(rotation[2])@size)


class DisplacementCollisionLimit(CollisionAvoidanceLimit):
    """Planning-only unit adapter; no collision shapes or physics are changed.

    In Mink1.3 the contact upper bound has velocity units (distance/dt),
    while build_ik optimizes displacement and returns velocity=delta_q/dt.
    This fixed conversion makes the original distance barrier constrain that
    displacement. Linearization still cannot guarantee nonlinear feasibility;
    paths need explicit geometry checking and actual drive playback.
    """

    def __init__(self, *args, **kwargs):
        if importlib.metadata.version('mink') != '1.3.0':
            raise ValueError('Contact unit adapter requires pinned Mink1.3.0')
        super().__init__(*args, **kwargs)

    def compute_qp_inequalities(self, configuration, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive planning dt required')
        inequality = super().compute_qp_inequalities(configuration, dt)
        if inequality.inactive:
            return inequality
        return Constraint(G=inequality.G, h=inequality.h*dt)


class GoosePlanningCollisionLimit(DisplacementCollisionLimit):
    """Retain explicitly requested static-world versus floating-root pairs.

    Native Mink's parent-child heuristic also filters world/root collision.
    MuJoCo treats world contacts as an exception to its parent filter. Restore
    only intended static-world pairs with a dynamic weld, compatible masks
    and no explicit exclusion. Ordinary robot parent pairs stay untouched.
    Static plane/box pairs constrain every original box corner: the nearest
    corner can switch during rotation while its single native row is feasible.
    All rows remain planning displacement constraints, not physical contacts.
    """

    def compute_qp_inequalities(self, configuration, dt):
        native = super().compute_qp_inequalities(configuration, dt)
        m, data = self.model, configuration.data
        rows, bounds = [], []
        signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
        for a, b in self.geom_id_pairs:
            if (m.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE
                    and m.geom_type[a] == mujoco.mjtGeom.mjGEOM_BOX):
                a, b = b, a
            if (m.geom_type[a] != mujoco.mjtGeom.mjGEOM_PLANE
                    or m.geom_type[b] != mujoco.mjtGeom.mjGEOM_BOX
                    or m.body_weldid[m.geom_bodyid[a]] != 0):
                continue
            normal = data.geom_xmat[a].reshape(3, 3)[:, 2]
            corners = signs*m.geom_size[b]
            world = corners@data.geom_xmat[b].reshape(3, 3).T+data.geom_xpos[b]
            distance = (world-data.geom_xpos[a])@normal
            if distance.min() >= self.collision_detection_distance:
                continue
            for point, height in zip(world, distance):
                jacobian = np.zeros((3, m.nv))
                mujoco.mj_jac(m, data, jacobian, None, point, int(m.geom_bodyid[b]))
                rows.append(-normal@jacobian)
                bounds.append(self.gain*max(height-self.minimum_distance_from_collisions, 0.)
                    +self.bound_relaxation*dt)
        if not rows:
            return native
        if native.inactive:
            return Constraint(G=np.asarray(rows), h=np.asarray(bounds))
        return Constraint(G=np.vstack((native.G, rows)), h=np.r_[native.h, bounds])

    def _construct_geom_id_pairs(self, geom_pairs):
        pairs = set(super()._construct_geom_id_pairs(geom_pairs))
        m = self.model
        excluded = set(map(int, m.exclude_signature))
        for left, right in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for a, b in itertools.product(left, right):
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
                compatible = bool((m.geom_contype[a]&m.geom_conaffinity[b])
                                  or (m.geom_contype[b]&m.geom_conaffinity[a]))
                low, high = sorted((ba, bb))
                if ((wa == 0) != (wb == 0)) and compatible and (low << 16)+high not in excluded:
                    pairs.add(tuple(sorted((a, b))))
        return list(pairs)
