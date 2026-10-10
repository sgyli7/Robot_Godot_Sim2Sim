"""Read-only terrain queries matching infinite physical MuJoCo planes.

MuJoCo's plane ray query uses the finite visual extent. Plane collisions do
not. Intersect actual plane poses analytically and retain native queries for
all other terrain shapes. Only a private query model has its groups changed.
This observer never advances, edits, or qualifies the physical simulation.
"""

import copy

import mujoco
import numpy as np


class TerrainSurfaceQuery:
    """Filter actual terrain roles without changing the live collision model."""

    def __init__(self, model, terrain_geom_ids):
        self.model = copy.copy(model)
        ids = np.asarray(terrain_geom_ids, dtype=np.int32)
        if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= model.ngeom)):
            raise ValueError("Declare distinct, valid terrain geometry IDs")
        self.model.geom_group[:] = 1
        self.model.geom_group[ids] = 0
        self.data = mujoco.MjData(self.model)
        self.groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
        self.groups[0] = 1
        self.plane_ids = ids[self.model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_PLANE]

    def update(self, qpos):
        """Set private kinematics to the observed physical state, without a step."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("Expected one finite, complete physical qpos")
        self.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.model, self.data)

    def ray(self, origin, direction):
        """Return distance and actual terrain ID; (-1, -1) means no surface."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        if (origin.shape != (3,) or direction.shape != (3,)
                or not np.isfinite(origin).all() or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1., rtol=0., atol=1e-8)):
            raise ValueError("Ray origin must be finite and direction a unit vector")
        hit = np.full(1, -1, dtype=np.int32)
        distance = float(mujoco.mj_ray(self.model, self.data, origin, direction,
                                     self.groups, 1, -1, hit))
        geom_id = int(hit[0])
        for plane_id in self.plane_ids:
            normal = self.data.geom_xmat[plane_id].reshape(3, 3)[:, 2]
            denominator = float(normal @ direction)
            # Physical plane support is its positive normal face. A ray
            # parallel to it or leaving that face has no support intersection.
            if denominator >= -1e-12:
                continue
            candidate = float(normal @ (self.data.geom_xpos[plane_id] - origin)
                              / denominator)
            if candidate >= 0. and (distance < 0. or candidate < distance):
                distance, geom_id = candidate, int(plane_id)
        return distance, geom_id
