"""Collision support geometry for the pinned native MuJoCo source backend."""
from __future__ import annotations

import mujoco
import numpy as np


ENGINE_VERSION = "3.10.0"
EXPORT_REVISION = "goose_native_convex_support_v2"


def require_supported_engine() -> None:
    if mujoco.__version__ != ENGINE_VERSION:
        raise ValueError(f"Collision support export requires MuJoCo {ENGINE_VERSION}")


def collision_mesh_vertices(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, dict]:
    """Return the native support points in the compiled geom-local frame.

    MuJoCo retains raw vertices even when maxhullvert limits the collision hull.
    Follow mjc_initCCDObj's graph selection, including its <10-vertex fallback.
    The internal graph layout is version pinned; do not infer it on a new backend.
    Reference: google-deepmind/mujoco 3.10.0, engine_collision_convex.c/.h.
    """
    require_supported_engine()
    if not 0 <= mesh_id < model.nmesh:
        raise ValueError("Invalid compiled mesh id")
    address, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[address:address + count].astype(float)
    graph_address = int(model.mesh_graphadr[mesh_id])
    source = "all_vertices"
    if graph_address >= 0 and count >= 10:
        graph = model.mesh_graph
        support_count = int(graph[graph_address])
        first = graph_address + 2 + support_count
        last = first + support_count
        if support_count < 4 or last > len(graph):
            raise ValueError("Invalid compiled collision graph")
        indices = graph[first:last]
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("Compiled support index outside mesh")
        # Preserve raw order for retained points; graph traversal order is not
        # part of the shape and must not reorder otherwise unchanged target hulls.
        vertices = vertices[np.sort(indices)]
        source = "compiled_convex_graph"
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Invalid compiled collision support vertices")
    return vertices, {"source": source, "raw_vertex_count": count,
                      "support_vertex_count": len(vertices)}


def collision_geom_vertices(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Support vertices in geom-local coordinates for mesh or native box."""
    require_supported_engine()
    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return collision_mesh_vertices(model, int(model.geom_dataid[geom_id]))[0]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        return np.asarray([[sx*size[0], sy*size[1], sz*size[2]]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    raise ValueError("Collision support export requires native mesh or box")
