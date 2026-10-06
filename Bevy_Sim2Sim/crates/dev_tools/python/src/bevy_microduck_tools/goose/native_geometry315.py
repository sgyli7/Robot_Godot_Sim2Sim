"""Version-pinned collision support for the native MuJoCo3.15 candidate.

The compiled graph schema was checked against upstream3.15
engine_collision_convex.c. Old3.10 exports retain their separate guard.
"""
from __future__ import annotations

import mujoco
import numpy as np

ENGINE_VERSION = "3.15.0"
EXPORT_REVISION = "goose_native_convex_support315_v1"


def require_supported_engine() -> None:
    if mujoco.__version__ != ENGINE_VERSION:
        raise ValueError(f"Collision support export requires MuJoCo {ENGINE_VERSION}")


def collision_mesh_vertices(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, dict]:
    """Select mjc_initCCDObj support points, including its small-mesh fallback."""
    require_supported_engine()
    if not 0 <= mesh_id < model.nmesh:
        raise ValueError("Invalid compiled mesh id")
    address, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[address:address+count].astype(float)
    graph_address = int(model.mesh_graphadr[mesh_id])
    source = "all_vertices"
    if graph_address >= 0 and count >= 10:
        graph = model.mesh_graph
        support_count = int(graph[graph_address])
        first = graph_address+2+support_count
        last = first+support_count
        if support_count < 4 or last > len(graph):
            raise ValueError("Invalid compiled collision graph")
        indices = graph[first:last]
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("Compiled support index outside mesh")
        vertices = vertices[np.sort(indices)]
        source = "compiled_convex_graph"
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Invalid compiled collision support vertices")
    return vertices, {"source": source, "raw_vertex_count": count,
                      "support_vertex_count": len(vertices)}


def collision_geom_vertices(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return compiled geom-local mesh support or native box corners."""
    require_supported_engine()
    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return collision_mesh_vertices(model, int(model.geom_dataid[geom_id]))[0]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        return np.asarray([[sx*size[0], sy*size[1], sz*size[2]]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    raise ValueError("Collision support export requires native mesh or box")
