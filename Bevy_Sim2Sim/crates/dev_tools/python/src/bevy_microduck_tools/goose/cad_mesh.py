"""CAD contact inputs only; body mass and inertia never come from these meshes."""

import numpy as np


def restore_triangle_cells(vertices, quads, tolerance_m=1e-9):
    """Reverse Goose's three planar quad cells per source triangle exactly.

    Reject any other layout, winding, or displaced midpoint/centre. This is
    lossless representation recovery, not decimation or a topology repair.
    Coordinates stay in the CAD frame before the assembly lift.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    quads = np.asarray(quads)
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
            or quads.ndim != 2 or quads.shape[1] != 4
            or not np.issubdtype(quads.dtype, np.integer)
            or not len(quads) or len(quads) % 3
            or tolerance_m <= 0 or not np.isfinite(tolerance_m)
            or np.min(quads) < 0 or np.max(quads) >= len(vertices)):
        raise ValueError("Invalid source quad tessellation")
    cells = quads.reshape(-1, 3, 4)
    if (not np.all(cells[:, :, 2] == cells[:, :1, 2])
            or not np.all(cells[:, :, 1] == np.roll(cells[:, :, 3], -1, axis=1))):
        raise ValueError("Source is not the declared shared-edge triangle-cell layout")
    corners = cells[:, :, 0]
    if np.any(np.diff(np.sort(corners, axis=1), axis=1) == 0):
        raise ValueError("Source triangle has repeated corners")
    maximum = 0.0
    for start in range(0, len(cells), 16384):
        section = cells[start:start + 16384]
        points = vertices[section[:, :, 0]]
        midpoints = (points + np.roll(points, -1, axis=1)) * 0.5
        maximum = max(maximum, float(np.max(np.linalg.norm(
            vertices[section[:, :, 1]] - midpoints, axis=2))))
        maximum = max(maximum, float(np.max(np.linalg.norm(
            vertices[section[:, 0, 2]] - points.mean(axis=1), axis=1))))
    if maximum > tolerance_m:
        raise ValueError("Source triangle cells are not planar midpoint subdivisions")
    import trimesh

    mesh = trimesh.Trimesh(vertices.copy(), corners.copy(), process=False)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Restored source triangle mesh is not a closed positive solid")
    return mesh, dict(method="inverse_shared_edge_triangle_quad_cells",
                     source_quad_count=len(quads), restored_triangle_count=len(corners),
                     maximum_midpoint_centre_error_m=maximum, geometric_decimation=False)


def native_surface_mesh(brep_path, *, deflection_mm, angle_rad, seam_digits=10):
    """Tessellate a read-only millimetre BREP with absolute OCCT deflection.

    The caller verifies native identity and independently screens the returned
    SI mesh. Coincident face-edge nodes are joined; no coordinates are moved,
    holes filled, normals repaired, or source BREP saved.
    """
    if (not 0 < deflection_mm < 1 or not 0 < angle_rad < 1
            or seam_digits != 10):
        raise ValueError("Invalid frozen contact-input meshing parameters")
    from build123d import import_brep
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    import trimesh

    shape = import_brep(str(brep_path))
    if not shape.is_valid or len(shape.solids()) != 1 or shape.volume <= 0:
        raise ValueError("Native input must be one valid positive solid")
    # Match the source manifest's adaptive B-spline volume measurement. The
    # default shape.volume integral can bias thin skins; this is diagnostic
    # geometry volume only and does not replace the robot's frozen SI ledger.
    properties = GProp_GProps()
    integration_error = BRepGProp.VolumeProperties_s(
        shape.wrapped, properties, 1e-7, True, False)
    volume_mm3 = float(properties.Mass())
    if not np.isfinite(volume_mm3) or volume_mm3 <= 0:
        raise ValueError("Native adaptive volume is invalid")
    # Discard cached display triangulations only; do not alter native surfaces.
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(
        shape.wrapped, deflection_mm, False, angle_rad, False)
    status = int(mesher.GetStatusFlags())
    if status or not mesher.IsDone():
        raise ValueError(f"OCCT triangulation failed: status={status}")
    # build123d's extractor reuses this absolute mesh when its tolerance passes.
    vertices, faces = shape.tessellate(deflection_mm, angle_rad)
    points = np.asarray([tuple(v) for v in vertices], dtype=np.float64) / 1000
    mesh = trimesh.Trimesh(points, faces, process=False)
    original_node_count = len(points)
    mesh.merge_vertices(digits_vertex=seam_digits)
    mesh.remove_unreferenced_vertices()
    return shape, mesh, dict(method="OCCT_absolute_BREP_surface_triangulation",
                             native_volume_mm3=volume_mm3, native_face_count=len(shape.faces()),
                             native_volume_integration_error=float(integration_error),
                             native_volume_integration_epsilon=1e-7,
                             status_flags=status, deflection_mm=deflection_mm, relative=False,
                             angular_deflection_rad=angle_rad, pre_seam_node_count=original_node_count,
                             vertices=len(mesh.vertices), triangles=len(mesh.faces),
                             seam_merge_digits_in_m=seam_digits, geometry_repaired=False)
