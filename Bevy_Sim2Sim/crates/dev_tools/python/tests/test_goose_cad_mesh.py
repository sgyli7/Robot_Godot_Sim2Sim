"""Guard exact source recovery and the native cavity-preserving input seam."""

import numpy as np
import pytest
import trimesh

from bevy_microduck_tools.goose.cad_mesh import native_surface_mesh, restore_triangle_cells


def triangle_cells(mesh):
    vertices = mesh.vertices.tolist()
    quads = []
    shared = {}
    for a, b, c in mesh.faces:
        mids = []
        for i, j in [(a, b), (b, c), (c, a)]:
            edge = tuple(sorted((int(i), int(j))))
            if edge not in shared:
                shared[edge] = len(vertices)
                vertices.append(((mesh.vertices[i] + mesh.vertices[j]) / 2).tolist())
            mids.append(shared[edge])
        centre = len(vertices)
        vertices.append(mesh.vertices[[a, b, c]].mean(axis=0).tolist())
        ab, bc, ca = mids
        quads.extend([(a, ab, centre, ca), (b, bc, centre, ab), (c, ca, centre, bc)])
    return np.asarray(vertices), np.asarray(quads)


def test_recovery_keeps_cad_before_lift_and_exact_closed_surface():
    source = trimesh.creation.box(extents=[0.02, 0.03, 0.004])
    source.apply_translation([0.18, -0.07, 0.55])
    mesh, metadata = restore_triangle_cells(*triangle_cells(source))
    assert np.array_equal(mesh.triangles, source.triangles)
    assert mesh.volume == pytest.approx(source.volume, rel=1e-12)
    assert mesh.euler_number == source.euler_number
    assert metadata["geometric_decimation"] is False


@pytest.mark.parametrize("damage", ["midpoint", "centre", "index", "layout", "open", "reverse"])
def test_recovery_rejects_corrupted_source_cells(damage):
    source = trimesh.creation.box(extents=[0.02, 0.03, 0.004])
    if damage == "reverse":
        source.invert()
    vertices, quads = triangle_cells(source)
    if damage == "midpoint":
        vertices[quads[0, 1], 2] += 0.0001
    elif damage == "centre":
        vertices[quads[0, 2], 2] += 0.0001
    elif damage == "index":
        quads[0, 0] = -1
    elif damage == "layout":
        quads[[0, 1]] = quads[[1, 0]]
    elif damage == "open":
        quads = quads[:-3]
    with pytest.raises(ValueError):
        restore_triangle_cells(vertices, quads)


def test_native_input_keeps_cylinder_port_and_source_bytes(tmp_path):
    from build123d import Cylinder, export_brep

    shape = Cylinder(20, 10) - Cylinder(10, 20)
    path = tmp_path / "native.brep"
    export_brep(shape, path)
    before = path.read_bytes()
    native, mesh, metadata = native_surface_mesh(path, deflection_mm=0.2, angle_rad=0.3)
    assert path.read_bytes() == before
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.euler_number == 0
    assert not mesh.contains([[0, 0, 0]])[0]
    assert mesh.contains([[0.015, 0, 0]])[0]
    assert abs(mesh.volume / (native.volume / 1e9) - 1) < 0.02
    assert metadata["relative"] is False
    assert metadata["geometry_repaired"] is False
