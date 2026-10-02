"""Use native distance queries to check source/target collision shape parity."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from bevy_microduck_tools.goose.artifacts import export_rapier_plant, sha256, write_json
from bevy_microduck_tools.goose.native_geometry import (
    ENGINE_VERSION, EXPORT_REVISION, collision_mesh_vertices)


pytestmark = pytest.mark.skipif(mujoco.__version__ != ENGINE_VERSION,
                               reason="Collision graph regression pins MuJoCo 3.10.0")


def mesh_model(points, cap, position=(.23, -.17, .11), quaternion=(1., 0., 0., 0.)):
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", ccd_tolerance="1e-12", ccd_iterations="200")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="shape", maxhullvert=str(cap),
                  vertex=" ".join(map(str, points.ravel())))
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="torso")
    ET.SubElement(body, "geom", name="shape", type="mesh", mesh="shape",
                  pos=" ".join(map(str, position)), quat=" ".join(map(str, quaternion)))
    ET.SubElement(world, "geom", name="probe", type="sphere", size=".0001", pos="1 0 0")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml


def native_distance(model, data, point):
    model.geom_pos[model.geom("probe").id] = point
    mujoco.mj_forward(model, data)
    return mujoco.mj_geomDistance(model, data, model.geom("shape").id,
                                 model.geom("probe").id, 1., None)


@pytest.mark.parametrize("cap,rings,expected_count", [(8, 32, 8), (-1, 32, 64), (4, 4, 8)])
def test_exported_support_reproduces_native_distance_with_vertex_caps(cap, rings, expected_count):
    points = np.array([[.1*np.cos(t), .07*np.sin(t), z] for z in (-.03, .03)
                       for t in np.linspace(0., 2.*np.pi, rings, endpoint=False)])
    source, data, _ = mesh_model(points, cap, quaternion=(.9238795325, 0., .3826834324, 0.))
    gid = source.geom("shape").id
    support, metadata = collision_mesh_vertices(source, int(source.geom_dataid[gid]))
    assert len(support) == expected_count
    assert metadata["raw_vertex_count"] == 2*rings
    exported, exported_data, _ = mesh_model(support, -1, source.geom_pos[gid], source.geom_quat[gid])
    raw = source.mesh_vert.copy().astype(float)
    world_points = raw @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    # Query separation away from vertex-touch EPA degeneracy. The reconstruction
    # recompiles float mesh coordinates; compare geometric distances, not solver
    # iteration paths through a vertex exactly on a tiny probe sphere.
    probes = data.geom_xpos[gid] + 1.3*(world_points-data.geom_xpos[gid])
    for point in probes:
        assert native_distance(exported, exported_data, point) == pytest.approx(
            native_distance(source, data, point), abs=2e-8)
    if cap == 8:
        # Independent native oracle: the old full-vertex export collides where
        # the capped source hull has a positive separation of several cm.
        planes = ConvexHull(support).equations
        separation = (raw @ planes[:, :3].T + planes[:, 3]).max(axis=1)
        point = world_points[int(separation.argmax())]
        old, old_data, _ = mesh_model(raw, -1, source.geom_pos[gid], source.geom_quat[gid])
        assert native_distance(source, data, point) > .03
        assert native_distance(old, old_data, point) < 0.


def test_plant_export_records_actual_support_and_keeps_compiled_frame(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    points = np.array([[.1*np.cos(t), .07*np.sin(t), z] for z in (-.03, .03)
                       for t in np.linspace(0., 2.*np.pi, 32, endpoint=False)])
    model, _, xml = mesh_model(points, 8)
    model_path, contract_path, output = (tmp_path / n for n in ("robot.xml", "contract.json", "plant.json"))
    model_path.write_text(xml)
    # Small geometry conformance rig, with no motor or task qualification.
    write_json(contract_path, {"bodies": [], "joints": [], "passive_contacts": [],
        "collision_geometries": [{"name": "shape"}], "beak_transmission": {
            "closed_crank_angle_in_xz_rad": 0., "crank_radius_m": .01,
            "jaw_axis_world_m": [0., 0., 0.]}})
    bundle = {"candidate": "geometry_conformance", "model_path": str(model_path),
              "contract_path": str(contract_path), "model_sha256": sha256(model_path),
              "source_contract_sha256": sha256(contract_path)}
    export_rapier_plant(bundle, output)
    plant = json.loads(output.read_text())
    collider = plant["colliders"][0]
    expected, _ = collision_mesh_vertices(model, 0)
    np.testing.assert_array_equal(collider["vertices_local_m"], expected)
    gid = model.geom("shape").id
    np.testing.assert_array_equal(collider["local_position_m"], model.geom_pos[gid])
    np.testing.assert_array_equal(collider["local_rotation_wxyz"], model.geom_quat[gid])
    assert collider["native_support"]["support_vertex_count"] == 8
    assert plant["geometry_export"]["revision"] == EXPORT_REVISION
    assert plant["geometry_export"]["reduced_mesh_count"] == 1
    assert plant["geometry_export"]["exported_vs_compiled_frame_max_error"] < 1e-10


def test_unsupported_backend_cannot_silently_reinterpret_collision_graph(monkeypatch):
    points = np.array([[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.1, .1)])
    model, _, _ = mesh_model(points, -1)
    monkeypatch.setattr(mujoco, "__version__", "3.13.0")
    with pytest.raises(ValueError, match="requires MuJoCo 3.10.0"):
        collision_mesh_vertices(model, 0)
