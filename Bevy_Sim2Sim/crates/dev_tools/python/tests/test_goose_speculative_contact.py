"""Actual frozen hulls and native constraint seam; external models stay in backups."""
import copy
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.speculative_contact import (
    build_reference, make_source_runtime, validate_prediction)


@pytest.fixture
def prediction(tmp_path, monkeypatch):
    package = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not package:
        pytest.skip("Verified external 004 package required")
    monkeypatch.chdir(tmp_path)
    package = Path(package).resolve(strict=True)
    return build_reference(package / "robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml",
        package / "robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        package, tmp_path / "prediction")


def test_native_pair_mixing_and_scene_identity(prediction):
    from mjlab.scene import Scene
    from bevy_microduck_tools.goose.mjlab_env import make_development_env_cfg
    model, path = prediction
    native = mujoco.MjModel.from_xml_path(str(model))
    cfg = make_development_env_cfg(model, path)
    scene = Scene(cfg.scene, "cpu").compile()
    cfg.sim.mujoco.apply(scene)
    validate_prediction(scene, json.loads(path.read_text()), entity_prefix="robot/")
    for name in ("pair_margin", "pair_gap", "pair_solref", "pair_solimp", "pair_friction",
                 "body_mass", "body_inertia", "body_ipos", "mesh_vert", "geom_margin",
                 "exclude_signature", "eq_data", "actuator_trnid"):
        np.testing.assert_array_equal(getattr(scene, name), getattr(native, name))
    np.testing.assert_array_equal(native.pair_solimp,
        np.tile([.925, .97, .001, .5, 2.], (9, 1)))
    contract = json.loads(path.read_text())
    assert not contract["training_release"] and not contract["upstream_baseline"]["source_qualified"]
    for mutation in ("material_surface_offset_m", "discovery_band_m"):
        bad = copy.deepcopy(contract)
        bad["ground_prediction"][mutation] += .001
        with pytest.raises(ValueError):
            validate_prediction(native, bad)
    native.pair_margin[0] += .001
    with pytest.raises(ValueError):
        validate_prediction(native, contract)


def test_actual_positive_gap_cpu_warp_reference_without_integrating(prediction):
    import mujoco_warp as mjw
    import warp as wp
    from bevy_microduck_tools.goose.speculative_contact_gpu import SpeculativeContactAdapter
    model, path = prediction
    runtime = make_source_runtime(model, path)
    m, d = runtime.model, runtime.data
    # Actual front-fallen cold pose, never a runtime correction.
    d.qpos[3:7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    from sai_agent.goose.convex_support import compiled_body_vertices
    mujoco.mj_kinematics(m, d)
    material_z = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            bid = m.geom_bodyid[g]
            points = compiled_body_vertices(m, g) @ d.xmat[bid].reshape(3, 3).T + d.xpos[bid]
            material_z.append(float(points[:, 2].min()))
    d.qpos[2] += .002 - min(material_z)
    mujoco.mj_step1(m, d)
    with wp.ScopedDevice("cpu"):
        flags = m.opt.disableflags
        m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        try:
            wm = mjw.put_model(m)
        finally:
            m.opt.disableflags = flags
        wd = mjw.put_data(m, d, nconmax=128, njmax=512)
        adapter = SpeculativeContactAdapter(wm, wd, m, runtime.contract)
        adapter.apply()
        adapter.assert_valid()
        runtime._planar_sole_quadrature()
        contacts = {n: getattr(wd.contact, n).numpy() for n in ("geom", "dist", "pos", "efc_address")}
        ref, vel, pos = wd.efc.aref.numpy()[0], wd.efc.vel.numpy()[0], wd.efc.pos.numpy()[0]
        expected = {(tuple(c.geom), tuple(np.round(c.pos, 5))): c for c in d.contact
                    if c.efc_address >= 0 and runtime.ground in c.geom
                    and not any(g in runtime.foot_geoms for g in c.geom)}
        compared = 0
        for i in range(int(wd.nacon.numpy()[0])):
            key = (tuple(contacts["geom"][i]), tuple(np.round(contacts["pos"][i].astype(float), 5)))
            if key not in expected:
                continue
            compared += 1
            row = contacts["efc_address"][i, 0]
            gap = contacts["dist"][i]
            np.testing.assert_allclose(ref[row], -vel[row]/.02-gap/.02**2, atol=3e-4, rtol=2e-6)
            np.testing.assert_allclose(pos[row], gap, atol=1e-7)
            np.testing.assert_allclose(ref[row], d.efc_aref[expected[key].efc_address], atol=3e-4, rtol=2e-6)
        assert compared == len(expected) > 0
        assert d.time == float(wd.time.numpy()[0]) == runtime.physics_integrations == 0
