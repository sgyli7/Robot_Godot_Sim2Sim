"""Native mass/inertia, grip axis and hollow-handle checks; no holding claim."""
import mujoco
import numpy as np
import pytest
from mjlab.entity import EntityCfg, Entity

from bevy_microduck_tools.goose.move_payloads import (
    MovePayload, PayloadContact, make_move_payload_spec, make_move_payload_entity_cfg,
)


def contact():
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="mouth", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(.02, .03, .004), contype=2, conaffinity=3,
        friction=(.65, .01, .002), solref=(.005, 1.), margin=0.)
    return PayloadContact.from_model(spec.compile(), "mouth")


@pytest.mark.parametrize("family,leaves", [("cylinder_grip", 3), ("open_handle", 4)])
@pytest.mark.parametrize("grams", [100, 200, 300, 500])
def test_native_free_payload_has_declared_mass_positive_inertia_and_grip_axis(family, leaves, grams):
    surface = contact()
    m = make_move_payload_spec(payload=MovePayload(family, grams), contact=surface).compile()
    d = mujoco.MjData(m)
    mujoco.mj_fwdPosition(m, d)
    body, grip = m.body("object").id, m.geom("grip").id
    assert m.njnt == 1 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.nq == 7 and m.nv == 6 and m.neq == m.nu == 0
    assert m.ngeom == leaves and m.body_mass[body] == pytest.approx(grams/1000., abs=1e-12)
    assert (m.body_inertia[body] > 0).all() and np.isfinite(m.body_ipos[body]).all()
    np.testing.assert_allclose(d.geom_xmat[grip].reshape(3, 3)[:, 2], [0., 1., 0.], atol=1e-12)
    for key in ("friction", "solref", "solimp"):
        np.testing.assert_allclose(getattr(m, "geom_"+key),
            np.broadcast_to(dict(surface.fields)[key], getattr(m, "geom_"+key).shape), atol=0.)
    assert ((m.geom_contype == 2) & (m.geom_conaffinity == 3)).all()


def test_actual_handle_opening_is_empty_and_bars_remain_collidable():
    m = make_move_payload_spec(payload=MovePayload("open_handle", 300), contact=contact()).compile()
    d = mujoco.MjData(m)
    mujoco.mj_fwdPosition(m, d)
    geomid = np.zeros(1, dtype=np.int32)
    hole = mujoco.mj_ray(m, d, np.array([.025, 0., -.1]), np.array([0., 0., 1.]),
                       None, 1, -1, geomid)
    assert hole == -1
    bar = mujoco.mj_ray(m, d, np.array([0., 0., -.1]), np.array([0., 0., 1.]),
                      None, 1, -1, geomid)
    assert bar > 0 and geomid[0] == m.geom("grip").id


def test_native_entity_owns_only_reset_pose_not_robot_attachment():
    state = EntityCfg.InitialStateCfg(pos=(1., 2., .5))
    cfg = make_move_payload_entity_cfg(payload=MovePayload("cylinder_grip", 100),
                                      contact=contact(), initial_state=state)
    native = Entity(cfg)
    model = native.spec.compile()
    assert model.njnt == 1 and model.neq == model.nu == 0
    assert cfg.init_state is state


@pytest.mark.parametrize("family,mass", [("unknown", 100), ("cylinder_grip", True),
                                       ("open_handle", 0), ("open_handle", 1000)])
def test_unnamed_payload_or_load_is_rejected(family, mass):
    with pytest.raises(ValueError, match="named family"):
        MovePayload(family, mass)
