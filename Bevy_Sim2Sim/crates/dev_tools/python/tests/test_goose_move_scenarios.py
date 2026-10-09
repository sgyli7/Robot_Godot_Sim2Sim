"""Check generated terrain and actual integrated impulse, without PPO/GPU."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.move_scenarios import (
    HorizontalBodyImpulse, make_move_impulse, make_move_terrain,
)


def source_ground_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="ground" type="plane" size="2 2 .01" contype="1"
        conaffinity="2" friction=".65 .01 .002" solref=".005 1"/>
        </worldbody></mujoco>''')


@pytest.mark.parametrize("height", [.005, .010, .020, .050])
def test_formal_native_steps_reach_declared_height_and_keep_real_floor(height):
    cfg = make_move_terrain(ground_model=source_ground_model(), seed=6147,
        height_m=height, evaluation=True)
    gen = cfg.terrain_generator
    assert gen.difficulty_range == (1., 1.)
    sub = gen.sub_terrains["staggered_steps"]
    sub.size = gen.size
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = sub.function(1., spec, np.random.default_rng(gen.seed))
    model = spec.compile()
    top = model.geom_pos[:, 2] + model.geom_size[:, 2]
    assert np.count_nonzero(np.isclose(top, height, atol=1e-10)) >= 10
    assert np.count_nonzero(np.isclose(top, 0., atol=1e-10)) >= 1
    assert top.max() == pytest.approx(height)
    assert output.origin[2] == 0.
    # Scattered positions and yaw come from the actual native generator.
    assert np.ptp(model.geom_pos[top > 0., :2], axis=0).min() > 1.
    assert np.ptp(model.geom_quat[top > 0., 3]) > .05


def test_actual_native_terrain_entity_inherits_frozen_contact_parameters():
    from mjlab.terrains.terrain_entity import TerrainEntity
    source = source_ground_model()
    cfg = make_move_terrain(ground_model=source, seed=6150,
        height_m=.020, evaluation=True)
    terrain = TerrainEntity(cfg, device="cpu")
    model = terrain.spec.compile()
    assert model.nhfield == 3 and model.ngeom > 10
    for name in ("contype", "conaffinity", "condim", "priority", "group",
                 "friction", "solref", "solimp", "margin", "gap", "solmix"):
        actual = getattr(model, "geom_" + name)
        expected = np.broadcast_to(getattr(source, "geom_" + name)[0], actual.shape)
        np.testing.assert_array_equal(actual, expected)


class CpuWrenchAsset:
    """Real MuJoCo body plus the upstream event's ordinary wrench API."""

    def __init__(self, model, data):
        self.model, self.mj_data = model, data
        self.num_bodies = 1
        self.data = SimpleNamespace(
            body_external_wrench=torch.zeros(1, 1, 6),
            body_com_quat_w=torch.tensor([[[1., 0., 0., 0.]]]))

    def write_external_wrench_to_sim(self, forces, torques, *, env_ids, body_ids):
        assert list(body_ids) == [0] and env_ids.tolist() == [0]
        self.data.body_external_wrench[env_ids, 0, :3] = forces[:, 0]
        self.data.body_external_wrench[env_ids, 0, 3:] = torques[:, 0]
        self.mj_data.xfrc_applied[1] = self.data.body_external_wrench[0, 0].numpy()


@pytest.mark.parametrize("hz", [50, 200])
@pytest.mark.parametrize("impulse", [.5, 1., 3.])
def test_native_pulse_delivers_same_horizontal_momentum_at_both_frequencies(hz, impulse):
    model = mujoco.MjModel.from_xml_string(f'''
        <mujoco><option timestep="{1/hz}" gravity="0 0 0"/>
        <worldbody><body name="robot"><freejoint/>
        <inertial pos="0 0 0" mass="10.430690821" diaginertia=".1 .1 .1"/>
        <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
        </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    asset = CpuWrenchAsset(model, data)
    env = SimpleNamespace(scene={"robot": asset}, step_dt=.02, num_envs=1, device="cpu")
    cfg = make_move_impulse(impulse_ns=impulse, body_point_offset=(0., 0., .1))
    cfg.params["asset_cfg"].body_ids = [0]
    torch.manual_seed(6148)
    event = HorizontalBodyImpulse(cfg, env)
    event._interval_time_left.zero_()  # Explicitly trigger this isolated fixture.
    event(env, None, **cfg.params)
    force = data.xfrc_applied[1, :3].copy()
    assert force[2] == 0.
    assert np.linalg.norm(force) == pytest.approx(impulse/.02, rel=1e-6)
    np.testing.assert_allclose(data.xfrc_applied[1, 3:],
        np.cross([0., 0., .1], force), rtol=1e-6, atol=1e-7)
    applied = np.zeros(3)
    for _ in range(hz//50):
        applied += data.xfrc_applied[1, :3] * model.opt.timestep
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(.02)
    assert np.linalg.norm(applied) == pytest.approx(impulse, rel=1e-6)
    np.testing.assert_allclose(data.qvel[:3] * model.body_mass[1], applied,
        rtol=1e-6, atol=1e-7)
    event(env, None, **cfg.params)
    assert not bool(event._active[0])
    assert not data.xfrc_applied[1].any()  # No unintended second control Tick.
