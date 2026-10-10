"""Query compiled native terrain surfaces and actual material inheritance."""
import mujoco
import numpy as np
import pytest
from mjlab.terrains.terrain_entity import TerrainEntity

from bevy_microduck_tools.goose.move_terrain_primitives import (
    MoveRampTerrainCfg, make_move_primitive_terrain,
    make_move_continuous_primitive_terrain,
)


def source_ground():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="ground" type="plane" size="2 2 .01" contype="1"
        conaffinity="2" friction=".65 .01 .002" solref=".005 1"/>
        </worldbody></mujoco>''')


@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("difficulty", [0., 1.])
def test_real_native_ramp_rays_have_declared_grade_and_continuous_seams(inverted, difficulty):
    cfg = MoveRampTerrainCfg(inverted=inverted)
    cfg.size = (4., 4.)
    spec = mujoco.MjSpec(); spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty, spec, np.random.default_rng(6401))
    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    assert model.nhfield == 0 and model.ngeom == 3
    assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    def height(x):
        hit = np.full(1, -1, np.int32)
        distance = mujoco.mj_ray(model, data, np.array([x, 2., 1.]),
            np.array([0., 0., -1.]), np.ones(mujoco.mjNGROUP, np.uint8), 1, -1, hit)
        assert distance >= 0 and hit[0] >= 0
        return 1. - distance
    xs = np.array([1.2, 1.8, 2.4, 2.8])
    grade = np.degrees(np.arctan(np.polyfit(xs, [height(x) for x in xs], 1)[0]))
    assert grade == pytest.approx((-5. if inverted else 5.) * difficulty, abs=1e-9)
    for seam in (1., 3.):
        assert abs(height(seam - 1e-7) - height(seam + 1e-7)) < 1e-7
    assert height(output.origin[0]) == pytest.approx(output.origin[2], abs=1e-12)


@pytest.mark.parametrize("height", [.005, .010, .020, .050])
def test_compiled_primitive_scene_preserves_materials_and_has_no_heightfields(height):
    source = source_ground()
    cfg = make_move_primitive_terrain(ground_model=source, seed=6402,
        height_m=height, evaluation=True)
    terrain = TerrainEntity(cfg, device="cpu")
    model = terrain.spec.compile()
    assert model.nhfield == 0 and model.ngeom > 40
    assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    for name in ("contype", "conaffinity", "condim", "priority", "group",
                 "friction", "solref", "solimp", "margin", "gap", "solmix"):
        actual = getattr(model, "geom_" + name)
        np.testing.assert_array_equal(actual,
            np.broadcast_to(getattr(source, "geom_" + name)[0], actual.shape))
    rough = cfg.terrain_generator.sub_terrains["random_rough"]
    rough.size = (4., 4.)
    spec = mujoco.MjSpec(); spec.worldbody.add_body(name="terrain")
    rough.function(1., spec, np.random.default_rng(6402))
    grid = spec.compile()
    tops = grid.geom_pos[:, 2] + grid.geom_size[:, 2]
    assert tops.max() == pytest.approx(height / 2)
    assert tops.min() >= -height / 2 - 1e-12
    assert np.ptp(tops) > .7 * height


@pytest.mark.parametrize("height", [.005, .020])
def test_continuous_grid_covers_the_transition_out_of_native_platform(height):
    terrain = make_move_continuous_primitive_terrain(ground_model=source_ground(),
        seed=6402, height_m=height, evaluation=True)
    rough = terrain.terrain_generator.sub_terrains["random_rough"]
    rough.size = (4., 4.)
    spec = mujoco.MjSpec(); spec.worldbody.add_body(name="terrain")
    output = rough.function(1., spec, np.random.default_rng(6402))
    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    # Sample the previously open platform ring and the rest of the tile.
    # Offset avoids exact shared-edge floating-point ray ambiguity.
    for x in np.linspace(.1, 3.9, 39) + 1e-6:
        for y in (1.45, 1.95, 2.05, 2.55):
            hit = np.full(1, -1, np.int32)
            distance = mujoco.mj_ray(model, data, np.array([x, y, 1.]),
                np.array([0., 0., -1.]), np.ones(mujoco.mjNGROUP, np.uint8), 1, -1, hit)
            assert distance >= 0 and hit[0] >= 0, (x, y)
            assert -height / 2 - 1e-12 <= 1. - distance <= height / 2 + 1e-12
    np.testing.assert_allclose(output.origin, [2., 2., height / 2], atol=1e-12)
    original = make_move_primitive_terrain(ground_model=source_ground(), seed=6402,
        height_m=height, evaluation=True)
    assert original.terrain_generator.sub_terrains["random_rough"].platform_width == 1.
