"""Static native hfield geometry and scene seams, with no physics rollout."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.terrain_entity import TerrainEntity

from bevy_microduck_tools.goose.move_continuous_hfield import (
    SignedContinuousHfieldCfg, make_move_signed_continuous_hfield,
    with_move_signed_continuous_hfield,
)


def compile_profile(profile, difficulty=1., grid_points=257):
    cfg = SignedContinuousHfieldCfg(profile=profile, grid_points=grid_points)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty, spec, np.random.default_rng(6272))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    return cfg, output, model, data


def surface(model, data, x, y, groups=None):
    hit = np.full(1, -1, np.int32)
    if groups is None:
        groups = np.ones(mujoco.mjNGROUP, np.uint8)
    distance = mujoco.mj_ray(model, data, np.array([x, y, 1.]),
        np.array([0., 0., -1.]), groups, 1, -1, hit)
    assert distance >= 0 and hit[0] >= 0, (x, y)
    return 1. - distance, int(hit[0])


def test_native_bump_has_twenty_mm_peak_and_flat_entry_exit():
    _, output, model, data = compile_profile("bump20")
    assert model.nhfield == model.ngeom == 1
    assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.hfield_nrow[0] == model.hfield_ncol[0] == 257
    np.testing.assert_array_equal(model.hfield_size[0, :2], [2., 2.])
    for x, y, expected in [(0.5, 2., 0.), (1.5, 2., .010), (2., 2., .020),
                           (2.5, 2., .010), (3.5, 2., 0.), (2., .5, 0.)]:
        assert surface(model, data, x, y)[0] == pytest.approx(expected, abs=2e-9)
    np.testing.assert_array_equal(output.origin, [.5, 2., 0.])
    assert data.time == 0.


def test_native_dip_keeps_negative_surface_without_a_filling_plane():
    _, _, model, data = compile_profile("dip20")
    assert model.nhfield == model.ngeom == 1
    for x, expected in [(.5, 0.), (1.5, -.010), (2., -.020),
                        (2.5, -.010), (3.5, 0.)]:
        z, hit = surface(model, data, x, 2.)
        assert z == pytest.approx(expected, abs=2e-9)
        assert model.geom_type[hit] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.geom_pos[0, 2] == pytest.approx(-.020)
    np.testing.assert_array_equal(model.hfield_size[0, 2:], [.020, .100])


def test_variable_surface_has_both_signed_five_degree_centreline_segments():
    _, _, model, data = compile_profile("variable_slope")
    dx = 4. / (257 - 1)
    # Independent native rays, one segment on each quarter of the two lobes.
    for x, degrees in [(1.171875, 5.), (1.546875, -5.),
                       (2.421875, -5.), (2.796875, 5.)]:
        z0, _ = surface(model, data, x, 2.)
        z1, _ = surface(model, data, x + dx, 2.)
        actual = np.degrees(np.arctan((z1 - z0) / dx))
        assert actual == pytest.approx(degrees, abs=1e-5)
    # The profile is asymmetric in X/Y: this catches transposed row/column data.
    positive = surface(model, data, 1.375, 2.)[0]
    negative = surface(model, data, 2.625, 2.)[0]
    assert positive > .020 and negative == pytest.approx(-positive, abs=2e-9)
    assert surface(model, data, 2., 1.375)[0] == pytest.approx(0., abs=2e-9)
    for x, y in [(.5, 2.), (3.5, 2.), (2., .5), (2., 3.5)]:
        assert surface(model, data, x, y)[0] == pytest.approx(0., abs=2e-9)


@pytest.mark.parametrize("profile", ["bump20", "dip20", "variable_slope"])
def test_manifest_samples_cover_native_endpoints_and_both_flat_strips(profile):
    cfg, _, model, data = compile_profile(profile)
    manifest = cfg.geometry_manifest()
    assert manifest["revision"] == "signed_continuous_hfield_v1"
    assert manifest["profile"] == profile
    assert manifest["sample_axes"] == {"row": "Y", "column": "X"}
    assert manifest["grid_spacing_m"] == 4. / (257 - 1)
    assert manifest["domain_xy_m"] == [[0., 4.], [0., 4.]]
    samples = cfg.height_samples()
    assert manifest["height_samples_sha256"] == hashlib.sha256(
        samples.astype("<f8").tobytes()).hexdigest()
    import bevy_microduck_tools.goose.move_continuous_hfield as component
    assert manifest["source_sha256"] == hashlib.sha256(
        Path(component.__file__).read_bytes()).hexdigest()
    native = (model.geom_pos[0, 2]
              + model.hfield_size[0, 2] * model.hfield_data.reshape(257, 257))
    np.testing.assert_allclose(native, samples, atol=2e-9, rtol=0.)
    for row, column in [(0, 0), (256, 256), (77, 101), (128, 88)]:
        x, y = column * 4. / (257 - 1), row * 4. / (257 - 1)
        assert surface(model, data, x, y)[0] == pytest.approx(samples[row, column], abs=2e-9)
    for a in [.01, .5, 1., 3., 3.5, 3.99]:
        for b in [.25, 1.5, 2., 2.75, 3.75]:
            assert surface(model, data, a, b)[0] == pytest.approx(0., abs=2e-9)
            assert surface(model, data, b, a)[0] == pytest.approx(0., abs=2e-9)
    # The native surface is continuous across grid triangles and the compact
    # support boundary; it is not claimed to be a differentiable collider.
    for x in [1., 1.375, 1.75, 2.25, 2.625, 3.]:
        left = surface(model, data, x - 1e-8, 2.)[0]
        right = surface(model, data, x + 1e-8, 2.)[0]
        assert abs(left - right) < 4e-9


@pytest.mark.parametrize("profile", ["bump20", "dip20", "variable_slope"])
def test_native_zero_difficulty_is_flat_and_intermediate_levels_are_explicit(profile):
    cfg, _, model, data = compile_profile(profile, difficulty=0.)
    assert model.nhfield == model.ngeom == 1
    assert np.count_nonzero(model.hfield_data) == 0
    assert surface(model, data, 1.375, 2.)[0] == pytest.approx(0., abs=1e-12)
    level = cfg.geometry_manifest(1. / 3)
    assert level["difficulty"] == 1. / 3
    assert level["height_range_m"] == pytest.approx(
        [cfg.height_samples().min() / 3, cfg.height_samples().max() / 3])


@pytest.mark.parametrize("difficulty", [1. / 3, 2. / 3])
@pytest.mark.parametrize("profile", ["bump20", "dip20", "variable_slope"])
def test_native_intermediate_levels_keep_actual_signed_height_and_grade(profile, difficulty):
    cfg, _, model, data = compile_profile(profile, difficulty)
    if profile in ("bump20", "dip20"):
        target = (.020 if profile == "bump20" else -.020) * difficulty
        assert surface(model, data, 2., 2.)[0] == pytest.approx(target, abs=2e-9)
    else:
        dx = 4. / (257 - 1)
        z0 = surface(model, data, 1.171875, 2.)[0]
        z1 = surface(model, data, 1.1875, 2.)[0]
        actual = np.degrees(np.arctan((z1 - z0) / dx))
        expected = np.degrees(np.arctan(difficulty * np.tan(np.radians(5.))))
        assert actual == pytest.approx(expected, abs=1e-5)
    manifest = cfg.geometry_manifest(difficulty)
    actual = model.geom_pos[0, 2] + model.hfield_size[0, 2] * model.hfield_data
    assert [actual.min(), actual.max()] == pytest.approx(manifest["height_range_m"], abs=2e-9)


@pytest.mark.parametrize("difficulty", [True, -.01, 1.01, float("nan"), float("inf")])
def test_named_geometry_rejects_invalid_difficulty(difficulty):
    with pytest.raises(ValueError, match="difficulty"):
        SignedContinuousHfieldCfg().height_samples(difficulty)


def test_named_geometry_rejects_other_size_or_profile():
    with pytest.raises(ValueError, match="4m"):
        SignedContinuousHfieldCfg(size=(4., 3.)).height_samples()
    with pytest.raises(ValueError, match="profile"):
        SignedContinuousHfieldCfg(profile="random_rough").height_samples()


@pytest.mark.parametrize("grid_points", [True, 65., 33, 129])
def test_grid_resolution_requires_a_named_candidate(grid_points):
    with pytest.raises(ValueError, match="grid"):
        SignedContinuousHfieldCfg(grid_points=grid_points).height_samples()


@pytest.mark.parametrize("profile", ["bump20", "dip20", "variable_slope"])
def test_capacity_candidate_preserves_features_with_bounded_native_surface_error(profile):
    cfg, output, coarse, data = compile_profile(profile, grid_points=65)
    _, _, fine, fine_data = compile_profile(profile)
    manifest = cfg.geometry_manifest()
    assert manifest["revision"] == "signed_continuous_hfield_capacity_v2"
    assert manifest["grid_shape"] == [65, 65]
    assert coarse.hfield_nrow[0] == coarse.hfield_ncol[0] == 65
    assert coarse.nhfield == coarse.ngeom == 1
    assert coarse.geom_type[0] == mujoco.mjtGeom.mjGEOM_HFIELD
    np.testing.assert_array_equal(output.origin, [.5, 2., 0.])
    errors = [abs(surface(coarse, data, x, y)[0] - surface(fine, fine_data, x, y)[0])
              for x in np.linspace(1., 3., 65) for y in np.linspace(1., 3., 17)]
    assert max(errors) <= (.0015 if profile == "variable_slope" else .0002)
    for x, y in [(.5, 2.), (3.5, 2.), (2., .5), (2., 3.5)]:
        assert surface(coarse, data, x, y)[0] == pytest.approx(0., abs=2e-9)
    if profile in ("bump20", "dip20"):
        assert surface(coarse, data, 2., 2.)[0] == pytest.approx(
            -.020 if profile == "dip20" else .020, abs=2e-9)
    else:
        # Measure the actual native piecewise surface, not a title or ideal derivative.
        heights = np.array([surface(coarse, data, x, 2.)[0]
                            for x in np.linspace(0., 4., 65)])
        slopes = np.degrees(np.arctan(np.diff(heights) / .0625))
        assert [slopes.min(), slopes.max()] == pytest.approx([-5., 5.], abs=1e-5)
        assert heights.max() > .020 and heights.min() < -.020
    assert data.time == fine_data.time == 0.


def flat_fixture():
    source = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="2 2 .01" contype="1"
      conaffinity="2" condim="4" priority="2" group="3" friction=".65 .01 .002"
      solref=".005 1" solmix=".7" margin=".001" gap=".0001"/>
      </worldbody></mujoco>''')

    def robot_spec():
        return mujoco.MjSpec.from_string('''<mujoco><worldbody>
          <body name="torso" pos="0 0 .3"><freejoint name="root"/>
          <geom name="sole" type="box" size=".06 .04 .01" mass="1"
          contype="2" conaffinity="1" friction=".8 .01 .002"/>
          </body></worldbody></mujoco>''')

    def flat_scene(spec):
        spec.option.timestep = .005
        spec.option.iterations = 17
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(2., 2., .01), contype=1, conaffinity=2)

    sensor = ContactSensorCfg(name="ground_contact",
        primary=ContactMatch(mode="geom", entity="robot", pattern="sole"),
        secondary=ContactMatch(mode="geom", pattern="ground"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
        track_air_time=True, history_length=4)
    cfg = SimpleNamespace(scene=SceneCfg(num_envs=2,
        entities={"robot": EntityCfg(spec_fn=robot_spec)},
        sensors=(sensor,), spec_fn=flat_scene),
        policy_dt=.02, physics_dt=.005, actor_dim=65, action_dim=18,
        rewards={"native_recipe": "frozen"}, events={"unrelated_reset": "frozen"})
    return cfg, source


@pytest.mark.parametrize("evaluation", [False, True])
def test_native_generator_has_all_three_profiles_and_inherits_source_materials(evaluation):
    _, source = flat_fixture()
    cfg = make_move_signed_continuous_hfield(ground_model=source,
        seed=6272, evaluation=evaluation)
    generator = cfg.terrain_generator
    assert generator.size == (4., 4.) and generator.curriculum
    assert generator.num_rows == (1 if evaluation else 4)
    assert generator.num_cols == 3
    assert generator.difficulty_range == ((1., 1.) if evaluation else (0., 1.))
    assert [t.profile for t in generator.sub_terrains.values()] == [
        "bump20", "dip20", "variable_slope"]
    assert all(t.proportion == 1. / 3 for t in generator.sub_terrains.values())
    terrain = TerrainEntity(cfg, device="cpu")
    model = terrain.spec.compile()
    assert model.nhfield == (3 if evaluation else 12)
    assert np.sum(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD) == model.nhfield
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    for field in ("contype", "conaffinity", "condim", "priority", "group", "friction",
                  "solref", "solimp", "margin", "gap", "solmix"):
        actual = getattr(model, "geom_" + field)
        np.testing.assert_array_equal(actual, np.broadcast_to(
            getattr(source, "geom_" + field)[0], actual.shape))
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    origins = terrain.terrain_origins.numpy().reshape(-1, 3)
    for x, y, z in origins:
        assert z == 0. and surface(model, data, x, y)[0] == pytest.approx(0.)
    if evaluation:
        dip = model.geom("signed_continuous_hfield_v1_dip20_1_surface").id
        x, y = data.geom_xpos[dip, :2]
        assert surface(model, data, x, y)[0] == pytest.approx(-.020, abs=2e-9)
    assert data.time == 0.


def test_opt_in_scene_removes_plane_preserves_robot_control_and_native_origins():
    cfg, source = flat_fixture()
    old = Scene(cfg.scene, device="cpu").compile()
    candidate = with_move_signed_continuous_hfield(cfg, ground_model=source,
        seed=6272, evaluation=True)
    new = Scene(candidate.scene, device="cpu").compile()
    assert cfg.scene.terrain is None and cfg.events == {"unrelated_reset": "frozen"}
    assert cfg.scene.sensors[0].secondary.mode == "geom"
    assert candidate.scene.sensors[0].secondary == ContactMatch(mode="body", pattern="terrain")
    for field in ("primary", "fields", "reduce", "num_slots", "track_air_time", "history_length"):
        assert getattr(candidate.scene.sensors[0], field) == getattr(cfg.scene.sensors[0], field)
    for field in ("policy_dt", "physics_dt", "actor_dim", "action_dim", "rewards"):
        assert getattr(candidate, field) == getattr(cfg, field)
    reset = candidate.events["move_root_origin_reset"]
    assert reset.mode == "reset" and reset.func.__name__ == "reset_root_state_uniform"
    assert reset.params["pose_range"] == reset.params["velocity_range"] == {}
    assert candidate.events["unrelated_reset"] == "frozen"
    assert new.nhfield == 3 and not np.any(new.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert new.opt.timestep == old.opt.timestep == .005
    assert new.opt.iterations == old.opt.iterations == 17
    for field in ("mass", "inertia", "ipos", "iquat"):
        np.testing.assert_array_equal(getattr(new, "body_" + field)[new.body("robot/torso").id],
            getattr(old, "body_" + field)[old.body("robot/torso").id])
    for field in ("type", "pos", "axis", "range", "limited"):
        np.testing.assert_array_equal(getattr(new, "jnt_" + field), getattr(old, "jnt_" + field))
    for field in ("type", "size", "pos", "quat", "contype", "conaffinity", "friction", "solref", "solimp"):
        np.testing.assert_array_equal(getattr(new, "geom_" + field)[new.geom("robot/sole").id],
            getattr(old, "geom_" + field)[old.geom("robot/sole").id])
    data = mujoco.MjData(new)
    mujoco.mj_fwdPosition(new, data)
    groups = np.zeros(mujoco.mjNGROUP, np.uint8)
    groups[3] = 1
    z, hit = surface(new, data, 0., 0., groups=groups)
    assert z == pytest.approx(-.020, abs=2e-9)
    assert new.geom_type[hit] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert data.time == 0.


@pytest.mark.parametrize("existing_reset", ["move_root_origin_reset", "move_root_patch_reset"])
def test_existing_origin_or_patch_reset_is_rejected(existing_reset):
    cfg, source = flat_fixture()
    cfg.events[existing_reset] = "frozen"
    with pytest.raises(ValueError, match="reset profile"):
        with_move_signed_continuous_hfield(cfg, ground_model=source, seed=6272)
