"""Check native sampled surfaces and reject missing reset data before GPU."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.terrains.terrain_entity import TerrainEntity

from bevy_microduck_tools.goose.move_scenarios import make_move_terrain
from bevy_microduck_tools.goose.move_spawn_patches import (
    PATCH_NAME, make_move_patch_reset, with_move_spawn_patches,
)


def terrain_cfg(height):
    ground = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="2 2 .01" group="0"
      contype="1" conaffinity="2"/></worldbody></mujoco>''')
    return SimpleNamespace(scene=SimpleNamespace(terrain=make_move_terrain(
        ground_model=ground, seed=6157, height_m=height, evaluation=True)),
        events={}, actor_dim=65, policy_dt=.02, physics_dt=.005)


@pytest.mark.parametrize("height", [.010, .020, .050])
def test_native_patches_lie_on_actual_rough_surface_and_cover_sole_corners(height):
    original = terrain_cfg(height)
    cfg = with_move_spawn_patches(original, num_patches=4)
    old_rough = original.scene.terrain.terrain_generator.sub_terrains["random_rough"]
    assert old_rough.flat_patch_sampling is None and not old_rough.scale_with_difficulty
    assert old_rough.border_width == .25
    assert cfg.actor_dim == 65 and cfg.events == {} and cfg.physics_dt == .005
    terrain = TerrainEntity(cfg.scene.terrain, device="cpu")
    model = terrain.spec.compile();data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    assert terrain.flat_patches[PATCH_NAME].shape == (1, 5, 4, 3)
    group = np.zeros(mujoco.mjNGROUP, dtype=np.uint8);group[0] = 1
    for patch in terrain.flat_patches[PATCH_NAME][0, 1].numpy():
        for dx in (-.05, .11):
            for dy in (-.106, .106):
                point = patch + np.array([dx, dy, .2])
                hit = np.full(1, -1, dtype=np.int32)
                distance = mujoco.mj_ray(model, data, point,
                    np.array([0., 0., -1.]), group, 1, -1, hit)
                assert hit[0] >= 0 and distance >= 0
                assert abs(point[2] - distance - patch[2]) <= .001 + 1e-6
    assert data.time == 0.


def test_missing_or_nonfinite_patches_cannot_fall_back_to_uniform_reset():
    event = make_move_patch_reset()
    assert event.mode == "reset"
    for patches in ({}, {PATCH_NAME: torch.full((1, 1, 1, 3), float("nan"))}):
        env = SimpleNamespace(scene=SimpleNamespace(
            terrain=SimpleNamespace(flat_patches=patches)))
        with pytest.raises(ValueError, match="admitted finite"):
            event.func(env, None, **event.params)


def test_invalid_border_and_existing_patch_contract_are_rejected():
    cfg = terrain_cfg(.020)
    with pytest.raises(ValueError, match="complete footprint"):
        with_move_spawn_patches(cfg, border_width_m=.25)
    with pytest.raises(ValueError, match="retain an interior"):
        with_move_spawn_patches(cfg, border_width_m=2.)
    with pytest.raises(ValueError, match="positive SI"):
        with_move_spawn_patches(cfg, num_patches=True)
    candidate = with_move_spawn_patches(cfg)
    with pytest.raises(ValueError, match="unmodified"):
        with_move_spawn_patches(candidate)
