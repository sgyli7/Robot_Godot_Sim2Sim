"""Opt-in native terrain spawn patches, independent of frozen flat scenes.

The signed rough-height preset's center origin need not lie on its normalized
heightfield surface. Use upstream footprint sampling and reset semantics.
Full-foot collision and raw-ray admission remain mandatory: native sampling
can fall back to a center when no valid patch exists. Moving reference-state
overlays must not overwrite this reset; they need separate terrain admission.
"""
from __future__ import annotations

import copy
from math import isfinite

import torch
from mjlab.envs.mdp.events import reset_root_state_from_flat_patches
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg
from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg

REVISION = "goose_move_native_spawn_patches_v1"
PATCH_NAME = "move_spawn"


def with_move_spawn_patches(cfg, *, patch_radius_m=.2, border_width_m=.5,
                            max_height_diff_m=.001, num_patches=16):
    """Copy the named terrain profile and enable native footprint sampling.

    This explicitly changes rough terrain's border and difficulty scaling,
    never a frozen flat model, contact material or driver. Other terrain types
    keep their existing native center origins. A wider rough border supplies
    feasible initial support; qualification must still traverse the rough
    interior. The reset event is returned separately by make_move_patch_reset
    so it cannot silently conflict with an existing reference-state boundary.
    """
    if (any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in
            (patch_radius_m, border_width_m, max_height_diff_m))
            or isinstance(num_patches, bool) or not isinstance(num_patches, int)
            or num_patches <= 0):
        raise ValueError("Move spawn patches require finite positive SI parameters")
    terrain = cfg.scene.terrain
    generator = None if terrain is None else terrain.terrain_generator
    rough = None if generator is None else generator.sub_terrains.get("random_rough")
    if not isinstance(rough, HfRandomUniformTerrainCfg) or rough.flat_patch_sampling:
        raise ValueError("Move spawn patches require an unmodified native rough profile")
    if (border_width_m < 2 * patch_radius_m + rough.horizontal_scale
            or border_width_m * 2 >= min(generator.size)):
        raise ValueError("Rough border must fit a complete footprint and retain an interior")
    result = copy.deepcopy(cfg)
    rough = result.scene.terrain.terrain_generator.sub_terrains["random_rough"]
    rough.border_width = border_width_m
    rough.scale_with_difficulty = True
    rough.flat_patch_sampling = {PATCH_NAME: FlatPatchSamplingCfg(
        num_patches=num_patches, patch_radius=patch_radius_m,
        max_height_diff=max_height_diff_m)}
    return result


def reset_move_root_on_patches(env, env_ids, *, patch_name, asset_cfg):
    """Guard the native episode-reset API against silent uniform fallback."""
    terrain = env.scene.terrain
    patches = None if terrain is None else terrain.flat_patches.get(patch_name)
    if (patches is None or patches.ndim != 4 or patches.shape[-1] != 3
            or patches.shape[2] == 0 or not torch.isfinite(patches).all()):
        raise ValueError("Move reset requires admitted finite native spawn patches")
    reset_root_state_from_flat_patches(env, env_ids, patch_name=patch_name,
        asset_cfg=asset_cfg)


def make_move_patch_reset():
    """Explicit native reset event; not installed into frozen flat profiles."""
    return EventTermCfg(func=reset_move_root_on_patches, mode="reset",
        params={"patch_name": PATCH_NAME, "asset_cfg": SceneEntityCfg("robot")})
