"""Separate primitive-terrain candidate; frozen heightfield scenes stay intact.

Random grid and scattered steps use installed native mjlab generators. Ramps
are three native boxes with a continuous, exactly declared top plane. No robot
shape, actuator, sensor history, physics option or policy contract is changed.
Scene construction never grants CPU/GPU or locomotion qualification.
"""
from dataclasses import dataclass
from math import cos, radians, sin, tan

import mujoco
import numpy as np
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import config as terrain_presets
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from .move_scenarios import make_move_terrain, with_move_terrain

REVISION = "goose_move_primitive_scenes_v1"
CONTINUOUS_REVISION = "goose_move_continuous_primitive_scenes_v2"


@dataclass(kw_only=True)
class MoveRampTerrainCfg(SubTerrainCfg):
    """Flat entry, exact signed ramp and flat exit, without a hidden plane."""

    inverted: bool = False
    slope_degrees: float = 5.
    entry_fraction: float = .25

    def function(self, difficulty, spec, rng):
        del rng
        if (not 0. <= difficulty <= 1. or self.slope_degrees != 5.
                or self.entry_fraction != .25 or min(self.size) <= 0.):
            raise ValueError("Move ramps require the frozen5deg geometry and native difficulty")
        length, width = self.size
        pad = length * self.entry_fraction
        run = length - 2 * pad
        beta = radians(self.slope_degrees * difficulty) * (-1 if self.inverted else 1)
        rise = abs(run * tan(beta))
        entry, exit_height = (rise, 0.) if self.inverted else (0., rise)
        body = spec.body("terrain")
        base = .1
        pieces = []
        for center_x, top in ((pad / 2, entry), (length - pad / 2, exit_height)):
            pieces.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(pad / 2, width / 2, (top + base) / 2),
                pos=(center_x, width / 2, (top - base) / 2)))
        # Place the top face, rather than the geom center, at the declared
        # ramp endpoints. Its projection spans exactly [pad,length-pad].
        thickness = rise + base
        pieces.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(run / (2 * cos(beta)), width / 2, thickness / 2),
            pos=(length / 2 + sin(beta) * thickness / 2, width / 2,
                 (entry + exit_height) / 2 - cos(beta) * thickness / 2),
            quat=(cos(beta / 2), 0., -sin(beta / 2), 0.)))
        return TerrainOutput(origin=np.array([pad / 2, width / 2, entry]),
            geometries=[TerrainGeometry(geom=geom, color=(.4, .5, .6, 1.)) for geom in pieces])


def make_move_primitive_terrain(*, ground_model, seed, height_m, evaluation=False,
                              ground_name="ground"):
    """New named scene, retaining the frozen ground materials and native course.

    Random grid heights range within±height/2, so neighbouring cells differ
    by at most the named tier. Native merging is disabled: actual children
    remain visible and counted. Formal ramps fix difficulty at one and are
    independently checked with rays against compiled geometry.
    """
    terrain = make_move_terrain(ground_model=ground_model, seed=seed,
        height_m=height_m, evaluation=evaluation, ground_name=ground_name)
    generator = terrain.terrain_generator
    generator.sub_terrains["random_rough"] = terrain_presets.box_random_grid(
        proportion=.2, grid_width=.4,
        grid_height_range=(height_m / 2, height_m / 2) if evaluation else (0., height_m / 2),
        platform_width=1., border_width=.5, holes=False, merge_similar_heights=False)
    generator.sub_terrains["slope_positive"] = MoveRampTerrainCfg(proportion=.15)
    generator.sub_terrains["slope_negative"] = MoveRampTerrainCfg(proportion=.15, inverted=True)
    return terrain


def with_move_primitive_terrain(cfg, *, ground_model, seed, height_m,
                              evaluation=False, ground_name="ground"):
    """Reuse plane removal and contact mapping; use native terrain origins.

    The inherited reset owns root position/orientation and velocity only at
    an episode boundary. It uses the primitive generator's declared entry
    origin, and does not alter joint, action, drive or sensor histories.
    Existing patch/origin resets are rejected to avoid double placement.
    """
    if any(name in cfg.events for name in ("move_root_patch_reset", "move_root_origin_reset")):
        raise ValueError("Primitive terrain requires an unmodified flat reset profile")
    result = with_move_terrain(cfg, ground_model=ground_model, seed=seed,
        height_m=height_m, evaluation=evaluation, ground_name=ground_name)
    result.scene.terrain = make_move_primitive_terrain(ground_model=ground_model,
        seed=seed, height_m=height_m, evaluation=evaluation, ground_name=ground_name)
    result.events["move_root_origin_reset"] = EventTermCfg(func=reset_root_state_uniform,
        mode="reset", params={"pose_range": {}, "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot")})
    return result


def make_move_continuous_primitive_terrain(*, ground_model, seed, height_m,
                                         evaluation=False, ground_name="ground"):
    """Opt into an aligned platform; preserve the frozen v1 scene factory.

    The native generator removes grid cells by their centers. Its 1m
    platform leaves 0.1m gaps beside a 0.4m grid on the declared 4m tile.
    A three-cell platform fills precisely the removed cells. Native grid,
    random heights, source material and episode origin remain unchanged.
    This is a new scene candidate and inherits no v1 dynamic qualification.
    """
    terrain = make_move_primitive_terrain(ground_model=ground_model, seed=seed,
        height_m=height_m, evaluation=evaluation, ground_name=ground_name)
    if terrain.terrain_generator.size != (4., 4.):
        raise ValueError("Continuous Move grid requires the declared 4m tile")
    rough = terrain.terrain_generator.sub_terrains["random_rough"]
    rough.platform_width = 3 * rough.grid_width
    return terrain
