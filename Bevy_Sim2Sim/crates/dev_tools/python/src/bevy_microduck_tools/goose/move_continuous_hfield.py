"""Opt-in signed continuous terrain; existing Goose scenes remain unchanged.

The named profile fixes a 4m tile and a grid that includes both endpoints.
Native MuJoCo heightfields own the collision surface. Static construction
and ray tests do not grant CPU/GPU rollout or locomotion qualification.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, radians, sin, tan
from pathlib import Path

import mujoco
import numpy as np
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainGeometry, TerrainOutput,
)

from .move_scenarios import make_move_terrain, with_move_terrain

REVISION = "signed_continuous_hfield_v1"
CAPACITY_REVISION = "signed_continuous_hfield_capacity_v2"
TILE_SIZE_M = (4., 4.)
GRID_POINTS = 257
CAPACITY_GRID_POINTS = 65
REQUIRED_HEIGHT_M = .020
BASE_THICKNESS_M = .100
PROFILES = ("bump20", "dip20", "variable_slope")


def _raised_cosine(coordinates, start, end):
    unit = np.clip((coordinates - start) / (end - start), 0., 1.)
    return (1. - np.cos(2 * np.pi * unit)) / 2


@dataclass(kw_only=True)
class SignedContinuousHfieldCfg(SubTerrainCfg):
    """Compact continuous surface with one-meter flat strips on all sides."""

    size: tuple[float, float] = TILE_SIZE_M
    profile: str = "bump20"
    grid_points: int = GRID_POINTS

    @property
    def revision(self):
        """Different collision discretizations never inherit qualification."""
        return REVISION if self.grid_points == GRID_POINTS else CAPACITY_REVISION

    def height_samples(self, difficulty=1.):
        """Physical meters, rows along Y and columns along X, including edges."""
        if (self.size != TILE_SIZE_M or isinstance(difficulty, bool)
                or not isfinite(difficulty) or not 0. <= difficulty <= 1.):
            raise ValueError("Signed hfield requires the named4m tile and finite native difficulty")
        if self.profile not in PROFILES:
            raise ValueError("Unknown signed hfield profile")
        if (type(self.grid_points) is not int
                or self.grid_points not in (GRID_POINTS, CAPACITY_GRID_POINTS)):
            raise ValueError("Signed hfield requires a named257 or65 point grid")
        coordinates = np.linspace(0., 4., self.grid_points)
        window = _raised_cosine(coordinates, 1., 3.)
        if self.profile == "variable_slope":
            # Calibrate the actual adjacent grid heights, not just the ideal
            # curve derivative, to +/-5deg along the Y=2m centreline. The
            # resulting lobe height exceeds20mm and is not called bump20.
            dx = 4. / (self.grid_points - 1)
            height = 2 * dx * tan(radians(5.)) / sin(2 * np.pi * dx / .75)
            along_x = (_raised_cosine(coordinates, 1., 1.75)
                       - _raised_cosine(coordinates, 2.25, 3.))
            return height * difficulty * np.outer(window, along_x)
        sign = -1. if self.profile == "dip20" else 1.
        return sign * REQUIRED_HEIGHT_M * difficulty * np.outer(window, window)

    def geometry_manifest(self, difficulty=1.):
        """Source and sample identity for freezing a separately named scene.

        Save ``height_samples(difficulty)`` with this receipt. The ideal curve
        is C1 at its support edges; the real collider is native piecewise
        triangular interpolation. Positive/negative20mm are full-difficulty
        bump/dip requirements, not a claim about easier curriculum levels.
        """
        samples = self.height_samples(difficulty)
        low, high = float(samples.min()), float(samples.max())
        elevation = high - low
        normalized = ((samples - low) / elevation if elevation > 0
                      else np.zeros_like(samples))
        dx = 4. / (self.grid_points - 1)
        centre_slopes = np.degrees(np.arctan(
            np.diff(samples[(self.grid_points - 1) // 2]) / dx))
        return {
            "revision": self.revision, "profile": self.profile,
            "geometry_type": "native_mujoco_hfield", "difficulty": float(difficulty),
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "height_samples_sha256": sha256(samples.astype("<f8").tobytes()).hexdigest(),
            "normalized_samples_sha256": sha256(normalized.astype("<f4").tobytes()).hexdigest(),
            "height_sample_encoding": "float64_little_endian_row_major_meters",
            "native_sample_encoding": "float32_little_endian_row_major_normalized",
            "sample_axes": {"row": "Y", "column": "X"},
            "grid_shape": [self.grid_points, self.grid_points], "grid_spacing_m": dx,
            "domain_xy_m": [[0., 4.], [0., 4.]], "height_range_m": [low, high],
            "entry_x_m": [0., 1.], "exit_x_m": [3., 4.],
            "flat_side_y_m": [[0., 1.], [3., 4.]],
            "origin_local_m": [.5, 2., 0.],
            "native_size_m": [2., 2., max(elevation, .001), BASE_THICKNESS_M],
            "native_geom_pos_local_m": [2., 2., low],
            "centreline_slope_range_degrees": [float(centre_slopes.min()), float(centre_slopes.max())],
            "height_interpolation": "native_piecewise_triangles",
            "qualification": False,
        }

    def function(self, difficulty, spec, rng):
        """Add one true native hfield, never a plane or a stack of boxes."""
        del rng  # v1 geometry is canonical; native generator still owns origins.
        heights = self.height_samples(difficulty)
        low, high = float(heights.min()), float(heights.max())
        elevation = high - low
        # A zero difficulty is exactly flat data. MuJoCo only requires the
        # asset's size to be positive; it does not require nonzero samples.
        normalized = ((heights - low) / elevation if elevation > 0
                      else np.zeros_like(heights))
        body = spec.body("terrain")
        if body is None:
            raise ValueError("Signed hfield requires the native terrain body")
        name = f"{self.revision}_{self.profile}_{len(spec.hfields)}"
        field = spec.add_hfield(name=name,
            size=(2., 2., max(elevation, .001), BASE_THICKNESS_M),
            nrow=self.grid_points, ncol=self.grid_points,
            userdata=normalized.astype(np.float32).ravel().tolist())
        geom = body.add_geom(name=name + "_surface",
            type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
            pos=(2., 2., low))
        return TerrainOutput(origin=np.array([.5, 2., 0.]),
            geometries=[TerrainGeometry(geom=geom, hfield=field)])


def make_move_signed_continuous_hfield(*, ground_model, seed,
                                     evaluation=False, ground_name="ground",
                                     grid_points=GRID_POINTS):
    """Separate native scene: three canonical4m surfaces, not random boxes.

    The inherited materials, collision masks and border stay native. Training
    levels span0..1; evaluation fixes full20mm bump/dip and centreline+/-5deg.
    Each origin is on the1m flat entry, not in the depression or on the bump.
    No current scene, curriculum, policy input or physics option is rewired.
    """
    terrain = make_move_terrain(ground_model=ground_model, seed=seed,
        height_m=REQUIRED_HEIGHT_M, evaluation=evaluation, ground_name=ground_name)
    generator = terrain.terrain_generator
    generator.sub_terrains = {profile: SignedContinuousHfieldCfg(
        profile=profile, grid_points=grid_points,
        proportion=1. / len(PROFILES)) for profile in PROFILES}
    generator.num_cols = len(PROFILES)
    return terrain


def with_move_signed_continuous_hfield(cfg, *, ground_model, seed,
                                     evaluation=False, ground_name="ground",
                                     grid_points=GRID_POINTS):
    """Opt-in copied scene with native origin reset and no filling plane.

    Reuse the existing native callback, plane removal, material-pair guards
    and contact-role mapping. The new reset only places the root at an episode
    boundary. Dynamic surface measurements, capacity, CPU/GPU contact parity
    and locomotion admission are separate requirements, not granted here.
    """
    if any(name in cfg.events for name in ("move_root_patch_reset", "move_root_origin_reset")):
        raise ValueError("Signed hfield requires an unmodified flat reset profile")
    result = with_move_terrain(cfg, ground_model=ground_model, seed=seed,
        height_m=REQUIRED_HEIGHT_M, evaluation=evaluation, ground_name=ground_name)
    result.scene.terrain = make_move_signed_continuous_hfield(
        ground_model=ground_model, seed=seed,
        evaluation=evaluation, ground_name=ground_name, grid_points=grid_points)
    result.events["move_root_origin_reset"] = EventTermCfg(func=reset_root_state_uniform,
        mode="reset", params={"pose_range": {}, "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot")})
    return result
