"""Opt-in native vertical rays at received sole vertices; no physical edits.

The native sensor owns raycasts, inside/miss handling and reset timing. A
read-only view lets the native height reduction see its physical frames;
the public sensor then groups those frames into the original two feet.
"""
from dataclasses import dataclass, replace
import copy
import numpy as np
import mujoco
from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from bevy_microduck_tools.goose.native_geometry315 import collision_geom_vertices

REVISION = 'goose_whole_sole_native_height221_v1'
FOOT_NAMES = ('right', 'left')
VERTICES_PER_FOOT = 8


class _PhysicalFrameView(TerrainHeightSensor):
    """Only used by the upstream pure data reduction; never sensed or reset."""
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    @property
    def num_frames(self):
        return self.source._num_frames

    @property
    def num_rays_per_frame(self):
        return self.source._num_rays_per_frame


class WholeSoleHeightSensor(TerrainHeightSensor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._physical_view = _PhysicalFrameView(self)

    @property
    def num_frames(self):
        return len(FOOT_NAMES)

    @property
    def num_rays_per_frame(self):
        return VERTICES_PER_FOOT * self._num_rays_per_frame

    def _compute_data(self):
        native = TerrainHeightSensor._compute_data(self._physical_view)
        if native.heights.shape[1] != len(FOOT_NAMES) * VERTICES_PER_FOOT:
            raise ValueError('Whole-sole sensor requires received eight vertices per foot')
        batch = native.heights.shape[0]
        grouped = native.heights.reshape(batch, len(FOOT_NAMES), VERTICES_PER_FOOT)
        positions = native.frame_pos_w.reshape(batch, len(FOOT_NAMES), VERTICES_PER_FOOT, 3).mean(dim=2)
        quaternions = native.frame_quat_w.reshape(batch, len(FOOT_NAMES), VERTICES_PER_FOOT, 4)[:, :, 0]
        return replace(native, heights=grouped.amin(dim=2),
            frame_pos_w=positions, frame_quat_w=quaternions)


@dataclass
class WholeSoleHeightSensorCfg(TerrainHeightSensorCfg):
    def build(self):
        return WholeSoleHeightSensor(self)


def with_whole_sole_native_height(cfg):
    result = copy.deepcopy(cfg)
    matches = [s for s in result.scene.sensors if s.name == 'goose_feet_height']
    if len(matches) != 1 or not isinstance(matches[0], TerrainHeightSensorCfg):
        raise ValueError('Requires exactly one existing native Goose foot-height sensor')
    old = matches[0]
    if isinstance(old, WholeSoleHeightSensorCfg):
        raise ValueError('Whole-sole height sensing cannot be attached twice')
    if old.ray_alignment != 'world' or old.include_geom_groups != (0,) or old.reduction != 'min':
        raise ValueError('Requires the frozen native vertical terrain-only minimum profile')
    entity = result.scene.entities['robot']
    original = entity.spec_fn
    frames = tuple(ObjRef(type='site', entity='robot', name=f'{side}_sole_vertex_{i}')
        for side in FOOT_NAMES for i in range(VERTICES_PER_FOOT))

    def sensed_spec():
        spec = original()
        model = spec.compile()
        for side in FOOT_NAMES:
            geom = model.geom(f'{side}_flexible_sole').id
            vertices = collision_geom_vertices(model, geom)
            if vertices.shape != (VERTICES_PER_FOOT, 3):
                raise ValueError('Received rigid sole must have exactly eight convex support vertices')
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            local = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
            body = spec.body(model.body(int(model.geom_bodyid[geom])).name)
            for index, point in enumerate(local):
                body.add_site(name=f'{side}_sole_vertex_{index}', pos=point,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.001, .001, .001),
                    rgba=(0., 1., 0., 0.), group=5)
        return spec

    entity.spec_fn = sensed_spec
    fields = dict(vars(old))
    fields.update(frame=frames,
        pattern=RingPatternCfg(rings=(), include_center=True), debug_vis=False)
    replacement = WholeSoleHeightSensorCfg(**fields)
    result.scene.sensors = tuple(replacement if sensor is old else sensor
        for sensor in result.scene.sensors)
    if result.observations['actor'] != cfg.observations['actor'] or result.rewards != cfg.rewards:
        raise ValueError('Whole-sole sensing must preserve Actor and native reward definitions')
    return result
