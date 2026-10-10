from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity.mdp.rewards import feet_swing_height

from bevy_microduck_tools.goose.move_sole_height import (
    WholeSoleHeightSensor, WholeSoleHeightSensorCfg, with_whole_sole_native_height)


def sensor_for_heights(heights):
    heights = torch.as_tensor(heights, dtype=torch.float32).reshape(-1, 16)
    sensor = WholeSoleHeightSensor(WholeSoleHeightSensorCfg(name="height", frame=(),
        pattern=RingPatternCfg(rings=(), include_center=True)))
    sensor._ctx = object()
    sensor._num_frames, sensor._num_rays_per_frame, sensor._num_rays = 16, 1, 16
    sensor._distances = heights.clone()
    sensor._normals_w = torch.zeros((*heights.shape, 3))
    sensor._normals_w[..., 2] = 1.
    sensor._frame_pos_w = torch.zeros((*heights.shape, 3))
    sensor._frame_pos_w[..., 2] = heights
    sensor._frame_quat_w = torch.zeros((*heights.shape, 4))
    sensor._frame_quat_w[..., 0] = 1.
    sensor._hit_pos_w = torch.zeros_like(sensor._frame_pos_w)
    sensor._pos_w = sensor._frame_pos_w[:, 0]
    sensor._quat_w = sensor._frame_quat_w[:, 0]
    return sensor


def test_tilted_foot_reports_lowest_received_vertex_in_original_right_left_order():
    sensor = sensor_for_heights([[.035, .045, .006, .018, .055, .04, .02, .03,
                                 .022, .028, .034, .031, .05, .06, .07, .08]])
    assert sensor.num_frames == 2 and sensor.num_rays == 16
    assert sensor.num_rays_per_frame == 8
    torch.testing.assert_close(sensor.data.heights, torch.tensor([[.006, .022]]))
    assert sensor.data.frame_pos_w.shape == (1, 2, 3)
    assert sensor.data.frame_quat_w.shape == (1, 2, 4)


def test_native_inside_terrain_and_miss_rules_survive_grouping_and_cache_reset():
    sensor = sensor_for_heights([[.03] * 16, [.04] * 16])
    sensor._normals_w[0, 2, 2] = -1.
    sensor._distances[0, 8:] = -1.
    sensor._frame_pos_w[0, 8:, 2] = .015
    torch.testing.assert_close(sensor.data.heights[0], torch.tensor([0., .015]))
    sensor._frame_pos_w[1, :, 2] = .007
    sensor._distances[1, :] = .007
    # Cached data changes only on the native update/reset boundary.
    torch.testing.assert_close(sensor.data.heights[1], torch.tensor([.04, .04]))
    sensor.reset(torch.tensor([1]))
    torch.testing.assert_close(sensor.data.heights[1], torch.tensor([.007, .007]))


def test_upstream_swing_reward_keeps_two_feet_and_scores_actual_minimum_at_landing():
    height = sensor_for_heights([[.006] + [.04] * 7 + [.022] + [.05] * 7])
    landing = torch.zeros((1, 2), dtype=torch.bool)
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.zeros((1, 2))),
        compute_first_contact=lambda dt: landing)
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02,
        scene={"height": height, "contact": contact}, extras={"log": {}},
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[.4, 0., 0.]])))
    cfg = RewardTermCfg(func=feet_swing_height, weight=-.25,
        params={"height_sensor_name": "height"})
    reward = feet_swing_height(cfg, env)
    args = dict(sensor_name="contact", height_sensor_name="height", target_height=.02,
        command_name="velocity", command_threshold=.01)
    assert reward.peak_heights.shape == (1, 2)
    torch.testing.assert_close(reward(env, **args), torch.zeros(1))
    landing[:] = True
    contact.data.found[:] = 1
    torch.testing.assert_close(reward(env, **args), torch.tensor([.5]))
    assert reward.peak_heights.count_nonzero() == 0


def make_cfg():
    def spec():
        return mujoco.MjSpec.from_string('''<mujoco><worldbody>
          <body name="right_foot"><geom name="right_flexible_sole" type="box"
            size=".04 .02 .002" pos=".01 0 0" quat=".9238795325 0 .3826834324 0"/></body>
          <body name="left_foot"><geom name="left_flexible_sole" type="box"
            size=".04 .02 .002" pos="-.01 0 0"/></body>
        </worldbody></mujoco>''')
    sensor = TerrainHeightSensorCfg(name="goose_feet_height",
        frame=tuple(ObjRef(type="site", name=f"{s}_sole_center", entity="robot") for s in ("right", "left")),
        pattern=RingPatternCfg.single_ring(radius=.01, num_samples=6),
        ray_alignment="world", include_geom_groups=(0,), reduction="min")
    return SimpleNamespace(scene=SimpleNamespace(sensors=(sensor,), entities={
        "robot": SimpleNamespace(spec_fn=spec)}), observations={"actor": [65]}, rewards=["native"])


def test_opt_in_adds_only_massless_frames_at_actual_rotated_collision_vertices():
    cfg = make_cfg()
    changed = with_whole_sole_native_height(cfg)
    old = cfg.scene.entities["robot"].spec_fn().compile()
    new = changed.scene.entities["robot"].spec_fn().compile()
    assert new.nsite == old.nsite + 16 and new.nbody == old.nbody and new.ngeom == old.ngeom
    for name in ("body_mass", "body_inertia", "geom_pos", "geom_quat", "geom_size"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))
    geom = new.geom("right_flexible_sole").id
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, new.geom_quat[geom])
    for index, signs in enumerate(((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                                   (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1))):
        expected = rotation.reshape(3, 3) @ (np.asarray(signs) * new.geom_size[geom]) + new.geom_pos[geom]
        np.testing.assert_allclose(new.site(f"right_sole_vertex_{index}").pos, expected)
    assert not isinstance(cfg.scene.sensors[0], WholeSoleHeightSensorCfg)
    assert changed.observations == cfg.observations and changed.rewards == cfg.rewards
    with pytest.raises(ValueError, match="twice"):
        with_whole_sole_native_height(changed)
    cfg.scene.sensors = ()
    with pytest.raises(ValueError, match="exactly one"):
        with_whole_sole_native_height(cfg)
