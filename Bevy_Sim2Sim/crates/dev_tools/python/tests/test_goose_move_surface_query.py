import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.move_surface_query import TerrainSurfaceQuery


def test_infinite_collision_plane_is_queryable_beyond_visual_extent():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="2 2 .01"/>
      <body pos="3 0 .099"><freejoint/><geom type="sphere" size=".1" group="1"/>
      </body></worldbody></mujoco>''')
    physical = mujoco.MjData(model)
    mujoco.mj_forward(model, physical)
    assert physical.ncon and physical.contact[0].dist < 0.
    hit = np.full(1, -1, dtype=np.int32)
    native_groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    native_groups[0] = 1
    assert mujoco.mj_ray(model, physical, np.array([3., 0., 1.]),
                        np.array([0., 0., -1.]), native_groups, 1, -1, hit) < 0.
    groups = model.geom_group.copy()
    qpos = physical.qpos.copy()
    query = TerrainSurfaceQuery(model, [0])
    query.update(qpos)
    assert query.ray([3., 0., 1.], [0., 0., -1.]) == (1., 0)
    np.testing.assert_array_equal(model.geom_group, groups)
    np.testing.assert_array_equal(physical.qpos, qpos)
    assert physical.time == query.data.time == 0.


def test_actual_tilted_translated_plane_pose_and_native_nearest_box():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size=".1 .1 .01" pos="0 0 .2" euler="0 5 0"/>
      <geom type="box" pos="0 0 .4" size=".2 .2 .05"/>
      <geom type="box" pos="0 0 .8" size=".1 .1 .05"/>
    </worldbody></mujoco>''')
    query = TerrainSurfaceQuery(model, [0, 1])
    query.update(np.empty(0))
    distance, hit = query.ray([0., 0., 1.], [0., 0., -1.])
    assert hit == 1 and distance == pytest.approx(.55)
    distance, hit = query.ray([3., 0., 1.], [0., 0., -1.])
    assert hit == 0
    assert 1. - distance == pytest.approx(.2 - 3. * np.tan(np.deg2rad(5.)))
    assert query.ray([0., 0., -.1], [0., 0., -1.]) == (-1., -1)


def test_query_rejects_ambiguous_roles_or_invalid_states():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="1 1 .01"/>
    </worldbody></mujoco>''')
    with pytest.raises(ValueError):
        TerrainSurfaceQuery(model, [0, 0])
    query = TerrainSurfaceQuery(model, [0])
    with pytest.raises(ValueError):
        query.update([1.])
    query.update([])
    with pytest.raises(ValueError):
        query.ray([0., 0., 1.], [0., 0., -2.])
