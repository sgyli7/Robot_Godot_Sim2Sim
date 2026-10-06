"""Physical geometry outcomes of the planning seam, using native Mink QP."""
import itertools

import pytest
import numpy as np

mink = pytest.importorskip('mink')
mujoco = pytest.importorskip('mujoco')
pytest.importorskip('scipy')
from scipy.spatial.transform import Rotation

from bevy_microduck_tools.goose.kinematic_contact import (
    DisplacementCollisionLimit, GoosePlanningCollisionLimit, box_support_height)

ROOT_BOX = '''<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .01"/>
<body name="box" pos="0 0 .101"><freejoint/>
<geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
</body></worldbody></mujoco>'''
CHILD_BOX = '''<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .01"/>
<body name="root" pos="0 0 .101"><freejoint/>
<inertial mass=".1" pos="0 0 0" diaginertia=".01 .01 .01"/>
<body name="box"><joint name="yaw" axis="0 0 1"/>
<geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
</body></body></worldbody></mujoco>'''


def private_displacement(xml, limit_type, dt):
    model = mujoco.MjModel.from_xml_string(xml)
    cfg = mink.Configuration(model)
    task = mink.FrameTask('box','body',10.,0.)
    task.set_target(mink.SE3.from_translation(np.array([0.,0.,.09])))
    limit = limit_type(model,[(['box_geom'],['floor'])],gain=.85,
        minimum_distance_from_collisions=0.,collision_detection_distance=.05)
    freeze = mink.DofFreezingTask(model,list(range(3,model.nv)))
    velocity = mink.solve_ik(cfg,[task],dt,'daqp',limits=[limit],constraints=[freeze])
    cfg.integrate_inplace(velocity,dt)
    return model,cfg,limit


@pytest.mark.parametrize('dt',[.01,.02,.04])
def test_displacement_constraint_keeps_box_above_plane(dt):
    # Different dt values are private IK unit tests, not physics substeps.
    _, native, _ = private_displacement(CHILD_BOX,mink.CollisionAvoidanceLimit,dt)
    _, corrected, _ = private_displacement(CHILD_BOX,DisplacementCollisionLimit,dt)
    assert native.q[2]-.1 < -.009
    assert corrected.q[2]-.1 >= .00015-1e-10
    np.testing.assert_allclose(corrected.q[3:7],[1.,0.,0.,0.],atol=1e-12,rtol=0.)
    assert corrected.data.time == 0.


def test_requested_world_root_collision_matches_native_physics_pair():
    model,native,unadapted = private_displacement(ROOT_BOX,mink.CollisionAvoidanceLimit,.02)
    assert unadapted.geom_id_pairs == []
    assert native.q[2]-.1 < -.009
    # Native MuJoCo's world contact is a parent-filter exception.
    mujoco.mj_collision(model,native.data)
    assert native.data.ncon > 0
    _,corrected,limit = private_displacement(ROOT_BOX,GoosePlanningCollisionLimit,.02)
    assert limit.geom_id_pairs == [(0,1)]
    assert corrected.q[2]-.1 >= .00015-1e-10


def test_pair_adaptation_preserves_masks_and_robot_parent_filter():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="floor" type="plane" size="1 1 .01"/>
    <body name="root" pos="0 0 .3"><freejoint/>
    <geom name="root_geom" type="sphere" size=".1"/>
    <body name="child" pos="0 0 -.03"><joint/>
    <geom name="child_geom" type="sphere" size=".1"/></body></body>
    </worldbody></mujoco>''')
    pairs=[(['floor','root_geom','child_geom'],['floor','root_geom','child_geom'])]
    before=(model.geom_contype.copy(),model.geom_conaffinity.copy(),model.geom_size.copy())
    limit=GoosePlanningCollisionLimit(model,pairs)
    assert (1,2) not in limit.geom_id_pairs
    assert (0,1) in limit.geom_id_pairs
    assert (0,2) in limit.geom_id_pairs
    for actual,original in zip((model.geom_contype,model.geom_conaffinity,model.geom_size),before):
        assert np.array_equal(actual,original)
    model.geom_contype[1]=model.geom_conaffinity[1]=0
    assert (0,1) not in GoosePlanningCollisionLimit(model,pairs).geom_id_pairs


def test_world_pair_respects_explicit_model_exclusion():
    xml=ROOT_BOX.replace('</mujoco>',
        '<contact><exclude body1="world" body2="box"/></contact></mujoco>')
    model=mujoco.MjModel.from_xml_string(xml)
    assert model.nexclude == 1
    limit=GoosePlanningCollisionLimit(model,[(['box_geom'],['floor'])])
    assert limit.geom_id_pairs == []


@pytest.mark.parametrize('dt',[0.,-1.,float('nan'),float('inf')])
def test_invalid_planning_time_is_rejected(dt):
    model=mujoco.MjModel.from_xml_string(ROOT_BOX)
    cfg=mink.Configuration(model)
    limit=GoosePlanningCollisionLimit(model,[(['box_geom'],['floor'])])
    with pytest.raises(ValueError,match='Positive planning dt'):
        limit.compute_qp_inequalities(cfg,dt)


@pytest.mark.parametrize('angle',[0.,45.,83.,90.])
def test_rotating_sole_goal_clears_all_box_corners(angle):
    size=np.array([.079,.03,.0154933878])
    rotation=Rotation.from_euler('y',angle,degrees=True).as_matrix()
    height=box_support_height(size,rotation)
    corners=np.array(list(itertools.product((-1.,1.),repeat=3)))*size
    world=corners@rotation.T+np.array([0.,0.,height])
    assert world[:,2].min() >= -1e-12
    assert abs(world[:,2].min()) < 1e-12


def test_invalid_box_goal_does_not_silently_project_inputs():
    with pytest.raises(ValueError):
        box_support_height([.1,-.1,.1],np.eye(3))
    with pytest.raises(ValueError):
        box_support_height([.1,.1,.1],np.diag([1.,1.,-1.]))
