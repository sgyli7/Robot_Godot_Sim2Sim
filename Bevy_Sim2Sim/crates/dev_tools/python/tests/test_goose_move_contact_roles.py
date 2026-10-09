"""Actual native contact surfaces and strict declared-role failures."""
import mujoco
import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.move_contact_roles import (
    MoveContactRole, build_move_contact_roles, measure_move_contact_depths,
)


def native_model(*, height, undeclared=False, explicit_pair=False):
    pair = '<contact><pair geom1="foot" geom2="robot_shell"/>'
    if explicit_pair:
        pair += '<pair geom1="foot" geom2="ground"/>'
    pair += '</contact>'
    mask = 'contype="0" conaffinity="0"' if explicit_pair else ''
    extra = '<body name="unknown" pos="2 0 1"><geom name="unknown_geom" size=".1"/></body>' if undeclared else ''
    return mujoco.MjModel.from_xml_string(f'''<mujoco>{pair}<worldbody>
      <geom name="ground" type="plane" pos="0 0 {height}" size="3 3 .01" {mask}/>
      <body name="robot" pos="0 0 {height + .08}"><freejoint/>
        <geom name="foot" size=".1" {mask}/>
        <body name="robot_child" pos=".17 0 0"><geom name="robot_shell" size=".1"/></body>
      </body>
      <body name="payload" pos="0 .17 {height + .08}"><freejoint/><geom name="payload_geom" size=".1"/></body>
      <body name="obstacle" pos="-.17 0 {height + .08}"><geom name="obstacle_geom" size=".1"/></body>
      {extra}</worldbody></mujoco>''')


def roles_for(model):
    return build_move_contact_roles(model, robot_root_body="robot",
        terrain_root_bodies=(), terrain_geom_names=("ground",),
        payload_root_bodies=("payload",), obstacle_root_bodies=("obstacle",),
        expected_robot_leaves=2)


@pytest.mark.parametrize("height", [0., 1.])
def test_native_contacts_distinguish_self_ground_payload_obstacle_above_world_zero(height):
    model = native_model(height=height)
    roles = roles_for(model)
    assert roles.count(MoveContactRole.ROBOT) == 2
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_collision(model, data)
    assert data.ncon > 0 and data.time == 0.
    distance = torch.tensor(np.array([c.dist for c in data.contact[:data.ncon]]))
    pairs = torch.tensor(np.array([[c.geom1, c.geom2] for c in data.contact[:data.ncon]]), dtype=torch.int64)
    result = measure_move_contact_depths(distance_m=distance, geom_ids=pairs,
        world_ids=torch.zeros(data.ncon, dtype=torch.int64),
        role_lookup=torch.tensor(roles), contact_count=data.ncon, num_worlds=1)
    assert float(result['terrain_m'][0]) == pytest.approx(.02, abs=1e-12)
    assert float(result['self_m'][0]) == pytest.approx(.03, abs=1e-12)
    assert float(result['payload_contact_m'][0]) == pytest.approx(.03, abs=1e-12)
    assert float(result['obstacle_contact_m'][0]) == pytest.approx(.03, abs=1e-12)
    assert float(result['payload_terrain_m'][0]) == pytest.approx(.02, abs=1e-12)
    assert data.time == 0.


def test_actual_contact_depths_stay_separate_between_worlds():
    model = native_model(height=1.)
    distances, pairs, world_ids = [], [], []
    for world, clearance in enumerate((0., .005)):
        data = mujoco.MjData(model)
        data.qpos[2] += clearance
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_collision(model, data)
        distances.extend(c.dist for c in data.contact[:data.ncon])
        pairs.extend([c.geom1, c.geom2] for c in data.contact[:data.ncon])
        world_ids.extend([world] * data.ncon)
        assert data.time == 0.
    result = measure_move_contact_depths(distance_m=torch.tensor(distances, dtype=torch.float64),
        geom_ids=torch.tensor(pairs), world_ids=torch.tensor(world_ids),
        role_lookup=torch.tensor(roles_for(model)), contact_count=len(distances), num_worlds=2)
    assert result['terrain_m'].tolist() == pytest.approx([.02, .015], abs=1e-12)
    assert result['self_m'].tolist() == pytest.approx([.03, .03], abs=1e-12)


def test_undeclared_or_ambiguous_native_colliders_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="one declared role"):
        roles_for(native_model(height=0., undeclared=True))
    model = native_model(height=0.)
    with pytest.raises(ValueError, match="one declared role"):
        build_move_contact_roles(model, robot_root_body="robot",
            terrain_root_bodies=("robot",), terrain_geom_names=("ground",),
            payload_root_bodies=("payload",), obstacle_root_bodies=("obstacle",),
            expected_robot_leaves=2)


def test_native_explicit_pairs_keep_zero_mask_colliders_in_role_inventory():
    model = native_model(height=0., explicit_pair=True)
    roles = roles_for(model)
    assert roles[model.geom("foot").id] == MoveContactRole.ROBOT
    assert roles[model.geom("ground").id] == MoveContactRole.TERRAIN


def test_occupied_invalid_contact_fails_while_unused_capacity_is_ignored():
    distance = torch.tensor([-.02, float('nan')])
    pairs = torch.tensor([[0, 1], [-1, -1]])
    world = torch.tensor([0, -1])
    kwargs = dict(distance_m=distance, geom_ids=pairs, world_ids=world,
        role_lookup=torch.tensor([MoveContactRole.ROBOT, MoveContactRole.TERRAIN]),
        num_worlds=1)
    result = measure_move_contact_depths(contact_count=1, **kwargs)
    assert float(result['terrain_m'][0]) == pytest.approx(.02)
    assert float(result['self_m'][0]) == 0.
    with pytest.raises(ValueError, match="Invalid occupied"):
        measure_move_contact_depths(contact_count=2, **kwargs)
    with pytest.raises(ValueError, match="count, shape"):
        measure_move_contact_depths(contact_count=3, **kwargs)
