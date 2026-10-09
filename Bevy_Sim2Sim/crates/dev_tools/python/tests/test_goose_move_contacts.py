"""Actual compiled scenes and native signed contacts catch old flat assumptions."""
import mujoco
import pytest
import torch

from bevy_microduck_tools.goose.move_contacts import ContactRole, SceneContactRoles


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="step" type="box" pos="0 0 .1" size=".5 .5 .1"/>
      <body name="robot" pos="0 0 .249"><freejoint/>
        <geom name="robot/foot" type="sphere" size=".05" mass="1"/>
        <geom name="robot/body" type="sphere" pos="0 0 .2" size=".05" mass="1"/>
      </body>
      <body name="load" pos="2 0 .3"><freejoint/>
        <geom name="payload/handle" type="box" size=".01 .01 .01" mass=".1"/>
      </body>
      <geom name="obstacle" type="sphere" pos="3 0 .3" size=".1"/>
      <geom name="visual" contype="0" conaffinity="0" type="sphere" size=".01" pos="4 0 1"/>
    </worldbody></mujoco>''')


def roles(compiled):
    return SceneContactRoles.from_model(compiled, {
        "step": ContactRole.TERRAIN,
        "robot/foot": ContactRole.ROBOT,
        "robot/body": ContactRole.ROBOT,
        "payload/handle": ContactRole.PAYLOAD,
        "obstacle": ContactRole.OBSTACLE,
    })


def test_elevated_step_contact_uses_actual_surface_not_world_zero():
    compiled = model()
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    assert data.ncon > 0
    assert data.geom_xpos[compiled.geom("robot/foot").id, 2] > .05
    contacts = data.contact[:data.ncon]
    value = roles(compiled).depth_by_world(
        torch.tensor(contacts.geom.copy()), torch.tensor(contacts.dist.copy()),
        torch.zeros(data.ncon, dtype=torch.long),
        active=torch.ones(data.ncon, dtype=torch.bool), num_worlds=1)
    assert value["terrain"][0].item() == pytest.approx(.001, abs=1e-9)
    assert value["self"][0].item() == 0.


def test_nonzero_payload_and_obstacle_ids_are_not_self_contact():
    compiled = model()
    foot, body, load, obstacle = (compiled.geom(name).id for name in
        ("robot/foot", "robot/body", "payload/handle", "obstacle"))
    geoms = torch.tensor([[foot, body], [foot, load], [obstacle, body], [-1, -1]])
    values = roles(compiled).depth_by_world(geoms,
        torch.tensor([-.0001, -.002, -.003, float('nan')]),
        torch.tensor([0, 0, 1, -1]), active=torch.tensor([True, True, True, False]), num_worlds=2)
    assert values["self"].tolist() == pytest.approx([.0001, 0.])
    assert values["payload"].tolist() == pytest.approx([.002, 0.])
    assert values["obstacle"].tolist() == pytest.approx([0., .003])
    assert values["terrain"].tolist() == [0., 0.]


def test_missing_colliding_geometry_role_and_bad_active_ids_fail():
    compiled = model()
    with pytest.raises(ValueError, match="no declared role"):
        SceneContactRoles.from_model(compiled, {"step": ContactRole.TERRAIN})
    with pytest.raises(ValueError, match="outside the scene"):
        roles(compiled).depth_by_world(torch.tensor([[-1, 0]]), torch.tensor([0.]),
            torch.tensor([0]), active=torch.tensor([True]), num_worlds=1)
