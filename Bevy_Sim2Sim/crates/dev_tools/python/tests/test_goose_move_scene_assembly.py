"""Compile native rough scenes and check real elevated contacts, no rollout."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from bevy_microduck_tools.goose.move_contacts import ContactRole, SceneContactRoles
from bevy_microduck_tools.goose.move_scenarios import with_move_terrain


def fixture():
    ground = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="2 2 .01" contype="1"
      conaffinity="2" friction=".65 .01 .002" solref=".005 1"/>
      </worldbody></mujoco>''')
    def robot_spec():
        return mujoco.MjSpec.from_string('''<mujoco><worldbody>
          <body name="torso" pos="0 0 .03"><freejoint name="root"/>
          <geom name="sole" type="box" size=".06 .04 .01" mass="1"
          contype="2" conaffinity="1" friction=".8 .01 .002"/>
          </body></worldbody></mujoco>''')
    def flat_scene(spec):
        spec.option.timestep = .005
        spec.option.iterations = 17
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(2.,2.,.01), contype=1, conaffinity=2,
            friction=(.65,.01,.002), solref=(.005,1.))
    sensor = ContactSensorCfg(name="ground_contact",
        primary=ContactMatch(mode="geom", entity="robot", pattern="sole"),
        secondary=ContactMatch(mode="geom", pattern="ground"),
        fields=("found","force"), reduce="netforce", num_slots=1,
        track_air_time=True, history_length=4)
    cfg = SimpleNamespace(scene=SceneCfg(num_envs=2,
        entities={"robot":EntityCfg(spec_fn=robot_spec)},
        sensors=(sensor,), spec_fn=flat_scene),
        policy_dt=.02, physics_dt=.005, actor_dim=65, action_dim=18,
        rewards={"native_recipe":"frozen"})
    return cfg, ground


@pytest.mark.parametrize("height", [.005,.020])
def test_native_scene_replaces_plane_and_preserves_robot_and_sensor(height):
    cfg, ground = fixture()
    old = Scene(cfg.scene, device="cpu").compile()
    candidate = with_move_terrain(cfg, ground_model=ground, seed=6160,
        height_m=height, evaluation=True)
    new = Scene(candidate.scene, device="cpu").compile()
    assert cfg.scene.terrain is None
    assert cfg.scene.sensors[0].secondary.mode == "geom"
    assert candidate.scene.sensors[0].secondary == ContactMatch(mode="body",pattern="terrain")
    for field in ("primary","fields","reduce","num_slots","track_air_time","history_length"):
        assert getattr(candidate.scene.sensors[0],field)==getattr(cfg.scene.sensors[0],field)
    assert candidate.rewards==cfg.rewards and candidate.policy_dt==cfg.policy_dt
    assert candidate.physics_dt==cfg.physics_dt and candidate.actor_dim==65 and candidate.action_dim==18
    assert mujoco.mj_name2id(new,mujoco.mjtObj.mjOBJ_GEOM,"ground")==-1
    assert new.nhfield==3 and new.ngeom>10
    assert new.opt.timestep==old.opt.timestep==.005
    assert new.opt.iterations==old.opt.iterations==17
    for field in ("mass","inertia","ipos","iquat"):
        np.testing.assert_array_equal(getattr(new,"body_"+field)[new.body("robot/torso").id],
            getattr(old,"body_"+field)[old.body("robot/torso").id])
    for field in ("type","pos","axis","range","limited"):
        np.testing.assert_array_equal(getattr(new,"jnt_"+field),getattr(old,"jnt_"+field))
    for field in ("type","size","pos","quat","contype","conaffinity","friction","solref","solimp"):
        np.testing.assert_array_equal(getattr(new,"geom_"+field)[new.geom("robot/sole").id],
            getattr(old,"geom_"+field)[old.geom("robot/sole").id])
    terrain = new.body("terrain").id
    ids=np.flatnonzero(new.geom_bodyid==terrain)
    assert len(ids)>10
    for field in ("contype","conaffinity","friction","solref","solimp"):
        actual=getattr(new,"geom_"+field)[ids]
        np.testing.assert_array_equal(actual,np.broadcast_to(getattr(ground,"geom_"+field)[0],actual.shape))


def test_elevated_native_box_contact_is_terrain_not_self():
    cfg, ground=fixture()
    scene=Scene(with_move_terrain(cfg,ground_model=ground,seed=6161,
        height_m=.020,evaluation=True).scene,device="cpu")
    model=scene.compile();data=mujoco.MjData(model)
    mujoco.mj_fwdPosition(model,data)
    terrain=model.body("terrain").id
    box=int(np.flatnonzero((model.geom_bodyid==terrain)
        &(model.geom_type==mujoco.mjtGeom.mjGEOM_BOX)
        &np.isclose(model.geom_size[:,2],.010))[0])
    top=data.geom_xpos[box,2]+model.geom_size[box,2]
    data.qpos[:3]=[*data.geom_xpos[box,:2],top+.010-.005]
    mujoco.mj_forward(model,data)  # Private collision query; no integration.
    mapping={model.geom(i).name:(ContactRole.ROBOT if model.geom(i).name.startswith("robot/") else ContactRole.TERRAIN)
        for i in range(model.ngeom)}
    roles=SceneContactRoles.from_model(model,mapping)
    geoms=torch.tensor(np.asarray(data.contact.geom[:data.ncon]).copy())
    dist=torch.tensor(np.asarray(data.contact.dist[:data.ncon]).copy())
    assert data.ncon>0
    depths=roles.depth_by_world(geoms,dist,torch.zeros(data.ncon,dtype=torch.long),
        active=torch.ones(data.ncon,dtype=torch.bool),num_worlds=1)
    assert depths["terrain"].item()==pytest.approx(.005,abs=2e-6)
    assert depths["self"].item()==0. and depths["payload"].item()==0.
    assert data.time==0.


def test_missing_ground_sensor_and_existing_terrain_are_rejected():
    cfg,ground=fixture();cfg.scene.sensors=()
    with pytest.raises(ValueError,match="ground-contact"):
        with_move_terrain(cfg,ground_model=ground,seed=6162,height_m=.020)
    cfg,ground=fixture();cfg.scene.terrain=object()
    with pytest.raises(ValueError,match="flat native scene"):
        with_move_terrain(cfg,ground_model=ground,seed=6162,height_m=.020)


def test_explicit_material_pair_is_not_silently_dropped():
    cfg,_=fixture()
    ground=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="2 2 .01"/>
      <geom name="probe" type="sphere" size=".01" pos="0 0 .02"/>
      </worldbody><contact><pair name="material" geom1="ground" geom2="probe"/></contact></mujoco>''')
    with pytest.raises(ValueError,match="material pairs"):
        with_move_terrain(cfg,ground_model=ground,seed=6163,height_m=.020)
