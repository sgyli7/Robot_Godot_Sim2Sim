"""Physical six-DoF force/moment response and strict diagnostic scope."""
import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.condensed_contact import ContactMaterial
from bevy_microduck_tools.goose.condensed_free_contact import free_step, free_step_be_metric


@pytest.fixture(autouse=True)
def outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def fixture(frictionless=True):
    geoms = []
    bindings = []
    for index, (x,y) in enumerate([(x,y) for x in [-.08,0.,.08] for y in [-.025,.025]]):
        for stop in (False,True):
            geoms.append(f'<geom name="p{index}_{int(stop)}" type="box" size=".002 .01 .001" pos="{x} {y} {0.0015 if stop else 0.}"/>')
            bindings.append((f'p{index}_{int(stop)}',f'p{index}',stop))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".02" gravity="0 0 0" solver="Newton" integrator="implicit"
              iterations="100" tolerance="1e-12"><flag autoreset="disable"/></option>
      <default><geom condim="{1 if frictionless else 3}" friction="0 0 0" margin=".1"/></default>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .0007"><inertial mass=".293" pos="0 0 0" diaginertia=".001 .002 .003"/>
          <freejoint/>{''.join(geoms)}</body></worldbody></mujoco>''')
    materials = tuple(ContactMaterial(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name),
                                     patch,23357.0304,2.,stop) for name,patch,stop in bindings)
    return model,mujoco.MjData(model),materials


@pytest.mark.parametrize('torque', [.003,-.003])
def test_coupled_translation_and_rotation_match_physical_backward_euler(torque, monkeypatch):
    model,data,materials = fixture()
    before = data.qpos.copy()
    data.xfrc_applied[1] = [0.,0.,-30.,torque,-torque,0.]
    calls = {'step1':0,'step2':0}
    for name in calls:
        native = getattr(mujoco,'mj_'+name)
        def counted(m,d,native=native,name=name):
            calls[name] += 1
            native(m,d)
        monkeypatch.setattr(mujoco,'mj_'+name,counted)
    rows = free_step(model,data,materials)
    mass = np.diag([.293,.293,.293,.001,.002,.003])
    tangent = np.zeros((6,6))
    load = data.xfrc_applied[1].copy()
    for row in rows:
        if row['travel_stop']:
            continue
        x,y,_ = row['point_world_m']
        j = np.array([0.,0.,1.,y,-x,0.])
        tangent += (.02*row['point_damping_n_s_m']+.02**2*row['point_stiffness_n_m'])*np.outer(j,j)
        load -= row['point_stiffness_n_m']*row['gap_m']*j
    expected = np.linalg.solve(mass+tangent,.02*load)
    np.testing.assert_allclose(data.qvel,expected,atol=2e-12,rtol=0.)
    np.testing.assert_allclose(data.qpos[:3],before[:3]+.02*expected[:3],atol=1e-13)
    assert abs(data.qvel[3]) > 1e-5 and abs(data.qvel[4]) > 1e-5
    assert data.time == pytest.approx(.02)
    assert calls == {'step1':1,'step2':1}
    assert not any(w.number for w in data.warning)


def test_friction_rejected_without_integrating():
    model,data,materials = fixture(False)
    with pytest.raises(ValueError,match='frictionless'):
        free_step(model,data,materials)
    assert data.time == 0.


def test_guided_or_whole_body_topology_rejected_without_integrating():
    model,data,materials = fixture()
    model.jnt_type[0] = int(mujoco.mjtJoint.mjJNT_SLIDE)
    with pytest.raises(ValueError,match='physical free foot'):
        free_step(model,data,materials)
    assert data.time == 0.


@pytest.mark.parametrize('axis', [3,4])
@pytest.mark.parametrize('angular_speed', [-2.5,2.5])
def test_be_metric_contact_force_uses_completed_velocity_with_offset_com(axis, angular_speed):
    model,data,materials = fixture()
    # Declared analytical fixture modification, not a candidate mass edit.
    model.body_ipos[1] = [.01,0.,-.005]
    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    data.qvel[axis] = angular_speed
    data.qvel[2] = -.02
    data.xfrc_applied[1,:3] = [0.,0.,-30.]
    rows = free_step_be_metric(model,data,materials)
    jacobian = np.asarray(data.efc_J).reshape(data.nefc,6)
    for row in rows:
        if row['travel_stop']:
            continue
        velocity = float(jacobian[row['row']]@data.qvel)
        expected = max(0.,-row['point_stiffness_n_m']*(row['gap_m']+.02*velocity)
                           -row['point_damping_n_s_m']*velocity)
        assert data.efc_force[row['row']] == pytest.approx(expected,abs=2e-10)
    assert data.time == pytest.approx(.02)
    assert not any(w.number for w in data.warning)


def test_be_metric_rejects_undeclared_second_damping_solve():
    model,data,materials = fixture()
    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    with pytest.raises(ValueError,match='eulerdamp disable'):
        free_step_be_metric(model,data,materials)
    assert data.time == 0.
