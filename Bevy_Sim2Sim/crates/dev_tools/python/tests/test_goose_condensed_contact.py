"""Analytic physical-load checks for the isolated 20ms normal-contact experiment."""

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.condensed_contact import ContactMaterial, guided_step

K = 23357.0304
C = 2.0
H = 0.02


@pytest.fixture(autouse=True)
def logs_stay_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def fixture(mass, gap=-0.0003, backing=False, frictionless=True):
    stop = '<geom name="stop" type="box" pos="0 0 .0015" size=".01 .01 .001"/>' if backing else ''
    dim = 1 if frictionless else 3
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".02" gravity="0 0 0" solver="Newton" integrator="implicit"
              iterations="100" tolerance="1e-12"><flag autoreset="disable"/></option>
      <default><geom condim="{dim}" friction="0 0 0" margin="1"/></default>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 {gap+.001}">
          <inertial mass="{mass}" pos="0 0 0" diaginertia=".001 .001 .001"/>
          <joint type="slide" axis="0 0 1" limited="false"/>
          <geom name="soft" type="box" size=".01 .01 .001"/>{stop}
        </body>
      </worldbody></mujoco>''')
    materials = [ContactMaterial(1, 'physical_patch', K, C)]
    if backing:
        materials.append(ContactMaterial(2, 'physical_patch', K, C, True))
    return model, mujoco.MjData(model), tuple(materials)


@pytest.mark.parametrize('mass', [0.29298065203309476, 2.0, 10.0])
@pytest.mark.parametrize('force', [5.0, 10.0, 20.0, 30.0])
def test_native_step_matches_coupled_backward_euler(mass, force, monkeypatch):
    model, data, materials = fixture(mass)
    data.qvel[0] = -0.02
    data.qfrc_applied[0] = -force
    before_mass = model.body_mass.copy()
    before_inertia = model.body_inertia.copy()
    calls = {'step1': 0, 'step2': 0}
    for name in calls:
        native = getattr(mujoco, 'mj_'+name)

        def counted(m, d, native=native, name=name):
            calls[name] += 1
            native(m, d)

        monkeypatch.setattr(mujoco, 'mj_'+name, counted)
    rows = guided_step(model, data, materials)
    expected_v = (mass*(-0.02)-H*force-H*K*(-0.0003))/(mass+H*C+H*H*K)
    assert data.qvel[0] == pytest.approx(expected_v, abs=1e-12)
    assert data.qpos[0] == pytest.approx(H*expected_v, abs=1e-13)
    assert data.time == pytest.approx(H)
    assert calls == {'step1': 1, 'step2': 1}
    assert sum(row['point_stiffness_n_m'] for row in rows) == pytest.approx(K)
    np.testing.assert_array_equal(model.body_mass, before_mass)
    np.testing.assert_array_equal(model.body_inertia, before_inertia)


@pytest.mark.parametrize('mass', [0.29298065203309476, 10.0])
def test_static_load_balance_is_independent_of_contact_mass(mass):
    model, data, materials = fixture(mass)
    data.qfrc_applied[0] = -K*0.0003
    guided_step(model, data, materials)
    assert abs(data.qvel[0]) < 1e-12
    assert abs(data.qpos[0]) < 1e-13


def test_backing_limits_overload_without_projection_or_substeps():
    model, data, materials = fixture(0.29298065203309476, gap=0.0005, backing=True)
    data.qfrc_applied[0] = -40.0
    for tick in range(100):
        previous = float(data.qpos[0])
        guided_step(model, data, materials)
        compression = -0.0005-float(data.qpos[0])
        assert compression <= 0.0015+1e-8
        assert data.qpos[0] == pytest.approx(previous+H*data.qvel[0], abs=1e-13)
        assert data.time == pytest.approx((tick+1)*H)
    assert compression == pytest.approx(0.0015, abs=1e-8)
    assert not any(w.number for w in data.warning)


def test_release_dissipates_physical_spring_energy():
    mass = 0.29298065203309476
    model, data, materials = fixture(mass)
    previous = 0.5*K*0.0003**2
    for _ in range(100):
        guided_step(model, data, materials)
        compression = max(0.0, 0.0003-float(data.qpos[0]))
        energy = 0.5*mass*data.qvel[0]**2+0.5*K*compression**2
        assert energy <= previous+1e-12
        previous = energy


def test_frictional_rows_are_rejected():
    model, data, materials = fixture(1.0, frictionless=False)
    with pytest.raises(ValueError, match='frictionless ground'):
        guided_step(model, data, materials)
    assert data.time == 0.0


def test_other_degrees_of_freedom_are_rejected_before_integration():
    model, data, materials = fixture(1.0)
    model.jnt_type[0] = int(mujoco.mjtJoint.mjJNT_HINGE)
    with pytest.raises(ValueError, match='one-slide'):
        guided_step(model, data, materials)
    assert data.time == 0.0
