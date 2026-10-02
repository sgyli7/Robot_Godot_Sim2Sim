"""Physical spring and first-Tick stop conformance, separate from M0 admission."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose import implicit_metric
from bevy_microduck_tools.goose.artifacts import DT
from bevy_microduck_tools.goose.experimental_runtime import GooseExperimentalSourceRuntime
from bevy_microduck_tools.goose.runtime import GooseSourceRuntime


@pytest.fixture(autouse=True)
def logs_stay_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def fixture():
    root = ET.fromstring('''<mujoco>
      <option timestep=".02" gravity="0 0 0"/>
      <worldbody><body><inertial mass=".0002232" pos="0 0 0" diaginertia="1e-8 1e-8 1e-8"/>
        <joint name="pad" type="slide" axis="0 0 1" range="0 .0015" limited="true"
          stiffness="23357.0304" damping="2" armature="0"/>
      </body></worldbody>
    </mujoco>''')
    physical = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    ledger = implicit_metric.prepare_xml(root, physical)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return physical, model, ledger


@pytest.mark.parametrize("force", [5., 10., 20., 30.])
def test_native_first_tick_matches_physical_backward_euler_without_hidden_steps(force, monkeypatch):
    physical, model, ledger = fixture()
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = force
    calls = {"step1": 0, "step2": 0}
    for name in calls:
        native = getattr(mujoco, "mj_"+name)

        def counted(m, d, native=native, name=name):
            calls[name] += 1
            native(m, d)

        monkeypatch.setattr(mujoco, "mj_"+name, counted)
    implicit_metric.native_step(model, data)
    expected = force*DT**2/(.0002232+DT*2.+DT**2*23357.0304)
    assert data.qpos[0] == pytest.approx(expected, abs=1e-14)
    assert data.time == pytest.approx(.02)
    assert calls == {"step1": 1, "step2": 1}
    np.testing.assert_array_equal(model.body_mass, physical.body_mass)
    np.testing.assert_array_equal(model.body_inertia, physical.body_inertia)
    assert ledger["joints"][0]["physical_armature"] == 0.
    assert ledger["joints"][0]["compiled_armature"] > 9.


@pytest.mark.parametrize("q,force,expected", [(0., 40., .0015), (.0015, -40., 0.)])
def test_both_physical_bounds_stop_first_tick_without_projection(q, force, expected):
    _, model, _ = fixture()
    data = mujoco.MjData(model)
    data.qpos[0] = q
    data.qfrc_applied[0] = force
    implicit_metric.native_step(model, data)
    assert data.qpos[0] == pytest.approx(expected, abs=1e-8)
    assert data.qpos[0] == pytest.approx(q+DT*data.qvel[0], abs=1e-14)
    assert sum(data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)) == 2
    assert not any(w.number for w in data.warning)


def test_unforced_release_dissipates_physical_energy():
    _, model, _ = fixture()
    data = mujoco.MjData(model)
    data.qpos[0] = .0012
    data.qvel[0] = .1
    previous = .5*.0002232*data.qvel[0]**2+.5*23357.0304*data.qpos[0]**2
    for _ in range(100):
        implicit_metric.native_step(model, data)
        energy = .5*.0002232*data.qvel[0]**2+.5*23357.0304*data.qpos[0]**2
        assert energy <= previous+1e-10
        previous = energy
    assert previous < 1e-10


def test_experimental_runtime_rejects_an_ordinary_candidate(tmp_path):
    contract = tmp_path/"contract.json"
    contract.write_text('{"schema":"goose_50hz_candidate_si_v1","candidate":"goose_460_full50_v1"}')
    with pytest.raises(ValueError, match="explicit unqualified v2"):
        GooseExperimentalSourceRuntime(tmp_path/"missing_model.xml", contract)


def test_default_runtime_cannot_silently_use_compiled_numerical_inertia(tmp_path):
    contract = tmp_path/"contract.json"
    contract.write_text('{"numerical_metric":{"revision":"goose_joint_backward_euler_predictive_v1"}}')
    with pytest.raises(ValueError, match="explicit experimental runtime"):
        GooseSourceRuntime(tmp_path/"missing_model.xml", contract)


@pytest.mark.parametrize("downward_force", [1., 10., 100.])
def test_predictive_normal_contact_uses_actual_gap_and_one_integration(downward_force, monkeypatch):
    root = ET.fromstring('''<mujoco><option timestep=".02" gravity="0 0 0" iterations="100" tolerance="1e-12"/>
      <worldbody><geom type="plane" size="1 1 .1" condim="1"/>
        <body pos="0 0 .0105"><inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
          <joint name="vertical" type="slide" axis="0 0 1" limited="false"/>
          <geom type="box" size=".01 .01 .01" condim="1" margin=".002"/>
        </body>
      </worldbody></mujoco>''')
    physical = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    implicit_metric.prepare_xml(root, physical)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = -downward_force
    calls = {"step1": 0, "step2": 0}
    for name in calls:
        native = getattr(mujoco, "mj_"+name)

        def counted(m, d, native=native, name=name):
            calls[name] += 1
            native(m, d)

        monkeypatch.setattr(mujoco, "mj_"+name, counted)
    implicit_metric.native_step(model, data, normal_contact_mode="predictive_rigid")
    expected_displacement = max(-.0005, -downward_force*DT**2)
    assert data.qpos[0] == pytest.approx(expected_displacement, abs=1e-8)
    assert data.qpos[0] == pytest.approx(DT*data.qvel[0], abs=1e-14)
    assert data.time == pytest.approx(DT)
    assert calls == {"step1": 1, "step2": 1}
    assert all(c.dist == pytest.approx(.0005) for c in data.contact)
    assert not any(w.number for w in data.warning)
