"""Actual native contact activation, physical boundary and Coulomb force checks."""
import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose import ground_discrete as method
from bevy_microduck_tools.goose.predictive_discrete import advance


@pytest.fixture(autouse=True)
def native_logs_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def contact_box():
    if mujoco.__version__ != "3.13.0":
        pytest.skip("Ground experiment is pinned to native 3.13.0")
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".02" integrator="discrete" solver="Newton" cone="elliptic" iterations="100" tolerance="1e-12">
        <flag autoreset="disable" eulerdamp="disable" diagexact="enable"/>
      </option>
      <worldbody>
        <geom name="ground" type="plane" size="1 1 .1" margin=".01" friction=".6 .001 .0001"/>
        <body pos="0 0 .025">
          <freejoint/>
          <inertial mass="1" pos="0 0 0" diaginertia=".0002666666667 .0002666666667 .0002666666667"/>
          <geom type="box" size=".02 .02 .02" friction=".6 .001 .0001"/>
        </body>
      </worldbody>
    </mujoco>''')
    return model, mujoco.MjData(model)


def test_wrong_engine_rejected_before_files(tmp_path):
    if mujoco.__version__ == "3.13.0":
        pytest.skip("Preserved earlier engine supplies this boundary check")
    with pytest.raises(ValueError, match="isolated MuJoCo 3.13.0"):
        method.GooseGroundDiscreteRuntime(tmp_path/"absent.xml", tmp_path/"absent.json")


def test_detection_margin_does_not_become_physical_surface(contact_box, monkeypatch):
    model, data = contact_box
    data.qvel[2] = -.4
    calls = []
    native_euler = mujoco.mj_Euler

    def counted_euler(m, d):
        calls.append(float(d.time))
        native_euler(m, d)

    def prepare(m, d):
        assert d.time == 0 and d.ncon == 4
        assert all(c.dist == pytest.approx(.005, abs=1e-12) for c in d.contact)
        return method.prepare_ground_constraints(m, d, m.geom("ground").id)

    monkeypatch.setattr(mujoco, "mj_Euler", counted_euler)
    receipt = advance(model, data, prepare_constraints=prepare)
    assert calls == [0.]
    assert data.time == .02 and receipt["integrations"] == 1
    assert receipt["constraint_preparation"]["ground_rows"] == 12
    assert data.qpos[2] == pytest.approx(.02, abs=1e-6)
    assert not any(w.number for w in data.warning)


def test_native_forces_stay_in_coulomb_cone_and_dissipate(contact_box):
    model, data = contact_box
    data.qpos[2] = .02
    data.qvel[:3] = [.2, 0., -.1]
    receipt = advance(model, data, prepare_constraints=lambda m, d:
        method.prepare_ground_constraints(m, d, m.geom("ground").id))
    assert receipt["integrations"] == 1
    velocity = np.empty(data.nefc)
    mujoco.mj_mulJacVec(model, data, velocity, data.qvel)
    normal_sum, power = 0., 0.
    for index, contact in enumerate(data.contact):
        wrench = np.empty(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        assert wrench[0] >= -1e-8
        assert np.linalg.norm(wrench[1:3]/contact.friction[:2]) <= wrench[0]+1e-6
        normal_sum += wrench[0]
        address = contact.efc_address
        power += float(wrench[1:3] @ velocity[address+1:address+3])
    assert normal_sum > 9.81
    assert power <= 1e-6


def test_preparation_cannot_skip_corrected_solve(contact_box):
    model, data = contact_box
    before = data.qpos.copy()
    with pytest.raises(ValueError, match="requires the corrected solve"):
        advance(model, data, predictive=False, prepare_constraints=lambda m, d: None)
    assert data.time == 0
    np.testing.assert_array_equal(data.qpos, before)


def test_stale_broadphase_margin_rejected_before_row_changes(contact_box):
    model, data = contact_box
    mujoco.mj_forward(model, data)
    model.body_margin[0] = 0.
    regularization = data.efc_R.copy()
    with pytest.raises(ValueError, match="stale body-margin cache"):
        method.prepare_ground_constraints(model, data, model.geom("ground").id)
    assert data.time == 0
    np.testing.assert_array_equal(data.efc_R, regularization)
