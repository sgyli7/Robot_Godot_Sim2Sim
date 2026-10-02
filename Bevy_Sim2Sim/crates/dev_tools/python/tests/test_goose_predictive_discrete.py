"""Actual scalar stops, time advancement and post-solve force sensor conformance."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose import predictive_discrete as method


@pytest.fixture(autouse=True)
def native_logs_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def predictive_pad(tmp_path):
    if mujoco.__version__ != "3.13.0":
        pytest.skip("Native effective metric is pinned to 3.13.0")
    root = ET.fromstring('''<mujoco>
      <option timestep=".02" integrator="discrete" solver="Newton" iterations="100" tolerance="1e-12" gravity="0 0 0">
        <flag autoreset="disable"/>
      </option>
      <worldbody><body name="pad">
        <inertial mass=".002" pos="0 0 0" diaginertia=".00001 .00001 .00001"/>
        <joint name="compression" type="slide" axis="0 0 1" stiffness="23357" damping="1.39" range="0 .0015" limited="true"/>
        <site name="tip" pos="0 0 0"/>
      </body></worldbody>
      <sensor><force site="tip"/></sensor>
    </mujoco>''')
    source = tmp_path/"source.xml"
    ET.ElementTree(root).write(source, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(source))
    method.prepare_limits(root, model)
    target = tmp_path/"predictive.xml"
    ET.ElementTree(root).write(target, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(target))
    return model, mujoco.MjData(model)


def test_unsupported_engine_is_rejected_before_loading(tmp_path):
    if mujoco.__version__ == "3.13.0":
        pytest.skip("The preserved earlier engine supplies this boundary check")
    with pytest.raises(ValueError, match="isolated MuJoCo 3.13.0"):
        method.GoosePredictiveDiscreteRuntime(tmp_path/"absent.xml", tmp_path/"absent.json")


def test_overload_and_release_respect_real_travel_without_extra_advance(predictive_pad, monkeypatch):
    model, data = predictive_pad
    calls = {"advance": 0}
    native_euler = mujoco.mj_Euler

    def counted_advance(m, d):
        calls["advance"] += 1
        native_euler(m, d)

    monkeypatch.setattr(mujoco, "mj_Euler", counted_advance)
    for tick in range(100):
        data.qfrc_applied[0] = 40. if tick < 50 else 0.
        receipt = method.advance(model, data)
        assert receipt["integrations"] == 1 and receipt["coordinate_writes"] == 0
        assert receipt["preview_constraint_solves"] == receipt["corrected_constraint_solves"] == 1
        assert calls["advance"] == tick+1
        assert data.time == pytest.approx((tick+1)*.02, abs=1e-12)
        assert -.00000002 <= data.qpos[0] <= .00150002
        assert not any(w.number for w in data.warning)
    assert abs(data.qpos[0]) < 1e-6


def test_actual_native_force_sensor_recomputed_after_corrected_solve(predictive_pad, monkeypatch):
    model, data = predictive_pad
    native_sensor = mujoco.mj_sensorAcc
    native_force = mujoco.mj_fwdConstraint
    count = {"sensor": 0, "constraint": 0}

    def counted_force(m, d):
        count["constraint"] += 1
        native_force(m, d)
        # The first native preview contains a force-sensor RNE cache.
        assert d.flg_rnepost == 1

    def counted_sensor(m, d):
        count["sensor"] += 1
        assert d.flg_rnepost == 0
        native_sensor(m, d)
        assert d.flg_rnepost == 1

    monkeypatch.setattr(mujoco, "mj_fwdConstraint", counted_force)
    monkeypatch.setattr(mujoco, "mj_sensorAcc", counted_sensor)
    data.qfrc_applied[0] = 40.
    method.advance(model, data)
    assert count == {"sensor": 1, "constraint": 1}
    assert np.isfinite(data.sensordata).all()


def test_bad_solver_fails_before_state_advance(predictive_pad):
    model, data = predictive_pad
    model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
    qpos = data.qpos.copy()
    with pytest.raises(ValueError, match="Unsupported"):
        method.advance(model, data)
    assert data.time == 0.
    np.testing.assert_array_equal(data.qpos, qpos)


def test_exact_profile_requires_actual_compiled_option(tmp_path):
    if mujoco.__version__ != "3.13.0":
        pytest.skip("The exact profile uses native 3.13.0")
    import json
    from test_goose_50hz import fixture_runtime
    from bevy_microduck_tools.goose.artifacts import sha256
    original, path = fixture_runtime(tmp_path, native_discrete=True)
    model_path = tmp_path/"robot.xml"
    tree = ET.parse(model_path)
    rows = method.prepare_limits(tree.getroot(), original.model)
    tree.write(model_path, encoding="unicode")
    contract = json.loads(path.read_text())
    contract.update(candidate=method.CANDIDATE_EXACT, model_sha256=sha256(model_path))
    contract["native_discrete"].update(revision=method.REVISION,
        predictive_limits={"rows": rows, "exact_diagonal": True})
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Compiled exact-diagonal flag"):
        method.GoosePredictiveDiscreteRuntime(model_path, path)
    tree.getroot().find("./option/flag").set("diagexact", "enable")
    tree.write(model_path, encoding="unicode")
    contract["model_sha256"] = sha256(model_path)
    path.write_text(json.dumps(contract))
    controller = method.GoosePredictiveDiscreteRuntime(model_path, path)
    obs, state = controller.step(np.zeros(18))
    assert obs.shape == (65,)
    assert state["controller_updates"] == state["physics_integrations"] == 1
    assert controller.advance_receipt["integrations"] == 1
