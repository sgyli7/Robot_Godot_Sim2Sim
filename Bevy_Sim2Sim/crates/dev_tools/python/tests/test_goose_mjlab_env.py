"""Actual upstream manager-loop checks on a small CPU drive conformance rig."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")
from bevy_microduck_tools.goose.artifacts import CANDIDATES, sha256
from bevy_microduck_tools.goose.mjlab_baseline import build_reference
from bevy_microduck_tools.goose.mjlab_env import (
    GooseDevelopmentEnv, make_development_env_cfg, native_tick_clock_error,
)
from bevy_microduck_tools.goose.runtime import GooseSourceRuntime
from test_goose_50hz import fixture_runtime
from test_goose_batch_drive import nominal_graph
from test_goose_mjlab_baseline import make_euler_reference


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("seconds", (0., 64., 1800.))
def test_native_clock_accepts_one_step_after_long_running_time(dtype, seconds):
    before = torch.tensor([seconds], dtype=dtype)
    after = before + .02
    error = native_tick_clock_error(before, after)
    if dtype == torch.float32 and seconds == 64.:
        assert error.item() > 2e-6  # The previous guard falsely rejected this Tick.
    assert torch.equal(error, (after-before-.02).abs())


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("seconds", (0., 64., 1800.))
def test_native_clock_rejects_missing_or_duplicate_integrations(dtype, seconds):
    before = torch.tensor([seconds, seconds], dtype=dtype)
    expected = before + .02
    for wrong in (before, expected + .02):
        with pytest.raises(RuntimeError, match="exactly one"):
            native_tick_clock_error(before, wrong)
    with pytest.raises(RuntimeError, match="exactly one"):
        native_tick_clock_error(before, torch.stack((expected[0], before[1])))


def test_native_clock_rejects_nonfinite_or_unresolvable_timestamps():
    for before, after in ((0., float("nan")), (float("inf"), float("inf")),
                          (float(2**24), float(2**24))):
        with pytest.raises(RuntimeError, match="cannot resolve"):
            native_tick_clock_error(torch.tensor([before]), torch.tensor([after]))


@pytest.fixture(autouse=True)
def native_logs_stay_in_test_temp_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(params=("implicitfast", "Euler"))
def reference(tmp_path, request):
    original, path = fixture_runtime(tmp_path)
    model = tmp_path / "robot.xml"
    tree = ET.fromstring(model.read_text())
    option = tree.find("option")
    option.set("ccd_tolerance", "0.00000002")
    ET.SubElement(option, "flag", multiccd="enable")
    rotor = ET.SubElement(tree.find("./worldbody/body"), "body", name="beak_input_rotor")
    ET.SubElement(rotor, "inertial", mass=".01", pos="0 0 0", diaginertia=".001 .001 .001")
    ET.SubElement(rotor, "joint", name="beak_input_rotor", axis="0 1 0", range="-1 1")
    tree.find("actuator")[5].set("joint", "beak_input_rotor")
    model.write_text(ET.tostring(tree, encoding="unicode"))
    contract = nominal_graph(original.contract)
    contract.update(candidate=CANDIDATES[1], model_sha256=sha256(model),
                    torque_updates_per_tick=1, policy_calls_per_tick=1)
    path.write_text(json.dumps(contract))
    reference = build_reference(model, path, tmp_path / "reference")
    return make_euler_reference(*reference) if request.param == "Euler" else reference


def test_upstream_loop_commits_once_and_matches_native_drive_and_actor(reference):
    model, contract = reference
    cfg = make_development_env_cfg(model, contract)
    env = GooseDevelopmentEnv(cfg, "cpu")
    assert env.physics_dt == env.step_dt == .02
    assert env.single_action_space.shape == (18,)
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 65)
    term = env.action_manager.get_term("goose")
    assert term._entity.joint_names[int(term.motor_ids[5])] == "beak_input_rotor"
    assert int(term.qids[5]) == int(env.sim.mj_model.joint("robot/beak_hinge").qposadr[0])
    refs = [GooseSourceRuntime(model, contract) for _ in range(2)]
    class Prepared(Exception): pass
    def stop_before_integral(): raise Prepared()
    for r in refs: r._integrate = stop_before_integral
    calls = 0
    upstream_step = env.sim.step
    def counted_step():
        nonlocal calls
        calls += 1
        upstream_step()
    env.sim.step = counted_step
    actions = np.zeros((2, 18))
    actions[1, 5] = .03
    for tick in range(3):
        before_q = env.sim.data.qpos.cpu().numpy().copy()
        before_v = env.sim.data.qvel.cpu().numpy().copy()
        for i, r in enumerate(refs):
            r.data.qpos[:], r.data.qvel[:] = before_q[i], before_v[i]
            with pytest.raises(Prepared): r.step(actions[i])
        obs, reward, terminated, timeout, _ = env.step(torch.tensor(actions, dtype=torch.float64))
        assert calls == tick+1
        assert not bool(terminated.any() | timeout.any())
        assert reward.shape == (2,)
        np.testing.assert_allclose(env.sim.data.time.cpu(), (tick+1)*.02, atol=2e-6)
        np.testing.assert_allclose(term.drive.last_tau.cpu(), [r.last_tau for r in refs], atol=5e-12)
        np.testing.assert_array_equal(env.sim.data.ctrl.cpu(), np.array([r.last_tau for r in refs], dtype=np.float32))
        assert term.drive.prepared_ticks.tolist() == term.drive.completed_ticks.tolist() == [tick+1]*2
        for i, r in enumerate(refs):
            r.data.qpos[:], r.data.qvel[:] = env.sim.data.qpos[i].cpu(), env.sim.data.qvel[i].cpu()
            mujoco.mj_kinematics(r.model, r.data); mujoco.mj_comPos(r.model, r.data)
            r.actions = actions[i].copy()
            r.phase = (r.phase+2*np.pi*r.contract["phase_frequency_hz"]*.02) % (2*np.pi)
            np.testing.assert_allclose(obs["actor"][i].cpu(), r.observations(), atol=2e-6, rtol=0)
    assert not torch.equal(env.sim.data.qpos[0], env.sim.data.qpos[1])
    env.reset(env_ids=torch.tensor([0]))
    assert term.drive.completed_ticks.tolist() == [0, 3]
    assert not bool(term.drive.actions[0].any()) and bool(term.drive.actions[1].any())


@pytest.mark.parametrize("mutation", ["decimation", "timestep", "auto_reset", "identity"])
def test_invalid_development_configuration_is_rejected_before_environment_mutation(reference, mutation):
    model, contract = reference
    cfg = make_development_env_cfg(model, contract)
    if mutation == "decimation": cfg.decimation = 4
    elif mutation == "timestep": cfg.sim.mujoco.timestep = .005
    elif mutation == "auto_reset": cfg.auto_reset = True
    else:
        payload = json.loads(contract.read_text())
        payload["joints"][0]["kp_nm_rad"] *= 2
        contract.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="50 Hz|identity changed"):
        GooseDevelopmentEnv(cfg, "cpu")


@pytest.mark.parametrize("mutation", ["integrator", "implicit_damping", "physical_damping"])
def test_euler_configuration_cannot_override_its_contract_before_environment_creation(reference, mutation):
    model, contract = make_euler_reference(*reference)
    cfg = make_development_env_cfg(model, contract)
    if mutation == "integrator": cfg.sim.mujoco.integrator = "implicitfast"
    elif mutation == "implicit_damping": cfg.sim.mujoco.disableflags = ()
    else: cfg.sim.mujoco.disableflags += ("damper",)
    with pytest.raises(ValueError, match="integration profile changed"):
        GooseDevelopmentEnv(cfg, "cpu")


def test_pending_preparation_cannot_reset_physics_or_read_actor(reference):
    from bevy_microduck_tools.goose.mjlab_env import actor_observation
    model, contract = reference
    env = GooseDevelopmentEnv(make_development_env_cfg(model, contract), "cpu")
    env.reset()
    env.action_manager.process_action(torch.ones((2, 18)))
    before = env.sim.data.qpos.clone(), env.sim.data.qvel.clone(), env.sim.data.time.clone()
    with pytest.raises(RuntimeError, match="pending/failed"):
        env.reset()
    assert all(torch.equal(a, b) for a, b in zip(before, (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.time)))
    with pytest.raises(RuntimeError, match="successful Tick history"):
        actor_observation(env)


def test_two_actual_integrals_are_rejected_without_erasing_the_failed_state(reference):
    model, contract = reference
    env = GooseDevelopmentEnv(make_development_env_cfg(model, contract), "cpu")
    env.reset()
    actual_step = env.sim.step
    def intentionally_invalid_two_steps():
        actual_step(); actual_step()  # Failure injection, never a production setting.
    env.sim.step = intentionally_invalid_two_steps
    with pytest.raises(RuntimeError, match="one actual 20ms"):
        env.step(torch.zeros((2, 18)))
    term = env.action_manager.get_term("goose")
    assert term.drive.prepared_ticks.tolist() == [1, 1]
    assert term.drive.completed_ticks.tolist() == [0, 0]
    np.testing.assert_allclose(env.sim.data.time.cpu(), .04, atol=2e-6)
    before = env.sim.data.qpos.clone(), env.sim.data.time.clone()
    with pytest.raises(RuntimeError, match="pending/failed"):
        env.reset()
    assert torch.equal(before[0], env.sim.data.qpos) and torch.equal(before[1], env.sim.data.time)


def test_step_nonfinite_acceleration_aborts_before_forward_can_replace_evidence(reference):
    model, contract = reference
    env = GooseDevelopmentEnv(make_development_env_cfg(model, contract), "cpu")
    env.reset()
    actual_step = env.sim.step
    def injected_bad_acceleration():
        actual_step()
        env.sim.data.qacc[0, 0] = float("nan")
    env.sim.step = injected_bad_acceleration
    forwards = 0
    actual_forward = env.sim.forward
    def counted_forward():
        nonlocal forwards
        forwards += 1
        actual_forward()
    env.sim.forward = counted_forward
    with pytest.raises(FloatingPointError, match="Nonfinite Goose"):
        env.step(torch.zeros((2, 18)))
    assert forwards == 0 and not bool(torch.isfinite(env.sim.data.qacc).all())
    assert env.action_manager.get_term("goose").drive.completed_ticks.tolist() == [0, 0]
    with pytest.raises(RuntimeError, match="pending/failed"):
        env.reset()


def test_scene_preserves_native_material_options_and_distinguishes_default_and_loaded_birth(reference):
    from mjlab.scene import Scene
    model, contract = reference
    native = mujoco.MjModel.from_xml_path(str(model))
    cfg = make_development_env_cfg(model, contract)
    assert cfg.scene.entities["robot"].init_state.pos[2] == pytest.approx(native.qpos0[2]+.002)
    loaded = native.qpos0.copy(); loaded[2] -= .001
    loaded_cfg = make_development_env_cfg(model, contract, development_initial_qpos=loaded)
    assert loaded_cfg.scene.entities["robot"].init_state.pos[2] == pytest.approx(loaded[2])
    wrapped = Scene(cfg.scene, "cpu").compile()
    cfg.sim.mujoco.apply(wrapped)
    assert cfg.sim.mujoco.integrator == json.loads(contract.read_text())["integrator"].lower()
    assert wrapped.narena == native.narena
    for field in dir(native.opt):
        if not field.startswith("_") and not callable(value := getattr(native.opt, field)):
            np.testing.assert_array_equal(getattr(wrapped.opt, field), value)
    for field in ("contype", "conaffinity", "condim", "friction", "solref", "solimp", "margin", "gap", "priority", "solmix"):
        np.testing.assert_array_equal(getattr(wrapped, "geom_"+field)[wrapped.geom("ground").id],
                                      getattr(native, "geom_"+field)[native.geom("ground").id])
    assert json.loads(contract.read_text())["training_release"] is False
