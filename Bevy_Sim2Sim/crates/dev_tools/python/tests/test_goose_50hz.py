"""Regression checks for Goose timing, inertia and admission boundaries."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from bevy_microduck_tools.goose.artifacts import combine_rigid_properties, DT, JOINT_ORDER, sha256, write_json
from bevy_microduck_tools.goose.preflight import require_m0_admission, REQUIRED_M0_CHECKS
from bevy_microduck_tools.goose.runtime import GooseSourceRuntime
from bevy_microduck_tools.goose.task import TaskGoal


@pytest.fixture(autouse=True)
def native_logs_stay_in_test_temp_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_full_inertia_condensation_preserves_mass_moments():
    parts = [(2., np.array([1., 0., 0.]), np.diag([.1, .2, .3])),
             (.5, np.array([0., 2., 0.]), np.array([[.05, .01, 0.], [.01, .06, .02], [0., .02, .07]]))]
    mass, center, inertia = combine_rigid_properties(parts)
    original_about_origin = sum((tensor + m*(np.eye(3)*c.dot(c)-np.outer(c,c)) for m,c,tensor in parts), start=np.zeros((3,3)))
    combined_about_origin = inertia + mass*(np.eye(3)*center.dot(center)-np.outer(center,center))
    np.testing.assert_allclose(combined_about_origin, original_about_origin, atol=1e-15)
    np.testing.assert_allclose(mass*center, [2., 1., 0.], atol=1e-15)
    assert inertia[0, 1] != 0  # Off-diagonal terms must survive.


def test_admission_cannot_use_finite_only_or_stale_target():
    candidate = {"candidate": "goose_460_full50_v1", "model_sha256": "m", "contract_sha256": "c",
                 "timing": {"physics_dt_s": DT, "torque_dt_s": DT, "policy_dt_s": DT, "substeps": 1}}
    source = {**candidate, "engine": "mujoco_cpu", "source_complete": True,
              "checks": {"numerical": {"status": "passed"}, "pad_curve": {"status": "not_checked"}}}
    target = {**candidate, "engine": "rapier", "target_complete": True, "checks": {"numerical": {"status": "passed"}}}
    with pytest.raises(ValueError, match="untested"):
        require_m0_admission(candidate, source, target)
    source["checks"] = {name: {"status": "passed"} for name in REQUIRED_M0_CHECKS}
    target["candidate"] = "goose_460_condensed50_v1"
    with pytest.raises(ValueError, match="identity"):
        require_m0_admission(candidate, source, target)


def fixture_runtime(tmp_path, skill="locomotion", *, native_discrete=False):
    # Real MuJoCo integration in a deliberately small controller conformance rig.
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 -9.81",
                  integrator="discrete" if native_discrete else "implicit")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="1 1 .1")
    torso = ET.SubElement(world, "body", name="torso", pos="0 0 1")
    ET.SubElement(torso, "freejoint")
    ET.SubElement(torso, "inertial", mass="1", pos="0 0 0", diaginertia=".1 .1 .1")
    specs = []
    actuators = ET.SubElement(root, "actuator")
    for index, name in enumerate(JOINT_ORDER):
        body = ET.SubElement(torso, "body", name=name)
        ET.SubElement(body, "inertial", mass=".01", pos=".01 0 0", diaginertia=".001 .001 .001")
        ET.SubElement(body, "joint", name=name, axis="0 1 0", range="-1 1", damping=".01")
        ET.SubElement(actuators, "motor", name=name+"_motor", joint=name, ctrlrange="-2 2", ctrllimited="true")
        specs.append({"name": name, "kp_nm_rad": 2., "kd_nm_s_rad": .1, "torque_peak_limit_nm": 2.,
                      "continuous_design_limit_nm": 1., "speed_limit_rad_s": 1., "action_scale_rad": 1.,
                      "q_neutral_rad": 0., "range_rad": [-1., 1.]})
    model = tmp_path / "robot.xml"
    ET.ElementTree(root).write(model, encoding="unicode")
    module = tmp_path / "src/sai_agent/goose/stage_one_gravity.py"
    module.parent.mkdir(parents=True)
    module.write_text('import numpy as np\nclass NominalNeckGravity:\n def __init__(self,contract):pass\n def __call__(self,q,orientation):return np.zeros(5)\n')
    contract = {"physics_dt_s": DT, "torque_dt_s": DT, "policy_dt_s": DT,
                "physics_steps_per_tick": 1, "joint_order": list(JOINT_ORDER), "model_sha256": sha256(model),
                "joints": specs, "passive_contacts": [], "collision_geometries": [], "passive_linkage_joints": [],
                "bodies": [], "phase_frequency_hz": 1.2, "positive_mechanical_power_limit_w": 350.,
                "source_checkpoint": {"source_root": str(tmp_path), "source_module_sha256": {"src/sai_agent/goose/stage_one_gravity.py": sha256(module)}}}
    path = tmp_path / "contract.json"
    runtime_type = GooseSourceRuntime
    if native_discrete:
        from bevy_microduck_tools.goose.native_discrete_runtime import (
            CANDIDATE, ENGINE_VERSION, REVISION, GooseDiscreteSourceRuntime)
        contract.update(candidate=CANDIDATE, integrator="discrete", native_discrete={
            "revision": REVISION, "engine_version": ENGINE_VERSION,
            "physical_parameters_changed": False})
        runtime_type = GooseDiscreteSourceRuntime
    write_json(path, contract)
    return runtime_type(model, path, skill=skill), path


def test_real_tick_has_single_integration_and_20ms_slew(tmp_path):
    runtime, _ = fixture_runtime(tmp_path)
    action = np.ones(18)
    before = runtime.data.time
    _, state = runtime.step(action)
    assert runtime.data.time-before == pytest.approx(.02)
    assert state["physics_integrations"] == state["controller_updates"] == 1
    np.testing.assert_allclose(runtime.target, .02)
    assert runtime.observations().shape == (65,)
    for _ in range(9):
        runtime.step(action)
    assert runtime.data.time == pytest.approx(.2)
    assert runtime.physics_integrations == 10


def test_static_action_inverse_matches_actual_settled_drive_without_changing_semantics(tmp_path):
    from bevy_microduck_tools.goose.drive_static import static_drive_targets
    runtime, _ = fixture_runtime(tmp_path)
    q = np.linspace(-.2, .2, 18)
    required = np.linspace(-.4, .4, 18)
    result = static_drive_targets(runtime.contract, q, required, np.zeros(5))
    assert result["feasible"] and not result["physical_pose_qualified"]
    assert max(result["minimum_slew_updates"]) > 1
    assert not np.allclose(result["stationary_first_update_torque_nm"], required)
    # A settled development start is declared explicitly; it is not a cold reset.
    runtime.data.qpos[runtime.qidx] = q
    runtime.target = np.asarray(result["bounded_target_rad"])
    _, state = runtime.step(result["action"])
    np.testing.assert_allclose(runtime.last_tau, required, atol=1e-12)
    assert state["physics_integrations"] == state["controller_updates"] == 1
    assert state["time_s"] == pytest.approx(DT)


def test_static_action_check_rejects_peak_only_or_clipped_holding_efforts(tmp_path):
    from bevy_microduck_tools.goose.drive_static import static_drive_targets
    runtime, _ = fixture_runtime(tmp_path)
    q, required = np.zeros(18), np.zeros(18)
    required[6] = 1.2  # Below peak 2 Nm but above continuous 1 Nm.
    q[7], required[7] = .99, .2  # Legal pose, impossible positive target offset.
    result = static_drive_targets(runtime.contract, q, required, np.zeros(5))
    assert not result["feasible"]
    assert not result["axis_checks"][6]["within_continuous_cap"]
    assert not result["axis_checks"][7]["target_reproduces_static_effort"]
    assert result["action"][7] == 1.
    assert result["stationary_settled_torque_nm"][7] == pytest.approx(.02)
    assert result["physics_integrations"] == 0


def test_static_beak_effort_has_no_extra_feedforward_or_out_of_range_target(tmp_path):
    from bevy_microduck_tools.goose.drive_static import static_drive_targets
    runtime, _ = fixture_runtime(tmp_path)
    runtime.contract["joints"][5].update(range_rad=[0., .55], action_scale_rad=.55)
    required = np.zeros(18)
    required[:5] = .3
    required[5] = -.01
    result = static_drive_targets(runtime.contract, np.zeros(18), required, np.full(5, .3))
    assert all(row["feasible"] for row in result["axis_checks"][:5])
    assert not result["axis_checks"][5]["feasible"]
    assert result["bounded_target_rad"][5] == 0.
    assert result["stationary_settled_torque_nm"][5] == 0.
    with pytest.raises(ValueError, match="5 finite"):
        static_drive_targets(runtime.contract, np.zeros(18), required, np.zeros(18))


def test_recovery_retains_low_com_state_no_auto_reset(tmp_path):
    runtime, _ = fixture_runtime(tmp_path, skill="recovery")
    runtime.data.qpos[2] = .05
    _, state = runtime.step(np.zeros(18))
    assert state["height_m"] < .18
    assert state["failure"] is False
    assert state["auto_reset"] is False
    assert runtime.age == 1


def test_actor_pose_and_goal_use_post_integration_state_without_extra_solve(tmp_path, monkeypatch):
    import mujoco
    runtime, _ = fixture_runtime(tmp_path, skill="pickup")
    runtime.goal = TaskGoal("object", (1., 2., 1.), (1., 0., 0., 0.), (2., 0., 1.))
    runtime.data.qvel[:6] = [.3, -.2, .1, .6, -.4, .2]
    calls = {"integrations": 0, "extra_forward_solves": 0}
    native_step, native_forward = mujoco.mj_step, mujoco.mj_forward

    def counted_step(model, data):
        calls["integrations"] += 1
        native_step(model, data)

    def counted_forward(model, data):
        calls["extra_forward_solves"] += 1
        native_forward(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    monkeypatch.setattr(mujoco, "mj_forward", counted_forward)
    obs, state = runtime.step(np.zeros(18))
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, runtime.data.qpos[3:7])
    rotation = rotation.reshape(3, 3)
    np.testing.assert_allclose(runtime.data.xmat[runtime.torso].reshape(3, 3), rotation, atol=1e-14)
    np.testing.assert_allclose(obs[3:6], rotation.T @ [0., 0., -1.], atol=1e-7)
    np.testing.assert_allclose(obs[65:68], rotation.T @ (np.array([1., 2., 1.])-runtime.data.qpos[:3]), atol=1e-7)
    expected_com = (runtime.model.body_mass[:, None]*runtime.data.xipos).sum(axis=0)/runtime.model.body_mass.sum()
    assert state["height_m"] == pytest.approx(expected_com[2], abs=1e-14)
    assert calls == {"integrations": 1, "extra_forward_solves": 0}
    assert runtime.data.time == pytest.approx(DT)


def test_native_warning_is_preserved_and_aborts_without_reset(tmp_path):
    import mujoco
    runtime, _ = fixture_runtime(tmp_path)
    assert runtime.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    runtime.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 1
    with pytest.raises(FloatingPointError, match="no automatic reset"):
        runtime.step(np.zeros(18))
    assert runtime.data.time == pytest.approx(DT)
    assert runtime.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number == 1


def test_pickup_observation_is_82_without_privileged_mass(tmp_path):
    runtime, _ = fixture_runtime(tmp_path, skill="pickup")
    obs = runtime.observations()
    assert obs.shape == (82,)
    assert obs[-1] == 0
    assert obs[68] == 0 and obs[75] == 1  # invalid pose zero; approach onehot retained
    runtime.goal = TaskGoal("object", (1., 0., 1.), (1., 0., 0., 0.), (2., 0., 1.))
    runtime.pickup_stage = 4
    extension = runtime.observations()[65:]
    np.testing.assert_allclose(extension[:3], [1., 0., -.002], atol=1e-6)
    assert extension[-1] == 1 and extension[14] == 1
    assert extension[10:16].sum() == 1


def test_old_source_timing_cannot_enter_goose_runtime(tmp_path):
    runtime, path = fixture_runtime(tmp_path)
    contract = json.loads(path.read_text())
    contract["physics_dt_s"] = .0001
    write_json(path, contract)
    with pytest.raises(ValueError, match=".02s"):
        GooseSourceRuntime(tmp_path/"robot.xml", path)


def test_frozen_experiment_and_candidate_outputs_cannot_be_overwritten(tmp_path):
    from bevy_microduck_tools.goose.artifacts import build_candidate
    from bevy_microduck_tools.goose.preflight import main, run_candidate
    marker = tmp_path / "source_receipt.json"
    marker.write_text("frozen experiment")
    with pytest.raises(FileExistsError, match="not empty"):
        build_candidate(tmp_path, tmp_path, {}, {}, "goose_460_full50_v1")
    with pytest.raises(FileExistsError, match="Frozen source"):
        run_candidate({"model_path": str(tmp_path/"robot.xml")}, {})
    with pytest.raises(FileExistsError, match="not empty"):
        main(["--source-root", str(tmp_path), "--output", str(tmp_path)])
    assert marker.read_text() == "frozen experiment"


def test_pickup_goal_uses_body_frame_and_quaternion_sign():
    goal = TaskGoal("object", (0., 2., 0.), (-1., 0., 0., 0.), (0., 3., 0.))
    body_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    extension = goal.actor_extension(np.zeros(3), np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]), body_rotation, phase=1)
    np.testing.assert_allclose(extension[:3], [2., 0., 0.], atol=1e-7)
    np.testing.assert_allclose(extension[3:7], [np.sqrt(.5), 0., 0., -np.sqrt(.5)], atol=1e-7)
    np.testing.assert_allclose(extension[7:10], [3., 0., 0.], atol=1e-7)
    assert extension[11] == 1


def test_invalid_pickup_goal_retains_stage_and_low_reach(tmp_path):
    runtime, _ = fixture_runtime(tmp_path, skill="pickup")
    runtime.goal = TaskGoal("stale", (0., 0., 0.), (1., 0., 0., 0.), (0., 0., 0.), valid=False)
    runtime.pickup_stage = 3
    extension = runtime.observations()[65:]
    assert extension[:10].sum() == 0 and extension[-1] == 0 and extension[13] == 1
    runtime.data.qpos[2] = .05
    _, state = runtime.step(np.zeros(18))
    assert state["failure"] is False and state["task_goal_invalid"] is True


def test_payload_static_gravity_is_si_virtual_work():
    import mujoco
    from bevy_microduck_tools.goose.reach_static import generalized_gravity
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/><worldbody>
      <body name="link"><joint name="pitch" axis="0 1 0"/>
        <inertial mass="2" pos="1 0 0" diaginertia=".1 .1 .1"/>
        <site name="grip" pos="2 0 0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    torque = generalized_gravity(model, data, payload_mass_kg=.3, grip_site_id=model.site("grip").id)
    assert torque[0] == pytest.approx(-(2*9.81*1 + .3*9.81*2))


def test_double_foot_hull_screen_is_explicit_geometric_condition():
    from bevy_microduck_tools.goose.reach_static import _convex_hull, inside_support
    hull = _convex_hull([[-1,-1], [1,-1], [1,1], [-1,1], [0,0]])
    assert inside_support([0,0], hull)
    assert inside_support([1,0], hull)
    assert not inside_support([1.01,0], hull)


@pytest.mark.parametrize("shape", ["sphere", "box"])
def test_native_kelvin_step_matches_backward_euler_without_substep(shape):
    import mujoco
    from bevy_microduck_tools.goose.contact_diagnostic import fixture, kelvin_native_step, K, C
    mass, force = .29298065203309465, 20.
    model = fixture(shape,mass)
    data = mujoco.MjData(model)
    points, native_force = kelvin_native_step(model,data,force)
    expected_velocity = (-force*DT + K*1e-6*DT)/(mass+C*DT+K*DT**2)
    assert data.qvel[0] == pytest.approx(expected_velocity, abs=1e-11)
    assert data.qpos[0] == pytest.approx(expected_velocity*DT, abs=1e-11)
    assert data.time == pytest.approx(DT)
    assert points == (1 if shape == "sphere" else 4)
    assert native_force == pytest.approx(K*(1e-6-data.qpos[0])-C*data.qvel[0],abs=1e-10)


def test_native_kelvin_contact_rejects_over_travel():
    from bevy_microduck_tools.goose.contact_diagnostic import one_case
    result = one_case("sphere",1.,40.,"step")
    assert result["status"] == "failed"
    assert result["failure_reason"] == "travel_exceeded_1_55mm"
    assert result["ticks_completed"] < 300


def test_pickup_relative_position_clipping_matches_rust_boundary():
    goal = TaskGoal("far", (50., -30., 100.), (1., 0., 0., 0.), (-50., 100., -100.))
    extension = goal.actor_extension(np.zeros(3), np.array([1., 0., 0., 0.]), np.eye(3), phase=0)
    np.testing.assert_array_equal(extension[:3], [20., -20., 20.])
    np.testing.assert_array_equal(extension[7:10], [-20., 20., -20.])


def native_jaw_fixture(tmp_path):
    # A portable pin/stops conformance fixture. The delivered-body diagnostic
    # is separate, hash-bound evidence; these masses are explicitly synthetic.
    xml = '''<mujoco><compiler angle="radian"/><option timestep=".02" gravity="0 0 0" integrator="implicit"/>
    <worldbody><body name="head_roll">
      <body name="beak_hinge" pos=".031 0 -.025">
        <inertial mass=".124" pos=".02 0 -.01" diaginertia=".0001 .0001 .0001"/>
        <joint name="beak_hinge" axis="0 1 0" range="0 .55" damping=".01"/>
        <site name="jaw_output_pin" pos="-.006 0 -.010392304845413264"/>
        <site name="jaw_grip_load_point" pos=".06 0 -.023"/>
      </body>
      <body name="beak_input_rotor">
        <inertial mass=".017" pos="0 0 0" diaginertia=".00001 .00001 .00001"/>
        <joint name="beak_input_rotor" axis="0 1 0" range="0 .55" armature=".024" damping=".001" frictionloss=".08"/>
        <body name="beak_coupler_link" pos="-.006 0 -.010392304845413264">
          <inertial mass=".005" pos=".015 0 -.012" diaginertia=".000001 .000001 .000001"/>
          <joint name="beak_coupler_link" axis="0 1 0" range="-.55 0" damping=".001"/>
          <site name="coupler_output_pin" pos=".031 0 -.025"/>
        </body>
      </body>
    </body></worldbody><actuator><motor joint="beak_input_rotor"/></actuator>
    <equality><joint joint1="beak_input_rotor" joint2="beak_hinge" solref=".002 1" solimp=".999 .9999 .0001"/></equality>
    </mujoco>'''
    source = tmp_path / "jaw.xml"
    source.write_text(xml)
    from bevy_microduck_tools.goose.mouth_constraints import derive_fixture
    return derive_fixture(source, tmp_path/"predictive_jaw.xml", "native_pin_implicit_metric_predictive_stop")


@pytest.mark.parametrize("torque", [4.4, -4.4])
def test_native_pin_and_predictive_stop_preserve_full_range(tmp_path, torque):
    from bevy_microduck_tools.goose.mouth_constraints import one_case
    model = native_jaw_fixture(tmp_path)
    result = one_case(model, .275, torque_nm=torque, predictive=True)
    assert result["local_checks_passed"]
    assert result["integrations"] == 100
    expected = .55 if torque > 0 else 0.
    assert result["rows"][-1]["angles_rad"]["beak_hinge"] == pytest.approx(expected, abs=1e-5)
    assert result["rows"][-1]["time_s"] == pytest.approx(2.)
    assert model.neq == 1  # Actual output pin; no hidden joint mimics.


def test_native_pin_transmits_twenty_newton_resistance(tmp_path):
    from bevy_microduck_tools.goose.mouth_constraints import one_case
    model = native_jaw_fixture(tmp_path)
    result = one_case(model, .275, 20., predictive=True)
    assert result["local_checks_passed"]
    assert max(abs(t) for row in result["rows"] for t in row["qfrc_constraint_nm"]) > .1
    assert result["max_output_pin_distance_m"] < .0001


def test_implicit_damping_metric_matches_physical_backward_euler():
    import mujoco
    from bevy_microduck_tools.goose.mouth_constraints import apply_implicit_damping_metric
    root = ET.fromstring('''<mujoco><option timestep=".02" gravity="0 0 0"/>
      <worldbody><body name="slider"><joint type="slide" axis="0 0 1" armature=".03" damping="2"/>
        <inertial mass=".1" pos="0 0 0" diaginertia=".01 .01 .01"/>
      </body></worldbody></mujoco>''')
    apply_implicit_damping_metric(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qvel[0], data.qfrc_applied[0] = .8, 2.
    mujoco.mj_step(model, data)
    physical_inertia = .1+.03
    expected = (physical_inertia*.8+DT*2.)/(physical_inertia+DT*2.)
    assert data.qvel[0] == pytest.approx(expected, abs=1e-12)
    assert data.qpos[0] == pytest.approx(DT*expected, abs=1e-12)
    assert model.body_mass.sum() == pytest.approx(.1)


def test_native_discrete_contract_cannot_silently_enter_default_runtime(tmp_path):
    _, path = fixture_runtime(tmp_path)
    contract = json.loads(path.read_text())
    contract["native_discrete"] = {"engine_version": "3.13.0"}
    write_json(path, contract)
    with pytest.raises(ValueError, match="explicit versioned runtime"):
        GooseSourceRuntime(tmp_path/"robot.xml", path)


def test_native_discrete_engine_guard_precedes_model_loading(tmp_path):
    import mujoco
    from bevy_microduck_tools.goose.native_discrete_runtime import ENGINE_VERSION, GooseDiscreteSourceRuntime
    if mujoco.__version__ == ENGINE_VERSION:
        pytest.skip("This check requires the preserved earlier engine")
    with pytest.raises(ValueError, match="isolated MuJoCo 3.13.0"):
        GooseDiscreteSourceRuntime(tmp_path/"absent.xml", tmp_path/"absent.json")


def test_native_discrete_identity_and_compiled_integrator_must_agree(tmp_path):
    import mujoco
    from bevy_microduck_tools.goose.native_discrete_runtime import (
        CANDIDATE, ENGINE_VERSION, REVISION, GooseDiscreteSourceRuntime)
    if mujoco.__version__ != ENGINE_VERSION:
        pytest.skip("Native discrete conformance uses the isolated 3.13.0 engine")
    _, path = fixture_runtime(tmp_path)
    contract = json.loads(path.read_text())
    with pytest.raises(ValueError, match="frozen native discrete identity"):
        GooseDiscreteSourceRuntime(tmp_path/"robot.xml", path)
    contract.update(candidate=CANDIDATE, integrator="discrete", native_discrete={
        "revision": REVISION, "engine_version": ENGINE_VERSION,
        "physical_parameters_changed": False})
    write_json(path, contract)
    with pytest.raises(ValueError, match="Compiled model"):
        GooseDiscreteSourceRuntime(tmp_path/"robot.xml", path)


def test_native_discrete_real_tick_matches_native_map_without_extra_solve(tmp_path, monkeypatch):
    import mujoco
    from bevy_microduck_tools.goose.native_discrete_runtime import ENGINE_VERSION
    if mujoco.__version__ != ENGINE_VERSION:
        pytest.skip("Native discrete conformance uses the isolated 3.13.0 engine")
    controller, _ = fixture_runtime(tmp_path, native_discrete=True)
    initial_qpos = controller.data.qpos.copy()
    initial_qvel = controller.data.qvel.copy()
    calls = {"step": 0, "forward": 0}
    native_step, native_forward = mujoco.mj_step, mujoco.mj_forward

    def counted_step(model, data):
        calls["step"] += 1
        native_step(model, data)

    def counted_forward(model, data):
        calls["forward"] += 1
        native_forward(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    monkeypatch.setattr(mujoco, "mj_forward", counted_forward)
    obs, state = controller.step(np.ones(18))
    assert calls == {"step": 1, "forward": 0}
    assert state["controller_updates"] == state["physics_integrations"] == 1
    assert controller.data.time == pytest.approx(.02)
    assert obs.shape == (65,)
    np.testing.assert_allclose(controller.target, .02)
    np.testing.assert_allclose(controller.data.qacc,
                               (controller.data.qvel-initial_qvel)/DT, atol=1e-12)
    reference = mujoco.MjData(controller.model)
    reference.qpos[:] = initial_qpos
    reference.qvel[:] = initial_qvel
    reference.ctrl[:] = controller.last_tau
    native_step(controller.model, reference)
    np.testing.assert_array_equal(controller.data.qpos, reference.qpos)
    np.testing.assert_array_equal(controller.data.qvel, reference.qvel)
    assert not any(w.number for w in controller.data.warning)


def test_native_discrete_rejects_python_outside_project_contract(tmp_path, monkeypatch):
    import mujoco
    from bevy_microduck_tools.goose import native_discrete_runtime as native
    if mujoco.__version__ != native.ENGINE_VERSION:
        pytest.skip("Python guard conformance uses the isolated 3.13.0 engine")
    monkeypatch.setattr(native.sys, "version_info", (3, 13, 15))
    with pytest.raises(ValueError, match="project Python 3.12"):
        native.GooseDiscreteSourceRuntime(tmp_path/"absent.xml", tmp_path/"absent.json")
