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


def fixture_runtime(tmp_path, skill="locomotion"):
    # Real MuJoCo integration in a deliberately small controller conformance rig.
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 -9.81", integrator="implicit")
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
    write_json(path, contract)
    return GooseSourceRuntime(model, path, skill=skill), path


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
