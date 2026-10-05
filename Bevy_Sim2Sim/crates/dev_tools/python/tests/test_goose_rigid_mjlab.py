"""Native C controller parity and upstream mixed XML actuator compilation."""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.rigid_native import build_reference, make_source_runtime
from bevy_microduck_tools.goose.batch_drive import BatchedRigidNativeDrive
from bevy_microduck_tools.goose.mjlab_baseline import make_entity_cfg
from bevy_microduck_tools.goose.mjlab_env import RigidGooseActionCfg, make_development_env_cfg


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not root:
        pytest.skip("Verified original004 package required")
    monkeypatch.chdir(tmp_path)
    root = Path(root)
    model, contract = build_reference(
        root / "robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml",
        root / "robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        root, tmp_path / "rigid", upstream=True)
    return model, contract, make_source_runtime(model, contract)


@pytest.mark.parametrize("limited", (False, True))
def test_same_state_native_controls_caps_force_history_observations(source, limited):
    _, _, rt = source
    drive = BatchedRigidNativeDrive(rt.contract, 1)
    if limited:
        rt.strength, rt.delay = .7, 1
        rt.thermal[:] = (rt.cont*.7)**2+1.
        drive.strength[:] = .7
        drive.delay[:] = True
        drive.thermal[:] = torch.from_numpy(rt.thermal)
    for tick in range(25):
        action = .025*np.sin(np.arange(18)*.3+tick*.1)
        action[5] = .1
        control = drive.prepare(action[None], rt.data.qpos[rt.qidx][None],
            rt.data.qvel[rt.vidx][None], rt.data.qpos[3:7][None],
            motor_velocity=rt.data.qvel[rt.motor_vids][None])
        rt.step(action)
        np.testing.assert_allclose(control[0], rt.data.ctrl, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(torch.stack((-drive.cap, drive.cap), -1)[0],
            rt.model.actuator_forcerange, atol=1e-12, rtol=1e-12)
        drive.commit(rt.last_tau[None])
        np.testing.assert_allclose(drive.thermal[0], rt.thermal, atol=1e-12)
        np.testing.assert_array_equal(drive.last_tau[0], rt.last_tau)
        obs = drive.observations(rt.data.qpos[rt.qidx][None], rt.data.qvel[rt.vidx][None],
            rt.data.qvel[3:6][None], rt.data.xmat[rt.torso].reshape(1, 3, 3), rt.commands[None])
        np.testing.assert_array_equal(obs[0], rt.observations().astype(np.float32))
        assert rt.physics_integrations == tick+1 == int(drive.completed_ticks[0])


def test_entity_keeps_native_si_actuator_order_and_real_beak(source):
    model, contract, rt = source
    cfg = make_entity_cfg(model, contract)
    entity = cfg.build()
    wrapped = entity.spec.compile()
    for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat", "jnt_range",
                 "dof_armature", "actuator_trnid", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(wrapped, name), getattr(rt.model, name))
    assert wrapped.ngeom == 11 and wrapped.nu == 18 and wrapped.neq == 1
    assert wrapped.joint(int(wrapped.actuator_trnid[5, 0])).name == "beak_input_rotor"
    assert [len(a.target_names_expr) for a in cfg.articulation.actuators] == [17, 1]
    assert [a.command_field for a in cfg.articulation.actuators] == ["position", "effort"]
    assert [wrapped.actuator(i).name for i in range(18)] == [rt.model.actuator(i).name for i in range(18)]
    assert not wrapped.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    profile = make_development_env_cfg(model, contract)
    assert isinstance(profile.actions["goose"], RigidGooseActionCfg)
    assert profile.decimation == 1 and profile.sim.mujoco.timestep == .02
    assert profile.sim.mujoco.integrator == "implicitfast"


@pytest.mark.parametrize("direction", (-1, 1))
def test_overspeed_keeps_bounded_braking_and_cpu_batch_actual_force_parity(
        source, tmp_path, direction):
    import json

    _, parent_contract, original = source
    cfg = json.loads(parent_contract.read_text())["rigid_native"]
    model, contract = build_reference(cfg["parent_model"], cfg["parent_contract"],
        cfg["source_root"], tmp_path / "braking", upstream=True,
        primitive_feet=True, braking=True)
    rt = make_source_runtime(model, contract)
    rt.data.qvel[rt.vidx[1]] = direction*rt.speed[1]*1.5
    drive = BatchedRigidNativeDrive(rt.contract, 1)
    for tick in range(20):
        velocity = rt.data.qvel[rt.motor_vids].copy()
        control = drive.prepare(np.zeros((1, 18)),
            rt.data.qpos[rt.qidx][None], rt.data.qvel[rt.vidx][None],
            rt.data.qpos[3:7][None], motor_velocity=velocity[None])
        if tick == 0:
            assert float(drive.force_lower[0, 1]) == (-rt.peak[1] if direction > 0 else 0.)
            assert float(drive.force_upper[0, 1]) == (0. if direction > 0 else rt.peak[1])
        rt.step(np.zeros(18))
        np.testing.assert_allclose(control[0], rt.data.ctrl, atol=1e-12)
        limits = torch.stack((drive.force_lower, drive.force_upper), -1)
        np.testing.assert_allclose(limits[0], rt.model.actuator_forcerange, atol=1e-12)
        if tick == 0:
            assert rt.last_tau[1]*direction < 0
        drive.commit(rt.last_tau[None])
        np.testing.assert_allclose(drive.thermal[0], rt.thermal, atol=1e-12)
        assert float(np.maximum(rt.last_tau*velocity, 0).sum()) <= (
            rt.contract["positive_mechanical_power_limit_w"]+1e-6)
    for name in ("body_mass", "body_ipos", "body_inertia", "jnt_range",
                 "actuator_gainprm", "actuator_biasprm", "eq_data"):
        np.testing.assert_array_equal(getattr(original.model, name), getattr(rt.model, name))
    assert rt.physics_integrations == 20 == int(drive.completed_ticks[0])


def test_posture_course_reward_change_keeps_real_transitions_and_actor(source):
    from bevy_microduck_tools.goose.foot_curriculum import (
        make_foot_course_cfg, make_rigid_posture_course_cfg)
    from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv

    model, contract, _ = source
    environments = []
    try:
        for factory in (make_foot_course_cfg, make_rigid_posture_course_cfg):
            cfg = factory(model, contract, num_envs=2, seed=67)
            cfg.commands["velocity"].rel_standing_envs = 1.
            env = GooseDevelopmentEnv(cfg, "cpu")
            env.reset()
            environments.append(env)
        before, after = environments
        for tick in range(16):
            action = torch.zeros((2, 18))
            action[1, 0] = .005*np.sin(tick*.02)
            first, _, old_fail, old_timeout, _ = before.step(action)
            second, _, new_fail, new_timeout, _ = after.step(action)
            for name in ("qpos", "qvel", "ctrl", "time"):
                torch.testing.assert_close(getattr(before.sim.data, name),
                    getattr(after.sim.data, name), atol=0, rtol=0)
            for name in ("last_tau", "thermal", "target", "completed_ticks"):
                torch.testing.assert_close(
                    getattr(before.action_manager.get_term("goose").drive, name),
                    getattr(after.action_manager.get_term("goose").drive, name), atol=0, rtol=0)
            assert int(before.sim.wp_data.nacon.numpy()[0]) == int(after.sim.wp_data.nacon.numpy()[0])
            torch.testing.assert_close(first["actor"], second["actor"], atol=0, rtol=0)
            torch.testing.assert_close(old_fail, new_fail, atol=0, rtol=0)
            torch.testing.assert_close(old_timeout, new_timeout, atol=0, rtol=0)
            index = after.reward_manager.active_terms.index("failed_episode")
            torch.testing.assert_close(after.reward_manager._step_reward[:, index],
                -200.*after.termination_manager.terminated.float(), atol=0, rtol=0)
            assert after.reward_manager.get_term_cfg("pose").func.__class__.__name__ == "variable_posture"
        assert after.action_manager.get_term("goose").drive.completed_ticks.tolist() == [16,16]
        assert before.contact_adapter is after.contact_adapter is None
    finally:
        for env in environments:
            env.close()


def test_motion_course_records_depth_and_retains_actual_fall_and_timeout(source):
    from bevy_microduck_tools.goose.foot_curriculum import (
        make_rigid_motion_course_cfg, rigid_collision_depth)

    model, contract, _ = source
    cfg = make_rigid_motion_course_cfg(model, contract, num_envs=2, seed=67)
    assert set(cfg.terminations) == {"fallen", "time_out"}
    assert cfg.terminations["time_out"].time_out
    assert cfg.metrics["rigid_contact_depth_m"].func is rigid_collision_depth
    assert cfg.sim.mujoco.timestep == .02 and cfg.decimation == 1
    assert cfg.auto_reset is False
    assert cfg.rewards["failed_episode"].weight == -200.


def test_native_gym_same_step_reset_critic_and_physics_identity(source):
    import gymnasium as gym
    from bevy_microduck_tools.goose.native_cpu_env import (
        NativeGooseStandingEnv, NativeGooseRslEnv)

    model, contract, direct = source
    initial = direct.data.qpos.copy()
    task = NativeGooseStandingEnv(model, contract, initial, episode_ticks=3,
        contact_limit_m=.05)
    task.reset()
    direct.reset()
    for tick in range(12):
        action = .005*np.sin(np.arange(18)+tick*.1)
        direct.step(action)
        _, _, _, _, info = task.step(action)
        for name in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(task.runtime.data, name),
                getattr(direct.data, name))
        for name in ("target", "thermal", "last_tau", "actions"):
            np.testing.assert_array_equal(getattr(task.runtime, name), getattr(direct, name))
        assert info["actual_integrations"] == 1
        assert task.runtime.physics_integrations == tick+1
    task.close()
    vector = gym.vector.SyncVectorEnv([
        lambda: NativeGooseStandingEnv(model, contract, initial,
            episode_ticks=3, contact_limit_m=.05) for _ in range(2)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    adapter = NativeGooseRslEnv(vector, privileged_critic=True)
    try:
        assert adapter.max_episode_length == 3
        for tick in range(5):
            observations, _, _, _ = adapter.step(torch.zeros((2, 18)))
            assert observations["actor"].shape == (2, 65)
            assert observations["critic"].shape == (2, 69)
            torch.testing.assert_close(observations["critic"][:, :65],
                observations["actor"], atol=0, rtol=0)
        assert adapter.real_integrations == 10
        assert sum(vector.call("total_integrations")) == 10
    finally:
        adapter.close()


def test_reference_reward_preserves_real_physics_and_actor(source, tmp_path):
    import hashlib
    from bevy_microduck_tools.goose.guided_training import (
        make_guided_velocity_cfg, reference_action)
    from bevy_microduck_tools.goose.mature_training import make_mature_velocity_cfg
    from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv

    model, contract, _ = source
    reference = tmp_path / "reference.npz"
    np.savez_compressed(reference, joint_pos=np.zeros((128, 12)))
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    original = make_mature_velocity_cfg(model, contract, num_envs=2, seed=109)
    guided = make_guided_velocity_cfg(model, contract, reference, digest,
        num_envs=2, seed=109)
    original.commands = guided.commands
    environments = []
    try:
        for cfg in (original, guided):
            env = GooseDevelopmentEnv(cfg, "cpu")
            env.reset()
            environments.append(env)
        before, after = environments
        for tick in range(8):
            action = torch.full((2, 18), .005*np.sin(tick*.1))
            first = before.step(action)[0]
            second = after.step(action)[0]
            for name in ("qpos", "qvel", "ctrl", "time"):
                torch.testing.assert_close(getattr(before.sim.data, name),
                    getattr(after.sim.data, name), atol=0, rtol=0)
            for name in ("last_tau", "target", "thermal", "actions", "completed_ticks"):
                torch.testing.assert_close(
                    getattr(before.action_manager.get_term("goose").drive, name),
                    getattr(after.action_manager.get_term("goose").drive, name), atol=0, rtol=0)
            torch.testing.assert_close(first["actor"], second["actor"], atol=0, rtol=0)
            assert first["actor"].shape == (2, 65)
            clock = after.sim.data.time.clone()
            reference_action(after, reference, digest)
            torch.testing.assert_close(after.sim.data.time.clone(), clock, atol=0, rtol=0)
        assert after.action_manager.get_term("goose").drive.completed_ticks.tolist() == [8, 8]
    finally:
        for env in environments:
            env.close()


def test_primitive_feet_fit_original_bottom_and_keep_si_and_real_controller(tmp_path, monkeypatch):
    import json
    from scipy.spatial import ConvexHull
    from bevy_microduck_tools.goose.native_geometry import collision_geom_vertices

    root = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not root:
        pytest.skip("Verified original004 package required")
    monkeypatch.chdir(tmp_path)
    root = Path(root)
    original_path = root / "robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml"
    model, contract = build_reference(original_path,
        root / "robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        root, tmp_path / "primitive", upstream=True, primitive_feet=True)
    rt = make_source_runtime(model, contract)
    parent = mujoco.MjModel.from_xml_path(str(original_path))
    for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat",
                 "jnt_range", "dof_armature", "exclude_signature", "eq_data"):
        np.testing.assert_array_equal(getattr(rt.model, name), getattr(parent, name))
    wrapped = make_entity_cfg(model, contract).build().spec.compile()
    assert wrapped.ngeom == 11 and wrapped.nu == 18 and wrapped.neq == 1
    assert sum(wrapped.geom_type == mujoco.mjtGeom.mjGEOM_BOX) == 2
    for fit in json.loads(contract.read_text())["rigid_native"]["primitive_foot"]["feet"]:
        g = parent.geom(fit["geom"]).id
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, parent.geom_quat[g])
        original = collision_geom_vertices(parent, g)@rotation.reshape(3, 3).T+parent.geom_pos[g]
        primitive = collision_geom_vertices(rt.model, rt.model.geom(fit["geom"]).id)
        primitive += np.asarray(fit["center_body_m"])
        planes = ConvexHull(original).equations
        assert (primitive@planes[:, :3].T+planes[:, 3]).max() <= 1e-8
        np.testing.assert_allclose(primitive[:, 2].min(), original[:, 2].min(), atol=1e-9)
    drive = BatchedRigidNativeDrive(rt.contract, 1)
    for tick in range(20):
        action = .005*np.sin(np.arange(18)*.3+tick*.1)
        control = drive.prepare(action[None], rt.data.qpos[rt.qidx][None],
            rt.data.qvel[rt.vidx][None], rt.data.qpos[3:7][None],
            motor_velocity=rt.data.qvel[rt.motor_vids][None])
        rt.step(action)
        np.testing.assert_allclose(control[0], rt.data.ctrl, atol=1e-12, rtol=1e-12)
        drive.commit(rt.last_tau[None])
        np.testing.assert_array_equal(drive.last_tau[0], rt.last_tau)
    assert rt.physics_integrations == 20 and int(drive.completed_ticks[0]) == 20


def test_depth_check_keeps_reward_snapshot_and_reads_current_geometry(source):
    from bevy_microduck_tools.goose.foot_curriculum import (
        rigid_collision_depth, rigid_contact_limit_exceeded)
    from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv
    from mjlab.managers.termination_manager import TerminationTermCfg

    model, contract, _ = source
    cfg = make_development_env_cfg(model, contract, num_envs=2)
    cfg.terminations["depth"] = TerminationTermCfg(
        func=rigid_contact_limit_exceeded, params={"max_depth_m": .05})
    env = GooseDevelopmentEnv(cfg, "cpu")
    fields = ("qpos", "qvel", "xpos", "xquat", "xipos", "subtree_com", "cvel",
              "geom_xpos", "geom_xmat", "time")
    captured = {}
    original_termination = env.termination_manager.compute
    original_reward = env.reward_manager.compute

    def termination():
        captured.update({name: getattr(env.sim.data, name).clone() for name in fields})
        return original_termination()

    def reward(dt):
        for name in fields:
            torch.testing.assert_close(getattr(env.sim.data, name).clone(), captured[name],
                                       atol=0, rtol=0)
        return original_reward(dt)

    env.termination_manager.compute = termination
    env.reward_manager.compute = reward
    try:
        env.reset()
        for tick in range(8):
            actions = torch.zeros((2, 18))
            actions[0, 6:] = .01 if tick % 2 else -.01
            env.step(actions)
            before = {name: getattr(env.sim.data, name).clone() for name in fields}
            depth = rigid_collision_depth(env)
            for name in fields:
                torch.testing.assert_close(getattr(env.sim.data, name).clone(), before[name],
                                           atol=0, rtol=0)
            host = mujoco.MjData(env.sim.mj_model)
            from bevy_microduck_tools.goose.native_geometry import collision_geom_vertices
            for world in range(2):
                host.qpos[:] = env.sim.data.qpos[world].numpy()
                mujoco.mj_kinematics(env.sim.mj_model, host)
                expected = max(0., max(-float((
                    collision_geom_vertices(env.sim.mj_model, geom)
                    @ host.geom_xmat[geom].reshape(3, 3).T
                    + host.geom_xpos[geom])[:, 2].min())
                    for geom in range(env.sim.mj_model.ngeom)
                    if env.sim.mj_model.geom_type[geom] in (
                        mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX)))
                assert float(depth[world]) == pytest.approx(expected, abs=2e-6)
        assert env._sim_step_counter == 8
        assert env.action_manager.get_term("goose").drive.completed_ticks.tolist() == [8, 8]
    finally:
        env.close()
