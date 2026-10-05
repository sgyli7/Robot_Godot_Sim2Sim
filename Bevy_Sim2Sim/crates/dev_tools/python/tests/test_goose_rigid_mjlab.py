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
