"""Actual native sensor instrumentation must preserve the source plant."""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from bevy_microduck_tools.goose.mjlab_baseline import build_task_proxy_reference
from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv
from bevy_microduck_tools.goose.foot_curriculum import (
    CONTACT_SENSOR, HEIGHT_SENSOR, SITE_NAMES, add_foot_sensors, make_foot_course_cfg)
from bevy_microduck_tools.goose.source_training import make_sole_bounded_forward_cfg


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not package:
        pytest.skip("Frozen external 004 package required")
    package = Path(package)
    return build_task_proxy_reference(
        package/"robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml",
        package/"robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        package, tmp_path/"candidate")


def test_sensors_leave_real_physics_torque_contacts_and_actor_unchanged(candidate):
    environments = []
    try:
        for factory in (make_sole_bounded_forward_cfg, make_foot_course_cfg):
            cfg = factory(*candidate, num_envs=2, seed=43)
            cfg.commands["velocity"].rel_standing_envs = 1.
            env = GooseDevelopmentEnv(cfg, "cpu")
            env.reset()
            environments.append(env)
        before, after = environments
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                "jnt_axis", "jnt_range", "geom_pos", "geom_quat", "geom_type",
                "geom_contype", "geom_conaffinity", "geom_margin", "geom_friction",
                "geom_solref", "geom_solimp", "dof_damping", "dof_armature",
                "dof_frictionloss", "eq_data", "eq_solref", "eq_solimp",
                "actuator_gainprm", "actuator_trnid", "mesh_vert", "mesh_face",
                "exclude_signature"):
            np.testing.assert_array_equal(getattr(before.sim.mj_model, field),
                                          getattr(after.sim.mj_model, field))
        assert before.sim.mj_model.ngeom == after.sim.mj_model.ngeom == 12
        assert before.sim.mj_model.nbody == after.sim.mj_model.nbody == 22
        assert after.sim.mj_model.nsite == before.sim.mj_model.nsite+2
        assert after.sim.mj_model.nu == 18 and after.sim.mj_model.opt.timestep == .02
        for tick in range(20):
            action = torch.zeros((2, 18))
            action[1, 0] = .005*np.sin(tick*.02)
            old_obs, *_ = before.step(action)
            new_obs, *_ = after.step(action)
            for field in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "time"):
                torch.testing.assert_close(getattr(before.sim.data, field),
                    getattr(after.sim.data, field), atol=0, rtol=0)
            old = before.action_manager.get_term("goose")
            new = after.action_manager.get_term("goose")
            torch.testing.assert_close(old.drive.last_tau, new.drive.last_tau, atol=0, rtol=0)
            torch.testing.assert_close(old_obs["actor"], new_obs["actor"], atol=0, rtol=0)
            assert new_obs["actor"].shape == (2, 65)
            count = int(before.sim.wp_data.nacon.numpy()[0])
            assert count == int(after.sim.wp_data.nacon.numpy()[0])
            for field in ("geom", "pos", "dist", "efc_address", "worldid"):
                np.testing.assert_array_equal(
                    getattr(before.sim.wp_data.contact, field).numpy()[:count],
                    getattr(after.sim.wp_data.contact, field).numpy()[:count])
            for world, size in enumerate(before.sim.data.nefc.numpy()):
                np.testing.assert_array_equal(before.sim.wp_data.efc.force.numpy()[world, :size],
                    after.sim.wp_data.efc.force.numpy()[world, :size])
        assert after.action_manager.get_term("goose").drive.completed_ticks.tolist() == [20,20]
        for name in ("feet_air_time", "feet_clearance"):
            index = after.reward_manager.active_terms.index(name)
            assert not after.reward_manager._step_reward[:, index].any()
        with pytest.raises(ValueError, match="already attached"):
            add_foot_sensors(after.cfg, candidate[1])
    finally:
        for env in environments:
            env.close()


def test_height_and_contact_identify_the_recorded_actual_lift(candidate):
    trace_path = os.environ.get("GOOSE_SUPPORT_LIFT_TRACE")
    if not trace_path:
        pytest.skip("Actual legal-action support/lift trajectory required")
    trace = np.load(trace_path)
    # Sensor-only replay of recorded actual physics, not training assistance.
    env = GooseDevelopmentEnv(make_foot_course_cfg(*candidate, num_envs=1), "cpu")
    try:
        env.reset()
        model = env.sim.mj_model
        host = mujoco.MjData(model)
        for index in (0, 99, 499, 510, 513):
            env.sim.data.qpos[:] = torch.tensor(trace["qpos"][index], dtype=torch.float32)
            env.sim.data.qvel[:] = torch.tensor(trace["qvel"][index], dtype=torch.float32)
            env.sim.forward()
            env.scene.update(dt=0.)
            env.sim.sense()
            host.qpos[:] = env.sim.data.qpos[0].numpy()
            mujoco.mj_kinematics(model, host)
            site_ids = [model.site("robot/"+name).id for name in SITE_NAMES]
            expected = host.site_xpos[site_ids, 2].clip(min=0.)
            height = env.scene[HEIGHT_SENSOR].data.heights[0].numpy()
            np.testing.assert_allclose(height, expected, atol=1e-6, rtol=0)
        contact = env.scene[CONTACT_SENSOR].data
        assert height[0] > .01 and height[1] < .0015
        assert contact.found[0, 0] == 0 and contact.found[0, 1] > 0
        assert abs(float(contact.force[0, 1, 2])) > 1.
        assert env._sim_step_counter == 0 and env.sim.data.time[0] == 0.
    finally:
        env.close()
