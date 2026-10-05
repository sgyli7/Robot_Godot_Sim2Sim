"""Same-state source regression and actual vector reset/integration boundaries."""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from bevy_microduck_tools.goose.native_cpu_env import NativeGooseRecoveryEnv, NativeGooseRslEnv
from bevy_microduck_tools.goose.speculative_contact import (
    build_reference, build_tilted_sole_reference, build_material_share_reference, make_source_runtime,
    require_pair_margin_backend, validate_prediction,
)


@pytest.fixture
def source(tmp_path, monkeypatch):
    package = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not package:
        pytest.skip("Verified external004 package required")
    monkeypatch.chdir(tmp_path)
    package = Path(package).resolve(strict=True)
    model, contract = build_reference(
        package / "robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml",
        package / "robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        package, tmp_path / "source")
    rt = make_source_runtime(model, contract)
    return model, contract, rt.data.qpos.copy()


def test_actual_native_state_torque_observation_matches_source(source):
    env = NativeGooseRecoveryEnv(*source)
    env.reset()
    ref = make_source_runtime(source[0], source[1])
    for tick in range(20):
        action = np.zeros(18)
        action[5] = .1*np.sin(tick*.2)
        expected, _ = ref.step(action)
        observation, reward, done, timeout, info = env.step(action)
        np.testing.assert_array_equal(env.runtime.data.qpos, ref.data.qpos)
        np.testing.assert_array_equal(env.runtime.data.qvel, ref.data.qvel)
        np.testing.assert_array_equal(env.runtime.last_tau, ref.last_tau)
        np.testing.assert_array_equal(observation, expected.astype(np.float32))
        assert info["actual_integrations"] == 1 and env.total_integrations == tick+1
        assert not done and not timeout and np.isfinite(reward)
    env.close()


def test_gym_same_step_reset_does_not_skip_next_physics_tick(source):
    import torch
    env = gym.vector.SyncVectorEnv([
        lambda: NativeGooseRecoveryEnv(*source, episode_ticks=2) for _ in range(2)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    wrapper = NativeGooseRslEnv(env)
    try:
        for tick in range(5):
            obs, reward, done, info = wrapper.step(torch.zeros((2, 18)))
            assert obs["actor"].shape == (2, 65)
            assert done.tolist() == [tick % 2 == 1]*2
            assert info["time_outs"].tolist() == done.tolist()
            assert wrapper.real_integrations == (tick+1)*2
        assert env.get_attr("total_integrations") == (5, 5)
        assert env.get_attr("episodes") == (3, 3)
    finally:
        wrapper.close()


@pytest.mark.parametrize("material_share", (False, True))
def test_tilted_discovery_preserves_upright_source_and_rejects_gpu(source, tmp_path, material_share):
    import json
    model, contract = build_tilted_sole_reference(source[0], source[1], tmp_path / "tilted")
    if material_share:
        model, contract = build_material_share_reference(model, contract, tmp_path / "material_share")
    env = NativeGooseRecoveryEnv(model, contract, source[2])
    env.reset()
    old = make_source_runtime(source[0], source[1])
    for tick in range(4):
        expected, _ = old.step(np.zeros(18))
        actual, _, _, _, info = env.step(np.zeros(18))
        np.testing.assert_array_equal(env.runtime.data.qpos, old.data.qpos)
        np.testing.assert_array_equal(env.runtime.data.qvel, old.data.qvel)
        np.testing.assert_array_equal(env.runtime.last_tau, old.last_tau)
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        assert info["actual_integrations"] == 1
    env.close()
    frozen = json.loads(contract.read_text())
    native = mujoco.MjModel.from_xml_path(str(model))
    assert native.ngeom == 12 and native.npair == 11
    with pytest.raises(ValueError, match="native CPU candidate only"):
        require_pair_margin_backend(frozen, None)
    gid = native.geom("right_flexible_sole").id
    pair = next(i for i in range(native.npair) if gid in (native.pair_geom1[i], native.pair_geom2[i]))
    native.pair_solimp[pair, 0] += .01
    with pytest.raises(ValueError, match="priority1 sole material"):
        validate_prediction(native, frozen)


@pytest.fixture
def rigid_source(tmp_path, monkeypatch):
    from bevy_microduck_tools.goose.rigid_native import build_reference, make_source_runtime
    package = os.environ.get("GOOSE_FROZEN_TASK_PROXY_PACKAGE")
    if not package:
        pytest.skip("Verified external004 package required")
    monkeypatch.chdir(tmp_path)
    package = Path(package).resolve(strict=True)
    model, contract = build_reference(
        package / "robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml",
        package / "robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json",
        package, tmp_path / "rigid")
    return model, contract, make_source_runtime(model, contract)


def test_rigid_native_preserves_si_and_actual_beak_rotor(rigid_source):
    import json
    _, path, rt = rigid_source
    contract = json.loads(path.read_text())
    parent = mujoco.MjModel.from_xml_path(contract["rigid_native"]["parent_model"])
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                 "jnt_pos", "jnt_axis", "jnt_range", "dof_armature", "dof_damping",
                 "mesh_vert", "geom_bodyid", "exclude_signature", "eq_data"):
        np.testing.assert_array_equal(getattr(rt.model, name), getattr(parent, name))
    assert rt.model.nbody == 22 and rt.model.ngeom == 12 and rt.model.nu == 18
    assert rt.model.npair == 0 and rt.model.neq == 1
    assert rt.model.joint(int(rt.model.actuator_trnid[5, 0])).name == "beak_input_rotor"
    assert rt.qidx[5] == rt.model.joint("beak_hinge").qposadr[0]
    for index in range(18):
        if index != 5:
            assert rt.model.actuator_biasprm[index, 2] == -rt.kd[index]
    assert contract["contact_mapping"] == {
        "method": "native_rigid_contact", "custom_callbacks": False}


def test_rigid_actual_one_tick_force_limits_and_observations(rigid_source):
    _, _, rt = rigid_source
    for tick in range(30):
        action = np.zeros(18)
        action[5] = .15
        before, target = rt.data.time, rt.target.copy()
        observations, info = rt.step(action)
        assert observations.shape == (65,) and np.isfinite(observations).all()
        assert rt.data.time == pytest.approx(before+.02)
        assert rt.physics_integrations == tick+1 == rt.controller_updates
        assert np.all(np.abs(rt.last_tau) <= rt.peak+1e-9)
        assert np.all(np.abs(rt.target-target) <= rt.speed*.02+1e-12)
        assert not info["auto_reset"]


def test_rigid_contact_violation_is_retained_failed_episode(rigid_source, tmp_path):
    from bevy_microduck_tools.goose.native_cpu_env import NativeGooseStandingEnv
    model, contract, rt = rigid_source
    env = NativeGooseStandingEnv(model, contract, rt.data.qpos.copy(),
        trajectory_directory=tmp_path / "episodes")
    env.reset()
    action = np.zeros(18)
    action[10] = .5
    action[16] = -.5
    _, reward, done, timeout, info = env.step(action)
    assert info["actual_integrations"] == 1 and env.total_integrations == 1
    assert info["contact_limit_failure"] and done and not timeout and reward < 0
    files = list((tmp_path / "episodes").glob("*_failed.npz"))
    assert len(files) == 1
    with np.load(files[0]) as saved:
        assert saved["time"].tolist() == [.02]
        assert saved["terminated"].tolist() == [True]
        np.testing.assert_array_equal(saved["action"][0], action)
    env.reset()
    assert env.runtime.data.time == 0 and env.total_integrations == 1
    env.close()


def test_rigid_recovery_still_aborts_on_unadmitted_contact(rigid_source):
    model, contract, rt = rigid_source
    env = NativeGooseRecoveryEnv(model, contract, rt.data.qpos.copy())
    env.reset()
    action = np.zeros(18)
    action[10] = .5
    action[16] = -.5
    with pytest.raises(RuntimeError, match="physical guard"):
        env.step(action)
    assert env.total_integrations == 1 and env.runtime.data.time == .02
    env.close()
