"""Same-state source regression and actual vector reset/integration boundaries."""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from bevy_microduck_tools.goose.native_cpu_env import NativeGooseRecoveryEnv, NativeGooseRslEnv
from bevy_microduck_tools.goose.speculative_contact import build_reference, make_source_runtime


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
