"""Actual snapshot/reset/inference-mode seams for the bent stance course."""
import json
import os
from pathlib import Path

import pytest
import torch

from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.stance_curriculum import (
    GooseBentStanceEnv, initialize_stance_actor, make_bent_standing_cfg, read_stance)
from bevy_microduck_tools.goose.source_training import GooseRslEnv, make_runner_cfg
from mjlab.rl.runner import MjlabOnPolicyRunner


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = os.environ.get("GOOSE_BENT_STANCE_PROFILE")
    if not path:
        pytest.skip("Actual frozen bent-stance profile required")
    root = Path(path)
    return root/"candidate/robot.xml", root/"candidate/contract.json", root/"stance.json"


def test_reset_after_inference_retains_selected_snapshot_and_other_world(profile):
    model, contract, stance = profile
    state = json.loads(stance.read_text())
    env = GooseBentStanceEnv(make_bent_standing_cfg(*profile, sha256(stance), num_envs=2),
        "cpu", stance_path=stance, stance_sha256=sha256(stance))
    try:
        obs, _ = env.reset()
        action = env.action_manager.get_term("goose")
        raw_buffer, effort_buffer = action._raw, action._effort
        mean = torch.tensor(state["drive"]["actions"]).expand(2, -1)
        with torch.inference_mode():
            for _ in range(5):
                env.step(mean.clone())
        assert action._raw is raw_buffer and action._effort is effort_buffer
        assert not action._raw.is_inference() and not action._effort.is_inference()
        saved = {n: getattr(env.sim.data, n)[1].clone() for n in ("qpos", "qvel", "ctrl", "time")}
        obs, _ = env.reset(env_ids=torch.tensor([0]))
        for name, value in saved.items():
            torch.testing.assert_close(getattr(env.sim.data, name)[1], value, atol=0, rtol=0)
        assert action.drive.completed_ticks.tolist() == [0, 5]
        torch.testing.assert_close(action.drive.target[0],
            action.drive.target.new_tensor(state["drive"]["target"]), atol=0, rtol=0)
        torch.testing.assert_close(obs["actor"][0, 45:63], mean[0], atol=0, rtol=0)
        assert env._sim_step_counter == 5
        env.step(mean)
        assert action.drive.completed_ticks.tolist() == [1, 6]
        assert env._sim_step_counter == 6
    finally:
        env.close()


def test_fresh_mean_initialization_uses_original_action_interface(profile):
    _, _, stance = profile
    state = json.loads(stance.read_text())
    env = GooseBentStanceEnv(make_bent_standing_cfg(*profile, sha256(stance), num_envs=2),
        "cpu", stance_path=stance, stance_sha256=sha256(stance))
    try:
        env.reset()
        wrapper = GooseRslEnv(env)
        runner = MjlabOnPolicyRunner(wrapper, make_runner_cfg(), None, "cpu")
        std = runner.alg.actor.distribution.log_std_param.detach().clone()
        initialize_stance_actor(runner, state["drive"]["actions"])
        torch.testing.assert_close(runner.alg.actor.distribution.log_std_param, std, atol=0, rtol=0)
        obs = wrapper.get_observations()
        expected = torch.tensor(state["drive"]["actions"]).expand(2, -1)
        for delta in (0., .1, -.1):
            obs["actor"][:, 9:27] += delta
            torch.testing.assert_close(runner.alg.actor(obs), expected, atol=1e-7, rtol=0)
        assert runner.alg.actor.obs_dim == 65 and wrapper.real_integrations == 0
        with pytest.raises(ValueError):
            initialize_stance_actor(runner, [float("nan")]*18)
    finally:
        env.close()


def test_stance_identity_and_illegal_history_fail_before_physics(profile, tmp_path):
    model, contract, source = profile
    state = json.loads(source.read_text())
    state["drive"]["target"][9] = 100.
    bad = tmp_path/"bad_stance.json"
    bad.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="drive limits"):
        read_stance(bad, sha256(bad), model, contract)
    with pytest.raises(ValueError, match="identity"):
        read_stance(source, "0"*64, model, contract)
