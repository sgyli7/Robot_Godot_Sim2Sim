"""Reference reset must never erase corrupt input or a pending physical Tick."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv
from bevy_microduck_tools.goose.moving_initialization import GooseMovingStateEnv
from bevy_microduck_tools.goose import moving_initialization as moving


def reset_shell(tmp_path, monkeypatch, *, before=None, pending=None):
    path = tmp_path / "bank.pt"
    path.write_bytes(b"owned reference bytes")
    env = object.__new__(GooseMovingStateEnv)
    env._moving_path = path
    env._moving_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    term = SimpleNamespace(_before_time=before,
        drive=SimpleNamespace(_pending_action=pending))
    env.action_manager = SimpleNamespace(get_term=lambda _: term)
    calls = []
    monkeypatch.setattr(GooseDevelopmentEnv, "reset", lambda *a, **kw: calls.append(kw))
    return env, path, calls


def test_changed_reference_bytes_rejected_before_native_reset(tmp_path, monkeypatch):
    env, path, calls = reset_shell(tmp_path, monkeypatch)
    path.write_bytes(b"changed reference bytes")
    with pytest.raises(ValueError, match="changed before reset"):
        env.reset()
    assert calls == []


@pytest.mark.parametrize("before,pending", [
    (torch.tensor([4.]), None), (None, torch.zeros(1,18)),
])
def test_pending_tick_is_preserved_before_native_reset(tmp_path, monkeypatch, before, pending):
    env, _, calls = reset_shell(tmp_path, monkeypatch, before=before, pending=pending)
    with pytest.raises(RuntimeError, match="Preserve pending physical Tick"):
        env.reset()
    assert calls == []


@pytest.mark.parametrize("fraction", [-.01, 1.01, float("nan"), float("inf"), True])
def test_bad_fraction_is_rejected_before_reading_bank(tmp_path, fraction):
    with pytest.raises(ValueError, match="Reference reset fraction"):
        GooseMovingStateEnv(None, "cpu", moving_bank_path=tmp_path/"missing.pt",
            moving_bank_sha256="missing", reference_fraction=fraction)


def test_mixed_partial_reset_preserves_cold_and_continuing_worlds(tmp_path, monkeypatch):
    env, _, _ = reset_shell(tmp_path, monkeypatch)
    env._reference_fraction, env._reset_world_cursor, env._moving_cursor = .5, 0, 0
    env.cold_reset_worlds = env.reference_reset_worlds = 0
    term = env.action_manager.get_term("goose")
    term.root_q = 0
    qpos = torch.full((3, 4), 100.)
    warm = torch.full((3, 2), 123.)
    env.scene = SimpleNamespace(num_envs=3, env_origins=torch.tensor([[10.,0,0],[20.,0,0],[30.,0,0]]),
        sensors={})
    env.episode_length_buf = torch.tensor([900, 901, 902])
    state = dict(buffers={"data.qpos": torch.tensor([1.,2,3,1]),
        "data.qacc_warmstart": torch.tensor([4.,5])}, episode_age=70)
    env._moving_states = [state]
    monkeypatch.setattr(moving, "_validate", lambda *args: None)
    monkeypatch.setattr(moving, "_world_buffers", lambda _: {
        "data.qpos": qpos, "data.qacc_warmstart": warm})
    command = SimpleNamespace(course_stage=8, _last_stage=8)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    env.common_step_counter = 999
    def cold_reset(_, **kw):
        ids = kw["env_ids"]
        qpos[ids] = 0
        qpos[ids,:3] = env.scene.env_origins[ids]
        qpos[ids,3] = 1
        warm[ids] = 0
        env.episode_length_buf[ids] = 0
        return "cold-observation", "cold-extras"
    monkeypatch.setattr(GooseDevelopmentEnv, "reset", cold_reset)
    env.sim = SimpleNamespace(device="cpu", forward=lambda: None, sense=lambda: None)
    env.extras = {}
    env.observation_manager = SimpleNamespace(compute=lambda **kw: qpos.clone())
    # Nonconsecutive IDs: first keeps cold birth, second restores reference.
    obs, _ = env.reset(env_ids=torch.tensor([2, 0]))
    assert torch.equal(obs[2], torch.tensor([30.,0,0,1]))
    assert torch.equal(obs[0], torch.tensor([11.,2,3,1]))
    assert torch.equal(obs[1], torch.full((4,), 100.))
    assert torch.equal(warm, torch.tensor([[4.,5],[123.,123],[0.,0]]))
    assert env.episode_length_buf.tolist() == [70, 901, 0]
    assert env.common_step_counter == 999 and command._last_stage == 8
    assert (env.cold_reset_worlds, env.reference_reset_worlds) == (1, 1)


def test_zero_fraction_returns_untouched_native_cold_reset(tmp_path, monkeypatch):
    env, _, calls = reset_shell(tmp_path, monkeypatch)
    env.scene, env.sim = SimpleNamespace(num_envs=3), SimpleNamespace(device="cpu")
    env._reference_fraction, env._reset_world_cursor, env._moving_cursor = 0., 0, 0
    env.cold_reset_worlds = env.reference_reset_worlds = 0
    env._moving_states = [object()]
    monkeypatch.setattr(moving, "_validate", lambda *args: pytest.fail("No reference was selected"))
    ids = torch.tensor([2, 0])
    assert env.reset(env_ids=ids) is None
    assert calls == [{"env_ids": ids}]
    assert env._moving_cursor == 0
    assert (env.cold_reset_worlds, env.reference_reset_worlds) == (2, 0)
