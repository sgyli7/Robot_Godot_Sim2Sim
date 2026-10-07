"""Reference reset must never erase corrupt input or a pending physical Tick."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv
from bevy_microduck_tools.goose.moving_initialization import GooseMovingStateEnv


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
