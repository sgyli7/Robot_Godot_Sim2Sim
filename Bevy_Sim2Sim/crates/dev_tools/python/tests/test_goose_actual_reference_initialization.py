"""Reject unsafe reference resets before the native environment can mutate."""
from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.actual_reference_initialization import ActualReferenceEnv
from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv


def guarded_env(tmp_path, monkeypatch):
    path = tmp_path / "reference.pt"
    path.write_bytes(b"frozen actual reference")
    action = SimpleNamespace(_before_time=None,
        drive=SimpleNamespace(_pending_action=None))
    env = object.__new__(ActualReferenceEnv)
    env._bank_path = path
    env._bank_digest = sha256(path)
    env._bank = {"states": [{}, {}, {}]}
    env._actual_reset_active = False
    env._forced_birth_frames = None
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device="cpu")
    env.action_manager = SimpleNamespace(get_term=lambda name: action)

    def forbid_native_reset(*args, **kwargs):
        pytest.fail("Unsafe reference reached mutable native reset")

    monkeypatch.setattr(GooseDevelopmentEnv, "reset", forbid_native_reset)
    return env, action


def test_pending_actual_tick_cannot_be_reset(tmp_path, monkeypatch):
    env, action = guarded_env(tmp_path, monkeypatch)
    action._before_time = torch.tensor([1., 1.])
    with pytest.raises(RuntimeError, match="incomplete native integral"):
        env.reset(env_ids=torch.tensor([0]))


def test_changed_bank_rejected_before_native_reset(tmp_path, monkeypatch):
    env, _ = guarded_env(tmp_path, monkeypatch)
    env._bank_path.write_bytes(b"changed actual reference")
    with pytest.raises(ValueError, match="changed before reset"):
        env.reset(env_ids=torch.tensor([0]))


@pytest.mark.parametrize("ids", ([0, 0], [-1], [2], []))
def test_invalid_world_selection_cannot_mutate(tmp_path, monkeypatch, ids):
    env, _ = guarded_env(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="distinct valid world IDs"):
        env.reset(env_ids=torch.tensor(ids, dtype=torch.long))


@pytest.mark.parametrize("frames", ([-1], [2], [0, 1]))
def test_unadmitted_reference_frame_cannot_mutate(tmp_path, monkeypatch, frames):
    env, _ = guarded_env(tmp_path, monkeypatch)
    env._forced_birth_frames = torch.tensor(frames)
    with pytest.raises(ValueError, match="outside the admitted finite clip"):
        env.reset(env_ids=torch.tensor([0]))
