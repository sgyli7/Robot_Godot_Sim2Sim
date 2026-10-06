"""Check upstream recurrent rollout/export/reset on the real65-value seam."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from onnx.reference import ReferenceEvaluator
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.models import RNNModel

from bevy_microduck_tools.goose.mature_training import (
    initialize_mature_actor, make_mature_recovery_memory_runner_cfg,
)


def make_actor():
    torch.manual_seed(113)
    cfg = make_mature_recovery_memory_runner_cfg(seed=113)["actor"]
    cfg.pop("class_name")
    cfg.pop("cnn_cfg")
    obs = TensorDict({"actor": torch.randn(2, 65)}, batch_size=[2])
    actor = RNNModel(obs, {"actor": ["actor"]}, "actor", 18, **cfg)
    return actor, obs


def test_cold_initialization_and_world_reset_preserve_other_memory():
    actor, obs = make_actor()
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor,
        optimizer=SimpleNamespace(state={})))
    initialize_mature_actor(runner, recovery=True)
    with torch.no_grad():
        assert torch.equal(actor(obs), torch.zeros(2, 18))
        old = actor.get_hidden_state().clone()
        assert old.shape == (1, 2, 256) and torch.count_nonzero(old)
        actor.reset(torch.tensor([True, False]))
        new = actor.get_hidden_state()
        assert torch.count_nonzero(new[:, 0]) == 0
        assert torch.equal(new[:, 1], old[:, 1])


def test_export_carries_same_public_sequence_and_cold_reset(tmp_path):
    actor, _ = make_actor()
    actor.eval()
    # Use the actual installed mjlab export, not a second model implementation.
    runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
    runner.alg = SimpleNamespace(get_policy=lambda: actor)
    runner.export_policy_to_onnx(str(tmp_path), "memory.onnx")
    evaluator = ReferenceEvaluator(str(tmp_path / "memory.onnx"))
    jit_actor = actor.as_jit().eval()
    rng = np.random.default_rng(113)
    inputs = rng.normal(size=(12, 1, 65)).astype(np.float32)
    hidden = np.zeros((1, 1, 256), dtype=np.float32)
    first = None
    actor.reset()
    with torch.no_grad():
        for value in inputs:
            obs = TensorDict({"actor": torch.from_numpy(value)}, batch_size=[1])
            expected = actor(obs).numpy()
            actions, hidden = evaluator.run(None, {"obs": value, "h_in": hidden})
            np.testing.assert_allclose(actions, expected, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(hidden, actor.get_hidden_state().numpy(),
                rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(jit_actor(torch.from_numpy(value)),
                torch.from_numpy(expected), rtol=2e-5, atol=2e-6)
            if first is None:
                first = actions.copy()
        jit_actor.reset()
        cold = evaluator.run(None, {"obs": inputs[0],
            "h_in": np.zeros_like(hidden)})[0]
        np.testing.assert_allclose(cold, first, rtol=0, atol=0)
        torch.testing.assert_close(jit_actor(torch.from_numpy(inputs[0])),
            torch.from_numpy(first), rtol=2e-5, atol=2e-6)


def test_reload_reproduces_weights_but_starts_declared_cold_memory(tmp_path):
    actor, obs = make_actor()
    with torch.no_grad():
        actor(obs)
    path = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), path)
    resumed, _ = make_actor()
    resumed.load_state_dict(torch.load(path, weights_only=True))
    assert resumed.get_hidden_state() is None
    before = deepcopy(actor)
    before.reset()
    with torch.no_grad():
        torch.testing.assert_close(resumed(obs), before(obs), rtol=0, atol=0)
