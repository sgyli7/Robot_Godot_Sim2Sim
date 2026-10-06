"""Counterexamples and axis semantics for the pinned tracking task."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.recovery_tracking import (
    canonical, load_author, qualified_support, reference_phase)


def test_canonical_axes_keep_real_ankles_upper_jaw_and_all18_gradients():
    q = torch.arange(18., requires_grad=True)
    output = canonical(q)
    torch.testing.assert_close(output[:6], q[[14, 13, 12, 15, 16, 17]])
    torch.testing.assert_close(output[6:12], q[[8, 7, 6, 9, 10, 11]])
    torch.testing.assert_close(output[15:21], q[:6])
    assert output[[12, 13, 14, 21, 22]].tolist() == [0.]*5
    output.sum().backward()
    torch.testing.assert_close(q.grad, torch.ones(18))


def test_reference_lookup_roundtrip_has_no_off_by_one_and_clamps_terminal():
    ticks = torch.arange(1000)
    recovered = (reference_phase(ticks, 627)*626).to(torch.int32)
    torch.testing.assert_close(recovered, ticks.clamp(max=626).to(torch.int32))


@pytest.mark.parametrize("counterexample", (
    "inverted_sole", "one_foot_only", "insufficient_support", "body_support",
    "low_body", "moving_body", "reclining_body"))
def test_success_rejects_known_false_positive_standing_signals(counterexample):
    values = dict(up=torch.ones(1), height=torch.tensor([.28]), nominal_height=.28,
        velocity=torch.zeros((1, 3)), angular=torch.zeros((1, 3)),
        normals=torch.ones((1, 2)), foot_loads=torch.tensor([[51.2, 51.2]]),
        nonfoot_load=torch.zeros(1), weight=102.4)
    assert qualified_support(**values).item()
    if counterexample == "inverted_sole":
        values["normals"][0, 1] = -1.
    elif counterexample == "one_foot_only":
        values["foot_loads"][0] = torch.tensor([102.4, 0.])
    elif counterexample == "insufficient_support":
        values["foot_loads"][0] = torch.tensor([8., 8.])
    elif counterexample == "body_support":
        values["nonfoot_load"][0] = 30.
    elif counterexample == "low_body":
        values["height"][0] = .16
    elif counterexample == "moving_body":
        values["velocity"][0, 0] = .1
    else:
        values["up"][0] = .65
    assert not qualified_support(**values).item()


def test_author_weights_formula_and_real_axis_subsets_are_unchanged():
    path = os.environ.get("GOOSE_TRACKING_AUTHOR_ROOT")
    if path is None:
        pytest.skip("Pinned HumanUP source required")
    root = Path(path)
    source, config = root/"g1waist_track.py", root/"g1waist_track_config.py"
    base = root.parent/"base/humanoid.py"
    cls, weights = load_author(source, sha256(source), config, sha256(config), base, sha256(base))
    assert weights["tracking_dof_error"] == 8 and weights["termination"] == -50
    ctx = cls()
    ctx.device = "cpu"
    ctx.cfg = SimpleNamespace(rewards=SimpleNamespace(tracking_sigma=.2))
    ctx.target_traj_length = 3
    ctx.episode_length_buf = torch.tensor([0, 1, 100])
    ctx.dof_pos_all_interp = canonical(torch.zeros((3, 18)))
    ctx.dof_pos = canonical(torch.full((3, 18), .1))
    expected = torch.exp(torch.full((3,), -18*.1**2/4))
    torch.testing.assert_close(ctx._reward_tracking_dof_error(), expected)
    ctx.torques = canonical(torch.arange(18.))[None, :]
    # Unchanged author's ankle/upper index sets now refer to the real Goose
    # ankles/upper motors; no neck/jaw torque is accidentally called an ankle.
    torch.testing.assert_close(ctx._reward_ankle_torques(),
        torch.linalg.vector_norm(torch.tensor([[16., 17., 10., 11.]]), dim=-1))
    torch.testing.assert_close(ctx._reward_upper_torques(),
        torch.linalg.vector_norm(torch.arange(6.)[None, :], dim=-1))
    with pytest.raises(ValueError, match="identity"):
        load_author(source, "0"*64, config, sha256(config), base, sha256(base))
