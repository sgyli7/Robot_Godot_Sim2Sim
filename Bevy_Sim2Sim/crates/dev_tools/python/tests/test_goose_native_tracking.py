"""Reference playback cannot teleport physics or advance on partial reset."""
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.command_manager import CommandTerm

from bevy_microduck_tools.goose.native_tracking import (
    AdmittedColdMotionCommand, GooseDevelopmentEnv, NativeTrackingColdResetEnv,
    MotionCommand, ReadOnlyMotionCommand,
)


def command_fixture():
    term = object.__new__(ReadOnlyMotionCommand)
    drive = SimpleNamespace(completed_ticks=torch.tensor([0, 12]), _pending_action=None)
    term._env = SimpleNamespace(action_manager=SimpleNamespace(
        get_term=lambda _: SimpleNamespace(drive=drive)))
    term.cfg = SimpleNamespace(sampling_mode="start")
    term.motion = SimpleNamespace(time_step_total=627)
    term.time_steps = torch.tensor([0, 13])
    term.update_relative_body_poses = lambda: None
    return term, drive


def test_cold_subset_reset_does_not_advance_running_reference():
    term, _ = command_fixture()
    term._resample_command(torch.tensor([0]))
    term.compute(0., env_ids=torch.tensor([0]))
    assert term.time_steps.tolist() == [1, 13]
    with pytest.raises(RuntimeError, match="genuine cold"):
        term._resample_command(torch.tensor([1]))
    assert term.time_steps.tolist() == [1, 13]


def test_current_command_api_accepts_scoped_reset_without_double_lookahead():
    term, _ = command_fixture()
    CommandTerm._check_update_command_signature(term)
    term._resample_command(torch.tensor([0]))
    term._update_command(torch.tensor([0]))
    assert term.time_steps.tolist() == [1, 13]
    term._update_command()
    assert term.time_steps.tolist() == [2, 14]


def test_scoped_reset_does_not_decay_adaptive_failure_history():
    term, _ = admitted_fixture()
    term.bin_failed_count = torch.tensor([2., 3.])
    term._current_bin_failed = torch.tensor([4., 5.])
    term._update_command(torch.tensor([0]))
    assert term.time_steps.tolist() == [0, 13]
    assert term.bin_failed_count.tolist() == [2., 3.]
    assert term._current_bin_failed.tolist() == [4., 5.]
    term._update_command()
    assert term.time_steps.tolist() == [1, 14]
    torch.testing.assert_close(term.bin_failed_count, torch.tensor([2.002, 3.002]))
    assert term._current_bin_failed.tolist() == [0., 0.]


def test_end_of_clip_holds_without_resampling_or_state_write():
    term, _ = command_fixture()
    term.time_steps[:] = torch.tensor([625, 626])
    term._resample_command = lambda _: pytest.fail("Clip end attempted a reset")
    for _ in range(3):
        term._update_command()
    assert term.time_steps.tolist() == [626, 626]
    for operation in (term._write_reference_state_to_sim, term.reset_to_frame):
        with pytest.raises(RuntimeError, match="actual robot state"):
            operation()
    assert term.time_steps.tolist() == [626, 626]


def test_clip_timeout_observes_final_frame_without_restarting_physics():
    from bevy_microduck_tools.goose.native_tracking import native_motion_clip_ended

    term, _ = command_fixture()
    term.time_steps[:] = torch.tensor([625, 626])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    term._resample_command = lambda _: pytest.fail("Timeout attempted a state reset")
    assert native_motion_clip_ended(env).tolist() == [False, True]
    assert term.time_steps.tolist() == [625, 626]
    term._update_command()
    assert native_motion_clip_ended(env).tolist() == [True, True]
    assert term.time_steps.tolist() == [626, 626]


def test_pending_tick_and_non20ms_reference_updates_are_rejected():
    term, drive = command_fixture()
    drive._pending_action = torch.zeros((2, 18))
    with pytest.raises(RuntimeError, match="genuine cold"):
        term._resample_command(torch.tensor([0]))
    for dt in (.005, .01, .04):
        with pytest.raises(ValueError, match="20ms"):
            term.compute(dt)
    assert term.time_steps.tolist() == [0, 13]


def admitted_fixture():
    term, drive = command_fixture()
    term.__class__ = AdmittedColdMotionCommand
    term.cfg = SimpleNamespace(sampling_mode="adaptive", adaptive_alpha=.001)
    term._env._reference_birth_ids = None
    term.eligible_frames = torch.tensor([15, 19, 426, 626])
    term.birth_proposal = torch.zeros(2, dtype=torch.long)
    return term, drive


def test_adaptive_native_proposals_map_only_to_admitted_births(monkeypatch):
    term, _ = admitted_fixture()
    def proposals(command, ids):
        command.time_steps[ids] = torch.tensor([17, 625])
    monkeypatch.setattr(MotionCommand, "_adaptive_sampling", proposals)
    term._adaptive_sampling(torch.tensor([0, 1]))
    assert term.birth_proposal.tolist() == [17, 625]
    assert term.time_steps.tolist() == [15, 626]  # Earlier frame wins a tie.


def test_reference_state_restart_outside_selected_cold_reset_is_rejected(monkeypatch):
    term, drive = admitted_fixture()
    monkeypatch.setattr(MotionCommand, "_resample_command",
                        lambda *args: pytest.fail("Attempted running state write"))
    with pytest.raises(RuntimeError, match="genuine cold"):
        term._resample_command(torch.tensor([0]))
    term._env._reference_birth_ids = torch.tensor([0])
    drive.completed_ticks[:] = 0
    with pytest.raises(RuntimeError, match="genuine cold"):
        term._resample_command(torch.tensor([1]))
    drive._pending_action = torch.zeros((2, 18))
    with pytest.raises(RuntimeError, match="genuine cold"):
        term._resample_command(torch.tensor([0]))


def test_invalid_birth_joint_is_rejected_before_native_projection(monkeypatch):
    term, _ = admitted_fixture()
    term._env._reference_birth_ids = torch.tensor([0])
    term.robot = SimpleNamespace(data=SimpleNamespace(
        soft_joint_pos_limits=torch.tensor([[[-1., 1.]], [[-1., 1.]]])))
    monkeypatch.setattr(MotionCommand, "_write_reference_state_to_sim",
                        lambda *args: pytest.fail("Native writer would project joint"))
    root = torch.zeros((1, 3))
    with pytest.raises(RuntimeError, match="joint projection"):
        term._write_reference_state_to_sim(torch.tensor([0]), root,
            torch.tensor([[1., 0., 0., 0.]]), root, root,
            torch.tensor([[1.01]]), torch.zeros((1, 1)))


def test_failed_cold_reset_cannot_leave_reference_writes_authorized(monkeypatch):
    class ResetFixture(NativeTrackingColdResetEnv):
        @property
        def num_envs(self):
            return 2

        @property
        def device(self):
            return "cpu"

    env = object.__new__(ResetFixture)
    def fail_reset(self, **kwargs):
        assert self._reference_birth_ids.tolist() == [0]
        raise ValueError("Preserve failed cold reset")
    monkeypatch.setattr(GooseDevelopmentEnv, "reset", fail_reset)
    with pytest.raises(ValueError, match="Preserve failed"):
        env.reset(env_ids=torch.tensor([0]))
    assert env._reference_birth_ids is None


def test_native_action_rate_undoes_public_units_before_installed_penalty():
    from mjlab.envs import mdp
    from bevy_microduck_tools.goose.native_action_units import native_coordinate_action_rate_l2

    scale = tuple(torch.linspace(.02, .3, 18).tolist())
    raw = (torch.arange(54, dtype=torch.float64).reshape(3, 18)/10).requires_grad_()
    previous = raw.detach()*.4
    env = SimpleNamespace(action_manager=SimpleNamespace(action=raw, prev_action=previous))
    expected = mdp.action_rate_l2(env)
    factor = raw.new_tensor(scale)
    public = SimpleNamespace(action_manager=SimpleNamespace(action=raw*factor, prev_action=previous*factor))
    before = public.action_manager.action.clone()
    corrected = native_coordinate_action_rate_l2(public, scale)
    torch.testing.assert_close(corrected, expected)
    torch.testing.assert_close(torch.autograd.grad(corrected.sum(), raw, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), raw)[0])
    assert torch.equal(public.action_manager.action, before)
    assert not torch.allclose(mdp.action_rate_l2(public), expected)


@pytest.mark.parametrize('invalid', [(0.,)*18, (float('inf'),)*18, (.1,)*17])
def test_native_action_rate_rejects_invalid_coordinate_contract(invalid):
    from bevy_microduck_tools.goose.native_action_units import native_coordinate_action_rate_l2

    env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(2, 18), prev_action=torch.zeros(2, 18)))
    with pytest.raises(ValueError, match='positive finite18'):
        native_coordinate_action_rate_l2(env, invalid)
