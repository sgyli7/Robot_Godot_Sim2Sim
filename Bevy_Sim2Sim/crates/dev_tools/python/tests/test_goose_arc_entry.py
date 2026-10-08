import copy
import json
import math

import pytest

from bevy_microduck_tools.goose.move_arc_entry import GooseArcPhaseSelector

PHASE = 1.2566370964050293


@pytest.mark.parametrize("commands", [
    [(0, 0, 0), (.14, 0, .1), (.14, 0, .1)],
    [(.14, 0, 0), (.14, 0, .1), (.14, 0, -.1)],
    [(.27, 0, 0), (0, 0, 0), (.14, 0, .1)],
    [(0, .1, 0), (.14, 0, .1)],
    [(-.15, 0, 0), (.14, 0, -.1)],
])
def test_keeps_clock_across_cold_zero_slow_and_arc_changes(commands):
    selector = GooseArcPhaseSelector(PHASE)
    assert [selector.select(command) for command in commands] == [None] * len(commands)


@pytest.mark.parametrize("previous", [(0, 0, .3), (0, 0, -.3), (.27, 0, 0)])
@pytest.mark.parametrize("arc", [(.14, 0, .1), (.14, 0, -.1)])
def test_hot_entry_requests_one_restart_and_preserves_caller_commands(previous, arc):
    selector = GooseArcPhaseSelector(PHASE)
    commands = [list(previous), list(arc), list(arc)]
    saved = copy.deepcopy(commands)
    assert [selector.select(command) for command in commands] == [None, PHASE, None]
    assert commands == saved


def test_pause_snapshot_preserves_transition_at_boundary_and_reset_clears_it():
    selector = GooseArcPhaseSelector(PHASE)
    selector.select((.27, 0, 0))
    paused = GooseArcPhaseSelector.from_snapshot(json.loads(json.dumps(selector.snapshot())))
    assert paused.select((.14, 0, .1)) == selector.select((.14, 0, .1)) == PHASE
    paused.reset()
    assert paused.select((.14, 0, .1)) is None


@pytest.mark.parametrize("invalid", [(1, 2), (math.nan, 0, 0), (0, math.inf, 0)])
def test_invalid_command_does_not_corrupt_resumable_history(invalid):
    selector = GooseArcPhaseSelector(PHASE)
    selector.select((.27, 0, 0))
    before = selector.snapshot()
    with pytest.raises(ValueError):
        selector.select(invalid)
    assert selector.snapshot() == before
    assert selector.select((.14, 0, .1)) == PHASE


@pytest.mark.parametrize("phase,boundary", [(-.1, .17), (2*math.pi, .17),
    (math.nan, .17), (PHASE, 0), (PHASE, math.inf)])
def test_invalid_manifest_units_are_rejected(phase, boundary):
    with pytest.raises(ValueError):
        GooseArcPhaseSelector(phase, boundary)


@pytest.mark.parametrize("key,value", [("revision", "future"),
    ("previous_user_kind", "unknown"), ("reference_phase_radians", math.inf)])
def test_incompatible_pause_snapshot_is_rejected(key, value):
    snapshot = GooseArcPhaseSelector(PHASE).snapshot()
    snapshot[key] = value
    with pytest.raises(ValueError):
        GooseArcPhaseSelector.from_snapshot(snapshot)
