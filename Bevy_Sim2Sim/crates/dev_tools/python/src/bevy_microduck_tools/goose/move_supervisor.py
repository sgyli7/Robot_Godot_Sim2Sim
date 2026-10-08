"""Explicit command history for the source Move development actor bundle.

This selector sees user commands only. Each selected actor still receives
the actual public Actor65 observation and emits the original18 actions.
It establishes no speed, direction, physical or Bevy qualification.
"""
import math
from enum import Enum

REVISION = "goose_command_history_stop_selector_v1"


class MoveActorRole(str, Enum):
    MOVEMENT = "moving136"
    TRANSLATION_STOP = "alpha025"
    YAW_STOP = "alpha075"


class GooseMoveActorSelector:
    """Remember the last nonzero user command; select one actor per Tick.

    A mixed translation/yaw command uses the translation stopping actor.
    The genuine cold-start controller precedes this selector. Reset this
    state on a cold physics reset; preserve its snapshot across a pause.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._last_nonzero_kind = "translation"

    def select(self, command):
        if len(command) != 3:
            raise ValueError("Move commands require body vx, vy and yaw_rate")
        vx, vy, yaw_rate = (float(value) for value in command)
        if not all(math.isfinite(value) for value in (vx, vy, yaw_rate)):
            raise ValueError("Move commands must be finite")
        if vx != 0 or vy != 0:
            self._last_nonzero_kind = "translation"
            return MoveActorRole.MOVEMENT
        if yaw_rate != 0:
            self._last_nonzero_kind = "pure_yaw"
            return MoveActorRole.MOVEMENT
        return (MoveActorRole.YAW_STOP if self._last_nonzero_kind == "pure_yaw"
                else MoveActorRole.TRANSLATION_STOP)

    def snapshot(self):
        return {"revision": REVISION,
            "last_nonzero_kind": self._last_nonzero_kind}

    @classmethod
    def from_snapshot(cls, snapshot):
        if (set(snapshot) != {"revision", "last_nonzero_kind"}
                or snapshot["revision"] != REVISION
                or snapshot["last_nonzero_kind"] not in ("translation", "pure_yaw")):
            raise ValueError("Incompatible Move supervisor snapshot")
        result = cls()
        result._last_nonzero_kind = snapshot["last_nonzero_kind"]
        return result
