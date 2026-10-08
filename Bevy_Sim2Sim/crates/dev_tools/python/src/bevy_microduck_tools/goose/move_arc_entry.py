"""Optional command-only phase entry for the named source Move arc bundle.

The caller applies a returned phase before this Tick's Actor observation.
The existing drive still advances its clock once in its normal20ms commit.
This module changes no body, action history, drive target or physics setting.
It establishes no speed, physical or Bevy qualification.
"""
import math

REVISION = "goose_selective_direct_arc_phase_entry_v1"
_KINDS = ("zero", "translation", "pure_yaw", "fast", "arc_left", "arc_right")


class GooseArcPhaseSelector:
    """Restart only when pure yaw or straight fast motion enters an arc.

    Cold/zero, slow-to-arc and arc-to-arc transitions keep the current phase.
    Reference phase and fast boundary belong to the selected bundle manifest.
    Preserve this selector's snapshot across a pause; reset on a cold reset.
    """

    def __init__(self, reference_phase_radians, fast_command_boundary_m_s=.17):
        phase = float(reference_phase_radians)
        boundary = float(fast_command_boundary_m_s)
        if not math.isfinite(phase) or not 0 <= phase < 2 * math.pi:
            raise ValueError("Reference phase must be finite and in [0, 2pi)")
        if not math.isfinite(boundary) or boundary <= 0:
            raise ValueError("Fast command boundary must be finite and positive")
        self.reference_phase_radians = phase
        self.fast_command_boundary_m_s = boundary
        self.reset()

    def reset(self):
        self._previous_user_kind = "zero"

    def select(self, command):
        if len(command) != 3:
            raise ValueError("Move commands require body vx, vy and yaw_rate")
        vx, vy, yaw = (float(value) for value in command)
        if not all(math.isfinite(value) for value in (vx, vy, yaw)):
            raise ValueError("Move commands must be finite")
        if vx == vy == yaw == 0:
            kind = "zero"
        elif vx > 0 and vy == 0 and yaw != 0:
            kind = "arc_left" if yaw > 0 else "arc_right"
        elif vx == vy == 0:
            kind = "pure_yaw"
        elif vx > self.fast_command_boundary_m_s and vy == yaw == 0:
            kind = "fast"
        else:
            kind = "translation"
        restart = (kind in ("arc_left", "arc_right")
                   and self._previous_user_kind in ("pure_yaw", "fast"))
        self._previous_user_kind = kind
        return self.reference_phase_radians if restart else None

    def snapshot(self):
        return {"revision": REVISION,
                "reference_phase_radians": self.reference_phase_radians,
                "fast_command_boundary_m_s": self.fast_command_boundary_m_s,
                "previous_user_kind": self._previous_user_kind}

    @classmethod
    def from_snapshot(cls, snapshot):
        if (set(snapshot) != {"revision", "reference_phase_radians",
                "fast_command_boundary_m_s", "previous_user_kind"}
                or snapshot["revision"] != REVISION
                or snapshot["previous_user_kind"] not in _KINDS):
            raise ValueError("Incompatible arc phase snapshot")
        result = cls(snapshot["reference_phase_radians"],
                     snapshot["fast_command_boundary_m_s"])
        result._previous_user_kind = snapshot["previous_user_kind"]
        return result
