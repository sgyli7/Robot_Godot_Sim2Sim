"""Goose pickup goal extension shared by training and target interface."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PHASES = ("approach", "bend_align", "close_beak", "lift", "carry", "place")


def quaternion_product(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


@dataclass(frozen=True)
class TaskGoal:
    object_id: str
    position_world_m: tuple[float, float, float]
    rotation_world_wxyz: tuple[float, float, float, float]
    placement_world_m: tuple[float, float, float]
    valid: bool = True

    def actor_extension(self, torso_position_world_m, torso_rotation_world_wxyz, torso_rotation_matrix, phase):
        if phase not in range(len(PHASES)):
            raise ValueError("Unknown pickup stage")
        if not self.valid:
            return invalid_goal_extension(phase)
        if not self.object_id:
            raise ValueError("Valid goal requires object identity and one of six phases")
        position, placement = np.asarray(self.position_world_m), np.asarray(self.placement_world_m)
        objq, torsoq = np.asarray(self.rotation_world_wxyz), np.asarray(torso_rotation_world_wxyz)
        if (position.shape != (3,) or placement.shape != (3,) or objq.shape != (4,)
                or torsoq.shape != (4,) or not all(np.isfinite(v).all() for v in (position, placement, objq, torsoq))
                or np.linalg.norm(objq) < 1e-12 or np.linalg.norm(torsoq) < 1e-12):
            raise ValueError("Invalid goal/body pose")
        objq = objq / np.linalg.norm(objq)
        torsoq = torsoq / np.linalg.norm(torsoq)
        inverse = torsoq * [1, -1, -1, -1]
        relative = quaternion_product(inverse, objq)
        # Deterministic equivalent sign, with lexicographic tie at a pi rotation.
        first = next((v for v in relative if abs(v) > 1e-12), 1.)
        if first < 0:
            relative = -relative
        inverse_rotation = np.asarray(torso_rotation_matrix).T
        phases = np.zeros(6)
        phases[phase] = 1
        return np.clip(np.r_[inverse_rotation @ (position - torso_position_world_m), relative,
                     inverse_rotation @ (placement - torso_position_world_m), phases, 1.], -20, 20).astype(np.float32)


def invalid_goal_extension(phase=0):
    if phase not in range(len(PHASES)):
        raise ValueError("Unknown pickup stage")
    value = np.zeros(17, dtype=np.float32)
    value[10 + phase] = 1
    return value
