"""Explicit scene contact roles for rough terrain and real carried objects.

Geom zero has no special meaning. Only declared robot geoms form self contact;
terrain, carried objects and obstacles are separate even when their IDs are
nonzero. Penetration uses MuJoCo's signed distance to the contacted surface.
This measurement seam does not modify collision filtering or apply forces.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

REVISION = "goose_move_scene_contact_roles_v1"


class ContactRole(IntEnum):
    VISUAL = 0
    ROBOT = 1
    TERRAIN = 2
    PAYLOAD = 3
    OBSTACLE = 4


@dataclass(frozen=True)
class SceneContactRoles:
    roles: tuple[ContactRole, ...]
    geom_names: tuple[str, ...]

    @classmethod
    def from_model(cls, model, assignments: dict[str, ContactRole]):
        """Require an explicit, unambiguous role for every colliding geom."""
        names = tuple(model.geom(index).name for index in range(model.ngeom))
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("Scene geoms require unique names for frozen role mapping")
        if set(assignments) - set(names):
            raise ValueError("Role mapping names a geom absent from the compiled scene")
        roles = []
        for index, name in enumerate(names):
            colliding = bool(model.geom_contype[index] or model.geom_conaffinity[index])
            role = assignments.get(name, ContactRole.VISUAL)
            if not isinstance(role, ContactRole):
                raise ValueError("Scene contact roles must use ContactRole values")
            if colliding and role is ContactRole.VISUAL:
                raise ValueError("Colliding geom has no declared role: " + name)
            roles.append(role)
        if ContactRole.ROBOT not in roles or ContactRole.TERRAIN not in roles:
            raise ValueError("Move scene requires robot and terrain roles")
        return cls(tuple(roles), names)

    def depth_by_world(self, geoms, distance, world_ids, *, active, num_worlds):
        """Return separate max depths from a CPU or GPU native contact buffer.

        Padded contacts are ignored. Invalid IDs in actual active contacts fail
        instead of silently treating a payload/obstacle as part of the robot.
        Negative signed distance means penetration. No world-height assumption
        appears here: an elevated step's side/top remains a terrain surface.
        """
        if num_worlds <= 0 or geoms.ndim != 2 or geoms.shape[1] != 2:
            raise ValueError("Native contacts require positive world count and Nx2 geoms")
        shape = (len(geoms),)
        if distance.shape != shape or world_ids.shape != shape or active.shape != shape:
            raise ValueError("Contact fields have inconsistent shapes")
        if active.dtype != torch.bool:
            raise ValueError("Active contacts require an explicit boolean mask")
        if (not torch.isfinite(distance[active]).all()
                or ((geoms[active] < 0) | (geoms[active] >= len(self.roles))).any()
                or ((world_ids[active] < 0) | (world_ids[active] >= num_worlds)).any()):
            raise ValueError("Active native contact is nonfinite or outside the scene")
        role = torch.tensor(self.roles, device=geoms.device, dtype=torch.long)
        pair = role[geoms.long().clamp(0, len(self.roles)-1)]
        robot0, robot1 = pair[:, 0] == ContactRole.ROBOT, pair[:, 1] == ContactRole.ROBOT
        masks = {"self": robot0 & robot1}
        for name, other in (("terrain", ContactRole.TERRAIN),
                            ("payload", ContactRole.PAYLOAD),
                            ("obstacle", ContactRole.OBSTACLE)):
            masks[name] = (robot0 & (pair[:, 1] == other)) | (robot1 & (pair[:, 0] == other))
        result = {}
        for name, mask in masks.items():
            depth = torch.where(active & mask, (-distance).clamp_min(0), 0.)
            values = torch.zeros(num_worlds, device=distance.device, dtype=distance.dtype)
            values.scatter_reduce_(0, world_ids.long().clamp(0, num_worlds-1), depth,
                reduce="amax", include_self=True)
            result[name] = values
        return result
