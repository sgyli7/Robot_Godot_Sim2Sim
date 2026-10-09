"""Declared native collision roles and signed-surface diagnostics for Move.

This opt-in measurement module never changes a model, collision mask, action or
world state. Depths come from actual native contact distances, not world z=0.
Contact depth alone is not evidence that a payload is held or an impact passed.
Callers still admit contact capacity, substep sampling and GPU reset separately.
"""
from __future__ import annotations

from enum import IntEnum

import mujoco
import torch

REVISION = "goose_move_declared_contact_roles_v1"


class MoveContactRole(IntEnum):
    UNDECLARED = 0
    ROBOT = 1
    TERRAIN = 2
    PAYLOAD = 3
    OBSTACLE = 4


def build_move_contact_roles(model: mujoco.MjModel, *,
                             robot_root_body="robot/torso",
                             terrain_root_bodies=("terrain",),
                             terrain_geom_names=(), payload_root_bodies=(),
                             obstacle_root_bodies=(), expected_robot_leaves=14):
    """Return immutable geom-id roles, checking every colliding native leaf.

    Roles follow explicit body ancestry; a terrain box is never a robot leaf.
    Explicit named terrain geoms support a flat world plane. Native explicit
    pairs also declare colliding leaves even when their masks are zero. Missing
    or ambiguous declarations fail before a runtime can drop these contacts.
    """
    declarations = {
        MoveContactRole.ROBOT: (robot_root_body,),
        MoveContactRole.TERRAIN: terrain_root_bodies,
        MoveContactRole.PAYLOAD: payload_root_bodies,
        MoveContactRole.OBSTACLE: obstacle_root_bodies,
    }
    try:
        roots = {role: {model.body(name).id for name in names}
                 for role, names in declarations.items()}
        terrain_geoms = {model.geom(name).id for name in terrain_geom_names}
    except KeyError as error:
        raise ValueError("Move contact role names must exist in the frozen model") from error
    explicit_pair_geoms = set(model.pair_geom1.tolist()) | set(model.pair_geom2.tolist())
    roles = [int(MoveContactRole.UNDECLARED)] * model.ngeom
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]
                or geom in explicit_pair_geoms):
            continue
        ancestors = set()
        body = int(model.geom_bodyid[geom])
        while True:
            ancestors.add(body)
            if body == 0:
                break
            body = int(model.body_parentid[body])
        matches = {role for role, ids in roots.items() if ancestors & ids}
        if geom in terrain_geoms:
            matches.add(MoveContactRole.TERRAIN)
        if len(matches) != 1:
            raise ValueError(f"Move collider {model.geom(geom).name!r} needs one declared role")
        roles[geom] = int(matches.pop())
    if (isinstance(expected_robot_leaves, bool) or not isinstance(expected_robot_leaves, int)
            or expected_robot_leaves <= 0
            or roles.count(MoveContactRole.ROBOT) != expected_robot_leaves):
        raise ValueError("Move robot leaf count differs from the frozen contract")
    return tuple(roles)


def measure_move_contact_depths(*, distance_m: torch.Tensor,
                                geom_ids: torch.Tensor, world_ids: torch.Tensor,
                                role_lookup: torch.Tensor,
                                contact_count: int, num_worlds: int):
    """Reduce completed native signed contact depths by role and world.

    Pass the actual occupied contact count and cache role_lookup on the same
    device. Unused capacity may contain arbitrary data; occupied invalid ids,
    undeclared roles or nonfinite distances fail rather than disappearing.
    Sampling the completed contacts must not advance or reset the live world.
    """
    if (isinstance(contact_count, bool) or not isinstance(contact_count, int)
            or isinstance(num_worlds, bool) or not isinstance(num_worlds, int)
            or num_worlds <= 0 or distance_m.ndim != 1
            or geom_ids.shape != (len(distance_m), 2)
            or world_ids.shape != distance_m.shape or role_lookup.ndim != 1
            or not distance_m.is_floating_point()
            or any(t.dtype != torch.int64 for t in (geom_ids, world_ids, role_lookup))
            or any(t.device != distance_m.device for t in (geom_ids, world_ids, role_lookup))
            or not 0 <= contact_count <= len(distance_m)):
        raise ValueError("Move contact buffers must match native count, shape and device")
    distance = distance_m[:contact_count]
    pairs = geom_ids[:contact_count]
    worlds = world_ids[:contact_count]
    if (not torch.isfinite(distance).all() or (pairs < 0).any()
            or (pairs >= len(role_lookup)).any() or (worlds < 0).any()
            or (worlds >= num_worlds).any()):
        raise ValueError("Invalid occupied native Move contact")
    roles = role_lookup[pairs]
    if ((roles <= MoveContactRole.UNDECLARED).any()
            or (roles > MoveContactRole.OBSTACLE).any()):
        raise ValueError("Occupied Move contacts require declared roles")

    def pair(first, second):
        return ((roles[:, 0] == first) & (roles[:, 1] == second)) | (
            (roles[:, 0] == second) & (roles[:, 1] == first))

    masks = {
        "self_m": pair(MoveContactRole.ROBOT, MoveContactRole.ROBOT),
        "terrain_m": pair(MoveContactRole.ROBOT, MoveContactRole.TERRAIN),
        "payload_contact_m": pair(MoveContactRole.ROBOT, MoveContactRole.PAYLOAD),
        "obstacle_contact_m": pair(MoveContactRole.ROBOT, MoveContactRole.OBSTACLE),
        "payload_terrain_m": pair(MoveContactRole.PAYLOAD, MoveContactRole.TERRAIN),
    }
    penetration = (-distance).clamp_min(0)
    result = {}
    for name, mask in masks.items():
        value = distance_m.new_zeros(num_worlds)
        value.scatter_reduce_(0, worlds, torch.where(mask, penetration, 0.),
                              reduce="amax", include_self=True)
        result[name] = value
    return result
