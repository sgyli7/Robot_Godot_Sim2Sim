"""Named free payload assets for Move, using native MuJoCo/mjlab entities.

These are test objects, not hardware load qualifications. Their grip origin is
the object frame origin and their grip axis is Y. Reset placement is supplied
by the task; this module never attaches an object or advances its pose. Holding
and transport must subsequently pass with the real mouth/contact dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import sqrt

import mujoco
from mjlab.entity import EntityCfg

REVISION = "goose_move_real_payload_assets_v1"
MASS_TIERS_G = (100, 200, 300, 500)
FAMILIES = ("cylinder_grip", "open_handle", "open_handle_wide")


@dataclass(frozen=True)
class MovePayload:
    family: str
    mass_g: int

    def __post_init__(self):
        if (self.family not in FAMILIES or type(self.mass_g) is not int
                or self.mass_g not in MASS_TIERS_G):
            raise ValueError("Payload requires a named family and 100/200/300/500g tier")

    @property
    def challenge(self):
        return self.mass_g == 500


@dataclass(frozen=True)
class PayloadContact:
    """Copy nominal contact properties from the actual frozen mouth surface."""
    source_geom: str
    fields: tuple[tuple[str, object], ...]

    @classmethod
    def from_model(cls, model, mouth_geom_name: str):
        gid = model.geom(mouth_geom_name).id
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            raise ValueError("Payload contact requires a colliding mouth surface")
        fields = {name: tuple(float(x) for x in getattr(model, "geom_" + name)[gid])
                  for name in ("friction", "solref", "solimp")}
        fields.update({name: float(getattr(model, "geom_" + name)[gid])
                       for name in ("margin", "gap", "solmix")})
        fields.update({name: int(getattr(model, "geom_" + name)[gid])
                       for name in ("contype", "conaffinity", "condim", "priority", "group")})
        return cls(mouth_geom_name, tuple(fields.items()))


def make_move_payload_spec(*, payload: MovePayload, contact: PayloadContact):
    """Native compound mass/inertia, one free body, no actuators or equalities.

    The cylinder family is a 12mm x 70mm grip bar, connecting stem and lower
    weight. The handle family has a 12mm grip bar and two arms around a real
    opening, joined to a solid case. The separately named wide handle places
    its side arms at +/-60mm to clear the measured 103.74mm upper bill; the
    original narrow handle remains unchanged. All three/four leaves count.
    Component masses scale together within each family; MuJoCo aggregates the
    COM and inertia. No claim of exact manufactured material density is made.
    """
    if not isinstance(payload, MovePayload) or not isinstance(contact, PayloadContact):
        raise ValueError("Payload assets require frozen geometry and contact records")
    spec = mujoco.MjSpec()
    spec.modelname = f"{REVISION}_{payload.family}_{payload.mass_g}g"
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_free")
    mass = payload.mass_g / 1000.
    properties = dict(contact.fields)
    y_axis = (sqrt(.5), -sqrt(.5), 0., 0.)
    x_axis = (sqrt(.5), 0., sqrt(.5), 0.)

    def geom(name, kind, size, fraction, pos=(0., 0., 0.), quat=(1., 0., 0., 0.)):
        body.add_geom(name=name, type=kind, size=size, mass=mass*fraction,
                      pos=pos, quat=quat, **properties)

    if payload.family == "cylinder_grip":
        geom("grip", mujoco.mjtGeom.mjGEOM_CYLINDER, (.006, .035, 0.), .08, quat=y_axis)
        geom("stem", mujoco.mjtGeom.mjGEOM_CYLINDER, (.010, .0175, 0.), .04,
             pos=(0., 0., -.0175))
        geom("weight", mujoco.mjtGeom.mjGEOM_CYLINDER, (.030, .025, 0.), .88,
             pos=(0., 0., -.055))
    else:
        half_span = .060 if payload.family == "open_handle_wide" else .027
        case_half_width = .065 if payload.family == "open_handle_wide" else .035
        geom("grip", mujoco.mjtGeom.mjGEOM_CAPSULE, (.006, half_span, 0.), .04, quat=y_axis)
        for side in (-1, 1):
            geom("arm_left" if side > 0 else "arm_right", mujoco.mjtGeom.mjGEOM_CAPSULE,
                 (.004, .0225, 0.), .03, pos=(.0225, side*half_span, 0.), quat=x_axis)
        geom("case", mujoco.mjtGeom.mjGEOM_BOX, (.035, case_half_width, .035), .90,
             pos=(.080, 0., -.035))
    return spec


def make_move_payload_entity_cfg(*, payload: MovePayload, contact: PayloadContact,
                                 initial_state: EntityCfg.InitialStateCfg):
    """Reuse the native entity/reset implementation; no online pose controller."""
    if not isinstance(initial_state, EntityCfg.InitialStateCfg):
        raise ValueError("Payload reset placement must be an explicit native initial state")
    return EntityCfg(spec_fn=partial(make_move_payload_spec, payload=payload, contact=contact),
                     init_state=initial_state)
