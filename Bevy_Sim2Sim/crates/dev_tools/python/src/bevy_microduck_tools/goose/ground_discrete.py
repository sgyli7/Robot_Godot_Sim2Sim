"""Versioned CPU ground-contact experiment on the native discrete metric.

The original pad springs remain physical. Ground rows use their actual gap,
finite regularization and the native elliptic Coulomb cone. Detection margin
does not become the contact boundary. Self and object contacts are unchanged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import sha256
from .native_discrete_runtime import _require_platform
from .predictive_discrete import CANDIDATE_EXACT as PARENT, advance
from .runtime import GooseSourceRuntime

CANDIDATE = "goose_460_full50_discrete_ground_v6"
REVISION = "goose_native_discrete_ground_predictive_v1"
LAW = {"revision": REVISION, "scope": "static_ground_only", "ground_geom": "ground",
       "activation_margin_m": .01, "regularization_scale": 1e-5,
       "normal_regularization_floor": 1e-12, "actual_gap_boundary": True,
       "preserve_native_cone_ratios": True, "physical_contact_law_changed": True}


def prepare_ground_constraints(model, data, ground):
    """Modify only static-ground rows before the existing corrected solve."""
    _require_platform()
    if (model.opt.timestep != .02 or model.opt.cone != int(mujoco.mjtCone.mjCONE_ELLIPTIC)
            or model.geom_bodyid[ground] != 0
            or model.geom_type[ground] != int(mujoco.mjtGeom.mjGEOM_PLANE)):
        raise ValueError("Predictive ground requires a static plane and elliptic cone")
    if model.body_margin[0] < model.geom_margin[ground]:
        raise ValueError("Ground detection has a stale body-margin cache")
    changed, contacts = 0, 0
    for contact in data.contact:
        if ground not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
            continue
        row, dim = int(contact.efc_address), int(contact.dim)
        expected = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS if dim == 1 else mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC
        if data.efc_type[row] != int(expected) or contact.adhesion:
            raise ValueError("Unsupported ground constraint or adhesion")
        original = data.efc_R[row:row+dim].copy()
        if not np.isfinite(original).all() or np.any(original <= 0):
            raise FloatingPointError("Invalid native ground regularization")
        normal = max(LAW["normal_regularization_floor"], float(original[0])*LAW["regularization_scale"])
        regularization = normal*(original/original[0])
        reference = -data.efc_vel[row:row+dim]/.02
        reference[0] -= float(contact.dist)/.02**2
        data.efc_R[row:row+dim] = regularization
        data.efc_D[row:row+dim] = 1/regularization
        data.efc_aref[row:row+dim] = reference
        if data.nisland:
            indices = data.map_efc2iefc[row:row+dim]
            if np.any(indices < 0):
                raise RuntimeError("Ground rows absent from native island")
            data.iefc_R[indices] = regularization
            data.iefc_D[indices] = 1/regularization
            data.iefc_aref[indices] = reference
        changed += dim
        contacts += 1
    return {"ground_contacts": contacts, "ground_rows": changed,
            "actual_gap_boundary": True, "coordinate_writes": 0}


def build_candidate(parent_model: Path, parent_contract: Path, destination: Path):
    """Preserve the v5 parent and freeze the declared ground intervention."""
    _require_platform()
    contract = json.loads(parent_contract.read_text())
    if (contract.get("candidate") != PARENT or contract["model_sha256"] != sha256(parent_model)
            or not contract["native_discrete"]["predictive_limits"].get("exact_diagonal")):
        raise ValueError("Frozen exact-diagonal v5 parent required")
    if destination.exists():
        raise FileExistsError("Preserve the prior ground candidate")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    ground = root.find("./worldbody/geom[@name='ground']")
    if ground is None or ground.get("type") != "plane":
        raise ValueError("Parent has no declared static ground plane")
    root.set("model", CANDIDATE)
    ground.set("margin", str(LAW["activation_margin_m"]))
    destination.mkdir(parents=True)
    model_path = destination/"robot.xml"
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode")
    contract = copy.deepcopy(contract)
    contract.update(candidate=CANDIDATE, model_sha256=sha256(model_path),
                    status="SOURCE_GROUND_CONTACT_UNQUALIFIED", training_release=False,
                    ground_contact=copy.deepcopy(LAW))
    contract["native_discrete"].update(revision=REVISION, physical_parameters_changed=True,
        unchanged_body_drive_parameters=True, parent_model_sha256=sha256(parent_model),
        parent_contract_sha256=sha256(parent_contract),
        physical_parameter_changes=["ground detection margin", "ground row reference and regularization"])
    path = destination/"contract.json"
    path.write_text(json.dumps(contract, indent=2)+"\n")
    return model_path, path


class GooseGroundDiscreteRuntime(GooseSourceRuntime):
    """Keep the 65/18 controller; select the declared ground law explicitly."""
    def __init__(self, model_path: Path, contract_path: Path, **kwargs):
        _require_platform()
        contract = json.loads(contract_path.read_text())
        native = contract.get("native_discrete", {})
        if (contract.get("candidate") != CANDIDATE or contract.get("ground_contact") != LAW
                or native.get("revision") != REVISION or native.get("engine_version") != "3.13.0"
                or native.get("physical_parameters_changed") is not True
                or native.get("unchanged_body_drive_parameters") is not True
                or not native.get("predictive_limits", {}).get("exact_diagonal")):
            raise ValueError("Explicit frozen ground-contact identity required")
        super().__init__(model_path, contract_path, **kwargs)
        if (not self.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
                or self.model.geom_margin[self.ground] != LAW["activation_margin_m"]
                or self.model.body_margin[0] < self.model.geom_margin[self.ground]):
            raise ValueError("Compiled ground profile disagrees with the contract")
        self.advance_receipt = None

    def _integrate(self):
        self.advance_receipt = advance(self.model, self.data,
            prepare_constraints=lambda model, data: prepare_ground_constraints(model, data, self.ground))
