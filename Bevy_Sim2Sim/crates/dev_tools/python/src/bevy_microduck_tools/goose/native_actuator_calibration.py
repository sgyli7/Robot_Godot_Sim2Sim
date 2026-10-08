"""Explicit ankle roll gain calibration on the frozen native discrete plant.

This development candidate changes two position gains only. It retains the
native integrator, force limits, damping, SI and public action targets, and
inherits no physical or policy qualification from its parent.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import sha256, write_json
from .upstream_discrete import DiscreteSourceRuntime, PLANT_FIELDS
from .upstream_discrete_mjlab import make_source_runtime as make_parent_runtime

CANDIDATE = "goose_task_proxy_11_discrete_ankle_kp10_v1"
REVISION = "goose_native_ankle_roll_gain10_v1"
AXES = (11, 17)
GAINS = {"right_ankle_roll": 10.0, "left_ankle_roll": 10.0}


def build_reference(parent_model, parent_contract, destination):
    """Freeze one controller change after validating the unchanged parent."""
    parent_model, parent_contract, destination = map(
        Path, (parent_model, parent_contract, destination))
    parent = make_parent_runtime(parent_model, parent_contract)
    contract = copy.deepcopy(parent.contract)
    if destination.exists():
        raise FileExistsError("Preserve the existing calibrated candidate")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    root.set("model", CANDIDATE)
    actuators = list(root.find("actuator"))
    for index in AXES:
        name = contract["joints"][index]["name"]
        if (name not in GAINS or actuators[index].tag != "position"
                or float(actuators[index].get("kp")) != 20.0
                or contract["joints"][index]["kp_nm_rad"] != 20.0):
            raise ValueError("Calibration requires the original ankle roll gains")
        actuators[index].set("kp", str(GAINS[name]))
        contract["joints"][index]["kp_nm_rad"] = GAINS[name]
    destination.mkdir(parents=True)
    for relative in contract["asset_sha256"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_model.parent / relative, target)
    model_path = destination / "robot.xml"
    tree.write(model_path, encoding="unicode")
    contract.update(candidate=CANDIDATE, candidate_id=CANDIDATE,
        controller_revision=REVISION, model_sha256=sha256(model_path),
        source_runtime="bevy_microduck_tools.goose.native_actuator_calibration.make_source_runtime",
        status="NATIVE_GAIN_CALIBRATION_DEVELOPMENT_ONLY", training_release=False,
        optimizer_updates=0, native_actuator_calibration={
            "revision": REVISION, "module_sha256": sha256(Path(__file__)),
            "parent_model": str(parent_model.resolve()),
            "parent_contract": str(parent_contract.resolve()),
            "parent_model_sha256": sha256(parent_model),
            "parent_contract_sha256": sha256(parent_contract),
            "position_gains_nm_rad": GAINS,
            "source_qualified": False, "target_qualified": False,
            "policy_qualified": False, "hardware_qualified": False})
    contract_path = destination / "contract.json"
    write_json(contract_path, contract)
    make_source_runtime(model_path, contract_path)
    return model_path, contract_path


def make_source_runtime(model_path, contract_path):
    """Validate every inherited field before constructing the native adapter."""
    model_path, contract_path = map(Path, (model_path, contract_path))
    contract = json.loads(contract_path.read_text())
    cfg = contract.get("native_actuator_calibration", {})
    if (contract["candidate"] != CANDIDATE
            or contract.get("candidate_id") != CANDIDATE
            or contract.get("controller_revision") != REVISION
            or cfg.get("revision") != REVISION
            or cfg.get("module_sha256") != sha256(Path(__file__))
            or cfg.get("position_gains_nm_rad") != GAINS
            or contract["model_sha256"] != sha256(model_path)
            or any(cfg.get(key) is not False for key in
                ("source_qualified", "target_qualified", "policy_qualified", "hardware_qualified"))):
        raise ValueError("Named native gain calibration identity changed")
    parent_model, parent_contract = map(Path, (cfg["parent_model"], cfg["parent_contract"]))
    if (sha256(parent_model) != cfg["parent_model_sha256"]
            or sha256(parent_contract) != cfg["parent_contract_sha256"]):
        raise ValueError("Calibration parent identity changed")
    parent = make_parent_runtime(parent_model, parent_contract).contract
    expected = copy.deepcopy(parent)
    for index in AXES:
        name = expected["joints"][index]["name"]
        expected["joints"][index]["kp_nm_rad"] = GAINS[name]
    allowed = {"candidate", "candidate_id", "controller_revision", "model_sha256",
        "source_runtime", "status", "optimizer_updates", "native_actuator_calibration"}
    if any(contract.get(k) != v for k, v in expected.items() if k not in allowed):
        raise ValueError("Gain calibration changed the inherited control contract")
    if set(contract)-set(expected) != {"native_actuator_calibration"}:
        raise ValueError("Gain calibration introduced unknown contract fields")
    for relative, digest in parent["asset_sha256"].items():
        if sha256(model_path.parent / relative) != digest:
            raise ValueError("Calibration geometry identity changed")
    original = mujoco.MjModel.from_xml_path(str(parent_model))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    for field in PLANT_FIELDS:
        wanted = getattr(original, field).copy()
        if field == "actuator_gainprm":
            wanted[list(AXES), 0] = 10.0
        elif field == "actuator_biasprm":
            wanted[list(AXES), 1] = -10.0
        if not np.array_equal(wanted, getattr(model, field)):
            raise ValueError(f"Calibration changed plant field {field}")
    for field in dir(original.opt):
        if field.startswith("_") or callable(getattr(original.opt, field)):
            continue
        if not np.array_equal(getattr(original.opt, field), getattr(model.opt, field)):
            raise ValueError(f"Calibration changed physics option {field}")
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    return DiscreteSourceRuntime(model, contract)
