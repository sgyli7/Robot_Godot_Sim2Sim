"""Versioned adoption of the unchanged discrete plant by current native mjlab.

mjlab1.6 owns model assembly, graph capture and stepping. Select the discrete
enum before its compiled model upload. The predecessor's mjlab1.3 cannot run
Warp3.15's removed linesearch option; keep that standalone source reference
and its dependency guard intact. This named adoption inherits no qualification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Literal
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mjlab.sim import MujocoCfg

from .artifacts import DT, sha256, write_json
from .batch_drive import BatchedRigidNativeDrive
from . import upstream_discrete as source

REVISION = "goose_upstream_discrete_mjlab160_v1"
CANDIDATE = "goose_task_proxy_11_discrete_mjlab160_v1"
VERSIONS = {"mjlab": "1.6.0", "mujoco": "3.15.0", "mujoco-warp": "3.15.0",
            "warp-lang": "1.15.0", "rsl-rl-lib": "5.4.2"}


def require_upstream_stack():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Discrete mjlab adoption requires project Python3.12")
    for distribution, version in VERSIONS.items():
        if importlib.metadata.version(distribution) != version:
            raise ValueError(f"Discrete mjlab adoption requires {distribution}=={version}")


def _read_source_reference(model_path, contract_path):
    """Validate frozen plant metadata without bypassing its old runtime guard."""
    contract = json.loads(contract_path.read_text())
    cfg = contract.get("upstream_discrete", {})
    if (contract["candidate"] != source.CANDIDATE
            or contract["model_sha256"] != sha256(model_path)
            or cfg.get("revision") != source.REVISION
            or cfg.get("versions") != source.VERSIONS
            or cfg.get("module_sha256") != sha256(Path(source.__file__))
            or cfg.get("drive_module_sha256") != sha256(Path(
                sys.modules[BatchedRigidNativeDrive.__module__].__file__))
            or cfg.get("option_changes") != ["integrator: implicitfast -> discrete"]
            or cfg.get("refsafe_enabled") is not True
            or contract["integrator"] != "discrete"):
        raise ValueError("Frozen standalone discrete reference identity changed")
    parent_model, parent_contract = Path(cfg["parent_model"]), Path(cfg["parent_contract"])
    if (sha256(parent_model) != cfg["parent_model_sha256"]
            or sha256(parent_contract) != cfg["parent_contract_sha256"]):
        raise ValueError("Discrete source parent identity changed")
    parent = source._validate_parent(parent_model, parent_contract)
    allowed = {"candidate", "integrator", "model_sha256", "source_runtime",
               "status", "training_release", "optimizer_updates", "upstream_discrete"}
    if any(contract.get(k) != v for k, v in parent.items() if k not in allowed):
        raise ValueError("Discrete source changed the inherited control contract")
    for relative, digest in contract["asset_sha256"].items():
        if sha256(model_path.parent/relative) != digest:
            raise ValueError("Discrete source geometry identity changed")
    original = mujoco.MjModel.from_xml_path(str(parent_model))
    compiled = mujoco.MjModel.from_xml_path(str(model_path))
    for field in source.PLANT_FIELDS:
        if not np.array_equal(getattr(original, field), getattr(compiled, field)):
            raise ValueError(f"Discrete source changed plant field {field}")
    for field in dir(original.opt):
        if field.startswith("_") or field == "integrator" or callable(getattr(original.opt, field)):
            continue
        if not np.array_equal(getattr(original.opt, field), getattr(compiled.opt, field)):
            raise ValueError(f"Discrete source changed option {field}")
    if (compiled.opt.timestep != DT
            or compiled.opt.integrator != mujoco.mjtIntegrator.mjINT_DISCRETE
            or compiled.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)):
        raise ValueError("Frozen source discrete/refsafe/timing profile changed")
    return contract, compiled


def build_reference(source_model: Path, source_contract: Path, destination: Path):
    """Name the dependency adoption, preserving every compiled physics field."""
    require_upstream_stack()
    source_model, source_contract, destination = map(Path,
        (source_model, source_contract, destination))
    contract, _ = _read_source_reference(source_model, source_contract)
    if destination.exists():
        raise FileExistsError("Preserve the existing mjlab adoption")
    tree = ET.parse(source_model)
    tree.getroot().set("model", CANDIDATE)
    destination.mkdir(parents=True)
    for relative in contract["asset_sha256"]:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_model.parent/relative, target)
    model_path = destination/"robot.xml"
    tree.write(model_path, encoding="unicode")
    new = copy.deepcopy(contract)
    new.update(candidate=CANDIDATE, model_sha256=sha256(model_path),
        source_runtime="bevy_microduck_tools.goose.upstream_discrete_mjlab.make_source_runtime",
        status="UPSTREAM_MJLAB_LINKAGE_REFERENCE_ONLY", training_release=False,
        optimizer_updates=0, upstream_mjlab={"revision": REVISION, "versions": VERSIONS,
            "module_sha256": sha256(Path(__file__)),
            "source_model": str(source_model.resolve()),
            "source_contract": str(source_contract.resolve()),
            "source_model_sha256": sha256(source_model),
            "source_contract_sha256": sha256(source_contract),
            "physics_option_changes": [], "decimation": 1,
            "custom_solver": False, "source_qualified": False,
            "target_qualified": False, "policy_qualified": False})
    contract_path = destination/"contract.json"
    write_json(contract_path, new)
    return model_path, contract_path


def make_source_runtime(model_path: Path, contract_path: Path):
    require_upstream_stack()
    model_path, contract_path = Path(model_path), Path(contract_path)
    contract = json.loads(contract_path.read_text())
    cfg = contract.get("upstream_mjlab", {})
    if (contract["candidate"] != CANDIDATE or cfg.get("revision") != REVISION
            or cfg.get("versions") != VERSIONS
            or cfg.get("module_sha256") != sha256(Path(__file__))
            or contract["model_sha256"] != sha256(model_path)
            or cfg.get("physics_option_changes") != []
            or cfg.get("decimation") != 1 or cfg.get("custom_solver") is not False
            or any(cfg.get(key) is not False for key in
                ("source_qualified", "target_qualified", "policy_qualified"))
            or contract.get("training_release") is not False):
        raise ValueError("Named mjlab adoption identity changed")
    parent_model, parent_contract = Path(cfg["source_model"]), Path(cfg["source_contract"])
    if (sha256(parent_model) != cfg["source_model_sha256"]
            or sha256(parent_contract) != cfg["source_contract_sha256"]):
        raise ValueError("Named mjlab source reference changed")
    parent, reference = _read_source_reference(parent_model, parent_contract)
    allowed = {"candidate", "model_sha256", "source_runtime", "status",
               "training_release", "optimizer_updates", "upstream_mjlab"}
    if any(contract.get(k) != v for k, v in parent.items() if k not in allowed):
        raise ValueError("Named mjlab adoption changed the control contract")
    for relative, digest in contract["asset_sha256"].items():
        if sha256(model_path.parent/relative) != digest:
            raise ValueError("Named mjlab adoption geometry changed")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    for field in source.PLANT_FIELDS:
        if not np.array_equal(getattr(reference, field), getattr(model, field)):
            raise ValueError(f"Named mjlab adoption changed plant field {field}")
    for field in dir(reference.opt):
        if field.startswith("_") or callable(getattr(reference.opt, field)):
            continue
        if not np.array_equal(getattr(reference.opt, field), getattr(model.opt, field)):
            raise ValueError(f"Named mjlab adoption changed option {field}")
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    return source.DiscreteSourceRuntime(model, contract)


@dataclass
class DiscreteMujocoCfg(MujocoCfg):
    integrator: Literal["discrete"] = "discrete"

    def apply(self, model: mujoco.MjModel) -> None:
        require_upstream_stack()
        if (self.integrator != "discrete" or self.timestep != DT
                or "refsafe" in self.disableflags):
            raise ValueError("Discrete mjlab bridge requires its safe 50Hz profile")
        inherited = copy.copy(self)
        inherited.integrator = "implicitfast"
        MujocoCfg.apply(inherited, model)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
