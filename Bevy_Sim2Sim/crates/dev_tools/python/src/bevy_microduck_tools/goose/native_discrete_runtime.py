"""Explicit MuJoCo 3.13 native effective-metric experiment, CPU admission only.

Original physical springs, armature, friction and topology are retained. This
does not qualify Rapier, MuJoCo Warp or training and never replaces v1/v2.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from .artifacts import DT, sha256
from .runtime import GooseSourceRuntime

ENGINE_VERSION = "3.13.0"
REVISION = "goose_native_discrete313_v1"
CANDIDATE = "goose_460_full50_discrete_v3"


def _require_platform():
    if mujoco.__version__ != ENGINE_VERSION:
        raise ValueError("Native discrete runtime requires isolated MuJoCo 3.13.0")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Native discrete runtime requires project Python 3.12")


def build_candidate(parent_model: Path, parent_contract: Path, destination: Path):
    """Freeze a new identity; change only numerical stepper and failure handling."""
    _require_platform()
    contract = json.loads(parent_contract.read_text())
    if (contract.get("candidate") != "goose_460_full50_v1"
            or contract["model_sha256"] != sha256(parent_model)
            or contract.get("numerical_metric") is not None
            or len(contract["bodies"]) != 33):
        raise ValueError("Frozen original 33-body full50 v1 parent required")
    if destination.exists():
        raise FileExistsError("Preserve the prior native discrete candidate")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    root.set("model", CANDIDATE)
    option = root.find("option")
    if option is None or float(option.get("timestep")) != DT:
        raise ValueError("Parent timestep is not 20ms")
    option.set("integrator", "discrete")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("autoreset", "disable")
    destination.mkdir(parents=True)
    model_path = destination/"robot.xml"
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode")
    contract = copy.deepcopy(contract)
    contract.update(candidate=CANDIDATE, integrator="discrete", model_sha256=sha256(model_path),
                    status="SOURCE_NATIVE_DISCRETE_UNQUALIFIED", training_release=False,
                    native_discrete={"revision": REVISION, "engine_version": ENGINE_VERSION,
                                     "parent_model_sha256": sha256(parent_model),
                                     "parent_contract_sha256": sha256(parent_contract),
                                     "physical_parameters_changed": False,
                                     "metric": "native M+hD+h^2K; qacc=(v_next-v)/h",
                                     "scope": "CPU source experiment; no target/GPU/M0 admission"})
    contract_path = destination/"contract.json"
    contract_path.write_text(json.dumps(contract, indent=2)+"\n")
    return model_path, contract_path


class GooseDiscreteSourceRuntime(GooseSourceRuntime):
    """Select native discrete explicitly; keep original 50Hz drive and observations."""

    def __init__(self, model_path: Path, contract_path: Path, **kwargs):
        _require_platform()
        contract = json.loads(contract_path.read_text())
        native = contract.get("native_discrete", {})
        if (contract.get("candidate") != CANDIDATE or contract.get("integrator") != "discrete"
                or native.get("revision") != REVISION
                or native.get("engine_version") != ENGINE_VERSION
                or native.get("physical_parameters_changed") is not False):
            raise ValueError("Explicit frozen native discrete identity required")
        super().__init__(model_path, contract_path, **kwargs)
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_DISCRETE:
            raise ValueError("Compiled model is not the declared native discrete integrator")

    def _integrate(self):
        """One native effective-metric solve and one position update, no inner loop."""
        mujoco.mj_step(self.model, self.data)
