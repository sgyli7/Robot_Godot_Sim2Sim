"""Explicitly unqualified CPU runtime for a versioned numerical experiment."""
from __future__ import annotations

import json

import numpy as np

from .implicit_metric import REVISION, native_step
from .runtime import GooseSourceRuntime


class GooseExperimentalSourceRuntime(GooseSourceRuntime):
    """Share drive/observation/reset code without replacing the default solver."""
    def __init__(self, model_path, contract_path, **kwargs):
        contract = json.loads(contract_path.read_text())
        if (contract.get("schema") != "goose_50hz_experimental_si_v2"
                or contract.get("candidate") != "goose_460_full50_be_v2"
                or contract.get("status") != "M0_UNQUALIFIED"
                or contract.get("training_release") is not False
                or contract.get("numerical_metric", {}).get("revision") != REVISION):
            raise ValueError("Experimental runtime needs an explicit unqualified v2 contract")
        super().__init__(model_path, contract_path, **kwargs)
        ledger = contract["numerical_metric"]["joints"]
        for entry in ledger:
            joint = self.model.joint(entry["joint"])
            did = int(joint.dofadr[0])
            if (did != entry["dof"] or self.model.dof_armature[did] != entry["compiled_armature"]
                    or self.model.dof_damping[did] != entry["compiled_damping"]):
                raise ValueError("Experimental model differs from frozen numerical metric")
        if len(ledger) != self.model.njnt-1 or not np.isfinite(self.model.dof_armature).all():
            raise ValueError("Experimental metric does not cover the full scalar tree")

    def _integrate(self):
        native_step(self.model, self.data)
