"""Named, unadmitted non-foot ground prediction on the native contact seam.

The discovery band does not define the material surface. Normal constraints
use the true signed gap and one 20 ms closing velocity; the native solver,
its regularization/friction cone and the inherited sole law remain in place.
No state correction, extra integration or replacement solver is implemented.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import DT, sha256, write_json
from .mjlab_baseline import build_task_proxy_reference, TASK_PROXY_CANDIDATE

CANDIDATE = "goose_task_proxy_11_speculative_ground_v1"
PAIR_MARGIN_CANDIDATE = "goose_task_proxy_11_speculative_ground_pair_margin_v1"
PREDICTION_CANDIDATES = (CANDIDATE, PAIR_MARGIN_CANDIDATE)
REVISION = "goose_nonfoot_ground_velocity_constraint_v1"
BAND = .08


def validate_prediction(model, contract, *, entity_prefix=""):
    """Reject a missing adapter, inflated rest surface or different pair set."""
    p = contract.get("ground_prediction", {})
    if (contract.get("candidate") not in PREDICTION_CANDIDATES
            or p.get("revision") != REVISION
            or p.get("discovery_band_m") != BAND
            or p.get("material_surface_offset_m") != 0.
            or p.get("normal_reference") != "minus_velocity_over_dt_minus_true_gap_over_dt_squared"
            or p.get("upstream_solver_unchanged") is not True
            or contract.get("upstream_baseline", {}).get("contact_adapter_revision") != REVISION
            or model.opt.timestep != DT
            or model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError("Explicit named speculative-ground contract required")
    ground = model.geom("ground").id
    feet = {model.geom(entity_prefix+row["geom"]).id
            for row in contract["contact_mapping"]["ground_contact_quadrature"]}
    targets = {g for g in range(model.ngeom)
               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and g not in feet}
    declared = {model.geom(entity_prefix+n).id for n in p.get("nonfoot_geoms", ())}
    pairs = {tuple(sorted((int(a), int(b))))
             for a, b in zip(model.pair_geom1, model.pair_geom2)}
    if (len(targets) != 9 or declared != targets or model.npair != 9
            or pairs != {tuple(sorted((ground, g))) for g in targets}
            or np.any(model.pair_margin != BAND)
            or np.any(model.pair_gap != 0)
            or np.any(model.pair_dim != 3)):
        raise ValueError("Only the nine original non-foot/ground pairs may predict")
    for i, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2)):
        friction = np.maximum(model.geom_friction[a], model.geom_friction[b])
        expected_friction = np.array([friction[0], friction[0], friction[1], friction[2], friction[2]])
        if (model.geom_priority[a] != model.geom_priority[b]
                or model.geom_solmix[a] != model.geom_solmix[b] or model.geom_solmix[a] != 1.
                or not np.array_equal(model.pair_solref[i], model.geom_solref[a])
                or not np.array_equal(model.geom_solref[a], model.geom_solref[b])
                or not np.array_equal(model.pair_solimp[i], .5*(model.geom_solimp[a]+model.geom_solimp[b]))
                or not np.array_equal(model.pair_friction[i], expected_friction)
                or np.any(model.pair_solreffriction[i])):
            raise ValueError("Prediction changed native material mixing")
    return ground, targets


def require_pair_margin_backend(contract, model):
    """Bind the isolated, named Warp discovery fix; never patch an import live."""
    if contract.get("candidate") != PAIR_MARGIN_CANDIDATE:
        return
    import importlib.metadata
    import mujoco_warp
    from mujoco_warp._src.types import BroadphaseType

    backend = contract.get("prediction_backend", {})
    if (backend.get("revision") != "goose_warp_explicit_pair_discovery_v1"
            or backend.get("parent_version") != "3.8.1"
            or importlib.metadata.version("mujoco-warp") != "3.8.1"
            or model.opt.broadphase != BroadphaseType.NXN
            or set(backend.get("module_sha256", {})) != {"collision_driver.py", "collision_primitive.py", "solver.py", "forward.py"}):
        raise ValueError("Named pair-margin backend identity required")
    root = Path(mujoco_warp.__file__).parent / "_src"
    for name, expected in backend["module_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Wrong frozen pair-margin backend: {name}")


def _preserve_physical_arrays(before, after):
    for name in dir(before):
        old = getattr(before, name)
        if isinstance(old, np.ndarray) and (name.startswith(("body_", "dof_", "geom_", "jnt_", "eq_", "actuator_", "mesh_")) or name in ("qpos0", "exclude_signature")):
            if not np.array_equal(old, getattr(after, name)):
                raise ValueError(f"Prediction changed parent physical array: {name}")
    for name in dir(before.opt):
        value = getattr(before.opt, name)
        if not name.startswith("_") and name != "disableflags" and not callable(value):
            if not np.array_equal(value, getattr(after.opt, name)):
                raise ValueError(f"Prediction changed native physical option: {name}")


def build_reference(parent_model: Path, parent_contract: Path, source_root: Path,
                    destination: Path):
    """Freeze a new identity; never overwrite the parent or inherit its gate."""
    if destination.exists():
        raise FileExistsError("Preserve the previous prediction candidate")
    base_model, base_contract = build_task_proxy_reference(
        parent_model, parent_contract, source_root, destination / "parent")
    native = mujoco.MjModel.from_xml_path(str(base_model))
    contract = json.loads(base_contract.read_text())
    foot_names = {row["geom"] for row in contract["contact_mapping"]["ground_contact_quadrature"]}
    targets = [g for g in range(native.ngeom)
               if native.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
               and native.geom(g).name not in foot_names]
    ground = native.geom("ground").id
    tree = ET.parse(base_model)
    root = tree.getroot()
    root.set("model", CANDIDATE)
    contacts = root.find("contact")
    if contacts is None:
        contacts = ET.SubElement(root, "contact")
    if contacts.findall("pair"):
        raise ValueError("Prediction parent must not already contain explicit pairs")
    for g in targets:
        # Freeze the parent's native equal-priority material mixing; ground
        # and hull impedance differ. An explicit pair must retain their blend.
        if not (np.array_equal(native.geom_solref[g], native.geom_solref[ground])
                and native.geom_priority[g] == native.geom_priority[ground]
                and native.geom_solmix[g] == native.geom_solmix[ground] == 1.
                and native.geom_condim[g] == native.geom_condim[ground] == 3):
            raise ValueError("Unknown native ground material mixing")
        friction = np.maximum(native.geom_friction[g], native.geom_friction[ground])
        solimp = .5 * (native.geom_solimp[g] + native.geom_solimp[ground])
        ET.SubElement(contacts, "pair", name="prediction_"+native.geom(g).name,
            geom1="ground", geom2=native.geom(g).name, condim="3", margin=str(BAND), gap="0",
            friction=" ".join(map(str, (friction[0], friction[0], friction[1], friction[2], friction[2]))),
            solref=" ".join(map(str, native.geom_solref[g])),
            solimp=" ".join(map(str, solimp)))
    out = destination / "candidate"
    out.mkdir()
    for relative in contract["asset_sha256"]:
        shutil.copy2(base_model.parent / relative, out / relative)
    model_path = out / "robot.xml"
    tree.write(model_path, encoding="unicode")
    runtime = source_root / "src/sai_agent/goose/task_proxy_runtime.py"
    contract = copy.deepcopy(contract)
    contract.update(candidate=CANDIDATE, model_sha256=sha256(model_path),
        status="SPECULATIVE_GROUND_UNADMITTED", training_release=False,
        ground_prediction={"revision": REVISION, "discovery_band_m": BAND,
            "material_surface_offset_m": 0., "upstream_solver_unchanged": True,
            "normal_reference": "minus_velocity_over_dt_minus_true_gap_over_dt_squared",
            "nonfoot_geoms": [native.geom(g).name for g in targets]},
        native_cpu_parent={"model_path": str(parent_model.resolve()),
            "contract_path": str(parent_contract.resolve()),
            "model_sha256": sha256(parent_model), "contract_sha256": sha256(parent_contract),
            "source_root": str(source_root.resolve()), "runtime_sha256": sha256(runtime)})
    contract["upstream_baseline"].update(parent_candidate=TASK_PROXY_CANDIDATE,
        parent_model_sha256=sha256(base_model), parent_contract_sha256=sha256(base_contract),
        contact_adapter_revision=REVISION, source_qualified=False, target_qualified=False)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    validate_prediction(model, contract)
    _preserve_physical_arrays(native, model)
    path = out / "contract.json"
    write_json(path, contract)
    return model_path, path


def rebind_cpu_rows(model, data, contract):
    """Native soft unilateral constraint in one-step, zero-offset gap units."""
    ground, targets = validate_prediction(model, contract)
    rows = []
    for cid, contact in enumerate(data.contact):
        if ground not in contact.geom or contact.efc_address < 0:
            continue
        gid = int(contact.geom[1] if contact.geom[0] == ground else contact.geom[0])
        if gid not in targets:
            continue
        row = int(contact.efc_address)
        if contact.dim != 3 or not np.isfinite(data.efc_D[row]) or data.efc_D[row] <= 0:
            raise RuntimeError("Unsupported native predictive contact row")
        gap = float(contact.dist)
        velocity = float(data.efc_vel[row])
        ref = -velocity / DT - gap / DT**2
        data.efc_pos[row] = gap
        data.efc_aref[row] = ref
        if data.nisland:
            island_row = int(data.map_efc2iefc[row])
            data.iefc_aref[island_row] = ref
        rows.append((cid, row, gap, velocity, ref, float(data.efc_D[row])))
    return rows


def make_source_runtime(model_path: Path, contract_path: Path, *, skill="recovery"):
    """Reuse the pinned source controller and sole law, at explicit construction."""
    contract = json.loads(contract_path.read_text())
    if sha256(model_path) != contract["model_sha256"]:
        raise ValueError("Prediction model/contract hash mismatch")
    native = mujoco.MjModel.from_xml_path(str(model_path))
    validate_prediction(native, contract)
    p = contract["native_cpu_parent"]
    pm, pc, sr = Path(p["model_path"]), Path(p["contract_path"]), Path(p["source_root"])
    if (sha256(pm) != p["model_sha256"] or sha256(pc) != p["contract_sha256"]
            or sha256(sr / "src/sai_agent/goose/task_proxy_runtime.py") != p["runtime_sha256"]):
        raise ValueError("Pinned source controller identity changed")
    sys.path.insert(0, str(sr / "src"))
    module = importlib.import_module("sai_agent.goose.task_proxy_runtime")
    if sha256(Path(module.__file__)) != p["runtime_sha256"]:
        raise ValueError("A different source runtime is already imported")

    class SpeculativeSourceRuntime(module.TaskProxyRuntime):
        def __init__(self):
            # Parent loader verifies its own immutable SI/model/source modules.
            super().__init__(pm, pc, skill=skill)
            _preserve_physical_arrays(self.model, native)
            expected = int(self.model.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
            expected &= ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            if int(native.opt.disableflags) != expected:
                raise ValueError("Prediction integration flags changed")
            self.model = native
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            self.data = mujoco.MjData(native)
            self.contract = contract
            self.predictive_rows = []
            self.reset()

        def _planar_sole_quadrature(self):
            super()._planar_sole_quadrature()
            self.predictive_rows = rebind_cpu_rows(self.model, self.data, self.contract)

        def _nonfoot_ground_contact(self):
            # Discovery-only rows do not count as actual material contact.
            return any(self.ground in c.geom and c.dist <= 0. and
                int(c.geom[1] if c.geom[0] == self.ground else c.geom[0])
                not in self.foot_geoms for c in self.data.contact)

    return SpeculativeSourceRuntime()
