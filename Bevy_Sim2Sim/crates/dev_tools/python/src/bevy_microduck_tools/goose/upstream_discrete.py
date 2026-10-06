"""Versioned adoption of MuJoCo's upstream discrete integrator.

The existing rigid controller is reused. This is a new plant identity, with no
inherited physical, policy or target qualification. Old candidates and their
dependency guards remain unchanged.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .artifacts import DT, JOINT_ORDER, sha256, write_json
from . import rigid_native
from .batch_drive import BatchedRigidNativeDrive

CANDIDATE = "goose_task_proxy_11_upstream_discrete315_v1"
REVISION = "goose_upstream_discrete315_v1"
VERSIONS = {"mjlab": "1.3.0", "mujoco": "3.15.0", "mujoco-warp": "3.15.0",
            "warp-lang": "1.15.0", "rsl-rl-lib": "5.0.1"}

# These fields determine the inherited plant, actuation and public axis layout.
# Both models are compiled by the same engine. Changing the integrator does not
# permit changes to any of these arrays.
PLANT_FIELDS = (
    "body_parentid", "body_mass", "body_inertia", "body_ipos", "body_iquat",
    "body_pos", "body_quat", "jnt_type", "jnt_pos", "jnt_axis", "jnt_range",
    "jnt_qposadr", "jnt_dofadr", "jnt_limited", "jnt_margin", "jnt_solref",
    "jnt_solimp", "dof_armature", "dof_damping", "dof_frictionloss", "qpos0",
    "geom_type", "geom_bodyid", "geom_dataid", "geom_size", "geom_pos",
    "geom_quat", "geom_contype", "geom_conaffinity", "geom_friction",
    "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_condim",
    "mesh_vert", "mesh_graph", "eq_type", "eq_obj1id", "eq_obj2id",
    "eq_data", "eq_solref", "eq_solimp", "actuator_trnid", "actuator_gear",
    "actuator_gainprm", "actuator_biasprm", "actuator_forcelimited",
    "actuator_forcerange", "actuator_ctrllimited", "actuator_ctrlrange",
)


def require_upstream_stack():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The discrete candidate requires project Python 3.12")
    for distribution, version in VERSIONS.items():
        if importlib.metadata.version(distribution) != version:
            raise ValueError(f"Discrete candidate requires {distribution}=={version}")


def _validate_parent(model_path, contract_path):
    contract = json.loads(contract_path.read_text())
    cfg = contract.get("rigid_native", {})
    if (contract["candidate"] != rigid_native.BRAKING_CANDIDATE
            or cfg.get("revision") != rigid_native.REVISION
            or cfg.get("module_sha256") != sha256(Path(rigid_native.__file__))
            or contract["model_sha256"] != sha256(model_path)
            or sha256(Path(cfg["parent_model"])) != cfg["parent_model_sha256"]
            or sha256(Path(cfg["parent_contract"])) != cfg["parent_contract_sha256"]
            or contract["integrator"] != "implicitfast"):
        raise ValueError("Frozen rigid braking parent identity changed")
    for relative, digest in contract["asset_sha256"].items():
        if sha256(model_path.parent / relative) != digest:
            raise ValueError("Parent geometry identity changed")
    return contract


def build_reference(parent_model: Path, parent_contract: Path, destination: Path):
    """Freeze one explicit integrator change, preserving the rigid H parent."""
    require_upstream_stack()
    parent_model, parent_contract, destination = map(Path,
        (parent_model, parent_contract, destination))
    contract = _validate_parent(parent_model, parent_contract)
    if destination.exists():
        raise FileExistsError("Preserve the existing discrete candidate")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    root.set("model", CANDIDATE)
    root.find("option").set("integrator", "discrete")
    destination.mkdir(parents=True)
    for relative in contract["asset_sha256"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_model.parent / relative, target)
    model_path = destination / "robot.xml"
    tree.write(model_path, encoding="unicode")
    new = copy.deepcopy(contract)
    new.update(candidate=CANDIDATE, integrator="discrete",
        model_sha256=sha256(model_path),
        source_runtime="bevy_microduck_tools.goose.upstream_discrete.make_source_runtime",
        status="UPSTREAM_RUNTIME_REFERENCE_ONLY", training_release=False,
        optimizer_updates=0,
        upstream_discrete={"revision": REVISION, "versions": VERSIONS,
            "module_sha256": sha256(Path(__file__)),
            "drive_module_sha256": sha256(Path(sys.modules[BatchedRigidNativeDrive.__module__].__file__)),
            "parent_model": str(parent_model.resolve()),
            "parent_contract": str(parent_contract.resolve()),
            "parent_model_sha256": sha256(parent_model),
            "parent_contract_sha256": sha256(parent_contract),
            "option_changes": ["integrator: implicitfast -> discrete"],
            "refsafe_enabled": True, "decimation": 1,
            "custom_solver": False, "source_qualified": False,
            "target_qualified": False, "policy_qualified": False})
    path = destination / "contract.json"
    write_json(path, new)
    return model_path, path


def make_source_runtime(model_path: Path, contract_path: Path):
    """Reuse the original 18-axis step/thermal/history implementation."""
    require_upstream_stack()
    model_path, contract_path = Path(model_path), Path(contract_path)
    contract = json.loads(contract_path.read_text())
    cfg = contract.get("upstream_discrete", {})
    if (contract["candidate"] != CANDIDATE or cfg.get("revision") != REVISION
            or cfg.get("versions") != VERSIONS
            or cfg.get("module_sha256") != sha256(Path(__file__))
            or contract["model_sha256"] != sha256(model_path)
            or contract["integrator"] != "discrete"
            or contract["joint_order"] != list(JOINT_ORDER)
            or cfg.get("option_changes") != ["integrator: implicitfast -> discrete"]
            or cfg.get("refsafe_enabled") is not True
            or any(contract[k] != DT for k in
                ("physics_dt_s", "torque_dt_s", "policy_dt_s"))
            or any(contract[k] != 1 for k in
                ("physics_steps_per_tick", "torque_updates_per_tick", "policy_calls_per_tick"))):
        raise ValueError("Discrete candidate model/controller/timing identity changed")
    parent_model, parent_contract = Path(cfg["parent_model"]), Path(cfg["parent_contract"])
    if (sha256(parent_model) != cfg["parent_model_sha256"]
            or sha256(parent_contract) != cfg["parent_contract_sha256"]):
        raise ValueError("Discrete candidate parent identity changed")
    parent = _validate_parent(parent_model, parent_contract)
    # The entire public/control contract is inherited; only the named artifact
    # identity and integrator metadata may differ.
    allowed = {"candidate", "integrator", "model_sha256", "source_runtime",
               "status", "training_release", "optimizer_updates", "upstream_discrete"}
    if any(contract.get(k) != v for k, v in parent.items() if k not in allowed):
        raise ValueError("Discrete adoption changed the inherited control contract")
    for relative, digest in contract["asset_sha256"].items():
        if sha256(model_path.parent / relative) != digest:
            raise ValueError("Discrete candidate geometry changed")
    drive_path = Path(sys.modules[BatchedRigidNativeDrive.__module__].__file__)
    if cfg.get("drive_module_sha256") != sha256(drive_path):
        raise ValueError("Discrete controller module identity changed")
    parent_model_compiled = mujoco.MjModel.from_xml_path(str(parent_model))
    parent_model_compiled.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    if (model.opt.timestep != DT or model.opt.integrator != mujoco.mjtIntegrator.mjINT_DISCRETE
            or model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)):
        raise ValueError("Discrete integration or refsafe profile changed")
    for name in PLANT_FIELDS:
        if not np.array_equal(getattr(parent_model_compiled, name), getattr(model, name)):
            raise ValueError(f"Discrete adoption changed plant field {name}")
    # Preserve every other option, including contact solver, flags and capacity.
    for name in dir(model.opt):
        if name.startswith("_") or name == "integrator":
            continue
        value = getattr(model.opt, name)
        if callable(value):
            continue
        if not np.array_equal(getattr(parent_model_compiled.opt, name), value):
            raise ValueError(f"Discrete adoption changed option {name}")
    return DiscreteSourceRuntime(model, contract)


class DiscreteSourceRuntime:
    """Thin native step adapter; the existing batch drive owns all control state.

    The old CPU TaskProxyRuntime remains pinned to 3.10 and is never patched or
    instantiated with a false version. No contact correction, reward, policy,
    optimizer or alternate integration is implemented here.
    """

    def __init__(self, model, contract):
        self.model, self.contract = model, contract
        self.data = mujoco.MjData(model)
        self.drive = BatchedRigidNativeDrive(contract, 1, device="cpu")
        self.qidx = np.array([model.joint(name).qposadr[0] for name in JOINT_ORDER])
        self.vidx = np.array([model.joint(name).dofadr[0] for name in JOINT_ORDER])
        self.motor_vids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.torso = model.body("torso").id
        self.commands = np.zeros(3)
        self.reset()

    @property
    def last_tau(self):
        return self.drive.last_tau[0].numpy()

    def reset(self):
        # Same nominal birth convention as the original published runtime.
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[2] += .002
        d.qpos[self.qidx] = self.drive.neutral.numpy()
        for joint in self.contract["passive_linkage_joints"]:
            source = int(m.joint(joint["mimic_joint"]).qposadr[0])
            target = int(m.joint(joint["name"]).qposadr[0])
            d.qpos[target] = joint["mimic_multiplier"]*d.qpos[source]+joint["mimic_offset_rad"]
        self.drive.reset(torch.tensor([0]))
        mujoco.mj_forward(m, d)
        self.physics_integrations = self.controller_updates = 0
        return self.observations()

    def observations(self):
        d = self.data
        return self.drive.observations(d.qpos[self.qidx][None], d.qvel[self.vidx][None],
            d.qvel[None, 3:6], d.xmat[self.torso].reshape(1, 3, 3),
            self.commands[None])[0].numpy()

    def step(self, action):
        m, d = self.model, self.data
        ctrl = self.drive.prepare(np.asarray(action)[None], d.qpos[self.qidx][None],
            d.qvel[self.vidx][None], d.qpos[None, 3:7],
            motor_velocity=d.qvel[self.motor_vids][None])[0].numpy()
        m.actuator_forcerange[:] = np.stack((self.drive.force_lower[0].numpy(),
                                            self.drive.force_upper[0].numpy()), axis=-1)
        d.ctrl[:] = ctrl
        before = float(d.time)
        mujoco.mj_step(m, d)
        self.physics_integrations += 1
        self.controller_updates += 1
        if (abs(d.time-before-DT) > 1e-12
                or not all(np.isfinite(getattr(d, name)).all() for name in
                    ("qpos", "qvel", "qacc", "ctrl", "actuator_force"))
                or any(w.number for w in d.warning)):
            raise FloatingPointError("Discrete native integration/finite guard failed")
        self.drive.commit(d.actuator_force[None])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        height = float(d.subtree_com[self.torso, 2])
        upright = float(d.xmat[self.torso].reshape(3, 3)[2, 2])
        return self.observations(), {"height_m": height, "upright": upright,
            "failure": height < .18 or upright < .65, "auto_reset": False,
            "physics_integrations": self.physics_integrations,
            "controller_updates": self.controller_updates, "time_s": float(d.time)}
