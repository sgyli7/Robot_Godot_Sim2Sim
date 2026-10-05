"""Native MJCF adoption for upstream mjlab; no replacement training machinery.

The reference preserves the inherited plant and collision geometry. It is a
model integration artifact, not physical admission or a registered RL task.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

from .artifacts import CANDIDATES, DT, JOINT_ORDER, sha256, write_json

CANDIDATE = "goose_460_condensed_mjlab50_reference_v1"
EULER_CANDIDATE = "goose_task_collision_condensed_connect50_euler_reference_v1"
TASK_PROXY_CANDIDATE = "goose_task_proxy_11_mjw_contact_v1"
TASK_PROXY_ADAPTER_REVISION = "goose_task_proxy_mjw_contact_adapter_v1"
NATIVE_PARENTS = {
    CANDIDATES[1]: CANDIDATE,
    "goose_task_collision_v1_condensed50":
        "goose_task_collision_condensed_mjlab50_reference_v1",
}
VERSIONS = {"mjlab": "1.3.0", "mujoco": "3.10.0", "mujoco-warp": "3.8.1",
            "warp-lang": "1.12.0", "rsl-rl-lib": "5.0.1"}


def require_upstream_stack():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Goose training requires project Python 3.12")
    for distribution, version in VERSIONS.items():
        if importlib.metadata.version(distribution) != version:
            raise ValueError(f"Goose baseline requires {distribution}=={version}")


def _read_native_contract(model_path: Path, contract_path: Path):
    contract = json.loads(contract_path.read_text())
    if (contract.get("native_discrete") is not None
            or contract.get("numerical_metric") is not None
            or contract.get("ground_contact") is not None):
        raise ValueError("CPU experimental contracts cannot enter the upstream baseline")
    if (contract["model_sha256"] != sha256(model_path)
            or contract["joint_order"] != list(JOINT_ORDER)
            or any(contract[k] != DT for k in ("physics_dt_s", "torque_dt_s", "policy_dt_s"))
            or any(contract[k] != 1 for k in ("physics_steps_per_tick",
                                              "torque_updates_per_tick", "policy_calls_per_tick"))):
        raise ValueError("Native model identity, axis order or 50 Hz timing mismatch")
    return contract


def build_reference(parent_model: Path, parent_contract: Path, destination: Path):
    """Reuse the existing mass-conserving condensation and native MJCF exporter."""
    require_upstream_stack()
    contract = _read_native_contract(parent_model, parent_contract)
    candidate = NATIVE_PARENTS.get(contract["candidate"])
    if candidate is None:
        raise ValueError("The frozen condensed native parent is required")
    if destination.exists():
        raise FileExistsError("Preserve the previous native reference")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    root.set("model", candidate)
    # implicitfast is supported by the pinned mjlab/Warp stack. No CPU callback.
    root.find("option").set("integrator", "implicitfast")
    # AUTORESET is a C-only failure-recovery flag rejected by pinned Warp.
    # GooseSourceRuntime disables it on its own C model. Warp.step has no
    # implicit reset: its caller must abort and retain nonfinite states.
    flag = root.find("./option/flag")
    if flag is not None:
        flag.attrib.pop("autoreset", None)
        if not flag.attrib:
            root.find("option").remove(flag)
    destination.mkdir(parents=True)
    path = destination / "robot.xml"
    tree.write(path, encoding="unicode")
    parent_candidate = contract["candidate"]
    contract = copy.deepcopy(contract)
    contract.update(candidate=candidate, integrator="implicitfast", model_sha256=sha256(path),
        status="NATIVE_MODEL_REFERENCE_ONLY", training_release=False, optimizer_updates=0,
        upstream_baseline={"versions": VERSIONS, "parent_candidate": parent_candidate,
            "parent_model_sha256": sha256(parent_model),
            "parent_contract_sha256": sha256(parent_contract), "decimation": 1,
            "native_mj_step": True, "custom_constraint_callbacks": False,
            "collision_proxy_qualified": False, "source_qualified": False,
            "target_qualified": False,
            "native_runtime_autoreset_disabled": True,
            "warp_implicit_autoreset": False,
            "external_nonfinite_abort_required": True,
            "option_changes": ["implicitfast", "C-only autoreset flag omitted from Warp XML"]})
    write_json(destination / "contract.json", contract)
    return path, destination / "contract.json"


def build_task_proxy_reference(parent_model: Path, parent_contract: Path, source_root: Path,
                               destination: Path):
    """Freeze the independently checked 004 -> Warp compatibility candidate.

    MULTICCD is disabled; the explicit four-point/contact-row adapter remains
    mandatory. This exporter never grants source or learned-policy admission.
    """
    require_upstream_stack()
    contract = _read_native_contract(parent_model, parent_contract)
    if (contract.get("schema") != "goose_task_proxy_si_v1"
            or contract.get("candidate") != "goose_task_proxy_11_v1"
            or contract.get("runtime_revision") != "goose_task_proxy_be_contact_v1"
            or contract.get("contact_mapping", {}).get("method") != "whole_sole_native_manifold_backward_euler_v1"):
        raise ValueError("Frozen task-proxy parent required")
    source_root = source_root.resolve(strict=True)
    module = source_root / "src/sai_agent/goose/stage_one_gravity.py"
    if sha256(module) != contract["source_module_sha256"][module.name]:
        raise ValueError("Task-proxy nominal gravity module identity mismatch")
    for relative, digest in contract["asset_sha256"].items():
        asset = (parent_model.parent / relative).resolve(strict=True)
        if not asset.is_relative_to(parent_model.parent.resolve()) or sha256(asset) != digest:
            raise ValueError("Task-proxy asset identity mismatch")
    if destination.exists():
        raise FileExistsError("Preserve the previous task-proxy candidate")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    root.set("model", TASK_PROXY_CANDIDATE)
    option = root.find("option")
    if option.get("integrator") != "Euler" or option.get("timestep") != "0.02":
        raise ValueError("Task-proxy Euler/50Hz identity changed")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("multiccd", "disable")
    flag.attrib.pop("autoreset", None)  # Warp has no implicit numerical reset.
    destination.mkdir(parents=True)
    for relative in contract["asset_sha256"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_model.parent / relative, target)
    model_path = destination / "robot.xml"
    tree.write(model_path, encoding="unicode")
    contract = copy.deepcopy(contract)
    contract.update(candidate=TASK_PROXY_CANDIDATE, integrator="Euler", model_sha256=sha256(model_path),
        status="GPU_CONTACT_DEVELOPMENT_CANDIDATE", training_release=False, optimizer_updates=0,
        source_checkpoint={"source_root": str(source_root),
            "source_module_sha256": {"src/sai_agent/goose/stage_one_gravity.py": sha256(module)}},
        upstream_baseline={"versions": VERSIONS, "parent_candidate": "goose_task_proxy_11_v1",
            "parent_model_sha256": sha256(parent_model), "parent_contract_sha256": sha256(parent_contract),
            "contact_adapter_revision": TASK_PROXY_ADAPTER_REVISION,
            "contact_adapter_required": True, "decimation": 1,
            "option_changes": ["MULTICCD disabled", "C-only autoreset guard omitted"],
            "source_qualified": False, "target_qualified": False,
            "external_nonfinite_abort_required": True})
    contract_path = destination / "contract.json"
    write_json(contract_path, contract)
    return model_path, contract_path


def make_entity_cfg(model_path: Path, contract_path: Path):
    """Wrap native XML motors as efforts; the Goose target controller is separate."""
    require_upstream_stack()
    import mujoco
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    contract = _read_native_contract(model_path, contract_path)
    from .rigid_native import (CANDIDATES as RIGID_CANDIDATES, MJLAB_CANDIDATE,
        make_source_runtime as make_rigid_runtime)
    is_rigid = contract["candidate"] in RIGID_CANDIDATES
    from .speculative_contact import PREDICTION_CANDIDATES, REVISION as PREDICTION_REVISION, validate_prediction
    is_prediction = contract["candidate"] in PREDICTION_CANDIDATES
    is_proxy = contract["candidate"] == TASK_PROXY_CANDIDATE or is_prediction
    is_euler = contract["candidate"] == EULER_CANDIDATE or is_proxy
    expected_integrator = "Euler" if is_euler else "implicitfast"
    if (not is_euler and not is_rigid and contract["candidate"] not in NATIVE_PARENTS.values()
            or contract["integrator"] != expected_integrator):
        raise ValueError("Explicit native mjlab reference required")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if is_rigid:
        if contract["candidate"] != MJLAB_CANDIDATE:
            raise ValueError("Explicit rigid mjlab candidate without C-only flags required")
        make_rigid_runtime(model_path, contract_path)  # Full identity,0 integrals.
    if model.nu != 18 or model.opt.timestep != DT:
        raise ValueError("Native model does not expose the Goose 18 motors at 50 Hz")
    expected_enum = (mujoco.mjtIntegrator.mjINT_EULER if is_euler
                     else mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    if model.opt.integrator != expected_enum:
        raise ValueError("Compiled native integration profile differs from its contract")
    if is_proxy:
        baseline = contract.get("upstream_baseline", {})
        adapter_revision = PREDICTION_REVISION if is_prediction else TASK_PROXY_ADAPTER_REVISION
        if (baseline.get("contact_adapter_revision") != adapter_revision
                or baseline.get("contact_adapter_required") is not True
                or contract.get("runtime_revision") != "goose_task_proxy_be_contact_v1"
                or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
                or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
                or model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER)):
            raise ValueError("Task-proxy contact adapter/integration profile changed")
        if is_prediction:
            validate_prediction(model, contract)
    elif is_euler:
        flow = contract.get("native_integration_flow", {})
        required = {"revision": "goose_native_euler_eulerdamp_disabled_v1",
                    "dt_s": DT, "decimation": 1, "integrations_per_tick": 1,
                    "physical_dampers_enabled": True,
                    "implicit_joint_damping_integration_enabled": False}
        if (any(flow.get(key) != value for key, value in required.items())
                or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
                or model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER)):
            raise ValueError("Native Euler integration profile must retain physical damping")
    positions = {model.joint(i).name: float(model.qpos0[model.jnt_qposadr[i]])
                 for i in range(model.njnt) if model.jnt_type[i] in (2, 3)}
    positions.update({joint["name"]: joint["q_neutral_rad"] for joint in contract["joints"]})
    for joint in contract["passive_linkage_joints"]:
        positions[joint["name"]] = (positions[joint["mimic_joint"]] * joint["mimic_multiplier"]
                                    + joint["mimic_offset_rad"])

    def robot_spec():
        _read_native_contract(model_path, contract_path)
        spec = mujoco.MjSpec.from_file(str(model_path))
        # The task scene owns its plane, as in the MicroDuck/industry workflow.
        if is_prediction:
            for pair in list(spec.pairs):
                spec.delete(pair)
        spec.delete(spec.geom("ground"))
        return spec

    root_pos = model.qpos0[:3].copy()
    root_pos[2] += .002  # Existing declared cold-birth clearance, initial state only.
    motor_targets = tuple(model.joint(int(model.actuator_trnid[i, 0])).name
                          for i in range(model.nu))
    actuator_groups = (XmlActuatorCfg(target_names_expr=motor_targets, command_field="effort"),)
    if is_rigid:
        actuator_groups = (
            XmlActuatorCfg(target_names_expr=tuple(n for i, n in enumerate(motor_targets) if i != 5),
                command_field="position"),
            XmlActuatorCfg(target_names_expr=(motor_targets[5],), command_field="effort"))
    return EntityCfg(spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(root_pos), rot=tuple(model.qpos0[3:7]),
                                            joint_pos=positions),
        articulation=EntityArticulationInfoCfg(actuators=actuator_groups),
        sort_actuators=False, collisions=())
