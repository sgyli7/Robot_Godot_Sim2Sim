"""Native MJCF adoption for upstream mjlab; no replacement training machinery.

The reference preserves the inherited plant and collision geometry. It is a
model integration artifact, not physical admission or a registered RL task.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from .artifacts import CANDIDATES, DT, JOINT_ORDER, sha256, write_json

CANDIDATE = "goose_460_condensed_mjlab50_reference_v1"
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
    if contract["candidate"] != CANDIDATES[1]:
        raise ValueError("The frozen condensed native parent is required")
    if destination.exists():
        raise FileExistsError("Preserve the previous native reference")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    root.set("model", CANDIDATE)
    # implicitfast is supported by the pinned mjlab/Warp stack. No CPU callback.
    root.find("option").set("integrator", "implicitfast")
    flag = root.find("./option/flag")
    if flag is None:
        flag = ET.SubElement(root.find("option"), "flag")
    flag.set("autoreset", "disable")
    destination.mkdir(parents=True)
    path = destination / "robot.xml"
    tree.write(path, encoding="unicode")
    contract = copy.deepcopy(contract)
    contract.update(candidate=CANDIDATE, integrator="implicitfast", model_sha256=sha256(path),
        status="NATIVE_MODEL_REFERENCE_ONLY", training_release=False, optimizer_updates=0,
        upstream_baseline={"versions": VERSIONS, "parent_model_sha256": sha256(parent_model),
            "parent_contract_sha256": sha256(parent_contract), "decimation": 1,
            "native_mj_step": True, "custom_constraint_callbacks": False,
            "collision_proxy_qualified": False, "source_qualified": False,
            "target_qualified": False, "option_changes": ["implicitfast", "autoreset disabled"]})
    write_json(destination / "contract.json", contract)
    return path, destination / "contract.json"


def make_entity_cfg(model_path: Path, contract_path: Path):
    """Wrap native XML motors as efforts; the Goose target controller is separate."""
    require_upstream_stack()
    import mujoco
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    contract = _read_native_contract(model_path, contract_path)
    if contract["candidate"] != CANDIDATE or contract["integrator"] != "implicitfast":
        raise ValueError("Explicit native mjlab reference required")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nu != 18 or model.opt.timestep != DT:
        raise ValueError("Native model does not expose the Goose 18 motors at 50 Hz")
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
        spec.delete(spec.geom("ground"))
        return spec

    root_pos = model.qpos0[:3].copy()
    root_pos[2] += .002  # Existing declared cold-birth clearance, initial state only.
    motor_targets = tuple(model.joint(int(model.actuator_trnid[i, 0])).name
                          for i in range(model.nu))
    return EntityCfg(spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(root_pos), rot=tuple(model.qpos0[3:7]),
                                            joint_pos=positions),
        articulation=EntityArticulationInfoCfg(actuators=(
            XmlActuatorCfg(target_names_expr=motor_targets, command_field="effort"),)),
        sort_actuators=False, collisions=())
