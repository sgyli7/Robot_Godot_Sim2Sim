"""Computer-only rigid feet and upstream native implicit position actuators.

Reuse the verified source's state, observation and target conventions. Native
MuJoCo owns contact and integration; no sole quadrature or constraint callback
is used. This is a separate controller/model, never a soft-sole qualification.
"""
from pathlib import Path
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .artifacts import DT, JOINT_ORDER, sha256, write_json

CANDIDATE = "goose_task_proxy_11_rigid_native_v1"
MJLAB_CANDIDATE = "goose_task_proxy_11_rigid_mjlab_v1"
CANDIDATES = (CANDIDATE, MJLAB_CANDIDATE)
REVISION = "goose_rigid_native_implicit_drive_v1"


def build_reference(parent_model, parent_contract, source_root, destination, *, upstream=False):
    parent_model, parent_contract, source_root, destination = map(
        Path, (parent_model, parent_contract, source_root, destination))
    contract = json.loads(parent_contract.read_text())
    if (contract["candidate"] != "goose_task_proxy_11_v1"
            or sha256(parent_model) != contract["model_sha256"]
            or contract["joint_order"] != list(JOINT_ORDER)):
        raise ValueError("Verified original task proxy required")
    if destination.exists():
        raise FileExistsError("Preserve existing rigid candidate")
    tree = ET.parse(parent_model)
    root = tree.getroot()
    candidate = MJLAB_CANDIDATE if upstream else CANDIDATE
    root.set("model", candidate)
    option = root.find("option")
    option.set("integrator", "implicitfast")
    flags = option.find("flag")
    flags.set("eulerdamp", "enable")
    flags.set("multiccd", "disable")
    if upstream:
        flags.attrib.pop("autoreset", None)  # Unsupported C-only flag in Warp.
    else:
        flags.set("autoreset", "disable")
    # Use the same ordinary rigid contact material as the body envelopes.
    # No positive discovery/rest margins, foundation springs or replacement
    # contact points. Geometry, friction, SI and self filters stay inherited.
    for geom in root.findall(".//geom"):
        if geom.get("name") in ("right_flexible_sole", "left_flexible_sole"):
            geom.set("solref", ".005 1")
            geom.set("solimp", ".95 .99 .001")
            geom.set("margin", "0")
            geom.set("priority", "0")
    actuator = root.find("actuator")
    for index, old in enumerate(list(actuator)):
        joint = contract["joints"][index]
        if index == 5:  # Actual four-bar input rotor stays a torque motor.
            continue
        old.tag = "position"
        old.set("kp", str(joint["kp_nm_rad"]))
        old.set("kv", str(joint["kd_nm_s_rad"]))
        old.set("forcelimited", "true")
        peak = joint["torque_peak_limit_nm"]
        old.set("forcerange", f"{-peak} {peak}")
        old.attrib.pop("ctrlrange", None)
        old.set("ctrllimited", "false")
    destination.mkdir(parents=True)
    for relative, digest in contract["asset_sha256"].items():
        src = parent_model.parent / relative
        if sha256(src) != digest:
            raise ValueError("Original geometry changed")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    model_path = destination / "robot.xml"
    tree.write(model_path, encoding="unicode")
    contract = copy.deepcopy(contract)
    contract.update(candidate=candidate, candidate_id=candidate,
        runtime_revision=REVISION, controller_revision=REVISION,
        source_runtime="bevy_microduck_tools.goose.rigid_native.make_source_runtime",
        integrator="implicitfast", model_sha256=sha256(model_path),
        status="COMPUTER_RIGID_NATIVE_DEVELOPMENT", training_release=False,
        source_checkpoint={"source_root": str(source_root.resolve()),
            "source_module_sha256": {"src/sai_agent/goose/stage_one_gravity.py":
                sha256(source_root / "src/sai_agent/goose/stage_one_gravity.py")}},
        contact_mapping={"method": "native_rigid_contact", "custom_callbacks": False},
        rigid_native={"revision": REVISION, "module_sha256": sha256(Path(__file__)),
            "foot_sensor_sites": [{"body": p["body"], "geom": p["geom"],
                "center_body_m": np.asarray(p["bottom_corners_body_m"], dtype=float).mean(0).tolist()}
                for p in contract["contact_mapping"]["ground_contact_quadrature"]],
            "source_root": str(source_root.resolve()),
            "parent_model": str(parent_model.resolve()),
            "parent_contract": str(parent_contract.resolve()),
            "parent_model_sha256": sha256(parent_model),
            "parent_contract_sha256": sha256(parent_contract),
            "parent_runtime": contract["source_runtime"],
            "position_actuators": 17, "beak_input_effort_actuators": 1,
            "native_contact_only": True, "integrations_per_tick": 1,
            "C_only_autoreset_flag_omitted_for_Warp": upstream,
            "hardware_qualification": False, "soft_sole_qualification": False})
    contract_path = destination / "contract.json"
    write_json(contract_path, contract)
    return model_path, contract_path


def make_source_runtime(model_path, contract_path):
    model_path, contract_path = Path(model_path), Path(contract_path)
    contract = json.loads(contract_path.read_text())
    cfg = contract["rigid_native"]
    if (contract["candidate"] not in CANDIDATES or cfg["revision"] != REVISION
            or cfg["module_sha256"] != sha256(Path(__file__))
            or contract["model_sha256"] != sha256(model_path)
            or sha256(Path(cfg["parent_model"])) != cfg["parent_model_sha256"]
            or sha256(Path(cfg["parent_contract"])) != cfg["parent_contract_sha256"]
            or contract["joint_order"] != list(JOINT_ORDER)
            or contract["contact_mapping"] != {
                "method": "native_rigid_contact", "custom_callbacks": False}
            or any(contract[k] != 1 for k in ("physics_steps_per_tick",
                "torque_updates_per_tick", "policy_calls_per_tick"))
            or any(contract[k] != DT for k in ("physics_dt_s", "policy_dt_s", "torque_dt_s"))):
        raise ValueError("Rigid model/controller/parent identity changed")
    for relative, digest in contract["asset_sha256"].items():
        if sha256(model_path.parent / relative) != digest:
            raise ValueError("Rigid candidate geometry changed")
    source = Path(cfg["source_root"]) / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from sai_agent.goose.task_proxy_runtime import TaskProxyRuntime
    # Constructor verifies every source module and original SI/asset hash.
    # This original instance is never stepped; reuse only its public state
    # conventions, reset, gravity and observations.
    base = TaskProxyRuntime(Path(cfg["parent_model"]), Path(cfg["parent_contract"]))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    # C evaluation must preserve failed states. Warp has no implicit reset.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    if (model.opt.timestep != DT or model.nu != 18 or model.nbody != 22
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            or model.npair != 0):
        raise ValueError("Native rigid timing/topology changed")

    class RigidRuntime(type(base)):
        def __init__(self):
            self.__dict__.update(base.__dict__)
            self.model, self.data, self.contract = model, mujoco.MjData(model), contract
            self.motor_vids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
            self.reset()

        def step(self, action):
            action = np.asarray(action, dtype=float)
            if action.shape != (18,) or not np.isfinite(action).all():
                raise ValueError("18 finite action targets required")
            action = np.clip(action, -1, 1)
            command = self.actions if self.delay else action
            desired = np.clip(self.neutral + self.scale * command,
                self.ranges[:, 0], self.ranges[:, 1])
            self.target += np.clip(desired - self.target, -self.speed * DT, self.speed * DT)
            q, qd = self.data.qpos[self.qidx], self.data.qvel[self.vidx]
            feed = np.zeros(18)
            feed[:5] = self.gravity(q[:6], self.data.qpos[3:7])
            cap = self.peak * self.strength * np.clip(1 - np.abs(qd)/(self.speed*1.3), 0, 1)
            cap = np.minimum(cap, np.where(self.thermal > (self.cont*self.strength)**2,
                self.cont, self.peak)*self.strength)
            # Conservative pre-Tick power cap independent of native feedback.
            bound = float(np.sum(cap * np.abs(self.data.qvel[self.motor_vids])))
            limit = self.contract["positive_mechanical_power_limit_w"]
            if bound > limit:
                cap *= limit / bound
            self.model.actuator_forcerange[:] = np.stack((-cap, cap), axis=1)
            self.data.ctrl[:] = self.target + feed/self.kp
            self.data.ctrl[5] = np.clip(self.kp[5]*(self.target[5]-q[5])-self.kd[5]*qd[5],
                -cap[5], cap[5])
            before = float(self.data.time)
            mujoco.mj_step(self.model, self.data)  # Exactly one native 20ms step.
            self.physics_integrations += 1
            self.controller_updates += 1
            self.last_tau = self.data.actuator_force.copy()
            self.thermal += DT/2*(self.last_tau**2-self.thermal)
            if (abs(self.data.time-before-DT) > 1e-12
                    or np.any(np.abs(self.last_tau) > cap+1e-9)
                    or not all(np.isfinite(getattr(self.data, n)).all()
                        for n in ("qpos", "qvel", "qacc", "ctrl", "actuator_force"))
                    or any(w.number for w in self.data.warning)):
                raise FloatingPointError("Native rigid integration/force/finite guard failed")
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self.age += 1
            self.phase = (self.phase+2*np.pi*self.contract["phase_frequency_hz"]*DT)%(2*np.pi)
            self.actions = action
            upright = float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])
            height = float(self.data.subtree_com[self.torso, 2])
            nonfoot = self._nonfoot_ground_contact()
            return self.observations(), {"failure": height < .18 or upright < .65,
                "height_m": height, "upright": upright, "nonfoot_ground_contact": nonfoot,
                "auto_reset": False, "physics_integrations": self.physics_integrations,
                "controller_updates": self.controller_updates, "time_s": float(self.data.time)}

    return RigidRuntime()
