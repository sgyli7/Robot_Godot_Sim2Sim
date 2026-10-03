"""Goose-only single-integration 50 Hz CPU source runtime, without training.

The previous checkpoint environment cannot be monkeypatched to support this
contract: its inner torque loop and its locomotion-only terminations differ.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import mujoco

from .artifacts import DT, JOINT_ORDER, sha256
from .task import TaskGoal, invalid_goal_extension

RUNTIME_REVISION = "goose_source_canonical_contacts_v3"


class GooseSourceRuntime:
    def __init__(self, model_path: Path, contract_path: Path, *, skill="locomotion", seed=17):
        if skill not in ("locomotion", "recovery", "pickup"):
            raise ValueError("Unknown Goose skill")
        self.skill = skill
        self.contract = json.loads(contract_path.read_text())
        if self.contract.get("native_discrete") is not None and type(self) is GooseSourceRuntime:
            raise ValueError("Native discrete contract requires its explicit versioned runtime")
        if self.contract.get("numerical_metric") is not None and type(self) is GooseSourceRuntime:
            raise ValueError("Numerical metric requires the explicit experimental runtime")
        if any(self.contract[k] != DT for k in ("physics_dt_s", "torque_dt_s", "policy_dt_s")):
            raise ValueError("Goose requires one .02s physics/torque/policy update")
        if self.contract.get("physics_steps_per_tick") != 1 or self.contract["joint_order"] != list(JOINT_ORDER):
            raise ValueError("Incorrect Goose integration count or axis order")
        if sha256(model_path) != self.contract["model_sha256"]:
            raise ValueError("Model/contract hash mismatch")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        if self.model.opt.timestep != DT or self.model.nu != 18:
            raise ValueError("Model timestep/action mismatch")
        # Preserve a failed state for the independent evaluator. MuJoCo's
        # native numerical recovery must not silently reset a failed rollout.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        self.names = list(JOINT_ORDER)
        joints = self.contract["joints"]
        self.qidx = np.array([int(self.model.joint(n).qposadr[0]) for n in self.names])
        self.vidx = np.array([int(self.model.joint(n).dofadr[0]) for n in self.names])
        self.kp = np.array([j["kp_nm_rad"] for j in joints])
        self.kd = np.array([j["kd_nm_s_rad"] for j in joints])
        self.peak = np.array([j["torque_peak_limit_nm"] for j in joints])
        self.cont = np.array([j["continuous_design_limit_nm"] for j in joints])
        self.speed = np.array([j["speed_limit_rad_s"] for j in joints])
        self.scale = np.array([j["action_scale_rad"] for j in joints])
        self.neutral = np.array([j["q_neutral_rad"] for j in joints])
        self.ranges = np.array([j["range_rad"] for j in joints])
        self.torso = self.model.body("torso").id
        self.ground = self.model.geom("ground").id
        self.foot_geoms = {self.model.geom(p["name"]).id for p in self.contract["passive_contacts"]}
        self.foot_geoms.update(self.model.geom(g["name"]).id for g in self.contract["collision_geometries"]
                               if g["part"] in ("right_flexible_sole", "left_flexible_sole"))
        module_path = Path(self.contract["source_checkpoint"]["source_root"]) / "src/sai_agent/goose/stage_one_gravity.py"
        expected = self.contract["source_checkpoint"]["source_module_sha256"]["src/sai_agent/goose/stage_one_gravity.py"]
        if sha256(module_path) != expected:
            raise ValueError("Nominal gravity source identity mismatch")
        module = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("goose_nominal_gravity", module_path))
        spec.loader.exec_module(module)
        self.gravity = module.NominalNeckGravity(self.contract)
        self.rng = np.random.default_rng(seed)
        self.nominal = {name: getattr(self.model, name).copy() for name in ("body_mass", "body_inertia", "body_ipos", "geom_friction")}
        self.goal = None
        self.pickup_stage = 0
        self.commands = np.zeros(3)
        self.reset()

    def reset(self, *, randomize=False):
        m, d = self.model, self.data
        for name, values in self.nominal.items():
            getattr(m, name)[:] = values
        self.strength = 1.0
        self.delay = 0
        if randomize:
            for body in self.contract["bodies"]:
                bid = m.body(body["name"]).id
                factor = self.rng.uniform(1 - body["mass_relative_design_uncertainty"], 1 + body["mass_relative_design_uncertainty"])
                m.body_mass[bid] *= factor
                m.body_inertia[bid] *= factor * self.rng.uniform(*body["inertia_multiplier_range"])
                m.body_ipos[bid] += self.rng.uniform(-body["com_randomization_m"], body["com_randomization_m"], 3)
            m.geom_friction[:, 0] = self.rng.uniform(*self.contract["foot_friction_range"])
            self.strength = self.rng.uniform(*self.contract["strength_multiplier_range"])
            self.delay = self.rng.integers(*[self.contract["latency_policy_steps"][0], self.contract["latency_policy_steps"][1] + 1])
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        d.qpos[2] += .002
        d.qpos[self.qidx] = self.neutral
        for joint in self.contract["passive_linkage_joints"]:
            source, target = m.joint(joint["mimic_joint"]), m.joint(joint["name"])
            d.qpos[int(target.qposadr[0])] = joint["mimic_multiplier"] * d.qpos[int(source.qposadr[0])] + joint["mimic_offset_rad"]
        mujoco.mj_forward(m, d)
        self.actions = np.zeros(18)
        self.target = self.neutral.copy()
        self.thermal = np.zeros(18)
        self.last_tau = np.zeros(18)
        self.phase = 0.0
        self.age = 0
        self.physics_integrations = 0
        self.controller_updates = 0
        self.nominal_com_height = float(d.subtree_com[self.torso, 2])
        return self.observations()

    def observations(self):
        d = self.data
        rotation = d.xmat[self.torso].reshape(3, 3)
        base = np.r_[d.qvel[3:6] * .25, rotation.T @ [0., 0., -1.], self.commands,
                     d.qpos[self.qidx] - self.neutral, d.qvel[self.vidx] * .1,
                     self.actions, np.sin(self.phase), np.cos(self.phase)].astype(np.float32)
        base = np.clip(base, -20, 20)
        if self.skill == "pickup":
            if self.goal is None:
                extension = invalid_goal_extension(self.pickup_stage)
            elif isinstance(self.goal, TaskGoal):
                extension = self.goal.actor_extension(d.xpos[self.torso], d.xquat[self.torso], rotation, self.pickup_stage)
            else:
                raise ValueError("Pickup requires a typed TaskGoal")
            base = np.r_[base, extension]
        if base.shape != ((82 if self.skill == "pickup" else 65),) or not np.isfinite(base).all():
            raise FloatingPointError("Nonfinite/wrong-size Goose observation")
        return base

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError("Goose action must be 18 finite values")
        action = np.clip(action, -1, 1)
        command = self.actions if self.delay else action
        desired = np.clip(self.neutral + self.scale * command, self.ranges[:, 0], self.ranges[:, 1])
        self.target += np.clip(desired - self.target, -self.speed * DT, self.speed * DT)
        q, qd = self.data.qpos[self.qidx], self.data.qvel[self.vidx]
        torque = self.kp * (self.target - q) - self.kd * qd
        torque[:5] += self.gravity(q[:6], self.data.qpos[3:7])
        cap = self.peak * self.strength * np.clip(1 - np.abs(qd) / (self.speed * 1.3), 0, 1)
        cap = np.minimum(cap, np.where(self.thermal > (self.cont * self.strength) ** 2, self.cont, self.peak) * self.strength)
        torque = np.clip(torque, -cap, cap)
        positive = torque * qd > 0
        power = float(np.sum(torque[positive] * qd[positive]))
        if power > self.contract["positive_mechanical_power_limit_w"]:
            torque[positive] *= self.contract["positive_mechanical_power_limit_w"] / power
        self.thermal += DT / 2 * (torque ** 2 - self.thermal)
        self.data.ctrl[:] = torque
        self.last_tau = torque.copy()
        before = float(self.data.time)
        # No nstep argument and no inner physics/torque loop.
        self._integrate()
        self.physics_integrations += 1
        self.controller_updates += 1
        if abs(self.data.time - before - DT) > 1e-12:
            raise RuntimeError("Integration did not advance exactly one 20ms tick")
        finite = all(np.isfinite(getattr(self.data, name)).all() for name in ("qpos", "qvel", "qacc", "ctrl"))
        if not finite or any(w.number for w in self.data.warning):
            raise FloatingPointError("Nonfinite Goose state or native solver warning; no automatic reset")
        # mj_step leaves pose-derived arrays at the pre-integration state.
        # Refresh kinematics and COM only: no extra integration, collision
        # detection, constraint solve, or replacement of this Tick's forces.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.age += 1
        self.phase = (self.phase + 2 * np.pi * self.contract["phase_frequency_hz"] * DT) % (2 * np.pi)
        self.actions = action
        upright = float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])
        height = float(self.data.subtree_com[self.torso, 2])
        nonfoot = self._nonfoot_ground_contact()
        # Recovery explicitly permits body contact and low COM. No auto reset;
        # a scorer must observe stable recovery and continuation separately.
        failure = False if self.skill in ("recovery", "pickup") else bool(height < .18 or upright < .65 or nonfoot)
        return self.observations(), {"failure": failure, "height_m": height, "upright": upright,
                                     "nonfoot_ground_contact": nonfoot, "auto_reset": False,
                                     "task_goal_invalid": self.skill == "pickup" and (self.goal is None or not self.goal.valid),
                                     "recovery_requested": upright < .65,
                                     "physics_integrations": self.physics_integrations,
                                     "controller_updates": self.controller_updates, "time_s": float(self.data.time)}

    def _integrate(self):
        """Default native implicit step; experimental subclasses must be explicit."""
        mujoco.mj_step(self.model, self.data)

    def _nonfoot_ground_contact(self):
        # Warp's public host bridge populates canonical geom[2]. The legacy
        # geom1/geom2 members can remain zero in MuJoCo 3.10 bridge results.
        return any(self.ground in c.geom and
                   int(c.geom[1] if c.geom[0] == self.ground else c.geom[0])
                   not in self.foot_geoms for c in self.data.contact)
