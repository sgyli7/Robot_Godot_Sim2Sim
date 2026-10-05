"""Goose robot terms in upstream mjlab, with explicit development admission.

No trainer, integrator or generic environment is implemented here. The small
environment subclass guards the public reset boundary before upstream resets
physics; upstream owns the scene, managers, stepping, forward and observations.
This manual-reset development profile is not registered as a trainable task.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from .artifacts import DT, JOINT_ORDER, sha256
from .batch_drive import BatchedGooseDrive, BatchedRigidNativeDrive
from .mjlab_baseline import (
    TASK_PROXY_CANDIDATE, _read_native_contract, make_entity_cfg, require_upstream_stack)

REVISION = "goose_mjlab_development_env_v1"


def _require_development_timing(cfg):
    if cfg.decimation != 1 or cfg.sim.mujoco.timestep != DT or cfg.auto_reset:
        raise ValueError("Goose development requires 50 Hz, decimation=1, auto_reset=false")


@dataclass(kw_only=True)
class GooseActionCfg(ActionTermCfg):
    model_path: Path
    contract_path: Path
    contract_sha256: str
    motor_joint_names: tuple[str, ...]

    def build(self, env):
        return GooseAction(self, env)


@dataclass(kw_only=True)
class RigidGooseActionCfg(GooseActionCfg):
    def build(self, env):
        return RigidGooseAction(self, env)


class GooseAction(ActionTerm):
    """Original ordered position action -> effort-only XML actuator targets."""

    def __init__(self, cfg, env):
        _require_development_timing(env.cfg)
        if sha256(cfg.contract_path) != cfg.contract_sha256:
            raise ValueError("Goose development contract identity changed")
        super().__init__(cfg, env)
        contract = json.loads(cfg.contract_path.read_text())
        self.drive = BatchedGooseDrive(contract, self.num_envs, device=self.device)
        model = env.sim.mj_model
        prefix = cfg.entity_name+"/"
        self.qids = torch.tensor([int(model.joint(prefix+n).qposadr[0]) for n in JOINT_ORDER],
                                 device=self.device)
        self.vids = torch.tensor([int(model.joint(prefix+n).dofadr[0]) for n in JOINT_ORDER],
                                 device=self.device)
        self.torso = model.body(prefix+"torso").id
        root = int(model.body_jntadr[self.torso])
        if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Goose requires its original free torso")
        self.root_q = int(model.jnt_qposadr[root])
        self.root_v = int(model.jnt_dofadr[root])
        targets, names = self._entity.find_joints(cfg.motor_joint_names, preserve_order=True)
        if names != list(cfg.motor_joint_names) or len(names) != 18:
            raise ValueError("Goose motor target order mismatch")
        # The observable beak hinge and actual input rotor remain distinct.
        if names[5] != "beak_input_rotor":
            raise ValueError("The sixth Goose motor must drive beak_input_rotor")
        self.motor_ids = torch.tensor(targets, device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._raw = torch.zeros((self.num_envs, 18), dtype=torch.float64, device=self.device)
        self._effort = torch.zeros_like(self._raw)
        self._before_time = None
        self._applied = False

    @property
    def action_dim(self):
        return 18

    @property
    def raw_action(self):
        return self._raw

    @torch.no_grad()
    def process_actions(self, actions):
        _require_development_timing(self._env.cfg)
        d = self._env.sim.data
        self._effort.copy_(self.drive.prepare(actions, d.qpos[:, self.qids], d.qvel[:, self.vids],
                                             d.qpos[:, self.root_q+3:self.root_q+7]))
        # Keep constructor-owned buffers mutable across inference-mode changes.
        # Replacing them with clones created in inference mode breaks reset.
        self._raw.copy_(actions)
        self._before_time = d.time.clone()
        self._applied = False

    def apply_actions(self):
        if self._before_time is None or self._applied:
            raise RuntimeError("Exactly one Goose effort write per prepared Tick")
        self._entity.set_joint_effort_target(self._effort.to(torch.float32), joint_ids=self.motor_ids)
        self._applied = True

    def verify_actual_step(self):
        if self._before_time is None or not self._applied:
            raise RuntimeError("No applied Goose Tick to commit")
        d, wp_data = self._env.sim.data, self._env.sim.wp_data
        if not all(bool(torch.isfinite(getattr(d, n)).all()) for n in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl")):
            raise FloatingPointError("Nonfinite Goose state; preserve failed Tick without reset")
        if bool(((d.time-self._before_time-DT).abs() > 2e-6).any()):
            raise RuntimeError("Goose Tick must contain exactly one actual 20ms integration")
        if bool((d.nefc >= wp_data.njmax).any()) or int(d.nacon[0]) >= wp_data.naconmax:
            raise RuntimeError("Goose constraint/contact capacity exhausted; no reset")
        if adapter := self._env.contact_adapter:
            adapter.assert_valid()
        return (d.time-self._before_time-DT).abs()

    def commit_after_step(self):
        self.verify_actual_step()
        self.drive.commit()
        self._before_time = None

    def reset(self, env_ids=None):
        self.drive.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._effort[ids] = 0
        self.commands[ids] = 0


class RigidGooseAction(GooseAction):
    """Native XML position feedback plus the real beak effort transmission."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        contract = json.loads(cfg.contract_path.read_text())
        self.drive = BatchedRigidNativeDrive(contract, self.num_envs, device=self.device)
        model = env.sim.mj_model
        names = [model.joint(int(model.actuator_trnid[i, 0])).name
            for i in range(model.nu)]
        if model.nu != 18 or names != [cfg.entity_name+"/"+n for n in cfg.motor_joint_names]:
            raise ValueError("Rigid native actuator declaration order changed")
        self.motor_vids = torch.tensor(model.jnt_dofadr[model.actuator_trnid[:, 0]],
            device=self.device, dtype=torch.long)
        self.position_axes = torch.tensor([i for i in range(18) if i != 5], device=self.device)
        self._solver_force = torch.zeros_like(self._raw)
        self._forces_captured = False
        if "actuator_forcerange" not in env.sim.expanded_fields:
            env.sim.expand_model_fields(("actuator_forcerange",))

    @torch.no_grad()
    def process_actions(self, actions):
        _require_development_timing(self._env.cfg)
        d = self._env.sim.data
        self._effort.copy_(self.drive.prepare(actions, d.qpos[:, self.qids], d.qvel[:, self.vids],
            d.qpos[:, self.root_q+3:self.root_q+7], motor_velocity=d.qvel[:, self.motor_vids]))
        self._raw.copy_(actions)
        self._before_time = d.time.clone()
        self._applied = self._forces_captured = False

    def apply_actions(self):
        if self._before_time is None or self._applied:
            raise RuntimeError("Exactly one rigid command batch per Tick")
        self._env.sim.model.actuator_forcerange.copy_(torch.stack(
            (-self.drive.cap, self.drive.cap), dim=-1).to(torch.float32))
        self._entity.set_joint_position_target(self._effort[:, self.position_axes].float(),
            joint_ids=self.motor_ids[self.position_axes])
        self._entity.set_joint_effort_target(self._effort[:, 5:6].float(),
            joint_ids=self.motor_ids[5:6])
        self._applied = True

    def verify_actual_step(self):
        result = super().verify_actual_step()
        if not self._forces_captured:
            self._solver_force.copy_(self._env.sim.data.actuator_force)
            self._forces_captured = True
        return result

    def commit_after_step(self):
        self.verify_actual_step()
        self.drive.commit(self._solver_force)
        self._before_time = None

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._solver_force[ids] = 0
        limits = self.drive.peak.expand(self.num_envs, -1)
        self._env.sim.model.actuator_forcerange[ids] = torch.stack(
            (-limits[ids], limits[ids]), dim=-1).float()


def commit_history(env, env_ids=None):
    """Upstream step event runs after physics/forward, before actor observation."""
    del env_ids
    env.action_manager.get_term("goose").commit_after_step()


def physics_tick_metric(env):
    """Check the completed integral before forward can replace solved arrays."""
    return env.action_manager.get_term("goose").verify_actual_step()


def actor_observation(env):
    action = env.action_manager.get_term("goose")
    if action._before_time is not None:
        raise RuntimeError("Actor observation requires the successful Tick history commit")
    d = env.sim.data
    return action.drive.observations(d.qpos[:, action.qids], d.qvel[:, action.vids],
        d.qvel[:, action.root_v+3:action.root_v+6], d.xmat[:, action.torso], action.commands)


class GooseDevelopmentEnv(ManagerBasedRlEnv):
    """Use the upstream loop, guarding failed-state reset before mutation."""

    def __init__(self, cfg, device, **kwargs):
        require_upstream_stack()
        _require_development_timing(cfg)
        action = cfg.actions.get("goose")
        if not isinstance(action, GooseActionCfg) or sha256(action.contract_path) != action.contract_sha256:
            raise ValueError("Goose development contract identity changed")
        metric = cfg.metrics.get("goose_dt_error_s")
        event = cfg.events.get("commit_history")
        if (len(cfg.actions) != 1 or metric is None or not metric.per_substep
                or metric.func is not physics_tick_metric or event is None
                or event.func is not commit_history or event.mode != "step"):
            raise ValueError("Goose requires its pre-forward guard and post-step commit")
        contract = _read_native_contract(action.model_path, action.contract_path)
        if (cfg.sim.mujoco.integrator != contract["integrator"].lower()
                or contract["integrator"] == "Euler" and (
                    "eulerdamp" not in cfg.sim.mujoco.disableflags
                    or "damper" in cfg.sim.mujoco.disableflags)):
            raise ValueError("Goose development integration profile changed")
        self.contact_adapter = None
        super().__init__(cfg, device, **kwargs)
        from .speculative_contact import PREDICTION_CANDIDATES
        if contract["candidate"] == TASK_PROXY_CANDIDATE or contract["candidate"] in PREDICTION_CANDIDATES:
            from .task_proxy_contact import FrozenContactAdapter
            from .speculative_contact_gpu import SpeculativeContactAdapter
            import warp as wp

            if self.sim.wp_model.callback.control is not None:
                raise ValueError("Task-proxy contact seam already occupied")
            with wp.ScopedDevice(self.sim.wp_device):
                adapter_type = (SpeculativeContactAdapter if contract["candidate"] in PREDICTION_CANDIDATES
                                else FrozenContactAdapter)
                self.contact_adapter = adapter_type(self.sim.wp_model, self.sim.wp_data,
                    self.sim.mj_model, contract, entity_prefix=action.entity_name+"/")
                self.sim.wp_model.callback.control = self._apply_task_proxy_contact
                # Public mjlab graphs must capture the new public Warp callback.
                self.sim.create_graph()

    def _apply_task_proxy_contact(self, model, data):
        """Native forward/step1 contact seam; no torque update or integration."""
        if model is not self.sim.wp_model or data is not self.sim.wp_data:
            raise RuntimeError("Unexpected task-proxy contact callback world")
        self.contact_adapter.apply()

    def reset(self, **kwargs):
        action = self.action_manager.get_term("goose")
        if action._before_time is not None:
            raise RuntimeError("Preserve the pending/failed Goose Tick before resetting physics")
        return super().reset(**kwargs)


def make_development_env_cfg(model_path: Path, contract_path: Path, *, num_envs=2,
                             development_initial_qpos=None):
    """Native scene/manager config; an explicit loaded dev state is optional.

    The default declared +2mm cold birth is preserved. Passing a frozen loaded
    qpos initializes the entity only, and never replaces its qualification.
    No reward or PPO configuration is attached to this integration profile.
    """
    require_upstream_stack()
    contract = _read_native_contract(model_path, contract_path)
    native = mujoco.MjModel.from_xml_path(str(model_path))
    entity = make_entity_cfg(model_path, contract_path)
    if development_initial_qpos is not None:
        pose = np.asarray(development_initial_qpos, dtype=float)
        if pose.shape != (native.nq,) or not np.isfinite(pose).all():
            raise ValueError("Frozen development qpos must match the native model")
        entity.init_state.pos = tuple(pose[:3])
        entity.init_state.rot = tuple(pose[3:7])
        entity.init_state.joint_pos = {native.joint(i).name: float(pose[native.jnt_qposadr[i]])
            for i in range(native.njnt) if native.jnt_type[i] in (2, 3)}
    motors = tuple(native.joint(int(native.actuator_trnid[i, 0])).name for i in range(native.nu))
    ground = native.geom("ground").id

    def add_native_ground(spec):
        # MjSpec.attach does not inherit a child's mjOption. Copy the native
        # option record before compilation; MujocoCfg then applies the same
        # values for the fields it owns, without losing flags/other options.
        for field in dir(native.opt):
            if not field.startswith("_") and not callable(value := getattr(native.opt, field)):
                setattr(spec.option, field, value)
        spec.memory = native.narena
        # Preserve the original plane's physical properties, rather than using
        # terrain defaults that could silently change friction/masks/materials.
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=native.geom_pos[ground], quat=native.geom_quat[ground], size=native.geom_size[ground],
            contype=int(native.geom_contype[ground]), conaffinity=int(native.geom_conaffinity[ground]),
            condim=int(native.geom_condim[ground]), friction=native.geom_friction[ground],
            solref=native.geom_solref[ground], solimp=native.geom_solimp[ground],
            margin=float(native.geom_margin[ground]), gap=float(native.geom_gap[ground]),
            priority=int(native.geom_priority[ground]), solmix=float(native.geom_solmix[ground]))
        # Scene-owned ground pairs keep the frozen explicit material mixing.
        # Entity attachment cannot retain references to the deleted child plane.
        for i in range(native.npair):
            a, b = int(native.pair_geom1[i]), int(native.pair_geom2[i])
            if ground not in (a, b):
                raise ValueError("Only explicit prediction ground pairs are supported")
            gid = b if a == ground else a
            spec.add_pair(name=native.pair(i).name, geomname1="ground",
                geomname2="robot/"+native.geom(gid).name,
                condim=int(native.pair_dim[i]), solref=native.pair_solref[i],
                solreffriction=native.pair_solreffriction[i], solimp=native.pair_solimp[i],
                margin=float(native.pair_margin[i]), gap=float(native.pair_gap[i]),
                friction=native.pair_friction[i])

    opt = native.opt
    sim_options = MujocoCfg(timestep=DT, integrator=contract["integrator"].lower(), impratio=float(opt.impratio),
        cone="elliptic" if int(opt.cone) else "pyramidal",
        jacobian={0: "dense", 1: "sparse", 2: "auto"}[int(opt.jacobian)],
        solver={0: "pgs", 1: "cg", 2: "newton"}[int(opt.solver)],
        iterations=int(opt.iterations), tolerance=float(opt.tolerance),
        ls_iterations=int(opt.ls_iterations), ls_tolerance=float(opt.ls_tolerance),
        ccd_iterations=int(opt.ccd_iterations), gravity=tuple(opt.gravity),
        disableflags=tuple(name.removeprefix("mjDSBL_").lower() for name in dir(mujoco.mjtDisableBit)
            if name.startswith("mjDSBL_") and int(opt.disableflags) & int(getattr(mujoco.mjtDisableBit, name))),
        enableflags=tuple(name.removeprefix("mjENBL_").lower() for name in dir(mujoco.mjtEnableBit)
            if name.startswith("mjENBL_") and int(opt.enableflags) & int(getattr(mujoco.mjtEnableBit, name))))
    from .rigid_native import CANDIDATES as RIGID_CANDIDATES
    action_type = RigidGooseActionCfg if contract["candidate"] in RIGID_CANDIDATES else GooseActionCfg
    return ManagerBasedRlEnvCfg(scene=SceneCfg(num_envs=num_envs, entities={"robot": entity},
        terrain=None, spec_fn=add_native_ground),
        actions={"goose": action_type(entity_name="robot", model_path=model_path, contract_path=contract_path,
            contract_sha256=sha256(contract_path), motor_joint_names=motors)},
        observations={"actor": ObservationGroupCfg(terms={"base": ObservationTermCfg(func=actor_observation)},
            enable_corruption=False, concatenate_terms=True)},
        events={"reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
                "commit_history": EventTermCfg(func=commit_history, mode="step")},
        metrics={"goose_dt_error_s": MetricsTermCfg(func=physics_tick_metric, per_substep=True)},
        sim=SimulationCfg(mujoco=sim_options, nconmax=128, njmax=512),
        decimation=1, auto_reset=False, episode_length_s=0., seed=17)
