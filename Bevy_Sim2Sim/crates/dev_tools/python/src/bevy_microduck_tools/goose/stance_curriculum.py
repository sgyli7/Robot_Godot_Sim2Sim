"""Frozen bent-stance initial conditions in native mjlab/RSL.

This profile loads an actual recorded state only at an explicit episode reset.
It is a development spawn, not proof of game cold birth or full hot migration.
The upstream loop, original 65/18 interface and effort controller are retained.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from .artifacts import JOINT_ORDER, sha256
from .batch_drive import _quat_matrix
from .foot_curriculum import add_foot_sensors, make_foot_course_cfg
from .mjlab_env import GooseDevelopmentEnv
from .source_training import completed_root_state, make_standing_cfg, sole_travel_exceeded

REVISION = "goose_bent_snapshot_standing_v1"
FORWARD_REVISION = "goose_bent_snapshot_native_forward_v1"


def read_stance(path: Path, digest: str, model_path: Path, contract_path: Path):
    if sha256(path) != digest:
        raise ValueError("Frozen bent stance identity changed")
    state = json.loads(path.read_text())
    if (state["revision"] != REVISION or state["joint_order"] != list(JOINT_ORDER)
            or state["model_sha256"] != sha256(model_path)
            or state["contract_sha256"] != sha256(contract_path)):
        raise ValueError("Bent stance must bind the same named plant and joint order")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    for field, length in (("qpos", model.nq), ("qvel", model.nv),
                          ("qacc_warmstart", model.nv), ("ctrl", model.nu)):
        value = np.asarray(state[field], dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError("Bent stance contains invalid physical state")
    if abs(np.linalg.norm(state["qpos"][3:7])-1.) > 1e-8:
        raise ValueError("Bent stance needs a unit free-root quaternion")
    drive = state["drive"]
    for field in ("actions", "target", "thermal", "last_tau"):
        value = np.asarray(drive[field], dtype=float)
        if value.shape != (18,) or not np.isfinite(value).all():
            raise ValueError("Bent stance contains invalid drive history")
    contract = json.loads(contract_path.read_text())
    ranges = np.asarray([j["range_rad"] for j in contract["joints"]])
    peaks = np.asarray([j["torque_peak_limit_nm"] for j in contract["joints"]])
    if (np.max(np.abs(drive["actions"])) > 1. or np.min(drive["thermal"]) < 0.
            or np.any(np.abs(drive["last_tau"]) > peaks+1e-12)
            or np.any(np.asarray(drive["target"]) < ranges[:, 0])
            or np.any(np.asarray(drive["target"]) > ranges[:, 1])
            or not np.isfinite(drive["phase"]) or not 0 <= drive["phase"] < 2*np.pi):
        raise ValueError("Bent stance violates the original drive limits")
    return state, model


def stance_height_reward(env, target_z):
    return torch.exp(-(completed_root_state(env)[0][:, 2]-target_z).square()/.0025)


def _apply_stance_initial_state(cfg, state, model):
    """Share one declared reset pose and standing-height target across tasks."""
    pose, velocity = np.asarray(state["qpos"]), np.asarray(state["qvel"])
    initial = cfg.scene.entities["robot"].init_state
    initial.pos, initial.rot = tuple(pose[:3]), tuple(pose[3:7])
    initial.lin_vel = tuple(velocity[:3])
    rotation = _quat_matrix(torch.tensor(pose[3:7]).reshape(1, 4))[0].numpy()
    initial.ang_vel = tuple(rotation@velocity[3:6])  # Native root API uses world angular velocity.
    initial.joint_pos = {model.joint(i).name: float(pose[model.jnt_qposadr[i]])
                         for i in range(model.njnt) if model.jnt_type[i] == 3}
    initial.joint_vel = {model.joint(i).name: float(velocity[model.jnt_dofadr[i]])
                         for i in range(model.njnt) if model.jnt_type[i] == 3}
    cfg.rewards["height"] = RewardTermCfg(func=stance_height_reward, weight=1.,
        params={"target_z": float(pose[2])})
    cfg.terminations["sole_travel"] = TerminationTermCfg(func=sole_travel_exceeded,
        params={"max_depth_m": .0015})
    return cfg


def make_bent_standing_cfg(model_path: Path, contract_path: Path, stance_path: Path,
                           stance_sha256: str, *, num_envs=16, seed=47,
                           episode_length_s=12.):
    state, model = read_stance(stance_path, stance_sha256, model_path, contract_path)
    cfg = make_standing_cfg(model_path, contract_path, num_envs=num_envs,
                           seed=seed, episode_length_s=episode_length_s)
    return add_foot_sensors(_apply_stance_initial_state(cfg, state, model), contract_path)


def make_bent_foot_course_cfg(model_path: Path, contract_path: Path, stance_path: Path,
                             stance_sha256: str, *, num_envs=16, seed=49,
                             episode_length_s=12.):
    """Reuse native forward commands and swing rewards from the held stance.

    The actual controller and 65/18 interface remain identical to standing.
    Only the task, declared initial conditions and height target are combined;
    the upstream command manager owns goals and the policy must create steps.
    """
    state, model = read_stance(stance_path, stance_sha256, model_path, contract_path)
    cfg = make_foot_course_cfg(model_path, contract_path, num_envs=num_envs,
                              seed=seed, episode_length_s=episode_length_s)
    return _apply_stance_initial_state(cfg, state, model)


class GooseBentStanceEnv(GooseDevelopmentEnv):
    """Seed physical/drive history once at public reset, before fresh observation."""

    def __init__(self, cfg, device, *, stance_path: Path, stance_sha256: str, **kwargs):
        action = cfg.actions["goose"]
        self._stance_path, self._stance_sha256 = stance_path, stance_sha256
        self._stance, _ = read_stance(stance_path, stance_sha256,
                                      action.model_path, action.contract_path)
        super().__init__(cfg, device, **kwargs)

    def reset(self, **kwargs):
        if sha256(self._stance_path) != self._stance_sha256:
            raise ValueError("Preserve reset: frozen stance identity changed")
        super().reset(**kwargs)
        ids = kwargs.get("env_ids")
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        action = self.action_manager.get_term("goose")
        data = self.sim.data
        expected = data.qpos.new_tensor(self._stance["qpos"])
        actual = data.qpos[ids].clone()
        actual[:, action.root_q:action.root_q+3] -= self.scene.env_origins[ids]
        if not torch.allclose(actual, expected.expand_as(actual), atol=1e-6, rtol=0):
            raise ValueError("Native reset did not load the declared bent stance")
        for field in ("qacc_warmstart", "ctrl"):
            getattr(data, field)[ids] = getattr(data, field).new_tensor(self._stance[field])
        for field in ("target", "actions", "thermal", "last_tau", "phase"):
            getattr(action.drive, field)[ids] = action.drive.target.new_tensor(self._stance["drive"][field])
        action._raw[ids] = action._raw.new_tensor(self._stance["drive"]["actions"])
        action._effort[ids] = action._effort.new_tensor(self._stance["drive"]["last_tau"])
        # Refresh sensors/observation at the reset boundary only; no integral.
        self.sim.forward()
        self.scene.update(dt=0.)
        self.sim.sense()
        # The parent's reset cached an observation before drive seeding.
        # Force a fresh public computation; this Actor group has no stacking.
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.extras


def initialize_stance_actor(runner, mean_action):
    """Fresh native policy starts at a physically held full target action.

    A zero final weight matrix gives this exact mean for every initial input;
    PPO learns feedback through the unchanged native MLP. No runtime residual
    helper or inverse kinematics is inserted, and Gaussian state stays intact.
    """
    if runner.alg.optimizer.state:
        raise ValueError("Stance mean initialization requires a fresh optimizer")
    actor = runner.alg.actor
    mean = torch.as_tensor(mean_action, dtype=torch.float32, device=runner.device)
    if actor.obs_dim != 65 or mean.shape != (18,) or not torch.isfinite(mean).all() or mean.abs().max() > 1:
        raise ValueError("Stance Actor must retain the original finite 65/18 interface")
    final = actor.mlp[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(mean)
