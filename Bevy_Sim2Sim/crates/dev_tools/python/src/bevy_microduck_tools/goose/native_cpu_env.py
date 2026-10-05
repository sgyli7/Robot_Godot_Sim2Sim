"""Native MuJoCo task in Gymnasium; physics stays on CPU at one20ms step.

Gymnasium owns vector workers/reset scheduling and RSL owns PPO. This module
only supplies the robot task and the existing65/18 interface. GPU physics and
game qualifications are separate from this bounded native source pilot.
"""
from pathlib import Path
import json

import gymnasium as gym
import mujoco
import numpy as np

from .speculative_contact import NATIVE_CPU_CANDIDATES, make_source_runtime
from .rigid_native import (
    CANDIDATE as RIGID_CANDIDATE, MJLAB_CANDIDATES,
    make_source_runtime as make_rigid_runtime)
from .native_geometry import collision_geom_vertices

REVISION = "goose_native_cpu_front_recovery_pilot_v1"
RIGID_CPU_CANDIDATES = (RIGID_CANDIDATE, *MJLAB_CANDIDATES)


class NativeGooseRecoveryEnv(gym.Env):
    metadata = {"render_modes": []}
    contact_limit_terminates_episode = False

    def __init__(self, model_path, contract_path, initial_qpos, *,
                 episode_ticks=700, trajectory_directory=None, contact_limit_m=.005):
        candidate = json.loads(Path(contract_path).read_text())["candidate"]
        factory = make_rigid_runtime if candidate in RIGID_CPU_CANDIDATES else make_source_runtime
        self.runtime = factory(Path(model_path), Path(contract_path))
        rt = self.runtime
        if rt.contract["candidate"] not in (*NATIVE_CPU_CANDIDATES, *RIGID_CPU_CANDIDATES):
            raise ValueError("Native CPU pilot requires its named native contact source")
        self.initial_qpos = np.asarray(initial_qpos, dtype=float).copy()
        if self.initial_qpos.shape != (rt.model.nq,) or not np.isfinite(self.initial_qpos).all():
            raise ValueError("Frozen recovery birth must match the native model")
        self.nominal_com_height = float(rt.data.subtree_com[rt.torso, 2])
        self.supports = {g: collision_geom_vertices(rt.model, g)
            for g in range(rt.model.ngeom)
            if rt.model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_MESH,
                mujoco.mjtGeom.mjGEOM_BOX)}
        if len(self.supports) != 11 or rt.model.opt.timestep != .02:
            raise ValueError("Native CPU physical identity changed")
        self.motor_vids = rt.model.jnt_dofadr[rt.model.actuator_trnid[:, 0]]
        if not 0 < contact_limit_m <= .05:
            raise ValueError("Native CPU episode contact limit must be in (0,50mm]")
        self.contact_limit_m = float(contact_limit_m)
        self.episode_ticks = int(episode_ticks)
        if self.episode_ticks <= 0:
            raise ValueError("Finite positive episode length required")
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (65,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (18,), np.float32)
        self.total_integrations = 0
        self.episodes = 0
        self.records = []
        self.directory = Path(trajectory_directory) if trajectory_directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def material_z(self):
        m, d = self.runtime.model, self.runtime.data
        return np.array([(v @ d.geom_xmat[g].reshape(3, 3).T
            + d.geom_xpos[g])[:, 2].min() for g, v in self.supports.items()])

    def critic_motion(self):
        rt = self.runtime
        rotation = rt.data.xmat[rt.torso].reshape(3, 3)
        return np.r_[rotation.T @ rt.data.qvel[:3], rt.data.qpos[2]].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._save_episode()
        rt = self.runtime
        rt.reset()
        # A declared cold birth only, with all original drive/history reset.
        rt.data.qpos[:] = self.initial_qpos
        rt.data.qvel[:] = 0.
        mujoco.mj_forward(rt.model, rt.data)
        self.records = []
        self.episodes += 1
        assert rt.physics_integrations == 0 and rt.data.time == 0.
        return rt.observations().astype(np.float32), {
            "reset_integrations": 0, "critic_motion": self.critic_motion()}

    def step(self, action):
        rt = self.runtime
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
            raise ValueError("Original bounded18-axis action required")
        previous = rt.actions.copy()
        before = rt.data.time
        observations, info = rt.step(action)
        self.total_integrations += 1
        z = self.material_z()
        up = float(info["upright"])
        height = float(info["height_m"] / self.nominal_com_height)
        speed = rt.data.qvel[:3]
        angular = rt.data.qvel[3:6]
        stable = float(up >= .95 and height >= .85) * np.exp(
            -(speed @ speed)/.04**2 - (angular @ angular)/.3**2)
        power = float(np.maximum(rt.last_tau * rt.data.qvel[self.motor_vids], 0).sum())
        reward = .02*(4.*.5*(up+1.) + 2.*np.clip(height, 0., 1.) + 3.*stable
            - .01*np.square(action-previous).mean() - .001*power)
        row = dict(qpos=rt.data.qpos.copy(), qvel=rt.data.qvel.copy(),
            qacc_warmstart=rt.data.qacc_warmstart.copy(), torque=rt.last_tau.copy(),
            target=rt.target.copy(), thermal=rt.thermal.copy(), action=action.copy(),
            observation=observations.copy(), min_collision_z=z, time=rt.data.time,
            upright=up, height_ratio=height, reward=reward)
        self.records.append(row)
        depth_failed = bool(-z.min() > self.contact_limit_m)
        if (abs(rt.data.time-before-.02) > 1e-10 or not np.isfinite(z).all()
                or depth_failed and not self.contact_limit_terminates_episode):
            self._save_episode(failed=True)
            raise RuntimeError("Native CPU pilot failed the frozen20ms/5mm physical guard")
        timeout = rt.physics_integrations >= self.episode_ticks
        return observations.astype(np.float32), float(reward), False, timeout, {
            "actual_integrations": 1, "total_integrations": self.total_integrations,
            "max_penetration_m": float(max(0., -z.min())), "upright": up,
            "height_ratio": height, "episode_tick": rt.physics_integrations,
            "contact_limit_failure": depth_failed, "critic_motion": self.critic_motion()}

    def _save_episode(self, *, failed=False):
        if self.directory and self.records:
            name = f"episode_{self.episodes:06d}" + ("_failed" if failed else "")
            path = self.directory / (name+".npz")
            if path.exists():
                raise FileExistsError("Preserve the previously saved native episode")
            np.savez_compressed(path, **{k: np.array([r[k] for r in self.records])
                for k in self.records[0]})
        self.records = []

    def close(self):
        self._save_episode()


class NativeGooseStandingEnv(NativeGooseRecoveryEnv):
    """Standing/velocity task using the same rigid plant and original65 slots."""
    contact_limit_terminates_episode = True

    def __init__(self, *args, command=(0., 0., 0.), **kwargs):
        super().__init__(*args, **kwargs)
        if self.runtime.contract["candidate"] not in RIGID_CPU_CANDIDATES:
            raise ValueError("Computer standing task requires its rigid native profile")
        self.command = np.asarray(command, dtype=float)
        if self.command.shape != (3,) or not np.isfinite(self.command).all():
            raise ValueError("Three finite base-frame velocity commands required")
        self.initial_xy = self.initial_qpos[:2].copy()

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self.runtime.commands[:] = self.command
        return self.runtime.observations().astype(np.float32), info

    def step(self, action):
        previous = self.runtime.actions.copy()
        obs, _, _, timeout, info = super().step(action)
        rt = self.runtime
        rotation = rt.data.xmat[rt.torso].reshape(3, 3)
        velocity = rotation.T @ rt.data.qvel[:3]
        angular = rt.data.qvel[3:6]  # Native free-joint angular velocity is local.
        tracking = np.exp(-np.square(velocity[:2]-self.command[:2]).sum()/.04)
        turning = np.exp(-(angular[2]-self.command[2])**2/.25)
        tilt = float(rotation[2, :2] @ rotation[2, :2])
        height = info["height_ratio"]
        rate = float(np.square(np.asarray(action)-previous).mean())
        power = float(np.maximum(rt.last_tau*rt.data.qvel[self.motor_vids], 0).sum())
        reward = .02*(2.*tracking + turning + 2.*np.clip(height, 0, 1)
            - 2.*tilt - .1*rate - .001*power)
        terminated = bool(info["upright"] < .65 or height < .65
            or info["contact_limit_failure"])
        if terminated:
            reward -= 1.
        self.records[-1]["reward"] = reward
        self.records[-1]["terminated"] = terminated
        if terminated:
            self._save_episode(failed=True)
        return obs, float(reward), terminated, timeout, info


class NativeGooseRslEnv:
    """Thin RSL tensor adapter for Gymnasium's standard AsyncVectorEnv."""

    def __init__(self, env, *, device="cpu", seed=53, privileged_critic=False):
        import torch
        from tensordict import TensorDict
        if env.autoreset_mode != gym.vector.AutoresetMode.SAME_STEP:
            raise ValueError("Same-step reset prevents a skipped integration nextTick")
        self.env, self.device = env, device
        self.num_envs, self.num_actions = env.num_envs, 18
        lengths = env.call("episode_ticks")
        if not lengths or len(set(lengths)) != 1:
            raise ValueError("RSL requires a common declared native episode length")
        self.max_episode_length = int(lengths[0])
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.cfg = {"task": REVISION, "physics_device": "cpu", "physics_dt_s": .02,
            "decimation": 1, "actor": 65, "actions": 18}
        # Successful vector returns only. After worker failure, saved worker
        # trajectories are authoritative for integrations in the failed call.
        self.real_integrations = 0
        self.common_step_counter = 0
        self.actor_batches = 0
        self.privileged_critic = privileged_critic
        obs, info = env.reset(seed=seed)
        self.obs = self._observations(obs, info)

    def _observations(self, obs, info):
        import torch
        from tensordict import TensorDict
        actor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        groups = {"actor": actor}
        if self.privileged_critic:
            motion = torch.as_tensor(np.asarray(info["critic_motion"]),
                dtype=torch.float32, device=self.device)
            if motion.shape != (self.num_envs, 4) or not torch.isfinite(motion).all():
                raise ValueError("Native privileged Critic requires four finite motion values")
            groups["critic"] = torch.cat((actor, motion), dim=-1)
        return TensorDict(groups, [self.num_envs])

    def get_observations(self):
        return self.obs

    @property
    def unwrapped(self):
        return self

    def step(self, actions):
        import torch
        from tensordict import TensorDict
        obs, reward, terminated, timeout, info = self.env.step(
            actions.detach().cpu().numpy().clip(-1., 1.))
        counts = np.asarray(info.get("actual_integrations", np.zeros(self.num_envs))).copy()
        ended = terminated | timeout
        if ended.any():
            counts[ended] = np.asarray(info["final_info"]["actual_integrations"])[ended]
        if not np.all(counts == 1):
            raise RuntimeError("Each vector worker must perform one actual integration")
        self.real_integrations += self.num_envs
        self.common_step_counter += 1
        self.actor_batches += 1
        self.episode_length_buf.add_(1)
        self.episode_length_buf[torch.tensor(ended, device=self.device)] = 0
        self.obs = self._observations(obs, info)
        return self.obs, torch.tensor(reward, device=self.device, dtype=torch.float32), \
            torch.tensor(ended, device=self.device), {
                "time_outs": torch.tensor(timeout, device=self.device), "log": {}}

    def close(self):
        self.env.close()
