"""Native MuJoCo task in Gymnasium; physics stays on CPU at one20ms step.

Gymnasium owns vector workers/reset scheduling and RSL owns PPO. This module
only supplies the robot task and the existing65/18 interface. GPU physics and
game qualifications are separate from this bounded native source pilot.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .speculative_contact import CANDIDATE, make_source_runtime

REVISION = "goose_native_cpu_front_recovery_pilot_v1"


class NativeGooseRecoveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_path, contract_path, initial_qpos, *,
                 episode_ticks=700, trajectory_directory=None):
        self.runtime = make_source_runtime(Path(model_path), Path(contract_path))
        rt = self.runtime
        if rt.contract["candidate"] != CANDIDATE:
            raise ValueError("Native CPU pilot requires its explicitly admitted contact source")
        self.initial_qpos = np.asarray(initial_qpos, dtype=float).copy()
        if self.initial_qpos.shape != (rt.model.nq,) or not np.isfinite(self.initial_qpos).all():
            raise ValueError("Frozen recovery birth must match the native model")
        self.nominal_com_height = float(rt.data.subtree_com[rt.torso, 2])
        from sai_agent.goose.convex_support import compiled_body_vertices
        self.supports = {g: compiled_body_vertices(rt.model, g)
            for g in range(rt.model.ngeom)
            if rt.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
        if len(self.supports) != 11 or rt.model.opt.timestep != .02:
            raise ValueError("Native CPU physical identity changed")
        self.motor_vids = rt.model.jnt_dofadr[rt.model.actuator_trnid[:, 0]]
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
        return np.array([(v @ d.xmat[int(m.geom_bodyid[g])].reshape(3, 3).T
            + d.xpos[int(m.geom_bodyid[g])])[:, 2].min() for g, v in self.supports.items()])

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
        return rt.observations().astype(np.float32), {"reset_integrations": 0}

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
        row = dict(qpos=rt.data.qpos.copy(), qvel=rt.data.qvel.copy(), torque=rt.last_tau.copy(),
            target=rt.target.copy(), thermal=rt.thermal.copy(), action=action.copy(),
            observation=observations.copy(), min_collision_z=z, time=rt.data.time,
            upright=up, height_ratio=height, reward=reward)
        self.records.append(row)
        if (abs(rt.data.time-before-.02) > 1e-10 or not np.isfinite(z).all() or -z.min() > .005):
            self._save_episode(failed=True)
            raise RuntimeError("Native CPU pilot failed the frozen20ms/5mm physical guard")
        timeout = rt.physics_integrations >= self.episode_ticks
        return observations.astype(np.float32), float(reward), False, timeout, {
            "actual_integrations": 1, "total_integrations": self.total_integrations,
            "max_penetration_m": float(max(0., -z.min())), "upright": up,
            "height_ratio": height, "episode_tick": rt.physics_integrations}

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


class NativeGooseRslEnv:
    """Thin RSL tensor adapter for Gymnasium's standard AsyncVectorEnv."""

    def __init__(self, env, *, device="cpu", seed=53):
        import torch
        from tensordict import TensorDict
        if env.autoreset_mode != gym.vector.AutoresetMode.SAME_STEP:
            raise ValueError("Same-step reset prevents a skipped integration nextTick")
        self.env, self.device = env, device
        self.num_envs, self.num_actions = env.num_envs, 18
        self.max_episode_length = 700
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.cfg = {"task": REVISION, "physics_device": "cpu", "physics_dt_s": .02,
            "decimation": 1, "actor": 65, "actions": 18}
        # Successful vector returns only. After worker failure, saved worker
        # trajectories are authoritative for integrations in the failed call.
        self.real_integrations = 0
        self.common_step_counter = 0
        self.actor_batches = 0
        obs, _ = env.reset(seed=seed)
        self.obs = TensorDict({"actor": torch.tensor(obs, device=device)}, [self.num_envs])

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
        self.obs = TensorDict({"actor": torch.tensor(obs, device=self.device)}, [self.num_envs])
        return self.obs, torch.tensor(reward, device=self.device, dtype=torch.float32), \
            torch.tensor(ended, device=self.device), {
                "time_outs": torch.tensor(timeout, device=self.device), "log": {}}

    def close(self):
        self.env.close()
