"""Goose-owned moving reference states at explicit episode resets.

The native mjlab step and RSL optimizer remain unchanged. Reference states
carry integration, driver, command, action and contact/reward history; they
are development initial conditions, never runtime body assistance or experts.
"""
from pathlib import Path

import torch

from .artifacts import sha256
from .mjlab_env import GooseDevelopmentEnv, actor_observation

REVISION = "goose_completed_moving_initialization_v1"
DATA_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
    "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "eq_active")
DRIVE_FIELDS = ("actions", "target", "thermal", "last_tau", "strength",
    "delay", "phase", "prepared_ticks", "completed_ticks", "cap",
    "force_lower", "force_upper")
COMMAND_FIELDS = ("sampled_commands", "bucket_ids", "warm_prefix_active",
    "vel_command_b", "vel_command_w", "is_standing_env", "is_heading_env",
    "is_world_env", "is_forward_env", "time_left", "command_counter")
ACTION_FIELDS = ("_raw", "_effort", "_solver_force", "commands")
MANAGER_FIELDS = ("_action", "_prev_action", "_prev_prev_action")


def _world_buffers(env):
    """Explicit state inventory for this pinned Goose/mjlab profile."""
    term = env.action_manager.get_term("goose")
    command = env.command_manager.get_term("velocity")
    buffers = {}
    for prefix, owner, fields in (("data", env.sim.data, DATA_FIELDS),
            ("drive", term.drive, DRIVE_FIELDS), ("action", term, ACTION_FIELDS),
            ("action_manager", env.action_manager, MANAGER_FIELDS),
            ("command", command, COMMAND_FIELDS)):
        for name in fields:
            value = getattr(owner, name)
            if value.shape[0] != env.num_envs:
                raise ValueError(f"Moving state world layout changed: {prefix}.{name}")
            buffers[f"{prefix}.{name}"] = value
    buffers["model.actuator_forcerange"] = env.sim.model.actuator_forcerange
    buffers["reward.peak_heights"] = env.reward_manager.get_term_cfg(
        "foot_swing_height").func.peak_heights
    for name, sensor in env.scene.sensors.items():
        air = getattr(sensor, "_air_time_state", None)
        if air is not None:
            for field, value in vars(air).items():
                buffers[f"sensor.{name}.air.{field}"] = value
        for field, value in (getattr(sensor, "_history_state", None) or {}).items():
            buffers[f"sensor.{name}.history.{field}"] = value
    return buffers


def _identity(env):
    cfg = env.cfg.actions["goose"]
    return dict(revision=REVISION, model_sha256=sha256(cfg.model_path),
        contract_sha256=sha256(cfg.contract_path), nq=env.sim.mj_model.nq,
        nv=env.sim.mj_model.nv, dt_s=env.physics_dt, decimation=env.cfg.decimation)


@torch.no_grad()
def capture_completed_moving_state(env, world_id=0):
    """Capture an own actual completed Tick; caller records source provenance."""
    term = env.action_manager.get_term("goose")
    if (term._before_time is not None or term.drive._pending_action is not None
            or bool(env._manual_reset_pending[world_id]) or env.cfg.auto_reset):
        raise ValueError("Capture requires a completed nonterminal actual Tick")
    buffers = {key: value[world_id].detach().cpu().clone()
        for key, value in _world_buffers(env).items()}
    buffers["data.qpos"][term.root_q:term.root_q+3] -= env.scene.env_origins[world_id].cpu()
    command = env.command_manager.get_term("velocity")
    return dict(identity=_identity(env), buffers=buffers,
        episode_age=int(env.episode_length_buf[world_id]),
        common_step_counter=env.common_step_counter,
        command_stage=command._last_stage,
        actor_observation=actor_observation(env)[world_id].cpu().clone())


def _validate(env, state):
    if state["identity"] != _identity(env):
        raise ValueError("Moving reference model/contract/timing changed")
    inventory = _world_buffers(env)
    if state["buffers"].keys() != inventory.keys():
        raise ValueError("Moving reference history inventory changed")
    for name, target in inventory.items():
        source = state["buffers"][name]
        if (source.shape != target.shape[1:] or source.dtype != target.dtype
                or not bool(torch.isfinite(source).all())):
            raise ValueError(f"Wrong-size/nonfinite moving reference: {name}")
    if state["episode_age"] < env.command_manager.get_term("velocity").cfg.standing_ticks:
        raise ValueError("A moving reference cannot bypass its recorded cold standing prefix")
    q = state["buffers"]["data.qpos"]
    term = env.action_manager.get_term("goose")
    if abs(float(q[term.root_q+3:term.root_q+7].norm())-1.) > 2e-6:
        raise ValueError("Moving reference contains invalid native quaternion")
    if (state["buffers"]["drive.actions"].abs().max() > 1.
            or state["buffers"]["drive.thermal"].min() < 0.
            or bool(state["buffers"]["command.warm_prefix_active"])):
        raise ValueError("Moving reference has invalid action, thermal or prefix state")
    target = state["buffers"]["drive.target"]
    ranges = term.drive.ranges.cpu()
    if (bool((target < ranges[:,0]).any()) or bool((target > ranges[:,1]).any())
            or bool((state["buffers"]["drive.strength"] <= 0).any())
            or not 0 <= float(state["buffers"]["drive.phase"]) < 2*torch.pi):
        raise ValueError("Moving reference violates drive targets, strength or phase")


class GooseMovingStateEnv(GooseDevelopmentEnv):
    """Restore own full history only inside a genuine public reset boundary.

    Source episode age is retained: this is reference-state initialization,
    not a demonstration of cold birth or an in-game policy transition.
    Global PPO/curriculum time belongs to the runner and is never restored.
    """

    def __init__(self, cfg, device, *, moving_bank_path: Path,
                 moving_bank_sha256: str, **kwargs):
        self._moving_path = moving_bank_path
        self._moving_digest = moving_bank_sha256
        if sha256(moving_bank_path) != moving_bank_sha256:
            raise ValueError("Moving reference bank identity changed")
        bank = torch.load(moving_bank_path, map_location="cpu", weights_only=True)
        if bank["revision"] != REVISION or not bank["states"]:
            raise ValueError("Empty or wrong-version moving reference bank")
        self._moving_states = bank["states"]
        self._moving_cursor = 0
        super().__init__(cfg, device, **kwargs)
        # These native-drive buffers normally appear at first prepare. A
        # reference reset must load them before that first physical Tick.
        drive = self.action_manager.get_term("goose").drive
        drive.cap = drive.peak.repeat(self.num_envs, 1)
        drive.force_lower = -drive.cap.clone()
        drive.force_upper = drive.cap.clone()
        for state in self._moving_states:
            _validate(self, state)

    @torch.no_grad()
    def reset(self, **kwargs):
        if sha256(self._moving_path) != self._moving_digest:
            raise ValueError("Moving bank changed before reset")
        term = self.action_manager.get_term("goose")
        if term._before_time is not None or term.drive._pending_action is not None:
            raise RuntimeError("Preserve pending physical Tick before moving reset")
        ids = kwargs.get("env_ids")
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        states = [self._moving_states[(self._moving_cursor+k) % len(self._moving_states)]
            for k in range(len(ids))]
        for state in states:
            _validate(self, state)  # Validate before native reset can mutate a world.
        super().reset(**kwargs)
        self._moving_cursor = (self._moving_cursor+len(ids)) % len(self._moving_states)
        inventory = _world_buffers(self)
        for key, target in inventory.items():
            source = torch.stack([s["buffers"][key] for s in states]).to(self.device)
            if key == "data.qpos":
                source[:,term.root_q:term.root_q+3] += self.scene.env_origins[ids]
            target[ids] = source
        self.episode_length_buf[ids] = torch.tensor(
            [s["episode_age"] for s in states], device=self.device)
        command = self.command_manager.get_term("velocity")
        # Curriculum stage is global runner time. A partial reference reset
        # must not rewind it or resample commands in the continuing worlds.
        command._last_stage = command.course_stage
        # Native forward/sense refresh derived state, with zero integrations.
        self.sim.forward()
        self.sim.sense()
        # Restore solver memory and contact/reward history after the refresh;
        # deriving reset sensors must not count another physical contact Tick.
        inventory = _world_buffers(self)
        for key, target in inventory.items():
            if key == "data.qacc_warmstart" or key.startswith(("sensor.", "reward.")):
                target[ids] = torch.stack([s["buffers"][key] for s in states]).to(self.device)
        for sensor in self.scene.sensors.values():
            sensor._invalidate_cache()
        self.obs_buf = self.observation_manager.compute(update_history=True, env_ids=ids)
        return self.obs_buf, self.extras
