"""Consume the real Pollen registry, retaining its entire training machinery."""

from __future__ import annotations

import copy
import importlib.metadata
import inspect
import math
import subprocess
import sys
from pathlib import Path

from .serialization import identity, qualified, sha256_file, snapshot

SOURCE_COMMIT = "5946fd9cdbc58956424420153e51975af3b30d77"
SKILL_TASKS = {
    "standing": "Mjlab-VelStand-Flat-MicroDuck",
    "walking": "Mjlab-Velocity-Rough-MicroDuck",
    "sit_stand": "Mjlab-SitStand-Flat-MicroDuck",
    "ground_pick": "Mjlab-GroundPick-Flat-MicroDuck",
    "kick_left": "Mjlab-BallKick-Flat-MicroDuck",
    "kick_right": "Mjlab-BallKick-Flat-MicroDuck",
    "roller": "Mjlab-Velocity-Flat-MicroDuck-Rollers",
    "roller_crouch": "Mjlab-RollerCrouch-Flat-MicroDuck",
    "roulade": "Mjlab-Roulade-Flat-MicroDuck",
}
JOINT_ORDER = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def load_source(root: Path) -> dict:
    root = root.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Upstream commit changed: {commit}; expected {SOURCE_COMMIT}")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = [name for name in tracked if name and (
        name.startswith("src/") or name in {"pyproject.toml", "uv.lock", "scripts/train.py"}
    )]
    # Reject scientific source modifications, while explicitly excluding dirty inference code.
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root, text=True).splitlines()
    if set(dirty) & set(files):
        raise ValueError(f"Modified adopted scientific source: {sorted(set(dirty) & set(files))}")
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import mjlab_microduck.tasks  # Executes actual upstream registration.
    from mjlab.tasks import registry
    registered = {task for task in registry.list_tasks() if "MicroDuck" in task}
    if len(registered) != 33 or not set(SKILL_TASKS.values()) <= registered:
        raise ValueError(f"Incomplete adopted registry: {len(registered)} MicroDuck tasks")
    module_path = Path(inspect.getfile(mjlab_microduck.tasks)).resolve()
    if not module_path.is_relative_to(root):
        raise ValueError(f"Wrong imported upstream package: {module_path}")
    versions = {}
    for distribution in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "better-actuator-models", "onnx"):
        versions[distribution] = importlib.metadata.version(distribution)
    return {
        "commit": commit, "root": str(root), "python": sys.executable,
        "versions": versions, "ignored_unadopted_changes": dirty,
        "files": {name: sha256_file(root / name) for name in files},
        "registry": registry,
    }


def delay_plan(minimum: int, maximum: int, original_dt: float, dt: float,
               update_period: int = 0, hold_prob: float = 0.0, per_env_phase: bool = True) -> dict:
    seconds = [lag * original_dt for lag in range(minimum, maximum + 1)]
    ticks = [math.ceil(seconds_value / dt - 1e-12) for seconds_value in seconds]
    period_s = update_period * original_dt
    return {
        "sample_delay_seconds": seconds, "mapped_ticks": ticks,
        "probabilities": [1 / len(seconds)] * len(seconds),
        "quantization_residual_seconds": [tick * dt - sec for tick, sec in zip(ticks, seconds)],
        "update_period_seconds": period_s,
        "update_period_ticks": math.ceil(period_s / dt - 1e-12) if update_period else 0,
        "hold_prob": hold_prob, "per_env_phase": per_env_phase,
        "selection": "latest timestamp <= now - sampled delay; causal zero-order hold",
        "initial_history": "explicit reset hold until first causally eligible sample",
    }


def make_skill_cfg(source: dict, skill: str, *, substeps: int = 1,
                   num_envs: int = 1, seed: int = 1000001, play: bool = False):
    if skill not in SKILL_TASKS:
        raise ValueError(f"No reviewed independent recipe for {skill!r}; Sprint is not aliased to Walk")
    if substeps < 1:
        raise ValueError("source substeps must be positive")
    registry = source["registry"]
    task = SKILL_TASKS[skill]
    cfg = registry.load_env_cfg(task, play=play)
    if skill == "kick_left":
        from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import make_microduck_ball_kick_env_cfg
        cfg = make_microduck_ball_kick_env_cfg(play=play, kick_foot="left")
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_STANDUP_ROBOT_CFG, MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
    )
    original = snapshot(cfg)
    original_physics_dt = cfg.sim.mujoco.timestep
    original_policy_dt = original_physics_dt * cfg.decimation
    is_roller = skill in {"roller", "roller_crouch"}
    cfg.scene.entities["robot"] = copy.deepcopy(
        MICRODUCK_WALK_ROLLERS_ROBOT_CFG if is_roller else MICRODUCK_STANDUP_ROBOT_CFG
    )
    cfg.sim.mujoco.timestep = 1 / (60 * substeps)
    cfg.decimation = substeps
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False  # Terminal state must be captured before any reset.
    timing = {"physics_dt": cfg.sim.mujoco.timestep, "policy_dt": 1 / 60,
              "source_substeps": substeps, "original_physics_dt": original_physics_dt,
              "original_policy_dt": original_policy_dt, "observation": {}, "motor": [], "schedules": []}
    def scale_steps(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "step" and isinstance(item, int):
                    seconds = item * original_policy_dt
                    mapped = math.ceil(seconds * 60 - 1e-12) if item else 0
                    value[key] = mapped
                    timing["schedules"].append({"path": f"{path}.step", "source_steps": item,
                                                "seconds": seconds, "mapped_steps": mapped,
                                                "quantization_residual_seconds": mapped / 60 - seconds})
                else:
                    scale_steps(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                scale_steps(item, f"{path}[{index}]")
    # Upstream schedule dicts are keyed by global policy steps. Their physical
    # duration remains fixed even though each new policy step lasts 1/60 second.
    for manager_name in ("events", "curriculum", "rewards", "terminations"):
        for name, term in getattr(cfg, manager_name).items():
            scale_steps(term.params, f"{manager_name}.{name}.params")
    for group, group_cfg in cfg.observations.items():
        timing["observation"][group] = {}
        for name, term in group_cfg.terms.items():
            if term.delay_max_lag:
                plan = delay_plan(term.delay_min_lag, term.delay_max_lag, original_policy_dt,
                                  1 / 60, term.delay_update_period, term.delay_hold_prob,
                                  term.delay_per_env_phase)
                timing["observation"][group][name] = plan
                term.delay_min_lag = min(plan["mapped_ticks"])
                term.delay_max_lag = max(plan["mapped_ticks"])
                term.delay_update_period = plan["update_period_ticks"]
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        plan = delay_plan(actuator.delay_min_lag, actuator.delay_max_lag, original_physics_dt,
                          cfg.sim.mujoco.timestep, actuator.delay_update_period,
                          actuator.delay_hold_prob, actuator.delay_per_env_phase)
        timing["motor"].append(plan)
        actuator.delay_min_lag = min(plan["mapped_ticks"])
        actuator.delay_max_lag = max(plan["mapped_ticks"])
        actuator.delay_update_period = plan["update_period_ticks"]
    adoption = {
        "skill": skill, "upstream_task": task, "family": "roller_allcollisions" if is_roller else "leg_allcollisions",
        "original_config": original, "effective_config": snapshot(cfg), "timing_plan": timing,
        "explicit_changes": ["full collision family shared across skills", "policy 60 Hz", "terminal auto reset disabled",
                             "physical delay distribution preserved in seconds; causal hold and quantization recorded",
                             "global-step schedule boundaries retain their original physical seconds"] +
                            (["left-foot factory override"] if skill == "kick_left" else []),
        "runner": snapshot(registry.load_runner_cls(task)),
        "rl_config": snapshot(registry.load_rl_cfg(task)),
    }
    adoption["sha256"] = identity(adoption)
    return cfg, adoption


def inventory(source: dict) -> dict:
    registry = source["registry"]
    entries = {}
    for task in registry.list_tasks():
        if "MicroDuck" not in task:
            continue
        cfg = registry.load_env_cfg(task)
        runner = registry.load_runner_cls(task)
        entries[task] = {"config": snapshot(cfg), "rl_config": snapshot(registry.load_rl_cfg(task)),
                         "runner": snapshot(runner), "runner_mro": [qualified(cls) for cls in runner.__mro__],
                         "events": snapshot(cfg.events), "curriculum": snapshot(cfg.curriculum),
                         "rewards": snapshot(cfg.rewards), "terminations": snapshot(cfg.terminations),
                         "observations": snapshot(cfg.observations)}
    result = {key: value for key, value in source.items() if key != "registry"}
    result.update(registered_tasks=entries, skill_tasks=SKILL_TASKS,
                  sprint={"status": "independent recipe and acceptance contract required", "training_allowed": False})
    result["sha256"] = identity(result)
    return result
