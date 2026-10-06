"""Thin, source-pinned HumanUP tracking task on the original Goose drive.

Only episode initialization and learning signals belong to this task. Native
MuJoCo/Warp, the original drive, Actor65 and one20ms integral remain unchanged.
An unqualified discovery reference never confers physical or policy success.
"""
import ast
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .artifacts import DT, JOINT_ORDER, sha256
from .mjlab_env import (
    RigidGooseAction, RigidGooseActionCfg, actor_observation,
    make_development_env_cfg)
from .native_geometry import collision_geom_vertices
from .source_training import episode_timeout

REVISION = "goose_humanup_constrained_tracking_v1"
CONTACT_SENSOR = "goose_tracking_ground"
# Author: left leg6, right leg6, waist3, upper8. Unused coordinates are zero;
# the18 real coordinates retain their original meaning, limits and units.
GOOSE_INDICES = (14, 13, 12, 15, 16, 17, 8, 7, 6, 9, 10, 11, 0, 1, 2, 3, 4, 5)
AUTHOR_INDICES = (*range(12), *range(15, 21))


def canonical(values, *, fill=0.):
    result = values.new_full((*values.shape[:-1], 23), fill)
    result[..., list(AUTHOR_INDICES)] = values[..., list(GOOSE_INDICES)]
    return result


def reference_phase(ticks, frames):
    """Preserve integer reference indexing through the author's divide/multiply.

    A one-ULP upward rounding avoids selecting Tick t-1 after a floating point
    roundtrip. This is reference lookup only, never the physical/Actor clock.
    """
    value = ticks.clamp(0, frames-1).double()/(frames-1)
    return torch.nextafter(value, torch.full_like(value, float("inf")))


def load_author(source_path, source_sha256, config_path, config_sha256,
                base_path, base_sha256):
    """Execute unchanged authored method ASTs from explicitly pinned source.

    Keep the upstream file and its license intact; importing IsaacGym is not
    necessary to execute its tensor reward functions. No AST is rewritten.
    """
    for path, digest in ((source_path, source_sha256), (config_path, config_sha256),
                         (base_path, base_sha256)):
        if sha256(path) != digest:
            raise ValueError("Pinned HumanUP source identity changed")
    tree = ast.parse(Path(config_path).read_text())
    root = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "G1WaistTrackCfg")
    rewards = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == "rewards")
    scales = next(n for n in rewards.body if isinstance(n, ast.ClassDef) and n.name == "scales")
    weights = {n.targets[0].id: ast.literal_eval(n.value)
               for n in scales.body if isinstance(n, ast.Assign)}
    weights = {name: weight for name, weight in weights.items() if weight != 0}
    wanted = {"_get_phase", *("_reward_"+name for name in weights)}
    # Resolve inherited author methods first, then exact G1 overrides.
    methods = {}
    for path in (base_path, source_path):
        source = ast.parse(Path(path).read_text())
        methods.update({n.name: n for c in source.body if isinstance(c, ast.ClassDef)
                        for n in c.body if isinstance(n, ast.FunctionDef) and n.name in wanted})
    if methods.keys() != wanted:
        raise ValueError("Complete authored tracking method set required")
    namespace = {"torch": torch, "quat_rotate_inverse": lambda q, v:
                 quat_apply_inverse(torch.cat((q[..., 3:4], q[..., :3]), -1), v)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=list(methods.values()), type_ignores=[])),
                 str(source_path), "exec"), namespace)
    return type("PinnedHumanupTracking", (), {name: namespace[name] for name in wanted}), weights


def qualified_support(up, height, nominal_height, velocity, angular, normals,
                      foot_loads, nonfoot_load, weight):
    """Existing physical hold criteria; this is a metric, not a new reward."""
    return ((up >= .95) & (height >= .85*nominal_height)
            & (velocity.norm(dim=-1) <= .04) & (angular.norm(dim=-1) <= .2)
            & (normals.min(-1).values >= .95)
            & (foot_loads.min(-1).values >= .1*weight)
            & (foot_loads.sum(-1) >= .9*weight) & (nonfoot_load <= 1.))


@dataclass(kw_only=True)
class TrackingActionCfg(RigidGooseActionCfg):
    reference_path: Path
    reference_sha256: str
    author_path: Path
    author_sha256: str
    author_config_path: Path
    author_config_sha256: str
    author_base_path: Path
    author_base_sha256: str
    align_cold_target: bool = True

    def build(self, env):
        return TrackingAction(self, env)


class TrackingAction(RigidGooseAction):
    """Capture solved forces before upstream forward; original actuation only."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if sha256(cfg.reference_path) != cfg.reference_sha256:
            raise ValueError("Tracking reference changed")
        with np.load(cfg.reference_path) as data:
            if (str(data["model_sha256"]) != sha256(cfg.model_path)
                    or str(data["contract_sha256"]) != sha256(cfg.contract_path)
                    or float(data["dt_s"]) != DT
                    or data["joint_order"].tolist() != list(JOINT_ORDER)):
                raise ValueError("Reference plant, clock or joint order changed")
            q = data["joint_q"].copy()
            head = data["head_height_m"].copy()
            self.nominal_height = float(data["nominal_COM_height_m"])
            self.head_factor = float(data["head_height_factor"])
            self.base_factor = float(data["base_height_factor"])
            self.feet_factor = float(data["feet_XY_factor"])
        if (q.ndim != 2 or q.shape[1] != 18 or len(q) < 2
                or head.shape != (len(q),) or not np.isfinite(q).all()
                or not np.isfinite(head).all() or self.nominal_height <= 0):
            raise ValueError("Finite ordered reference and explicit units required")
        limits = self.drive.ranges.cpu().numpy()
        lower = np.maximum(limits[:, 0], (self.drive.neutral-self.drive.scale).cpu().numpy())
        upper = np.minimum(limits[:, 1], (self.drive.neutral+self.drive.scale).cpu().numpy())
        if (np.any(q < lower-1e-12) or np.any(q > upper+1e-12)
                or np.any(np.abs(np.diff(q, axis=0)) >
                          self.drive.speed.cpu().numpy()*DT+1e-12)):
            raise ValueError("Reference target exceeds original18-axis ranges")
        self.reference = torch.as_tensor(q, device=self.device, dtype=torch.float64)
        self.head_reference = torch.as_tensor(head, device=self.device, dtype=torch.float64)
        self.author, _ = load_author(cfg.author_path, cfg.author_sha256,
            cfg.author_config_path, cfg.author_config_sha256,
            cfg.author_base_path, cfg.author_base_sha256)
        self.ground_forces = torch.zeros((self.num_envs, 11, 3),
                                         device=self.device, dtype=torch.float64)
        self._ground_captured = False
        self.hold = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_hold = torch.zeros_like(self.hold)
        self.context = None
        m = env.sim.mj_model
        self.leaves = [g for g in range(m.ngeom) if m.geom_type[g] in
                       (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX)]
        if len(self.leaves) != 11 or env.contact_adapter is not None:
            raise ValueError("Original11-leaf native plant required")
        self.feet = [m.body("robot/"+s+"_ankle_roll").id for s in ("right", "left")]
        self.foot_slots = [self.leaves.index(m.geom("robot/"+s+"_flexible_sole").id)
                           for s in ("right", "left")]
        self.head = m.body("robot/head_roll").id
        self.vertices = [(g, torch.as_tensor(collision_geom_vertices(m, g),
                          device=self.device, dtype=torch.float32)) for g in self.leaves]

    def process_actions(self, actions):
        super().process_actions(actions)
        self._ground_captured = False
        self.context = None

    def verify_actual_step(self):
        result = super().verify_actual_step()
        if not self._ground_captured:
            # The per-substep metric calls this after integration and before
            # forward can solve a different contact state. Primary geom force
            # uses the native sensor's opposite sign; verify in admission.
            self.ground_forces.copy_(-self._env.scene[CONTACT_SENSOR].data.force)
            self._ground_captured = True
        return result

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if self.cfg.align_cold_target:
            # Explicit cold initialization, after reset event pose writes.
            # No changes to previous action, thermal, phase or mechanical state.
            self.drive.target[ids] = self._env.sim.data.qpos[ids][:, self.qids].double().clamp(
                self.drive.ranges[:, 0], self.drive.ranges[:, 1])
        self.ground_forces[ids] = 0
        self.hold[ids] = self.max_hold[ids] = 0
        self.context = None


def snapshot(env):
    term = env.action_manager.get_term("goose")
    if term.context is not None:
        return term.context
    m = env.sim.mj_model
    with wp.ScopedDevice(env.sim.wp_device):
        if not hasattr(term, "private_data"):
            term.private_data = mjw.make_data(m, nworld=env.num_envs,
                nconmax=0, nccdmax=0, njmax=0, njmax_nnz=0)
        d = term.private_data
        wp.copy(d.qpos, env.sim.wp_data.qpos)
        wp.copy(d.qvel, env.sim.wp_data.qvel)
        mjw.kinematics(env.sim.wp_model, d)
        mjw.com_pos(env.sim.wp_model, d)
        mjw.com_vel(env.sim.wp_model, d)
    xpos, ipos = wp.to_torch(d.xpos), wp.to_torch(d.xipos)
    rotation = wp.to_torch(d.xmat).reshape(env.num_envs, m.nbody, 3, 3)
    quat, com, cvel = wp.to_torch(d.xquat), wp.to_torch(d.subtree_com), wp.to_torch(d.cvel)
    qpos, qvel = wp.to_torch(d.qpos), wp.to_torch(d.qvel)
    roots = torch.as_tensor(m.body_rootid, device=env.device, dtype=torch.long)
    mass = torch.as_tensor(m.body_mass, device=env.device, dtype=ipos.dtype)
    inertial_velocity = cvel[:, :, 3:6] + torch.cross(
        cvel[:, :, :3], ipos-com[:, roots], dim=-1)
    velocity = (inertial_velocity*mass[None, :, None]).sum(1)/mass.sum()
    height = com[:, term.torso, 2]-env.scene.env_origins[:, 2]
    angular = cvel[:, term.torso, :3]
    normals = rotation[:, term.feet, 2, 2]
    loads = term.ground_forces[:, term.foot_slots, 2]
    nonfoot = term.ground_forces[:, :, 2].sum(-1)-loads.sum(-1)
    ctx = term.author()
    ctx.device = env.device
    ctx.cfg = SimpleNamespace(rewards=SimpleNamespace(
        tracking_sigma=.2, soft_torque_limit=.95, min_dist=.25, max_dist=1.))
    ctx.target_traj_length = len(term.reference)
    ctx.episode_length_buf = env.episode_length_buf
    ctx._get_phase = lambda: reference_phase(env.episode_length_buf, len(term.reference))
    ctx.dof_pos_all_interp = canonical(term.reference)
    ctx.dof_pos = canonical(qpos[:, term.qids])
    ctx.dof_vel = canonical(qvel[:, term.vids])
    ctx.dof_pos_limits = canonical(term.drive.ranges.T).T
    # RewardManager runs before post-forward drive.commit. The captured force
    # is this integral's effort; drive.last_tau still describes the prior Tick.
    ctx.torques = canonical(term._solver_force)
    ctx.torque_limits = canonical(term.drive.peak, fill=1.)
    ctx.base_lin_vel = torch.einsum("bij,bj->bi", rotation[:, term.torso].transpose(1, 2), velocity)
    ctx.base_ang_vel = torch.einsum("bij,bj->bi", rotation[:, term.torso].transpose(1, 2), angular)
    state = torch.zeros((env.num_envs, m.nbody, 13), device=env.device)
    state[:, :, :3] = xpos-env.scene.env_origins[:, None, :]
    state[:, term.feet, :2] *= term.feet_factor
    ctx.rigid_body_states = state
    ctx.rigid_body_rot = torch.cat((quat[:, :, 1:], quat[:, :, :1]), -1)
    ctx.feet_indices = term.feet
    ctx.gravity_vec = state.new_tensor([0., 0., -1.]).expand(env.num_envs, -1)
    ctx.actions = canonical(env.action_manager.action*term.drive.scale/.5)
    ctx.last_actions = canonical(env.action_manager.prev_action*term.drive.scale/.5)
    ctx.reset_buf = env.termination_manager.terminated.clone()
    ctx.time_out_buf = env.termination_manager.time_outs.clone()
    ctx.motor_velocity = canonical(qvel[:, term.motor_vids])
    ctx.actual = dict(up=rotation[:, term.torso, 2, 2], height=height,
        velocity=velocity, angular=angular, normals=normals, loads=loads,
        nonfoot=nonfoot, weight=mass.sum()*9.81,
        head_height=ipos[:, term.head, 2]-env.scene.env_origins[:, 2])
    positions, matrices = wp.to_torch(d.geom_xpos), wp.to_torch(d.geom_xmat)
    lowest = [(v@matrices[:, g].reshape(env.num_envs, 3, 3)[:, 2, :].T
               +positions[:, g, 2]).min(0).values for g, v in term.vertices]
    ctx.depth = -torch.stack(lowest).min(0).values.clamp(max=0)
    term.context = ctx
    return ctx


def reward(env, name):
    ctx = snapshot(env)
    # Energy uses actual motor speed, including the beak input rotor, rather
    # than observable jaw speed. All other authored terms use logical axes.
    if name == "energy":
        logical = ctx.dof_vel
        ctx.dof_vel = ctx.motor_velocity
        try:
            return ctx._reward_energy()
        finally:
            ctx.dof_vel = logical
    if name == "termination":
        ctx.reset_buf = env.termination_manager.terminated
        ctx.time_out_buf = env.termination_manager.time_outs
    return getattr(ctx, "_reward_"+name)()


def contact_failure(env):
    return snapshot(env).depth > .005


def tracking_failure(env):
    term = env.action_manager.get_term("goose")
    a = snapshot(env).actual
    frame = env.episode_length_buf.clamp(max=len(term.reference)-1)
    head_error = (a["head_height"]-term.head_reference[frame]).abs()*term.head_factor
    return ((head_error > .2) | (a["velocity"].norm(dim=-1) > 2.5)
            | (a["height"]*term.base_factor > 1.2) | (a["height"] < 0))


def record_hold(env, env_ids=None):
    del env_ids
    term = env.action_manager.get_term("goose")
    a = snapshot(env).actual
    stable = qualified_support(a["up"], a["height"], term.nominal_height,
        a["velocity"], a["angular"], a["normals"], a["loads"], a["nonfoot"], a["weight"])
    term.hold.copy_(torch.where(stable, term.hold+1, 0))
    term.max_hold.copy_(torch.maximum(term.hold, term.max_hold))


def critic_state(env):
    # ObservationManager queries term dimensions before TerminationManager
    # exists. No physics has run at that construction boundary.
    if not hasattr(env, "termination_manager"):
        return torch.zeros((env.num_envs, 27), device=env.device)
    term = env.action_manager.get_term("goose")
    ctx = snapshot(env)
    a = ctx.actual
    frame = env.episode_length_buf.clamp(max=len(term.reference)-1)
    # No reference, force, height or episode-time value reaches Actor65.
    return torch.cat((a["velocity"], a["height"][:, None], a["loads"]/a["weight"],
                      a["normals"], frame[:, None]/(len(term.reference)-1),
                      term.reference[frame]), -1).float()


def make_tracking_cfg(model_path, contract_path, reference_path, reference_sha256,
                      author_path, author_sha256, author_config_path,
                      author_config_sha256, author_base_path, author_base_sha256,
                      *, initial_qpos, num_envs=16, seed=113,
                      align_cold_target=True):
    """Fixed original birth; no random state injection or reference root writes.

    Single nominal pilot freezes regularization1 and disables domain
    randomization/RSI. This is a declared Goose adaptation, not full HumanUP.
    """
    _, weights = load_author(author_path, author_sha256,
        author_config_path, author_config_sha256, author_base_path, author_base_sha256)
    cfg = make_development_env_cfg(model_path, contract_path, num_envs=num_envs,
                                  development_initial_qpos=initial_qpos)
    cfg.seed = seed
    cfg.episode_length_s = 13.
    old = cfg.actions["goose"]
    cfg.actions["goose"] = TrackingActionCfg(
        **{f.name: getattr(old, f.name) for f in fields(old) if f.init},
        reference_path=Path(reference_path), reference_sha256=reference_sha256,
        author_path=Path(author_path), author_sha256=author_sha256,
        author_config_path=Path(author_config_path), author_config_sha256=author_config_sha256,
        author_base_path=Path(author_base_path), author_base_sha256=author_base_sha256,
        align_cold_target=align_cold_target)
    m = mujoco.MjModel.from_xml_path(str(model_path))
    leaves = tuple(m.geom(g).name for g in range(m.ngeom) if m.geom_type[g] in
                   (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX))
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name=CONTACT_SENSOR, primary=ContactMatch(mode="geom", entity="robot", pattern=leaves),
        secondary=ContactMatch(mode="geom", pattern="ground"),
        fields=("force",), reduce="netforce", num_slots=1, global_frame=True))
    cfg.rewards = {name: RewardTermCfg(func=reward, weight=weight, params={"name": name})
                   for name, weight in weights.items()}
    cfg.terminations = {"time_out": TerminationTermCfg(func=episode_timeout, time_out=True),
        "contact_failure": TerminationTermCfg(func=contact_failure),
        "tracking_failure": TerminationTermCfg(func=tracking_failure)}
    cfg.events["tracking_hold"] = EventTermCfg(func=record_hold, mode="step")
    cfg.metrics["tracking_depth"] = MetricsTermCfg(func=lambda env: snapshot(env).depth)
    cfg.metrics["tracking_hold"] = MetricsTermCfg(func=lambda env:
        env.action_manager.get_term("goose").hold.float())
    cfg.observations = {
        "actor": ObservationGroupCfg(terms={"base": ObservationTermCfg(func=actor_observation)},
            enable_corruption=False, concatenate_terms=True),
        "critic": ObservationGroupCfg(terms={"base": ObservationTermCfg(func=actor_observation),
            "task": ObservationTermCfg(func=critic_state)},
            enable_corruption=False, concatenate_terms=True)}
    return cfg
