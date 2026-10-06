"""Training-only native BeyondMimic feedback teacher on the frozen Goose.

Reference observations are explicit teacher inputs, never the public Actor65.
The installed reward functions/weights are retained. Reference playback cannot
write running robot state. A separately named RSI profile admits writes only
inside the environment's genuine cold reset, with original drive history reset.
"""
import copy
from dataclasses import asdict, dataclass

import mujoco
import torch
from mjlab.envs import mdp as native_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from .artifacts import DT
from .foot_curriculum import rigid_contact_limit_exceeded
from .mjlab_env import (
    RigidGooseAction, RigidGooseActionCfg, actor_observation,
    GooseDevelopmentEnv, make_development_env_cfg,
)

REVISION = "goose_native_tracking_teacher_v1"
RSI_REVISION = "goose_native_tracking_admitted_rsi_v1"
GROUND_SENSOR = "goose_teacher_ground"


class ColdAlignedRigidAction(RigidGooseAction):
    """Initialize the original persistent target at the genuine cold pose."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ground_forces = torch.zeros((self.num_envs, 11, 3),
            device=self.device, dtype=torch.float64)
        self._ground_captured = False

    def process_actions(self, actions):
        super().process_actions(actions)
        self._ground_captured = False

    def verify_actual_step(self):
        result = super().verify_actual_step()
        if not self._ground_captured:
            # Sensor primary receives the opposite signed pair force. Capture
            # the actual completed solve before the upstream forward refresh.
            self.ground_forces.copy_(-self._env.scene[GROUND_SENSOR].data.force)
            self._ground_captured = True
        return result

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.drive.target[ids] = self._env.sim.data.qpos[ids][:, self.qids].double().clamp(
            self.drive.ranges[:, 0], self.drive.ranges[:, 1])
        self.ground_forces[ids] = 0


@dataclass(kw_only=True)
class ColdAlignedRigidActionCfg(RigidGooseActionCfg):
    def build(self, env):
        return ColdAlignedRigidAction(self, env)


class ReadOnlyMotionCommand(MotionCommand):
    """Native reference math with no simulated state writes or clip resets.

    Keep the native full-reset first observation's one-frame lookahead. A
    zero-dt partial reset refreshes poses without advancing unaffected worlds.
    Exactly one frame advances per real20ms command update; the end holds.
    """

    def _resample_command(self, env_ids):
        drive = self._env.action_manager.get_term("goose").drive
        if (self.cfg.sampling_mode != "start" or drive._pending_action is not None
                or bool((drive.completed_ticks[env_ids] != 0).any())):
            raise RuntimeError("Reference restart requires a genuine cold episode reset")
        self.time_steps[env_ids] = min(1, self.motion.time_step_total-1)

    def compute(self, dt, env_ids=None):
        if dt == 0.:
            self.update_relative_body_poses()
        elif env_ids is None and abs(dt-DT) <= 1e-12:
            super().compute(dt)
        else:
            raise ValueError("Reference clock requires a genuine20ms Tick or cold reset")

    def _update_command(self, env_ids=None):
        # The cold sampler already supplies the one-frame lookahead. A scoped
        # native reset refresh must not advance it again or advance other worlds.
        if env_ids is None:
            self.time_steps.add_(1).clamp_(max=self.motion.time_step_total-1)
        self.update_relative_body_poses()

    def _write_reference_state_to_sim(self, *args, **kwargs):
        raise RuntimeError("Training reference cannot write actual robot state")

    def reset_to_frame(self, *args, **kwargs):
        raise RuntimeError("Reference scrubbing cannot reset actual robot state")


@dataclass(kw_only=True)
class ReadOnlyMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        return ReadOnlyMotionCommand(self, env)


class NativeTrackingColdResetEnv(GooseDevelopmentEnv):
    """Authorize native reference births only during the actual reset call."""

    def reset(self, **kwargs):
        if getattr(self, "_reference_birth_ids", None) is not None:
            raise RuntimeError("Nested reference birth is not a cold reset")
        ids = kwargs.get("env_ids")
        self._reference_birth_ids = (torch.arange(self.num_envs, device=self.device)
            if ids is None else torch.as_tensor(ids, device=self.device, dtype=torch.long))
        try:
            return super().reset(**kwargs)
        finally:
            self._reference_birth_ids = None


class AdmittedColdMotionCommand(ReadOnlyMotionCommand):
    """Use native adaptive proposals and native birth-state writer at reset.

    Proposals map to the nearest pre-admitted exact MotionLoader frame (ties to
    the earlier frame). This explicit restriction changes the birth distribution;
    it does not repair the reference trajectory or grant recovery qualification.
    Reference advancement and native failure-bin EMA continue at 50Hz. Clip end
    holds the reference and cannot reinitialize an already running robot.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        frames = torch.as_tensor(cfg.eligible_frames, device=self.device, dtype=torch.long)
        if (frames.numel() == 0 or bool((frames < 0).any())
                or bool((frames >= self.motion.time_step_total).any())
                or not torch.equal(frames, frames.unique(sorted=True))):
            raise ValueError("Reference birth frames must be admitted, ordered and unique")
        self.eligible_frames = frames
        self.birth_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.birth_proposal = self.birth_frame.clone()
        self.birth_count = self.birth_frame.clone()

    def _require_cold_reset(self, env_ids):
        drive = self._env.action_manager.get_term("goose").drive
        authorized = getattr(self._env, "_reference_birth_ids", None)
        if (authorized is None or not bool(torch.isin(env_ids, authorized).all())
                or drive._pending_action is not None
                or bool((drive.completed_ticks[env_ids] != 0).any())):
            raise RuntimeError("Reference state writes require a genuine cold reset")

    def _adaptive_sampling(self, env_ids):
        MotionCommand._adaptive_sampling(self, env_ids)
        proposals = self.time_steps[env_ids].clone()
        nearest = (proposals[:, None]-self.eligible_frames[None, :]).abs().argmin(-1)
        self.birth_proposal[env_ids] = proposals
        self.time_steps[env_ids] = self.eligible_frames[nearest]

    def _resample_command(self, env_ids):
        self._require_cold_reset(env_ids)
        if self.cfg.sampling_mode != "adaptive":
            raise RuntimeError("Admitted RSI requires the named native adaptive sampler")
        MotionCommand._resample_command(self, env_ids)
        self.birth_frame[env_ids] = self.time_steps[env_ids]
        self.birth_count[env_ids] += 1
        # Preserve the stock full-reset lookahead, without advancing other worlds.
        self.time_steps[env_ids] = (self.time_steps[env_ids]+1).clamp(
            max=self.motion.time_step_total-1)

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori,
                                    root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        self._require_cold_reset(env_ids)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        if (not all(bool(torch.isfinite(x).all()) for x in
                    (root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel))
                or bool((joint_pos < limits[:, :, 0]).any())
                or bool((joint_pos > limits[:, :, 1]).any())):
            raise RuntimeError("Admitted reference birth must not require joint projection")
        MotionCommand._write_reference_state_to_sim(self, env_ids, root_pos, root_ori,
            root_lin_vel, root_ang_vel, joint_pos, joint_vel)
        action = self._env.action_manager.get_term("goose")
        action.drive.target[env_ids] = self._env.sim.data.qpos[env_ids][
            :, action.qids].double()
        action.ground_forces[env_ids] = 0

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        # Native adaptive failure-bin update; no stock clip-end state overwrite.
        if env_ids is None:
            self.bin_failed_count = (self.cfg.adaptive_alpha*self._current_bin_failed
                +(1-self.cfg.adaptive_alpha)*self.bin_failed_count)
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class AdmittedColdMotionCommandCfg(MotionCommandCfg):
    eligible_frames: tuple[int, ...]

    def build(self, env):
        return AdmittedColdMotionCommand(self, env)


def make_native_tracking_teacher_cfg(model_path, contract_path, motion_file, *,
                                    cold_qpos, num_envs=256, seed=133):
    cfg = make_development_env_cfg(model_path, contract_path, num_envs=num_envs,
        development_initial_qpos=cold_qpos)
    if not isinstance(cfg.actions["goose"], RigidGooseActionCfg):
        raise ValueError("Tracking teacher requires the frozen rigid native drive")
    original = cfg.actions["goose"]
    cfg.actions["goose"] = ColdAlignedRigidActionCfg(**vars(original))
    native = mujoco.MjModel.from_xml_path(str(model_path))
    body_names = tuple(native.body(i).name for i in range(1, native.nbody))
    stock = make_tracking_env_cfg()
    cfg.observations = copy.deepcopy(stock.observations)
    for group in cfg.observations.values():
        group.enable_corruption = False  # First named nominal-condition pilot.
        for name, function in (("base_lin_vel", native_mdp.base_lin_vel),
                               ("base_ang_vel", native_mdp.base_ang_vel)):
            group.terms[name].func = function
            group.terms[name].params = {}  # Native root-link point, no invented IMU.
    cfg.observations["public_actor"] = ObservationGroupCfg(
        terms={"base": ObservationTermCfg(func=actor_observation)},
        enable_corruption=False, concatenate_terms=True)
    cfg.commands = {"motion": ReadOnlyMotionCommandCfg(entity_name="robot",
        motion_file=str(motion_file), anchor_body_name="torso", body_names=body_names,
        sampling_mode="start", resampling_time_range=(1e9, 1e9), debug_vis=False,
        pose_range={}, velocity_range={}, joint_position_range=(0., 0.))}
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
        fields=("found", "force"), reduce="none", num_slots=1, history_length=4),
        ContactSensorCfg(name=GROUND_SENSOR,
            primary=ContactMatch(mode="geom", entity="robot", pattern=tuple(
                native.geom(i).name for i in range(native.ngeom)
                if int(native.geom_type[i]) in (mujoco.mjtGeom.mjGEOM_BOX,
                                               mujoco.mjtGeom.mjGEOM_MESH))),
            secondary=ContactMatch(mode="geom", pattern="ground"),
            fields=("force",), reduce="netforce", num_slots=1, global_frame=True))
    cfg.rewards = copy.deepcopy(stock.rewards)
    cfg.terminations = copy.deepcopy(stock.terminations)
    # Goose's two feet and head are the relevant physical end effectors.
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "right_ankle_roll", "left_ankle_roll", "head_roll")
    cfg.terminations["contact_domain_failure"] = TerminationTermCfg(
        func=rigid_contact_limit_exceeded, params={"max_depth_m": .05})
    cfg.episode_length_s = 13.  # Includes the original12.52s reference/3s hold.
    cfg.seed = seed
    return cfg


def make_native_tracking_runner_cfg(*, seed=133):
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.seed = seed
    cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",)}
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    cfg.experiment_name = "goose_tracking_teacher"
    cfg.run_name = REVISION
    cfg.clip_actions = 1.
    return asdict(cfg)


def make_native_tracking_rsi_cfg(model_path, contract_path, motion_file, *,
                                cold_qpos, eligible_frames, num_envs=256, seed=133):
    """Same teacher/rewards/physics; a separately declared native birth course."""
    cfg = make_native_tracking_teacher_cfg(model_path, contract_path, motion_file,
        cold_qpos=cold_qpos, num_envs=num_envs, seed=seed)
    fields = vars(cfg.commands["motion"]).copy()
    fields["sampling_mode"] = "adaptive"
    cfg.commands["motion"] = AdmittedColdMotionCommandCfg(
        **fields, eligible_frames=tuple(eligible_frames))
    return cfg
