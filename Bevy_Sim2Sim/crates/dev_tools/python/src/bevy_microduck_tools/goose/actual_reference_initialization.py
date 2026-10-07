"""Own actual reference history at genuine native tracking episode resets.

The public Actor65, physical model and original drive are unchanged. A private
tracking teacher may use the installed native reference observations. This
module supplies reference births only; mjlab steps and RSL learning stay native.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from .native_tracking import (
    ReadOnlyMotionCommand, ReadOnlyMotionCommandCfg, make_native_tracking_teacher_cfg,
    native_motion_clip_ended)
from .mjlab_env import GooseDevelopmentEnv
from .moving_initialization import DATA_FIELDS, DRIVE_FIELDS, MANAGER_FIELDS
from .native_action_units import native_coordinate_action_rate_l2
from .mature_training import native_physical_action_std
from .artifacts import sha256

REVISION = "goose_actual_reference_history_v1"

class ActualHistoryMotionCommand(ReadOnlyMotionCommand):
    """Native adaptive proposals select genuine completed-history births."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.birth_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.birth_count = self.birth_frame.clone()

    def _resample_command(self, env_ids):
        action = self._env.action_manager.get_term('goose')
        if (action.drive._pending_action is not None
                or bool((action.drive.completed_ticks[env_ids] != 0).any())
                or not getattr(self._env, '_actual_reset_active', False)):
            raise RuntimeError('Actual reference sampling requires genuine episode reset')
        if self.cfg.sampling_mode == 'adaptive':
            MotionCommand._adaptive_sampling(self, env_ids)
            self.time_steps[env_ids] = self.time_steps[env_ids].clamp(max=self.motion.time_step_total-2)
        elif self.cfg.sampling_mode == 'start':
            self.time_steps[env_ids] = 0
        else:
            raise ValueError('Undeclared reference birth course')
        forced = getattr(self._env, '_forced_birth_frames', None)
        if forced is not None:
            if len(forced) != len(env_ids):
                raise ValueError('Explicit admission birth count changed')
            self.time_steps[env_ids] = forced
        self.birth_frame[env_ids] = self.time_steps[env_ids]
        self.birth_count[env_ids] += 1
        self.time_steps[env_ids] += 1

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        if env_ids is None and self.cfg.sampling_mode == 'adaptive':
            self.bin_failed_count = (self.cfg.adaptive_alpha*self._current_bin_failed
                + (1-self.cfg.adaptive_alpha)*self.bin_failed_count)
            self._current_bin_failed.zero_()

@dataclass(kw_only=True)
class ActualHistoryMotionCommandCfg(ReadOnlyMotionCommandCfg):
    def build(self, env):
        return ActualHistoryMotionCommand(self, env)

def make_actual_reference_tracking_cfg(model_path, contract_path, motion_file,
                                       actual_bank_path, *, num_envs=128,
                                       adaptive=True, seed=163, floor_limit_m=.005):
    """Native9-term private teacher with explicit actual-history RSI births.

    Native action-rate is evaluated in the same fixed coordinates as the
    mapped actor/Gaussian. Other8 native functions, parameters and all9 weights
    are unchanged. This course does not grant full fallen-state qualification.
    """
    if not 0 < floor_limit_m <= .005:
        raise ValueError("Actual-history reference course requires the fixed5mm gate")
    model_path, contract_path, motion_file = map(Path,
        (model_path, contract_path, motion_file))
    bank = torch.load(Path(actual_bank_path), weights_only=True)
    if bank.get('revision') != 'goose_full_actual_partial_reference_bank_v1':
        raise ValueError("Wrong actual reference bank schema")
    if (bank.get('model_sha256') != sha256(model_path)
            or bank.get('contract_sha256') != sha256(contract_path)
            or bank.get('motion_sha256') != sha256(motion_file)
            or bank.get('dt_s') != .02 or bank.get('decimation') != 1
            or not bank.get('source_physical_valid') or not bank.get('states')):
        raise ValueError("Actual reference provenance/model/timing changed")
    cfg = make_native_tracking_teacher_cfg(Path(model_path), Path(contract_path),
        Path(motion_file), cold_qpos=bank['states'][0]['data']['qpos'].numpy(),
        num_envs=num_envs, seed=seed)
    fields = vars(cfg.commands['motion']).copy()
    fields['sampling_mode'] = 'adaptive' if adaptive else 'start'
    cfg.commands['motion'] = ActualHistoryMotionCommandCfg(**fields)
    cfg.rewards['action_rate_l2'].func = native_coordinate_action_rate_l2
    cfg.rewards['action_rate_l2'].params = dict(public_action_scale=native_physical_action_std(
        json.loads(Path(contract_path).read_text())))
    cfg.terminations['contact_domain_failure'].params['max_depth_m'] = floor_limit_m
    cfg.terminations['motion_clip_end'] = TerminationTermCfg(func=native_motion_clip_ended, time_out=True)
    cfg.episode_length_s = (len(bank['states'])-1)*.02
    return cfg

class ActualReferenceEnv(GooseDevelopmentEnv):
    """Preserve integration, driver and action history at real episode reset.

    A scoped reset preserves the continuing worlds' complete explicit state
    inventory. No reference state writes occur during the native physical step.
    """
    def __init__(self, cfg, device, *, bank_path, bank_sha256):
        self._bank_path = Path(bank_path)
        self._bank_digest = bank_sha256
        if sha256(self._bank_path) != bank_sha256:
            raise ValueError('Actual reference bank changed')
        self._bank = torch.load(self._bank_path, weights_only=True)
        if (self._bank.get('revision') != 'goose_full_actual_partial_reference_bank_v1'
                or not self._bank.get('source_physical_valid') or not self._bank.get('states')):
            raise ValueError('Invalid actual reference bank schema')
        self._actual_reset_active = False
        self._forced_birth_frames = None
        super().__init__(cfg, device)
        action = self.action_manager.get_term('goose')
        drive = action.drive
        drive.cap = drive.peak.expand(self.num_envs, -1).clone()
        drive.force_lower = -drive.cap.clone()
        drive.force_upper = drive.cap.clone()
        if (self._bank['model_sha256'] != sha256(cfg.actions['goose'].model_path)
                or self._bank['contract_sha256'] != sha256(cfg.actions['goose'].contract_path)
                or self._bank['motion_sha256'] != sha256(Path(cfg.commands['motion'].motion_file))
                or self._bank['dt_s'] != .02 or self._bank['decimation'] != 1):
            raise ValueError('Actual reference model/motion/timing changed')
        for frame, state in enumerate(self._bank['states']):
            if (state['frame'] != frame
                    or state['source_tick'] != self._bank['source_completed_tick']+state['frame']):
                raise ValueError('Actual completed reference provenance changed')
            for n in DATA_FIELDS:
                t = getattr(self.sim.data, n)
                if state['data'][n].shape != t.shape[1:]:
                    raise ValueError('Native reference field shape changed: '+n)
                if not bool(torch.isfinite(state['data'][n]).all()):
                    raise ValueError('Nonfinite native reference field: '+n)
            for n in DRIVE_FIELDS:
                t = getattr(drive, n)
                if state['drive'][n].shape != t.shape[1:]:
                    raise ValueError('Original drive history shape changed: '+n)
                if (state['drive'][n].dtype != t.dtype
                        or not bool(torch.isfinite(state['drive'][n]).all())):
                    raise ValueError('Invalid original drive history: '+n)
            if (int(state['drive']['prepared_ticks']) != state['source_tick']
                    or int(state['drive']['completed_ticks']) != state['source_tick']
                    or abs(float(state['data']['time'])-state['source_tick']*.02) > 2e-6
                    or state['last_actions'].shape != (3, 18)
                    or not bool(torch.isfinite(state['last_actions']).all())
                    or bool((state['last_actions'].abs() > 1.).any())
                    or state['actuator_forcerange'].shape != (18, 2)
                    or not bool(torch.isfinite(state['actuator_forcerange']).all())):
                raise ValueError('Actual completed driver/action/force history changed')

    def _world_buffers(self):
        action = self.action_manager.get_term('goose')
        motion = self.command_manager.get_term('motion')
        out = {}
        for prefix, owner, fields in (
                ('data', self.sim.data, DATA_FIELDS), ('drive', action.drive, DRIVE_FIELDS),
                ('action', action, ('_raw', '_effort', '_solver_force', 'commands', 'ground_forces')),
                ('action_manager', self.action_manager, MANAGER_FIELDS),
                ('motion', motion, ('time_steps', 'birth_frame', 'birth_count', 'time_left',
                    'command_counter', 'body_pos_relative_w', 'body_quat_relative_w'))):
            for n in fields:
                out[prefix+'.'+n] = getattr(owner, n)
        out['model.actuator_forcerange'] = self.sim.model.actuator_forcerange
        out['episode.age'] = self.episode_length_buf
        out['episode.manual_pending'] = self._manual_reset_pending
        for name, sensor in self.scene.sensors.items():
            for n, v in (getattr(sensor, '_history_state', None) or {}).items():
                out['sensor.'+name+'.'+n] = v
        return out

    @torch.no_grad()
    def reset(self, **kwargs):
        action = self.action_manager.get_term('goose')
        if (self._actual_reset_active or action._before_time is not None
                or action.drive._pending_action is not None):
            raise RuntimeError('No reset during incomplete native integral')
        if sha256(self._bank_path) != self._bank_digest:
            raise ValueError('Actual reference bank changed before reset')
        ids = kwargs.get('env_ids')
        ids = (torch.arange(self.num_envs, device=self.device) if ids is None
            else torch.as_tensor(ids, device=self.device, dtype=torch.long))
        if (ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel()
                or bool((ids < 0).any()) or bool((ids >= self.num_envs).any())):
            raise ValueError('Actual reference reset requires distinct valid world IDs')
        forced = self._forced_birth_frames
        if forced is not None and (forced.ndim != 1 or len(forced) != len(ids)
                or bool((forced < 0).any())
                or bool((forced >= len(self._bank['states'])-1).any())):
            raise ValueError('Actual reference birth frame is outside the admitted finite clip')
        continuing = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        continuing[ids] = False
        before = {k: v[continuing].clone() for k, v in self._world_buffers().items()}
        self._actual_reset_active = True
        try:
            super().reset(**kwargs)
            motion = self.command_manager.get_term('motion')
            states = [self._bank['states'][int(f)] for f in motion.birth_frame[ids].cpu()]
            for n in DATA_FIELDS:
                target = getattr(self.sim.data, n)
                value = torch.stack([s['data'][n] for s in states]).to(target)
                if n == 'qpos':
                    value[:, action.root_q:action.root_q+3] += self.scene.env_origins[ids]
                target[ids] = value
            for n in DRIVE_FIELDS:
                target = getattr(action.drive, n)
                target[ids] = torch.stack([s['drive'][n] for s in states]).to(target)
            self.sim.model.actuator_forcerange[ids] = torch.stack(
                [s['actuator_forcerange'] for s in states]).to(self.sim.model.actuator_forcerange)
            for offset, n in enumerate(MANAGER_FIELDS):
                target = getattr(self.action_manager, n)
                target[ids] = torch.stack([s['last_actions'][2-offset] for s in states]).to(target)
            action._raw[ids] = self.action_manager.action[ids].double()
            action._effort[ids] = self.sim.data.ctrl[ids].double()
            action._solver_force[ids] = action.drive.last_tau[ids]
            action.commands[ids] = 0
            self.episode_length_buf[ids] = 0  # Relative course age; original drive clock is preserved.
            warm = self.sim.data.qacc_warmstart.clone()
            self.sim.forward()
            motion.update_relative_body_poses()
            self.sim.sense()
            self.sim.data.qacc_warmstart.copy_(warm)
            # Native global forward/sense must not consume a history Tick for
            # worlds that were not reset. Restore those explicit buffers.
            inventory = self._world_buffers()
            for k, value in before.items():
                inventory[k][continuing] = value
            for sensor in self.scene.sensors.values():
                sensor._invalidate_cache()
            self.obs_buf = self.observation_manager.compute(update_history=True, env_ids=ids)
            return self.obs_buf, self.extras
        finally:
            self._actual_reset_active = False
