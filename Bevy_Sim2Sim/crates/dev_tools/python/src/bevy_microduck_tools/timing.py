"""Physical-second causal delays; preserve the source sampling distribution."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from copy import deepcopy


class TimedHistory:
    """Reference interface shared conceptually with the Rust timed queue."""

    def __init__(self, fallback):
        self.fallback = deepcopy(fallback)
        self.samples = deque()

    def reset(self, fallback) -> None:
        self.fallback = deepcopy(fallback)
        self.samples.clear()

    def append(self, timestamp: float, value) -> None:
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("History timestamps must strictly increase")
        self.samples.append((timestamp, deepcopy(value)))

    def read(self, now: float, delay_seconds: float):
        if delay_seconds < 0:
            raise ValueError("Negative delay is acausal")
        index = bisect_right([time for time, _ in self.samples], now - delay_seconds) - 1
        return deepcopy(self.samples[index][1] if index >= 0 else self.fallback)


def install_delays(env, timing_plan: dict) -> None:
    """Install actual second-based adapters into upstream observation/actuator loops.

    mjlab private seams are pinned to 1.3.0 and covered by the source integration
    check. Circular storage remains GPU-native; timestamps and selection use f64.
    """
    import torch
    from mjlab.utils.buffers.delay_buffer import DelayBuffer

    class TimestampDelay(DelayBuffer):
        def __init__(self, original, plan, dt, fallback, *, before_integration=False):
            super().__init__(min_lag=min(plan["mapped_ticks"]), max_lag=max(plan["mapped_ticks"]),
                             batch_size=original.batch_size, device=original.device,
                             per_env=original.per_env, hold_prob=plan["hold_prob"],
                             update_period=plan["update_period_ticks"], per_env_phase=plan["per_env_phase"])
            self._delay_choices = torch.tensor(plan["sample_delay_seconds"], device=self.device, dtype=torch.float64)
            self._current_seconds = torch.zeros(self.batch_size, device=self.device, dtype=torch.float64)
            self._capacity = self.max_lag + 2
            self._timestamps = torch.full((self.batch_size, self._capacity), -torch.inf, device=self.device, dtype=torch.float64)
            self._origin = torch.full((self.batch_size,), env._sim_step_counter, device=self.device, dtype=torch.long)
            self._pointer = torch.full((self.batch_size,), -1, device=self.device, dtype=torch.long)
            self._last_append = torch.full((self.batch_size,), -torch.inf, device=self.device, dtype=torch.float64)
            self._last_compute = self._last_append.clone()
            self._history = None
            self._cached = None
            self._before_integration = before_integration
            self._latest_reset_counter = env._sim_step_counter
            self._dt = dt
            self._fallback_fn = fallback
            self._fallback = None
            self.fallback_count = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
            self.append_count = torch.zeros_like(self.fallback_count)
            self._append_key = None
            self._compute_key = None

        def now(self):
            # The environment increments this counter before actuator compute.
            ticks = env._sim_step_counter - self._origin - int(self._before_integration)
            return ticks.clamp_min(0).to(torch.float64) * env.physics_dt

        @property
        def is_initialized(self):
            return self._history is not None

        def reset(self, batch_ids=None):
            super().reset(batch_ids)
            ids = slice(None) if batch_ids is None else batch_ids
            self._origin[ids] = env._sim_step_counter
            self._latest_reset_counter = env._sim_step_counter
            self._timestamps[ids] = -torch.inf
            self._pointer[ids] = -1
            self._last_append[ids] = -torch.inf
            self._last_compute[ids] = -torch.inf
            self.append_count[ids] = 0
            self.fallback_count[ids] = 0
            self._current_seconds[ids] = 0
            self._append_key = None
            self._compute_key = None
            # Current fallback is recomputed after reset events, on next append.
            if self._fallback is not None:
                self._fallback[ids] = self._fallback_fn()[ids]
            self._fallback_pending = ids

        def append(self, data):
            if self._before_integration and env._sim_step_counter == self._latest_reset_counter:
                return  # Reset write/forward is outside the real motor-sample phase.
            if self._append_key == env._sim_step_counter:
                return  # Identical physical timestamp: retain its first sample.
            # PPO collects in inference_mode but resets later outside it. History
            # storage must remain mutable normal tensors across those boundaries.
            with torch.inference_mode(False), torch.no_grad():
                if self._fallback is None:
                    self._fallback = self._fallback_fn().detach().clone()
                    self._history = torch.zeros((self.batch_size, self._capacity, *data.shape[1:]), device=self.device, dtype=data.dtype)
                    self._cached = self._fallback.clone()
                if hasattr(self, "_fallback_pending"):
                    ids = self._fallback_pending
                    self._fallback[ids] = self._fallback_fn()[ids]
                    del self._fallback_pending
                timestamp = self.now()
                fresh = timestamp > self._last_append
                self._pointer[fresh] = (self._pointer[fresh] + 1) % self._capacity
                rows = fresh.nonzero(as_tuple=False).squeeze(-1)
                slots = self._pointer[rows]
                self._history[rows, slots] = data[rows]
                self._timestamps[rows, slots] = timestamp[rows]
                self._last_append[fresh] = timestamp[fresh]
                self.append_count += fresh
                self._append_key = env._sim_step_counter

        def _update_lags(self, fresh):
            if self.update_period:
                mask = (self._step_count + self._phase_offsets) % self.update_period == 0
            else:
                mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            mask &= fresh
            sampled = self._sample_lags(mask)
            self._current_lags.copy_(torch.where(mask, sampled, self._current_lags))
            self._step_count += fresh

        def _sample_lags(self, mask):
            size = (self.batch_size,) if self.per_env else (1,)
            choices = torch.randint(len(self._delay_choices), size, device=self.device, generator=self.generator)
            seconds = self._delay_choices[choices].expand(self.batch_size)
            if self.hold_prob:
                mask = mask & (torch.rand(self.batch_size, device=self.device) >= self.hold_prob)
            self._current_seconds.copy_(torch.where(mask, seconds, self._current_seconds))
            return torch.ceil(self._current_seconds / self._dt - 1e-12).to(torch.long)

        def compute(self):
            if self._before_integration and env._sim_step_counter == self._latest_reset_counter:
                fallback = self._fallback_fn()
                if self._cached is None:
                    return fallback
                warmup = env._sim_step_counter <= self._origin
                return torch.where(warmup[:, None], fallback, self._cached)
            if self._compute_key == env._sim_step_counter:
                return self._cached.clone()
            timestamp = self.now()
            fresh = timestamp > self._last_compute
            self._update_lags(fresh)
            cutoff = timestamp - self._current_seconds
            possible = self._timestamps <= cutoff.unsqueeze(-1) + 1e-12
            causal_times = torch.where(possible, self._timestamps, -torch.inf)
            newest, slots = causal_times.max(dim=-1)
            eligible = torch.isfinite(newest)
            delayed = self._history[torch.arange(self.batch_size, device=self.device), slots]
            shape = (self.batch_size,) + (1,) * (delayed.ndim - 1)
            result = torch.where(eligible.reshape(shape), delayed, self._fallback)
            self._cached.copy_(torch.where(fresh.reshape(shape), result, self._cached))
            self._last_compute[fresh] = timestamp[fresh]
            self.fallback_count += (~eligible) & fresh
            self._compute_key = env._sim_step_counter
            return self._cached.clone()

    manager = env.observation_manager
    for group, terms in timing_plan["observation"].items():
        for name, plan in terms.items():
            previous = manager._group_obs_term_delay_buffer[group][name]
            term_cfg = env.cfg.observations[group].terms[name]
            def fallback(term_cfg=term_cfg, name=name):
                value = term_cfg.func(env, **term_cfg.params)
                if name == "joint_vel":
                    return torch.zeros_like(value)
                if term_cfg.scale is not None:
                    value = value * torch.as_tensor(term_cfg.scale, device=env.device)
                return value
            manager._group_obs_term_delay_buffer[group][name] = TimestampDelay(previous, plan, 1 / 60, fallback)
    for actuator, plan in zip(env.scene["robot"].actuators, timing_plan["motor"], strict=True):
        def target_fallback(actuator=actuator):
            data = env.scene["robot"].data
            return data.default_joint_pos[:, actuator.target_ids] - data.encoder_bias[:, actuator.target_ids]
        actuator._delay_buffer = TimestampDelay(actuator._delay_buffer, plan, timing_plan["physics_dt"], target_fallback, before_integration=True)


def delay_diagnostics(env) -> dict:
    """Copy clocks only at diagnostic boundaries; no per-step host synchronization."""
    def record(buffer):
        def times(value):
            return [item if item != float("-inf") else None for item in value.detach().cpu().tolist()]
        return {"timestamps": [[value if value != float("-inf") else None for value in row]
                                for row in buffer._timestamps.detach().cpu().tolist()],
                "origin_physics_tick": buffer._origin.detach().cpu().tolist(),
                "append_count": buffer.append_count.detach().cpu().tolist(),
                "update_count": buffer._step_count.detach().cpu().tolist(),
                "phase_offset": buffer._phase_offsets.detach().cpu().tolist(),
                "last_sample_time": times(buffer._last_append),
                "last_read_time": times(buffer._last_compute),
                "current_delay_seconds": buffer._current_seconds.detach().cpu().tolist(),
                "fallback_count": buffer.fallback_count.detach().cpu().tolist()}
    return {"physics_tick": env._sim_step_counter,
            "observation": {group: {name: record(buffer) for name, buffer in terms.items() if hasattr(buffer, "append_count")}
                            for group, terms in env.observation_manager._group_obs_term_delay_buffer.items()},
            "motor": [record(actuator._delay_buffer) for actuator in env.scene["robot"].actuators]}
