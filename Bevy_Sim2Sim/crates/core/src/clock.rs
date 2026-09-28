//! Integer accumulation preserves fractional frames and overdue fixed steps.

use std::time::Duration;

use serde::Serialize;
use thiserror::Error;

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// A fixed-step accumulator whose time quantum is one nanosecond times its rate.
#[derive(Debug)]
pub struct FixedStepClock {
    rate_hz: u32,
    max_steps_per_frame: u32,
    pending_quanta: u128,
    remaining_frame_steps: u32,
    global_step: u64,
    paused: bool,
}

/// Readable timing evidence independent of a particular simulation.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ClockSnapshot {
    pub rate_hz: u32,
    pub global_step: u64,
    pub simulation_seconds: f64,
    pub pending_ticks: u128,
    pub pending_seconds: f64,
    pub paused: bool,
}

/// Invalid clock configuration or a tick committed outside its frame budget.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ClockError {
    #[error("fixed-step frequency and frame budget must both be positive")]
    InvalidConfiguration,
    #[error("cannot commit a tick while paused")]
    Paused,
    #[error("no fixed tick is available in the current frame budget")]
    NoTickAvailable,
    #[error("global tick counter overflow")]
    CounterOverflow,
}

impl FixedStepClock {
    /// Construct a clock without assumptions about game or episode semantics.
    pub fn new(rate_hz: u32, max_steps_per_frame: u32) -> Result<Self, ClockError> {
        if rate_hz == 0 || max_steps_per_frame == 0 {
            return Err(ClockError::InvalidConfiguration);
        }
        Ok(Self {
            rate_hz,
            max_steps_per_frame,
            pending_quanta: 0,
            remaining_frame_steps: 0,
            global_step: 0,
            paused: false,
        })
    }

    /// Add elapsed wall time and expose at most the configured frame budget.
    ///
    /// Paused frames contribute no wall time. Existing debt and fractional ticks
    /// remain intact; callers must not send the whole paused interval on resume.
    pub fn begin_frame(&mut self, elapsed: Duration) -> usize {
        self.remaining_frame_steps = 0;
        if self.paused {
            return 0;
        }
        self.pending_quanta += elapsed.as_nanos() * u128::from(self.rate_hz);
        self.remaining_frame_steps = self
            .pending_ticks()
            .min(u128::from(self.max_steps_per_frame)) as u32;
        self.remaining_frame_steps as usize
    }

    /// Commit only after the corresponding simulation step completes.
    pub fn commit_tick(&mut self) -> Result<u64, ClockError> {
        if self.paused {
            return Err(ClockError::Paused);
        }
        if self.remaining_frame_steps == 0 || self.pending_ticks() == 0 {
            return Err(ClockError::NoTickAvailable);
        }
        let next = self
            .global_step
            .checked_add(1)
            .ok_or(ClockError::CounterOverflow)?;
        self.pending_quanta -= NANOS_PER_SECOND;
        self.remaining_frame_steps -= 1;
        self.global_step = next;
        Ok(next)
    }

    /// Freeze/unfreeze time; resuming requires a fresh frame budget.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        self.remaining_frame_steps = 0;
    }

    /// Number of completed ticks, including across episode resets.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Whole overdue ticks; fractional remainder remains in the accumulator.
    pub fn pending_ticks(&self) -> u128 {
        self.pending_quanta / NANOS_PER_SECOND
    }

    /// Snapshot includes debt rather than silently discarding clipped time.
    pub fn snapshot(&self) -> ClockSnapshot {
        ClockSnapshot {
            rate_hz: self.rate_hz,
            global_step: self.global_step,
            simulation_seconds: self.global_step as f64 / f64::from(self.rate_hz),
            pending_ticks: self.pending_ticks(),
            pending_seconds: self.pending_quanta as f64
                / (NANOS_PER_SECOND as f64 * f64::from(self.rate_hz)),
            paused: self.paused,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_frames(fps: u64) -> Vec<u64> {
        let mut clock = FixedStepClock::new(60, 8).unwrap();
        let mut ticks = Vec::new();
        let mut last_nanos = 0;
        for frame in 1..=fps * 10 {
            let target_nanos = frame * 1_000_000_000 / fps;
            let budget = clock.begin_frame(Duration::from_nanos(target_nanos - last_nanos));
            last_nanos = target_nanos;
            for _ in 0..budget {
                ticks.push(clock.global_step());
                clock.commit_tick().unwrap();
            }
        }
        assert_eq!(clock.pending_ticks(), 0);
        ticks
    }

    #[test]
    fn frame_rate_does_not_change_tick_sequence() {
        let expected: Vec<_> = (0..600).collect();
        for fps in [30, 60, 144] {
            assert_eq!(run_frames(fps), expected);
        }
    }

    #[test]
    fn half_second_hitch_retains_all_thirty_ticks() {
        let mut clock = FixedStepClock::new(60, 8).unwrap();
        let mut budgets = Vec::new();
        let mut elapsed = Duration::from_millis(500);
        while clock.global_step() < 30 {
            let budget = clock.begin_frame(elapsed);
            elapsed = Duration::ZERO;
            budgets.push(budget);
            for _ in 0..budget {
                clock.commit_tick().unwrap();
            }
            assert!(clock.commit_tick().is_err());
        }
        assert_eq!(budgets, [8, 8, 8, 6]);
        assert_eq!(clock.pending_ticks(), 0);
    }

    #[test]
    fn pause_preserves_debt_and_fractional_time() {
        let mut clock = FixedStepClock::new(60, 8).unwrap();
        clock.begin_frame(Duration::from_millis(510));
        clock.commit_tick().unwrap();
        let before = clock.snapshot();
        clock.set_paused(true);
        assert_eq!(clock.begin_frame(Duration::from_secs(7)), 0);
        assert_eq!(clock.commit_tick(), Err(ClockError::Paused));
        assert_eq!(clock.snapshot().pending_seconds, before.pending_seconds);
        clock.set_paused(false);
        assert_eq!(clock.begin_frame(Duration::ZERO), 8);
        clock.commit_tick().unwrap();
        assert_eq!(clock.global_step(), 2);
    }
}
