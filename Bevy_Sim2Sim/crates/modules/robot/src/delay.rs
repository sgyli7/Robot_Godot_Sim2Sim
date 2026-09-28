//! Causal zero-order-hold delay, with explicit reset prehistory.

use crate::RobotError;
use std::collections::VecDeque;

/// Delay quantization is shared with the source adapter and recorded as evidence.
pub fn latency_ticks(seconds: f64, hz: u32) -> Result<u64, RobotError> {
    if !seconds.is_finite() || seconds < 0.0 || hz == 0 || seconds > 10.0 {
        return Err(RobotError::Contract(
            "invalid history latency or frequency".into(),
        ));
    }
    // Only removes floating-point error at an exact integer boundary.
    Ok((seconds * f64::from(hz) - 1e-10).max(0.0).ceil() as u64)
}

/// The reset value represents known constant prehistory, never a future sample.
pub struct TimedHistory<T> {
    samples: VecDeque<(u64, T)>,
    fallback: T,
    maximum_lag: u64,
}

impl<T: Clone> TimedHistory<T> {
    pub fn new(maximum_lag: u64, fallback: T) -> Result<Self, RobotError> {
        if maximum_lag > 600 {
            return Err(RobotError::Contract("history exceeds ten seconds".into()));
        }
        Ok(Self {
            samples: VecDeque::new(),
            fallback,
            maximum_lag,
        })
    }

    pub fn reset(&mut self, fallback: T) {
        self.samples.clear();
        self.fallback = fallback;
    }

    pub fn push(&mut self, tick: u64, value: T) -> Result<(), RobotError> {
        if self
            .samples
            .back()
            .is_some_and(|(previous, _)| *previous >= tick)
        {
            return Err(RobotError::Contract(
                "history ticks must strictly increase".into(),
            ));
        }
        self.samples.push_back((tick, value));
        while self.samples.len() > self.maximum_lag as usize + 1 {
            self.samples.pop_front();
        }
        Ok(())
    }

    /// Returns the selected value and whether explicit reset prehistory was used.
    pub fn sample(&self, now: u64, lag: u64) -> Result<(T, bool), RobotError> {
        if lag > self.maximum_lag {
            return Err(RobotError::Contract("lag exceeds allocated history".into()));
        }
        let Some(cutoff) = now.checked_sub(lag) else {
            return Ok((self.fallback.clone(), true));
        };
        let selected = self.samples.iter().rev().find(|(tick, _)| *tick <= cutoff);
        Ok(match selected {
            Some((_, value)) => (value.clone(), false),
            None => (self.fallback.clone(), true),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_delay_preserves_the_four_source_samples() {
        let samples = [0.015, 0.020, 0.025, 0.030];
        let mapped: Vec<_> = samples
            .into_iter()
            .map(|value| latency_ticks(value, 60).unwrap())
            .collect();
        assert_eq!(mapped, [1, 2, 2, 2]);
        assert_eq!(latency_ticks(1.28, 60).unwrap(), 77);
    }

    #[test]
    fn cold_start_and_reset_cannot_read_future_or_old_episode() {
        let mut history = TimedHistory::new(2, -1).unwrap();
        history.push(0, 10).unwrap();
        assert_eq!(history.sample(0, 2).unwrap(), (-1, true));
        history.push(1, 20).unwrap();
        history.push(2, 30).unwrap();
        assert_eq!(history.sample(2, 2).unwrap(), (10, false));
        assert!(history.push(2, 40).is_err());
        history.reset(-2);
        history.push(0, 50).unwrap();
        assert_eq!(history.sample(0, 1).unwrap(), (-2, true));
    }
}
