//! Globally ordered input events are independent of episode-local clocks.

use std::collections::BTreeMap;

use thiserror::Error;

/// An event's stamp references the next global physics boundary.
#[derive(Clone, Debug)]
pub struct TickEvent<T> {
    pub global_step: u64,
    pub sequence: u64,
    pub payload: T,
}

/// An ordered queue with monotone sequence admission and no episode reset API.
#[derive(Debug)]
pub struct TickEventQueue<T> {
    pending: BTreeMap<(u64, u64), T>,
    last_sequence: Option<u64>,
    last_consumed_tick: Option<u64>,
}

/// Invalid replay/input ordering is rejected instead of replayed twice.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EventError {
    #[error("event sequence must increase strictly")]
    NonMonotoneSequence,
    #[error("cannot insert an event at an already consumed global boundary")]
    BoundaryAlreadyConsumed,
    #[error("global event-consumption time must not move backward")]
    TimeMovedBackward,
}

impl<T> Default for TickEventQueue<T> {
    fn default() -> Self {
        Self {
            pending: BTreeMap::new(),
            last_sequence: None,
            last_consumed_tick: None,
        }
    }
}

impl<T> TickEventQueue<T> {
    /// Admit events once in their original monotonically numbered stream.
    pub fn push(&mut self, event: TickEvent<T>) -> Result<(), EventError> {
        if self
            .last_sequence
            .is_some_and(|last| event.sequence <= last)
        {
            return Err(EventError::NonMonotoneSequence);
        }
        if self
            .last_consumed_tick
            .is_some_and(|last| event.global_step <= last)
        {
            return Err(EventError::BoundaryAlreadyConsumed);
        }
        self.last_sequence = Some(event.sequence);
        self.pending
            .insert((event.global_step, event.sequence), event.payload);
        Ok(())
    }

    /// Consume due events in tick/sequence order before the tick's integration.
    pub fn drain_for_tick(&mut self, global_step: u64) -> Result<Vec<TickEvent<T>>, EventError> {
        if self
            .last_consumed_tick
            .is_some_and(|last| global_step < last)
        {
            return Err(EventError::TimeMovedBackward);
        }
        self.last_consumed_tick = Some(global_step);
        let mut events = Vec::new();
        while self
            .pending
            .first_key_value()
            .is_some_and(|(&(tick, _), _)| tick <= global_step)
        {
            let ((tick, sequence), payload) = self.pending.pop_first().unwrap();
            events.push(TickEvent {
                global_step: tick,
                sequence,
                payload,
            });
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_zero_and_multiple_same_tick_events_are_consumed_once() {
        let mut queue = TickEventQueue::default();
        for sequence in [1, 2] {
            queue
                .push(TickEvent {
                    global_step: 0,
                    sequence,
                    payload: sequence,
                })
                .unwrap();
        }
        let events = queue.drain_for_tick(0).unwrap();
        assert_eq!(
            events.iter().map(|e| e.sequence).collect::<Vec<_>>(),
            [1, 2]
        );
        assert!(queue.drain_for_tick(0).unwrap().is_empty());
        assert!(queue.drain_for_tick(1).unwrap().is_empty());
        assert_eq!(
            queue.drain_for_tick(0).unwrap_err(),
            EventError::TimeMovedBackward
        );
        assert_eq!(
            queue.push(TickEvent {
                global_step: 1,
                sequence: 2,
                payload: 9
            }),
            Err(EventError::NonMonotoneSequence)
        );
    }
}
