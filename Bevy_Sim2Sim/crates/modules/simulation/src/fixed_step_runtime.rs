//! Fixed-step lifecycle driver over one [`SimulationWorld`].
//!
//! Each due tick drains ordered events, synchronizes the scene, infers once,
//! integrates once, commits the shared clock, and publishes that snapshot.

use std::time::Duration;

use common_minigame::clock::{ClockError, ClockSnapshot, FixedStepClock};
use common_minigame::events::{EventError, TickEvent, TickEventQueue};
use thiserror::Error;

use crate::{
    BodyTorque, MAX_STEPS_PER_FRAME, PHYSICS_HZ, SimulationError, SimulationWorld, StepSnapshot,
};

/// Ticks completed in one display frame, plus the whole-tick debt still retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOutcome {
    /// Physics ticks integrated and published during this frame.
    pub completed_steps: u64,
    /// Whole ticks still owed after this frame's budget.
    pub pending_ticks: u128,
}

/// Scene, inference, and publish hooks for one physics boundary.
pub trait TickController<E> {
    /// Inferences this controller has completed.
    fn inference_count(&self) -> u64;

    /// Apply due events before inference and integration.
    fn synchronize_scene(
        &mut self,
        world: &mut SimulationWorld,
        events: Vec<TickEvent<E>>,
        boundary: u64,
    ) -> Result<(), String>;

    /// Run one inference and return the torques for this boundary.
    fn infer_and_actuate(
        &mut self,
        world: &SimulationWorld,
        boundary: u64,
    ) -> Result<Vec<BodyTorque>, String>;

    /// Observe the snapshot of the boundary that just completed.
    fn publish(&mut self, snapshot: &StepSnapshot) -> Result<(), String>;
}

/// Terminal failure of the fixed-step lifecycle.
#[derive(Debug, Error)]
pub enum FixedStepError {
    /// The supplied world has already integrated or advanced its global step.
    #[error("simulation world has already advanced")]
    WorldAlreadyAdvanced,
    /// A previous tick failed; later calls cannot continue or replay it.
    #[error("fixed-step runtime is halted: {0}")]
    Halted(String),
    /// The controller rejected this boundary.
    #[error("tick controller failed: {0}")]
    Controller(String),
    /// Inference did not advance by exactly one call.
    #[error("inference count changed from {before} to {after}, expected an increment of 1")]
    InferenceCount { before: u64, after: u64 },
    /// The completed snapshot and the clock do not name the same boundary.
    #[error("clock and simulation boundary disagree")]
    StepDisagreement,
    /// The shared clock rejected the tick.
    #[error(transparent)]
    Clock(#[from] ClockError),
    /// The ordered event queue rejected the operation.
    #[error(transparent)]
    Event(#[from] EventError),
    /// The physics world rejected the step.
    #[error(transparent)]
    Simulation(#[from] SimulationError),
}

/// Owner of one world, the shared fixed-step clock, and its ordered events.
pub struct FixedStepRuntime<E> {
    world: SimulationWorld,
    clock: FixedStepClock,
    events: TickEventQueue<E>,
    halted: Option<String>,
}

impl<E> FixedStepRuntime<E> {
    /// Take a world that has not yet integrated or advanced.
    ///
    /// Bodies and colliders may already be present. A nonzero `global_step`
    /// or `integration_count` is rejected.
    pub fn new(world: SimulationWorld) -> Result<Self, FixedStepError> {
        if world.global_step != 0 || world.integration_count != 0 {
            return Err(FixedStepError::WorldAlreadyAdvanced);
        }
        Ok(Self {
            world,
            clock: FixedStepClock::new(PHYSICS_HZ, MAX_STEPS_PER_FRAME)?,
            events: TickEventQueue::default(),
            halted: None,
        })
    }

    /// Queue one event. Admission failures leave the runtime running.
    pub fn enqueue(&mut self, event: TickEvent<E>) -> Result<(), FixedStepError> {
        self.ensure_active()?;
        self.events.push(event)?;
        Ok(())
    }

    /// Freeze the clock. Existing debt and fractional time stay in place.
    pub fn pause(&mut self) -> Result<(), FixedStepError> {
        self.ensure_active()?;
        self.clock.set_paused(true);
        Ok(())
    }

    /// Unfreeze the clock. The next frame supplies time since resume only.
    pub fn resume(&mut self) -> Result<(), FixedStepError> {
        self.ensure_active()?;
        self.clock.set_paused(false);
        Ok(())
    }

    /// Run every due tick for this display frame, in clock order.
    ///
    /// `elapsed` is wall time since the previous running frame. Paused
    /// intervals must be omitted; the clock already retained that debt.
    /// The first lifecycle error halts the runtime permanently.
    pub fn advance_frame(
        &mut self,
        elapsed: Duration,
        controller: &mut impl TickController<E>,
    ) -> Result<FrameOutcome, FixedStepError> {
        self.ensure_active()?;
        let budget = self.clock.begin_frame(elapsed);
        let mut completed_steps = 0u64;
        for _ in 0..budget {
            self.advance_tick(controller)?;
            completed_steps += 1;
        }
        Ok(FrameOutcome {
            completed_steps,
            pending_ticks: self.clock.pending_ticks(),
        })
    }

    /// Read body state without granting mutable world access.
    pub fn world_snapshot(&self) -> StepSnapshot {
        self.world.snapshot()
    }

    /// Read the shared clock, including retained debt.
    pub fn clock_snapshot(&self) -> ClockSnapshot {
        self.clock.snapshot()
    }

    fn ensure_active(&self) -> Result<(), FixedStepError> {
        match &self.halted {
            Some(reason) => Err(FixedStepError::Halted(reason.clone())),
            None => Ok(()),
        }
    }

    fn halt<T>(&mut self, result: Result<T, FixedStepError>) -> Result<T, FixedStepError> {
        result.map_err(|error| {
            if self.halted.is_none() {
                self.halted = Some(error.to_string());
            }
            error
        })
    }

    fn advance_tick(
        &mut self,
        controller: &mut impl TickController<E>,
    ) -> Result<(), FixedStepError> {
        let boundary = self.clock.global_step();
        let drained = self.events.drain_for_tick(boundary).map_err(Into::into);
        let events = self.halt(drained)?;

        let synced = controller
            .synchronize_scene(&mut self.world, events, boundary)
            .map_err(FixedStepError::Controller);
        self.halt(synced)?;
        // A scene callback may change bodies and colliders, but it must not
        // integrate. Detect that before inference and before our own step.
        let scene_kept_boundary =
            if self.world.global_step == boundary && self.world.integration_count == boundary {
                Ok(())
            } else {
                Err(FixedStepError::StepDisagreement)
            };
        self.halt(scene_kept_boundary)?;

        let before = controller.inference_count();
        let inferred = controller
            .infer_and_actuate(&self.world, boundary)
            .map_err(FixedStepError::Controller);
        let torques = self.halt(inferred)?;
        let after = controller.inference_count();
        if after.checked_sub(before) != Some(1) {
            return self.halt(Err(FixedStepError::InferenceCount { before, after }));
        }

        let stepped = self.world.step_with_torques(&torques).map_err(Into::into);
        let snapshot = self.halt(stepped)?;
        let committed = self.clock.commit_tick().map_err(Into::into);
        let clock_step = self.halt(committed)?;
        let agreed =
            if snapshot.global_step == clock_step && snapshot.integration_count == clock_step {
                Ok(())
            } else {
                Err(FixedStepError::StepDisagreement)
            };
        self.halt(agreed)?;

        let published = controller
            .publish(&snapshot)
            .map_err(FixedStepError::Controller);
        self.halt(published)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use common_minigame::events::EventError;

    /// Wall time for `ticks` at [`PHYSICS_HZ`], plus one nanosecond.
    ///
    /// Integer division of the rate would otherwise drop the last tick.
    fn duration_for_ticks(ticks: u64) -> Duration {
        Duration::from_nanos(ticks * 1_000_000_000 / u64::from(PHYSICS_HZ) + 1)
    }

    fn foundation_runtime() -> FixedStepRuntime<u64> {
        FixedStepRuntime::new(SimulationWorld::foundation()).unwrap()
    }

    #[derive(Default)]
    struct CountingController {
        inferences: u64,
        syncs: u64,
        publishes: u64,
    }

    impl TickController<u64> for CountingController {
        fn inference_count(&self) -> u64 {
            self.inferences
        }

        fn synchronize_scene(
            &mut self,
            _world: &mut SimulationWorld,
            _events: Vec<TickEvent<u64>>,
            _boundary: u64,
        ) -> Result<(), String> {
            self.syncs += 1;
            Ok(())
        }

        fn infer_and_actuate(
            &mut self,
            _world: &SimulationWorld,
            _boundary: u64,
        ) -> Result<Vec<BodyTorque>, String> {
            self.inferences += 1;
            Ok(Vec::new())
        }

        fn publish(&mut self, snapshot: &StepSnapshot) -> Result<(), String> {
            assert_eq!(snapshot.global_step, snapshot.integration_count);
            self.publishes += 1;
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Fault {
        None,
        SyncAt(u64),
        SkipInferenceIncrement,
        DoubleInference,
        Publish,
        NonFiniteTorque,
        IntegrateDuringSync,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Phase {
        Synchronize {
            boundary: u64,
            global_step: u64,
            integration_count: u64,
            events: Vec<(u64, u64, u64)>,
        },
        Infer {
            boundary: u64,
            global_step: u64,
            integration_count: u64,
        },
        Publish {
            global_step: u64,
            integration_count: u64,
        },
    }

    struct RecordingController {
        inferences: u64,
        publishes: u64,
        trace: Vec<Phase>,
        fault: Fault,
    }

    impl RecordingController {
        fn new(fault: Fault) -> Self {
            Self {
                inferences: 0,
                publishes: 0,
                trace: Vec::new(),
                fault,
            }
        }
    }

    impl TickController<u64> for RecordingController {
        fn inference_count(&self) -> u64 {
            self.inferences
        }

        fn synchronize_scene(
            &mut self,
            world: &mut SimulationWorld,
            events: Vec<TickEvent<u64>>,
            boundary: u64,
        ) -> Result<(), String> {
            self.trace.push(Phase::Synchronize {
                boundary,
                global_step: world.global_step,
                integration_count: world.integration_count,
                events: events
                    .iter()
                    .map(|event| (event.global_step, event.sequence, event.payload))
                    .collect(),
            });
            if matches!(self.fault, Fault::SyncAt(step) if step == boundary) {
                return Err("scene sync failed".to_string());
            }
            if matches!(self.fault, Fault::IntegrateDuringSync) {
                world
                    .step_with_torques(&[])
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        }

        fn infer_and_actuate(
            &mut self,
            world: &SimulationWorld,
            boundary: u64,
        ) -> Result<Vec<BodyTorque>, String> {
            self.trace.push(Phase::Infer {
                boundary,
                global_step: world.global_step,
                integration_count: world.integration_count,
            });
            match self.fault {
                Fault::SkipInferenceIncrement => Ok(Vec::new()),
                Fault::DoubleInference => {
                    self.inferences += 2;
                    Ok(Vec::new())
                }
                Fault::NonFiniteTorque => {
                    self.inferences += 1;
                    let body = world.probe_handle().map_err(|error| error.to_string())?;
                    Ok(vec![BodyTorque {
                        body,
                        world_torque: [f32::NAN, 0.0, 0.0],
                    }])
                }
                _ => {
                    self.inferences += 1;
                    Ok(Vec::new())
                }
            }
        }

        fn publish(&mut self, snapshot: &StepSnapshot) -> Result<(), String> {
            self.publishes += 1;
            self.trace.push(Phase::Publish {
                global_step: snapshot.global_step,
                integration_count: snapshot.integration_count,
            });
            if matches!(self.fault, Fault::Publish) {
                return Err("publish failed".to_string());
            }
            Ok(())
        }
    }

    fn assert_halted(runtime: &mut FixedStepRuntime<u64>, controller: &mut RecordingController) {
        let trace = controller.trace.clone();
        let inferences = controller.inferences;
        let publishes = controller.publishes;
        let integration = runtime.world_snapshot().integration_count;
        let clock_step = runtime.clock_snapshot().global_step;
        let second = runtime
            .advance_frame(Duration::from_secs(1), controller)
            .unwrap_err();
        assert!(matches!(second, FixedStepError::Halted(_)));
        assert_eq!(controller.trace, trace);
        assert_eq!(controller.inferences, inferences);
        assert_eq!(controller.publishes, publishes);
        assert_eq!(runtime.world_snapshot().integration_count, integration);
        assert_eq!(runtime.clock_snapshot().global_step, clock_step);
        assert!(matches!(
            runtime.enqueue(TickEvent {
                global_step: 100,
                sequence: 100,
                payload: 1,
            }),
            Err(FixedStepError::Halted(_))
        ));
        assert!(matches!(runtime.pause(), Err(FixedStepError::Halted(_))));
        assert!(matches!(runtime.resume(), Err(FixedStepError::Halted(_))));
    }

    #[test]
    fn constructor_accepts_idle_foundation_and_rejects_advanced_world() {
        let runtime = foundation_runtime();
        assert_eq!(runtime.clock_snapshot().rate_hz, PHYSICS_HZ);
        assert_eq!(runtime.clock_snapshot().global_step, 0);
        assert_eq!(runtime.world_snapshot().global_step, 0);
        assert_eq!(runtime.world_snapshot().integration_count, 0);

        let mut advanced = SimulationWorld::foundation();
        advanced.step_with_torques(&[]).unwrap();
        match FixedStepRuntime::<u64>::new(advanced) {
            Ok(_) => panic!("advanced world was accepted"),
            Err(error) => assert!(matches!(error, FixedStepError::WorldAlreadyAdvanced)),
        }
    }

    #[test]
    fn first_boundary_orders_events_inference_step_and_publish() {
        let mut runtime = foundation_runtime();
        runtime
            .enqueue(TickEvent {
                global_step: 0,
                sequence: 1,
                payload: 7,
            })
            .unwrap();
        let mut controller = RecordingController::new(Fault::None);
        let outcome = runtime
            .advance_frame(duration_for_ticks(1), &mut controller)
            .unwrap();
        assert_eq!(
            outcome,
            FrameOutcome {
                completed_steps: 1,
                pending_ticks: 0,
            }
        );
        assert_eq!(
            controller.trace,
            vec![
                Phase::Synchronize {
                    boundary: 0,
                    global_step: 0,
                    integration_count: 0,
                    events: vec![(0, 1, 7)],
                },
                Phase::Infer {
                    boundary: 0,
                    global_step: 0,
                    integration_count: 0,
                },
                Phase::Publish {
                    global_step: 1,
                    integration_count: 1,
                },
            ]
        );
        assert_eq!(controller.inferences, 1);
        assert_eq!(controller.publishes, 1);
        assert_eq!(runtime.clock_snapshot().global_step, 1);
        assert_eq!(runtime.world_snapshot().global_step, 1);
        assert_eq!(runtime.world_snapshot().integration_count, 1);
        assert_eq!(runtime.world_snapshot().torque_update_count, 1);

        let rejected = runtime.enqueue(TickEvent {
            global_step: 0,
            sequence: 2,
            payload: 8,
        });
        assert!(matches!(
            rejected,
            Err(FixedStepError::Event(EventError::BoundaryAlreadyConsumed))
        ));
        let next = runtime
            .advance_frame(duration_for_ticks(1), &mut controller)
            .unwrap();
        assert_eq!(next.completed_steps, 1);
        assert_eq!(controller.inferences, 2);
        assert_eq!(controller.publishes, 2);
    }

    #[test]
    fn six_hundred_ticks_match_inference_integration_and_publish() {
        let mut runtime = foundation_runtime();
        let mut controller = CountingController::default();
        let mut elapsed = Duration::from_secs(10);
        let mut frames = 0u32;
        while runtime.clock_snapshot().global_step < 600 {
            let outcome = runtime.advance_frame(elapsed, &mut controller).unwrap();
            assert!(outcome.completed_steps > 0);
            assert!(outcome.completed_steps <= u64::from(MAX_STEPS_PER_FRAME));
            assert_eq!(
                outcome.pending_ticks,
                runtime.clock_snapshot().pending_ticks
            );
            elapsed = Duration::ZERO;
            frames += 1;
            assert!(frames <= 75);
        }
        assert_eq!(frames, 75);
        assert_eq!(controller.syncs, 600);
        assert_eq!(controller.inferences, 600);
        assert_eq!(controller.publishes, 600);
        let world = runtime.world_snapshot();
        assert_eq!(world.global_step, 600);
        assert_eq!(world.integration_count, 600);
        assert_eq!(world.torque_update_count, 600);
        assert_eq!(runtime.clock_snapshot().global_step, 600);
        assert_eq!(runtime.clock_snapshot().pending_ticks, 0);
    }

    #[test]
    fn half_second_hitch_budgets_eight_and_retains_debt() {
        let mut runtime = foundation_runtime();
        let mut controller = CountingController::default();
        let mut elapsed = Duration::from_millis(500);
        let mut outcomes = Vec::new();
        for _ in 0..4 {
            let outcome = runtime.advance_frame(elapsed, &mut controller).unwrap();
            elapsed = Duration::ZERO;
            assert_eq!(
                outcome.pending_ticks,
                runtime.clock_snapshot().pending_ticks
            );
            outcomes.push((outcome.completed_steps, outcome.pending_ticks));
        }
        assert_eq!(outcomes, [(8, 22), (8, 14), (8, 6), (6, 0)]);
        assert_eq!(controller.inferences, 30);
        assert_eq!(controller.publishes, 30);
        assert_eq!(runtime.world_snapshot().integration_count, 30);
        assert_eq!(runtime.clock_snapshot().global_step, 30);
    }

    #[test]
    fn pause_and_resume_preserve_debt() {
        let mut runtime = foundation_runtime();
        let mut controller = CountingController::default();
        let first = runtime
            .advance_frame(Duration::from_millis(510), &mut controller)
            .unwrap();
        assert_eq!(first.completed_steps, 8);
        runtime
            .enqueue(TickEvent {
                global_step: 8,
                sequence: 1,
                payload: 1,
            })
            .unwrap();
        let before = runtime.clock_snapshot();
        runtime.pause().unwrap();
        let paused = runtime
            .advance_frame(Duration::from_secs(7), &mut controller)
            .unwrap();
        assert_eq!(paused.completed_steps, 0);
        assert_eq!(controller.inferences, 8);
        let during = runtime.clock_snapshot();
        assert!(during.paused);
        assert_eq!(during.global_step, before.global_step);
        assert_eq!(during.pending_ticks, before.pending_ticks);
        assert_eq!(during.pending_seconds, before.pending_seconds);
        runtime.resume().unwrap();
        assert!(!runtime.clock_snapshot().paused);
        let resumed = runtime
            .advance_frame(Duration::ZERO, &mut controller)
            .unwrap();
        assert_eq!(resumed.completed_steps, 8);
        assert_eq!(controller.inferences, 16);
        assert_eq!(controller.publishes, 16);
        assert_eq!(runtime.clock_snapshot().global_step, 16);
        assert_eq!(runtime.world_snapshot().integration_count, 16);
    }

    #[test]
    fn same_boundary_events_arrive_in_sequence_order() {
        let mut runtime = foundation_runtime();
        for (global_step, sequence, payload) in [(0, 1, 10), (0, 2, 20), (1, 3, 30), (2, 4, 40)] {
            runtime
                .enqueue(TickEvent {
                    global_step,
                    sequence,
                    payload,
                })
                .unwrap();
        }
        let mut controller = RecordingController::new(Fault::None);
        runtime
            .advance_frame(duration_for_ticks(2), &mut controller)
            .unwrap();
        assert_eq!(
            controller.trace,
            vec![
                Phase::Synchronize {
                    boundary: 0,
                    global_step: 0,
                    integration_count: 0,
                    events: vec![(0, 1, 10), (0, 2, 20)],
                },
                Phase::Infer {
                    boundary: 0,
                    global_step: 0,
                    integration_count: 0,
                },
                Phase::Publish {
                    global_step: 1,
                    integration_count: 1,
                },
                Phase::Synchronize {
                    boundary: 1,
                    global_step: 1,
                    integration_count: 1,
                    events: vec![(1, 3, 30)],
                },
                Phase::Infer {
                    boundary: 1,
                    global_step: 1,
                    integration_count: 1,
                },
                Phase::Publish {
                    global_step: 2,
                    integration_count: 2,
                },
            ]
        );
    }

    #[test]
    fn failure_halts_and_rejects_later_frames() {
        let mut runtime = foundation_runtime();
        runtime
            .enqueue(TickEvent {
                global_step: 0,
                sequence: 1,
                payload: 1,
            })
            .unwrap();
        let mut controller = RecordingController::new(Fault::SyncAt(0));
        let error = runtime
            .advance_frame(duration_for_ticks(4), &mut controller)
            .unwrap_err();
        assert!(
            matches!(error, FixedStepError::Controller(ref reason) if reason == "scene sync failed")
        );
        assert_eq!(runtime.world_snapshot().integration_count, 0);
        assert_eq!(runtime.clock_snapshot().global_step, 0);
        assert_halted(&mut runtime, &mut controller);

        let mut runtime = foundation_runtime();
        runtime
            .enqueue(TickEvent {
                global_step: 1,
                sequence: 1,
                payload: 2,
            })
            .unwrap();
        let mut controller = RecordingController::new(Fault::SyncAt(1));
        let error = runtime
            .advance_frame(duration_for_ticks(4), &mut controller)
            .unwrap_err();
        assert!(matches!(error, FixedStepError::Controller(_)));
        assert_eq!(controller.inferences, 1);
        assert_eq!(controller.publishes, 1);
        assert_eq!(runtime.world_snapshot().integration_count, 1);
        assert!(controller.trace.iter().any(|phase| matches!(
            phase,
            Phase::Synchronize {
                boundary: 1,
                events,
                ..
            } if events == &vec![(1, 1, 2)]
        )));
        assert_halted(&mut runtime, &mut controller);

        let mut runtime = foundation_runtime();
        let mut controller = RecordingController::new(Fault::SkipInferenceIncrement);
        let error = runtime
            .advance_frame(duration_for_ticks(1), &mut controller)
            .unwrap_err();
        assert!(matches!(
            error,
            FixedStepError::InferenceCount {
                before: 0,
                after: 0
            }
        ));
        assert_eq!(runtime.world_snapshot().torque_update_count, 0);
        assert_eq!(controller.publishes, 0);
        assert_halted(&mut runtime, &mut controller);

        let mut runtime = foundation_runtime();
        let mut controller = RecordingController::new(Fault::DoubleInference);
        let error = runtime
            .advance_frame(duration_for_ticks(1), &mut controller)
            .unwrap_err();
        assert!(matches!(
            error,
            FixedStepError::InferenceCount {
                before: 0,
                after: 2
            }
        ));
        assert_eq!(runtime.world_snapshot().integration_count, 0);
        assert_halted(&mut runtime, &mut controller);

        let mut runtime = foundation_runtime();
        let mut controller = RecordingController::new(Fault::NonFiniteTorque);
        let error = runtime
            .advance_frame(duration_for_ticks(1), &mut controller)
            .unwrap_err();
        assert!(matches!(
            error,
            FixedStepError::Simulation(SimulationError::NonFiniteTorque)
        ));
        assert_eq!(controller.inferences, 1);
        assert_eq!(controller.publishes, 0);
        assert_eq!(runtime.world_snapshot().torque_update_count, 0);
        assert_eq!(runtime.clock_snapshot().global_step, 0);
        assert_halted(&mut runtime, &mut controller);

        let mut runtime = foundation_runtime();
        let mut controller = RecordingController::new(Fault::Publish);
        let error = runtime
            .advance_frame(duration_for_ticks(2), &mut controller)
            .unwrap_err();
        assert!(
            matches!(error, FixedStepError::Controller(ref reason) if reason == "publish failed")
        );
        assert_eq!(controller.publishes, 1);
        assert_eq!(runtime.world_snapshot().integration_count, 1);
        assert_eq!(runtime.clock_snapshot().global_step, 1);
        assert_halted(&mut runtime, &mut controller);

        let mut runtime = foundation_runtime();
        let mut controller = RecordingController::new(Fault::IntegrateDuringSync);
        let error = runtime
            .advance_frame(duration_for_ticks(1), &mut controller)
            .unwrap_err();
        assert!(matches!(error, FixedStepError::StepDisagreement));
        assert_eq!(controller.publishes, 0);
        assert_eq!(controller.inferences, 0);
        assert_eq!(runtime.world_snapshot().integration_count, 1);
        assert_eq!(runtime.clock_snapshot().global_step, 0);
        assert_halted(&mut runtime, &mut controller);
    }
}
