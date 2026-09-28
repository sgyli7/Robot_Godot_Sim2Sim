//! Optional development verification; never called by a release game by default.

use std::{path::Path, time::Duration};

use common_minigame::{
    clock::{ClockError, ClockSnapshot, FixedStepClock},
    events::{EventError, TickEvent, TickEventQueue},
};
use serde::Serialize;
use simulation_minigame::{
    BodyTorque, MAX_STEPS_PER_FRAME, PHYSICS_HZ, SimulationError, SimulationWorld,
    StepConfiguration, StepSnapshot, WorldCounts,
};
use thiserror::Error;

/// Foundation evidence intentionally excludes real robot and policy qualification.
#[derive(Debug, Serialize)]
pub struct FoundationReport {
    pub schema_version: u32,
    pub scope: &'static str,
    pub passed: bool,
    pub unqualified: Vec<&'static str>,
    pub physics_configuration: StepConfiguration,
    pub completed_clock: ClockSnapshot,
    pub physics_steps: u64,
    pub control_boundaries: u64,
    pub inference_count: u64,
    pub tick_zero_input_consumed_before_first_integration: bool,
    pub initial_counts: WorldCounts,
    pub counts_after_steps: WorldCounts,
    pub reset_counts: Vec<ResetRecord>,
    pub final_state: StepSnapshot,
    pub final_state_after_resets: StepSnapshot,
    pub events: Vec<EventRecord>,
}

/// Count-only reset evidence does not qualify full robot/contact reset behavior.
#[derive(Debug, Serialize)]
pub struct ResetRecord {
    pub episode_id: u64,
    pub global_step: u64,
    pub episode_step: u64,
    pub counts: WorldCounts,
    pub previous_handle_invalid: bool,
}

/// The event record names the pre-integration global boundary.
#[derive(Debug, Serialize)]
pub struct EventRecord {
    pub global_step: u64,
    pub sequence: u64,
    pub integrations_before_event: u64,
    pub label: &'static str,
}

/// Actual failures retain their cause and produce a nonzero CLI result.
#[derive(Debug, Error)]
pub enum VerificationError {
    #[error(transparent)]
    Clock(#[from] ClockError),
    #[error(transparent)]
    Event(#[from] EventError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("foundation verification failed: {0}")]
    Invariant(&'static str),
}

/// Run the same single-step world for 600 ticks and then 20 non-integrating resets.
///
/// The offline clock receives exactly ten seconds of debt. No policy Session is
/// loaded, and the count-only reset fixture cannot stand in for robot gates.
pub fn verify_foundation(output: &Path) -> Result<FoundationReport, VerificationError> {
    let mut clock = FixedStepClock::new(PHYSICS_HZ, MAX_STEPS_PER_FRAME)?;
    let mut world = SimulationWorld::foundation();
    let initial_counts = world.counts();
    let probe = world.probe_handle()?;
    let mut queue = TickEventQueue::default();
    queue.push(TickEvent {
        global_step: 0,
        sequence: 1,
        payload: "foundation_initial_input",
    })?;
    queue.push(TickEvent {
        global_step: 10,
        sequence: 2,
        payload: "foundation_end_torque_pulse",
    })?;
    let mut events = Vec::new();
    let mut rows = Vec::with_capacity(600);
    let mut elapsed = Duration::from_secs(10);
    while clock.global_step() < 600 {
        let budget = clock.begin_frame(elapsed);
        elapsed = Duration::ZERO;
        for _ in 0..budget {
            let before = world.snapshot();
            for event in queue.drain_for_tick(clock.global_step())? {
                events.push(EventRecord {
                    global_step: event.global_step,
                    sequence: event.sequence,
                    integrations_before_event: before.integration_count,
                    label: event.payload,
                });
            }
            let torque = if clock.global_step() < 10 {
                vec![BodyTorque {
                    body: probe,
                    world_torque: [0.0001, 0.0, 0.0],
                }]
            } else {
                Vec::new()
            };
            let snapshot = world.step_with_torques(&torque)?;
            clock.commit_tick()?;
            if snapshot.global_step != clock.global_step() {
                return Err(VerificationError::Invariant(
                    "clock/backend tick disagreement",
                ));
            }
            rows.push(snapshot);
        }
    }
    let final_state = world.snapshot();
    let counts_after_steps = world.counts();
    if final_state.integration_count != 600
        || final_state.torque_update_count != 600
        || counts_after_steps != initial_counts
        || clock.pending_ticks() != 0
    {
        return Err(VerificationError::Invariant(
            "600-step counts/debt mismatch",
        ));
    }
    let mut reset_counts = Vec::with_capacity(20);
    for _ in 0..20 {
        let previous = world.probe_handle()?;
        world.reset_probe()?;
        let counts = world.counts();
        let state = world.snapshot();
        let previous_handle_invalid = world.world.bodies.get(previous).is_none();
        if counts != initial_counts
            || state.global_step != 600
            || state.episode_step != 0
            || !previous_handle_invalid
        {
            return Err(VerificationError::Invariant(
                "reset changed scene/counts/global clock",
            ));
        }
        reset_counts.push(ResetRecord {
            episode_id: state.episode_id,
            global_step: state.global_step,
            episode_step: state.episode_step,
            counts,
            previous_handle_invalid,
        });
    }
    let tick_zero_input_consumed_before_first_integration = events
        .first()
        .is_some_and(|event| event.global_step == 0 && event.integrations_before_event == 0);
    if !tick_zero_input_consumed_before_first_integration || events.len() != 2 {
        return Err(VerificationError::Invariant(
            "input boundary coverage failed",
        ));
    }
    let report = FoundationReport {
        schema_version: 1,
        scope: "foundation_only",
        passed: true,
        unqualified: vec![
            "microduck_body_and_joints",
            "bam",
            "observation_contract",
            "onnx_policy_inference",
            "source_target_behavior",
            "robot_contact_reset",
            "science_station_collision",
            "rendering",
            "1080p_60fps",
        ],
        physics_configuration: world.configuration(),
        completed_clock: clock.snapshot(),
        physics_steps: final_state.integration_count,
        control_boundaries: final_state.torque_update_count,
        inference_count: 0,
        tick_zero_input_consumed_before_first_integration,
        initial_counts,
        counts_after_steps,
        reset_counts,
        final_state,
        final_state_after_resets: world.snapshot(),
        events,
    };
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("trace.json"), serde_json::to_vec_pretty(&rows)?)?;
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
