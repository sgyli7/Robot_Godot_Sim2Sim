//! Explicit finite AGILE initialization followed by unchanged matched T1 chunks.
//! This development owner wraps one existing world; no load-time integrations,
//! synthetic VLA actions, contact sensors or automatic runtime fallback.

use super::{
    runner::{G1Measurement, G1ProgressCounts},
    task_objects::TaskObjectFrame,
    task_runner::{
        ArenaBodyStep, ArenaTaskCommand, ArenaTaskExecution, ArenaTaskRunner, ArenaTaskRunnerConfig,
    },
};
use robot_minigame::{RobotError, g1::definition::G1BodyFrame};
use serde::Serialize;

pub const STATIC_STARTUP_TICKS: u64 = 60;

#[derive(Clone, Debug)]
pub enum StaticStartupCommand {
    Startup { episode_id: u64 },
    OriginalVla(ArenaTaskCommand),
    VisualTransfer(super::static_transfer::StaticVisualTransferGoal),
    MemoryPlace(super::static_place::StaticMemoryPlaceGoal),
    ObservationWithdrawal(super::static_observe::StaticObservationWithdrawalGoal),
}
impl StaticStartupCommand {
    pub(super) fn validate(&self) -> Result<(), RobotError> {
        match self {
            Self::Startup { episode_id } if *episode_id != 0 => Ok(()),
            Self::OriginalVla(command) => command.validate(),
            Self::VisualTransfer(goal) => goal.validate(),
            Self::MemoryPlace(goal) => goal.validate(),
            Self::ObservationWithdrawal(goal) => goal.validate(),
            _ => Err(invalid("invalid explicit static startup episode")),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticStartupExecution {
    Startup {
        ticks: u64,
        stable_self_velocity_ticks: u32,
        completed: bool,
        ready: bool,
        original_vla_output: bool,
    },
    OriginalVla(ArenaTaskExecution),
    VisualTransfer(super::static_transfer::StaticVisualTransferStep),
    MemoryPlace(super::static_place::StaticMemoryPlaceStep),
    ObservationWithdrawal(super::static_observe::StaticObservationWithdrawalStep),
}

#[derive(Clone, Debug, Serialize)]
pub struct StaticStartupStep {
    pub execution: StaticStartupExecution,
    pub body: ArenaBodyStep,
}

pub(super) struct StaticStartupRunner {
    owner: ArenaTaskRunner,
    episode_id: u64,
    stable_ticks: u32,
    ready: bool,
    halted: bool,
    kinematics: super::static_transfer::StaticLeftPalmKinematics,
    last_original_execution: Option<ArenaTaskExecution>,
    transfer: Option<super::static_transfer::StaticVisualTransfer>,
    placement_geometry: Option<super::static_place::StaticPlacementGeometry>,
    placement: Option<super::static_place::StaticMemoryPlace>,
    observation_withdrawal: Option<super::static_observe::StaticObservationWithdrawal>,
}
impl StaticStartupRunner {
    pub(super) fn load(config: &ArenaTaskRunnerConfig) -> Result<Self, RobotError> {
        if config.body.profile() != task_minigame::types::TaskProfile::StaticApple {
            return Err(invalid("static startup requires original AGILE/N1.7"));
        }
        let super::task_runner::ArenaTaskBodyConfig::StaticAgile(body) = &config.body else {
            return Err(invalid("static transfer preparation requires AGILE"));
        };
        let definition = robot_minigame::g1::definition::G1Definition::load(
            &body.definition,
            &body.definition_sha256,
        )?;
        let kinematics = super::static_transfer::StaticLeftPalmKinematics::new(&definition)?;
        // Prepare immutable public geometry before the physical clock starts.
        let placement_geometry = body
            .task_objects
            .as_ref()
            .map(super::static_place::StaticPlacementGeometry::load)
            .transpose()?;
        Ok(Self {
            kinematics,
            last_original_execution: None,
            transfer: None,
            placement_geometry,
            placement: None,
            observation_withdrawal: None,
            owner: ArenaTaskRunner::load_static_predictive_constraint_diagnostic(config)?,
            episode_id: config.body.episode_id(),
            stable_ticks: 0,
            ready: false,
            halted: false,
        })
    }
    pub(super) fn initial_frame(&self) -> Result<G1BodyFrame, RobotError> {
        self.owner.initial_frame()
    }
    pub(super) fn measurement(&self) -> Result<G1Measurement, RobotError> {
        self.owner.measurement()
    }
    pub(super) fn task_objects(&self) -> Result<Option<TaskObjectFrame>, RobotError> {
        self.owner.task_object_frame()
    }
    pub(super) fn progress_counts(&self) -> G1ProgressCounts {
        let mut counts = self.owner.progress_counts();
        counts.halted |= self.halted;
        counts
    }
    pub(super) fn step_with_guard(
        &mut self,
        command: &StaticStartupCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<StaticStartupStep, RobotError> {
        if self.halted {
            return Err(invalid("static startup owner failed; reset required"));
        }
        let result = (|| {
            command.validate()?;
            match command {
                StaticStartupCommand::Startup { episode_id } => {
                    if *episode_id != self.episode_id
                        || self.owner.progress_counts().integration_count >= STATIC_STARTUP_TICKS
                    {
                        return Err(invalid(
                            "startup changed episode or exceeded finite initialization",
                        ));
                    }
                    let body = self.owner.diagnostic_static_startup_step(guard)?;
                    let ArenaBodyStep::StaticAgile(step) = &body else {
                        return Err(invalid("startup selected foreign body"));
                    };
                    // Self velocity and IMU only. Object/contact/world auditor
                    // fields do not participate in this admission decision.
                    let speed = step
                        .measurement
                        .root_velocity_source
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        .sqrt();
                    let [w, x, y, z] = step.measurement.root_rotation_wxyz;
                    let upright = 1. - 2. * (x * x + y * y);
                    if !speed.is_finite() || ![w, x, y, z].iter().all(|v| v.is_finite()) {
                        return Err(invalid("nonfinite startup self state"));
                    }
                    self.stable_ticks = if speed <= 0.03 && upright > 0.95 {
                        self.stable_ticks + 1
                    } else {
                        0
                    };
                    let completed = step.integration_count == STATIC_STARTUP_TICKS;
                    self.ready = completed && self.stable_ticks >= 20;
                    Ok(StaticStartupStep {
                        execution: StaticStartupExecution::Startup {
                            ticks: step.integration_count,
                            stable_self_velocity_ticks: self.stable_ticks,
                            completed,
                            ready: self.ready,
                            original_vla_output: false,
                        },
                        body,
                    })
                }
                StaticStartupCommand::OriginalVla(command) => {
                    if !self.ready
                        || command.chunk.profile != task_minigame::types::TaskProfile::StaticApple
                        || command.chunk.observation.episode_id != self.episode_id
                        || command.chunk.observation.sim_time_ns < STATIC_STARTUP_TICKS * 20_000_000
                    {
                        return Err(invalid(
                            "original task requires completed startup and a newer actual image",
                        ));
                    }
                    if self.transfer.is_some() {
                        return Err(invalid(
                            "original VLA cannot replace an admitted geometric phase",
                        ));
                    }
                    let step = self.owner.step_with_guard(command, guard)?;
                    self.last_original_execution = Some(step.execution.clone());
                    Ok(StaticStartupStep {
                        execution: StaticStartupExecution::OriginalVla(step.execution),
                        body: step.body,
                    })
                }
                StaticStartupCommand::VisualTransfer(goal) => {
                    goal.validate()?;
                    let state = self.owner.measurement()?;
                    if self.transfer.is_none() {
                        let execution = self.last_original_execution.as_ref().ok_or_else(|| {
                            invalid("static transfer has no original executed grasp")
                        })?;
                        if execution.frame_index != 39
                            || execution.admitted_chunks != 2
                            || state.source_tick != 140
                        {
                            return Err(invalid(
                                "static transfer requires two complete original chunks",
                            ));
                        }
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| invalid(e.to_string()))?
                            .as_millis() as u64;
                        if goal.observation.captured_at_unix_ms > now
                            || now - goal.observation.captured_at_unix_ms > 2000
                        {
                            return Err(invalid("static transfer current camera wall age expired"));
                        }
                        let command = self.owner.executed_static_command(execution)?;
                        self.transfer = Some(super::static_transfer::StaticVisualTransfer::new(
                            goal.clone(),
                            &state,
                            command,
                        )?);
                    }
                    let transfer = self.transfer.as_mut().unwrap();
                    if transfer.goal() != goal {
                        return Err(invalid("static transfer goal changed during execution"));
                    }
                    let step = transfer.update(&state, &self.kinematics)?;
                    let body = self
                        .owner
                        .step_static_visual_transfer_with_guard(&step.command, guard)?;
                    Ok(StaticStartupStep {
                        execution: StaticStartupExecution::VisualTransfer(step),
                        body,
                    })
                }
                StaticStartupCommand::ObservationWithdrawal(goal) => {
                    let state = self.owner.measurement()?;
                    if self.observation_withdrawal.is_none() {
                        let command = self
                            .placement
                            .as_ref()
                            .and_then(|p| p.completed_command())
                            .ok_or_else(|| {
                                invalid("observation withdrawal requires completed typed placement")
                            })?;
                        self.observation_withdrawal =
                            Some(super::static_observe::StaticObservationWithdrawal::new(
                                goal.clone(),
                                &state,
                                command,
                            )?);
                    }
                    let withdrawal = self.observation_withdrawal.as_mut().unwrap();
                    if withdrawal.goal() != goal {
                        return Err(invalid("observation withdrawal goal changed"));
                    }
                    let step = withdrawal.update(&state, &self.kinematics)?;
                    let body = self
                        .owner
                        .step_static_observation_withdrawal_with_guard(&step.command, guard)?;
                    Ok(StaticStartupStep {
                        execution: StaticStartupExecution::ObservationWithdrawal(step),
                        body,
                    })
                }
                StaticStartupCommand::MemoryPlace(goal) => {
                    let state = self.owner.measurement()?;
                    if self.placement.is_none() {
                        let command = self
                            .transfer
                            .as_ref()
                            .and_then(|t| t.completed_command())
                            .ok_or_else(|| {
                                invalid("placement requires completed typed static transfer")
                            })?;
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| invalid(e.to_string()))?
                            .as_millis() as u64;
                        if goal.current_observation.captured_at_unix_ms > now
                            || now - goal.current_observation.captured_at_unix_ms > 2000
                        {
                            return Err(invalid("placement current observation wall age expired"));
                        }
                        let geometry = self.placement_geometry.as_ref().ok_or_else(|| {
                            invalid("placement requires original public task geometry")
                        })?;
                        self.placement = Some(super::static_place::StaticMemoryPlace::new(
                            goal.clone(),
                            &state,
                            command,
                            geometry,
                        )?);
                    }
                    let placement = self.placement.as_mut().unwrap();
                    if placement.goal() != goal {
                        return Err(invalid("placement goal changed during execution"));
                    }
                    let step = placement.update(&state, &self.kinematics)?;
                    let body = self
                        .owner
                        .step_static_memory_place_with_guard(&step.command, guard)?;
                    Ok(StaticStartupStep {
                        execution: StaticStartupExecution::MemoryPlace(step),
                        body,
                    })
                }
            }
        })();
        if result.is_err() {
            self.halted = true;
        }
        result
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g1::worker::{G1WorkerPhase, StaticStartupWorker, TimedCommand, WorkerSnapshot};
    use std::{
        fs,
        sync::Arc,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    #[test]
    #[ignore = "requires explicit local frozen models and saved original T1 chunk; 121 real startup/task ticks"]
    fn real_static_startup_reset_and_stale_image_gate() -> Result<(), RobotError> {
        let path =
            std::env::var("G1_STATIC_STARTUP_TEST_CONFIG").map_err(|e| invalid(e.to_string()))?;
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(path).map_err(|e| invalid(e.to_string()))?)
                .map_err(|e| invalid(e.to_string()))?;
        let config: ArenaTaskRunnerConfig =
            serde_json::from_value(value["runner"].clone()).map_err(|e| invalid(e.to_string()))?;
        let initial_episode = config.body.episode_id();
        let fixture =
            std::env::var("G1_STATIC_STARTUP_TEST_CHUNK").map_err(|e| invalid(e.to_string()))?;
        let saved: task_minigame::policy::PolicyActionChunk =
            serde_json::from_slice(&fs::read(fixture).map_err(|e| invalid(e.to_string()))?)
                .map_err(|e| invalid(e.to_string()))?;
        let worker = StaticStartupWorker::spawn_static_startup(config)?;
        let trace = worker.subscribe_steps(256)?;
        let wait = |episode: u64,
                    ticks: u64,
                    phase: G1WorkerPhase|
         -> Result<Arc<WorkerSnapshot<StaticStartupStep>>, RobotError> {
            let end = Instant::now() + Duration::from_secs(15);
            loop {
                if let Some(snapshot) = worker.take_latest() {
                    if snapshot.episode_id == episode
                        && snapshot.timing.episode_integrations == ticks
                        && snapshot.phase == phase
                    {
                        return Ok(snapshot);
                    }
                    if snapshot.phase == G1WorkerPhase::Failed && phase != G1WorkerPhase::Failed {
                        return Err(invalid(snapshot.reason.clone().unwrap_or_default()));
                    }
                }
                if Instant::now() > end {
                    return Err(invalid("bounded startup test timeout"));
                }
                std::thread::sleep(Duration::from_millis(4));
            }
        };
        let initialize = |episode: u64| -> Result<(), RobotError> {
            wait(episode, 0, G1WorkerPhase::Paused)?;
            worker.submit(TimedCommand {
                episode_id: episode,
                valid_until_sim_ns: 60 * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(8),
                command: StaticStartupCommand::Startup {
                    episode_id: episode,
                },
            })?;
            let snapshot = wait(episode, 60, G1WorkerPhase::Paused)?;
            assert!(matches!(
                snapshot.step.as_ref().map(|s| &s.execution),
                Some(StaticStartupExecution::Startup {
                    completed: true,
                    ready: true,
                    ..
                })
            ));
            std::thread::sleep(Duration::from_millis(100));
            if let Some(snapshot) = worker.take_latest() {
                assert_eq!(snapshot.timing.episode_integrations, 60);
                assert_eq!(snapshot.phase, G1WorkerPhase::Paused);
            }
            Ok(())
        };
        let original = |episode: u64, image_tick: u64| -> StaticStartupCommand {
            let mut chunk = saved.clone();
            // Offline admission fixture only: no actual RGB/VLA is claimed.
            chunk.observation.episode_id = episode;
            chunk.observation.sim_time_ns = image_tick * 20_000_000;
            chunk.observation.captured_at_unix_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            StaticStartupCommand::OriginalVla(ArenaTaskCommand {
                chunk: Arc::new(chunk),
                scheduled_start_sim_ns: None,
            })
        };
        initialize(initial_episode)?;
        worker.submit(TimedCommand {
            episode_id: initial_episode,
            valid_until_sim_ns: 61 * 20_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(8),
            command: original(initial_episode, 0),
        })?;
        wait(initial_episode, 60, G1WorkerPhase::Failed)?;
        worker.reset()?;
        let second = initial_episode + 1;
        initialize(second)?;
        worker.submit(TimedCommand {
            episode_id: second,
            valid_until_sim_ns: 61 * 20_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(8),
            command: original(second, 60),
        })?;
        let first_task = wait(second, 61, G1WorkerPhase::Paused)?;
        assert!(
            matches!(first_task.step.as_ref().map(|s|&s.execution),Some(StaticStartupExecution::OriginalVla(e)) if e.frame_index==0 && e.admitted_chunks==1)
        );
        worker.reset()?;
        let third = second + 1;
        let reset = wait(third, 0, G1WorkerPhase::Paused)?;
        assert_eq!(reset.timing.total_integrations, 121);
        assert_eq!(reset.timing.total_successful_inferences, 121);
        assert!(
            worker
                .submit(TimedCommand {
                    episode_id: second,
                    valid_until_sim_ns: 60 * 20_000_000,
                    valid_until_wall: Instant::now() + Duration::from_secs(8),
                    command: StaticStartupCommand::Startup { episode_id: second }
                })
                .is_err()
        );
        let records = trace.drain();
        assert_eq!(records.len(), 121);
        assert_eq!(trace.dropped_records(), 0);
        if let Ok(output) = std::env::var("G1_STATIC_STARTUP_TEST_TRACE") {
            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output)
                .map_err(|e| invalid(e.to_string()))?;
            for record in &records {
                writeln!(file,"{}",serde_json::json!({"episode_id":record.episode_id,"generation":record.generation,
                    "owner_total_integrations":record.timing.total_integrations,
                    "owner_episode_integrations":record.timing.episode_integrations,
                    "execution":record.step.execution,"body":record.step.body,
                    "offline_admission_fixture":true,"fresh_vla_calls":0,"qualified":false})).map_err(|e|invalid(e.to_string()))?;
            }
        }
        println!(
            "actual_native_integrations=121 body_inferences=121 fresh_vla_calls=0 old_image_rejected_without_integration=true reset_zero_integrations=true old_episode_rejected=true"
        );
        worker.shutdown()?;
        Ok(())
    }
}
