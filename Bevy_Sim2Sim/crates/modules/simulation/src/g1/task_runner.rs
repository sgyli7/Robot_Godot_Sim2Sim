//! Owner-side task action scheduling over distinct native body controllers.
//!
//! Only decoded commands enter this boundary. The existing runner still owns
//! every motor/contact and performs exactly one integration. No network request,
//! rendered frame or task-object truth is read from this control path.

use std::{
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use robot_minigame::{RobotError, g1::definition::G1Definition};
use serde::{Deserialize, Serialize};
use task_minigame::{
    policy::{
        ACTION_JOINT_NAMES, ARENA_ACTION_PERIOD_NS, ActionLimits, PolicyActionChunk,
        PolicyActionQueue, PolicyTick, profile_contract,
    },
    types::{ObservationStamp, TaskProfile},
};

use super::{
    agile_runner::{AgileRunner, AgileRunnerConfig, AgileStep},
    runner::{G1Measurement, G1ProgressCounts, G1Runner, G1RunnerConfig, G1Step},
    task_objects::TaskObjectFrame,
    task_policy::{ArenaControllerCommand, controller_command},
};

/// A profile is selected by its concrete body configuration, never guessed from
/// a model reply or substituted when a different body fails.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArenaTaskBodyConfig {
    StaticAgile(AgileRunnerConfig),
    MobileHomieV2(G1RunnerConfig),
}

impl ArenaTaskBodyConfig {
    pub fn profile(&self) -> TaskProfile {
        match self {
            Self::StaticAgile(_) => TaskProfile::StaticApple,
            Self::MobileHomieV2(_) => TaskProfile::MobileBox,
        }
    }

    pub fn episode_id(&self) -> u64 {
        match self {
            Self::StaticAgile(config) => config.episode_id,
            Self::MobileHomieV2(config) => config.episode_id,
        }
    }

    pub(crate) fn with_episode(&self, episode_id: u64) -> Self {
        match self {
            Self::StaticAgile(config) => Self::StaticAgile(AgileRunnerConfig {
                episode_id,
                ..config.clone()
            }),
            Self::MobileHomieV2(config) => Self::MobileHomieV2(G1RunnerConfig {
                episode_id,
                ..config.clone()
            }),
        }
    }

    fn definition_identity(&self) -> (&Path, &str) {
        match self {
            Self::StaticAgile(config) => (&config.definition, &config.definition_sha256),
            Self::MobileHomieV2(config) => (&config.definition, &config.definition_sha256),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArenaTaskRunnerConfig {
    pub body: ArenaTaskBodyConfig,
    /// Explicit controller envelope, further constrained by the frozen URDF.
    pub limits: ActionLimits,
    pub max_observation_age_ns: u64,
    /// Wall-clock admission age also expires while the simulation is paused.
    pub max_observation_wall_age_ms: u64,
}

/// Immutable, bounded whole-chunk replacement. Re-submitting an identical chunk
/// does not restart its execution timeline; changing it under the same sequence
/// is rejected before another control update.
#[derive(Clone, Debug)]
pub struct ArenaTaskCommand {
    pub chunk: Arc<PolicyActionChunk>,
}

impl ArenaTaskCommand {
    pub(crate) fn validate(&self) -> Result<(), RobotError> {
        let chunk = &self.chunk;
        let contract = profile_contract(chunk.profile);
        if chunk.observation.episode_id == 0
            || chunk.sequence_id == 0
            || chunk.model_revision != contract.revision
            || chunk.action_period_ns != ARENA_ACTION_PERIOD_NS
            || chunk.frames.len() != contract.action_horizon
            || chunk.frames.iter().any(|frame| {
                !frame.base_height_m.is_finite()
                    || frame.base_height_m <= 0.
                    || frame.joint_targets().iter().any(|v| !v.is_finite())
                    || frame.navigate_mps_rps.iter().any(|v| !v.is_finite())
            })
        {
            return Err(error("invalid matched task chunk before owner submission"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ArenaTaskExecution {
    pub profile: TaskProfile,
    pub sequence_id: u64,
    pub frame_index: usize,
    pub observation: ObservationStamp,
    pub execution_start_sim_ns: u64,
    pub observation_age_ns: u64,
    pub observation_wall_age_ms: u64,
    pub decoded_waist_targets_rad: [f32; 3],
    /// Admission is counted once per accepted chunk, never per WBC call.
    pub admitted_chunks: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArenaBodyStep {
    StaticAgile(Box<AgileStep>),
    MobileHomieV2(Box<G1Step>),
}

#[derive(Clone, Debug, Serialize)]
pub struct ArenaTaskStep {
    pub execution: ArenaTaskExecution,
    pub body: ArenaBodyStep,
}

enum ArenaBodyRunner {
    StaticAgile(AgileRunner),
    MobileHomieV2(G1Runner),
}

/// Scheduling state is exclusively advanced by the physics owner. The renderer
/// can replace an input chunk but cannot choose the action index or owner clock.
struct TaskChunkExecutor {
    profile: TaskProfile,
    limits: ActionLimits,
    queue: PolicyActionQueue,
    accepted: Option<Arc<PolicyActionChunk>>,
    admitted_chunks: u64,
    max_observation_wall_age_ms: u64,
}

impl TaskChunkExecutor {
    fn new(config: &ArenaTaskRunnerConfig) -> Result<Self, RobotError> {
        let profile = config.body.profile();
        Ok(Self {
            profile,
            limits: config.limits.clone(),
            queue: PolicyActionQueue::new(
                config.body.episode_id(),
                profile,
                config.limits.clone(),
                config.max_observation_age_ns,
            )
            .map_err(error)?,
            accepted: None,
            admitted_chunks: 0,
            max_observation_wall_age_ms: config.max_observation_wall_age_ms,
        })
    }

    fn next(
        &mut self,
        command: &ArenaTaskCommand,
        sim_ns: u64,
        now_wall_ms: u64,
    ) -> Result<(ArenaControllerCommand, ArenaTaskExecution), RobotError> {
        command.validate()?;
        if self.accepted.as_deref() != Some(command.chunk.as_ref()) {
            let wall_age = now_wall_ms
                .checked_sub(command.chunk.observation.captured_at_unix_ms)
                .ok_or_else(|| error("task observation is from a future wall clock"))?;
            if wall_age > self.max_observation_wall_age_ms {
                return Err(error("task observation expired before owner admission"));
            }
            // Same-sequence mutation and all stale/profile/limit failures are
            // rejected by whole-chunk admission before replacement.
            self.queue
                .accept((*command.chunk).clone(), sim_ns)
                .map_err(error)?;
            self.admitted_chunks += 1;
            self.accepted = Some(command.chunk.clone());
        }
        let PolicyTick::Action {
            sequence_id,
            frame_index,
            action,
            observation,
            execution_start_sim_ns,
            observation_age_ns,
        } = self.queue.tick(sim_ns).map_err(error)?
        else {
            return Err(error(
                "task action unavailable; simulation halted, reset required",
            ));
        };
        let admitted = controller_command(self.profile, &action, &self.limits).map_err(error)?;
        Ok((
            admitted.controller,
            ArenaTaskExecution {
                profile: self.profile,
                sequence_id,
                frame_index,
                observation,
                execution_start_sim_ns,
                observation_age_ns,
                observation_wall_age_ms: now_wall_ms
                    .saturating_sub(observation.captured_at_unix_ms),
                decoded_waist_targets_rad: admitted.decoded_waist_targets_rad,
                admitted_chunks: self.admitted_chunks,
            },
        ))
    }
}

/// One existing native runner and one small action queue live on the same owner.
pub struct ArenaTaskRunner {
    body: ArenaBodyRunner,
    executor: TaskChunkExecutor,
    halted: bool,
}

impl ArenaTaskRunner {
    pub fn load(config: &ArenaTaskRunnerConfig) -> Result<Self, RobotError> {
        config.limits.validate().map_err(error)?;
        let (path, hash) = config.body.definition_identity();
        let definition = G1Definition::load(path, hash)?;
        for (i, name) in ACTION_JOINT_NAMES.iter().enumerate() {
            let source = definition
                .model()
                .joints
                .iter()
                .find(|joint| joint.name == *name)
                .ok_or_else(|| error(format!("missing frozen task joint {name}")))?;
            if config.limits.joint_min_rad[i] < source.limits[0] as f32
                || config.limits.joint_max_rad[i] > source.limits[1] as f32
            {
                return Err(error(format!(
                    "task envelope exceeds frozen source joint limit: {name}"
                )));
            }
        }
        let executor = TaskChunkExecutor::new(config)?;
        let body = match &config.body {
            ArenaTaskBodyConfig::StaticAgile(body) => {
                ArenaBodyRunner::StaticAgile(AgileRunner::load(body)?)
            }
            ArenaTaskBodyConfig::MobileHomieV2(body) => {
                ArenaBodyRunner::MobileHomieV2(G1Runner::load(body)?)
            }
        };
        Ok(Self {
            body,
            executor,
            halted: false,
        })
    }

    pub fn measurement(&self) -> Result<G1Measurement, RobotError> {
        match &self.body {
            ArenaBodyRunner::StaticAgile(body) => body.measurement(),
            ArenaBodyRunner::MobileHomieV2(body) => body.measurement(),
        }
    }

    pub fn initial_frame(&self) -> Result<robot_minigame::g1::definition::G1BodyFrame, RobotError> {
        match &self.body {
            ArenaBodyRunner::StaticAgile(body) => body.initial_frame(),
            ArenaBodyRunner::MobileHomieV2(body) => body.initial_frame(),
        }
    }

    pub fn task_object_frame(&self) -> Result<Option<TaskObjectFrame>, RobotError> {
        match &self.body {
            ArenaBodyRunner::StaticAgile(body) => body.task_object_frame(),
            ArenaBodyRunner::MobileHomieV2(body) => body.task_object_frame(),
        }
    }

    pub fn progress_counts(&self) -> G1ProgressCounts {
        let mut counts = match &self.body {
            ArenaBodyRunner::StaticAgile(body) => body.progress_counts(),
            ArenaBodyRunner::MobileHomieV2(body) => body.progress_counts(),
        };
        counts.halted |= self.halted;
        counts
    }

    pub fn step_with_guard(
        &mut self,
        command: &ArenaTaskCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<ArenaTaskStep, RobotError> {
        if self.halted {
            return Err(error("task owner is halted; reset required"));
        }
        let result = (|| {
            guard()?;
            let sim_ns = self.measurement()?.sim_time_ns;
            let now_wall_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(error)?
                .as_millis()
                .try_into()
                .map_err(error)?;
            let (command, execution) = self.executor.next(command, sim_ns, now_wall_ms)?;
            let body = match (&mut self.body, command) {
                (
                    ArenaBodyRunner::StaticAgile(body),
                    ArenaControllerCommand::StaticAgile(command),
                ) => ArenaBodyStep::StaticAgile(Box::new(body.step_with_guard(&command, guard)?)),
                (
                    ArenaBodyRunner::MobileHomieV2(body),
                    ArenaControllerCommand::MobileHomieV2(command),
                ) => ArenaBodyStep::MobileHomieV2(Box::new(body.step_with_guard(&command, guard)?)),
                _ => {
                    return Err(error(
                        "task/body profile mismatch before physical actuation",
                    ));
                }
            };
            Ok(ArenaTaskStep { execution, body })
        })();
        if result.is_err() {
            self.halted = true;
            self.executor.queue.stop();
        }
        result
    }
}

fn error(value: impl std::fmt::Debug) -> RobotError {
    RobotError::Contract(format!("Arena task owner: {value:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use task_minigame::policy::PolicyActionFrame;

    fn executor() -> TaskChunkExecutor {
        let limits = ActionLimits {
            joint_min_rad: [-3.; 31],
            joint_max_rad: [3.; 31],
            base_height_min_m: 0.4,
            base_height_max_m: 0.9,
            navigation_abs_max: [1.; 3],
        };
        TaskChunkExecutor {
            profile: TaskProfile::StaticApple,
            queue: PolicyActionQueue::new(
                1,
                TaskProfile::StaticApple,
                limits.clone(),
                1_000_000_000,
            )
            .unwrap(),
            limits,
            accepted: None,
            admitted_chunks: 0,
            max_observation_wall_age_ms: 1000,
        }
    }

    fn command() -> ArenaTaskCommand {
        ArenaTaskCommand {
            chunk: Arc::new(PolicyActionChunk {
                profile: TaskProfile::StaticApple,
                observation: ObservationStamp {
                    episode_id: 1,
                    frame_id: 1,
                    sim_time_ns: 0,
                    captured_at_unix_ms: 0,
                },
                sequence_id: 1,
                model_revision: profile_contract(TaskProfile::StaticApple).revision.into(),
                action_period_ns: ARENA_ACTION_PERIOD_NS,
                frames: (0..40)
                    .map(|i| PolicyActionFrame {
                        left_arm: [i as f32 * 0.01; 7],
                        right_arm: [0.; 7],
                        left_hand: [0.; 7],
                        right_hand: [0.; 7],
                        waist: [0.; 3],
                        base_height_m: 0.75,
                        navigate_mps_rps: [0.; 3],
                    })
                    .collect(),
            }),
        }
    }

    #[test]
    fn repeated_whole_chunk_does_not_restart_native_controller_frames() {
        let mut owner = executor();
        let command = command();
        for i in 0..40 {
            let (controller, frame) = owner
                .next(&command, 300_000_000 + i * 20_000_000, 500 + i * 20)
                .unwrap();
            assert_eq!(frame.frame_index, i as usize);
            assert_eq!(frame.admitted_chunks, 1);
            assert_eq!(frame.execution_start_sim_ns, 300_000_000);
            assert_eq!(frame.observation_age_ns, 300_000_000 + i * 20_000_000);
            let ArenaControllerCommand::StaticAgile(command) = controller else {
                panic!("wrong body");
            };
            assert_eq!(command.upper_positions[0], i as f32 * 0.01);
        }
        assert!(owner.next(&command, 1_100_000_000, 1300).is_err());
    }

    #[test]
    fn same_sequence_mutation_and_profile_switch_cannot_replace_owner_timeline() {
        let mut owner = executor();
        let command = command();
        owner.next(&command, 0, 500).unwrap();
        let mut changed = (*command.chunk).clone();
        changed.frames[1].left_arm[0] = 2.;
        assert!(
            owner
                .next(
                    &ArenaTaskCommand {
                        chunk: Arc::new(changed)
                    },
                    20_000_000,
                    500
                )
                .is_err()
        );
        let (_, frame) = owner.next(&command, 20_000_000, 500).unwrap();
        assert_eq!(frame.frame_index, 1);
        let mut changed = (*command.chunk).clone();
        changed.profile = TaskProfile::MobileBox;
        changed.sequence_id = 2;
        changed.model_revision = profile_contract(TaskProfile::MobileBox).revision.into();
        changed.frames.resize(50, changed.frames[0].clone());
        assert!(
            owner
                .next(
                    &ArenaTaskCommand {
                        chunk: Arc::new(changed)
                    },
                    40_000_000,
                    500
                )
                .is_err()
        );
        assert_eq!(owner.admitted_chunks, 1);
    }

    #[test]
    fn a_paused_simulation_cannot_make_an_expired_image_fresh() {
        let mut owner = executor();
        let command = command();
        assert!(owner.next(&command, 0, 1001).is_err());
        assert_eq!(owner.admitted_chunks, 0);
        let (_, frame) = owner.next(&command, 0, 500).unwrap();
        assert_eq!(frame.frame_index, 0);
        assert_eq!(frame.observation_wall_age_ms, 500);
        let mut future = (*command.chunk).clone();
        future.sequence_id = 2;
        future.observation.frame_id = 2;
        future.observation.captured_at_unix_ms = 501;
        assert!(
            owner
                .next(
                    &ArenaTaskCommand {
                        chunk: Arc::new(future)
                    },
                    20_000_000,
                    500
                )
                .is_err()
        );
        assert_eq!(owner.admitted_chunks, 1);
    }
}
