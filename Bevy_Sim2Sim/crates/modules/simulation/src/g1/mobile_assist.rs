//! Typed handoff from original mobile VLA chunks to disclosed traditional grip
//! and self-state navigation on the same sole native physics owner.

use std::time::{SystemTime, UNIX_EPOCH};

use robot_minigame::{
    RobotError,
    g1::{contract::G1Command, definition::G1Definition},
};
use serde::Serialize;
use task_minigame::{
    policy::{ActionLimits, profile_contract},
    types::TaskProfile,
};

use super::{
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    mobile_lowering::{MobileGripLowering, MobileLowerGoal, MobileLowerStep},
    mobile_navigation::{
        MobileCarryGoal, MobileCarryNavigator, MobileNavigationStep, MobileScanGoal,
        MobileScanNavigator,
    },
    mobile_raise::{MobileGripRaising, MobileRaiseGoal, MobileRaiseStep},
    mobile_release::{MobileGripRelease, MobileReleaseGoal, MobileReleaseStep},
    mobile_restore::{MobileGripRestoring, MobileRestoreGoal, MobileRestoreStep},
    runner::{G1Measurement, G1ProgressCounts},
    task_objects::TaskObjectFrame,
    task_policy::{ArenaControllerCommand, controller_command},
    task_runner::{
        ArenaBodyStep, ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskExecution, ArenaTaskRunner,
        ArenaTaskRunnerConfig,
    },
};

#[derive(Clone, Debug)]
pub enum MobileAssistCommand {
    OriginalVla(ArenaTaskCommand),
    ClassicalCarry(MobileCarryGoal),
    ClassicalScan(MobileScanGoal),
    ClassicalReobserve(MobileScanGoal),
    ClassicalLower(MobileLowerGoal),
    ClassicalRaise(MobileRaiseGoal),
    ClassicalRelease(MobileReleaseGoal),
    ClassicalRestore(MobileRestoreGoal),
}

impl MobileAssistCommand {
    pub(crate) fn validate(&self) -> Result<(), RobotError> {
        match self {
            Self::OriginalVla(command) => {
                command.validate()?;
                if command.chunk.profile != TaskProfile::MobileBox {
                    return Err(invalid("assisted mobile owner rejects static VLA"));
                }
                Ok(())
            }
            Self::ClassicalCarry(goal) => goal.validate(),
            Self::ClassicalScan(goal) => goal.validate(),
            Self::ClassicalReobserve(goal) => goal.validate(),
            Self::ClassicalLower(goal) => goal.validate(),
            Self::ClassicalRaise(goal) => goal.validate(),
            Self::ClassicalRelease(goal) => goal.validate(),
            Self::ClassicalRestore(goal) => goal.validate(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MobileAssistExecution {
    OriginalVla(ArenaTaskExecution),
    ClassicalCarry {
        goal: MobileCarryGoal,
        grip: MobileGripReceipt,
        navigation: MobileNavigationStep,
        command: G1Command,
    },
    ClassicalScan {
        goal: MobileScanGoal,
        grip: MobileGripReceipt,
        navigation: MobileNavigationStep,
        command: G1Command,
    },
    ClassicalReobserve {
        goal: MobileScanGoal,
        grip: MobileGripReceipt,
        navigation: MobileNavigationStep,
        command: G1Command,
    },
    ClassicalLower {
        goal: MobileLowerGoal,
        lowering: MobileLowerStep,
    },
    ClassicalRaise {
        goal: MobileRaiseGoal,
        raising: MobileRaiseStep,
    },
    ClassicalRelease {
        goal: MobileReleaseGoal,
        opening: MobileReleaseStep,
    },
    ClassicalRestore {
        goal: MobileRestoreGoal,
        restoring: MobileRestoreStep,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileAssistStep {
    pub execution: MobileAssistExecution,
    pub body: ArenaBodyStep,
}

/// No second world, physical pose setter, task-object sensor or model client.
pub struct MobileAssistRunner {
    owner: ArenaTaskRunner,
    calibration: MobileGripCalibration,
    limits: ActionLimits,
    maximum_observation_wall_age_ms: u64,
    last_vla_execution: Option<ArenaTaskExecution>,
    last_vla_command: Option<G1Command>,
    carry: Option<CarryState>,
    scan: Option<ScanState>,
    reobserve: Option<ScanState>,
    lower: Option<MobileGripLowering>,
    raise: Option<MobileGripRaising>,
    release: Option<MobileGripRelease>,
    restore: Option<MobileGripRestoring>,
    original_transport_command: Option<G1Command>,
    halted: bool,
}

struct CarryState {
    navigator: MobileCarryNavigator,
    command: G1Command,
    grip: MobileGripReceipt,
}

struct ScanState {
    navigator: MobileScanNavigator,
    command: G1Command,
    grip: MobileGripReceipt,
}

impl MobileAssistRunner {
    /// Opt-in factory only: original complete T2 scene/50Hz/native motors with
    /// the already measured4PGS candidate. Normal task loading is unchanged.
    pub fn load(config: &ArenaTaskRunnerConfig) -> Result<Self, RobotError> {
        let ArenaTaskBodyConfig::MobileHomieV2(body) = &config.body else {
            return Err(invalid("mobile assist requires Homie_v2/N1.6"));
        };
        let definition = G1Definition::load(&body.definition, &body.definition_sha256)?;
        Ok(Self {
            owner: ArenaTaskRunner::load_mobile_constraint_diagnostic(config)?,
            calibration: MobileGripCalibration::new(&definition)?,
            limits: config.limits.clone(),
            maximum_observation_wall_age_ms: config.max_observation_wall_age_ms,
            last_vla_execution: None,
            last_vla_command: None,
            carry: None,
            scan: None,
            reobserve: None,
            lower: None,
            raise: None,
            release: None,
            restore: None,
            original_transport_command: None,
            halted: false,
        })
    }
    pub fn measurement(&self) -> Result<G1Measurement, RobotError> {
        self.owner.measurement()
    }
    pub fn initial_frame(&self) -> Result<robot_minigame::g1::definition::G1BodyFrame, RobotError> {
        self.owner.initial_frame()
    }
    pub fn task_object_frame(&self) -> Result<Option<TaskObjectFrame>, RobotError> {
        self.owner.task_object_frame()
    }
    pub fn progress_counts(&self) -> G1ProgressCounts {
        let mut counts = self.owner.progress_counts();
        counts.halted |= self.halted;
        counts
    }
    pub fn completed_carry(&self) -> bool {
        self.carry
            .as_ref()
            .is_some_and(|carry| carry.navigator.completed())
    }

    pub fn completed_skill(&self) -> bool {
        if let Some(release) = &self.release {
            release.completed()
        } else if let Some(reobserve) = &self.reobserve {
            reobserve.navigator.completed()
        } else if let Some(restore) = &self.restore {
            restore.completed()
        } else if let Some(raise) = &self.raise {
            raise.completed()
        } else if let Some(lower) = &self.lower {
            lower.completed()
        } else {
            self.completed_carry() || self.scan.as_ref().is_some_and(|s| s.navigator.completed())
        }
    }

    fn prepare_classical(
        &self,
        observation: &task_minigame::types::ObservationStamp,
        state: &G1Measurement,
    ) -> Result<(G1Command, MobileGripReceipt), RobotError> {
        if self.restore.as_ref().is_some_and(|r| !r.completed()) {
            return Err(invalid(
                "cannot replace active transport-posture restoration",
            ));
        }
        if self
            .reobserve
            .as_ref()
            .is_some_and(|r| !r.navigator.completed())
        {
            return Err(invalid("cannot replace active visual reobservation turn"));
        }
        let previous = self
            .last_vla_execution
            .as_ref()
            .ok_or_else(|| invalid("classical skill has no original VLA predecessor"))?;
        if previous.frame_index + 1 != profile_contract(TaskProfile::MobileBox).action_horizon {
            return Err(invalid(
                "classical handoff requires an original complete VLA chunk",
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| invalid(e.to_string()))?
            .as_millis() as u64;
        let age = now
            .checked_sub(observation.captured_at_unix_ms)
            .ok_or_else(|| invalid("classical observation is from a future wall clock"))?;
        if age > self.maximum_observation_wall_age_ms {
            return Err(invalid(
                "classical observation expired before owner handoff",
            ));
        }
        if let Some(carry) = &self.carry {
            if !carry.navigator.completed() {
                return Err(invalid("cannot replace an active carry"));
            }
            if let Some(restore) = &self.restore {
                return Ok((restore.command().clone(), carry.grip.clone()));
            }
            if let Some(raise) = &self.raise {
                if !raise.completed() {
                    return Err(invalid("cannot replace active raising"));
                }
                return Ok((raise.command().clone(), carry.grip.clone()));
            }
            return Ok((carry.command.clone(), carry.grip.clone()));
        }
        if let Some(scan) = &self.scan {
            if !scan.navigator.completed() {
                return Err(invalid("cannot replace an active scan"));
            }
            if let Some(lower) = &self.lower {
                if !lower.completed() {
                    return Err(invalid("cannot replace active lowering"));
                }
                return Ok((lower.command().clone(), scan.grip.clone()));
            }
            return Ok((scan.command.clone(), scan.grip.clone()));
        }
        let original = self
            .last_vla_command
            .as_ref()
            .ok_or_else(|| invalid("classical handoff lost original upper command"))?;
        let correction = self.calibration.correct(state, original)?;
        Ok((correction.command, correction.receipt))
    }

    pub fn step_with_guard(
        &mut self,
        command: &MobileAssistCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<MobileAssistStep, RobotError> {
        if self.halted {
            return Err(invalid("mobile assist halted; reset required"));
        }
        let result = self.step_inner(command, guard);
        if result.is_err() {
            self.halted = true;
        }
        result
    }
    fn step_inner(
        &mut self,
        command: &MobileAssistCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<MobileAssistStep, RobotError> {
        guard()?;
        command.validate()?;
        match command {
            MobileAssistCommand::OriginalVla(command) => {
                if self
                    .carry
                    .as_ref()
                    .is_some_and(|carry| !carry.navigator.completed())
                    || self
                        .scan
                        .as_ref()
                        .is_some_and(|scan| !scan.navigator.completed())
                    || self.lower.as_ref().is_some_and(|lower| !lower.completed())
                    || self.raise.as_ref().is_some_and(|raise| !raise.completed())
                    || self
                        .release
                        .as_ref()
                        .is_some_and(|release| !release.completed())
                    || self
                        .reobserve
                        .as_ref()
                        .is_some_and(|r| !r.navigator.completed())
                    || self.restore.as_ref().is_some_and(|r| !r.completed())
                {
                    return Err(invalid(
                        "cannot replace active classical carry with a VLA chunk",
                    ));
                }
                let step = self.owner.step_with_guard(command, guard)?;
                let admitted = controller_command(
                    TaskProfile::MobileBox,
                    &command.chunk.frames[step.execution.frame_index],
                    &self.limits,
                )
                .map_err(|e| invalid(format!("{e:?}")))?;
                let ArenaControllerCommand::MobileHomieV2(original) = admitted.controller else {
                    return Err(invalid("foreign controller in mobile handoff"));
                };
                self.last_vla_command = Some(original);
                self.last_vla_execution = Some(step.execution.clone());
                self.carry = None;
                self.scan = None;
                self.reobserve = None;
                self.lower = None;
                self.raise = None;
                self.release = None;
                self.restore = None;
                self.original_transport_command = None;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::OriginalVla(step.execution),
                    body: step.body,
                })
            }
            MobileAssistCommand::ClassicalCarry(goal) => {
                if self.release.is_some() {
                    return Err(invalid("carry after opening requires a new grasp/reset"));
                }
                let state = self.owner.measurement()?;
                if self.carry.as_ref().is_none_or(|carry| {
                    carry.navigator.completed() && carry.navigator.goal() != goal
                }) {
                    if self.carry.as_ref().is_some_and(|carry| {
                        goal.observation.frame_id <= carry.navigator.goal().observation.frame_id
                    }) {
                        return Err(invalid("next carry requires a newer actual camera frame"));
                    }
                    let navigator = MobileCarryNavigator::new(goal.clone(), &state)?;
                    let (command, grip) = self.prepare_classical(&goal.observation, &state)?;
                    if self.original_transport_command.is_none() {
                        let mut posture = command.clone();
                        // Cache the arm posture, not the predecessor VLA's
                        // navigation. This reference is never executed.
                        posture.navigation = [0.; 3];
                        self.original_transport_command = Some(posture);
                    }
                    self.carry = Some(CarryState {
                        navigator,
                        command,
                        grip,
                    });
                    self.scan = None;
                    self.lower = None;
                    self.raise = None;
                    self.reobserve = None;
                    self.restore = None;
                }
                let carry = self.carry.as_mut().unwrap();
                if carry.navigator.goal() != goal {
                    return Err(invalid(
                        "carry goal changed while executing; stop/reset required",
                    ));
                }
                let navigation = carry.navigator.update(&state)?;
                carry.command.navigation = navigation.navigation;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&carry.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalCarry {
                        goal: goal.clone(),
                        grip: carry.grip.clone(),
                        navigation,
                        command: carry.command.clone(),
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalReobserve(goal) => {
                if self.release.is_some() {
                    return Err(invalid("reobserve after opening requires new grasp/reset"));
                }
                let state = self.owner.measurement()?;
                if self.reobserve.is_none() {
                    let Some(raise) = &self.raise else {
                        return Err(invalid("reobserve requires matched completed raising"));
                    };
                    if !raise.completed()
                        || self.carry.as_ref().is_none_or(|c| !c.navigator.completed())
                        || goal.observation.frame_id <= raise.goal().observation.frame_id
                    {
                        return Err(invalid(
                            "reobserve requires a newer current raised RGB boundary",
                        ));
                    }
                    if self.restore.as_ref().is_some_and(|r| {
                        !r.completed() || goal.observation.frame_id <= r.goal().observation.frame_id
                    }) {
                        return Err(invalid(
                            "restored reobserve requires a newer completed-posture image",
                        ));
                    }
                    let [w, x, y, z] = state.root_rotation_wxyz;
                    let yaw = (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z));
                    let offset = goal.heading_yaw_source_rad - yaw;
                    if offset.sin().atan2(offset.cos()).abs() > 0.6 {
                        return Err(invalid("raised reobservation turn exceeds0.6rad"));
                    }
                    let navigator = MobileScanNavigator::new(goal.clone(), &state)?;
                    let (command, grip) = self.prepare_classical(&goal.observation, &state)?;
                    self.reobserve = Some(ScanState {
                        navigator,
                        command,
                        grip,
                    });
                }
                let reobserve = self.reobserve.as_mut().unwrap();
                if reobserve.navigator.goal() != goal {
                    return Err(invalid("reobserve goal changed while executing"));
                }
                let navigation = reobserve.navigator.update(&state)?;
                reobserve.command.navigation = navigation.navigation;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&reobserve.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalReobserve {
                        goal: goal.clone(),
                        grip: reobserve.grip.clone(),
                        navigation,
                        command: reobserve.command.clone(),
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalScan(goal) => {
                let state = self.owner.measurement()?;
                if self.carry.is_some() || self.lower.is_some() {
                    return Err(invalid(
                        "scan after carry requires a separately admitted next stage",
                    ));
                }
                if self.scan.is_none() {
                    let navigator = MobileScanNavigator::new(goal.clone(), &state)?;
                    let (command, grip) = self.prepare_classical(&goal.observation, &state)?;
                    if self.original_transport_command.is_none() {
                        let mut posture = command.clone();
                        posture.navigation = [0.; 3];
                        self.original_transport_command = Some(posture);
                    }
                    self.scan = Some(ScanState {
                        navigator,
                        command,
                        grip,
                    });
                }
                let scan = self.scan.as_mut().unwrap();
                if scan.navigator.goal() != goal {
                    return Err(invalid("scan goal changed while executing; reset required"));
                }
                let navigation = scan.navigator.update(&state)?;
                scan.command.navigation = navigation.navigation;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&scan.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalScan {
                        goal: goal.clone(),
                        grip: scan.grip.clone(),
                        navigation,
                        command: scan.command.clone(),
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalRestore(goal) => {
                if self.release.is_some() || self.reobserve.is_some() {
                    return Err(invalid(
                        "restore requires the stationary raised stage before observation motion",
                    ));
                }
                let state = self.owner.measurement()?;
                if self.restore.is_none() {
                    if self.raise.as_ref().is_none_or(|r| !r.completed())
                        || self.carry.as_ref().is_none_or(|c| !c.navigator.completed())
                    {
                        return Err(invalid(
                            "restore requires a completed raised carry boundary",
                        ));
                    }
                    let original = self
                        .original_transport_command
                        .as_ref()
                        .ok_or_else(|| invalid("episode transport posture absent"))?;
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    self.restore = Some(MobileGripRestoring::new(
                        goal.clone(),
                        &state,
                        command,
                        original,
                        &self.calibration,
                    )?);
                }
                let restore = self.restore.as_mut().unwrap();
                if restore.goal() != goal {
                    return Err(invalid("restore goal changed while executing"));
                }
                let restoring = restore.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&restoring.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalRestore {
                        goal: goal.clone(),
                        restoring,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalRelease(goal) => {
                let state = self.owner.measurement()?;
                if self.release.is_none() {
                    if self.carry.as_ref().is_none_or(|c| !c.navigator.completed())
                        || self.raise.as_ref().is_some_and(|r| !r.completed())
                    {
                        return Err(invalid("release requires completed stationary carry"));
                    }
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    self.release = Some(MobileGripRelease::new(
                        goal.clone(),
                        &state,
                        command,
                        &self.calibration,
                    )?);
                }
                let release = self.release.as_mut().unwrap();
                if release.goal() != goal {
                    return Err(invalid("release goal changed while executing"));
                }
                let opening = release.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&opening.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalRelease {
                        goal: goal.clone(),
                        opening,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalRaise(goal) => {
                if self.release.is_some() {
                    return Err(invalid("raising after opening requires new grasp/reset"));
                }
                let state = self.owner.measurement()?;
                if self.raise.is_none() {
                    if self.carry.as_ref().is_none_or(|c| !c.navigator.completed()) {
                        return Err(invalid("raising requires completed stationary carry"));
                    }
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    self.raise = Some(MobileGripRaising::new(goal.clone(), &state, command)?);
                }
                let raise = self.raise.as_mut().unwrap();
                if raise.goal() != goal {
                    return Err(invalid("raise goal changed while executing"));
                }
                let raising = raise.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&raising.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalRaise {
                        goal: goal.clone(),
                        raising,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalLower(goal) => {
                let state = self.owner.measurement()?;
                if self.lower.is_none() {
                    if self.carry.is_some()
                        || self.scan.as_ref().is_none_or(|s| !s.navigator.completed())
                    {
                        return Err(invalid(
                            "lowering requires the admitted completed stationary scan",
                        ));
                    }
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    self.lower = Some(MobileGripLowering::new(goal.clone(), &state, command)?);
                }
                let lower = self.lower.as_mut().unwrap();
                if lower.goal() != goal {
                    return Err(invalid("lower goal changed while executing"));
                }
                let lowering = lower.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&lowering.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalLower {
                        goal: goal.clone(),
                        lowering,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
        }
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use serde::Deserialize;
    use std::{fs, io::Write, path::Path, sync::Arc};
    use task_minigame::{policy::PolicyActionChunk, types::ObservationStamp};
    #[derive(Deserialize)]
    struct Sequence {
        schema: String,
        chunks: Vec<PolicyActionChunk>,
    }
    fn read(name: &str) -> Result<Vec<u8>, RobotError> {
        let path = std::env::var(name).map_err(|e| invalid(e.to_string()))?;
        let hash = std::env::var(format!("{name}_SHA256")).map_err(|e| invalid(e.to_string()))?;
        bound_bytes(Path::new(&path), &hash)
    }
    #[derive(Deserialize)]
    struct ScanLowerFixture {
        scan: MobileScanGoal,
        lower: MobileLowerGoal,
        carry: Option<MobileCarryGoal>,
        fine_carry: Option<MobileCarryGoal>,
        raise: Option<MobileRaiseGoal>,
        reobserve: Option<MobileScanGoal>,
        restore: Option<MobileRestoreGoal>,
        release: Option<MobileReleaseGoal>,
    }

    /// Frozen real-RGB grasp/scan prefix, then lowering and optional visual carry.
    /// This has zero fresh VLA calls and grants no autonomous/task qualification.
    #[test]
    #[ignore = "requires frozen four-grasp/scan goals/new output;<=3150native50HzTicks,0freshVLA"]
    fn real_mobile_scan_lowering_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        let mut goals: ScanLowerFixture =
            serde_json::from_slice(&read("G1_MOBILE_SCAN_LOWER_GOALS")?)
                .map_err(|e| invalid(e.to_string()))?;
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
        {
            return Err(invalid(
                "scanlower requires four original saved native replies",
            ));
        }
        let output =
            std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&output).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let mut owner = MobileAssistRunner::load(&config)?;
        let mut current = None;
        let mut lower_started = false;
        let mut lower_completed = false;
        let mut carry_started = false;
        let mut coarse_completed = false;
        let mut fine_started = false;
        let mut fine_completed = false;
        let mut raise_started = false;
        let mut reobserve_started = false;
        let mut restore_started = false;
        let mut release_started = false;
        let maximum_ticks = if goals.carry.is_some() { 3150 } else { 1300 };
        let mut actual_ticks = 0;
        let now = || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        };
        let result = (|| -> Result<(), RobotError> {
            for tick in 1..=maximum_ticks {
                if tick <= 200 && (tick - 1) % 50 == 0 {
                    let mut chunk = sequence.chunks[((tick - 1) / 50) as usize].clone();
                    chunk.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::OriginalVla(ArenaTaskCommand {
                        chunk: Arc::new(chunk),
                        scheduled_start_sim_ns: None,
                    }));
                } else if tick == 201 {
                    goals.scan.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalScan(goals.scan.clone()));
                } else if !lower_started && owner.completed_skill() {
                    goals.lower.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalLower(goals.lower.clone()));
                    lower_started = true;
                } else if lower_started
                    && !carry_started
                    && owner.completed_skill()
                    && let Some(goal) = &mut goals.carry
                {
                    goal.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalCarry(goal.clone()));
                    carry_started = true;
                } else if carry_started
                    && !raise_started
                    && owner.completed_skill()
                    && let Some(goal) = &mut goals.raise
                {
                    goal.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalRaise(goal.clone()));
                    raise_started = true;
                } else if raise_started
                    && !restore_started
                    && owner.completed_skill()
                    && let Some(goal) = &mut goals.restore
                {
                    goal.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalRestore(goal.clone()));
                    restore_started = true;
                } else if raise_started
                    && !reobserve_started
                    && owner.completed_skill()
                    && let Some(goal) = &mut goals.reobserve
                {
                    goal.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalReobserve(goal.clone()));
                    reobserve_started = true;
                } else if carry_started
                    && !fine_started
                    && owner.completed_skill()
                    && let Some(goal) = &mut goals.fine_carry
                {
                    goal.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalCarry(goal.clone()));
                    fine_started = true;
                } else if fine_started
                    && !release_started
                    && owner.completed_skill()
                    && let Some(goal) = &mut goals.release
                {
                    goal.observation.captured_at_unix_ms = now();
                    current = Some(MobileAssistCommand::ClassicalRelease(goal.clone()));
                    release_started = true;
                }
                let step = owner.step_with_guard(current.as_ref().unwrap(), &mut || Ok(()))?;
                if matches!(&step.execution, MobileAssistExecution::ClassicalLower { lowering, .. } if lowering.completed)
                {
                    lower_completed = true;
                }
                actual_ticks = tick;
                if carry_started && !fine_started && owner.completed_skill() {
                    coarse_completed = true;
                }
                if fine_started && !release_started && owner.completed_skill() {
                    fine_completed = true;
                }
                serde_json::to_writer(
                    &mut trace,
                    &serde_json::json!({
                        "saved_grasp_scan_fixture":true,"fresh_vla_calls":0,"qualified":false,
                        "robot_background_contacts_auditor_only":owner.owner.diagnostic_robot_background_contacts(),
                        "execution":step.execution,"body":step.body,
                    }),
                )
                .map_err(|e| invalid(e.to_string()))?;
                writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                if lower_started
                    && owner.completed_skill()
                    && (goals.carry.is_none() || carry_started)
                    && (goals.fine_carry.is_none() || fine_started)
                    && (goals.reobserve.is_none() || reobserve_started)
                    && (goals.restore.is_none() || restore_started)
                    && (goals.release.is_none() || release_started)
                {
                    return Ok(());
                }
            }
            Err(invalid("finite scanlower diagnostic deadline missed"))
        })();
        trace.flush().map_err(|e| invalid(e.to_string()))?;
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(file,&serde_json::json!({
            "actual_integrations":actual_ticks,"lower_started":lower_started,"carry_started":carry_started,
            "fine_carry_started":fine_started,"completed_fine_carry":fine_completed,
            "raise_started":raise_started,
            "reobserve_started":reobserve_started,
            "restore_started":restore_started,
            "release_started":release_started,"completed_release":release_started && result.is_ok(),
            "completed_lowering":lower_completed,"completed_carry":coarse_completed,
            "completed_selected_sequence":result.is_ok(),"failure":result.as_ref().err().map(ToString::to_string),
            "fresh_vla_calls":0,"saved_native_fixture_used":true,"qualified":false,
            "physics_hz":50,"integrations_per_tick":1,
        })).map_err(|e| invalid(e.to_string()))?;
        result
    }
    #[test]
    #[ignore = "requires frozen mobileconfig/four saved native replies/new output; <=2050real50Hzsteps,0VLA"]
    fn real_mobile_typed_assist_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
        {
            return Err(invalid(
                "typedassist diagnostic requires exactlyfour native reply fixtures",
            ));
        }
        let path = std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut receipt = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| invalid(e.to_string()))?;
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&path).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let mut owner = MobileAssistRunner::load(&config)?;
        let mut current = None;
        let mut actual_ticks = 0;
        let mut minimum_upright = 1_f32;
        let mut last_root = None;
        let mut walk_origin = None;
        let mut final_displacement = 0_f32;
        let mut maximum_walk_displacement = 0_f32;
        let result = (|| -> Result<(), RobotError> {
            for tick in 1..=2050 {
                if tick <= 200 && (tick - 1) % 50 == 0 {
                    let mut chunk = sequence.chunks[((tick - 1) / 50) as usize].clone();
                    // Offline wall-admission rebinding only. Original raw
                    // fixture stamps/action bytes remain in the bound input.
                    chunk.observation.captured_at_unix_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_err(|e| invalid(e.to_string()))?
                        .as_millis()
                        as u64;
                    current = Some(MobileAssistCommand::OriginalVla(ArenaTaskCommand {
                        chunk: Arc::new(chunk),
                        scheduled_start_sim_ns: None,
                    }));
                } else if tick == 201 {
                    let state = owner.measurement()?;
                    current = Some(MobileAssistCommand::ClassicalCarry(MobileCarryGoal {
                        observation: ObservationStamp {
                            episode_id: state.episode_id,
                            frame_id: 5,
                            sim_time_ns: state.sim_time_ns,
                            captured_at_unix_ms: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map_err(|e| invalid(e.to_string()))?
                                .as_millis()
                                as u64,
                        },
                        heading_yaw_source_rad: std::f32::consts::FRAC_PI_2,
                        relative_distance_m: 2.,
                    }));
                }
                let step = owner.step_with_guard(current.as_ref().unwrap(), &mut || Ok(()))?;
                actual_ticks = tick;
                let ArenaBodyStep::MobileHomieV2(body) = &step.body else {
                    return Err(invalid("typed mobile diagnostic changed body"));
                };
                if body.integration_count != tick
                    || body.inference.inference_count != tick
                    || body.torque_update_count != tick
                    || body.step_configuration.dt != 0.02
                    || body.step_configuration.physics_hz != 50
                    || body.step_configuration.num_solver_iterations != 1
                    || body.step_configuration.num_internal_pgs_iterations != 4
                    || body.step_configuration.max_ccd_substeps != 1
                {
                    return Err(invalid("typed mobile owner changed counts/timing"));
                }
                minimum_upright = minimum_upright.min(body.root_upright_cosine);
                let (phase, command) = match &step.execution {
                    MobileAssistExecution::OriginalVla(_) => ("saved_grasp_fixture", None),
                    MobileAssistExecution::ClassicalScan { .. }
                    | MobileAssistExecution::ClassicalReobserve { .. }
                    | MobileAssistExecution::ClassicalRestore { .. }
                    | MobileAssistExecution::ClassicalLower { .. }
                    | MobileAssistExecution::ClassicalRaise { .. }
                    | MobileAssistExecution::ClassicalRelease { .. } => {
                        return Err(invalid("unexpected scan in saved carry fixture"));
                    }
                    MobileAssistExecution::ClassicalCarry {
                        navigation,
                        command,
                        ..
                    } => {
                        let phase = match navigation.phase {
                            super::super::mobile_navigation::MobileNavigationPhase::Turn => {
                                "feedback_fixed_grip_turn"
                            }
                            super::super::mobile_navigation::MobileNavigationPhase::Walk => {
                                "feedback_fixed_grip_walk"
                            }
                            super::super::mobile_navigation::MobileNavigationPhase::Stop => {
                                "feedback_fixed_grip_stop"
                            }
                        };
                        (phase, Some(command))
                    }
                };
                if phase == "feedback_fixed_grip_walk" && walk_origin.is_none() {
                    walk_origin = last_root;
                }
                if let Some(origin) = walk_origin {
                    let origin: [f32; 3] = origin;
                    final_displacement = (body.root_position_source[0] - origin[0])
                        .hypot(body.root_position_source[1] - origin[1]);
                    if phase == "feedback_fixed_grip_walk" {
                        maximum_walk_displacement =
                            maximum_walk_displacement.max(final_displacement);
                    }
                }
                last_root = Some(body.root_position_source);
                serde_json::to_writer(&mut trace,&serde_json::json!({"phase":phase,"manual_body_command":command,"body":step.body,"execution":step.execution,"diagnostic_saved_grasp_fixture":true,"autonomous_execution":false})).map_err(|e|invalid(e.to_string()))?;
                writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                if minimum_upright < 0.95 {
                    return Err(invalid("independent diagnostic fall guard"));
                }
                if owner.completed_carry() {
                    break;
                }
            }
            Ok(())
        })();
        trace.flush().map_err(|e| invalid(e.to_string()))?;
        let passed = result.is_ok()
            && owner.completed_carry()
            && maximum_walk_displacement >= 2.
            && final_displacement >= 2.;
        serde_json::to_writer_pretty(&mut receipt,&serde_json::json!({"schema":"g1_mobile_typed_assist_diagnostic_v1","qualified":false,"autonomous_execution":false,"task_success_verified":false,"actual_vla_inferences":0,"original_saved_reply_count":4,"actual_ticks":actual_ticks,"counts":owner.progress_counts(),"physics_hz":50,"integrations_per_tick":1,"completed_carry":owner.completed_carry(),"minimum_upright":minimum_upright,"maximum_walk_displacement_m":maximum_walk_displacement,"final_displacement_m":final_displacement,"mechanical_only_passed":passed,"error":result.as_ref().err().map(ToString::to_string),"scope":"savednativegrasp/runtimeclassicalgrip/typedselfstatenav;notliveRGB,Qwen,release,or1x"})).map_err(|e|invalid(e.to_string()))?;
        receipt.flush().map_err(|e| invalid(e.to_string()))?;
        result?;
        if !passed {
            return Err(invalid("typed mechanical carry didnotmeet2mwalkandstop"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod worker_diagnostic {
    use super::super::worker::{G1WorkerPhase, MobileAssistWorker, TimedCommand};
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use serde::Deserialize;
    use std::{
        fs,
        io::Write,
        path::Path,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };
    use task_minigame::{policy::PolicyActionChunk, types::ObservationStamp};
    #[derive(Deserialize)]
    struct Sequence {
        schema: String,
        chunks: Vec<PolicyActionChunk>,
    }
    fn read(name: &str) -> Result<Vec<u8>, RobotError> {
        bound_bytes(
            Path::new(&std::env::var(name).map_err(|e| invalid(e.to_string()))?),
            &std::env::var(format!("{name}_SHA256")).map_err(|e| invalid(e.to_string()))?,
        )
    }
    #[test]
    #[ignore = "requires frozenmobileconfig/fourreplies/newoutput;one real50Hz backgroundhandoff,<=2050Ticks/50s/0VLA"]
    fn real_mobile_typed_worker_handoff() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
        {
            return Err(invalid("typedworkerrequiresfourboundednativefixtures"));
        }
        let path = std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|e| invalid(e.to_string()))?;
        let mut trace_file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(Path::new(&path).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let worker = MobileAssistWorker::spawn_mobile_assist(config.clone())?;
        let trace = worker.subscribe_steps(2048)?;
        let started = Instant::now();
        let mut latest = None;
        let mut next_chunk = 0;
        let mut goal_submitted = false;
        let mut completed_at = None;
        let mut observed_steps = 0_u64;
        let result = (|| -> Result<(), RobotError> {
            loop {
                for record in trace.drain() {
                    observed_steps += 1;
                    let (phase, command) = match &record.step.execution {
                        MobileAssistExecution::OriginalVla(_) => ("saved_grasp_fixture", None),
                        MobileAssistExecution::ClassicalScan { .. }
                        | MobileAssistExecution::ClassicalReobserve { .. }
                        | MobileAssistExecution::ClassicalRestore { .. }
                        | MobileAssistExecution::ClassicalLower { .. }
                        | MobileAssistExecution::ClassicalRaise { .. }
                        | MobileAssistExecution::ClassicalRelease { .. } => {
                            return Err(invalid("unexpected scan in saved carry fixture"));
                        }
                        MobileAssistExecution::ClassicalCarry {
                            navigation,
                            command,
                            ..
                        } => {
                            let phase = match navigation.phase {
                                super::super::mobile_navigation::MobileNavigationPhase::Turn => {
                                    "feedback_fixed_grip_turn"
                                }
                                super::super::mobile_navigation::MobileNavigationPhase::Walk => {
                                    "feedback_fixed_grip_walk"
                                }
                                super::super::mobile_navigation::MobileNavigationPhase::Stop => {
                                    "feedback_fixed_grip_stop"
                                }
                            };
                            (phase, Some(command))
                        }
                    };
                    serde_json::to_writer(&mut trace_file,&serde_json::json!({"phase":phase,"manual_body_command":command,"body":record.step.body,"execution":record.step.execution,"episode_id":record.episode_id,"generation":record.generation,"timing":record.timing,"diagnostic_saved_grasp_fixture":true,"autonomous_execution":false})).map_err(|e|invalid(e.to_string()))?;
                    writeln!(trace_file).map_err(|e| invalid(e.to_string()))?;
                }
                if let Some(snapshot) = worker.take_latest() {
                    latest = Some(snapshot);
                }
                if let Some(snapshot) = &latest {
                    if snapshot.phase == G1WorkerPhase::Failed {
                        return Err(invalid(format!(
                            "backgroundownerfailed: {:?}",
                            snapshot.reason
                        )));
                    }
                    if snapshot.phase == G1WorkerPhase::Paused && snapshot.frame.is_some() {
                        let ticks = snapshot.timing.episode_integrations;
                        if next_chunk<4 && ticks==next_chunk as u64*50 {
                            let mut chunk=sequence.chunks[next_chunk].clone();
                            chunk.observation.captured_at_unix_ms=SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|invalid(e.to_string()))?.as_millis() as u64;
                            worker.submit(TimedCommand {episode_id:config.body.episode_id(),valid_until_sim_ns:(next_chunk as u64+1)*1_000_000_000,valid_until_wall:Instant::now()+Duration::from_secs(2),command:MobileAssistCommand::OriginalVla(ArenaTaskCommand {chunk:Arc::new(chunk),scheduled_start_sim_ns:None})})?;
                            next_chunk+=1;
                        } else if next_chunk==4 && ticks==200 && !goal_submitted {
                            let goal=MobileCarryGoal {observation:ObservationStamp {episode_id:snapshot.episode_id,frame_id:5,sim_time_ns:4_000_000_000,captured_at_unix_ms:SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|invalid(e.to_string()))?.as_millis() as u64},heading_yaw_source_rad:std::f32::consts::FRAC_PI_2,relative_distance_m:2.};
                            worker.submit(TimedCommand {episode_id:snapshot.episode_id,valid_until_sim_ns:41_000_000_000,valid_until_wall:Instant::now()+Duration::from_secs(40),command:MobileAssistCommand::ClassicalCarry(goal)})?;goal_submitted=true;
                        } else if goal_submitted && snapshot.step.as_ref().is_some_and(|step|matches!(&step.execution,MobileAssistExecution::ClassicalCarry {navigation,..} if navigation.completed)) {
                            let entry=completed_at.get_or_insert((Instant::now(),ticks));
                            if ticks!=entry.1 {return Err(invalid("completed background skill performed an extra integration"));}
                            if entry.0.elapsed()>=Duration::from_secs(1){break;}
                        }
                    }
                }
                if started.elapsed() > Duration::from_secs(50) {
                    return Err(invalid("finite background handoff deadline missed"));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Ok(())
        })();
        trace_file.flush().map_err(|e| invalid(e.to_string()))?;
        let timing = latest.as_ref().map(|s| s.timing.clone());
        let dropped = trace.dropped_records();
        let counts = timing.as_ref().map(|t| t.episode_integrations).unwrap_or(0);
        let passed =
            result.is_ok() && completed_at.is_some() && dropped == 0 && observed_steps == counts;
        serde_json::to_writer_pretty(&mut file,&serde_json::json!({"schema":"g1_mobile_typed_background_handoff_v1","qualified":false,"autonomous_execution":false,"task_success_verified":false,"formal_one_x_qualified":false,"actual_vla_inferences":0,"original_saved_reply_count":4,"actual_ticks":counts,"observed_completed_steps":observed_steps,"trace_dropped":dropped,"timing":timing,"paused_after_completed_skill":completed_at.is_some(),"one_second_after_completion_adds_no_ticks":completed_at.is_some(),"background_handoff_only_passed":passed,"error":result.as_ref().err().map(ToString::to_string),"wall_seconds_including_load_pause_evidence":started.elapsed().as_secs_f64(),"scope":"savednativegrasp/actualbackgroundclock/typedclassicalcarry;notliveRGB,render/Qwen/release/formal1x"})).map_err(|e|invalid(e.to_string()))?;
        file.flush().map_err(|e| invalid(e.to_string()))?;
        worker.shutdown()?;
        result?;
        if !passed {
            return Err(invalid("backgroundcarrydidnotcompletewithfulltrace"));
        }
        Ok(())
    }
}
