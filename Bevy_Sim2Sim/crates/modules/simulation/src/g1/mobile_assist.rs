//! Typed handoff from original mobile VLA chunks to disclosed traditional grip
//! and self-state navigation on the same sole native physics owner.

use std::time::{SystemTime, UNIX_EPOCH};

use robot_minigame::{
    RobotError,
    g1::{contract::G1Command, definition::G1Definition},
};
use serde::Serialize;
use task_minigame::{policy::profile_contract, types::TaskProfile};

use super::{
    mobile_admission::MobileImageAdmission,
    mobile_center::{MobileCenterGoal, MobileCenterStep, MobileGripCentering},
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    mobile_hold::{MobileGripHolding, MobileHoldGoal, MobileHoldStep},
    mobile_lowering::{MobileGripLowering, MobileLowerGoal, MobileLowerStep},
    mobile_navigation::{
        MobileCarryGoal, MobileCarryNavigator, MobileNavigationStep, MobileScanGoal,
        MobileScanNavigator,
    },
    mobile_open::{MobileGraspOpening, MobileOpenGoal},
    mobile_raise::{MobileGripRaising, MobileRaiseGoal, MobileRaiseStep},
    mobile_regrasp::{MobileRegrasp, MobileRegraspGoal, MobileRegraspPhase, MobileRegraspStep},
    mobile_release::{MobileGripRelease, MobileReleaseGoal, MobileReleaseStep},
    mobile_restore::{MobileGripRestoring, MobileRestoreGoal, MobileRestoreStep},
    mobile_thumb::{MobileThumbGoal, MobileThumbPreparation, MobileThumbStep},
    mobile_wait::{MobileModelWaiting, MobileWaitGoal, MobileWaitStep},
    runner::{G1Measurement, G1ProgressCounts},
    task_objects::TaskObjectFrame,
    task_runner::{
        ArenaBodyStep, ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskExecution, ArenaTaskRunner,
        ArenaTaskRunnerConfig,
    },
};

#[derive(Clone, Debug)]
pub enum MobileAssistCommand {
    OriginalVla(ArenaTaskCommand),
    /// Original50frames, then a separately recorded traditional standing wait.
    /// The owner transitions without pausing or repeating a VLA frame.
    OriginalVlaThenWait(ArenaTaskCommand),
    /// Same unchanged original frames and finite wait, but the next original
    /// image may be acquired just after all frames of its predecessor chunk.
    /// Geometric classical skills still require stationary self admission.
    OriginalVlaBoundaryImageThenWait(ArenaTaskCommand),
    ObservedSkillThenWait(MobileObservedSkill),
    ClassicalCarry(MobileCarryGoal),
    ClassicalScan(MobileScanGoal),
    ClassicalReobserve(MobileScanGoal),
    ClassicalLower(MobileLowerGoal),
    ClassicalRaise(MobileRaiseGoal),
    /// Station-only initial pickup after completed calibrated grip settling.
    /// Separate from the original raise-after-carry contract.
    ClassicalPickupRaise(MobileRaiseGoal),
    ClassicalRelease(MobileReleaseGoal),
    ClassicalRestore(MobileRestoreGoal),
    ClassicalGripSettle(MobileHoldGoal),
    ClassicalGraspCenter(MobileCenterGoal),
    ClassicalGraspOpen(MobileOpenGoal),
    ClassicalRegrasp(MobileRegraspGoal),
    /// Finite lift test after a completed current-observation regrasp hold.
    ClassicalRegraspPickupRaise(MobileRaiseGoal),
    /// Fresh stationary verification of the unchanged, completed pickup target.
    ClassicalPickupHold(MobileHoldGoal),
    ClassicalHold(MobileHoldGoal),
    ClassicalThumbClearance(MobileThumbGoal),
    ClassicalModelWait(MobileWaitGoal),
}

#[derive(Clone, Debug, PartialEq)]
pub enum MobileObservedSkill {
    Carry(MobileCarryGoal),
    Scan(MobileScanGoal),
    GripSettle(MobileHoldGoal),
    Hold(MobileHoldGoal),
    Release(MobileReleaseGoal),
}
impl MobileObservedSkill {
    fn observation(&self) -> task_minigame::types::ObservationStamp {
        match self {
            Self::Carry(g) => g.observation,
            Self::Scan(g) => g.observation,
            Self::GripSettle(g) => g.observation,
            Self::Hold(g) => g.observation,
            Self::Release(g) => g.observation,
        }
    }
    fn command(&self) -> MobileAssistCommand {
        match self {
            Self::Carry(g) => MobileAssistCommand::ClassicalCarry(g.clone()),
            Self::Scan(g) => MobileAssistCommand::ClassicalScan(g.clone()),
            Self::GripSettle(g) => MobileAssistCommand::ClassicalGripSettle(g.clone()),
            Self::Hold(g) => MobileAssistCommand::ClassicalHold(g.clone()),
            Self::Release(g) => MobileAssistCommand::ClassicalRelease(g.clone()),
        }
    }
}

impl MobileAssistCommand {
    pub(crate) fn validate(&self) -> Result<(), RobotError> {
        match self {
            Self::ObservedSkillThenWait(skill) => skill.command().validate(),
            Self::OriginalVla(command)
            | Self::OriginalVlaThenWait(command)
            | Self::OriginalVlaBoundaryImageThenWait(command) => {
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
            Self::ClassicalPickupRaise(goal) | Self::ClassicalRegraspPickupRaise(goal) => {
                goal.validate()?;
                if goal.distance_m != 0.1 || goal.duration_ticks != 100 {
                    return Err(invalid(
                        "initial pickup requires the declared10cm/100Tick lift",
                    ));
                }
                Ok(())
            }
            Self::ClassicalRelease(goal) => goal.validate(),
            Self::ClassicalRestore(goal) => goal.validate(),
            Self::ClassicalGripSettle(goal) => goal.validate(),
            Self::ClassicalGraspCenter(goal) => goal.validate(),
            Self::ClassicalGraspOpen(goal) => goal.validate(),
            Self::ClassicalRegrasp(goal) => goal.validate(),
            Self::ClassicalHold(goal) | Self::ClassicalPickupHold(goal) => goal.validate(),
            Self::ClassicalThumbClearance(goal) => goal.validate(),
            Self::ClassicalModelWait(goal) => goal.validate(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MobileAssistExecution {
    OriginalVla(ArenaTaskExecution),
    ClassicalRegrasp {
        goal: MobileRegraspGoal,
        regrasp: MobileRegraspStep,
    },
    ClassicalGraspOpen {
        goal: MobileOpenGoal,
        opening: MobileReleaseStep,
    },
    ClassicalGraspCenter {
        goal: MobileCenterGoal,
        centering: MobileCenterStep,
    },
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
    ClassicalGripSettle {
        goal: MobileHoldGoal,
        holding: MobileHoldStep,
        grip: MobileGripReceipt,
    },
    ClassicalHold {
        goal: MobileHoldGoal,
        holding: MobileHoldStep,
    },
    ClassicalThumbClearance {
        goal: MobileThumbGoal,
        preparing: MobileThumbStep,
    },
    ClassicalModelWait {
        goal: MobileWaitGoal,
        waiting: MobileWaitStep,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileAssistStep {
    pub execution: MobileAssistExecution,
    pub body: ArenaBodyStep,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_admission: Option<MobileImageAdmission>,
}

/// No second world, physical pose setter, task-object sensor or model client.
pub struct MobileAssistRunner {
    owner: ArenaTaskRunner,
    station_fixture: bool,
    calibration: MobileGripCalibration,
    maximum_observation_wall_age_ms: u64,
    maximum_observation_age_ns: u64,
    continuous_skill: Option<MobileObservedSkill>,
    continuous_admission: Option<MobileImageAdmission>,
    rejected_observed_skill: Option<(MobileObservedSkill, super::mobile_wait::MobileRejectedImage)>,
    rejected_observation_frame_floor: u64,
    last_vla_execution: Option<ArenaTaskExecution>,
    last_vla_command: Option<G1Command>,
    last_controller_command: Option<G1Command>,
    waiting: Option<MobileModelWaiting>,
    carry: Option<CarryState>,
    scan: Option<ScanState>,
    reobserve: Option<ScanState>,
    lower: Option<MobileGripLowering>,
    raise: Option<MobileGripRaising>,
    pickup_raise: Option<MobileGripRaising>,
    pickup_hold: Option<MobileGripHolding>,
    release: Option<MobileGripRelease>,
    restore: Option<MobileGripRestoring>,
    grip_settle: Option<GripSettleState>,
    centering: Option<MobileGripCentering>,
    grasp_opening: Option<MobileGraspOpening>,
    regrasp: Option<MobileRegrasp>,
    hold: Option<MobileGripHolding>,
    thumb: Option<MobileThumbPreparation>,
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

struct GripSettleState {
    holding: MobileGripHolding,
    grip: MobileGripReceipt,
}

impl MobileAssistRunner {
    /// Opt-in factory only: original complete T2 scene/50Hz/native motors with
    /// the already measured4PGS candidate. Normal task loading is unchanged.
    pub fn load(config: &ArenaTaskRunnerConfig) -> Result<Self, RobotError> {
        Self::from_owner(
            config,
            ArenaTaskRunner::load_mobile_constraint_diagnostic(config)?,
            false,
        )
    }

    /// Distinct native-station transfer factory. Its completed original grasp
    /// uses the independently measured100Tick self-state hold before locomotion;
    /// the original complete-source factory retains its50Tick settling phase.
    pub fn load_mobile_station_fixture_diagnostic(
        config: &ArenaTaskRunnerConfig,
    ) -> Result<Self, RobotError> {
        Self::from_owner(
            config,
            ArenaTaskRunner::load_mobile_station_fixture_diagnostic(config)?,
            true,
        )
    }

    fn from_owner(
        config: &ArenaTaskRunnerConfig,
        owner: ArenaTaskRunner,
        station_fixture: bool,
    ) -> Result<Self, RobotError> {
        let ArenaTaskBodyConfig::MobileHomieV2(body) = &config.body else {
            return Err(invalid("mobile assist requires Homie_v2/N1.6"));
        };
        let definition = G1Definition::load(&body.definition, &body.definition_sha256)?;
        Ok(Self {
            owner,
            station_fixture,
            calibration: MobileGripCalibration::new(&definition)?,
            maximum_observation_wall_age_ms: config.max_observation_wall_age_ms,
            maximum_observation_age_ns: config.max_observation_age_ns,
            continuous_skill: None,
            continuous_admission: None,
            rejected_observed_skill: None,
            rejected_observation_frame_floor: 0,
            last_vla_execution: None,
            last_vla_command: None,
            last_controller_command: None,
            waiting: None,
            carry: None,
            scan: None,
            reobserve: None,
            lower: None,
            raise: None,
            pickup_raise: None,
            pickup_hold: None,
            release: None,
            restore: None,
            grip_settle: None,
            centering: None,
            grasp_opening: None,
            regrasp: None,
            hold: None,
            thumb: None,
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
        if let Some(waiting) = &self.waiting {
            return waiting.completed();
        }
        if self.continuous_skill.is_some() {
            return false;
        }
        self.completed_native_skill()
    }
    fn completed_native_skill(&self) -> bool {
        if let Some(regrasp) = &self.regrasp {
            return regrasp.completed();
        }
        if let Some(waiting) = &self.waiting {
            waiting.completed()
        } else if let Some(release) = &self.release {
            release.completed()
        } else if let Some(reobserve) = &self.reobserve {
            reobserve.navigator.completed()
        } else if let Some(restore) = &self.restore {
            restore.completed()
        } else if let Some(raise) = &self.raise {
            raise.completed()
        } else if let Some(lower) = &self.lower {
            lower.completed()
        } else if let Some(thumb) = &self.thumb {
            thumb.completed()
        } else if let Some(hold) = &self.hold {
            hold.completed()
        } else if let Some(opening) = &self.grasp_opening {
            opening.completed()
        } else if let Some(centering) = self
            .centering
            .as_ref()
            .filter(|_| self.grip_settle.is_none() && self.carry.is_none() && self.scan.is_none())
        {
            centering.completed()
        } else if self.carry.is_none() && self.scan.is_none() {
            if let Some(hold) = &self.pickup_hold {
                hold.completed()
            } else if let Some(pickup) = &self.pickup_raise {
                pickup.completed()
            } else {
                self.grip_settle
                    .as_ref()
                    .is_some_and(|s| s.holding.completed())
            }
        } else {
            self.carry.as_ref().is_some_and(|c| c.navigator.stopped())
                || self.scan.as_ref().is_some_and(|s| s.navigator.completed())
        }
    }

    fn skill_completed(&self, skill: &MobileObservedSkill) -> bool {
        match skill {
            MobileObservedSkill::Carry(g) => self
                .carry
                .as_ref()
                .is_some_and(|c| c.navigator.goal() == g && c.navigator.completed()),
            MobileObservedSkill::Scan(g) => self
                .scan
                .as_ref()
                .is_some_and(|s| s.navigator.goal() == g && s.navigator.completed()),
            MobileObservedSkill::GripSettle(g) => self
                .grip_settle
                .as_ref()
                .is_some_and(|s| s.holding.goal() == g && s.holding.completed()),
            MobileObservedSkill::Hold(g) => self
                .hold
                .as_ref()
                .is_some_and(|h| h.goal() == g && h.completed()),
            MobileObservedSkill::Release(g) => self
                .release
                .as_ref()
                .is_some_and(|r| r.goal() == g && r.completed()),
        }
    }
    fn classical_admission(
        &self,
        observation: task_minigame::types::ObservationStamp,
        state: &G1Measurement,
    ) -> Result<MobileImageAdmission, RobotError> {
        if let Some(admission) = &self.continuous_admission {
            if admission.matches(observation, state) {
                return Ok(admission.clone());
            }
        }
        MobileImageAdmission::at_current_boundary(observation, state)
    }

    fn prepare_classical(
        &self,
        observation: &task_minigame::types::ObservationStamp,
        state: &G1Measurement,
    ) -> Result<(G1Command, MobileGripReceipt), RobotError> {
        if self.regrasp.is_some() {
            return Err(invalid(
                "regrasp needs explicit fresh measured completion before lift/carry",
            ));
        }
        if self
            .grip_settle
            .as_ref()
            .is_some_and(|s| !s.holding.completed())
        {
            return Err(invalid("cannot replace active grip settling"));
        }
        if self.hold.as_ref().is_some_and(|h| !h.completed()) {
            return Err(invalid("cannot replace an active stationary hold"));
        }
        if self.pickup_hold.as_ref().is_some_and(|h| !h.completed()) {
            return Err(invalid("cannot replace active loaded-pickup verification"));
        }
        if self.thumb.as_ref().is_some_and(|t| !t.completed()) {
            return Err(invalid("cannot replace active thumb preparation"));
        }
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
            if !carry.navigator.stopped() {
                return Err(invalid("cannot replace an active carry"));
            }
            if let Some(thumb) = &self.thumb {
                return Ok((thumb.command().clone(), carry.grip.clone()));
            }
            if let Some(hold) = &self.hold {
                return Ok((hold.command().clone(), carry.grip.clone()));
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
        if let Some(settle) = &self.grip_settle {
            if observation.frame_id <= settle.holding.goal().observation.frame_id {
                return Err(invalid(
                    "transport turn requires a new image after grip settling",
                ));
            }
            if let Some(pickup) = &self.pickup_raise {
                if !pickup.completed() || observation.frame_id <= pickup.goal().observation.frame_id
                {
                    return Err(invalid(
                        "station turn requires a newer completed-pickup RGB boundary",
                    ));
                }
                // Keep the elevated measured-target posture. Returning the
                // pre-lift hold command would lower the box during the turn.
                return Ok((pickup.command().clone(), settle.grip.clone()));
            }
            return Ok((settle.holding.command().clone(), settle.grip.clone()));
        }
        let original = self
            .last_vla_command
            .as_ref()
            .ok_or_else(|| invalid("classical handoff lost original upper command"))?;
        if self.grasp_opening.is_some() {
            return Err(invalid(
                "opened pregrasp requires fresh250Tick RGB and verified insertion/source-close before hold or transport",
            ));
        }
        if let Some(center) = &self.centering {
            if !center.completed()
                || observation.frame_id <= center.goal().observation.frame_id
                || observation.sim_time_ns != state.sim_time_ns
                || state.source_tick != 250
            {
                return Err(invalid(
                    "closing requires a fresh250Tick image after one completed insertion",
                ));
            }
            let correction = self.calibration.correct(state, center.command())?;
            return Ok((correction.command, correction.receipt));
        }
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
        if !matches!(
            command,
            MobileAssistCommand::ObservedSkillThenWait(_)
                | MobileAssistCommand::ClassicalModelWait(_)
        ) {
            self.continuous_skill = None;
            self.continuous_admission = None;
            self.rejected_observed_skill = None;
        }
        let mut result = self.step_inner(command, guard);
        if let Ok(step) = &mut result {
            if !matches!(
                &step.execution,
                MobileAssistExecution::ClassicalModelWait { .. }
            ) {
                step.image_admission = self.continuous_admission.clone();
            }
            self.last_controller_command = Some(match &step.execution {
                MobileAssistExecution::OriginalVla(_) => {
                    self.last_vla_command.as_ref().unwrap().clone()
                }
                MobileAssistExecution::ClassicalRegrasp { regrasp, .. } => regrasp.command.clone(),
                MobileAssistExecution::ClassicalGraspCenter { centering, .. } => {
                    centering.command.clone()
                }
                MobileAssistExecution::ClassicalGraspOpen { opening, .. } => {
                    opening.command.clone()
                }
                MobileAssistExecution::ClassicalCarry { command, .. }
                | MobileAssistExecution::ClassicalScan { command, .. }
                | MobileAssistExecution::ClassicalReobserve { command, .. } => command.clone(),
                MobileAssistExecution::ClassicalLower { lowering, .. } => lowering.command.clone(),
                MobileAssistExecution::ClassicalRaise { raising, .. } => raising.command.clone(),
                MobileAssistExecution::ClassicalRelease { opening, .. } => opening.command.clone(),
                MobileAssistExecution::ClassicalRestore { restoring, .. } => {
                    restoring.command.clone()
                }
                MobileAssistExecution::ClassicalThumbClearance { preparing, .. } => {
                    preparing.command.clone()
                }
                MobileAssistExecution::ClassicalHold { holding, .. }
                | MobileAssistExecution::ClassicalGripSettle { holding, .. } => {
                    holding.command.clone()
                }
                MobileAssistExecution::ClassicalModelWait { waiting, .. } => {
                    waiting.command.clone()
                }
            });
        }
        if result.is_err() {
            self.halted = true;
        }
        result
    }
    fn continue_wait_after_image_rejection(
        &mut self,
        rejection: super::mobile_wait::MobileRejectedImage,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<MobileAssistStep, RobotError> {
        let goal = self
            .waiting
            .as_ref()
            .ok_or_else(|| {
                invalid("rejected image has no previously admitted finite standing wait")
            })?
            .goal()
            .clone();
        let mut step = self.step_inner(&MobileAssistCommand::ClassicalModelWait(goal), guard)?;
        let MobileAssistExecution::ClassicalModelWait { waiting, .. } = &mut step.execution else {
            return Err(invalid("rejected image did not preserve the standing wait"));
        };
        waiting.rejected_image = Some(rejection);
        Ok(step)
    }

    fn step_inner(
        &mut self,
        command: &MobileAssistCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<MobileAssistStep, RobotError> {
        guard()?;
        command.validate()?;
        if !matches!(
            command,
            MobileAssistCommand::ClassicalModelWait(_)
                | MobileAssistCommand::OriginalVlaThenWait(_)
                | MobileAssistCommand::OriginalVlaBoundaryImageThenWait(_)
                | MobileAssistCommand::ObservedSkillThenWait(_)
        ) {
            self.waiting = None;
        }
        match command {
            MobileAssistCommand::ObservedSkillThenWait(skill) => {
                let state = self.owner.measurement()?;
                // A rejected observation is discarded permanently. Continue
                // only the existing finite standing wait; do not retry it when
                // the self motion later changes or reset the waiting budget.
                if let Some((rejected, reason)) = &self.rejected_observed_skill
                    && rejected.observation() == skill.observation()
                {
                    if rejected != skill {
                        return Err(invalid("a rejected image cannot mutate into another skill"));
                    }
                    let reason = reason.clone();
                    return self.continue_wait_after_image_rejection(reason, guard);
                }
                if skill.observation().episode_id == state.episode_id
                    && skill.observation().frame_id <= self.rejected_observation_frame_floor
                {
                    return self.continue_wait_after_image_rejection(
                        super::mobile_wait::MobileRejectedImage {
                            observation: skill.observation(),
                            rejected_at_sim_ns: state.sim_time_ns,
                            reason: "previously discarded image cannot become an executed skill"
                                .into(),
                        },
                        guard,
                    );
                }
                if self.continuous_skill.as_ref() != Some(skill) {
                    if self
                        .continuous_skill
                        .as_ref()
                        .is_some_and(|previous| !self.skill_completed(previous))
                    {
                        return Err(invalid(
                            "cannot replace an active continuous observed skill",
                        ));
                    }
                    let waiting = self.waiting.as_ref().ok_or_else(|| {
                        invalid("observed skill requires an explicit stationary owner wait")
                    })?;
                    let admission = match waiting.admit_observed_skill(
                        skill.observation(),
                        &state,
                        &self.calibration,
                        self.maximum_observation_age_ns,
                    ) {
                        Ok(admission) => admission,
                        Err(error) => {
                            if skill.observation().episode_id == state.episode_id {
                                self.rejected_observation_frame_floor = self
                                    .rejected_observation_frame_floor
                                    .max(skill.observation().frame_id);
                            }
                            let rejection = super::mobile_wait::MobileRejectedImage {
                                observation: skill.observation(),
                                rejected_at_sim_ns: state.sim_time_ns,
                                reason: error.to_string(),
                            };
                            self.rejected_observed_skill = Some((skill.clone(), rejection.clone()));
                            return self.continue_wait_after_image_rejection(rejection, guard);
                        }
                    };
                    self.continuous_skill = Some(skill.clone());
                    self.continuous_admission = Some(admission);
                    self.rejected_observed_skill = None;
                }
                if self.skill_completed(skill) {
                    let goal =
                        self.waiting
                            .as_ref()
                            .map(|w| w.goal().clone())
                            .unwrap_or(MobileWaitGoal {
                                episode_id: state.episode_id,
                                request_id: skill.observation().frame_id,
                                execution_start_sim_ns: state.sim_time_ns,
                                duration_ticks: 200,
                            });
                    return self.step_inner(&MobileAssistCommand::ClassicalModelWait(goal), guard);
                }
                self.step_inner(&skill.command(), guard)
            }
            MobileAssistCommand::OriginalVlaThenWait(vla)
            | MobileAssistCommand::OriginalVlaBoundaryImageThenWait(vla) => {
                let boundary_image = matches!(
                    command,
                    MobileAssistCommand::OriginalVlaBoundaryImageThenWait(_)
                );
                let command = vla;
                if command.scheduled_start_sim_ns.is_some() {
                    return Err(invalid(
                        "waited VLA uses current owner admission, not a future action slot",
                    ));
                }
                let state = self.owner.measurement()?;
                if let Some(previous) = &self.last_vla_execution {
                    if command.chunk.sequence_id == previous.sequence_id {
                        // Compare against the accepted original chunk as well;
                        // an equal sequence cannot mutate it during standing.
                        if !self.owner.is_current_mobile_chunk(command) {
                            return Err(invalid("waited original observation identity changed"));
                        }
                        if previous.frame_index + 1
                            == profile_contract(TaskProfile::MobileBox).action_horizon
                        {
                            let goal = self.waiting.as_ref().map(|w| w.goal().clone()).unwrap_or(
                                MobileWaitGoal {
                                    episode_id: state.episode_id,
                                    request_id: previous.sequence_id,
                                    execution_start_sim_ns: state.sim_time_ns,
                                    duration_ticks: 200,
                                },
                            );
                            return self
                                .step_inner(&MobileAssistCommand::ClassicalModelWait(goal), guard);
                        }
                    } else if boundary_image {
                        if !original_boundary_image_window(
                            previous,
                            &state,
                            command,
                            self.maximum_observation_age_ns,
                        ) {
                            return Err(invalid(
                                "next original image is not from the completed predecessor observation window",
                            ));
                        }
                    } else if self.waiting.as_ref().is_none_or(|w| {
                        !w.admits_image(&state, command.chunk.observation.sim_time_ns)
                    }) {
                        return Err(invalid(
                            "replacement VLA image is outside the current stationary waiting interval",
                        ));
                    }
                } else if state.source_tick != 0 || command.chunk.observation.sim_time_ns != 0 {
                    return Err(invalid(
                        "first waited VLA requires the real zero-Tick image",
                    ));
                }
                self.step_inner(&MobileAssistCommand::OriginalVla(command.clone()), guard)
            }
            MobileAssistCommand::ClassicalModelWait(goal) => {
                let state = self.owner.measurement()?;
                if self.waiting.is_none() {
                    let complete_vla = self.last_vla_execution.as_ref().is_some_and(|e| {
                        e.frame_index + 1 == profile_contract(TaskProfile::MobileBox).action_horizon
                            && Some(state.sim_time_ns)
                                == e.execution_start_sim_ns.checked_add(1_000_000_000)
                    });
                    if !self.completed_native_skill() && !complete_vla {
                        return Err(invalid(
                            "model waiting requires a completed VLA chunk or classical skill",
                        ));
                    }
                    let command = self.last_controller_command.as_ref().ok_or_else(|| {
                        invalid("model waiting has no actually executed predecessor")
                    })?;
                    self.waiting = Some(MobileModelWaiting::new(
                        goal.clone(),
                        &state,
                        command.clone(),
                    )?);
                }
                let waiting = self.waiting.as_mut().unwrap();
                if waiting.goal() != goal {
                    return Err(invalid("waiting goal changed during execution"));
                }
                let waiting_step = waiting.update(&state)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&waiting_step.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalModelWait {
                        goal: goal.clone(),
                        waiting: waiting_step,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
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
                        .pickup_raise
                        .as_ref()
                        .is_some_and(|raise| !raise.completed())
                    || self
                        .release
                        .as_ref()
                        .is_some_and(|release| !release.completed())
                    || self
                        .reobserve
                        .as_ref()
                        .is_some_and(|r| !r.navigator.completed())
                    || self.restore.as_ref().is_some_and(|r| !r.completed())
                    || self.hold.as_ref().is_some_and(|h| !h.completed())
                    || self.centering.as_ref().is_some_and(|c| !c.completed())
                    || self.grasp_opening.is_some()
                    || self.regrasp.is_some()
                    || self
                        .grip_settle
                        .as_ref()
                        .is_some_and(|s| !s.holding.completed())
                {
                    return Err(invalid(
                        "cannot replace active classical carry with a VLA chunk",
                    ));
                }
                let step = self.owner.step_with_guard(command, guard)?;
                let original = self.owner.executed_mobile_command(&step.execution)?;
                self.last_vla_command = Some(original);
                self.last_vla_execution = Some(step.execution.clone());
                self.carry = None;
                self.scan = None;
                self.reobserve = None;
                self.lower = None;
                self.raise = None;
                self.pickup_raise = None;
                self.release = None;
                self.restore = None;
                self.hold = None;
                self.grip_settle = None;
                self.centering = None;
                self.grasp_opening = None;
                self.regrasp = None;
                self.waiting = None;
                self.original_transport_command = None;
                Ok(MobileAssistStep {
                    image_admission: None,
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
                    let admission = self.classical_admission(goal.observation, &state)?;
                    let navigator =
                        MobileCarryNavigator::new_with_admission(goal.clone(), &state, &admission)?;
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
                    self.hold = None;
                }
                let carry = self.carry.as_mut().unwrap();
                if carry.navigator.goal() != goal {
                    return Err(invalid(
                        "carry goal changed while executing; stop/reset required",
                    ));
                }
                let navigation = match carry.navigator.update(&state) {
                    Err(RobotError::Contract(reason))
                        if self.station_fixture
                            && reason
                                == "carry blocked: less than3cm forward progress in1second; pause/reset required" =>
                    {
                        carry.navigator.begin_station_progress_stop()?;
                        carry.navigator.update(&state)?
                    }
                    result => result?,
                };
                carry.command.navigation = navigation.navigation;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&carry.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
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
                    image_admission: None,
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
                    let admission = self.classical_admission(goal.observation, &state)?;
                    let navigator =
                        MobileScanNavigator::new_with_admission(goal.clone(), &state, &admission)?;
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
                    image_admission: None,
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
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalRestore {
                        goal: goal.clone(),
                        restoring,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalRegrasp(goal) => {
                if !self.station_fixture
                    || self.carry.is_some()
                    || self.scan.is_some()
                    || self.pickup_raise.is_some()
                    || self.grasp_opening.is_some()
                    || self.centering.is_some()
                    || self.waiting.is_some()
                    || self.hold.is_some()
                    || self.release.is_some()
                {
                    return Err(invalid(
                        "regrasp belongs only after the original stationary station grip",
                    ));
                }
                let state = self.owner.measurement()?;
                if self.regrasp.as_ref().is_none_or(|r| r.goal() != goal) {
                    self.classical_admission(goal.observation, &state)?;
                    if !self.last_vla_execution.as_ref().is_some_and(|p| {
                        p.profile == TaskProfile::MobileBox
                            && p.sequence_id == 4
                            && p.frame_index == 49
                            && p.admitted_chunks == 4
                            && p.execution_start_sim_ns == 3_000_000_000
                    }) {
                        return Err(invalid(
                            "regrasp requires exactly four completed original chunks",
                        ));
                    }
                    let command = if let Some(previous) = &self.regrasp {
                        if !previous.completed() || !goal.follows(previous.goal()) {
                            return Err(invalid(
                                "regrasp requires its own completed predecessor and fresh next image",
                            ));
                        }
                        previous.command().clone()
                    } else {
                        let settle = self
                            .grip_settle
                            .as_ref()
                            .ok_or_else(|| invalid("regrasp lost source close predecessor"))?;
                        if goal.attempt != 1
                            || goal.phase != MobileRegraspPhase::Open
                            || state.source_tick != 300
                            || !settle.holding.completed()
                            || settle.holding.goal().observation.sim_time_ns != 4_000_000_000
                        {
                            return Err(invalid(
                                "first regrasp requires completed original200..300 source hold",
                            ));
                        }
                        settle.holding.command().clone()
                    };
                    self.regrasp = Some(MobileRegrasp::new(
                        goal.clone(),
                        &state,
                        command,
                        &self.calibration,
                    )?);
                }
                let regrasp = self
                    .regrasp
                    .as_mut()
                    .unwrap()
                    .update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&regrasp.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalRegrasp {
                        goal: goal.clone(),
                        regrasp,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalGraspOpen(goal) => {
                if !self.station_fixture
                    || self.carry.is_some()
                    || self.scan.is_some()
                    || self.grip_settle.is_some()
                    || self.pickup_raise.is_some()
                    || self.centering.is_some()
                {
                    return Err(invalid(
                        "pregrasp opening belongs only before the first station grip close",
                    ));
                }
                let state = self.owner.measurement()?;
                if self.grasp_opening.is_none() {
                    let previous = self
                        .last_vla_execution
                        .as_ref()
                        .ok_or_else(|| invalid("opening lacks original grasp predecessor"))?;
                    if previous.profile != TaskProfile::MobileBox
                        || previous.sequence_id != 4
                        || previous.frame_index != 49
                        || previous.admitted_chunks != 4
                        || previous.execution_start_sim_ns != 3_000_000_000
                    {
                        return Err(invalid(
                            "opening requires exactly four completed original grasp chunks",
                        ));
                    }
                    self.classical_admission(goal.observation, &state)?;
                    let original = self
                        .last_vla_command
                        .clone()
                        .ok_or_else(|| invalid("opening lost original upper targets"))?;
                    self.grasp_opening = Some(MobileGraspOpening::new(
                        goal.clone(),
                        &state,
                        original,
                        &self.calibration,
                    )?);
                }
                let opening = self.grasp_opening.as_mut().unwrap();
                if opening.goal() != goal {
                    return Err(invalid("pregrasp opening goal changed during execution"));
                }
                let step = opening.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&step.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalGraspOpen {
                        goal: goal.clone(),
                        opening: step,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalGraspCenter(goal) => {
                if !self.station_fixture
                    || self.carry.is_some()
                    || self.scan.is_some()
                    || self.grip_settle.is_some()
                    || self.pickup_raise.is_some()
                    || self.grasp_opening.is_some()
                {
                    return Err(invalid(
                        "centering belongs only before the first station grip close",
                    ));
                }
                let state = self.owner.measurement()?;
                if self.centering.is_none() {
                    let previous = self
                        .last_vla_execution
                        .as_ref()
                        .ok_or_else(|| invalid("centering lacks the original grasp predecessor"))?;
                    if previous.profile != TaskProfile::MobileBox
                        || previous.sequence_id != 4
                        || previous.frame_index != 49
                        || previous.admitted_chunks != 4
                        || previous.execution_start_sim_ns != 3_000_000_000
                    {
                        return Err(invalid(
                            "centering requires exactly four completed original grasp chunks",
                        ));
                    }
                    self.classical_admission(goal.observation, &state)?;
                    let original = self
                        .last_vla_command
                        .clone()
                        .ok_or_else(|| invalid("centering lost original upper targets"))?;
                    self.centering =
                        Some(MobileGripCentering::new(goal.clone(), &state, original)?);
                }
                let center = self.centering.as_mut().unwrap();
                if center.goal() != goal {
                    return Err(invalid("centering goal changed during execution"));
                }
                let centering = center.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&centering.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalGraspCenter {
                        goal: goal.clone(),
                        centering,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalGripSettle(goal) => {
                if self.carry.is_some() || self.scan.is_some() || self.release.is_some() {
                    return Err(invalid("grip settling belongs only before first transport"));
                }
                let state = self.owner.measurement()?;
                if self.grip_settle.is_none() {
                    let (mut command, grip) = self.prepare_classical(&goal.observation, &state)?;
                    // Original VLA navigation is not a stationary-hold command.
                    command.navigation = [0.; 3];
                    let admission = self.classical_admission(goal.observation, &state)?;
                    let holding = if self.station_fixture {
                        MobileGripHolding::new_with_admission(
                            goal.clone(),
                            &state,
                            command,
                            &admission,
                        )?
                    } else {
                        MobileGripHolding::new_grip_settle_with_admission(
                            goal.clone(),
                            &state,
                            command,
                            &admission,
                        )?
                    };
                    self.grip_settle = Some(GripSettleState { holding, grip });
                }
                let settle = self.grip_settle.as_mut().unwrap();
                if settle.holding.goal() != goal {
                    return Err(invalid("grip-settling goal changed during execution"));
                }
                let holding = settle.holding.update(&state)?;
                let grip = settle.grip.clone();
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&holding.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalGripSettle {
                        goal: goal.clone(),
                        holding,
                        grip,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalPickupHold(goal) => {
                let state = self.owner.measurement()?;
                let pickup = self.pickup_raise.as_ref().ok_or_else(|| {
                    invalid("loaded pickup hold lacks its original completed lift")
                })?;
                if !self.station_fixture
                    || !pickup.completed()
                    || self.carry.is_some()
                    || self.scan.is_some()
                    || self.release.is_some()
                    || goal.observation.frame_id <= pickup.goal().observation.frame_id
                {
                    return Err(invalid(
                        "loaded pickup hold is not a fresh pre-transport boundary",
                    ));
                }
                if self.pickup_hold.as_ref().is_none_or(|h| h.goal() != goal) {
                    if self.pickup_hold.as_ref().is_some_and(|h| {
                        !h.completed() || goal.observation.frame_id <= h.goal().observation.frame_id
                    }) {
                        return Err(invalid(
                            "loaded pickup hold cannot replace active or newer goal",
                        ));
                    }
                    let admission = self.classical_admission(goal.observation, &state)?;
                    self.pickup_hold = Some(MobileGripHolding::new_with_admission(
                        goal.clone(),
                        &state,
                        pickup.command().clone(),
                        &admission,
                    )?);
                }
                let holding = self.pickup_hold.as_mut().unwrap().update(&state)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&holding.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalHold {
                        goal: goal.clone(),
                        holding,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalHold(goal) => {
                if self.release.is_some() {
                    return Err(invalid("hold after release requires a new grasp/reset"));
                }
                let state = self.owner.measurement()?;
                if self.hold.is_none() {
                    if self.carry.as_ref().is_none_or(|c| !c.navigator.stopped()) {
                        return Err(invalid("hold requires completed carry"));
                    }
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    let admission = self.classical_admission(goal.observation, &state)?;
                    self.hold = Some(MobileGripHolding::new_with_admission(
                        goal.clone(),
                        &state,
                        command,
                        &admission,
                    )?);
                }
                let hold = self.hold.as_mut().unwrap();
                if hold.goal() != goal {
                    return Err(invalid("hold goal changed during execution"));
                }
                let holding = hold.update(&state)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&holding.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalHold {
                        goal: goal.clone(),
                        holding,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalThumbClearance(goal) => {
                let state = self.owner.measurement()?;
                if self.thumb.is_none() {
                    if !self.station_fixture
                        || self.release.is_some()
                        || self.hold.as_ref().is_none_or(|h| {
                            !h.completed()
                                || goal.observation.frame_id <= h.goal().observation.frame_id
                        })
                        || self.carry.as_ref().is_none_or(|c| !c.navigator.stopped())
                    {
                        return Err(invalid(
                            "thumb preparation requires completed stationary station hold",
                        ));
                    }
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    self.thumb = Some(MobileThumbPreparation::new(goal.clone(), &state, command)?);
                }
                let thumb = self.thumb.as_mut().unwrap();
                if thumb.goal() != goal {
                    return Err(invalid("thumb preparation goal changed while executing"));
                }
                let preparing = thumb.update(&state)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&preparing.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalThumbClearance {
                        goal: goal.clone(),
                        preparing,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalRelease(goal) => {
                let state = self.owner.measurement()?;
                if self.release.is_none() {
                    if self.carry.as_ref().is_none_or(|c| !c.navigator.stopped())
                        || self.raise.as_ref().is_some_and(|r| !r.completed())
                        || self.hold.as_ref().is_some_and(|h| {
                            !h.completed()
                                || goal.observation.frame_id <= h.goal().observation.frame_id
                        })
                        || self.thumb.as_ref().is_some_and(|t| {
                            !t.completed()
                                || goal.observation.frame_id <= t.goal().observation.frame_id
                        })
                    {
                        return Err(invalid("release requires completed stationary carry"));
                    }
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    let admission = self.classical_admission(goal.observation, &state)?;
                    self.release = Some(MobileGripRelease::new_with_admission(
                        goal.clone(),
                        &state,
                        command,
                        &self.calibration,
                        &admission,
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
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalRelease {
                        goal: goal.clone(),
                        opening,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalRegraspPickupRaise(goal) => {
                let state = self.owner.measurement()?;
                if !self.station_fixture
                    || self.carry.is_some()
                    || self.scan.is_some()
                    || self.release.is_some()
                {
                    return Err(invalid("regrasp pickup belongs before station transport"));
                }
                if self.pickup_raise.is_none() {
                    let regrasp = self
                        .regrasp
                        .as_ref()
                        .ok_or_else(|| invalid("regrasp pickup lost its actual closed hold"))?;
                    if !regrasp.completed()
                        || regrasp.goal().phase != MobileRegraspPhase::Hold
                        || state.source_tick != regrasp.goal().end_tick()
                        || goal.observation.frame_id <= regrasp.goal().observation.frame_id
                    {
                        return Err(invalid(
                            "regrasp pickup requires fresh RGB after its own completed100Tick hold",
                        ));
                    }
                    let command = regrasp.command().clone();
                    self.classical_admission(goal.observation, &state)?;
                    self.pickup_raise =
                        Some(MobileGripRaising::new(goal.clone(), &state, command)?);
                    self.regrasp = None;
                }
                let pickup = self.pickup_raise.as_mut().unwrap();
                if pickup.goal() != goal {
                    return Err(invalid("regrasp pickup goal changed during actual lift"));
                }
                let raising = pickup.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&raising.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalRaise {
                        goal: goal.clone(),
                        raising,
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
            MobileAssistCommand::ClassicalPickupRaise(goal) => {
                let state = self.owner.measurement()?;
                if !self.station_fixture
                    || self.carry.is_some()
                    || self.scan.is_some()
                    || self.release.is_some()
                    || self.raise.is_some()
                    || self.lower.is_some()
                    || self
                        .grip_settle
                        .as_ref()
                        .is_none_or(|s| !s.holding.completed())
                {
                    return Err(invalid(
                        "initial pickup requires the completed stationary station grip before any transport",
                    ));
                }
                if self.pickup_raise.is_none() {
                    self.classical_admission(goal.observation, &state)?;
                    let (command, _) = self.prepare_classical(&goal.observation, &state)?;
                    self.pickup_raise =
                        Some(MobileGripRaising::new(goal.clone(), &state, command)?);
                }
                let pickup = self.pickup_raise.as_mut().unwrap();
                if pickup.goal() != goal {
                    return Err(invalid("initial pickup goal changed while executing"));
                }
                let raising = pickup.update(&state, &self.calibration)?;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&raising.command, guard)?;
                Ok(MobileAssistStep {
                    image_admission: None,
                    execution: MobileAssistExecution::ClassicalRaise {
                        goal: goal.clone(),
                        raising,
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
                    image_admission: None,
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
                    image_admission: None,
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

fn original_boundary_image_window(
    previous: &ArenaTaskExecution,
    state: &G1Measurement,
    next: &ArenaTaskCommand,
    maximum_age_ns: u64,
) -> bool {
    let image = next.chunk.observation;
    previous.frame_index == 49
        && previous.sequence_id.checked_add(1) == Some(next.chunk.sequence_id)
        && image.episode_id == state.episode_id
        && image.frame_id > previous.observation.frame_id
        && state.source_tick.checked_mul(20_000_000) == Some(state.sim_time_ns)
        && previous
            .execution_start_sim_ns
            .checked_add(1_000_000_000)
            .is_some_and(|end| state.sim_time_ns >= end && image.sim_time_ns >= end)
        && previous
            .execution_start_sim_ns
            .checked_add(1_500_000_000)
            .is_some_and(|end| image.sim_time_ns < end)
        && image.sim_time_ns % 20_000_000 == 0
        && state
            .sim_time_ns
            .checked_sub(image.sim_time_ns)
            .is_some_and(|age| age <= maximum_age_ns)
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

    #[test]
    fn declared_pickup_goal_cannot_silently_replace_the_legacy_raise_contract() {
        let mut goal = MobileRaiseGoal {
            observation: ObservationStamp {
                episode_id: 1,
                frame_id: 6,
                sim_time_ns: 6_000_000_000,
                captured_at_unix_ms: 1,
            },
            distance_m: 0.1,
            duration_ticks: 100,
        };
        assert!(
            MobileAssistCommand::ClassicalPickupRaise(goal.clone())
                .validate()
                .is_ok()
        );
        goal.distance_m = 0.18;
        goal.duration_ticks = 150;
        assert!(
            MobileAssistCommand::ClassicalRaise(goal.clone())
                .validate()
                .is_ok()
        );
        assert!(
            MobileAssistCommand::ClassicalPickupRaise(goal.clone())
                .validate()
                .is_err()
        );
        goal.distance_m = 0.1;
        goal.duration_ticks = 100;
        goal.observation.episode_id = 0;
        assert!(
            MobileAssistCommand::ClassicalPickupRaise(goal)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn boundary_original_reply_cannot_replace_an_active_or_foreign_chunk() {
        let stamp = ObservationStamp {
            episode_id: 1,
            frame_id: 1,
            sim_time_ns: 0,
            captured_at_unix_ms: 1,
        };
        let mut previous = ArenaTaskExecution {
            profile: TaskProfile::MobileBox,
            sequence_id: 1,
            frame_index: 49,
            observation: stamp,
            execution_start_sim_ns: 0,
            observation_age_ns: 0,
            observation_wall_age_ms: 0,
            decoded_waist_targets_rad: [0.; 3],
            admitted_chunks: 1,
        };
        let mut state = G1Measurement {
            episode_id: 1,
            source_tick: 50,
            sim_time_ns: 1_000_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_velocity_source: [0.; 3],
            root_angular_velocity_body: [0.; 3],
        };
        let mut chunk = PolicyActionChunk {
            profile: TaskProfile::MobileBox,
            sequence_id: 2,
            observation: ObservationStamp {
                frame_id: 2,
                sim_time_ns: 1_000_000_000,
                ..stamp
            },
            model_revision: String::new(),
            action_period_ns: 20_000_000,
            frames: vec![],
        };
        let command = |c: &PolicyActionChunk| ArenaTaskCommand {
            chunk: Arc::new(c.clone()),
            scheduled_start_sim_ns: None,
        };
        assert!(original_boundary_image_window(
            &previous,
            &state,
            &command(&chunk),
            1_000_000_000
        ));
        previous.frame_index = 48;
        assert!(!original_boundary_image_window(
            &previous,
            &state,
            &command(&chunk),
            1_000_000_000
        ));
        previous.frame_index = 49;
        for tick in [49, 75] {
            chunk.observation.sim_time_ns = tick * 20_000_000;
            assert!(!original_boundary_image_window(
                &previous,
                &state,
                &command(&chunk),
                1_000_000_000
            ));
        }
        chunk.observation.sim_time_ns = 1_000_000_000;
        chunk.sequence_id = 3;
        assert!(!original_boundary_image_window(
            &previous,
            &state,
            &command(&chunk),
            1_000_000_000
        ));
        chunk.sequence_id = 2;
        chunk.observation.episode_id = 2;
        assert!(!original_boundary_image_window(
            &previous,
            &state,
            &command(&chunk),
            1_000_000_000
        ));
        chunk.observation.episode_id = 1;
        state.source_tick = 101;
        state.sim_time_ns = 2_020_000_000;
        assert!(!original_boundary_image_window(
            &previous,
            &state,
            &command(&chunk),
            1_000_000_000
        ));
    }

    #[derive(Deserialize)]
    struct SavedContinuousScan {
        chunks: Vec<PolicyActionChunk>,
        prefix_sequence_ids: Vec<usize>,
        scan: MobileScanGoal,
        maximum_ticks: u64,
        preserve_executed_grasp: bool,
    }

    #[test]
    #[ignore = "one saved RGB-derived forward correction versus unchanged grip, identical prefix and manual scan; no fresh perception/task qualification"]
    fn real_mobile_saved_visual_forward_grip_diagnostic() -> Result<(), RobotError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Comparison {
            chunks: Vec<PolicyActionChunk>,
            prefix_sequence_ids: Vec<usize>,
            source_image: ObservationStamp,
            forward_distance_m: f64,
            correction_ticks: u32,
            enable_translation: bool,
            #[serde(default)]
            typed_grip_settle: bool,
            fixed_scan_navigation: Vec<[f32; 3]>,
            settle_ticks: u32,
        }
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let mut fixture: Comparison =
            serde_json::from_slice(&read("G1_MOBILE_VISUAL_FORWARD_FIXTURE")?)
                .map_err(|e| invalid(e.to_string()))?;
        if fixture.chunks.len() != 4
            || fixture.prefix_sequence_ids.is_empty()
            || fixture.prefix_sequence_ids.len() > 1000
            || fixture
                .prefix_sequence_ids
                .iter()
                .any(|i| !(1..=4).contains(i))
            || !fixture.forward_distance_m.is_finite()
            || !(0. ..=0.05).contains(&fixture.forward_distance_m)
            || fixture.correction_ticks != 50
            || (fixture.typed_grip_settle && fixture.enable_translation)
            || fixture.settle_ticks != 50
            || fixture.fixed_scan_navigation.is_empty()
            || fixture.fixed_scan_navigation.len() > 600
            || fixture
                .fixed_scan_navigation
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err(invalid(
                "forward comparison exceeds its one fixed bounded candidate",
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        for chunk in &mut fixture.chunks {
            chunk.observation.captured_at_unix_ms = now;
        }
        let output =
            std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&output).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let mut owner = MobileAssistRunner::load(&config)?;
        for &sequence in &fixture.prefix_sequence_ids {
            let command = MobileAssistCommand::OriginalVlaBoundaryImageThenWait(ArenaTaskCommand {
                chunk: Arc::new(fixture.chunks[sequence - 1].clone()),
                scheduled_start_sim_ns: None,
            });
            let step = owner.step_with_guard(&command, &mut || Ok(()))?;
            serde_json::to_writer(
                &mut trace,
                &serde_json::json!({"phase":"saved_boundary_prefix","body":step.body}),
            )
            .map_err(|e| invalid(e.to_string()))?;
            writeln!(trace).map_err(|e| invalid(e.to_string()))?;
        }
        let state = owner.measurement()?;
        let saved_sim_proof = owner
            .waiting
            .as_ref()
            .ok_or_else(|| invalid("forward fixture has no standing self history"))?
            .admit_observed_skill(
                fixture.source_image,
                &state,
                &owner.calibration,
                owner.maximum_observation_age_ns,
            )?;
        let previous = owner
            .last_controller_command
            .as_ref()
            .ok_or_else(|| invalid("forward fixture lacks actual original command"))?;
        let calibrated = owner.calibration.correct(&state, previous)?;
        let mut command = calibrated.command;
        command.navigation = [0.; 3];
        let mut observed_goal = fixture.source_image;
        // Offline saved proof only: refresh wall admission in the diagnostic;
        // the historical simulation/frame identity and original input stay pinned.
        observed_goal.captured_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let settle = MobileAssistCommand::ObservedSkillThenWait(MobileObservedSkill::GripSettle(
            MobileHoldGoal {
                observation: observed_goal,
            },
        ));
        for i in 0..fixture.correction_ticks {
            let state = owner.measurement()?;
            if fixture.typed_grip_settle {
                let step = owner.step_with_guard(&settle, &mut || Ok(()))?;
                let MobileAssistExecution::ClassicalGripSettle { holding, .. } = &step.execution
                else {
                    return Err(invalid(
                        "typed grip settle unexpectedly entered another phase",
                    ));
                };
                if serde_json::to_value(&holding.command).unwrap()
                    != serde_json::to_value(&command).unwrap()
                    || holding.holding_ticks != i + 1
                {
                    return Err(invalid(
                        "typed grip settle changed the comparison command or Tick count",
                    ));
                }
                serde_json::to_writer(
                    &mut trace,
                    &serde_json::json!({
                        "phase":"typed_grip_settle", "index":i, "body":step.body,
                        "holding":holding, "saved_proof_only":true,
                    }),
                )
                .map_err(|e| invalid(e.to_string()))?;
                writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                continue;
            }
            let correction = if fixture.enable_translation {
                let changed = owner.calibration.translate(
                    &state,
                    &command,
                    [
                        fixture.forward_distance_m / f64::from(fixture.correction_ticks),
                        0.,
                        0.,
                    ],
                )?;
                command = changed.command;
                Some(changed.receipt)
            } else {
                None
            };
            let body = owner
                .owner
                .step_mobile_assist_with_guard(&command, &mut || Ok(()))?;
            serde_json::to_writer(&mut trace, &serde_json::json!({"phase":"fixed_preparation","index":i,"body":body,
                "manual_saved_command":command,"correction":correction,"autonomous_execution":false})).map_err(|e| invalid(e.to_string()))?;
            writeln!(trace).map_err(|e| invalid(e.to_string()))?;
        }
        for (i, navigation) in fixture
            .fixed_scan_navigation
            .iter()
            .copied()
            .chain(std::iter::repeat_n([0.; 3], fixture.settle_ticks as usize))
            .enumerate()
        {
            command.navigation = navigation;
            let body = owner
                .owner
                .step_mobile_assist_with_guard(&command, &mut || Ok(()))?;
            serde_json::to_writer(
                &mut trace,
                &serde_json::json!({"phase":"identical_fixed_navigation","index":i,"body":body,
                "manual_saved_command":command,"autonomous_execution":false}),
            )
            .map_err(|e| invalid(e.to_string()))?;
            writeln!(trace).map_err(|e| invalid(e.to_string()))?;
        }
        trace.flush().map_err(|e| invalid(e.to_string()))?;
        fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"g1_saved_rgb_forward_grip_comparison_v1","qualified":false,"autonomous_execution":false,
            "fresh_rgb":0,"fresh_vla_calls":0,"counts":owner.progress_counts(),"source_image":fixture.source_image,
            "saved_sim_self_proof":saved_sim_proof,"fresh_wall_age_qualified":false,
            "enabled_translation":fixture.enable_translation,"forward_distance_m":fixture.forward_distance_m,
            "typed_grip_settle":fixture.typed_grip_settle,
            "correction_ticks":fixture.correction_ticks,"fixed_navigation_ticks":fixture.fixed_scan_navigation.len(),
            "settle_ticks":fixture.settle_ticks,"source_gap_correction":calibrated.receipt,"no_parameter_sweep":true,
            "scope":"exact saved native prefix, one actual-RGB-derived target translation, fixed identical manual navigation; independent contact audit required",
        })).map_err(|e| invalid(e.to_string()))?).map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }

    #[test]
    #[ignore = "saved actual prefix: rejected input must preserve identical finite standing mechanics, then admit a distinct saved image; zero fresh RGB/VLA"]
    fn real_mobile_rejected_image_preserves_wait_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let mut fixture: SavedContinuousScan =
            serde_json::from_slice(&read("G1_MOBILE_CONTINUOUS_SCAN_FIXTURE")?)
                .map_err(|e| invalid(e.to_string()))?;
        if fixture.chunks.len() != 4
            || fixture.prefix_sequence_ids.len() > 1000
            || fixture
                .prefix_sequence_ids
                .iter()
                .any(|i| !(1..=4).contains(i))
        {
            return Err(invalid(
                "rejection test requires the exact bounded saved prefix",
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        for chunk in &mut fixture.chunks {
            chunk.observation.captured_at_unix_ms = now;
        }
        fixture.scan.observation.captured_at_unix_ms = now;
        let mut discarded = MobileAssistRunner::load(&config)?;
        let mut reference = MobileAssistRunner::load(&config)?;
        let mut recovered = MobileAssistRunner::load(&config)?;
        for &sequence in &fixture.prefix_sequence_ids {
            let command = MobileAssistCommand::OriginalVlaThenWait(ArenaTaskCommand {
                chunk: Arc::new(fixture.chunks[sequence - 1].clone()),
                scheduled_start_sim_ns: None,
            });
            let a = discarded.step_with_guard(&command, &mut || Ok(()))?;
            let b = reference.step_with_guard(&command, &mut || Ok(()))?;
            let c = recovered.step_with_guard(&command, &mut || Ok(()))?;
            assert_eq!(
                serde_json::to_value(&a.body).unwrap(),
                serde_json::to_value(&b.body).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&a.body).unwrap(),
                serde_json::to_value(&c.body).unwrap()
            );
        }
        let wait = discarded
            .waiting
            .as_ref()
            .ok_or_else(|| invalid("prefix lacks finite wait"))?;
        let elapsed_wait_ticks = (discarded.measurement()?.sim_time_ns
            - wait.goal().execution_start_sim_ns)
            / 20_000_000;
        let remaining = wait.goal().duration_ticks - elapsed_wait_ticks as u32;
        assert!(remaining > 3);
        let goal = wait.goal().clone();
        let mut stale = fixture.scan.clone();
        stale.observation = fixture.chunks[3].observation;
        assert!(stale.observation.frame_id < fixture.scan.observation.frame_id);
        let rejected =
            MobileAssistCommand::ObservedSkillThenWait(MobileObservedSkill::Scan(stale.clone()));
        for i in 0..remaining {
            let a = discarded.step_with_guard(&rejected, &mut || Ok(()))?;
            let b = reference.step_with_guard(
                &MobileAssistCommand::ClassicalModelWait(goal.clone()),
                &mut || Ok(()),
            )?;
            assert_eq!(
                serde_json::to_value(&a.body).unwrap(),
                serde_json::to_value(&b.body).unwrap()
            );
            let MobileAssistExecution::ClassicalModelWait {
                goal: actual_goal,
                waiting,
            } = &a.execution
            else {
                panic!("rejected image executed a skill")
            };
            assert_eq!(actual_goal, &goal);
            assert_eq!(
                waiting.rejected_image.as_ref().unwrap().observation,
                stale.observation
            );
            assert_eq!(waiting.completed, i + 1 == remaining);
            assert!(a.image_admission.is_none());
            if i < 3 {
                let c = recovered.step_with_guard(&rejected, &mut || Ok(()))?;
                assert_eq!(
                    serde_json::to_value(&a.body).unwrap(),
                    serde_json::to_value(&c.body).unwrap()
                );
            }
        }
        assert!(discarded.completed_skill());
        let before = discarded.progress_counts();
        assert!(
            discarded
                .step_with_guard(&rejected, &mut || Ok(()))
                .is_err()
        );
        let after = discarded.progress_counts();
        assert_eq!(after.integration_count, before.integration_count);
        assert_eq!(after.torque_update_count, before.torque_update_count);
        assert_eq!(
            after.inference_attempt_count,
            before.inference_attempt_count
        );
        assert_eq!(
            after.successful_inference_count,
            before.successful_inference_count
        );
        assert!(after.halted);
        let valid = MobileAssistCommand::ObservedSkillThenWait(MobileObservedSkill::Scan(
            fixture.scan.clone(),
        ));
        let admitted = recovered.step_with_guard(&valid, &mut || Ok(()))?;
        assert!(matches!(
            admitted.execution,
            MobileAssistExecution::ClassicalScan { .. }
        ));
        assert_eq!(
            admitted.image_admission.as_ref().unwrap().observation(),
            fixture.scan.observation
        );
        let output =
            std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"g1_rejected_image_finite_wait_comparison_v1", "qualified":false,
            "fresh_rgb":0,"fresh_vla_calls":0,"saved_prefix_ticks_per_owner":fixture.prefix_sequence_ids.len(),
            "same_physical_prefix_owners":3,"identical_rejection_and_reference_wait_ticks":remaining,
            "rejected_image":stale.observation,"new_distinct_saved_image":fixture.scan.observation,
            "expired_wait_adds_no_ticks":true,"waiting_budget_reset":false,"discarded_counts":before,
            "reference_counts":reference.progress_counts(),"recovered_counts":recovered.progress_counts(),
            "recovered_admission":admitted.image_admission,
            "scope":"saved mechanical control/rejection semantics only; not fresh perception or task success",
        })).map_err(|e| invalid(e.to_string()))?).map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }

    #[test]
    #[ignore = "one frozen continuous prefix and scan handoff comparison; no fresh RGB/VLA or autonomous qualification"]
    fn real_mobile_saved_continuous_scan_handoff_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let mut fixture: SavedContinuousScan =
            serde_json::from_slice(&read("G1_MOBILE_CONTINUOUS_SCAN_FIXTURE")?)
                .map_err(|e| invalid(e.to_string()))?;
        if fixture.chunks.len() != 4
            || fixture.prefix_sequence_ids.is_empty()
            || fixture.prefix_sequence_ids.len() > 1000
            || fixture
                .prefix_sequence_ids
                .iter()
                .any(|i| !(1..=4).contains(i))
            || fixture.maximum_ticks > 1500
            || fixture.maximum_ticks <= fixture.prefix_sequence_ids.len() as u64
        {
            return Err(invalid(
                "continuous scan fixture exceeds fixed bounded comparison",
            ));
        }
        let now = || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        };
        for chunk in &mut fixture.chunks {
            chunk.observation.captured_at_unix_ms = now();
        }
        fixture.scan.observation.captured_at_unix_ms = now();
        let output =
            std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&output).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let mut owner = MobileAssistRunner::load(&config)?;
        for &sequence in &fixture.prefix_sequence_ids {
            let command = MobileAssistCommand::OriginalVlaThenWait(ArenaTaskCommand {
                chunk: Arc::new(fixture.chunks[sequence - 1].clone()),
                scheduled_start_sim_ns: None,
            });
            let step = owner.step_with_guard(&command, &mut || Ok(()))?;
            serde_json::to_writer(&mut trace, &step).map_err(|e| invalid(e.to_string()))?;
            writeln!(trace).map_err(|e| invalid(e.to_string()))?;
        }
        let state = owner.measurement()?;
        let proof = owner
            .waiting
            .as_ref()
            .ok_or_else(|| invalid("saved scan has no current waiting history"))?
            .admit_observed_skill(
                fixture.scan.observation,
                &state,
                &owner.calibration,
                owner.maximum_observation_age_ns,
            )?;
        let mut navigator =
            MobileScanNavigator::new_with_admission(fixture.scan.clone(), &state, &proof)?;
        let original = owner
            .last_controller_command
            .as_ref()
            .ok_or_else(|| invalid("saved prefix lacks executed upper command"))?;
        let mut command = if fixture.preserve_executed_grasp {
            original.clone()
        } else {
            owner.calibration.correct(&state, original)?.command
        };
        let mut actual_ticks = fixture.prefix_sequence_ids.len() as u64;
        while actual_ticks < fixture.maximum_ticks {
            let state = owner.measurement()?;
            let navigation = if navigator.completed() {
                None
            } else {
                Some(navigator.update(&state)?)
            };
            command.navigation = navigation.as_ref().map_or([0.; 3], |n| n.navigation);
            let body = owner
                .owner
                .step_mobile_assist_with_guard(&command, &mut || Ok(()))?;
            serde_json::to_writer(&mut trace, &serde_json::json!({
                "body": body, "manual_saved_scan_command":command, "navigation":navigation,
                "image_admission":proof, "diagnostic_saved_fixture":true,"autonomous_execution":false,
            })).map_err(|e| invalid(e.to_string()))?;
            writeln!(trace).map_err(|e| invalid(e.to_string()))?;
            actual_ticks += 1;
        }
        trace.flush().map_err(|e| invalid(e.to_string()))?;
        fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"g1_saved_continuous_scan_handoff_comparison_v1","actual_ticks":actual_ticks,
            "counts":owner.progress_counts(),"preserve_executed_grasp":fixture.preserve_executed_grasp,
            "scan_completed":navigator.completed(),"fresh_rgb":0,"fresh_vla_calls":0,
            "qualified":false,"autonomous_execution":false,"parameter_sweeps":0,
            "scope":"exact saved prefix, one fixed handoff alternative; independent truth audit required",
        })).map_err(|e| invalid(e.to_string()))?).map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }

    #[test]
    #[ignore = "frozen original1..4chunk prefix then50/100/200explicit waiting Ticks;0freshVLA/renderer work"]
    fn real_mobile_model_wait_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        let prefix: usize = std::env::var("G1_MOBILE_WAIT_PREFIX_CHUNKS")
            .map_err(|e| invalid(e.to_string()))?
            .parse()
            .map_err(|e| invalid(format!("{e:?}")))?;
        let wait_ticks = std::env::var("G1_MOBILE_WAIT_DURATION_TICKS")
            .ok()
            .map(|v| v.parse::<u32>().map_err(|e| invalid(e.to_string())))
            .transpose()?
            .unwrap_or(50);
        let automatic_handoff =
            std::env::var("G1_MOBILE_WAIT_AUTOMATIC_HANDOFF").as_deref() == Ok("1");
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
            || !(1..=4).contains(&prefix)
            || ![50, 100, 200].contains(&wait_ticks)
            || (automatic_handoff && wait_ticks != 200)
        {
            return Err(invalid(
                "waiting diagnostic requires an exact original1..4chunk prefix",
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
        let mut records = 0;
        let mut automatic_command = None;
        let result = (|| -> Result<(), RobotError> {
            for (index, mut chunk) in sequence.chunks.into_iter().take(prefix).enumerate() {
                // Only saved fixture wall freshness is renewed. Original image
                // identity and simulation acquisition time remain unchanged.
                chunk.observation.captured_at_unix_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| invalid(e.to_string()))?
                    .as_millis() as u64;
                let original = ArenaTaskCommand {
                    chunk: Arc::new(chunk),
                    scheduled_start_sim_ns: None,
                };
                let command = if automatic_handoff && index + 1 == prefix {
                    automatic_command =
                        Some(MobileAssistCommand::OriginalVlaThenWait(original.clone()));
                    MobileAssistCommand::OriginalVla(original)
                } else {
                    MobileAssistCommand::OriginalVla(original)
                };
                for _ in 0..50 {
                    let step = owner.step_with_guard(&command, &mut || Ok(()))?;
                    serde_json::to_writer(&mut trace, &step).map_err(|e| invalid(e.to_string()))?;
                    writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                    records += 1;
                }
            }
            let state = owner.measurement()?;
            let command = automatic_command.unwrap_or(MobileAssistCommand::ClassicalModelWait(
                MobileWaitGoal {
                    episode_id: state.episode_id,
                    request_id: prefix as u64,
                    execution_start_sim_ns: state.sim_time_ns,
                    duration_ticks: wait_ticks,
                },
            ));
            for tick in 0..wait_ticks {
                let step = owner.step_with_guard(&command, &mut || Ok(()))?;
                assert_eq!(owner.completed_skill(), tick + 1 == wait_ticks);
                serde_json::to_writer(&mut trace, &step).map_err(|e| invalid(e.to_string()))?;
                writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                records += 1;
            }
            Ok(())
        })();
        trace.flush().map_err(|e| invalid(e.to_string()))?;
        let counts = owner.progress_counts();
        fs::write(output,serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"g1_saved_prefix_model_wait_diagnostic_v1","qualified":false,
            "original_prefix_chunks":prefix,"explicit_wait_ticks":wait_ticks,"trace_records":records,
            "actual_integrations":counts.integration_count,"actual_torque_updates":counts.torque_update_count,
            "actual_body_policy_inferences":counts.successful_inference_count,"fresh_vla_calls":0,
            "new_current_camera_used":false,"mechanical_owner_wait_clock_not_image_stamp":true,
            "saved_fixture_wall_age_renewed":true,"world_or_contact_truth_input":false,
            "automatic_owner_handoff_without_pause":automatic_handoff,
            "completed":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),
        })).map_err(|e|invalid(e.to_string()))?).map_err(|e|invalid(e.to_string()))?;
        result
    }
    #[derive(Deserialize)]
    struct AuxiliaryHoldFixture {
        scan: MobileScanGoal,
        coarse: MobileCarryGoal,
        fine: MobileCarryGoal,
        hold: MobileHoldGoal,
        fine_walk_speed_m_s: Option<f32>,
        maximum_native_ticks: Option<u64>,
        fine_segments_m: Option<Vec<f32>>,
        coarse_segments_m: Option<Vec<f32>>,
        stop_after_coarse: Option<bool>,
        observed_fine_goals: Option<Vec<MobileCarryGoal>>,
        mechanical_release: Option<MobileReleaseGoal>,
    }
    #[test]
    #[ignore = "frozen actual finite prefix then bounded standing;0freshVLA/renderer work"]
    fn real_mobile_auxiliary_hold_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        let mut goals: AuxiliaryHoldFixture =
            serde_json::from_slice(&read("G1_MOBILE_AUX_HOLD_GOALS")?)
                .map_err(|e| invalid(e.to_string()))?;
        let maximum_ticks = goals.maximum_native_ticks.unwrap_or(1700);
        if !(1700..=3150).contains(&maximum_ticks) {
            return Err(invalid("saved mechanical fixture exceeds3150Tick budget"));
        }
        let fine_speed = goals.fine_walk_speed_m_s;
        let segments = goals.fine_segments_m;
        let coarse_segments = goals.coarse_segments_m;
        let stop_after_coarse = goals.stop_after_coarse.unwrap_or(false);
        let observed_fine_goals = goals.observed_fine_goals;
        let mechanical_release = goals.mechanical_release;
        if let Some(observed) = &observed_fine_goals {
            if stop_after_coarse
                || coarse_segments.is_some()
                || segments.is_some()
                || fine_speed.is_some()
                || observed.is_empty()
                || observed.len() > 5
                || observed.iter().any(|goal| {
                    goal.validate().is_err()
                        || goal.relative_distance_m != 0.1
                        || goal.observation.episode_id != goals.scan.observation.episode_id
                })
                || observed.windows(2).any(|pair| {
                    pair[1].observation.frame_id <= pair[0].observation.frame_id
                        || pair[1].observation.sim_time_ns <= pair[0].observation.sim_time_ns
                })
            {
                return Err(invalid("invalid saved actual fine-goal comparison"));
            }
        }
        if mechanical_release.is_some() && observed_fine_goals.is_none() {
            return Err(invalid(
                "mechanical opening requires an exact observed prefix",
            ));
        }
        if let Some(distances) = &coarse_segments {
            if !stop_after_coarse
                || fine_speed.is_some()
                || segments.is_some()
                || distances.is_empty()
                || distances.len() > 5
                || distances
                    .iter()
                    .any(|d| !d.is_finite() || !(0.1..=0.3).contains(d))
                || (distances.iter().map(|d| d + 0.05).sum::<f32>()
                    - (goals.coarse.relative_distance_m + 0.05))
                    .abs()
                    > 0.00001
            {
                return Err(invalid("invalid finite coarse carry comparison"));
            }
        }
        if let Some(distances) = &segments {
            if fine_speed.is_some()
                || distances.is_empty()
                || distances.len() > 5
                || distances
                    .iter()
                    .any(|d| !d.is_finite() || !(0.1..=0.2).contains(d))
                || distances.iter().map(|d| d + 0.05).sum::<f32>() > 0.7
            {
                return Err(invalid("invalid finite segmented carry comparison"));
            }
        }
        let self_clock_suffix =
            fine_speed.is_some() || segments.is_some() || mechanical_release.is_some();
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
        {
            return Err(invalid(
                "auxiliary hold requires four frozen actual native replies",
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
        let now = || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        };
        let mut ticks = 0;
        let result = (|| -> Result<(), RobotError> {
            for mut chunk in sequence.chunks {
                chunk.observation.captured_at_unix_ms = now();
                let command = MobileAssistCommand::OriginalVla(ArenaTaskCommand {
                    chunk: Arc::new(chunk),
                    scheduled_start_sim_ns: None,
                });
                for _ in 0..50 {
                    let step = owner.step_with_guard(&command, &mut || Ok(()))?;
                    ticks += 1;
                    serde_json::to_writer(&mut trace, &step).map_err(|e| invalid(e.to_string()))?;
                    writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                }
            }
            goals.scan.observation.captured_at_unix_ms = now();
            goals.coarse.observation.captured_at_unix_ms = now();
            goals.fine.observation.captured_at_unix_ms = now();
            goals.hold.observation.captured_at_unix_ms = now();
            let mut commands = vec![MobileAssistCommand::ClassicalScan(goals.scan)];
            if let Some(distances) = &coarse_segments {
                for distance in distances {
                    let mut goal = goals.coarse.clone();
                    goal.relative_distance_m = *distance;
                    commands.push(MobileAssistCommand::ClassicalCarry(goal));
                }
            } else {
                commands.push(MobileAssistCommand::ClassicalCarry(goals.coarse));
            }
            if stop_after_coarse {
                // Finite saved-prefix comparison ends before any fine/hold action.
            } else if let Some(observed) = &observed_fine_goals {
                commands.extend(
                    observed
                        .iter()
                        .cloned()
                        .map(MobileAssistCommand::ClassicalCarry),
                );
            } else if let Some(distances) = &segments {
                for distance in distances {
                    let mut goal = goals.fine.clone();
                    goal.relative_distance_m = *distance;
                    commands.push(MobileAssistCommand::ClassicalCarry(goal));
                }
            } else {
                commands.push(MobileAssistCommand::ClassicalCarry(goals.fine));
            }
            let hold_stage = commands.len();
            if !stop_after_coarse {
                commands.push(MobileAssistCommand::ClassicalHold(goals.hold));
            }
            if let Some(goal) = &mechanical_release {
                commands.push(MobileAssistCommand::ClassicalRelease(goal.clone()));
            }
            for (stage, mut command) in commands.into_iter().enumerate() {
                if observed_fine_goals.is_some() && (2..hold_stage).contains(&stage) {
                    let MobileAssistCommand::ClassicalCarry(goal) = &mut command else {
                        unreachable!()
                    };
                    // Keep the saved actual image frame/sim identity; only
                    // renew the wall age for this explicitly offline fixture.
                    goal.observation.captured_at_unix_ms = now();
                }
                if coarse_segments.is_some() && stage > 1 {
                    let state = owner.measurement()?;
                    let MobileAssistCommand::ClassicalCarry(goal) = &mut command else {
                        unreachable!()
                    };
                    // Saved heading and self-clock only; no new RGB is claimed.
                    goal.observation.frame_id += (stage - 1) as u64;
                    goal.observation.sim_time_ns = state.sim_time_ns;
                    goal.observation.captured_at_unix_ms = now();
                }
                if segments.is_some() && stage > 2 && stage < hold_stage {
                    let state = owner.measurement()?;
                    let MobileAssistCommand::ClassicalCarry(goal) = &mut command else {
                        unreachable!()
                    };
                    // Saved-image heading + self-clock mechanical suffix only.
                    // Synthetic stamps are never passed off as new RGB frames.
                    goal.observation.frame_id += (stage - 2) as u64;
                    goal.observation.sim_time_ns = state.sim_time_ns;
                    goal.observation.captured_at_unix_ms = now();
                }
                if stage == hold_stage && self_clock_suffix {
                    let state = owner.measurement()?;
                    let MobileAssistCommand::ClassicalHold(goal) = &mut command else {
                        unreachable!()
                    };
                    // Self-clock mechanical hold only: there is no new image.
                    goal.observation.frame_id += segments.as_ref().map_or(0, |v| v.len() as u64);
                    goal.observation.sim_time_ns = state.sim_time_ns;
                    goal.observation.captured_at_unix_ms = now();
                }
                if stage == hold_stage + 1 && mechanical_release.is_some() {
                    let state = owner.measurement()?;
                    let MobileAssistCommand::ClassicalRelease(goal) = &mut command else {
                        unreachable!()
                    };
                    // Explicit mechanical self-clock suffix; this is not a
                    // fresh placement image or permission for live opening.
                    goal.observation.sim_time_ns = state.sim_time_ns;
                    goal.observation.captured_at_unix_ms = now();
                }
                let mut speed_selected = false;
                loop {
                    let step = owner.step_with_guard(&command, &mut || Ok(()))?;
                    if stage == 2 && !speed_selected {
                        if let Some(speed) = fine_speed {
                            owner
                                .carry
                                .as_mut()
                                .ok_or_else(|| invalid("fine causal test has no carry"))?
                                .navigator
                                .diagnostic_set_walk_speed(speed)?;
                        }
                        speed_selected = true;
                    }
                    ticks += 1;
                    let mut record =
                        serde_json::to_value(&step).map_err(|e| invalid(e.to_string()))?;
                    record["diagnostic_robot_background_contacts"] =
                        serde_json::json!(owner.owner.diagnostic_robot_background_contacts());
                    serde_json::to_writer(&mut trace, &record)
                        .map_err(|e| invalid(e.to_string()))?;
                    writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                    if owner.completed_skill() {
                        break;
                    }
                    if ticks >= maximum_ticks {
                        return Err(invalid("auxiliary hold exceeded declared fixture budget"));
                    }
                }
            }
            Ok(())
        })();
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(file,&serde_json::json!({"qualified":false,"actual_integrations":ticks,
            "fresh_vla_calls":0,"saved_actual_prefix":true,"new_current_camera_used":false,
            "completed_hold":result.is_ok() && !stop_after_coarse,"completed_coarse_comparison":result.is_ok() && stop_after_coarse,
            "fine_walk_speed_override_m_s":fine_speed,"maximum_native_ticks":maximum_ticks,
            "fine_segments_m":segments,"coarse_segments_m":coarse_segments,"stop_after_coarse":stop_after_coarse,
            "observed_fine_goals":observed_fine_goals,"mechanical_release":mechanical_release,
            "self_clock_suffix_stamps_without_new_images":self_clock_suffix || coarse_segments.is_some(),
            "error":result.as_ref().err().map(ToString::to_string),"scope":"finite saved-prefix mechanical comparison only; no fresh navigation/release/task qualification"})).map_err(|e|invalid(e.to_string()))?;
        result
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
                    MobileAssistExecution::ClassicalModelWait { .. } => {
                        ("traditional_model_wait", None)
                    }
                    MobileAssistExecution::ClassicalRegrasp { .. }
                    | MobileAssistExecution::ClassicalGraspCenter { .. }
                    | MobileAssistExecution::ClassicalGraspOpen { .. }
                    | MobileAssistExecution::ClassicalScan { .. }
                    | MobileAssistExecution::ClassicalReobserve { .. }
                    | MobileAssistExecution::ClassicalRestore { .. }
                    | MobileAssistExecution::ClassicalHold { .. }
                    | MobileAssistExecution::ClassicalThumbClearance { .. }
                    | MobileAssistExecution::ClassicalGripSettle { .. }
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
                        MobileAssistExecution::ClassicalModelWait { .. } => {
                            ("traditional_model_wait", None)
                        }
                        MobileAssistExecution::ClassicalRegrasp { .. }
                        | MobileAssistExecution::ClassicalGraspCenter { .. }
                        | MobileAssistExecution::ClassicalGraspOpen { .. }
                        | MobileAssistExecution::ClassicalScan { .. }
                        | MobileAssistExecution::ClassicalReobserve { .. }
                        | MobileAssistExecution::ClassicalRestore { .. }
                        | MobileAssistExecution::ClassicalHold { .. }
                        | MobileAssistExecution::ClassicalThumbClearance { .. }
                        | MobileAssistExecution::ClassicalGripSettle { .. }
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
