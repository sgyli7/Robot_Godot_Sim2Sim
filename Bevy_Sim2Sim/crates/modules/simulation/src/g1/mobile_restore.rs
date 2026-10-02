//! Restore the episode's calibrated transport posture using self FK only.

use super::{
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    runner::G1Measurement,
};
use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileRestoreGoal {
    pub observation: ObservationStamp,
    pub duration_ticks: u32,
}
impl MobileRestoreGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.observation.episode_id == 0
            || self.observation.frame_id == 0
            || self.observation.captured_at_unix_ms == 0
            || self.observation.sim_time_ns % 20_000_000 != 0
            || !(100..=200).contains(&self.duration_ticks)
        {
            return Err(invalid(
                "restoring requires a current image and 100–200 Ticks",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct MobileRestoreStep {
    pub restoring_ticks: u32,
    pub settling_ticks: u32,
    pub completed: bool,
    pub total_cartesian_restore_root_source_m: [f64; 3],
    pub correction: Option<MobileGripReceipt>,
    pub command: G1Command,
}
pub struct MobileGripRestoring {
    goal: MobileRestoreGoal,
    offset: [f64; 3],
    next_tick: u64,
    restoring_ticks: u32,
    settling_ticks: u32,
    completed: bool,
    command: G1Command,
}
impl MobileGripRestoring {
    pub fn new(
        goal: MobileRestoreGoal,
        state: &G1Measurement,
        current: G1Command,
        original: &G1Command,
        calibration: &MobileGripCalibration,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        if goal.observation.episode_id != state.episode_id
            || goal.observation.sim_time_ns != state.sim_time_ns
        {
            return Err(invalid(
                "restore handoff requires the current stationary boundary",
            ));
        }
        let offset = calibration.transport_restore_offset(state, &current, original)?;
        Ok(Self {
            goal,
            offset,
            next_tick: state.source_tick,
            restoring_ticks: 0,
            settling_ticks: 0,
            completed: false,
            command: current,
        })
    }
    pub fn goal(&self) -> &MobileRestoreGoal {
        &self.goal
    }
    pub fn completed(&self) -> bool {
        self.completed
    }
    pub fn command(&self) -> &G1Command {
        &self.command
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        calibration: &MobileGripCalibration,
    ) -> Result<MobileRestoreStep, RobotError> {
        if self.completed
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != self.next_tick * 20_000_000
        {
            return Err(invalid("foreign/repeated restore self-state boundary"));
        }
        let correction = if self.restoring_ticks < self.goal.duration_ticks {
            let corrected = calibration.translate(
                state,
                &self.command,
                self.offset.map(|v| v / f64::from(self.goal.duration_ticks)),
            )?;
            self.command = corrected.command;
            self.restoring_ticks += 1;
            Some(corrected.receipt)
        } else {
            self.settling_ticks += 1;
            self.completed = self.settling_ticks >= 100;
            None
        };
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or_else(|| invalid("restore Tick overflow"))?;
        Ok(MobileRestoreStep {
            restoring_ticks: self.restoring_ticks,
            settling_ticks: self.settling_ticks,
            completed: self.completed,
            total_cartesian_restore_root_source_m: self.offset,
            correction,
            command: self.command.clone(),
        })
    }
}
fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}
