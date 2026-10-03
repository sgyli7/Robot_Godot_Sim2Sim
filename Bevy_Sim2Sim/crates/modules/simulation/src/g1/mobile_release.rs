//! Finite palm opening and real physical settling; self sensors/FK only.

use super::{
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    runner::G1Measurement,
};
use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileReleaseGoal {
    pub observation: ObservationStamp,
    pub target_palm_gap_m: f64,
    pub duration_ticks: u32,
}
impl MobileReleaseGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.observation.episode_id == 0
            || self.observation.frame_id == 0
            || self.observation.captured_at_unix_ms == 0
            || self.observation.sim_time_ns % 20_000_000 != 0
            || !self.target_palm_gap_m.is_finite()
            || !(0.25..=0.35).contains(&self.target_palm_gap_m)
            || !(50..=150).contains(&self.duration_ticks)
        {
            return Err(invalid(
                "release requires current stamp,25–35cm palm gap and50–150Ticks",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileReleaseStep {
    pub command: G1Command,
    pub commanded_palm_gap_m: f64,
    pub opening_ticks: u32,
    pub settling_ticks: u32,
    pub completed: bool,
    pub correction: Option<MobileGripReceipt>,
}

pub struct MobileGripRelease {
    goal: MobileReleaseGoal,
    next_tick: u64,
    increment: f64,
    command: G1Command,
    opening_ticks: u32,
    settling_ticks: u32,
    completed: bool,
}
impl MobileGripRelease {
    pub fn new(
        goal: MobileReleaseGoal,
        state: &G1Measurement,
        command: G1Command,
        calibration: &MobileGripCalibration,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if goal.observation.episode_id != state.episode_id
            || goal.observation.sim_time_ns != state.sim_time_ns
            || command.navigation != [0.; 3]
        {
            return Err(invalid(
                "release needs completed stationary current carry boundary",
            ));
        }
        let increment = calibration.horizontal_opening_increment(
            state,
            &command,
            goal.target_palm_gap_m,
            goal.duration_ticks,
        )?;
        if !(0. ..=0.004).contains(&increment) {
            return Err(invalid(
                "release opening increment outside bounded envelope",
            ));
        }
        Ok(Self {
            goal,
            next_tick: state.source_tick,
            increment,
            command,
            opening_ticks: 0,
            settling_ticks: 0,
            completed: false,
        })
    }
    pub fn goal(&self) -> &MobileReleaseGoal {
        &self.goal
    }
    pub fn completed(&self) -> bool {
        self.completed
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        calibration: &MobileGripCalibration,
    ) -> Result<MobileReleaseStep, RobotError> {
        if self.completed
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != state.source_tick * 20_000_000
        {
            return Err(invalid("foreign/repeated/completed release state"));
        }
        let correction = if self.opening_ticks < self.goal.duration_ticks {
            let corrected = calibration.spread_horizontal(state, &self.command, self.increment)?;
            self.command = corrected.command;
            self.opening_ticks += 1;
            Some(corrected.receipt)
        } else {
            self.settling_ticks += 1;
            self.completed = self.settling_ticks >= 125;
            None
        };
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or_else(|| invalid("release Tick overflow"))?;
        Ok(MobileReleaseStep {
            command: self.command.clone(),
            commanded_palm_gap_m: calibration.commanded_gap(state, &self.command)?,
            opening_ticks: self.opening_ticks,
            settling_ticks: self.settling_ticks,
            completed: self.completed,
            correction,
        })
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_grasp_sized_opening_and_foreign_time_grid() {
        let mut goal = MobileReleaseGoal {
            observation: ObservationStamp {
                episode_id: 1,
                frame_id: 9,
                sim_time_ns: 20_000_000,
                captured_at_unix_ms: 1,
            },
            target_palm_gap_m: 0.3,
            duration_ticks: 100,
        };
        assert!(goal.validate().is_ok());
        goal.target_palm_gap_m = 0.2;
        assert!(goal.validate().is_err());
        goal.target_palm_gap_m = 0.351;
        assert!(goal.validate().is_err());
        goal.target_palm_gap_m = 0.3;
        goal.duration_ticks = 151;
        assert!(goal.validate().is_err());
        goal.duration_ticks = 100;
        goal.observation.sim_time_ns += 1;
        assert!(goal.validate().is_err());
    }
}
