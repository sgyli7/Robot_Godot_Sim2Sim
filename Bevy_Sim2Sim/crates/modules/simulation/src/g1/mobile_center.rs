//! One finite shared insertion before source-calibrated closing. Uses self state
//! and an admitted visual offset only; no box/contact/world pose input.

use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

use super::{
    mobile_admission::MobileImageAdmission,
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    runner::G1Measurement,
};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileCenterGoal {
    pub observation: ObservationStamp,
    pub common_offset_root_source_m: [f64; 3],
    pub duration_ticks: u32,
}

impl MobileCenterGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let s = self.observation;
        let offset = self.common_offset_root_source_m;
        let length = offset.iter().map(|x| x * x).sum::<f64>().sqrt();
        if s.episode_id == 0
            || s.frame_id == 0
            || s.captured_at_unix_ms == 0
            || s.sim_time_ns != 4_000_000_000
            || self.duration_ticks != 50
            || !offset.iter().all(|x| x.is_finite())
            || !(0.000_001..=0.05).contains(&length)
        {
            return Err(invalid(
                "pre-grip centering requires current200Tick RGB, one<=5cm/50Tick shared insertion",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileCenterStep {
    pub command: G1Command,
    pub completed_ticks: u32,
    pub commanded_common_translation_m: [f64; 3],
    pub correction: MobileGripReceipt,
    pub completed: bool,
    pub fresh_post_insertion_rgb_required_before_closing: bool,
}

pub struct MobileGripCentering {
    goal: MobileCenterGoal,
    command: G1Command,
    next_tick: u64,
    ticks: u32,
}

impl MobileGripCentering {
    pub fn new(
        goal: MobileCenterGoal,
        state: &G1Measurement,
        mut original: G1Command,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        original.validate()?;
        MobileImageAdmission::at_current_boundary(goal.observation, state)?;
        if state.source_tick != 200 {
            return Err(invalid(
                "pre-grip insertion cannot follow a previous close or transport",
            ));
        }
        // Keep the original VLA arm/finger target geometry until insertion
        // completes; do not apply the smaller calibrated squeeze gap here.
        original.navigation = [0.; 3];
        Ok(Self {
            goal,
            command: original,
            next_tick: state.source_tick,
            ticks: 0,
        })
    }
    pub fn goal(&self) -> &MobileCenterGoal {
        &self.goal
    }
    pub fn completed(&self) -> bool {
        self.ticks == self.goal.duration_ticks
    }
    pub fn command(&self) -> &G1Command {
        &self.command
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        calibration: &MobileGripCalibration,
    ) -> Result<MobileCenterStep, RobotError> {
        if self.completed()
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != self.next_tick * 20_000_000
        {
            return Err(invalid(
                "centering received repeated/reset/nonsequential owner state",
            ));
        }
        let increment = self
            .goal
            .common_offset_root_source_m
            .map(|x| x / f64::from(self.goal.duration_ticks));
        let correction = calibration.translate(state, &self.command, increment)?;
        self.command = correction.command;
        self.next_tick += 1;
        self.ticks += 1;
        Ok(MobileCenterStep {
            command: self.command.clone(),
            completed_ticks: self.ticks,
            commanded_common_translation_m: self
                .goal
                .common_offset_root_source_m
                .map(|x| x * f64::from(self.ticks) / f64::from(self.goal.duration_ticks)),
            correction: correction.receipt,
            completed: self.completed(),
            fresh_post_insertion_rgb_required_before_closing: true,
        })
    }
}

fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn goal() -> MobileCenterGoal {
        MobileCenterGoal {
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 5,
                sim_time_ns: 4_000_000_000,
                captured_at_unix_ms: 123,
            },
            common_offset_root_source_m: [0.04, 0., 0.],
            duration_ticks: 50,
        }
    }
    #[test]
    fn closing_or_transport_clock_cannot_enter_the_pre_grip_stage() {
        let g = goal();
        assert!(g.validate().is_ok());
        for clock in [0, 3_980_000_000, 5_000_000_000, 6_000_000_000] {
            let mut changed = g.clone();
            changed.observation.sim_time_ns = clock;
            assert!(changed.validate().is_err());
        }
    }
    #[test]
    fn one_finite_insertion_cannot_be_clipped_or_extended() {
        for offset in [
            [f64::NAN, 0., 0.],
            [0.050_001, 0., 0.],
            [0., 0., 0.],
            [0.04, 0.04, 0.],
        ] {
            let mut g = goal();
            g.common_offset_root_source_m = offset;
            assert!(g.validate().is_err());
        }
        for ticks in [0, 49, 51, 200] {
            let mut g = goal();
            g.duration_ticks = ticks;
            assert!(g.validate().is_err());
        }
    }
}
