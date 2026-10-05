//! Finite pre-grasp opening through the original bounded release IK primitive.
//! The sole owner must pause after50Ticks for a new paired RGB observation.

use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

use super::{
    mobile_grip::MobileGripCalibration,
    mobile_release::{MobileGripRelease, MobileReleaseGoal, MobileReleaseStep},
    runner::G1Measurement,
};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileOpenGoal {
    pub observation: ObservationStamp,
}

impl MobileOpenGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let s = self.observation;
        if s.episode_id == 0
            || s.frame_id == 0
            || s.captured_at_unix_ms == 0
            || s.sim_time_ns != 4_000_000_000
        {
            return Err(invalid(
                "pregrasp opening requires current200Tick image identity",
            ));
        }
        Ok(())
    }
}

pub struct MobileGraspOpening {
    goal: MobileOpenGoal,
    opening: MobileGripRelease,
    completed: bool,
}

impl MobileGraspOpening {
    pub fn new(
        goal: MobileOpenGoal,
        state: &G1Measurement,
        mut original: G1Command,
        calibration: &MobileGripCalibration,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        if state.source_tick != 200 {
            return Err(invalid(
                "opening cannot follow an initial close or transport",
            ));
        }
        original.navigation = [0.; 3];
        // Reuse the existing original-target horizontal opening and its2mm
        // per-palm/Tick and0.1rad joint bounds. No contacts or object poses.
        let opening = MobileGripRelease::new(
            MobileReleaseGoal {
                observation: goal.observation,
                target_palm_gap_m: 0.30,
                duration_ticks: 50,
            },
            state,
            original,
            calibration,
        )?;
        Ok(Self {
            goal,
            opening,
            completed: false,
        })
    }
    pub fn goal(&self) -> &MobileOpenGoal {
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
        if self.completed {
            return Err(invalid(
                "pregrasp opening is complete; fresh250Tick RGB required",
            ));
        }
        let mut step = self.opening.update(state, calibration)?;
        if step.settling_ticks != 0 || step.opening_ticks > 50 {
            return Err(invalid(
                "pregrasp opening cannot enter the release settling stage",
            ));
        }
        self.completed = step.opening_ticks == 50;
        step.completed = self.completed;
        Ok(step)
    }
}

fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_close_transport_and_reset_images_cannot_open() {
        let goal = MobileOpenGoal {
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 9,
                sim_time_ns: 4_000_000_000,
                captured_at_unix_ms: 123,
            },
        };
        assert!(goal.validate().is_ok());
        for ns in [0, 3_980_000_000, 5_000_000_000, 6_000_000_000] {
            let mut changed = goal.clone();
            changed.observation.sim_time_ns = ns;
            assert!(changed.validate().is_err());
        }
        let mut changed = goal;
        changed.observation.episode_id = 0;
        assert!(changed.validate().is_err());
    }
}
