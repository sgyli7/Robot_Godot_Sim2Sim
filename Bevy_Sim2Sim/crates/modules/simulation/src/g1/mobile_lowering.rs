//! Finite traditional Cartesian lowering using original self sensors/FK only.
//! Keeps palm orientations, gap and fingers; contact truth is acceptance-only.

use rapier3d::na::{Quaternion, UnitQuaternion, Vector3};
use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

use super::{
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    runner::G1Measurement,
};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileLowerGoal {
    pub observation: ObservationStamp,
    pub distance_m: f32,
    pub duration_ticks: u32,
}

impl MobileLowerGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.observation.episode_id == 0
            || self.observation.frame_id == 0
            || self.observation.captured_at_unix_ms == 0
            || self.observation.sim_time_ns % 20_000_000 != 0
            || !self.distance_m.is_finite()
            || !(0.01..=0.25).contains(&self.distance_m)
            || !(50..=200).contains(&self.duration_ticks)
            || self.distance_m / (self.duration_ticks as f32 * 0.02) > 0.1
        {
            return Err(invalid(
                "lowering requires finite current observation,1–25cm,50–200Ticks and<=0.1m/s",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileLowerStep {
    pub commanded_drop_m: f32,
    pub commanded_additional_retraction_m: f64,
    pub cartesian_increment_root_source_m: [f64; 3],
    pub lowering_ticks: u32,
    pub settling_ticks: u32,
    pub completed: bool,
    pub correction: Option<MobileGripReceipt>,
    pub command: G1Command,
}

pub struct MobileGripLowering {
    goal: MobileLowerGoal,
    next_tick: u64,
    lowering_ticks: u32,
    settling_ticks: u32,
    completed: bool,
    command: G1Command,
    retraction_m: f64,
}

impl MobileGripLowering {
    pub fn new(
        goal: MobileLowerGoal,
        state: &G1Measurement,
        command: G1Command,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if goal.observation.episode_id != state.episode_id
            || goal.observation.sim_time_ns != state.sim_time_ns
            || command.navigation != [0.; 3]
        {
            return Err(invalid(
                "lowering requires a fresh stationary completed skill boundary",
            ));
        }
        Ok(Self {
            goal,
            next_tick: state.source_tick,
            lowering_ticks: 0,
            settling_ticks: 0,
            completed: false,
            command,
            retraction_m: 0.,
        })
    }
    pub fn goal(&self) -> &MobileLowerGoal {
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
    ) -> Result<MobileLowerStep, RobotError> {
        if self.completed
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != self.next_tick * 20_000_000
            || !state.root_rotation_wxyz.iter().all(|q| q.is_finite())
            || (state.root_rotation_wxyz.iter().map(|q| q * q).sum::<f32>() - 1.).abs() > 2e-5
        {
            return Err(invalid("foreign/repeated/nonfinite lower self state"));
        }
        let mut increment = [0.; 3];
        let correction = if self.lowering_ticks < self.goal.duration_ticks {
            let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
            let root_rotation = UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z));
            let offset = root_rotation.inverse()
                * Vector3::new(
                    0.,
                    0.,
                    -f64::from(self.goal.distance_m) / f64::from(self.goal.duration_ticks),
                );
            increment = calibration.reach_preserving_offset(state, &self.command, offset.into())?;
            self.retraction_m += offset.x - increment[0];
            let corrected = calibration.translate(state, &self.command, increment)?;
            self.command = corrected.command;
            self.lowering_ticks += 1;
            Some(corrected.receipt)
        } else {
            self.settling_ticks += 1;
            self.completed = self.settling_ticks >= 100;
            None
        };
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or_else(|| invalid("lowering Tick overflow"))?;
        Ok(MobileLowerStep {
            commanded_additional_retraction_m: self.retraction_m,
            cartesian_increment_root_source_m: increment,
            commanded_drop_m: self.goal.distance_m * self.lowering_ticks as f32
                / self.goal.duration_ticks as f32,
            lowering_ticks: self.lowering_ticks,
            settling_ticks: self.settling_ticks,
            completed: self.completed,
            correction,
            command: self.command.clone(),
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
    fn lowering_rejects_excessive_distance_speed_and_foreign_stamp() {
        let mut goal = MobileLowerGoal {
            observation: ObservationStamp {
                episode_id: 1,
                frame_id: 1,
                sim_time_ns: 4_000_000_000,
                captured_at_unix_ms: 1,
            },
            distance_m: 0.18,
            duration_ticks: 150,
        };
        assert!(goal.validate().is_ok());
        goal.duration_ticks = 50;
        assert!(goal.validate().is_err());
        goal.duration_ticks = 150;
        goal.distance_m = 0.26;
        assert!(goal.validate().is_err());
        goal.distance_m = f32::NAN;
        assert!(goal.validate().is_err());
        goal.distance_m = 0.18;
        goal.observation.sim_time_ns += 1;
        assert!(goal.validate().is_err());
    }
}
