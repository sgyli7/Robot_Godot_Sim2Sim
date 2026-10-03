//! Finite traditional observation withdrawal after the typed static placement.
//! Fixed 10cm source+Y motion with open-hand self sensing; no object/contact truth.
use super::{
    runner::G1Measurement,
    static_transfer::{StaticCartesianReceipt, StaticLeftPalmKinematics, self_rotation},
};
use rapier3d::na::Vector3;
use robot_minigame::{RobotError, g1::agile::AgileCommand};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticObservationWithdrawalGoal {
    pub episode_id: u64,
    pub start_sim_time_ns: u64,
}
impl StaticObservationWithdrawalGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.episode_id == 0 || self.start_sim_time_ns != 715 * 20_000_000 {
            return Err(invalid(
                "observation withdrawal requires exact715Tick placement boundary",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct StaticObservationWithdrawalStep {
    pub goal: StaticObservationWithdrawalGoal,
    pub withdrawal_ticks: u32,
    pub phase: &'static str,
    pub completed: bool,
    pub command: AgileCommand,
    pub classical_geometry: Option<StaticCartesianReceipt>,
    pub world_or_contact_truth_input: bool,
    pub task_qualified: bool,
}
pub struct StaticObservationWithdrawal {
    goal: StaticObservationWithdrawalGoal,
    ticks: u32,
    command: AgileCommand,
}
impl StaticObservationWithdrawal {
    pub fn new(
        goal: StaticObservationWithdrawalGoal,
        state: &G1Measurement,
        command: AgileCommand,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        self_rotation(state)?;
        if state.episode_id != goal.episode_id
            || state.source_tick != 715
            || state.sim_time_ns != goal.start_sim_time_ns
            || state.joint_positions.len() != 43
            || state.joint_velocities.len() != 43
            || !state
                .joint_positions
                .iter()
                .chain(state.joint_velocities.iter())
                .all(|v| v.is_finite())
            || state.joint_positions[22..29].iter().any(|v| v.abs() > 0.05)
            || state.joint_velocities[22..29]
                .iter()
                .any(|v| v.abs() > 0.02)
            || command.upper_positions[7..14].iter().any(|v| *v != 0.)
            || command.navigation.iter().any(|v| v.abs() > 0.01)
        {
            return Err(invalid(
                "observation withdrawal requires matching715Tick open-hand self state",
            ));
        }
        Ok(Self {
            goal,
            ticks: 0,
            command,
        })
    }
    pub fn goal(&self) -> &StaticObservationWithdrawalGoal {
        &self.goal
    }
    pub(super) fn update(
        &mut self,
        state: &G1Measurement,
        fk: &StaticLeftPalmKinematics,
    ) -> Result<StaticObservationWithdrawalStep, RobotError> {
        if self.ticks >= 125
            || state.episode_id != self.goal.episode_id
            || state.source_tick != 715 + u64::from(self.ticks)
            || state.sim_time_ns != state.source_tick * 20_000_000
        {
            return Err(invalid(
                "observation withdrawal repeated/foreign/excessiveTick",
            ));
        }
        let (phase, receipt) = if self.ticks < 100 {
            let (command, receipt) = fk.translate(
                state,
                &self.command,
                (self_rotation(state)?.inverse() * Vector3::new(0., 0.001, 0.)).into(),
            )?;
            self.command = command;
            ("withdraw", Some(receipt))
        } else {
            ("hold", None)
        };
        self.ticks += 1;
        Ok(StaticObservationWithdrawalStep {
            goal: self.goal.clone(),
            withdrawal_ticks: self.ticks,
            phase,
            completed: self.ticks == 125,
            command: self.command.clone(),
            classical_geometry: receipt,
            world_or_contact_truth_input: false,
            task_qualified: false,
        })
    }
}
fn invalid(s: impl Into<String>) -> RobotError {
    RobotError::Contract(s.into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_closed_moving_nonfinite_or_reset_hand_before_withdrawal() {
        let goal = StaticObservationWithdrawalGoal {
            episode_id: 7,
            start_sim_time_ns: 715 * 20_000_000,
        };
        let state = G1Measurement {
            episode_id: 7,
            source_tick: 715,
            sim_time_ns: 715 * 20_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [0.; 3],
        };
        let command = AgileCommand {
            navigation: [0.; 3],
            pelvis_height: 0.72,
            upper_positions: [0.; 28],
        };
        assert!(StaticObservationWithdrawal::new(goal.clone(), &state, command.clone()).is_ok());
        let mut bad = state.clone();
        bad.joint_positions[22] = 0.1;
        assert!(StaticObservationWithdrawal::new(goal.clone(), &bad, command.clone()).is_err());
        let mut bad = state.clone();
        bad.joint_velocities[28] = 0.1;
        assert!(StaticObservationWithdrawal::new(goal.clone(), &bad, command.clone()).is_err());
        let mut bad = state.clone();
        bad.joint_positions[0] = f32::NAN;
        assert!(StaticObservationWithdrawal::new(goal.clone(), &bad, command.clone()).is_err());
        let mut bad = state.clone();
        bad.episode_id += 1;
        assert!(StaticObservationWithdrawal::new(goal.clone(), &bad, command.clone()).is_err());
        let mut bad = command.clone();
        bad.upper_positions[7] = 1.;
        assert!(StaticObservationWithdrawal::new(goal, &state, bad).is_err());
    }
}
