//! Declared finite stationary thumb preparation, followed by a new camera view.
//! Drives original finger motors only; no object/contact truth or pose setters.

use super::{mobile_admission::MobileImageAdmission, runner::G1Measurement};
use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

pub const LEFT_THUMB_CLEARANCE_TARGETS: [f32; 2] = [-0.040_685_26, -0.184_286_03];

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileThumbGoal {
    pub observation: ObservationStamp,
}
impl MobileThumbGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let s = self.observation;
        if s.episode_id == 0
            || s.frame_id == 0
            || s.captured_at_unix_ms == 0
            || s.sim_time_ns % 20_000_000 != 0
        {
            return Err(invalid(
                "thumb preparation requires a current native image identity",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileThumbStep {
    pub command: G1Command,
    pub preparation_ticks: u32,
    pub maximum_measured_tracking_error_rad: f32,
    pub completed: bool,
    pub fresh_release_observation_required: bool,
}

pub struct MobileThumbPreparation {
    goal: MobileThumbGoal,
    command: G1Command,
    start: [f32; 2],
    ticks: u32,
    next_tick: u64,
    completed: bool,
}
impl MobileThumbPreparation {
    pub fn new(
        goal: MobileThumbGoal,
        state: &G1Measurement,
        command: G1Command,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        MobileImageAdmission::at_current_boundary(goal.observation, state)?;
        validate_state(state)?;
        let start = [command.upper_positions[11], command.upper_positions[12]];
        if command.navigation != [0.; 3]
            || start.iter().any(|q| q.abs() > 0.01)
            || state.joint_positions[26..28].iter().any(|q| q.abs() > 0.02)
        {
            return Err(invalid(
                "thumb preparation needs stationary original near-zero fingers",
            ));
        }
        Ok(Self {
            goal,
            command,
            start,
            ticks: 0,
            next_tick: state.source_tick,
            completed: false,
        })
    }
    pub fn goal(&self) -> &MobileThumbGoal {
        &self.goal
    }
    pub fn command(&self) -> &G1Command {
        &self.command
    }
    pub fn completed(&self) -> bool {
        self.completed
    }
    pub fn update(&mut self, state: &G1Measurement) -> Result<MobileThumbStep, RobotError> {
        validate_state(state)?;
        if self.completed
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
        {
            return Err(invalid(
                "foreign/repeated/completed thumb preparation state",
            ));
        }
        let error = (0..2)
            .map(|slot| {
                (state.joint_positions[26 + slot] - LEFT_THUMB_CLEARANCE_TARGETS[slot]).abs()
            })
            .fold(0., f32::max);
        if self.ticks == 99 && error > 0.02 {
            return Err(invalid("thumb motor tracking failed; pause before release"));
        }
        self.ticks += 1;
        let fraction = (self.ticks as f32 / 50.).min(1.);
        for (slot, target) in LEFT_THUMB_CLEARANCE_TARGETS.iter().enumerate() {
            self.command.upper_positions[11 + slot] =
                self.start[slot] + (target - self.start[slot]) * fraction;
        }
        self.completed = self.ticks == 100;
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or_else(|| invalid("thumb Tick overflow"))?;
        Ok(MobileThumbStep {
            command: self.command.clone(),
            preparation_ticks: self.ticks,
            maximum_measured_tracking_error_rad: error,
            completed: self.completed,
            fresh_release_observation_required: true,
        })
    }
}

fn validate_state(state: &G1Measurement) -> Result<(), RobotError> {
    if state.source_tick.checked_mul(20_000_000) != Some(state.sim_time_ns)
        || state.joint_positions.len() != 43
        || !state.joint_positions.iter().all(|q| q.is_finite())
    {
        return Err(invalid("invalid thumb self measurement"));
    }
    Ok(())
}
fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(tick: u64) -> G1Measurement {
        G1Measurement {
            episode_id: 1,
            source_tick: tick,
            sim_time_ns: tick * 20_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [0.; 3],
        }
    }
    fn goal() -> MobileThumbGoal {
        MobileThumbGoal {
            observation: ObservationStamp {
                episode_id: 1,
                frame_id: 1,
                sim_time_ns: 0,
                captured_at_unix_ms: 1,
            },
        }
    }
    #[test]
    fn refuses_foreign_stamp_moving_command_and_nonfinite_self_state() {
        let mut g = goal();
        g.observation.episode_id = 2;
        assert!(MobileThumbPreparation::new(g, &state(0), G1Command::default()).is_err());
        let mut c = G1Command::default();
        c.navigation[0] = 0.1;
        assert!(MobileThumbPreparation::new(goal(), &state(0), c).is_err());
        let mut s = state(0);
        s.joint_positions[26] = f32::NAN;
        assert!(MobileThumbPreparation::new(goal(), &s, G1Command::default()).is_err());
    }
    #[test]
    fn changes_only_two_original_fingers_and_refuses_untracked_completion() {
        let c = G1Command::default();
        let mut p = MobileThumbPreparation::new(goal(), &state(0), c.clone()).unwrap();
        let first = p.update(&state(0)).unwrap();
        assert!(p.update(&state(0)).is_err());
        for i in 0..28 {
            if i != 11 && i != 12 {
                assert_eq!(first.command.upper_positions[i], c.upper_positions[i]);
            }
        }
        for tick in 1..99 {
            p.update(&state(tick)).unwrap();
        }
        assert!(p.update(&state(99)).is_err());
        assert!(!p.completed());
    }
    #[test]
    fn tracked_completion_still_requires_another_release_image() {
        let mut p = MobileThumbPreparation::new(goal(), &state(0), G1Command::default()).unwrap();
        for tick in 0..100 {
            let mut s = state(tick);
            s.joint_positions[26..28].copy_from_slice(&LEFT_THUMB_CLEARANCE_TARGETS);
            let step = p.update(&s).unwrap();
            assert!(step.fresh_release_observation_required);
            assert_eq!(step.completed, tick == 99);
        }
        assert!(p.completed());
        assert!(p.update(&state(100)).is_err());
    }
}
