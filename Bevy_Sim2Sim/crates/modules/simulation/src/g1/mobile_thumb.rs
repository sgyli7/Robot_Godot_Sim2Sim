//! Declared finite stationary thumb preparation, followed by a new camera view.
//! Drives original finger motors only; no object/contact truth or pose setters.

use super::{mobile_admission::MobileImageAdmission, runner::G1Measurement};
use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

pub const LEFT_THUMB_CLEARANCE_TARGETS: [f32; 2] = [-0.040_685_26, -0.184_286_03];
/// Conservative subset of the four original thumb joint limits, in radians.
pub const MAXIMUM_BOUNDED_THUMB_ANGLE_RAD: f32 = 0.25;
const THUMB_JOINTS: [usize; 4] = [26, 27, 40, 41];

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileThumbGoal {
    pub observation: ObservationStamp,
    /// Left thumb 0/1, right thumb 0/1; absent preserves the original left pose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounded_thumb_targets_rad: Option<[f32; 4]>,
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
        if self.bounded_thumb_targets_rad.is_some_and(|targets| {
            targets
                .iter()
                .any(|q| !q.is_finite() || q.abs() > MAXIMUM_BOUNDED_THUMB_ANGLE_RAD)
        }) {
            return Err(invalid(
                "thumb targets exceed the finite four-joint envelope",
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
    start: [f32; 4],
    targets: [f32; 4],
    joint_count: usize,
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
        let joint_count = if goal.bounded_thumb_targets_rad.is_some() {
            4
        } else {
            2
        };
        let start = THUMB_JOINTS.map(|joint| command.upper_positions[joint - 15]);
        let targets = goal.bounded_thumb_targets_rad.unwrap_or([
            LEFT_THUMB_CLEARANCE_TARGETS[0],
            LEFT_THUMB_CLEARANCE_TARGETS[1],
            0.,
            0.,
        ]);
        if command.navigation != [0.; 3]
            || start[..joint_count].iter().any(|q| q.abs() > 0.01)
            || THUMB_JOINTS[..joint_count]
                .iter()
                .any(|joint| state.joint_positions[*joint].abs() > 0.02)
        {
            return Err(invalid(
                "thumb preparation needs stationary original near-zero fingers",
            ));
        }
        Ok(Self {
            goal,
            command,
            start,
            targets,
            joint_count,
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
        let error = (0..self.joint_count)
            .map(|slot| (state.joint_positions[THUMB_JOINTS[slot]] - self.targets[slot]).abs())
            .fold(0., f32::max);
        if self.ticks == 99 && error > 0.02 {
            return Err(invalid("thumb motor tracking failed; pause before release"));
        }
        self.ticks += 1;
        let fraction = (self.ticks as f32 / 50.).min(1.);
        for (slot, target) in self.targets[..self.joint_count].iter().enumerate() {
            self.command.upper_positions[THUMB_JOINTS[slot] - 15] =
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
            bounded_thumb_targets_rad: None,
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
    #[test]
    fn bounded_both_hand_goal_drives_only_four_motors_and_checks_right_tracking() {
        let targets = [0.136_114_98, -0.25, 0.062_134_184, 0.181_736_25];
        let mut value = serde_json::to_value(goal()).unwrap();
        value["bounded_thumb_targets_rad"] = serde_json::json!(targets);
        let bounded: MobileThumbGoal = serde_json::from_value(value).unwrap();
        let command = G1Command::default();
        let mut preparation =
            MobileThumbPreparation::new(bounded, &state(0), command.clone()).unwrap();
        let first = preparation.update(&state(0)).unwrap();
        for slot in 0..28 {
            if let Some(i) = [11, 12, 25, 26].iter().position(|index| *index == slot) {
                assert!((first.command.upper_positions[slot] - targets[i] / 50.).abs() < 1e-8);
            } else {
                assert_eq!(
                    first.command.upper_positions[slot],
                    command.upper_positions[slot]
                );
            }
        }
        for tick in 1..99 {
            preparation.update(&state(tick)).unwrap();
        }
        let mut measured = state(99);
        measured.joint_positions[26..28].copy_from_slice(&targets[..2]);
        assert!(preparation.update(&measured).is_err());
        measured.joint_positions[40..42].copy_from_slice(&targets[2..]);
        let completed = preparation.update(&measured).unwrap();
        assert!(completed.completed && completed.fresh_release_observation_required);
        assert!(preparation.update(&state(100)).is_err());
    }
    #[test]
    fn bounded_targets_require_original_range_and_near_zero_right_fingers() {
        assert!(
            serde_json::to_value(goal())
                .unwrap()
                .get("bounded_thumb_targets_rad")
                .is_none()
        );
        for unsafe_angle in [0.251, -0.251, f32::NAN, f32::INFINITY] {
            let mut g = goal();
            g.bounded_thumb_targets_rad = Some([0., 0., unsafe_angle, 0.]);
            assert!(g.validate().is_err());
        }
        let mut g = goal();
        g.bounded_thumb_targets_rad = Some([0., -0.25, 0.06, 0.18]);
        let mut s = state(0);
        s.joint_positions[40] = 0.021;
        assert!(MobileThumbPreparation::new(g.clone(), &s, G1Command::default()).is_err());
        let mut command = G1Command::default();
        command.upper_positions[26] = 0.011;
        assert!(MobileThumbPreparation::new(g, &state(0), command.clone()).is_err());
        // The previously admitted profile still touches only the left fingers.
        let mut legacy = MobileThumbPreparation::new(goal(), &s, command.clone()).unwrap();
        assert_eq!(
            legacy.update(&s).unwrap().command.upper_positions[26],
            command.upper_positions[26]
        );
    }
}
