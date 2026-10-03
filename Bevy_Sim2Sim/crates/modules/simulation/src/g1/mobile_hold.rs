//! Bounded real standing using the original command and named self velocity.
//! No object pose/contact truth, camera inference or additional integration.

use super::mobile_admission::MobileImageAdmission;

use super::runner::G1Measurement;
use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileHoldGoal {
    pub observation: ObservationStamp,
}
impl MobileHoldGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let s = &self.observation;
        if s.episode_id == 0
            || s.frame_id == 0
            || s.captured_at_unix_ms == 0
            || s.sim_time_ns % 20_000_000 != 0
        {
            return Err(invalid("hold requires a current native camera identity"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileHoldStep {
    pub command: G1Command,
    pub holding_ticks: u32,
    pub stable_velocity_ticks: u32,
    pub self_speed_m_s: f32,
    pub completed: bool,
}

pub struct MobileGripHolding {
    goal: MobileHoldGoal,
    next_tick: u64,
    command: G1Command,
    holding_ticks: u32,
    stable_velocity_ticks: u32,
    minimum_ticks: u32,
    completed: bool,
}
impl MobileGripHolding {
    pub fn new(
        goal: MobileHoldGoal,
        state: &G1Measurement,
        command: G1Command,
    ) -> Result<Self, RobotError> {
        let admission = MobileImageAdmission::at_current_boundary(goal.observation, state)?;
        Self::new_with_admission(goal, state, command, &admission)
    }
    pub(super) fn new_with_admission(
        goal: MobileHoldGoal,
        state: &G1Measurement,
        command: G1Command,
        admission: &MobileImageAdmission,
    ) -> Result<Self, RobotError> {
        Self::bounded(goal, state, command, admission, 100)
    }
    /// Fixed one-second contact-settling phase before the first transport turn.
    /// Placement retains its separate two-second minimum.
    pub(super) fn new_grip_settle_with_admission(
        goal: MobileHoldGoal,
        state: &G1Measurement,
        command: G1Command,
        admission: &MobileImageAdmission,
    ) -> Result<Self, RobotError> {
        Self::bounded(goal, state, command, admission, 50)
    }
    fn bounded(
        goal: MobileHoldGoal,
        state: &G1Measurement,
        command: G1Command,
        admission: &MobileImageAdmission,
        minimum_ticks: u32,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if !admission.matches(goal.observation, state) || command.navigation != [0.; 3] {
            return Err(invalid(
                "standing hold requires a fresh stationary skill boundary",
            ));
        }
        Ok(Self {
            goal,
            next_tick: state.source_tick,
            command,
            holding_ticks: 0,
            stable_velocity_ticks: 0,
            minimum_ticks,
            completed: false,
        })
    }
    pub fn goal(&self) -> &MobileHoldGoal {
        &self.goal
    }
    pub fn command(&self) -> &G1Command {
        &self.command
    }
    pub fn completed(&self) -> bool {
        self.completed
    }
    pub fn update(&mut self, state: &G1Measurement) -> Result<MobileHoldStep, RobotError> {
        if self.completed
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != self.next_tick * 20_000_000
            || !state.root_velocity_source.iter().all(|v| v.is_finite())
        {
            return Err(invalid(
                "foreign/repeated/nonfinite/completed standing hold state",
            ));
        }
        let speed = state
            .root_velocity_source
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        self.holding_ticks += 1;
        self.stable_velocity_ticks = if speed <= 0.03 {
            self.stable_velocity_ticks + 1
        } else {
            0
        };
        self.completed =
            self.holding_ticks >= self.minimum_ticks && self.stable_velocity_ticks >= 20;
        if self.holding_ticks >= 250 && !self.completed {
            return Err(invalid(
                "bounded standing hold did not attain20stable self-velocity samples; pause required",
            ));
        }
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or_else(|| invalid("hold Tick overflow"))?;
        Ok(MobileHoldStep {
            command: self.command.clone(),
            holding_ticks: self.holding_ticks,
            stable_velocity_ticks: self.stable_velocity_ticks,
            self_speed_m_s: speed,
            completed: self.completed,
        })
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(tick: u64, speed: f32) -> G1Measurement {
        G1Measurement {
            episode_id: 1,
            source_tick: tick,
            sim_time_ns: tick * 20_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [speed, 0., 0.],
        }
    }
    fn hold() -> MobileGripHolding {
        MobileGripHolding::new(
            MobileHoldGoal {
                observation: ObservationStamp {
                    episode_id: 1,
                    frame_id: 1,
                    sim_time_ns: 0,
                    captured_at_unix_ms: 1,
                },
            },
            &state(0, 0.),
            G1Command {
                navigation: [0.; 3],
                pelvis_height: 0.75,
                torso_rpy: [0.; 3],
                upper_positions: [0.; 28],
            },
        )
        .unwrap()
    }
    #[test]
    fn actual_hold_budget_requires_stability_and_rejects_repeat_and_reset() {
        let mut h = hold();
        for tick in 0..99 {
            assert!(!h.update(&state(tick, 0.)).unwrap().completed);
        }
        assert!(h.update(&state(99, 0.)).unwrap().completed);
        assert!(h.update(&state(100, 0.)).is_err());
        let mut h = hold();
        h.update(&state(0, 0.)).unwrap();
        assert!(h.update(&state(0, 0.)).is_err());
        let mut foreign = state(1, 0.);
        foreign.episode_id = 2;
        assert!(h.update(&foreign).is_err());
    }
    #[test]
    fn transient_stability_cannot_accumulate_and_waiting_is_finite() {
        let mut h = hold();
        for tick in 0..249 {
            assert!(
                !h.update(&state(tick, if tick % 20 == 19 { 0.04 } else { 0. }))
                    .unwrap()
                    .completed
            );
        }
        assert!(h.update(&state(249, 0.04)).is_err());
    }
    #[test]
    fn grip_settle_has_a_distinct_minimum_and_requires_consecutive_self_stability() {
        let placement = hold();
        let admission =
            MobileImageAdmission::at_current_boundary(placement.goal.observation, &state(0, 0.))
                .unwrap();
        let mut h = MobileGripHolding::new_grip_settle_with_admission(
            placement.goal,
            &state(0, 0.),
            placement.command,
            &admission,
        )
        .unwrap();
        for tick in 0..50 {
            let speed = if tick == 40 { 0.04 } else { 0. };
            assert!(!h.update(&state(tick, speed)).unwrap().completed);
        }
        for tick in 50..60 {
            assert!(!h.update(&state(tick, 0.)).unwrap().completed);
        }
        assert!(h.update(&state(60, 0.)).unwrap().completed);
        assert!(h.update(&state(61, 0.)).is_err());
    }
}
