//! Bounded reacquisition after measured closed-grip RGB, on the original owner.
//! Each phase requires a new image at its actual boundary. No prop/contact input.

use robot_minigame::{RobotError, g1::contract::G1Command};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

use super::{
    mobile_admission::MobileImageAdmission,
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    mobile_hold::{MobileGripHolding, MobileHoldGoal},
    mobile_release::{MobileGripRelease, MobileReleaseGoal},
    runner::G1Measurement,
};

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MobileRegraspPhase {
    Open,
    Settle,
    Insert,
    Close,
    Hold,
}

impl MobileRegraspPhase {
    pub fn offset_ticks(self) -> u64 {
        match self {
            Self::Open => 0,
            Self::Settle => 50,
            Self::Insert => 150,
            Self::Close => 200,
            Self::Hold => 250,
        }
    }
    pub fn duration_ticks(self) -> u32 {
        match self {
            Self::Hold | Self::Settle => 100,
            Self::Open | Self::Insert | Self::Close => 50,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileRegraspGoal {
    pub attempt: u32,
    pub phase: MobileRegraspPhase,
    pub closed_observation: ObservationStamp,
    pub observation: ObservationStamp,
    pub common_offset_root_source_m: [f64; 3],
}

impl MobileRegraspGoal {
    pub fn start_tick(&self) -> u64 {
        300 + u64::from(self.attempt.saturating_sub(1)) * 350 + self.phase.offset_ticks()
    }
    pub fn end_tick(&self) -> u64 {
        self.start_tick() + u64::from(self.phase.duration_ticks())
    }
    pub fn validate(&self) -> Result<(), RobotError> {
        let c = self.closed_observation;
        let s = self.observation;
        let offset = self.common_offset_root_source_m;
        let length = offset.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(1..=2).contains(&self.attempt)
            || c.episode_id == 0
            || c.frame_id == 0
            || c.captured_at_unix_ms == 0
            || c.sim_time_ns != (300 + u64::from(self.attempt.saturating_sub(1)) * 350) * 20_000_000
            || s.episode_id != c.episode_id
            || s.sim_time_ns != self.start_tick() * 20_000_000
            || (self.phase == MobileRegraspPhase::Open && s != c)
            || (self.phase != MobileRegraspPhase::Open
                && (s.frame_id <= c.frame_id || s.captured_at_unix_ms <= c.captured_at_unix_ms))
            || !offset.iter().all(|x| x.is_finite())
            || !(0.000_001..=0.05).contains(&length)
        {
            return Err(invalid(
                "regrasp requires its current closed300/650 RGB, fresh phase image, <=5cm and at most two attempts",
            ));
        }
        Ok(())
    }
    pub fn follows(&self, previous: &Self) -> bool {
        if self.validate().is_err()
            || previous.validate().is_err()
            || self.start_tick() != previous.end_tick()
            || self.observation.episode_id != previous.observation.episode_id
            || self.observation.frame_id <= previous.observation.frame_id
            || self.observation.captured_at_unix_ms <= previous.observation.captured_at_unix_ms
        {
            return false;
        }
        if previous.phase == MobileRegraspPhase::Hold {
            self.phase == MobileRegraspPhase::Open && self.attempt == previous.attempt + 1
        } else {
            self.attempt == previous.attempt
                && self.closed_observation == previous.closed_observation
                && (previous.phase == MobileRegraspPhase::Settle
                    || self.common_offset_root_source_m == previous.common_offset_root_source_m)
                && matches!(
                    (previous.phase, self.phase),
                    (MobileRegraspPhase::Open, MobileRegraspPhase::Settle)
                        | (MobileRegraspPhase::Settle, MobileRegraspPhase::Insert)
                        | (MobileRegraspPhase::Insert, MobileRegraspPhase::Close)
                        | (MobileRegraspPhase::Close, MobileRegraspPhase::Hold)
                )
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileRegraspStep {
    pub command: G1Command,
    pub phase_ticks: u32,
    pub stable_velocity_ticks: u32,
    pub correction: Option<MobileGripReceipt>,
    pub completed: bool,
    pub fresh_next_boundary_rgb_required: bool,
}

pub struct MobileRegrasp {
    goal: MobileRegraspGoal,
    command: G1Command,
    next_tick: u64,
    ticks: u32,
    opening: Option<MobileGripRelease>,
    holding: Option<MobileGripHolding>,
}

impl MobileRegrasp {
    pub fn new(
        goal: MobileRegraspGoal,
        state: &G1Measurement,
        mut command: G1Command,
        calibration: &MobileGripCalibration,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        MobileImageAdmission::at_current_boundary(goal.observation, state)?;
        command.navigation = [0.; 3];
        command.validate()?;
        let opening = if goal.phase == MobileRegraspPhase::Open {
            Some(MobileGripRelease::new(
                MobileReleaseGoal {
                    observation: goal.observation,
                    target_palm_gap_m: 0.30,
                    duration_ticks: 50,
                },
                state,
                command.clone(),
                calibration,
            )?)
        } else {
            None
        };
        let holding = if matches!(
            goal.phase,
            MobileRegraspPhase::Hold | MobileRegraspPhase::Settle
        ) {
            Some(MobileGripHolding::new(
                MobileHoldGoal {
                    observation: goal.observation,
                },
                state,
                command.clone(),
            )?)
        } else {
            None
        };
        Ok(Self {
            goal,
            command,
            next_tick: state.source_tick,
            ticks: 0,
            opening,
            holding,
        })
    }
    pub fn goal(&self) -> &MobileRegraspGoal {
        &self.goal
    }
    pub fn command(&self) -> &G1Command {
        &self.command
    }
    pub fn completed(&self) -> bool {
        self.ticks == self.goal.phase.duration_ticks()
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        calibration: &MobileGripCalibration,
    ) -> Result<MobileRegraspStep, RobotError> {
        if self.completed()
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != self.next_tick * 20_000_000
        {
            return Err(invalid(
                "regrasp received repeated/reset/nonsequential/completed owner state",
            ));
        }
        let mut stable_velocity_ticks = 0;
        let correction = match self.goal.phase {
            MobileRegraspPhase::Open => {
                let step = self.opening.as_mut().unwrap().update(state, calibration)?;
                if step.settling_ticks != 0 || step.opening_ticks > 50 {
                    return Err(invalid("reopening cannot enter release settling"));
                }
                self.command = step.command;
                step.correction
            }
            MobileRegraspPhase::Insert => {
                let offset = self.goal.common_offset_root_source_m.map(|x| x / 50.);
                let corrected = calibration.translate(state, &self.command, offset)?;
                self.command = corrected.command;
                Some(corrected.receipt)
            }
            MobileRegraspPhase::Close => {
                let corrected =
                    calibration.approach_source_gap(state, &self.command, 50 - self.ticks)?;
                self.command = corrected.command;
                Some(corrected.receipt)
            }
            MobileRegraspPhase::Hold | MobileRegraspPhase::Settle => {
                let step = self.holding.as_mut().unwrap().update(state)?;
                if self.ticks == 99 && !step.completed {
                    return Err(invalid(
                        "regrasp original100Tick hold lacks consecutive self stability",
                    ));
                }
                stable_velocity_ticks = step.stable_velocity_ticks;
                self.command = step.command;
                None
            }
        };
        self.ticks += 1;
        self.next_tick += 1;
        Ok(MobileRegraspStep {
            command: self.command.clone(),
            phase_ticks: self.ticks,
            stable_velocity_ticks,
            correction,
            completed: self.completed(),
            fresh_next_boundary_rgb_required: true,
        })
    }
}
fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn goal(attempt: u32, phase: MobileRegraspPhase) -> MobileRegraspGoal {
        let tick = 300 + u64::from(attempt - 1) * 350;
        let closed = ObservationStamp {
            episode_id: 7,
            frame_id: tick,
            captured_at_unix_ms: tick,
            sim_time_ns: tick * 20_000_000,
        };
        let phase_tick = tick + phase.offset_ticks();
        MobileRegraspGoal {
            attempt,
            phase,
            closed_observation: closed,
            observation: ObservationStamp {
                frame_id: phase_tick,
                captured_at_unix_ms: phase_tick,
                sim_time_ns: phase_tick * 20_000_000,
                ..closed
            },
            common_offset_root_source_m: [0.042, 0., 0.],
        }
    }
    #[test]
    fn stale_reset_skipped_and_third_regrasp_cannot_reopen() {
        let open = goal(1, MobileRegraspPhase::Open);
        let settle = goal(1, MobileRegraspPhase::Settle);
        let insert = goal(1, MobileRegraspPhase::Insert);
        assert!(open.validate().is_ok() && settle.follows(&open) && insert.follows(&settle));
        let mut replanned = insert.clone();
        replanned.common_offset_root_source_m = [0.024, 0., 0.];
        assert!(replanned.follows(&settle));
        assert!(!goal(1, MobileRegraspPhase::Close).follows(&replanned));

        assert!(!goal(1, MobileRegraspPhase::Close).follows(&open));
        assert!(goal(2, MobileRegraspPhase::Open).follows(&goal(1, MobileRegraspPhase::Hold)));
        assert!(goal(3, MobileRegraspPhase::Open).validate().is_err());
        let mut foreign = insert.clone();
        foreign.observation.episode_id += 1;
        assert!(!foreign.follows(&open));
        let mut stale = insert;
        stale.observation = open.observation;
        assert!(!stale.follows(&open));
        let mut changed = goal(1, MobileRegraspPhase::Insert);
        changed.common_offset_root_source_m = [0.041, 0., 0.];
        assert!(!changed.follows(&open));
        for offset in [[0.; 3], [0.050001, 0., 0.], [f64::NAN, 0., 0.]] {
            let mut g = open.clone();
            g.common_offset_root_source_m = offset;
            assert!(g.validate().is_err());
        }
    }
}
