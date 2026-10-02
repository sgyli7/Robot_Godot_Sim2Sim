//! Finite traditional navigation from robot quaternion and measured velocity.
//! Targets come from an explicit caller/observation; no task-world truth is read.

use robot_minigame::RobotError;
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

use super::runner::G1Measurement;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileCarryGoal {
    pub observation: ObservationStamp,
    pub heading_yaw_source_rad: f32,
    pub relative_distance_m: f32,
}

impl MobileCarryGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.observation.episode_id == 0
            || self.observation.frame_id == 0
            || self.observation.sim_time_ns % 20_000_000 != 0
            || self.observation.captured_at_unix_ms == 0
            || !self.heading_yaw_source_rad.is_finite()
            || self.heading_yaw_source_rad.abs() > std::f32::consts::PI
            || !self.relative_distance_m.is_finite()
            || !(0.1..=2.5).contains(&self.relative_distance_m)
        {
            return Err(invalid("invalid bounded mobile carry observation/goal"));
        }
        Ok(())
    }
}

/// A bounded public-map search heading, followed by a stationary RGB boundary.
/// It carries no object position, walking distance or task-world observation.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileScanGoal {
    pub observation: ObservationStamp,
    pub heading_yaw_source_rad: f32,
}

impl MobileScanGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        MobileCarryGoal {
            observation: self.observation.clone(),
            heading_yaw_source_rad: self.heading_yaw_source_rad,
            relative_distance_m: 0.1,
        }
        .validate()
    }
}

/// Shares the measured finite turn controller; never enters its walking phase.
pub struct MobileScanNavigator {
    goal: MobileScanGoal,
    motion: MobileCarryNavigator,
}

impl MobileScanNavigator {
    pub fn new(goal: MobileScanGoal, state: &G1Measurement) -> Result<Self, RobotError> {
        goal.validate()?;
        let mut motion = MobileCarryNavigator::new(
            MobileCarryGoal {
                observation: goal.observation.clone(),
                heading_yaw_source_rad: goal.heading_yaw_source_rad,
                relative_distance_m: 0.1,
            },
            state,
        )?;
        motion.scan_only = true;
        Ok(Self { goal, motion })
    }
    pub fn goal(&self) -> &MobileScanGoal {
        &self.goal
    }
    pub fn completed(&self) -> bool {
        self.motion.completed()
    }
    pub fn update(&mut self, state: &G1Measurement) -> Result<MobileNavigationStep, RobotError> {
        self.motion.update(state)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MobileNavigationPhase {
    Turn,
    Walk,
    Stop,
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileNavigationStep {
    pub phase: MobileNavigationPhase,
    pub navigation: [f32; 3],
    pub own_velocity_odometry_xy_m: [f32; 2],
    pub completed: bool,
}

#[derive(Clone)]
pub struct MobileCarryNavigator {
    goal: MobileCarryGoal,
    next_state_tick: u64,
    phase: MobileNavigationPhase,
    phase_ticks: u64,
    heading_ready_ticks: u64,
    settling: bool,
    odometry: [f32; 2],
    completed: bool,
    scan_only: bool,
    forward_window_m: f32,
    forward_window_ticks: u32,
}

impl MobileCarryNavigator {
    pub fn new(goal: MobileCarryGoal, state: &G1Measurement) -> Result<Self, RobotError> {
        goal.validate()?;
        if goal.observation.episode_id != state.episode_id
            || goal.observation.sim_time_ns != state.sim_time_ns
        {
            return Err(invalid(
                "carry handoff requires observation from the current native boundary",
            ));
        }
        Ok(Self {
            goal,
            next_state_tick: state.source_tick,
            phase: MobileNavigationPhase::Turn,
            phase_ticks: 0,
            heading_ready_ticks: 0,
            settling: false,
            odometry: [0.; 2],
            completed: false,
            scan_only: false,
            forward_window_m: 0.,
            forward_window_ticks: 0,
        })
    }

    pub fn goal(&self) -> &MobileCarryGoal {
        &self.goal
    }
    pub fn completed(&self) -> bool {
        self.completed
    }

    /// One call per native pre-integration boundary. Repeats, resets, skipped
    /// states and a completed goal are errors, never another odometry update.
    pub fn update(&mut self, state: &G1Measurement) -> Result<MobileNavigationStep, RobotError> {
        if self.completed
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != self.next_state_tick
            || state.source_tick.checked_mul(20_000_000) != Some(state.sim_time_ns)
            || !state
                .root_rotation_wxyz
                .iter()
                .chain(state.root_velocity_source.iter())
                .all(|v| v.is_finite())
            || (state.root_rotation_wxyz.iter().map(|q| q * q).sum::<f32>() - 1.).abs() > 2e-5
        {
            return Err(invalid(
                "foreign/repeated/nonfinite/incoherent carry self state",
            ));
        }
        let [w, x, y, z] = state.root_rotation_wxyz;
        let heading = (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z));
        let delta = self.goal.heading_yaw_source_rad - heading;
        let error = delta.sin().atan2(delta.cos());
        self.phase_ticks += 1;
        let phase = self.phase;
        let navigation = match phase {
            MobileNavigationPhase::Turn => {
                self.heading_ready_ticks = if error.abs() < 0.06 {
                    self.heading_ready_ticks + 1
                } else {
                    0
                };
                if self.heading_ready_ticks >= 20 {
                    self.phase = if self.scan_only {
                        MobileNavigationPhase::Stop
                    } else {
                        MobileNavigationPhase::Walk
                    };
                    self.phase_ticks = 0;
                } else if self.phase_ticks >= 750 {
                    return Err(invalid("proprioceptive turn deadline missed"));
                }
                if self.settling {
                    if error.abs() >= 0.06 {
                        self.settling = false;
                    }
                } else if error.abs() <= 0.03 {
                    self.settling = true;
                }
                let yaw = if self.settling {
                    0.
                } else {
                    error.signum() * (0.8 * error.abs()).clamp(0.06, 0.4)
                };
                [0., 0., yaw]
            }
            MobileNavigationPhase::Walk => {
                for k in 0..2 {
                    self.odometry[k] += state.root_velocity_source[k] * 0.02;
                }
                self.forward_window_m += 0.02
                    * (state.root_velocity_source[0] * self.goal.heading_yaw_source_rad.cos()
                        + state.root_velocity_source[1] * self.goal.heading_yaw_source_rad.sin());
                self.forward_window_ticks += 1;
                if self.forward_window_ticks == 50 {
                    // One initial gait-start window is allowed. Thereafter,
                    // lateral drift cannot hide failure to advance toward goal.
                    if self.phase_ticks >= 100 && self.forward_window_m < 0.03 {
                        return Err(invalid(
                            "carry blocked: less than3cm forward progress in1second; pause/reset required",
                        ));
                    }
                    self.forward_window_m = 0.;
                    self.forward_window_ticks = 0;
                }
                if self.odometry[0].hypot(self.odometry[1]) >= self.goal.relative_distance_m + 0.05
                {
                    self.phase = MobileNavigationPhase::Stop;
                    self.phase_ticks = 0;
                    [0.; 3]
                } else if self.phase_ticks >= 1000 {
                    return Err(invalid("proprioceptive carry walk deadline missed"));
                } else {
                    [0.3, 0., (0.8 * error).clamp(-0.25, 0.25)]
                }
            }
            MobileNavigationPhase::Stop => {
                self.completed = self.phase_ticks >= 100;
                [0.; 3]
            }
        };
        self.next_state_tick = self
            .next_state_tick
            .checked_add(1)
            .ok_or_else(|| invalid("carry tick overflow"))?;
        Ok(MobileNavigationStep {
            phase,
            navigation,
            own_velocity_odometry_xy_m: self.odometry,
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
    fn state(tick: u64) -> G1Measurement {
        G1Measurement {
            episode_id: 7,
            source_tick: tick,
            sim_time_ns: tick * 20_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [0.; 3],
        }
    }
    fn goal() -> MobileCarryGoal {
        MobileCarryGoal {
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 5,
                sim_time_ns: 4_000_000_000,
                captured_at_unix_ms: 1,
            },
            heading_yaw_source_rad: std::f32::consts::FRAC_PI_2,
            relative_distance_m: 2.,
        }
    }
    #[test]
    fn carry_does_not_recount_old_episode_or_repeated_velocity() {
        let s = state(200);
        let mut nav = MobileCarryNavigator::new(goal(), &s).unwrap();
        nav.update(&s).unwrap();
        assert!(nav.clone().update(&s).is_err());
        assert!(nav.clone().update(&state(202)).is_err());
        let mut foreign = state(201);
        foreign.episode_id += 1;
        assert!(nav.clone().update(&foreign).is_err());
        let mut bad = state(201);
        bad.root_velocity_source[0] = f32::NAN;
        assert!(nav.clone().update(&bad).is_err());
        bad = state(201);
        bad.root_rotation_wxyz = [0.; 4];
        assert!(nav.update(&bad).is_err());
    }
    #[test]
    fn carry_turn_has_a_finite_deadline() {
        let mut nav = MobileCarryNavigator::new(goal(), &state(200)).unwrap();
        for tick in 200..949 {
            assert!(nav.update(&state(tick)).is_ok());
        }
        assert!(nav.update(&state(949)).is_err());
    }
    #[test]
    fn carry_goal_must_use_current_boundary_and_bounded_distance() {
        let mut g = goal();
        g.relative_distance_m = 2.6;
        assert!(MobileCarryNavigator::new(g, &state(200)).is_err());
        assert!(MobileCarryNavigator::new(goal(), &state(201)).is_err());
        let mut g = goal();
        g.heading_yaw_source_rad = f32::NAN;
        assert!(g.validate().is_err());
    }

    #[test]
    fn scan_stops_at_a_real_observation_boundary_without_walking() {
        let mut nav = MobileScanNavigator::new(
            MobileScanGoal {
                observation: goal().observation,
                heading_yaw_source_rad: 0.,
            },
            &state(200),
        )
        .unwrap();
        for tick in 200..320 {
            let step = nav.update(&state(tick)).unwrap();
            assert_ne!(step.phase, MobileNavigationPhase::Walk);
            assert_eq!(step.navigation, [0.; 3]);
            assert_eq!(step.completed, tick == 319);
        }
        assert!(nav.update(&state(320)).is_err());
    }
    #[test]
    fn blocked_walk_detects_no_forward_progress_even_with_lateral_drift() {
        let mut g = goal();
        g.heading_yaw_source_rad = 0.;
        let mut nav = MobileCarryNavigator::new(g, &state(200)).unwrap();
        for tick in 200..220 {
            nav.update(&state(tick)).unwrap();
        }
        for tick in 220..319 {
            let mut s = state(tick);
            s.root_velocity_source = [0., 0.3, 0.];
            assert!(nav.update(&s).is_ok());
        }
        let mut s = state(319);
        s.root_velocity_source = [0., 0.3, 0.];
        assert!(
            nav.update(&s)
                .unwrap_err()
                .to_string()
                .contains("carry blocked")
        );
    }
    #[test]
    fn progress_guard_accepts_a_gait_with_alternating_zero_speed() {
        let mut g = goal();
        g.heading_yaw_source_rad = 0.;
        let mut nav = MobileCarryNavigator::new(g, &state(200)).unwrap();
        for tick in 200..220 {
            nav.update(&state(tick)).unwrap();
        }
        for tick in 220..420 {
            let mut s = state(tick);
            s.root_velocity_source = [if tick % 2 == 0 { 0.6 } else { 0. }, 0., 0.];
            assert!(nav.update(&s).is_ok());
        }
    }
}
