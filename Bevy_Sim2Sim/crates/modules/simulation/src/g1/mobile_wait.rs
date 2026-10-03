//! Explicit finite standing while an observation/model operation is pending.
//! Uses the previously executed upper command and named self sensors only.
//! This is traditional control, never a repeated or restamped VLA action frame.

use super::{
    mobile_admission::MobileImageAdmission, mobile_grip::MobileGripCalibration,
    runner::G1Measurement,
};
use robot_minigame::{
    RobotError,
    g1::contract::{G1Command, JOINT_COUNT},
};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MobileWaitGoal {
    pub episode_id: u64,
    pub request_id: u64,
    /// Self-owner admission time; it is not an image acquisition stamp.
    pub execution_start_sim_ns: u64,
    pub duration_ticks: u32,
}

impl MobileWaitGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.episode_id == 0
            || self.request_id == 0
            || self.execution_start_sim_ns % 20_000_000 != 0
            || !(1..=200).contains(&self.duration_ticks)
            || self
                .execution_start_sim_ns
                .checked_add(u64::from(self.duration_ticks) * 20_000_000)
                .is_none()
        {
            return Err(invalid(
                "waiting requires a current self-owner identity and at most200Ticks",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileWaitStep {
    pub command: G1Command,
    pub waiting_ticks: u32,
    pub self_velocity_displacement_source_m: [f64; 3],
    pub self_upright_cosine: f32,
    pub self_speed_m_s: f32,
    pub stable_velocity_ticks: u32,
    pub observation_ready: bool,
    pub completed: bool,
}

pub struct MobileModelWaiting {
    goal: MobileWaitGoal,
    next_tick: u64,
    command: G1Command,
    ticks: u32,
    displacement: [f64; 3],
    previous_velocity: [f32; 3],
    stable_velocity_ticks: u32,
    self_history: Vec<(G1Measurement, [f64; 3])>,
}

impl MobileModelWaiting {
    pub fn new(
        goal: MobileWaitGoal,
        state: &G1Measurement,
        mut command: G1Command,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if goal.episode_id != state.episode_id || goal.execution_start_sim_ns != state.sim_time_ns {
            return Err(invalid("wait admission is not the current owner boundary"));
        }
        validate_self(state)?;
        command.navigation = [0.; 3];
        Ok(Self {
            goal,
            next_tick: state.source_tick,
            command,
            ticks: 0,
            displacement: [0.; 3],
            previous_velocity: state.root_velocity_source,
            stable_velocity_ticks: 0,
            self_history: vec![(state.clone(), [0.; 3])],
        })
    }

    pub fn goal(&self) -> &MobileWaitGoal {
        &self.goal
    }
    pub fn completed(&self) -> bool {
        self.ticks == self.goal.duration_ticks
    }

    /// An image keeps its acquisition clock. Only the current consecutive
    /// stationary interval may supply a replacement original action chunk.
    pub fn admits_image(&self, state: &G1Measurement, image_sim_ns: u64) -> bool {
        validate_self(state).is_ok()
            && state.episode_id == self.goal.episode_id
            && state.source_tick == self.next_tick
            && self.stable_velocity_ticks >= 20
            && state
                .root_velocity_source
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                <= 0.03
            && image_sim_ns % 20_000_000 == 0
            && image_sim_ns <= state.sim_time_ns
            && image_sim_ns
                >= self
                    .next_tick
                    .saturating_sub(u64::from(self.stable_velocity_ticks))
                    * 20_000_000
    }

    pub(super) fn admit_observed_skill(
        &self,
        observation: ObservationStamp,
        current: &G1Measurement,
        calibration: &MobileGripCalibration,
        maximum_image_age_ns: u64,
    ) -> Result<MobileImageAdmission, RobotError> {
        if !self.admits_image(current, observation.sim_time_ns) {
            return Err(invalid(
                "skill image is outside the current stationary self interval",
            ));
        }
        let current_odometry: [f64; 3] = std::array::from_fn(|i| {
            self.displacement[i]
                + 0.01 * f64::from(self.previous_velocity[i] + current.root_velocity_source[i])
        });
        let (image, image_odometry) = if observation.sim_time_ns == current.sim_time_ns {
            (current, &current_odometry)
        } else {
            let (image, odometry) = self
                .self_history
                .iter()
                .find(|(s, _)| s.sim_time_ns == observation.sim_time_ns)
                .ok_or_else(|| {
                    invalid("actual camera boundary is absent from bounded owner self history")
                })?;
            (image, odometry)
        };
        let displacement = std::array::from_fn(|i| current_odometry[i] - image_odometry[i]);
        let (palm_motion, rotation_change) =
            calibration.measured_palm_motion(image, current, displacement)?;
        MobileImageAdmission::from_stationary_wait(
            observation,
            current,
            displacement,
            palm_motion,
            rotation_change,
            maximum_image_age_ns,
        )
    }

    pub fn update(&mut self, state: &G1Measurement) -> Result<MobileWaitStep, RobotError> {
        if self.completed()
            || state.episode_id != self.goal.episode_id
            || state.source_tick != self.next_tick
            || state.sim_time_ns != self.next_tick * 20_000_000
        {
            return Err(invalid("foreign/repeated/completed waiting self state"));
        }
        let upright = validate_self(state)?;
        if self.ticks > 0 {
            for i in 0..3 {
                self.displacement[i] +=
                    0.01 * f64::from(self.previous_velocity[i] + state.root_velocity_source[i]);
            }
        }
        // A 0.3m/s predecessor needs a finite physical stopping interval.
        // This is a stop envelope, not permission to reuse an old image:
        // observation readiness has a separate low-speed consecutive gate.
        if self.displacement.iter().map(|x| x * x).sum::<f64>().sqrt() > 0.15 {
            return Err(invalid(
                "waiting stop odometry exceeds15cm; explicit pause required",
            ));
        }
        let speed = state
            .root_velocity_source
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        self.stable_velocity_ticks = if speed <= 0.03 {
            self.stable_velocity_ticks + 1
        } else {
            0
        };
        self.previous_velocity = state.root_velocity_source;
        if self
            .self_history
            .last()
            .is_none_or(|(s, _)| s.source_tick != state.source_tick)
        {
            if self.self_history.len() >= 201 {
                return Err(invalid("finite owner self-history budget exhausted"));
            }
            self.self_history.push((state.clone(), self.displacement));
        }
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or_else(|| invalid("waiting Tick overflow"))?;
        self.ticks += 1;
        Ok(MobileWaitStep {
            command: self.command.clone(),
            waiting_ticks: self.ticks,
            self_velocity_displacement_source_m: self.displacement,
            self_upright_cosine: upright,
            self_speed_m_s: speed,
            stable_velocity_ticks: self.stable_velocity_ticks,
            observation_ready: self.stable_velocity_ticks >= 20,
            completed: self.completed(),
        })
    }
}

fn validate_self(state: &G1Measurement) -> Result<f32, RobotError> {
    let q = state.root_rotation_wxyz;
    let norm = q.iter().map(|v| v * v).sum::<f32>();
    let upright = 1. - 2. * (q[1] * q[1] + q[2] * q[2]);
    if state.sim_time_ns
        != state
            .source_tick
            .checked_mul(20_000_000)
            .unwrap_or(u64::MAX)
        || state.joint_positions.len() != JOINT_COUNT
        || state.joint_velocities.len() != JOINT_COUNT
        || !state
            .joint_positions
            .iter()
            .chain(&state.joint_velocities)
            .chain(&state.root_velocity_source)
            .chain(&state.root_angular_velocity_body)
            .chain(&q)
            .all(|v| v.is_finite())
        || (norm - 1.).abs() > 0.00002
        || upright < 0.98
    {
        return Err(invalid(
            "waiting requires finite original self sensors and upright>=0.98",
        ));
    }
    Ok(upright)
}

fn invalid(message: impl Into<String>) -> RobotError {
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
            joint_positions: vec![0.; JOINT_COUNT],
            joint_velocities: vec![0.; JOINT_COUNT],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_velocity_source: [0.; 3],
            root_angular_velocity_body: [0.; 3],
        }
    }
    fn waiting() -> MobileModelWaiting {
        MobileModelWaiting::new(
            MobileWaitGoal {
                episode_id: 1,
                request_id: 1,
                execution_start_sim_ns: 1_000_000_000,
                duration_ticks: 50,
            },
            &state(50),
            G1Command {
                navigation: [0.1, 0., 0.],
                ..G1Command::default()
            },
        )
        .unwrap()
    }
    #[test]
    fn finite_wait_preserves_upper_command_and_has_its_own_clock() {
        let mut w = waiting();
        for tick in 50..100 {
            let step = w.update(&state(tick)).unwrap();
            assert_eq!(step.command.navigation, [0.; 3]);
            assert_eq!(
                step.command.upper_positions,
                G1Command::default().upper_positions
            );
            assert_eq!(step.completed, tick == 99);
            assert_eq!(step.observation_ready, tick >= 69);
        }
        assert!(w.update(&state(100)).is_err());
        let mut w = waiting();
        w.update(&state(50)).unwrap();
        assert!(w.update(&state(50)).is_err());
        let mut foreign = state(51);
        foreign.episode_id = 2;
        assert!(w.update(&foreign).is_err());
    }
    #[test]
    fn waiting_rejects_unstable_self_state_and_excess_motion() {
        let mut w = waiting();
        let mut s = state(50);
        s.root_rotation_wxyz = [0.; 4];
        assert!(w.update(&s).is_err());
        let mut w = waiting();
        let mut s = state(50);
        s.root_velocity_source[0] = f32::NAN;
        assert!(w.update(&s).is_err());
        let mut w = waiting();
        for tick in 50..88 {
            let mut s = state(tick);
            s.root_velocity_source[0] = 0.2;
            assert!(w.update(&s).is_ok());
        }
        let mut s = state(88);
        s.root_velocity_source[0] = 0.2;
        assert!(w.update(&s).is_err());
    }

    #[test]
    fn deadline_completion_does_not_admit_an_unsettled_image() {
        let mut w = waiting();
        for tick in 50..100 {
            let mut s = state(tick);
            if tick % 20 == 19 {
                s.root_velocity_source[0] = 0.04;
            }
            let step = w.update(&s).unwrap();
            assert!(!step.observation_ready);
            assert_eq!(step.completed, tick == 99);
        }
        let mut goal = w.goal().clone();
        goal.duration_ticks = 201;
        assert!(goal.validate().is_err());
        goal.duration_ticks = 100;
        assert!(MobileModelWaiting::new(goal, &state(51), G1Command::default()).is_err());
    }

    #[test]
    fn replacement_image_must_belong_to_current_stationary_interval() {
        let mut w = waiting();
        for tick in 50..70 {
            w.update(&state(tick)).unwrap();
        }
        assert!(w.admits_image(&state(70), 65 * 20_000_000));
        assert!(!w.admits_image(&state(70), 49 * 20_000_000));
        assert!(!w.admits_image(&state(70), 71 * 20_000_000));
        let mut moved = state(70);
        moved.root_velocity_source[0] = 0.04;
        assert!(!w.admits_image(&moved, 65 * 20_000_000));
        w.update(&moved).unwrap();
        assert!(!w.admits_image(&state(71), 65 * 20_000_000));
    }

    #[test]
    #[ignore = "frozen actual RGB/current-self admission check;0world integrations/0model calls"]
    fn saved_mobile_image_admission_self_diagnostic() -> Result<(), RobotError> {
        use super::super::mobile_navigation::{MobileCarryGoal, MobileCarryNavigator};
        use robot_minigame::g1::definition::G1Definition;
        use robot_minigame::g1::policy::bound_bytes;
        use std::{fs, path::Path};
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct SavedSelf {
            episode_id: u64,
            source_tick: u64,
            sim_time_ns: u64,
            joint_positions: Vec<f32>,
            joint_velocities: Vec<f32>,
            root_rotation_wxyz: [f32; 4],
            root_velocity_source: [f32; 3],
            root_angular_velocity_body: [f32; 3],
        }
        impl SavedSelf {
            fn measured(self) -> G1Measurement {
                G1Measurement {
                    episode_id: self.episode_id,
                    source_tick: self.source_tick,
                    sim_time_ns: self.sim_time_ns,
                    joint_positions: self.joint_positions,
                    joint_velocities: self.joint_velocities,
                    root_rotation_wxyz: self.root_rotation_wxyz,
                    root_velocity_source: self.root_velocity_source,
                    root_angular_velocity_body: self.root_angular_velocity_body,
                }
            }
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Case {
            goal: MobileWaitGoal,
            command: G1Command,
            states: Vec<SavedSelf>,
            current: SavedSelf,
            observation: ObservationStamp,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fixture {
            schema: String,
            definition: String,
            definition_sha256: String,
            cases: Vec<Case>,
        }
        let path = std::env::var("G1_MOBILE_SELF_ADMISSION_FIXTURE")
            .map_err(|e| invalid(e.to_string()))?;
        let expected = std::env::var("G1_MOBILE_SELF_ADMISSION_FIXTURE_SHA256")
            .map_err(|e| invalid(e.to_string()))?;
        let bytes = bound_bytes(Path::new(&path), &expected)?;
        let fixture: Fixture =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        if fixture.schema != "g1_saved_actual_image_self_admission_v1" || fixture.cases.len() != 4 {
            return Err(invalid(
                "self admission requires three actual replacement images and the final actual grip image",
            ));
        }
        let definition =
            G1Definition::load(Path::new(&fixture.definition), &fixture.definition_sha256)?;
        let calibration = MobileGripCalibration::new(&definition)?;
        let mut results = Vec::new();
        let result = (|| -> Result<(), RobotError> {
            for c in fixture.cases {
                let states: Vec<_> = c.states.into_iter().map(SavedSelf::measured).collect();
                let current = c.current.measured();
                let first = states
                    .first()
                    .ok_or_else(|| invalid("empty actual self history"))?;
                let mut waiting = MobileModelWaiting::new(c.goal, first, c.command)?;
                for s in &states {
                    waiting.update(s)?;
                }
                let admission = waiting.admit_observed_skill(
                    c.observation,
                    &current,
                    &calibration,
                    1_000_000_000,
                );
                let admission = match admission {
                    Ok(admission) => admission,
                    Err(error) => {
                        results.push(serde_json::json!({"admitted":false,"observation":c.observation,"execution_start_sim_ns":current.sim_time_ns,"error":error.to_string()}));
                        continue;
                    }
                };
                let goal = MobileCarryGoal {
                    observation: c.observation,
                    heading_yaw_source_rad: 0.,
                    relative_distance_m: 0.1,
                };
                if c.observation.sim_time_ns != current.sim_time_ns {
                    assert!(MobileCarryNavigator::new(goal.clone(), &current).is_err());
                }
                let mut navigation =
                    MobileCarryNavigator::new_with_admission(goal.clone(), &current, &admission)?;
                navigation.update(&current)?;
                assert_eq!(navigation.goal().observation, c.observation);
                let mut next = current.clone();
                next.source_tick += 1;
                next.sim_time_ns += 20_000_000;
                assert!(MobileCarryNavigator::new_with_admission(goal, &next, &admission).is_err());
                let mut old = c.observation;
                old.sim_time_ns = 0;
                assert!(
                    waiting
                        .admit_observed_skill(old, &current, &calibration, 1_000_000_000)
                        .is_err()
                );
                if c.observation.sim_time_ns != current.sim_time_ns {
                    let mut moved = current.clone();
                    moved.joint_positions[15] += 0.1;
                    assert!(
                        waiting
                            .admit_observed_skill(
                                c.observation,
                                &moved,
                                &calibration,
                                1_000_000_000
                            )
                            .is_err()
                    );
                }
                results.push(serde_json::json!({"admitted":true,"admission":admission}));
            }
            assert!(results.iter().any(|r| r["admitted"] == true));
            assert!(results.iter().any(|r| r["admitted"] == false));
            Ok(())
        })();
        let output =
            std::env::var("G1_MOBILE_SELF_ADMISSION_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(&mut file,&serde_json::json!({"schema":"g1_saved_actual_self_admission_result_v1","qualified":false,"actual_integrations":0,"actual_torque_updates":0,"fresh_vla_calls":0,"new_rgb_frames":0,"fixture_sha256":expected,"admissions":results,"complete":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)})).map_err(|e|invalid(e.to_string()))?;
        result
    }
}
