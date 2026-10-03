//! Finite classical grasp template retargeted from actual RGB and self sensors.
//! The reference is a disclosed observed trajectory, not new VLA inference.
//! No task-object pose, contacts, simulator world or coordinate setters enter.

use super::{
    runner::G1Measurement,
    static_transfer::{StaticLeftPalmKinematics, StaticPoseReceipt, self_rotation},
};
use rapier3d::na::Vector3;
use robot_minigame::{RobotError, g1::agile::AgileCommand};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticGraspIncrement {
    pub translation_source_m: [f64; 3],
    pub rotation_source_rad: [f64; 3],
    pub left_hand_targets: [f32; 7],
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticObservedGraspGoal {
    pub observation: ObservationStamp,
    pub image_sha256: String,
    pub reference_rgb_self_template_sha256: String,
    pub increments: Vec<StaticGraspIncrement>,
}
impl StaticObservedGraspGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let s = self.observation;
        let hash = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        };
        if s.episode_id == 0
            || s.frame_id == 0
            || s.captured_at_unix_ms == 0
            || s.sim_time_ns != 100 * 20_000_000
            || !hash(&self.image_sha256)
            || !hash(&self.reference_rgb_self_template_sha256)
            || self.increments.is_empty()
            || self.increments.len() > 400
            || self.increments.iter().any(|p| {
                !p.translation_source_m
                    .iter()
                    .chain(p.rotation_source_rad.iter())
                    .all(|v| v.is_finite())
                    || !p.left_hand_targets.iter().all(|v| v.is_finite())
                    || Vector3::from(p.translation_source_m).norm() > 0.001600001
                    || Vector3::from(p.rotation_source_rad).norm() > 0.010000001
            })
        {
            return Err(invalid(
                "observed grasp requires bounded RGB100/self geometry",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StaticObservedGraspStep {
    pub grasp_ticks: u32,
    pub completed: bool,
    pub classical_geometry: StaticPoseReceipt,
    pub maximum_hand_target_change_rad: f32,
    pub command: AgileCommand,
    pub original_vla_output: bool,
    pub world_or_contact_truth_input: bool,
    pub task_qualified: bool,
}

pub struct StaticObservedGrasp {
    goal: StaticObservedGraspGoal,
    ticks: usize,
    command: AgileCommand,
}
impl StaticObservedGrasp {
    pub fn new(
        goal: StaticObservedGraspGoal,
        state: &G1Measurement,
        command: AgileCommand,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if state.source_tick != 100
            || state.sim_time_ns != goal.observation.sim_time_ns
            || state.episode_id != goal.observation.episode_id
            || command.navigation.iter().any(|v| v.abs() > 0.01)
            || state.joint_positions.len() != 43
            || state.joint_positions[22..29]
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 0.05)
        {
            return Err(invalid(
                "observed grasp requires its actual open-hand100Tick self state",
            ));
        }
        Ok(Self {
            goal,
            ticks: 0,
            command,
        })
    }
    pub fn goal(&self) -> &StaticObservedGraspGoal {
        &self.goal
    }
    pub(super) fn completed_command(&self) -> Option<AgileCommand> {
        (self.ticks == self.goal.increments.len()).then(|| self.command.clone())
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        kinematics: &StaticLeftPalmKinematics,
    ) -> Result<StaticObservedGraspStep, RobotError> {
        if state.episode_id != self.goal.observation.episode_id
            || state.source_tick != 100 + self.ticks as u64
            || state.sim_time_ns != state.source_tick * 20_000_000
            || self.ticks >= self.goal.increments.len()
        {
            return Err(invalid(
                "observed grasp rejected stale, reset or exhausted state",
            ));
        }
        let p = &self.goal.increments[self.ticks];
        let rotation = self_rotation(state)?.inverse();
        let (mut command, receipt) = kinematics.move_by(
            state,
            &self.command,
            (rotation * Vector3::from(p.translation_source_m)).into(),
            (rotation * Vector3::from(p.rotation_source_rad)).into(),
        )?;
        let hand_step = command.upper_positions[7..14]
            .iter()
            .zip(p.left_hand_targets.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        // Same 1.2rad range / 50 ticks as the existing finite opening, now
        // applied explicitly to classical closure; actuator parameters stay fixed.
        if hand_step > 0.0240001 {
            return Err(invalid("observed grasp hand step exceeds0.024rad"));
        }
        command.upper_positions[7..14].copy_from_slice(&p.left_hand_targets);
        kinematics.commanded_left_palm(state, &command)?;
        self.command = command.clone();
        self.ticks += 1;
        Ok(StaticObservedGraspStep {
            grasp_ticks: self.ticks as u32,
            completed: self.ticks == self.goal.increments.len(),
            classical_geometry: receipt,
            maximum_hand_target_change_rad: hand_step,
            command,
            original_vla_output: false,
            world_or_contact_truth_input: false,
            task_qualified: false,
        })
    }
}
fn invalid(s: impl std::fmt::Display) -> RobotError {
    RobotError::Contract(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::g1::{definition::G1Definition, policy::bound_bytes};
    use std::{fs, path::Path};

    fn goal() -> StaticObservedGraspGoal {
        StaticObservedGraspGoal {
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 2,
                sim_time_ns: 100 * 20_000_000,
                captured_at_unix_ms: 1,
            },
            image_sha256: "a".repeat(64),
            reference_rgb_self_template_sha256: "b".repeat(64),
            increments: vec![StaticGraspIncrement {
                translation_source_m: [0.; 3],
                rotation_source_rad: [0.; 3],
                left_hand_targets: [0.; 7],
            }],
        }
    }
    #[test]
    fn observed_grasp_rejects_old_episode_closed_hand_and_foreign_truth() {
        let g = goal();
        let mut s = G1Measurement {
            episode_id: 7,
            source_tick: 100,
            sim_time_ns: 100 * 20_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [0.; 3],
        };
        assert!(
            StaticObservedGrasp::new(
                g.clone(),
                &s,
                AgileCommand {
                    navigation: [0.; 3],
                    pelvis_height: 0.75,
                    upper_positions: [0.; 28]
                }
            )
            .is_ok()
        );
        s.episode_id = 8;
        assert!(
            StaticObservedGrasp::new(
                g.clone(),
                &s,
                AgileCommand {
                    navigation: [0.; 3],
                    pelvis_height: 0.75,
                    upper_positions: [0.; 28]
                }
            )
            .is_err()
        );
        s.episode_id = 7;
        s.source_tick = 101;
        assert!(
            StaticObservedGrasp::new(
                g.clone(),
                &s,
                AgileCommand {
                    navigation: [0.; 3],
                    pelvis_height: 0.75,
                    upper_positions: [0.; 28]
                }
            )
            .is_err()
        );
        s.source_tick = 100;
        s.joint_positions[23] = 0.6;
        assert!(
            StaticObservedGrasp::new(
                g.clone(),
                &s,
                AgileCommand {
                    navigation: [0.; 3],
                    pelvis_height: 0.75,
                    upper_positions: [0.; 28]
                }
            )
            .is_err()
        );
        let mut v = serde_json::to_value(g).unwrap();
        v["apple_world_position"] = serde_json::json!([0, 0, 0]);
        assert!(serde_json::from_value::<StaticObservedGraspGoal>(v).is_err());
    }
    #[test]
    fn observed_grasp_rejects_unbounded_rotation_translation_and_horizon() {
        let mut g = goal();
        g.increments[0].rotation_source_rad = [0., 0., 0.02];
        assert!(g.validate().is_err());
        g.increments[0].rotation_source_rad = [0.; 3];
        g.increments[0].translation_source_m = [0.002, 0., 0.];
        assert!(g.validate().is_err());
        g.increments[0].translation_source_m = [0.; 3];
        g.increments[0].left_hand_targets[0] = f32::NAN;
        assert!(g.validate().is_err());
        g.increments[0].left_hand_targets = [0.; 7];
        g.increments.resize(401, g.increments[0].clone());
        assert!(g.validate().is_err());
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        definition: String,
        definition_sha256: String,
        state: SelfState,
        command: AgileCommand,
        goal: StaticObservedGraspGoal,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SelfState {
        episode_id: u64,
        source_tick: u64,
        sim_time_ns: u64,
        joint_positions: Vec<f32>,
        joint_velocities: Vec<f32>,
        root_rotation_wxyz: [f32; 4],
        root_angular_velocity_body: [f32; 3],
        root_velocity_source: [f32; 3],
    }
    #[test]
    #[ignore = "explicit bound actual RGB/self grasp preflight;0physics/model calls"]
    fn preflight_actual_observed_grasp() -> Result<(), RobotError> {
        let path = std::env::var("G1_OBSERVED_GRASP_FIXTURE").map_err(invalid)?;
        let hash = std::env::var("G1_OBSERVED_GRASP_FIXTURE_SHA256").map_err(invalid)?;
        let bytes = bound_bytes(Path::new(&path), &hash)?;
        if bytes.len() > 256 * 1024 {
            return Err(invalid("grasp fixture exceeds bound"));
        }
        let fixture: Fixture = serde_json::from_slice(&bytes).map_err(invalid)?;
        let definition =
            G1Definition::load(Path::new(&fixture.definition), &fixture.definition_sha256)?;
        let kinematics = StaticLeftPalmKinematics::new(&definition)?;
        let original = fixture.command.clone();
        let s = fixture.state;
        let mut state = G1Measurement {
            episode_id: s.episode_id,
            source_tick: s.source_tick,
            sim_time_ns: s.sim_time_ns,
            joint_positions: s.joint_positions,
            joint_velocities: s.joint_velocities,
            root_rotation_wxyz: s.root_rotation_wxyz,
            root_angular_velocity_body: s.root_angular_velocity_body,
            root_velocity_source: s.root_velocity_source,
        };
        assert!(
            kinematics
                .move_by(&state, &original, [0.; 3], [0., 0., 0.011])
                .is_err()
        );
        assert!(
            kinematics
                .move_by(&state, &original, [0.002, 0., 0.], [0.; 3])
                .is_err()
        );
        let mut grasp = StaticObservedGrasp::new(fixture.goal, &state, fixture.command)?;
        let mut steps = Vec::new();
        let mut failure = None;
        for index in 0..grasp.goal.increments.len() {
            match grasp.update(&state, &kinematics) {
                Ok(step) => {
                    assert_eq!(
                        step.command.upper_positions[14..],
                        original.upper_positions[14..]
                    );
                    assert_eq!(step.command.navigation, original.navigation);
                    assert_eq!(step.command.pelvis_height, original.pelvis_height);
                    steps.push(step);
                    state.source_tick += 1;
                    state.sim_time_ns += 20_000_000;
                }
                Err(e) => {
                    failure = Some(serde_json::json!({"index":index,"error":e.to_string()}));
                    break;
                }
            }
        }
        let report = serde_json::json!({"schema":"g1_actual_observed_grasp_joint_preflight_v1",
            "physics_integrations":0,"model_calls":0,"geometry_points":steps.len(),"failure":failure,
            "world_or_contact_truth_input":false,"task_qualified":false,"steps":steps});
        let output = std::env::var("G1_OBSERVED_GRASP_OUTPUT").map_err(invalid)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(invalid)?;
        std::io::Write::write_all(
            &mut file,
            &serde_json::to_vec_pretty(&report).map_err(invalid)?,
        )
        .map_err(invalid)?;
        Ok(())
    }
}
