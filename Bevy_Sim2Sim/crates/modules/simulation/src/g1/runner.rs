//! A single owner performs observation, real inference, explicit PD and one 50Hz
//! integration. Rendering receives immutable completed frames only.

use super::assembly::{self, G1Assembly};
use crate::{SimulationWorld, StepConfiguration, WorldCounts};
use rapier3d::prelude::*;
use robot_minigame::{
    RobotError,
    basis::engine_to_source_vector,
    g1::{
        actuator,
        contract::{G1Command, LOWER_HOME},
        definition::{G1BodyFrame, G1Definition, SourcePose},
        policy::{HomiePolicy, HomieResult},
    },
};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct G1RunnerConfig {
    pub episode_id: u64,
    pub definition: PathBuf,
    pub definition_sha256: String,
    pub ort_library: PathBuf,
    pub ort_sha256: String,
    pub stand_model: PathBuf,
    pub walk_model: PathBuf,
    /// Source Z-up free pelvis pose. This is not the external USD scene root.
    pub root_pose: SourcePose,
    /// Explicit diagnostic materials; do not claim PhysX cooking/material parity.
    pub robot_contact_friction: f32,
    pub floor_contact_friction: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct G1Step {
    pub frame: G1BodyFrame,
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub applied_torques: Vec<f32>,
    pub inference: HomieResult,
    pub root_position_source: [f32; 3],
    pub root_velocity_source: [f32; 3],
    pub root_upright_cosine: f32,
    pub active_contact_pairs: usize,
    pub integration_count: u64,
    pub torque_update_count: u64,
    pub step_configuration: StepConfiguration,
}

pub struct G1Runner {
    episode_id: u64,
    simulation: SimulationWorld,
    assembly: G1Assembly,
    policy: HomiePolicy,
    halted: bool,
}

impl G1Runner {
    pub fn load(config: &G1RunnerConfig) -> Result<Self, RobotError> {
        let definition = G1Definition::load(&config.definition, &config.definition_sha256)?;
        if !config.floor_contact_friction.is_finite() || config.floor_contact_friction < 0. {
            return Err(error("invalid G1 floor material"));
        }
        let policy = HomiePolicy::load(
            &config.ort_library,
            &config.ort_sha256,
            &config.stand_model,
            &config.walk_model,
        )?;
        let mut simulation = SimulationWorld::with_game_frequency(50).map_err(error)?;
        // Broad floor is the only fixture; all robot/environment contacts share
        // this exact world. No kinematic anchor, tether or invisible support.
        let floor = simulation
            .world
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(Vector::new(0., -0.25, 0.)));
        simulation.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(20., 0.25, 20.)
                .friction(config.floor_contact_friction)
                .restitution(0.)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_2,
                    Group::ALL,
                    InteractionTestMode::And,
                )),
            floor,
            &mut simulation.world.bodies,
        );
        let mut initial = [0.; 43];
        initial[..15].copy_from_slice(&LOWER_HOME);
        let assembly = assembly::build(
            &mut simulation.world,
            &definition,
            &config.root_pose,
            &initial,
            config.robot_contact_friction,
        )?;
        Ok(Self {
            episode_id: config.episode_id,
            simulation,
            assembly,
            policy,
            halted: false,
        })
    }

    pub fn initial_frame(&self) -> Result<G1BodyFrame, RobotError> {
        self.completed_frame(&self.simulation.snapshot())
    }
    fn completed_frame(&self, snapshot: &crate::StepSnapshot) -> Result<G1BodyFrame, RobotError> {
        let mut frame = self.assembly.frame(snapshot)?;
        frame.episode_id = self.episode_id;
        Ok(frame)
    }

    pub fn counts(&self) -> WorldCounts {
        self.simulation.counts()
    }

    /// All errors halt this candidate; a caller cannot skip a failed inference
    /// and continue integrating with a stale target or a synthetic observation.
    pub fn step(&mut self, command: &G1Command) -> Result<G1Step, RobotError> {
        if self.halted {
            return Err(error("G1 runner halted after a failed boundary"));
        }
        let result = self.step_inner(command);
        if result.is_err() {
            self.halted = true;
        }
        result
    }

    fn step_inner(&mut self, command: &G1Command) -> Result<G1Step, RobotError> {
        let state = self.assembly.state(&self.simulation.world)?;
        let inference = self.policy.infer(&state, command)?;
        let applied = actuator::torques(&inference.targets, &state.positions, &state.velocities)?;
        let torques = self.assembly.torques(&self.simulation.world, &applied)?;
        let snapshot = self.simulation.step_with_torques(&torques).map_err(error)?;
        if snapshot.integration_count != inference.inference_count
            || snapshot.torque_update_count != inference.inference_count
        {
            return Err(error("G1 policy/torque/physics count mismatch"));
        }
        let completed = self.assembly.state(&self.simulation.world)?;
        let root = &self.simulation.world.bodies[self.assembly.root_handle()];
        Ok(G1Step {
            frame: self.completed_frame(&snapshot)?,
            joint_positions: completed.positions.to_vec(),
            joint_velocities: completed.velocities.to_vec(),
            applied_torques: applied.to_vec(),
            inference,
            root_position_source: engine_to_source_vector(root.translation().to_array()),
            root_velocity_source: engine_to_source_vector(root.linvel().to_array()),
            root_upright_cosine: (*root.rotation() * Vector::Y).y,
            active_contact_pairs: snapshot.active_contact_pair_count,
            integration_count: snapshot.integration_count,
            torque_update_count: snapshot.torque_update_count,
            step_configuration: self.simulation.configuration(),
        })
    }
}
fn error(e: impl std::fmt::Display) -> RobotError {
    RobotError::Contract(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};

    fn config() -> G1RunnerConfig {
        let models = PathBuf::from(env::var("G1_MODEL_DIR").unwrap());
        G1RunnerConfig {
            episode_id: 0,
            definition: models.join("g1_physics.json"),
            definition_sha256: env::var("G1_DEFINITION_SHA256").unwrap(),
            ort_library: env::var("G1_ORT").unwrap().into(),
            ort_sha256: env::var("G1_ORT_SHA256").unwrap(),
            stand_model: models.join("stand.onnx"),
            walk_model: models.join("walk.onnx"),
            root_pose: SourcePose {
                position: [0., 0., 0.78],
                rotation_wxyz: [1., 0., 0., 0.],
            },
            robot_contact_friction: 0.5,
            floor_contact_friction: 1.,
        }
    }

    #[test]
    #[ignore = "requires frozen G1_MODEL_DIR, G1_DEFINITION_SHA256, G1_ORT, G1_ORT_SHA256, G1_T0_OUTPUT"]
    fn real_homie_t0_diagnostic() {
        let config = config();
        let mut runner = G1Runner::load(&config).unwrap();
        let counts = runner.counts();
        assert_eq!(
            (
                counts.bodies,
                counts.colliders,
                counts.multibody_joint_handles,
                counts.impulse_joints
            ),
            (54, 53, 52, 0)
        );
        let initial = runner.initial_frame().unwrap();
        assert_eq!(initial.source_tick, 0);
        let mut rows = Vec::new();
        let mut reason = "completed".to_string();
        let mut phases = Vec::new();
        let mut command = G1Command::default();
        for tick in 0..650 {
            let phase = match tick {
                0..100 => "stand",
                100..200 => "turn",
                200..350 => "walk",
                350..450 => "stop",
                _ => "reach",
            };
            if phases.last().map(String::as_str) != Some(phase) {
                phases.push(phase.to_string());
            }
            command.navigation = match phase {
                "turn" => [0., 0., 0.2],
                "walk" => [0.25, 0., 0.],
                _ => [0.; 3],
            };
            if phase == "reach" {
                let a = ((tick - 450) as f32 / 100.).min(1.);
                command.upper_positions[0] = -0.5 * a;
                command.upper_positions[3] = 0.3 * a;
            }
            match runner.step(&command) {
                Ok(step) => {
                    assert_eq!(step.step_configuration.physics_hz, 50);
                    assert_eq!(step.step_configuration.num_solver_iterations, 1);
                    assert_eq!(step.step_configuration.max_ccd_substeps, 1);
                    assert_eq!(step.step_configuration.additional_solver_iterations_max, 0);
                    assert_eq!(step.integration_count, step.frame.source_tick);
                    let fell =
                        step.root_position_source[2] < 0.35 || step.root_upright_cosine < 0.5;
                    rows.push(serde_json::json!({"phase":phase,"command":command,"step":step}));
                    if fell {
                        reason = "fell: pelvis below 0.35m or tilt above 60deg".into();
                        break;
                    }
                }
                Err(e) => {
                    reason = e.to_string();
                    break;
                }
            }
        }
        let report = serde_json::json!({"qualification":"diagnostic_only", "source_physics_hz":200,"source_control_hz":50,
            "target_physics_hz":50,"target_control_hz":50, "time_substeps":1,
            "limits":["four shape-less fixed sensor mass frames use explicit diagnostic origin/identity fallback",
                "PhysX friction coefficient is not a Rapier torque; joint friction disabled pending mapping",
                "source joint velocity constraint and convex cooking/material equivalence not yet qualified"],
            "definition_sha256":config.definition_sha256,"counts":counts,"initial":initial,"phases_entered":phases,"termination":reason,"steps":rows});
        fs::write(
            env::var("G1_T0_OUTPUT").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        println!(
            "G1_T0 termination={reason}; completed_ticks={}; phases={phases:?}",
            rows.len()
        );
        // This test validates an honest real trajectory/receipt, not T0 skill success.
        assert!(
            !rows.is_empty(),
            "no real physics boundary completed: {reason}"
        );
    }
}
