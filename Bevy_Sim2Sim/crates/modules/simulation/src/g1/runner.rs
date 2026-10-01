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

/// Readable even after a failed boundary; no rendered-frame counts are inferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct G1ProgressCounts {
    pub integration_count: u64,
    pub torque_update_count: u64,
    /// Actual ORT session calls, including failed calls.
    pub inference_attempt_count: u64,
    /// Outputs validated and accepted into the policy history.
    pub successful_inference_count: u64,
    pub halted: bool,
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

    pub fn progress_counts(&self) -> G1ProgressCounts {
        G1ProgressCounts {
            integration_count: self.simulation.integration_count,
            torque_update_count: self.simulation.torque_update_count,
            inference_attempt_count: self.policy.inference_attempt_count(),
            successful_inference_count: self.policy.inference_count(),
            halted: self.halted,
        }
    }

    /// All errors halt this candidate; a caller cannot skip a failed inference
    /// and continue integrating with a stale target or a synthetic observation.
    pub fn step(&mut self, command: &G1Command) -> Result<G1Step, RobotError> {
        self.step_with_guard(command, || Ok(()))
    }

    /// Check the caller's deadline/cancellation guard before inference and again
    /// after successful inference, before PD or integration. A rejected guard is
    /// propagated unchanged and permanently halts this candidate: its policy
    /// history may already have advanced, so it must be rebuilt, never resumed.
    pub fn step_with_guard(
        &mut self,
        command: &G1Command,
        mut guard: impl FnMut() -> Result<(), RobotError>,
    ) -> Result<G1Step, RobotError> {
        if self.halted {
            return Err(error("G1 runner halted after a failed boundary"));
        }
        let result = self.step_inner(command, &mut guard);
        if result.is_err() {
            self.halted = true;
        }
        result
    }

    fn step_inner(
        &mut self,
        command: &G1Command,
        guard: &mut impl FnMut() -> Result<(), RobotError>,
    ) -> Result<G1Step, RobotError> {
        guard()?;
        let state = self.assembly.state(&self.simulation.world)?;
        let inference = self.policy.infer(&state, command)?;
        guard()?;
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
            root_velocity_source: engine_to_source_vector(
                self.assembly
                    .current_root_velocity(&self.simulation.world)?
                    .linvel
                    .to_array(),
            ),
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
    #[ignore = "requires frozen G1_MODEL_DIR, G1_DEFINITION_SHA256, G1_ORT, G1_ORT_SHA256; one real inference, no integration"]
    fn real_homie_guard_rejects_after_inference_without_integration() {
        let mut runner = G1Runner::load(&config()).unwrap();
        let initial = serde_json::to_value(runner.initial_frame().unwrap()).unwrap();
        let mut guard_calls = 0;
        let result = runner.step_with_guard(&G1Command::default(), || {
            guard_calls += 1;
            if guard_calls == 1 {
                Ok(())
            } else {
                Err(error("test deadline expired after real inference"))
            }
        });
        assert!(
            matches!(result, Err(RobotError::Contract(message)) if message == "test deadline expired after real inference")
        );
        assert_eq!(guard_calls, 2);
        let rejected = G1ProgressCounts {
            integration_count: 0,
            torque_update_count: 0,
            inference_attempt_count: 1,
            successful_inference_count: 1,
            halted: true,
        };
        assert_eq!(runner.progress_counts(), rejected);
        assert_eq!(
            serde_json::to_value(runner.initial_frame().unwrap()).unwrap(),
            initial
        );
        assert!(runner.step(&G1Command::default()).is_err());
        assert_eq!(runner.progress_counts(), rejected);
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
        let report = serde_json::json!({"qualification":"diagnostic_only", "code_commit":env::var("G1_CODE_COMMIT").unwrap(), "t0_completed":reason == "completed",
            "usd_sha256":robot_minigame::g1::definition::USD_SHA256,"stand_sha256":robot_minigame::g1::policy::STAND_SHA256,"walk_sha256":robot_minigame::g1::policy::WALK_SHA256,"ort_sha256":config.ort_sha256, "root_pose":config.root_pose.position, "source_physics_hz":200,"source_control_hz":50,
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

    /// Separate development-only source-frequency diagnostic. It never calls the
    /// 50 Hz game runner or presents its 200 Hz ticks as game qualification.
    #[test]
    #[ignore = "same frozen G1 environment as T0; G1_T0_OUTPUT is a separate diagnostic report"]
    fn real_homie_source_frequency_diagnostic() {
        let config = config();
        let mut runner = G1Runner::load(&config).unwrap();
        runner.simulation.world.integration_parameters.dt = 1. / 200.;
        let mut rows = Vec::new();
        let mut termination = "completed_standing_6_seconds".to_string();
        let mut physics_count = 0_u64;
        let command = G1Command::default();
        'control: for _ in 0..300 {
            let state = runner.assembly.state(&runner.simulation.world).unwrap();
            let inference = runner.policy.infer(&state, &command).unwrap();
            // Source holds the target for four source physics steps and updates
            // its explicit IdealPD from fresh q/dq at each 200Hz boundary.
            for _ in 0..4 {
                let state = runner.assembly.state(&runner.simulation.world).unwrap();
                let tau =
                    actuator::torques(&inference.targets, &state.positions, &state.velocities)
                        .unwrap();
                let pairs = runner
                    .assembly
                    .torques(&runner.simulation.world, &tau)
                    .unwrap();
                for (_, body) in runner.simulation.world.bodies.iter_mut() {
                    body.reset_torques(true);
                }
                for pair in pairs {
                    runner.simulation.world.bodies[pair.body]
                        .add_torque(Vector::from_array(pair.world_torque), true);
                }
                runner.simulation.world.step();
                physics_count += 1;
                if !runner.simulation.world.quarantine().is_empty() {
                    termination = "native_nonfinite_quarantine".into();
                    break 'control;
                }
                let root = &runner.simulation.world.bodies[runner.assembly.root_handle()];
                let position = engine_to_source_vector(root.translation().to_array());
                let upright = (*root.rotation() * Vector::Y).y;
                let state = match runner.assembly.state(&runner.simulation.world) {
                    Ok(s) => s,
                    Err(e) => {
                        termination = e.to_string();
                        break 'control;
                    }
                };
                rows.push(serde_json::json!({"physics_tick":physics_count,"physics_seconds":physics_count as f64 / 200.,
                    "inference_count":inference.inference_count,"root_position_source":position,"upright_cosine":upright,
                    "positions":state.positions.to_vec(),"velocities":state.velocities.to_vec(),"action":inference.action,"torques":tau.to_vec()}));
                if position[2] < 0.35 || upright < 0.5 {
                    termination = "fell".into();
                    break 'control;
                }
            }
        }
        let report = serde_json::json!({"qualification":"source_frequency_diagnostic_only_not_game",
            "code_commit":env::var("G1_CODE_COMMIT").unwrap(),"definition_sha256":config.definition_sha256,
            "stand_sha256":robot_minigame::g1::policy::STAND_SHA256,"walk_sha256":robot_minigame::g1::policy::WALK_SHA256,
            "physics_hz":200,"control_hz":50,"source_target_hold_ticks":4,"solver_iterations":1,"ccd_substeps":1,
            "root_pose":config.root_pose.position,"termination":termination,"physics_count":physics_count,"rows":rows});
        fs::write(
            env::var("G1_T0_OUTPUT").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        println!("G1_SOURCE_200_50 termination={termination}; physical_ticks={physics_count}");
        assert!(!rows.is_empty());
    }

    /// Qualification must go red on failed standing. This is deliberately a
    /// separate entry from the trajectory capture test, which is not a skill gate.
    #[test]
    #[ignore = "requires frozen G1 env, G1_CODE_COMMIT and G1_T0_OUTPUT"]
    fn real_homie_stand_qualification() {
        let config = config();
        let definition = G1Definition::load(&config.definition, &config.definition_sha256).unwrap();
        let mut runner = G1Runner::load(&config).unwrap();
        let command = G1Command::default();
        let initial = runner.initial_frame().unwrap();
        let mut rows = Vec::new();
        let mut first_limit_violation = None;
        let mut termination = "completed_150_stand_ticks".to_string();
        for _ in 0..150 {
            match runner.step(&command) {
                Ok(step) => {
                    assert_eq!(step.integration_count, step.frame.source_tick);
                    assert_eq!(step.integration_count, step.inference.inference_count);
                    assert_eq!(step.step_configuration.physics_hz, 50);
                    assert_eq!(step.step_configuration.num_solver_iterations, 1);
                    assert_eq!(step.step_configuration.max_ccd_substeps, 1);
                    assert_eq!(step.step_configuration.additional_solver_iterations_max, 0);
                    if first_limit_violation.is_none() {
                        for (wbc, &joint_id) in definition.driven_joints().iter().enumerate() {
                            let joint = &definition.model().joints[joint_id];
                            let q = f64::from(step.joint_positions[wbc]);
                            // Numerical allowance is declared, fixed and is not
                            // adjusted between runs to manufacture a pass.
                            if q < joint.limits[0] - 0.001 || q > joint.limits[1] + 0.001 {
                                first_limit_violation = Some(
                                    serde_json::json!({"tick":step.frame.source_tick,"joint":joint.name,"position":q,"limits":joint.limits,"tolerance_rad":0.001}),
                                );
                                break;
                            }
                        }
                    }
                    let fell =
                        step.root_position_source[2] < 0.35 || step.root_upright_cosine < 0.5;
                    rows.push(step);
                    // Keep observing after the first limit violation until the
                    // fall trigger, so this same run reproduces the exact symptom.
                    if fell {
                        termination = "fell".into();
                        break;
                    }
                }
                Err(e) => {
                    termination = e.to_string();
                    break;
                }
            }
        }
        let passed = termination == "completed_150_stand_ticks" && first_limit_violation.is_none();
        let report = serde_json::json!({"qualification":"stand_150_ticks", "passed":passed,
            "code_commit":env::var("G1_CODE_COMMIT").unwrap(),"definition_sha256":config.definition_sha256,
            "usd_sha256":robot_minigame::g1::definition::USD_SHA256,
            "stand_sha256":robot_minigame::g1::policy::STAND_SHA256,"walk_sha256":robot_minigame::g1::policy::WALK_SHA256,
            "ort_sha256":config.ort_sha256,"randomness":"none: deterministic CPU ORT threads=1, identical initial state",
            "physics_hz":50,"control_hz":50,"root_pose":config.root_pose.position,"robot_friction":config.robot_contact_friction,
            "floor_friction":config.floor_contact_friction,"command":command,"initial":initial,
            "first_limit_violation":first_limit_violation,"termination":termination,"steps":rows});
        fs::write(
            env::var("G1_T0_OUTPUT").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        println!(
            "G1_STAND_QUAL passed={passed}; termination={termination}; completed_ticks={}; first_limit_violation={first_limit_violation:?}",
            rows.len()
        );
        assert!(
            passed,
            "real G1 standing failed: {termination}; first_limit_violation={first_limit_violation:?}"
        );
    }
}
