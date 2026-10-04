//! A single owner performs observation, real inference, actuation and one 50Hz
//! integration. Rendering receives immutable completed frames only.

use super::assembly::{self, G1Assembly};
use super::static_environment::{PreparedStaticEnvironment, StaticEnvironmentReceipt};
use super::task_objects::{
    TaskObjectFrame, TaskObjectKind, TaskObjectScene, TaskObjectSceneConfig,
};
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
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc};

/// Homie actuator discretization, recorded independently of policy identity.
/// Existing configurations retain the original explicit-PD path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum G1ActuatorBackend {
    #[default]
    ExternalExplicitPd,
    NativeForceBased,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1RunnerConfig {
    pub episode_id: u64,
    pub definition: PathBuf,
    pub definition_sha256: String,
    pub ort_library: PathBuf,
    pub ort_sha256: String,
    pub stand_model: PathBuf,
    pub walk_model: PathBuf,
    #[serde(default)]
    pub actuator_backend: G1ActuatorBackend,
    /// Source Z-up free pelvis pose. This is not the external USD scene root.
    pub root_pose: SourcePose,
    /// Explicit diagnostic materials; do not claim PhysX cooking/material parity.
    pub robot_contact_friction: f32,
    pub floor_contact_friction: f32,
    #[serde(default)]
    pub task_objects: Option<TaskObjectSceneConfig>,
    /// Prepared from the same public station geometry as rendering, before
    /// the owner clock starts. JSON cannot inject physics geometry here.
    #[serde(skip)]
    pub startup_environment: Option<Arc<PreparedStaticEnvironment>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct G1Step {
    pub actuator_backend: G1ActuatorBackend,
    pub frame: G1BodyFrame,
    /// Native state sampled at the same completed boundary as `frame`.
    pub measurement: G1Measurement,
    pub task_objects: Option<TaskObjectFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_static_environment: Option<StaticEnvironmentReceipt>,
    /// Completed last-solve contacts for an independent station auditor only.
    #[cfg(feature = "g1_constraint_diagnostic")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_station_contacts: Option<Vec<G1StationContact>>,
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    /// Applied external PD efforts. Native motor rows apply their own bounded
    /// impulses, so this is 43 zeros there, never a fabricated motor waveform.
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

/// Self state from the owner world, including its real pre-integration state.
/// No task-object poses or identities enter this observation.
#[derive(Clone, Debug, Serialize)]
pub struct G1Measurement {
    pub episode_id: u64,
    pub source_tick: u64,
    pub sim_time_ns: u64,
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub root_rotation_wxyz: [f32; 4],
    pub root_angular_velocity_body: [f32; 3],
    pub root_velocity_source: [f32; 3],
}

/// Last-solve evidence; never supplied to a policy or command source.
#[cfg(feature = "g1_constraint_diagnostic")]
#[derive(Clone, Debug, Serialize)]
pub struct G1StationContact {
    pub robot_body_index: usize,
    /// Last-solve other-body identity for independent obstacle diagnosis.
    pub other_body_handle: Option<[u32; 2]>,
    pub other_body_fixed: bool,
    /// Legacy all-point cache sum; inactive entries may retain old impulses.
    pub normal_impulse_n_s: f32,
    pub active_solver_normal_impulse_n_s: Option<f32>,
    /// Same published Tick's shape distance, independent of cached anchors.
    pub geometric_distance_after_step_m: Option<f32>,
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
    task_objects: Option<TaskObjectScene>,
    native_static_environment: Option<StaticEnvironmentReceipt>,
    policy: HomiePolicy,
    actuator_backend: G1ActuatorBackend,
    halted: bool,
}

impl G1Runner {
    #[cfg(feature = "g1_constraint_diagnostic")]
    pub fn load_mobile_station_fixture_diagnostic(
        config: &G1RunnerConfig,
    ) -> Result<Self, RobotError> {
        let mut owner = Self::load(config)?;
        owner.enable_mobile_station_fixture_diagnostic()?;
        Ok(owner)
    }

    pub fn load(config: &G1RunnerConfig) -> Result<Self, RobotError> {
        let station_fixtures = config
            .task_objects
            .as_ref()
            .and_then(|scene| scene.source_t2_background.as_ref())
            .is_some_and(|background| {
                background.selection
                    == robot_minigame::g1::task_fixtures::T2BackgroundSelection::StationTaskFixtures
            });
        if station_fixtures && config.startup_environment.is_none() {
            return Err(error(
                "station T2 fixtures require a prepared physical environment before loading models",
            ));
        }
        if config.startup_environment.is_some()
            && config.task_objects.is_some()
            && !station_fixtures
        {
            return Err(error(
                "mobile station stand entry cannot overlay an unvalidated task scene",
            ));
        }
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
        if config.actuator_backend == G1ActuatorBackend::NativeForceBased {
            assembly.set_force_based_targets(&mut simulation.world, &initial)?;
        }
        let task_objects = config
            .task_objects
            .as_ref()
            .map(|scene| {
                scene.load_in_owner_world(
                    &mut simulation,
                    &[TaskObjectKind::BrownBox, TaskObjectKind::BlueBin],
                )
            })
            .transpose()?;
        let native_static_environment = config
            .startup_environment
            .as_ref()
            .map(|environment| environment.replace_startup_floor(&mut simulation, floor))
            .transpose()?;
        Ok(Self {
            episode_id: config.episode_id,
            simulation,
            assembly,
            task_objects,
            native_static_environment,
            policy,
            actuator_backend: config.actuator_backend,
            halted: false,
        })
    }

    /// One explicit released T2 convergence candidate; normal loading stays at
    /// one PGS sweep. Requires the complete original background and force motors.
    #[cfg(any(test, feature = "g1_constraint_diagnostic"))]
    pub(super) fn enable_mobile_constraint_diagnostic(&mut self) -> Result<(), RobotError> {
        if self.simulation.integration_count != 0
            || self.progress_counts().inference_attempt_count != 0
            || self.actuator_backend != G1ActuatorBackend::NativeForceBased
            || self
                .task_objects
                .as_ref()
                .is_none_or(|scene| !scene.has_source_t2_background())
            || self
                .simulation
                .world
                .integration_parameters
                .num_internal_pgs_iterations
                != 1
        {
            return Err(error(
                "mobile convergence candidate requires a fresh released T2 scene and native force motors",
            ));
        }
        self.simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 4;
        Ok(())
    }

    /// Separate station transfer candidate. The original-scene factory still
    /// requires complete source coverage; neither factory adds time substeps.
    #[cfg(any(test, feature = "g1_constraint_diagnostic"))]
    pub(super) fn enable_mobile_station_fixture_diagnostic(&mut self) -> Result<(), RobotError> {
        if self.simulation.integration_count != 0
            || self.progress_counts().inference_attempt_count != 0
            || self.actuator_backend != G1ActuatorBackend::NativeForceBased
            || self
                .task_objects
                .as_ref()
                .is_none_or(|scene| !scene.has_station_t2_fixtures())
            || self
                .native_static_environment
                .as_ref()
                .is_none_or(|environment| {
                    !environment.original_broad_floor_removed
                        || environment.owner_integrations_at_installation != 0
                })
            || self
                .simulation
                .world
                .integration_parameters
                .num_internal_pgs_iterations
                != 1
        {
            return Err(error(
                "station fixture candidate requires a fresh same-world environment, frozen T2 support fixtures and native force motors",
            ));
        }
        self.simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 4;
        Ok(())
    }

    /// Test-private single comparison after the exactly reproduced grasp/scan
    /// prefix. No runtime setter or arbitrary solver search is exposed.
    #[cfg(all(test, feature = "g1_constraint_diagnostic"))]
    pub(super) fn saved_walk_sixteen_pgs_comparison(&mut self) -> Result<(), RobotError> {
        if self.simulation.integration_count != 996
            || self.progress_counts().successful_inference_count != 996
            || self.actuator_backend != G1ActuatorBackend::NativeForceBased
            || self
                .task_objects
                .as_ref()
                .is_none_or(|s| !s.has_source_t2_background())
            || self
                .simulation
                .world
                .integration_parameters
                .num_internal_pgs_iterations
                != 4
        {
            return Err(error(
                "saved walking comparison requires the exact996Tick original4PGS prefix",
            ));
        }
        self.simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 16;
        Ok(())
    }

    pub fn initial_frame(&self) -> Result<G1BodyFrame, RobotError> {
        self.completed_frame(&self.simulation.snapshot())
    }

    /// Truth for rendering/independent acceptance; absent from self-state inputs.
    pub fn task_object_frame(&self) -> Result<Option<TaskObjectFrame>, RobotError> {
        self.task_objects
            .as_ref()
            .map(|scene| {
                let mut frame = scene.frame(&self.simulation)?;
                self.assembly.annotate_task_contact_bodies(&mut frame);
                frame.episode_id = self.episode_id;
                Ok(frame)
            })
            .transpose()
    }

    #[cfg(feature = "g1_constraint_diagnostic")]
    fn native_station_contacts(&self) -> Option<Vec<G1StationContact>> {
        self.native_static_environment.as_ref()?;
        let world = &self.simulation.world;
        let handles = self.assembly.diagnostic_body_handles();
        Some(
            world
                .narrow_phase
                .contact_pairs()
                .filter_map(|pair| {
                    if !pair.has_any_active_contact() {
                        return None;
                    }
                    let left = world.colliders[pair.collider1]
                        .parent()
                        .and_then(|h| handles.iter().position(|r| *r == h));
                    let right = world.colliders[pair.collider2]
                        .parent()
                        .and_then(|h| handles.iter().position(|r| *r == h));
                    let (robot_body_index, other) = match (left, right) {
                        (Some(index), None) => (index, pair.collider2),
                        (None, Some(index)) => (index, pair.collider1),
                        _ => return None,
                    };
                    let other_body_fixed = world.colliders[other]
                        .parent()
                        .is_some_and(|h| world.bodies[h].is_fixed());
                    let position = |handle: rapier3d::prelude::ColliderHandle| {
                        let collider = &world.colliders[handle];
                        collider.parent().map_or_else(
                            || *collider.position(),
                            |body| {
                                *world.bodies[body].position()
                                    * collider.position_wrt_parent().copied().unwrap_or_default()
                            },
                        )
                    };
                    let distance = rapier3d::parry::query::distance(
                        &position(pair.collider1),
                        world.colliders[pair.collider1].shape(),
                        &position(pair.collider2),
                        world.colliders[pair.collider2].shape(),
                    )
                    .ok()
                    .filter(|d| d.is_finite() && *d >= 0.);
                    Some(G1StationContact {
                        robot_body_index,
                        other_body_handle: world.colliders[other].parent().map(|h| {
                            let (index, generation) = h.into_raw_parts();
                            [index, generation]
                        }),
                        other_body_fixed,
                        normal_impulse_n_s: pair.total_impulse_magnitude(),
                        active_solver_normal_impulse_n_s:
                            super::task_objects::active_solver_normal_impulse(pair),
                        geometric_distance_after_step_m: distance,
                    })
                })
                .collect(),
        )
    }

    /// Completed last-solve evidence only; no additional collision refresh/step.
    #[cfg(all(test, feature = "g1_constraint_diagnostic"))]
    pub(super) fn diagnostic_robot_background_contacts(&self) -> Vec<serde_json::Value> {
        let world = &self.simulation.world;
        let handles = self.assembly.diagnostic_body_handles();
        world.narrow_phase.contact_pairs().filter_map(|pair| {
            if !pair.has_any_active_contact() || pair.total_impulse_magnitude()<=1e-8 { return None; }
            let left=world.colliders[pair.collider1].parent().and_then(|h| handles.iter().position(|r| *r==h));
            let right=world.colliders[pair.collider2].parent().and_then(|h| handles.iter().position(|r| *r==h));
            let (robot,other)=match (left,right) {
                (Some(i),None)=>(i,pair.collider2),
                (None,Some(i))=>(i,pair.collider1),
                _=>return None,
            };
            let path=self.task_objects.as_ref()?.diagnostic_background_path(other)?;
            Some(serde_json::json!({"robot_body_index":robot,"background_collider_path":path,
                "normal_impulse_n_s":pair.total_impulse_magnitude(),"source_tick":self.simulation.integration_count}))
        }).collect()
    }

    /// Test-only causal comparison after an exactly matched200Tick grasp.
    /// Forces fresh contact geometry; no solver iterations, gains, time step,
    /// body pose or motor target is altered.
    #[cfg(all(test, feature = "g1_constraint_diagnostic"))]
    pub(super) fn diagnostic_refresh_contacts_after_grasp(&mut self) -> Result<(), RobotError> {
        if self.progress_counts().integration_count != 200
            || !self
                .task_objects
                .as_ref()
                .is_some_and(|s| s.has_source_t2_background())
            || self.simulation.world.integration_parameters.dt != 0.02
            || self
                .simulation
                .world
                .integration_parameters
                .num_internal_pgs_iterations
                != 4
        {
            return Err(error(
                "contact-refresh comparison requires matched200Tick mobile grasp",
            ));
        }
        self.simulation
            .world
            .integration_parameters
            .contact_recycling = false;
        Ok(())
    }

    pub fn measurement(&self) -> Result<G1Measurement, RobotError> {
        let state = self.assembly.state(&self.simulation.world)?;
        Ok(G1Measurement {
            episode_id: self.episode_id,
            source_tick: self.simulation.integration_count,
            sim_time_ns: self
                .simulation
                .integration_count
                .checked_mul(20_000_000)
                .ok_or_else(|| error("G1 observation timestamp overflow"))?,
            joint_positions: state.positions.to_vec(),
            joint_velocities: state.velocities.to_vec(),
            root_rotation_wxyz: state.root_rotation_wxyz,
            root_angular_velocity_body: state.root_angular_velocity_body,
            root_velocity_source: engine_to_source_vector(
                self.assembly
                    .current_root_velocity(&self.simulation.world)?
                    .linvel
                    .to_array(),
            ),
        })
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
        let applied = match self.actuator_backend {
            G1ActuatorBackend::ExternalExplicitPd => {
                actuator::torques(&inference.targets, &state.positions, &state.velocities)?
            }
            G1ActuatorBackend::NativeForceBased => {
                self.assembly
                    .set_force_based_targets(&mut self.simulation.world, &inference.targets)?;
                [0.; 43]
            }
        };
        let torques = match self.actuator_backend {
            G1ActuatorBackend::ExternalExplicitPd => {
                self.assembly.torques(&self.simulation.world, &applied)?
            }
            G1ActuatorBackend::NativeForceBased => Vec::new(),
        };
        let snapshot = self.simulation.step_with_torques(&torques).map_err(error)?;
        if snapshot.integration_count != inference.inference_count
            || snapshot.torque_update_count != inference.inference_count
        {
            return Err(error("G1 policy/torque/physics count mismatch"));
        }
        let measurement = self.measurement()?;
        let root = &self.simulation.world.bodies[self.assembly.root_handle()];
        Ok(G1Step {
            actuator_backend: self.actuator_backend,
            frame: self.completed_frame(&snapshot)?,
            joint_positions: measurement.joint_positions.clone(),
            joint_velocities: measurement.joint_velocities.clone(),
            applied_torques: applied.to_vec(),
            inference,
            root_position_source: engine_to_source_vector(root.translation().to_array()),
            root_velocity_source: measurement.root_velocity_source,
            measurement,
            task_objects: self.task_object_frame()?,
            native_static_environment: self.native_static_environment.clone(),
            #[cfg(feature = "g1_constraint_diagnostic")]
            native_station_contacts: self.native_station_contacts(),
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

    fn portable_mobile_configuration() -> serde_json::Value {
        serde_json::json!({"episode_id":1,"definition":"not-loaded.json",
            "definition_sha256":"1".repeat(64),"ort_library":"not-loaded.so",
            "ort_sha256":"2".repeat(64),"stand_model":"stand.onnx","walk_model":"walk.onnx",
            "root_pose":{"position":[0.,0.,0.78],"rotation_wxyz":[1.,0.,0.,0.]},
            "robot_contact_friction":0.5,"floor_contact_friction":1.0})
    }

    #[test]
    fn serialized_mobile_inputs_cannot_inject_prepared_environment() {
        let original = portable_mobile_configuration();
        let legacy: G1RunnerConfig = serde_json::from_value(original.clone()).unwrap();
        assert!(legacy.startup_environment.is_none());
        let mut injected = original;
        injected["startup_environment"] = serde_json::json!({"colliders":[]});
        assert!(serde_json::from_value::<G1RunnerConfig>(injected).is_err());
    }

    #[test]
    fn mobile_station_task_overlay_is_rejected_before_loading_models() {
        use super::super::static_environment::{StaticEnvironmentIdentity, StaticEnvironmentShape};
        let mut config: G1RunnerConfig =
            serde_json::from_value(portable_mobile_configuration()).unwrap();
        config.startup_environment = Some(Arc::new(
            PreparedStaticEnvironment::prepare(
                StaticEnvironmentIdentity {
                    source: "public fixture".into(),
                    model_sha256: "1".repeat(64),
                    manifest_sha256: "2".repeat(64),
                    layout_sha256: "3".repeat(64),
                },
                vec![StaticEnvironmentShape::Triangles {
                    vertices: vec![[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]],
                    indices: vec![[0, 1, 2]],
                }],
                1.,
            )
            .unwrap(),
        ));
        config.task_objects = Some(TaskObjectSceneConfig {
            definition: "not-loaded.json".into(),
            definition_sha256: "4".repeat(64),
            placements: vec![],
            source_t1_shelf: None,
            source_t2_background: None,
        });
        assert!(
            matches!(G1Runner::load(&config),Err(RobotError::Contract(message))
            if message.contains("unvalidated task scene"))
        );
        use robot_minigame::g1::task_fixtures::T2BackgroundSelection;
        config.task_objects.as_mut().unwrap().source_t2_background =
            Some(super::super::task_background::T2SourceBackgroundConfig {
                definition: "not-loaded-background.json".into(),
                definition_sha256: "5".repeat(64),
                environment_translation_source: [0., 0., 0.795],
                selection: T2BackgroundSelection::OriginalScene,
            });
        assert!(
            matches!(G1Runner::load(&config), Err(RobotError::Contract(message)) if message.contains("unvalidated task scene"))
        );
        config
            .task_objects
            .as_mut()
            .unwrap()
            .source_t2_background
            .as_mut()
            .unwrap()
            .selection = T2BackgroundSelection::StationTaskFixtures;
        config.startup_environment = None;
        assert!(
            matches!(G1Runner::load(&config), Err(RobotError::Contract(message)) if message.contains("prepared physical environment"))
        );
    }

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
            actuator_backend: match env::var("G1_ACTUATOR_BACKEND").as_deref() {
                Ok("native_force_based") => G1ActuatorBackend::NativeForceBased,
                Ok("external_explicit_pd") | Err(_) => G1ActuatorBackend::ExternalExplicitPd,
                Ok(other) => panic!("unknown G1_ACTUATOR_BACKEND: {other}"),
            },
            root_pose: SourcePose {
                position: [0., 0., 0.78],
                rotation_wxyz: [1., 0., 0., 0.],
            },
            robot_contact_friction: 0.5,
            floor_contact_friction: 1.,
            task_objects: None,
            startup_environment: None,
        }
    }

    #[test]
    #[ignore = "requires frozen G1_MODEL_DIR, G1_DEFINITION_SHA256, G1_ORT, G1_ORT_SHA256; one real inference, no integration"]
    fn real_homie_guard_rejects_after_inference_without_integration() {
        let mut runner = G1Runner::load(&config()).unwrap();
        let initial = serde_json::to_value(runner.initial_frame().unwrap()).unwrap();
        let measurement = runner.measurement().unwrap();
        assert_eq!(measurement.episode_id, 0);
        assert_eq!(measurement.source_tick, 0);
        assert_eq!(measurement.sim_time_ns, 0);
        assert_eq!(measurement.joint_positions.len(), 43);
        assert_eq!(measurement.joint_velocities, vec![0.; 43]);
        assert_eq!(measurement.root_velocity_source, [0.; 3]);
        assert_eq!(measurement.joint_positions[..15], LOWER_HOME,);
        let initial_measurement = serde_json::to_value(measurement).unwrap();
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
        assert_eq!(
            serde_json::to_value(runner.measurement().unwrap()).unwrap(),
            initial_measurement
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
                    assert_eq!(step.integration_count, step.inference.inference_count);
                    assert_eq!(step.integration_count, step.torque_update_count);
                    if config.actuator_backend == G1ActuatorBackend::NativeForceBased {
                        assert_eq!(step.applied_torques, vec![0.; 43]);
                        assert!(
                            runner
                                .simulation
                                .world
                                .bodies
                                .iter()
                                .all(|(_, body)| body.user_torque().length() == 0.)
                        );
                    }
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
        let report = serde_json::json!({"qualification":"diagnostic_only", "qualified":false, "code_commit":env::var("G1_CODE_COMMIT").unwrap(), "actuator_backend":config.actuator_backend,"t0_completed":reason == "completed",
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

    #[test]
    #[ignore = "requires frozen G1_T2_NATIVE_CONFIG and fresh G1_T0_OUTPUT; one 150-Tick complete-scene contact diagnostic"]
    fn real_released_t2_scene_contact_diagnostic() {
        use std::io::Write;
        let config_bytes = fs::read(env::var("G1_T2_NATIVE_CONFIG").unwrap()).unwrap();
        let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let config: G1RunnerConfig = serde_json::from_slice(&config_bytes).unwrap();
        let output = PathBuf::from(env::var("G1_T0_OUTPUT").unwrap());
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .unwrap();
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.with_extension("jsonl"))
            .unwrap();
        let mut runner = G1Runner::load(&config).unwrap();
        let initial = runner.task_object_frame().unwrap().unwrap();
        let background = initial.source_t2_background.as_ref().unwrap();
        assert_eq!(background.bodies.len(), 3);
        assert_eq!(
            background
                .bodies
                .iter()
                .map(|b| b.collider_count)
                .sum::<usize>(),
            250
        );
        assert_eq!(background.bodies.iter().filter(|b| b.dynamic).count(), 2);
        assert_eq!(initial.world_counts.bodies, 59);
        assert_eq!(initial.world_counts.colliders, 305);
        assert_eq!(runner.progress_counts().integration_count, 0);
        assert_eq!(
            runner
                .simulation
                .world
                .integration_parameters
                .num_internal_pgs_iterations,
            1
        );
        // Verify malformed mass data fails during preparation without a partial
        // background insert into this actual owner world.
        let bg_config = config
            .task_objects
            .as_ref()
            .unwrap()
            .source_t2_background
            .as_ref()
            .unwrap();
        let mut bad_config = bg_config.clone();
        bad_config.definition = env::var("G1_T2_INVALID_BACKGROUND").unwrap().into();
        bad_config.definition_sha256 = env::var("G1_T2_INVALID_BACKGROUND_SHA256").unwrap();
        let before = runner.counts();
        assert!(bad_config.prepare(&runner.simulation).is_err());
        assert_eq!(runner.counts(), before);
        runner.enable_mobile_constraint_diagnostic().unwrap();
        let mut termination = "completed".to_owned();
        let mut rows = 0;
        let mut minimum_upright = 1_f32;
        for _ in 0..150 {
            let step = match runner.step(&G1Command::default()) {
                Ok(s) => s,
                Err(e) => {
                    termination = e.to_string();
                    break;
                }
            };
            rows += 1;
            minimum_upright = minimum_upright.min(step.root_upright_cosine);
            assert_eq!(step.integration_count, rows);
            assert_eq!(step.torque_update_count, rows);
            assert_eq!(step.step_configuration.physics_hz, 50);
            assert_eq!(step.step_configuration.num_solver_iterations, 1);
            assert_eq!(step.step_configuration.num_internal_pgs_iterations, 4);
            assert_eq!(step.step_configuration.max_ccd_substeps, 1);
            serde_json::to_writer(&mut trace, &step).unwrap();
            writeln!(trace).unwrap();
            trace.flush().unwrap();
            if step.root_upright_cosine < 0.5 || step.root_position_source[2] < 0.35 {
                termination = "fell".into();
                break;
            }
        }
        let report = serde_json::json!({"qualified":false,"scope":"one complete released native T2 scene contact diagnostic, no VLA and no navigation",
            "code_commit":env::var("G1_CODE_COMMIT").unwrap(),"config":config_value,"initial":initial,
            "minimum_upright":minimum_upright,"completed_integrations":rows,"termination":termination,"counts":runner.progress_counts(),
            "final":runner.task_object_frame().unwrap(),"native_frequency_hz":50,"integrations_per_tick":1,
            "nonintegrating_pgs_candidate":4,"public_default_pgs":1,"vla_calls":0,"task_success":false});
        serde_json::to_writer(&mut file, &report).unwrap();
        writeln!(file).unwrap();
        println!(
            "G1_RELEASED_T2_SCENE integrations={rows} termination={termination} upright={minimum_upright}"
        );
        assert!(rows > 0, "no actual native scene step");
    }

    /// Single bounded baseline for the released T2 body, without task objects.
    /// Measures drift and every original joint limit; survival alone is not T0
    /// qualification. This never changes the owner's solver or time settings.
    #[test]
    #[ignore = "requires frozen G1 env, native_force_based, G1_CODE_COMMIT and fresh G1_T0_OUTPUT"]
    fn real_homie_released_stand30_diagnostic() {
        use std::{io::Write, time::Instant};
        let mut config = config();
        assert_eq!(config.actuator_backend, G1ActuatorBackend::NativeForceBased);
        // Published T2 root [0,.18,0], shifted with its floor by +.795m.
        // Flat-floor diagnostic only; no claim of original background contact.
        config.root_pose.position = [0., 0.18, 0.795];
        config.floor_contact_friction = 0.5;
        let output = PathBuf::from(env::var("G1_T0_OUTPUT").unwrap());
        let mut report_file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .unwrap();
        let trace_path = output.with_extension("jsonl");
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&trace_path)
            .unwrap();
        let definition = G1Definition::load(&config.definition, &config.definition_sha256).unwrap();
        let limits: Vec<_> = definition
            .driven_joints()
            .iter()
            .map(|&id| definition.model().joints[id].limits)
            .collect();
        let started = Instant::now();
        let mut runner = G1Runner::load(&config).unwrap();
        // One bounded algebraic convergence comparison, never a time substep
        // or a public runner default. Other values cannot request a sweep.
        let pgs = match env::var("G1_HOMIE_STAND_PGS_DIAGNOSTIC").as_deref() {
            Err(_) => 1,
            Ok("4") => {
                assert_eq!(
                    runner
                        .simulation
                        .world
                        .integration_parameters
                        .num_internal_pgs_iterations,
                    1
                );
                runner
                    .simulation
                    .world
                    .integration_parameters
                    .num_internal_pgs_iterations = 4;
                4
            }
            Ok(other) => panic!("unsupported bounded Homie PGS diagnostic: {other}"),
        };
        let initial = runner.measurement().unwrap();
        assert_eq!(initial.source_tick, 0);
        assert_eq!(initial.joint_positions[..15], LOWER_HOME);
        let command = G1Command::default();
        let mut reason = "completed".to_owned();
        let mut completed = 0;
        let mut minimum_upright = 1_f32;
        let mut minimum_height = f32::MAX;
        let mut maximum_drift = 0_f64;
        let mut first_limit_violation = None;
        let mut last_configuration = None;
        for _ in 0..1500 {
            let step = match runner.step(&command) {
                Ok(step) => step,
                Err(error) => {
                    reason = error.to_string();
                    break;
                }
            };
            completed += 1;
            let params = &step.step_configuration;
            assert_eq!(params.physics_hz, 50);
            assert_eq!(params.dt, 1. / 50.);
            assert_eq!(params.num_solver_iterations, 1);
            assert_eq!(params.num_internal_pgs_iterations, pgs);
            assert_eq!(params.max_ccd_substeps, 1);
            assert_eq!(params.additional_solver_iterations_max, 0);
            assert_eq!(step.integration_count, completed);
            assert_eq!(step.torque_update_count, completed);
            assert_eq!(step.inference.inference_count, completed);
            assert_eq!(step.inference.policy, "stand");
            assert_eq!(step.applied_torques, vec![0.; 43]);
            minimum_upright = minimum_upright.min(step.root_upright_cosine);
            minimum_height = minimum_height.min(step.root_position_source[2]);
            let drift = (f64::from(step.root_position_source[0]).powi(2)
                + (f64::from(step.root_position_source[1]) - 0.18).powi(2))
            .sqrt();
            maximum_drift = maximum_drift.max(drift);
            if first_limit_violation.is_none() {
                for (index, (&q, bounds)) in step
                    .measurement
                    .joint_positions
                    .iter()
                    .zip(&limits)
                    .enumerate()
                {
                    if f64::from(q) < bounds[0] - 1e-4 || f64::from(q) > bounds[1] + 1e-4 {
                        first_limit_violation = Some(serde_json::json!({"tick":completed,
                            "joint":robot_minigame::g1::contract::JOINT_NAMES[index],"position":q,"limits":bounds,"tolerance_rad":1e-4}));
                        break;
                    }
                }
            }
            let fell = step.root_position_source[2] < 0.35 || step.root_upright_cosine < 0.5;
            serde_json::to_writer(&mut trace, &serde_json::json!({"control_tick":completed,
                "measurement":step.measurement,"command":command,"inference":step.inference,
                "root_position_source":step.root_position_source,"root_upright_cosine":step.root_upright_cosine,
                "active_contact_pairs":step.active_contact_pairs,"integration_count":step.integration_count,
                "torque_update_count":step.torque_update_count,"step_configuration":params})).unwrap();
            trace.write_all(b"\n").unwrap();
            last_configuration = Some(step.step_configuration);
            if fell {
                reason = "fell: pelvis below .35m or tilt above60deg".into();
                break;
            }
        }
        trace.flush().unwrap();
        let report = serde_json::json!({"schema":"g1_released_homie_native_stand30_v1",
            "scope":"single flat-floor body diagnostic; no scene, VLA, task or real-time qualification",
            "qualified":false,"code_commit":env::var("G1_CODE_COMMIT").unwrap(),
            "config":{"definition":config.definition,"definition_sha256":config.definition_sha256,
                "ort_library":config.ort_library,"ort_sha256":config.ort_sha256,
                "stand_model":config.stand_model,"walk_model":config.walk_model,
                "actuator_backend":config.actuator_backend,"episode_id":config.episode_id,
                "root_position_source":config.root_pose.position,"root_rotation_wxyz":config.root_pose.rotation_wxyz,
                "robot_contact_friction":config.robot_contact_friction,"floor_contact_friction":config.floor_contact_friction,
                "task_objects":null},"initial_measurement":initial,"requested_controls":1500,
            "completed_controls":completed,"simulated_seconds":completed as f64*0.02,
            "termination":reason,"minimum_upright_cosine":minimum_upright,
            "minimum_root_height_m":minimum_height,"maximum_horizontal_drift_m":maximum_drift,
            "first_joint_limit_violation":first_limit_violation,"limit_tolerance_rad":1e-4,
            "progress_counts":runner.progress_counts(),"actual_step_configuration":last_configuration,
            "nonintegrating_pgs_diagnostic":pgs != 1,"native_default_changed":false,
            "execution_wall_seconds":started.elapsed().as_secs_f64(),"trace":trace_path,
            "stand_sha256":robot_minigame::g1::policy::STAND_SHA256,
            "walk_sha256":robot_minigame::g1::policy::WALK_SHA256,
            "source_profile":"release_0_2_1","source_physics_hz":200,"native_physics_hz":50,
            "limits":["robot convex cooking and four sensor mass defaults remain diagnostic",
                "flat floor omits original shelf, objects and background contacts",
                "PhysX joint friction and velocity constraint mapping not qualified"]});
        serde_json::to_writer_pretty(&mut report_file, &report).unwrap();
        report_file.flush().unwrap();
        println!(
            "G1_RELEASED_STAND30 termination={reason} controls={completed} max_drift_m={maximum_drift} min_upright={minimum_upright} first_limit={first_limit_violation:?}"
        );
        // The diagnostic reports failures honestly; this asserts measurement,
        // never successful body/task qualification.
        assert!(completed > 0, "no real native boundary completed: {reason}");
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
        let report = serde_json::json!({"qualification":"stand_150_ticks", "passed":passed,"actuator_backend":config.actuator_backend,
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

#[cfg(all(test, feature = "sim2sim_motor_row_trace"))]
#[path = "runner/actuator_diagnostic.rs"]
mod actuator_diagnostic;
