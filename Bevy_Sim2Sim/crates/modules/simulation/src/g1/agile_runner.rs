//! Independent T1 AGILE owner: real recurrence and native ForceBased motors.
//!
//! Angular velocity originates in the root body's COM motion and is expressed
//! in actor/root-link axes. A COM is a point, not principal-inertia orientation.
//! No principal-axis rotation, input normalization or Homie policy is applied.

use std::{cell::Cell, path::PathBuf};

use rapier3d::prelude::*;
use robot_minigame::{
    RobotError,
    basis::{engine_to_source_vector, source_to_engine_rotation},
    g1::{
        actuator::IdealPd,
        agile::{self, AgileCommand, AgilePolicy, AgileResult, AgileState},
        contract::{G1State, JOINT_COUNT, JOINT_NAMES},
        definition::{G1BodyFrame, G1Definition, SourcePose},
    },
};
use serde::{Deserialize, Serialize};

use super::{
    assembly::{self, G1Assembly},
    runner::{G1Measurement, G1ProgressCounts},
    task_objects::{TaskObjectFrame, TaskObjectKind, TaskObjectScene, TaskObjectSceneConfig},
};
use crate::{SimulationWorld, StepConfiguration, WorldCounts};

const TICK_NS: u64 = 20_000_000;
const SOURCE_LIMIT_TOLERANCE_RAD: f64 = 0.001;

/// This first T1 runtime has exactly one explicit candidate backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgileActuatorBackend {
    NativeForceBased,
}

/// All task defaults are supplied explicitly in the original 43-joint order.
/// The same values initialize native q and the recurrent q-minus-default input.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgileRunnerConfig {
    pub episode_id: u64,
    pub definition: PathBuf,
    pub definition_sha256: String,
    pub ort_library: PathBuf,
    pub ort_sha256: String,
    pub agile_model: PathBuf,
    pub root_pose: SourcePose,
    pub robot_contact_friction: f32,
    pub floor_contact_friction: f32,
    pub default_positions: Vec<f32>,
    #[serde(default)]
    pub task_objects: Option<TaskObjectSceneConfig>,
}

impl AgileRunnerConfig {
    /// Validate without loading a model or creating a physics world.
    pub fn validate(&self) -> Result<[f32; JOINT_COUNT], RobotError> {
        if self.default_positions.len() != JOINT_COUNT
            || self
                .default_positions
                .iter()
                .any(|value| !value.is_finite())
            || self
                .root_pose
                .position
                .iter()
                .any(|value| !value.is_finite() || !(*value as f32).is_finite())
            || [self.robot_contact_friction, self.floor_contact_friction]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.)
            || [&self.definition, &self.ort_library, &self.agile_model]
                .iter()
                .any(|path| path.as_os_str().is_empty())
            || !sha256_text(&self.definition_sha256)
            || !sha256_text(&self.ort_sha256)
        {
            return Err(error(
                "invalid explicit AGILE configuration/defaults/materials",
            ));
        }
        // Retain the existing source/engine unit-quaternion guard; no repair.
        source_to_engine_rotation(self.root_pose.rotation_wxyz.map(|value| value as f32))?;
        if let Some(shelf) = self
            .task_objects
            .as_ref()
            .and_then(|o| o.source_t1_shelf.as_ref())
        {
            shelf.validate()?;
            let expected = [
                shelf.environment_translation_source[0] + 0.25,
                shelf.environment_translation_source[1] + 0.08,
                shelf.environment_translation_source[2],
            ];
            if self
                .root_pose
                .position
                .iter()
                .zip(expected)
                .any(|(actual, expected)| (*actual - expected).abs() > 1e-8)
                || self.root_pose.rotation_wxyz[1..]
                    .iter()
                    .any(|v| v.abs() > 1e-8)
            {
                return Err(error(
                    "source shelf profile requires the original translated robot startup pose",
                ));
            }
        }
        let mut defaults = [0.; JOINT_COUNT];
        defaults.copy_from_slice(&self.default_positions);
        Ok(defaults)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct AgileJointLimit {
    pub name: &'static str,
    pub radians: [f64; 2],
}

#[derive(Clone, Debug, Serialize)]
pub struct AgileStep {
    pub actuator_backend: AgileActuatorBackend,
    pub frame: G1BodyFrame,
    pub measurement: G1Measurement,
    pub task_objects: Option<TaskObjectFrame>,
    pub source_t1_finger_material_bodies: Vec<usize>,
    pub inference: AgileResult,
    pub root_position_source: [f32; 3],
    pub root_velocity_source: [f32; 3],
    pub root_upright_cosine: f32,
    pub active_contact_pairs: usize,
    /// Complete 43-motor command updates at runtime boundaries, excluding setup.
    pub motor_update_count: u64,
    /// Empty external-torque boundaries, not native motor-row impulse counts.
    pub torque_update_count: u64,
    pub integration_count: u64,
    pub external_torque_contributions: usize,
    pub step_configuration: StepConfiguration,
}

/// Owns the only floor/robot world and the independent recurrent policy.
/// Public snapshots are read-only. Any operational or snapshot error latches
/// terminal failure; rebuilding is required even after a post-inference guard.
pub struct AgileRunner {
    episode_id: u64,
    simulation: SimulationWorld,
    assembly: G1Assembly,
    task_objects: Option<TaskObjectScene>,
    source_t1_finger_material_bodies: Vec<usize>,
    policy: AgilePolicy,
    parameters: [IdealPd; JOINT_COUNT],
    default_positions: [f32; JOINT_COUNT],
    source_limits: [AgileJointLimit; JOINT_COUNT],
    motor_update_count: u64,
    /// At most one accepted real output survives a later guard/integration failure.
    last_inference: Option<(u64, AgileResult)>,
    halted: Cell<bool>,
}

impl AgileRunner {
    pub fn load(config: &AgileRunnerConfig) -> Result<Self, RobotError> {
        let default_positions = config.validate()?;
        let parameters = source_parameters()?;
        let definition = G1Definition::load(&config.definition, &config.definition_sha256)?;
        let source_limits = std::array::from_fn(|index| AgileJointLimit {
            name: JOINT_NAMES[index],
            radians: definition.model().joints[definition.driven_joints()[index]].limits,
        });
        for (index, limit) in source_limits.iter().enumerate() {
            let position = f64::from(default_positions[index]);
            if position < limit.radians[0] - 1e-5 || position > limit.radians[1] + 1e-5 {
                return Err(error(format!(
                    "AGILE default outside source limit: {}",
                    limit.name
                )));
            }
        }
        let policy = AgilePolicy::load(
            &config.ort_library,
            &config.ort_sha256,
            &config.agile_model,
            default_positions,
        )?;
        let mut simulation = SimulationWorld::with_game_frequency(50).map_err(error)?;
        // Resolve the coupled contact/motor constraints within this one 20 ms
        // step. PGS sweeps do not integrate; num_solver_iterations stays one.
        // This G1 task owner does not change the shared or Homie defaults.
        simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 4;
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
        let assembly = assembly::build_with_actuator_parameters(
            &mut simulation.world,
            &definition,
            &config.root_pose,
            &default_positions,
            config.robot_contact_friction,
            &parameters,
        )?;
        // One non-integrating initial installation; boundary-update count starts at zero.
        assembly.set_force_based_targets_with_parameters(
            &mut simulation.world,
            &default_positions,
            &parameters,
        )?;
        if assembly.body_count() != 53
            || simulation.counts().bodies != 54
            || simulation.counts().colliders != 53
            || simulation.counts().multibody_joint_handles != 52
            || simulation.counts().impulse_joints != 0
            || !simulation.world.bodies[assembly.root_handle()].is_dynamic()
        {
            return Err(error(
                "AGILE native topology must be one floor plus a free 53-link robot",
            ));
        }
        // state/current_root_velocity validates the original unlocked six-DoF root.
        assembly.state(&simulation.world)?;
        assembly.current_root_velocity(&simulation.world)?;
        validate_clock(simulation.configuration())?;
        let source_t1_finger_material_bodies = if config
            .task_objects
            .as_ref()
            .is_some_and(|o| o.source_t1_shelf.is_some())
        {
            assembly.apply_t1_source_finger_material(&mut simulation.world, &definition)?
        } else {
            Vec::new()
        };
        let task_objects = config
            .task_objects
            .as_ref()
            .map(|scene| {
                scene.load_in_owner_world(
                    &mut simulation,
                    &[TaskObjectKind::Apple, TaskObjectKind::Plate],
                )
            })
            .transpose()?;
        let runner = Self {
            episode_id: config.episode_id,
            simulation,
            assembly,
            task_objects,
            source_t1_finger_material_bodies,
            policy,
            parameters,
            default_positions,
            source_limits,
            motor_update_count: 0,
            last_inference: None,
            halted: Cell::new(false),
        };
        runner.frame()?;
        runner.measurement()?;
        runner.task_object_frame()?;
        Ok(runner)
    }

    /// The actual current completed boundary, never a renderer's interpolated pose.
    pub fn frame(&self) -> Result<G1BodyFrame, RobotError> {
        let result = (|| {
            let snapshot = self.simulation.snapshot();
            if snapshot.episode_step != snapshot.integration_count {
                return Err(error("failed integration has no validated completed frame"));
            }
            let mut frame = self.assembly.frame(&snapshot)?;
            frame.episode_id = self.episode_id;
            Ok(frame)
        })();
        self.latch(result)
    }

    /// A load-time read; subsequent calls remain reads of the current boundary.
    pub fn initial_frame(&self) -> Result<G1BodyFrame, RobotError> {
        self.frame()
    }

    /// Render/acceptance truth at the same completed owner boundary as the robot.
    pub fn task_object_frame(&self) -> Result<Option<TaskObjectFrame>, RobotError> {
        self.latch(
            self.task_objects
                .as_ref()
                .map(|scene| {
                    let mut frame = scene.frame(&self.simulation)?;
                    self.assembly.annotate_task_contact_bodies(&mut frame);
                    frame.episode_id = self.episode_id;
                    Ok(frame)
                })
                .transpose(),
        )
    }

    pub fn state(&self) -> Result<AgileState, RobotError> {
        self.latch(
            self.assembly
                .state(&self.simulation.world)
                .and_then(|state| agile_state(&state)),
        )
    }

    pub fn measurement(&self) -> Result<G1Measurement, RobotError> {
        let result = (|| {
            let state = self.assembly.state(&self.simulation.world)?;
            let velocity = self
                .assembly
                .current_root_velocity(&self.simulation.world)?;
            Ok(G1Measurement {
                episode_id: self.episode_id,
                source_tick: self.simulation.integration_count,
                sim_time_ns: self
                    .simulation
                    .integration_count
                    .checked_mul(TICK_NS)
                    .ok_or_else(|| error("AGILE observation timestamp overflow"))?,
                joint_positions: state.positions.to_vec(),
                joint_velocities: state.velocities.to_vec(),
                root_rotation_wxyz: state.root_rotation_wxyz,
                root_angular_velocity_body: state.root_angular_velocity_body,
                root_velocity_source: engine_to_source_vector(velocity.linvel.to_array()),
            })
        })();
        self.latch(result)
    }

    pub fn counts(&self) -> WorldCounts {
        self.simulation.counts()
    }
    pub fn configuration(&self) -> StepConfiguration {
        self.simulation.configuration()
    }

    /// One isolated convergence comparison, with unchanged temporal cadence.
    /// This is absent from normal builds and cannot change a running episode.
    #[cfg(feature = "g1_constraint_diagnostic")]
    pub(super) fn set_diagnostic_constraint_sweeps(&mut self) -> Result<(), RobotError> {
        if self.simulation.integration_count != 0 || self.policy.inference_count() != 0 {
            return Err(error(
                "constraint diagnostic requires a fresh zero-Tick owner",
            ));
        }
        self.simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 16;
        validate_clock(self.configuration())
    }

    /// Predictive bounds are isolated from the normal loading path. No change
    /// to source gains, action semantics or the four nonintegrating sweeps.
    #[cfg(feature = "g1_constraint_diagnostic")]
    pub(super) fn enable_diagnostic_predictive_limits(&mut self) -> Result<(), RobotError> {
        if self.simulation.integration_count != 0 || self.policy.inference_count() != 0 {
            return Err(error(
                "predictive limit diagnostic requires a fresh zero-Tick owner",
            ));
        }
        if self
            .assembly
            .enable_diagnostic_predictive_limits(&mut self.simulation.world)?
            != JOINT_COUNT
        {
            return Err(error(
                "predictive limit diagnostic source joint coverage changed",
            ));
        }
        validate_clock(self.configuration())
    }
    pub fn motor_update_count(&self) -> u64 {
        self.motor_update_count
    }
    pub fn default_positions(&self) -> &[f32; JOINT_COUNT] {
        &self.default_positions
    }
    pub fn actuator_parameters(&self) -> &[IdealPd; JOINT_COUNT] {
        &self.parameters
    }
    pub fn source_limits(&self) -> &[AgileJointLimit; JOINT_COUNT] {
        &self.source_limits
    }
    /// Pre-integration source tick and the last accepted real ORT output.
    pub fn last_inference(&self) -> Option<(u64, &AgileResult)> {
        self.last_inference
            .as_ref()
            .map(|(tick, inference)| (*tick, inference))
    }

    pub fn progress_counts(&self) -> G1ProgressCounts {
        G1ProgressCounts {
            integration_count: self.simulation.integration_count,
            torque_update_count: self.simulation.torque_update_count,
            inference_attempt_count: self.policy.inference_attempt_count(),
            successful_inference_count: self.policy.inference_count(),
            halted: self.halted.get(),
        }
    }

    pub fn step(&mut self, command: &AgileCommand) -> Result<AgileStep, RobotError> {
        self.step_with_guard(command, || Ok(()))
    }

    /// A rejected caller guard, including after recurrence advances, is terminal.
    pub fn step_with_guard(
        &mut self,
        command: &AgileCommand,
        mut guard: impl FnMut() -> Result<(), RobotError>,
    ) -> Result<AgileStep, RobotError> {
        if self.halted.get() {
            return Err(error("AGILE runner halted; cold rebuild required"));
        }
        let result = self.step_inner(command, &mut guard);
        self.latch(result)
    }

    fn step_inner(
        &mut self,
        command: &AgileCommand,
        guard: &mut impl FnMut() -> Result<(), RobotError>,
    ) -> Result<AgileStep, RobotError> {
        guard()?;
        validate_clock(self.simulation.configuration())?;
        let state = self.state()?;
        self.check_source_positions(&state.positions)?;
        let inference = self.policy.infer(&state, command)?;
        self.last_inference = Some((self.simulation.integration_count, inference.clone()));
        guard()?;
        let next_motor_update = self
            .motor_update_count
            .checked_add(1)
            .ok_or_else(|| error("AGILE motor-update count overflow"))?;
        self.assembly.set_force_based_targets_with_parameters(
            &mut self.simulation.world,
            &inference.targets,
            &self.parameters,
        )?;
        self.motor_update_count = next_motor_update;
        // Exactly one existing 20 ms integration, with no external PD contributions.
        let snapshot = self.simulation.step_with_torques(&[]).map_err(error)?;
        if snapshot.integration_count != inference.inference_count
            || snapshot.torque_update_count != inference.inference_count
            || self.motor_update_count != inference.inference_count
            || self.policy.inference_attempt_count() != inference.inference_count
        {
            return Err(error(
                "AGILE inference/motor/empty-torque/integration count mismatch",
            ));
        }
        let frame = self.frame()?;
        let measurement = self.measurement()?;
        self.check_source_positions(&measurement.joint_positions)?;
        if frame.source_tick != snapshot.integration_count
            || measurement.source_tick != frame.source_tick
        {
            return Err(error("AGILE frame/measurement boundary mismatch"));
        }
        let root = &self.simulation.world.bodies[self.assembly.root_handle()];
        Ok(AgileStep {
            actuator_backend: AgileActuatorBackend::NativeForceBased,
            frame,
            measurement: measurement.clone(),
            task_objects: self.task_object_frame()?,
            source_t1_finger_material_bodies: self.source_t1_finger_material_bodies.clone(),
            inference,
            root_position_source: engine_to_source_vector(root.translation().to_array()),
            root_velocity_source: measurement.root_velocity_source,
            root_upright_cosine: (*root.rotation() * Vector::Y).y,
            active_contact_pairs: snapshot.active_contact_pair_count,
            motor_update_count: self.motor_update_count,
            torque_update_count: snapshot.torque_update_count,
            integration_count: snapshot.integration_count,
            external_torque_contributions: 0,
            step_configuration: self.simulation.configuration(),
        })
    }

    fn latch<T>(&self, result: Result<T, RobotError>) -> Result<T, RobotError> {
        if result.is_err() {
            self.halted.set(true);
        }
        result
    }

    fn check_source_positions(&self, positions: &[f32]) -> Result<(), RobotError> {
        for (position, limit) in positions.iter().zip(&self.source_limits) {
            let q = f64::from(*position);
            if q < limit.radians[0] - SOURCE_LIMIT_TOLERANCE_RAD
                || q > limit.radians[1] + SOURCE_LIMIT_TOLERANCE_RAD
            {
                return Err(error(format!(
                    "AGILE measured source limit exceeded at Tick {}: {}={q}, limits={:?}, tolerance={SOURCE_LIMIT_TOLERANCE_RAD}",
                    self.simulation.integration_count, limit.name, limit.radians,
                )));
            }
        }
        Ok(())
    }
}

fn agile_state(state: &G1State) -> Result<AgileState, RobotError> {
    state.validate()?;
    let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
    let state = AgileState {
        positions: state.positions,
        velocities: state.velocities,
        // assembly.state already expresses angular velocity in actor/link axes.
        // Angular velocity is identical at the COM and any point of a rigid body.
        root_angular_velocity_com: state.root_angular_velocity_body,
        projected_gravity_body: [
            (2. * (w * y - x * z)) as f32,
            (-2. * (y * z + w * x)) as f32,
            (-w * w + x * x + y * y - z * z) as f32,
        ],
    };
    state.validate()?;
    Ok(state)
}

fn source_parameters() -> Result<[IdealPd; JOINT_COUNT], RobotError> {
    let mut parameters = [agile::parameters(0)?; JOINT_COUNT];
    for (index, parameter) in parameters.iter_mut().enumerate() {
        *parameter = agile::parameters(index)?;
    }
    for parameter in &parameters {
        if ![
            parameter.stiffness,
            parameter.damping,
            parameter.effort_limit,
            parameter.velocity_limit,
            parameter.armature,
        ]
        .iter()
        .all(|value| value.is_finite())
            || parameter.stiffness <= 0.
            || parameter.damping < 0.
            || parameter.effort_limit <= 0.
            || parameter.velocity_limit <= 0.
            || parameter.armature <= 0.
            || parameter
                .physx_friction_coefficient
                .is_some_and(|value| !value.is_finite() || value < 0.)
        {
            return Err(error("invalid pinned AGILE actuator parameter"));
        }
    }
    Ok(parameters)
}

fn validate_clock(config: StepConfiguration) -> Result<(), RobotError> {
    if config.physics_hz != 50
        || config.dt != 0.02
        || config.num_solver_iterations != 1
        || !(matches!(config.num_internal_pgs_iterations, 1 | 4)
            || (cfg!(feature = "g1_constraint_diagnostic")
                && config.num_internal_pgs_iterations == 16))
        || config.max_ccd_substeps != 1
        || config.additional_solver_iterations_max != 0
    {
        return Err(error(
            "AGILE requires 50 Hz, one time step, and 1 or 4 non-integrating PGS passes",
        ));
    }
    Ok(())
}
fn sha256_text(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
fn error(value: impl std::fmt::Display) -> RobotError {
    RobotError::Contract(value.to_string())
}

#[cfg(test)]
#[path = "source_stand_replay.rs"]
mod source_stand_replay;

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::basis::source_to_engine_vector;
    use robot_minigame::g1::{
        definition::{ARENA_COMMIT, USD_SHA256},
        policy::bound_bytes,
    };
    use serde_json::{Value, json};
    use std::{
        env,
        fs::{self, OpenOptions},
        io::Write,
    };

    fn install_diagnostic_source_hulls(
        runner: &mut AgileRunner,
        config: &AgileRunnerConfig,
        report: &mut Value,
    ) -> Result<(), RobotError> {
        let Ok(path) = env::var("G1_AGILE_DIAGNOSTIC_ROBOT_COOKING_QUERY") else {
            return Ok(());
        };
        let sha = env::var("G1_AGILE_DIAGNOSTIC_ROBOT_COOKING_SHA256").map_err(error)?;
        let bytes = bound_bytes(std::path::Path::new(&path), &sha)?;
        let query: Value = serde_json::from_slice(&bytes).map_err(error)?;
        if query["schema"] != "g1_robot_physx_cooking_query_v1"
            || query["usd_sha256"] != USD_SHA256
            || query["all_robot_queries_succeeded"] != true
            || query["closed_timeline_was_stopped"] != true
            || query["physics_integrations"] != 0
            || query["timeline_started"] != false
            || query.get("error").is_some()
        {
            return Err(error("invalid zero-integration source robot cooking query"));
        }
        let definition = G1Definition::load(&config.definition, &config.definition_sha256)?;
        let mode = env::var("G1_AGILE_DIAGNOSTIC_ROBOT_COOKING_SCOPE")
            .unwrap_or_else(|_| "pelvis_contour_only".into());
        if !matches!(mode.as_str(), "pelvis_contour_only" | "all_source_meshes") {
            return Err(error("invalid bounded source-cooking comparison scope"));
        }
        let collisions = query["collisions"]
            .as_array()
            .ok_or_else(|| error("missing source collisions"))?;
        if collisions.len() != 52 {
            return Err(error("source collision coverage changed"));
        }
        let mut replacements = Vec::new();
        for source in collisions {
            if source["source_type"] != "Mesh"
                || (mode == "pelvis_contour_only" && source["body_name"] != "pelvis_contour_link")
            {
                continue;
            }
            let name = source["body_name"]
                .as_str()
                .ok_or_else(|| error("missing source body name"))?;
            let index = definition
                .model()
                .bodies
                .iter()
                .position(|b| b.name == name)
                .ok_or_else(|| error("foreign cooked body name"))?;
            let hulls = source["hulls"]
                .as_array()
                .ok_or_else(|| error("missing source hulls"))?;
            if hulls.len() != 1 || source["approximation"] != "convexHull" {
                return Err(error("source robot convex cooking contract changed"));
            }
            let hull = &hulls[0];
            let points: Vec<[f32; 3]> =
                serde_json::from_value(hull["vertices_body_local"].clone()).map_err(error)?;
            let indices: Vec<u32> =
                serde_json::from_value(hull["indices"].clone()).map_err(error)?;
            if points.len() < 4 || points.iter().flatten().any(|p| !p.is_finite()) {
                return Err(error("invalid source cooked vertices"));
            }
            let mut triangles = Vec::new();
            for polygon in hull["polygons"]
                .as_array()
                .ok_or_else(|| error("missing source polygons"))?
            {
                let start = polygon["index_base"]
                    .as_u64()
                    .ok_or_else(|| error("invalid source polygon start"))?
                    as usize;
                let count = polygon["num_vertices"]
                    .as_u64()
                    .ok_or_else(|| error("invalid source polygon size"))?
                    as usize;
                let face = indices
                    .get(
                        start
                            ..start
                                .checked_add(count)
                                .ok_or_else(|| error("source face overflow"))?,
                    )
                    .filter(|face| {
                        face.len() >= 3 && face.iter().all(|p| (*p as usize) < points.len())
                    })
                    .ok_or_else(|| error("invalid source polygon indices"))?;
                for i in 1..face.len() - 1 {
                    triangles.push([face[0], face[i], face[i + 1]]);
                }
            }
            let vertices = points
                .into_iter()
                .map(|p| Vector::from_array(source_to_engine_vector(p)))
                .collect();
            let shape = SharedShape::convex_mesh(vertices, &triangles)
                .ok_or_else(|| error("cannot construct copied source convex topology"))?;
            let body = runner.assembly.diagnostic_body_handles()[index];
            let handles = runner.simulation.world.bodies[body].colliders();
            if handles.len() != 1 {
                return Err(error("cooked mesh comparison needs one collider per body"));
            }
            replacements.push((handles[0], shape, name.to_owned()));
        }
        let expected = if mode == "pelvis_contour_only" { 1 } else { 38 };
        if replacements.len() != expected {
            return Err(error("cooked mesh comparison coverage changed"));
        }
        for (handle, shape, _) in &replacements {
            runner.simulation.world.colliders[*handle].set_shape(SharedShape::compound(vec![(
                rapier3d::math::Pose::IDENTITY,
                shape.clone(),
            )]));
        }
        report["diagnostic_source_cooking_overlay"] = json!({"query_sha256":sha,"scope":mode,
            "source_runtime_build":query["runtime_build"],"t1_matching_runtime_verified":false,
            "source_task_rollout_verified":false,"replaced_body_names":replacements.iter().map(|r|&r.2).collect::<Vec<_>>(),
            "copied_polygon_faces":true,"formal_owner_configuration_changed":false});
        Ok(())
    }

    fn configuration() -> AgileRunnerConfig {
        AgileRunnerConfig {
            episode_id: 1,
            definition: "definition.json".into(),
            definition_sha256: "0".repeat(64),
            ort_library: "libonnxruntime.so".into(),
            ort_sha256: "1".repeat(64),
            agile_model: "agile.onnx".into(),
            root_pose: SourcePose {
                position: [0., 0., 0.78],
                rotation_wxyz: [1., 0., 0., 0.],
            },
            robot_contact_friction: 0.5,
            floor_contact_friction: 1.,
            default_positions: vec![0.; JOINT_COUNT],
            task_objects: None,
        }
    }

    #[test]
    fn configuration_requires_explicit_finite_43_defaults_and_materials() {
        assert!(configuration().validate().is_ok());
        for case in 0..7 {
            let mut config = configuration();
            match case {
                0 => {
                    config.default_positions.pop();
                }
                1 => config.default_positions[16] = f32::NAN,
                2 => config.robot_contact_friction = -0.1,
                3 => config.floor_contact_friction = f32::INFINITY,
                4 => config.root_pose.rotation_wxyz = [2., 0., 0., 0.],
                5 => config.root_pose.position[0] = f64::MAX,
                6 => config.ort_sha256 = "A".repeat(64),
                _ => unreachable!(),
            }
            assert!(config.validate().is_err(), "case {case}");
        }
    }

    #[test]
    fn angular_axes_and_inverse_source_gravity_are_not_principal_inertia_axes() {
        let mut native = G1State {
            positions: std::array::from_fn(|i| i as f32 * 0.01),
            velocities: [0.2; JOINT_COUNT],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.1, -0.2, 0.3],
        };
        let actual = agile_state(&native).unwrap();
        assert_eq!(actual.positions, native.positions);
        assert_eq!(actual.velocities, native.velocities);
        assert_eq!(
            actual.root_angular_velocity_com,
            native.root_angular_velocity_body
        );
        assert_eq!(actual.projected_gravity_body, [0., 0., -1.]);
        native.root_rotation_wxyz = [
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            0.,
            0.,
        ];
        let actual = agile_state(&native).unwrap();
        assert!(actual.projected_gravity_body[0].abs() < 1e-6);
        assert!((actual.projected_gravity_body[1] + 1.).abs() < 1e-6);
        assert!(actual.projected_gravity_body[2].abs() < 1e-6);
        assert_eq!(actual.root_angular_velocity_com, [0.1, -0.2, 0.3]);
        native.root_rotation_wxyz = [2., 0., 0., 0.];
        assert!(agile_state(&native).is_err());
    }

    #[test]
    fn actuator_set_is_source_agile_and_clock_has_no_time_subdivision() {
        let parameters = source_parameters().unwrap();
        assert!((parameters[0].stiffness - 40.17924).abs() < 1e-5);
        assert!((parameters[3].armature - 0.025101925).abs() < 1e-8);
        assert_eq!(parameters[12].stiffness, 300.);
        let clock = StepConfiguration {
            dt: 0.02,
            physics_hz: 50,
            num_solver_iterations: 1,
            num_internal_pgs_iterations: 1,
            max_ccd_substeps: 1,
            additional_solver_iterations_max: 0,
        };
        assert!(validate_clock(clock).is_ok());
        assert!(
            validate_clock(StepConfiguration {
                num_internal_pgs_iterations: 4,
                ..clock
            })
            .is_ok()
        );
        assert!(
            validate_clock(StepConfiguration {
                num_internal_pgs_iterations: 0,
                ..clock
            })
            .is_err()
        );
        assert!(validate_clock(StepConfiguration { dt: 0.005, ..clock }).is_err());
        assert!(
            validate_clock(StepConfiguration {
                additional_solver_iterations_max: 1,
                ..clock
            })
            .is_err()
        );
    }

    fn run_stand(config: &AgileRunnerConfig, report: &mut Value) -> Result<(), RobotError> {
        let defaults = config.validate()?;
        for (index, expected) in [(16, 0.25), (17, 0.5), (30, -0.25), (31, -0.5)] {
            if (defaults[index] - expected).abs() > 1e-6 {
                return Err(error(
                    "stand diagnostic requires frozen T1 static open-arm shoulder defaults",
                ));
            }
        }
        let mut runner = AgileRunner::load(config)?;
        report["initial_task_objects"] = json!(runner.task_object_frame()?);
        install_diagnostic_source_hulls(&mut runner, config, report)?;
        let shelf = runner
            .task_objects
            .as_ref()
            .and_then(TaskObjectScene::diagnostic_shelf_collider);
        let shelf_robot_contacts = env::var("G1_AGILE_DIAGNOSTIC_SHELF_ROBOT_CONTACTS")
            .unwrap_or_else(|_| "original_enabled".into());
        match shelf_robot_contacts.as_str() {
            "original_enabled" => {}
            "diagnostic_disabled" => {
                let handle =
                    shelf.ok_or_else(|| error("shelf contact control needs source shelf"))?;
                runner.simulation.world.colliders[handle].set_collision_groups(
                    InteractionGroups::new(
                        Group::GROUP_2,
                        Group::ALL ^ Group::GROUP_1,
                        InteractionTestMode::And,
                    ),
                );
            }
            _ => return Err(error("unknown diagnostic shelf contact mode")),
        }
        report["diagnostic_shelf_robot_contacts"] = json!(shelf_robot_contacts);
        let contact_recycling = env::var("G1_AGILE_DIAGNOSTIC_CONTACT_RECYCLING")
            .unwrap_or_else(|_| "original_enabled".into());
        match contact_recycling.as_str() {
            "original_enabled" => {}
            "diagnostic_disabled" if shelf.is_some() => {
                runner
                    .simulation
                    .world
                    .integration_parameters
                    .contact_recycling = false;
            }
            _ => return Err(error("unknown or shelf-free contact recycling diagnostic")),
        }
        report["diagnostic_contact_recycling"] = json!(contact_recycling);
        report["diagnostic_contact_samples"] = json!([]);
        let definition = G1Definition::load(&config.definition, &config.definition_sha256)?;
        report["robot_collider_source_mapping"] = json!(runner
            .assembly
            .diagnostic_body_handles()
            .iter()
            .enumerate()
            .flat_map(|(body_index, handle)| {
                let name = &definition.model().bodies[body_index].name;
                runner.simulation.world.bodies[*handle].colliders().iter().map(move |collider| {
                    json!({"collider":format!("{collider:?}"),"body_index":body_index,"body_name":name})
                })
            })
            .collect::<Vec<_>>());
        // An explicit, bounded diagnostic changes only the number of constraint
        // sweeps before the existing integration. Rapier's num_solver_iterations
        // splits dt and stays at one. One pass is retained as the failing control.
        let passes = env::var("G1_AGILE_DIAGNOSTIC_PGS_PASSES")
            .map(|value| value.parse::<usize>().map_err(error))
            .unwrap_or(Ok(4))?;
        if !matches!(passes, 1 | 4) {
            return Err(error("diagnostic PGS passes must be exactly 1 or 4"));
        }
        runner
            .simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = passes;
        validate_clock(runner.configuration())?;
        report["diagnostic_pgs_passes"] = json!(passes);
        let ticks = env::var("G1_AGILE_DIAGNOSTIC_TICKS")
            .map(|value| value.parse::<u64>().map_err(error))
            .unwrap_or(Ok(150))?;
        if !(1..=1500).contains(&ticks) {
            return Err(error("diagnostic Tick budget must be in 1..=1500"));
        }
        report["requested_ticks"] = json!(ticks);
        report["initial_non_integrating_motor_installations"] = json!(1);
        report["actuator_parameters"] = json!(runner.actuator_parameters().to_vec());
        report["source_joint_limits"] = json!(runner.source_limits().to_vec());
        report["initial_frame"] = json!(runner.initial_frame()?);
        report["initial_measurement"] = json!(runner.measurement()?);
        report["initial_counts"] = json!(runner.progress_counts());
        report["world_counts"] = json!(runner.counts());
        report["step_configuration"] = json!(runner.configuration());
        let initial_root = runner.simulation.world.bodies[runner.assembly.root_handle()]
            .translation()
            .to_array();
        let mut upper = [0.; 28];
        upper.copy_from_slice(&defaults[15..]);
        let command = AgileCommand {
            navigation: [0.; 3],
            pelvis_height: 0.75,
            upper_positions: upper,
        };
        report["command"] = json!(command);
        let mut outcome = Ok(());
        let mut max_drift = 0_f64;
        for tick in 1..=ticks {
            if let Some(shelf) = shelf {
                if tick <= 10 || tick % 25 == 0 {
                    let mut pairs = Vec::new();
                    for (body_index, handle) in
                        runner.assembly.diagnostic_body_handles().iter().enumerate()
                    {
                        for collider in runner.simulation.world.bodies[*handle].colliders() {
                            if let Some(pair) = runner
                                .simulation
                                .world
                                .narrow_phase
                                .contact_pair(*collider, shelf)
                                .filter(|pair| pair.has_any_active_contact())
                            {
                                pairs.push(json!({"body_index":body_index,
                                    "body_name":definition.model().bodies[body_index].name,
                                    "impulse_magnitude_ns":pair.total_impulse_magnitude(),
                                    "min_geometric_distance_m":pair.manifolds.iter()
                                        .flat_map(|m|m.points.iter()).map(|p|p.dist)
                                        .fold(f32::INFINITY,f32::min),
                                    "min_solver_distance_m":pair.manifolds.iter()
                                        .flat_map(|m|m.data.solver_contacts.iter()).map(|p|p.dist)
                                        .fold(f32::INFINITY,f32::min)}));
                            }
                        }
                    }
                    report["diagnostic_contact_samples"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"completed_tick":tick-1,"shelf_robot_pairs":pairs}));
                }
            }
            match runner.step(&command) {
                Ok(step) => {
                    let root = runner.simulation.world.bodies[runner.assembly.root_handle()]
                        .translation()
                        .to_array();
                    let drift = ((f64::from(root[0]) - f64::from(initial_root[0])).powi(2)
                        + (f64::from(root[2]) - f64::from(initial_root[2])).powi(2))
                    .sqrt();
                    max_drift = max_drift.max(drift);
                    let violation = runner.source_limits.iter().enumerate().find_map(|(index, limit)| {
                        let q = f64::from(step.measurement.joint_positions[index]);
                        (q < limit.radians[0] - 1e-3 || q > limit.radians[1] + 1e-3)
                            .then(|| json!({"tick":tick,"joint":limit.name,"position":q,"source_limits":limit.radians,"tolerance_rad":0.001}))
                    });
                    let fall =
                        step.root_position_source[2] < 0.35 || step.root_upright_cosine < 0.5;
                    let external_torque_leak = runner
                        .simulation
                        .world
                        .bodies
                        .iter()
                        .any(|(_, body)| body.user_torque().length() != 0.);
                    report["steps"].as_array_mut().unwrap().push(json!(step));
                    if let Some(violation) = violation {
                        report["first_source_limit_violation"] = violation;
                        outcome = Err(error(
                            "measured source joint limit exceeded 0.001 rad tolerance",
                        ));
                    } else if fall {
                        outcome = Err(error(
                            "fell: pelvis below 0.35m or upright cosine below 0.5",
                        ));
                    } else if external_torque_leak {
                        outcome = Err(error(
                            "external torque leaked into native ForceBased candidate",
                        ));
                    }
                    if outcome.is_err() {
                        runner.halted.set(true);
                        break;
                    }
                }
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        report["max_horizontal_drift_m"] = json!(max_drift);
        report["final_counts"] = json!(runner.progress_counts());
        report["final_motor_update_count"] = json!(runner.motor_update_count());
        if let Some((source_tick, inference)) = runner.last_inference() {
            report["last_accepted_inference"] =
                json!({"source_tick_before_inference":source_tick,"inference":inference});
        }
        // Failed snapshots are not relabeled as a valid completed physics frame.
        match runner.frame() {
            Ok(frame) => report["last_finite_frame"] = json!(frame),
            Err(error) => report["last_frame_error"] = json!(error.to_string()),
        }
        match runner.measurement() {
            Ok(measurement) => report["last_finite_measurement"] = json!(measurement),
            Err(error) => report["last_measurement_error"] = json!(error.to_string()),
        }
        report["final_counts"] = json!(runner.progress_counts());
        outcome
    }

    /// Real CPU ONNX + native Rapier only when explicitly invoked by the owner.
    /// Save evidence before asserting failure; never count this as task success.
    #[test]
    #[ignore = "requires frozen G1_AGILE_CONFIG JSON and unique G1_AGILE_OUTPUT path; actual 150-Tick candidate"]
    fn real_agile_static_open_arm_stand_diagnostic() {
        let path = PathBuf::from(env::var("G1_AGILE_CONFIG").expect("G1_AGILE_CONFIG"));
        let output = PathBuf::from(env::var("G1_AGILE_OUTPUT").expect("G1_AGILE_OUTPUT"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .expect("new receipt output in an existing directory");
        let mut report = json!({
            "schema":"g1_agile_native_stand_diagnostic_v1", "qualified":false,
            "scope":"t1_agile_native_force_based_static_open_arm_stand_candidate_only",
            "actuator_backend":"native_force_based", "config_path":path,
            "config_file_sha256_verified":Value::Null, "code_commit_supplied":env::var("G1_CODE_COMMIT").ok(),
            "policy_arena_commit":"8b4a3a47fc53de23e8205089d71109a2e2348acd",
            "body_export_arena_commit":ARENA_COMMIT, "usd_sha256":USD_SHA256,
            "agile_model_sha256":agile::MODEL_SHA256, "contract_version":agile::CONTRACT_VERSION,
            "source_physics_hz":200,"source_control_hz":50,"native_physics_hz":50,"native_control_hz":50,
            "integrations_per_tick":1,"initial_non_integrating_motor_installations":0,"requested_ticks":150,
            "source_limit_tolerance_rad":0.001,"termination":"initializing","error":Value::Null,"steps":[],
            "sensor_axes":"root COM angular velocity expressed in actor/link axes; COM does not mean principal inertia axes",
            "limits":["source joint velocity constraints and PhysX joint friction are not equivalent native constraints",
                "convex cooking/contact material equivalence, long standing, real-time pacing and task success are unqualified",
                "motor-update and empty-torque counts are not measured motor efforts; motor row tracing is separate",
                "external execution wrapper must bind compiled source and test executable identity"]
        });
        let result = (|| -> Result<(), RobotError> {
            let bytes = if let Ok(hash) = env::var("G1_AGILE_CONFIG_SHA256") {
                let bytes = bound_bytes(&path, &hash)?;
                report["config_file_sha256_verified"] = json!(hash);
                bytes
            } else {
                fs::read(&path).map_err(error)?
            };
            report["original_config"] = serde_json::from_slice(&bytes).map_err(error)?;
            let config: AgileRunnerConfig = serde_json::from_slice(&bytes).map_err(error)?;
            report["definition_sha256"] = json!(config.definition_sha256);
            report["ort_sha256"] = json!(config.ort_sha256);
            report["default_positions"] = json!(config.default_positions);
            run_stand(&config, &mut report)
        })();
        let termination = match &result {
            Ok(()) => format!("completed_{}_ticks", report["requested_ticks"]),
            Err(error) => error.to_string(),
        };
        report["termination"] = json!(termination);
        report["completed_stand_budget_without_detected_failure"] = json!(result.is_ok());
        if let Err(error) = &result {
            report["error"] = json!(error.to_string());
        }
        serde_json::to_writer_pretty(&mut file, &report).expect("save finite diagnostic evidence");
        file.write_all(b"\n").unwrap();
        file.sync_all().unwrap();
        println!(
            "AGILE_NATIVE_STAND termination={termination}; ticks={}; qualified=false",
            report["steps"].as_array().unwrap().len()
        );
        assert!(
            result.is_ok(),
            "actual candidate failed; evidence preserved: {termination}"
        );
    }
}
