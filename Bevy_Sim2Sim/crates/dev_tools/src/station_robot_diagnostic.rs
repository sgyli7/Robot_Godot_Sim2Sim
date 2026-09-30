//! A bounded, dev-only 60 Hz Rapier/ORT experiment in one station world.
//!
//! Uses a deliberately simplified P-only torque law while native BAM previous-solve
//! load remains unavailable. Every pose is read after a real integration. This
//! cannot qualify the plant, policy, station contacts or any skill.

use crate::legacy_cpu_actor::LegacyCpuActor;
use common_minigame::events::TickEvent;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, Vector};
use rendering_minigame::{
    StationScene,
    geometry::{StationCollider, StationCollisionShape},
};
use robot_minigame::{
    ACTION_DIMENSION,
    basis::engine_to_source_vector,
    body_pose::RobotPoseFrame,
    contract::{Command, NativeState, PolicyContract},
    definition::RobotDefinition,
    policy::CpuPolicy,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{
    BodyTorque, PHYSICS_HZ, SimulationWorld, StepSnapshot,
    fixed_step_runtime::{FixedStepRuntime, FrameOutcome, TickController},
    robot_builder::{RobotAssembly, build_structure},
};
use std::{fs, path::Path, sync::Arc, time::Duration};

const LEGACY_HOME: [f32; ACTION_DIMENSION] = [
    0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049,
    -0.4530,
];

enum DiagnosticPolicy {
    Native {
        actor: CpuPolicy,
        contract: PolicyContract,
    },
    Legacy {
        actor: LegacyCpuActor,
        name: String,
    },
}

impl DiagnosticPolicy {
    fn observation(
        &self,
        state: &NativeState,
        command: &Command,
        previous_action: &[f32; ACTION_DIMENSION],
    ) -> Result<[f32; 61], String> {
        match self {
            Self::Native { contract, .. } => contract
                .observation(state, command, previous_action)
                .map_err(|error| error.to_string()),
            Self::Legacy { .. } => {
                let mut obs = [0.0_f32; 61];
                obs[..3].copy_from_slice(&state.gyro);
                obs[3..6].copy_from_slice(&state.projected_gravity);
                for index in 0..ACTION_DIMENSION {
                    obs[6 + index] = state.joint_position[index] - LEGACY_HOME[index];
                }
                obs[20..34].copy_from_slice(&state.joint_velocity);
                obs[34..48].copy_from_slice(previous_action);
                obs[48..51].copy_from_slice(&command.locomotion);
                obs[51..55].copy_from_slice(&command.head);
                obs[55..61].copy_from_slice(&command.body);
                if obs.iter().any(|value| !value.is_finite()) {
                    return Err("non-finite legacy observation".into());
                }
                Ok(obs)
            }
        }
    }

    fn infer(&mut self, observation: &[f32; 61]) -> Result<[f32; ACTION_DIMENSION], String> {
        match self {
            Self::Native { actor, .. } => {
                actor.infer(observation).map_err(|error| error.to_string())
            }
            Self::Legacy { actor, .. } => {
                actor.infer(observation).map_err(|error| error.to_string())
            }
        }
    }

    fn targets(&self, action: &[f32; ACTION_DIMENSION]) -> Result<[f32; ACTION_DIMENSION], String> {
        match self {
            Self::Native { contract, .. } => {
                contract.targets(action).map_err(|error| error.to_string())
            }
            Self::Legacy { name, .. } => {
                let scale = match name.as_str() {
                    "alpha_walking" => 1.1_f32,
                    "roller" | "roller_crouch" => 0.8_f32,
                    _ => 1.0_f32,
                };
                let target =
                    std::array::from_fn(|index| action[index] * scale + LEGACY_HOME[index]);
                if target.iter().any(|value: &f32| !value.is_finite()) {
                    return Err("non-finite legacy target".into());
                }
                Ok(target)
            }
        }
    }

    fn applied_targets(
        &self,
        raw: [f32; ACTION_DIMENSION],
        previous: Option<[f32; ACTION_DIMENSION]>,
    ) -> [f32; ACTION_DIMENSION] {
        match (self, previous) {
            (Self::Legacy { .. }, Some(previous)) => std::array::from_fn(|index| {
                let alpha = if (5..=8).contains(&index) {
                    0.5_f32
                } else {
                    0.7_f32
                };
                alpha * raw[index] + (1.0 - alpha) * previous[index]
            }),
            _ => raw,
        }
    }

    fn successful_inference_count(&self) -> u64 {
        match self {
            Self::Native { actor, .. } => actor.successful_inference_count(),
            Self::Legacy { actor, .. } => actor.successful_inference_count(),
        }
    }

    fn command(&self, tick: usize) -> Command {
        let mut command = Command::default();
        if let Self::Legacy { name, .. } = self {
            match name.as_str() {
                "alpha_walking" => command.locomotion[0] = 0.2,
                "roller" => command.locomotion[0] = 0.3,
                "alpha_sitstand" if tick >= 60 && tick < 270 => command.locomotion[0] = 1.0,
                "alpha_ground_pick" | "roller_crouch" => {
                    let (period, phase_end) = if name == "alpha_ground_pick" {
                        (4.0_f64, 0.8_f64)
                    } else {
                        (5.0_f64, 0.7_f64)
                    };
                    let time_seconds = tick as f64 / 60.0;
                    let phase = (time_seconds / period).clamp(0.0, phase_end);
                    command.locomotion[0] = (std::f64::consts::TAU * phase).cos() as f32;
                    command.locomotion[1] = (std::f64::consts::TAU * phase).sin() as f32;
                }
                _ => {}
            }
        }
        command
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn import_station(
    world: &mut SimulationWorld,
    colliders: &[StationCollider],
) -> Result<usize, String> {
    if world.counts().bodies != 0 || world.counts().colliders != 0 {
        return Err("station import requires an empty world".into());
    }
    let mut builders = Vec::new();
    for source in colliders.iter().filter(|source| source.owner == "station") {
        let builder = match &source.shape {
            StationCollisionShape::Convex { vertices } => {
                let points: Vec<Vector> =
                    vertices.iter().copied().map(Vector::from_array).collect();
                ColliderBuilder::convex_hull(&points)
                    .ok_or_else(|| format!("{}: invalid station convex hull", source.tag))?
            }
            StationCollisionShape::Triangles { vertices, indices } => {
                let points: Vec<Vector> =
                    vertices.iter().copied().map(Vector::from_array).collect();
                ColliderBuilder::trimesh(points, indices.clone())
                    .map_err(|error| format!("{}: station triangle mesh: {error:?}", source.tag))?
            }
        };
        builders.push(builder);
    }
    if builders.is_empty() {
        return Err("station has no static colliders".into());
    }
    let fixed = world.world.bodies.insert(RigidBodyBuilder::fixed());
    let count = builders.len();
    for builder in builders {
        // These station shapes are immutable. Materialize exactly Rapier's
        // default-density mass properties during loading instead of deriving
        // thousands of convex-hull inertias inside the first timed tick. Keep
        // collider insertion and collision-index construction in their original
        // order; no dynamics or collision step is performed here.
        let mut collider = builder.build();
        let mass_properties = collider.mass_properties();
        collider.set_mass_properties(mass_properties);
        world
            .world
            .colliders
            .insert_with_parent(collider, fixed, &mut world.world.bodies);
    }
    Ok(count)
}

fn native_state(
    world: &SimulationWorld,
    assembly: &RobotAssembly,
    definition: &RobotDefinition,
) -> Result<(NativeState, [f32; 3]), String> {
    let handle = assembly.body_handles()[1].ok_or("source root body handle is missing")?;
    let root = world
        .world
        .bodies
        .get(handle)
        .ok_or("source root handle is stale")?;
    let angular = root.rotation().inverse() * root.angvel();
    let gravity = root.rotation().inverse() * Vector::new(0.0, -1.0, 0.0);
    let feedback = assembly
        .actuator_joint_feedback(&world.world, definition)
        .map_err(|error| error.to_string())?;
    let mut joint_position = [0.0; ACTION_DIMENSION];
    let mut joint_velocity = [0.0; ACTION_DIMENSION];
    for (index, channel) in feedback.iter().enumerate() {
        joint_position[index] = channel.position;
        joint_velocity[index] = channel.velocity;
    }
    Ok((
        NativeState {
            gyro: engine_to_source_vector(angular.to_array()),
            projected_gravity: engine_to_source_vector(gravity.to_array()),
            joint_position,
            joint_velocity,
        },
        root.translation().to_array(),
    ))
}

struct PendingTick {
    row: Value,
    action: [f32; ACTION_DIMENSION],
    target: [f32; ACTION_DIMENSION],
}

struct DiagnosticController {
    policy: DiagnosticPolicy,
    definition: Arc<RobotDefinition>,
    assembly: RobotAssembly,
    previous_action: [f32; ACTION_DIMENSION],
    previous_target: Option<[f32; ACTION_DIMENSION]>,
    pending: Option<PendingTick>,
    report: Value,
    published_frames: Vec<Arc<RobotPoseFrame>>,
}

impl TickController<()> for DiagnosticController {
    fn inference_count(&self) -> u64 {
        self.policy.successful_inference_count()
    }

    fn synchronize_scene(
        &mut self,
        _world: &mut SimulationWorld,
        events: Vec<TickEvent<()>>,
        _boundary: u64,
    ) -> Result<(), String> {
        if !events.is_empty() {
            return Err("diagnostic station does not admit scene events".into());
        }
        Ok(())
    }

    fn infer_and_actuate(
        &mut self,
        world: &SimulationWorld,
        boundary: u64,
    ) -> Result<Vec<BodyTorque>, String> {
        if self.pending.is_some() {
            return Err("previous diagnostic tick was not published".into());
        }
        let tick = usize::try_from(boundary).map_err(|_| "diagnostic tick exceeds usize")?;
        let (state, root_before) = native_state(world, &self.assembly, &self.definition)?;
        let command = self.policy.command(tick);
        let observation = self
            .policy
            .observation(&state, &command, &self.previous_action)?;
        let action = {
            let _span = tracing::info_span!("sim2sim_onnx_inference").entered();
            self.policy.infer(&observation)?
        };
        let raw_target = self.policy.targets(&action)?;
        let target = self
            .policy
            .applied_targets(raw_target, self.previous_target);
        // Diagnostic only: this P-only law omits BAM, delays and dry-friction loads.
        let torque: [f32; ACTION_DIMENSION] = std::array::from_fn(|i| {
            (0.55 * (target[i] - state.joint_position[i])).clamp(-0.640_523_6, 0.640_523_6)
        });
        let contributions = self
            .assembly
            .actuator_body_torques(&world.world, &self.definition, &torque)
            .map_err(|error| error.to_string())?;
        self.pending = Some(PendingTick {
            row: json!({
                "tick_before":tick,
                "observation":observation.as_slice(),
                "command":{"locomotion":command.locomotion,"head":command.head,"body":command.body},
                "action":action,
                "raw_target":raw_target,
                "target":target,
                "torque":torque,
                "joint_position":state.joint_position,
                "joint_velocity":state.joint_velocity,
                "root_before":root_before,
            }),
            action,
            target,
        });
        Ok(contributions)
    }

    fn publish(&mut self, snapshot: &StepSnapshot) -> Result<(), String> {
        let pending = self
            .pending
            .take()
            .ok_or("diagnostic tick has no matching inference")?;
        let pose = self
            .assembly
            .pose_frame(snapshot)
            .map_err(|error| error.to_string())?;
        let mut row = pending.row;
        row["root_after"] = json!(pose.poses[0].translation);
        row["contact_pairs"] = json!(snapshot.contact_pair_count);
        row["active_contact_pairs"] = json!(snapshot.active_contact_pair_count);
        self.report["trace"]
            .as_array_mut()
            .ok_or("trace report changed type")?
            .push(row);
        self.report["pose_frames"]
            .as_array_mut()
            .ok_or("pose report changed type")?
            .push(json!(pose));
        self.published_frames.push(Arc::new(pose));
        self.previous_action = pending.action;
        self.previous_target = Some(pending.target);
        self.report["integration_count"] = json!(snapshot.integration_count);
        self.report["policy_inference_count"] = json!(self.policy.successful_inference_count());
        Ok(())
    }
}

/// A real diagnostic world and policy, owned together for offline or live driving.
pub struct DiagnosticSession {
    runtime: FixedStepRuntime<()>,
    controller: DiagnosticController,
    initial_pose: Arc<RobotPoseFrame>,
    definition: Arc<RobotDefinition>,
    scene: StationScene,
    ticks: usize,
}

impl DiagnosticSession {
    pub fn definition(&self) -> Arc<RobotDefinition> {
        self.definition.clone()
    }

    pub fn scene(&self) -> StationScene {
        self.scene.clone()
    }

    pub fn initial_pose(&self) -> Arc<RobotPoseFrame> {
        self.initial_pose.clone()
    }

    pub fn ticks(&self) -> usize {
        self.ticks
    }

    pub fn global_step(&self) -> u64 {
        self.runtime.clock_snapshot().global_step
    }

    pub fn pending_ticks(&self) -> u128 {
        self.runtime.clock_snapshot().pending_ticks
    }

    pub fn enable_rapier_counters(&mut self) {
        self.runtime.enable_rapier_counters();
    }

    pub fn rapier_counter_sample(&self) -> simulation_minigame::RapierCounterSample {
        self.runtime.rapier_counter_sample()
    }

    pub fn advance_frame(&mut self, elapsed: Duration) -> Result<FrameOutcome, String> {
        self.runtime
            .advance_frame(elapsed, &mut self.controller)
            .map_err(|error| error.to_string())
    }

    /// Every completed tick remains in the report; this is only the display handoff.
    pub fn take_published_frames(&mut self) -> Vec<Arc<RobotPoseFrame>> {
        std::mem::take(&mut self.controller.published_frames)
    }

    pub fn report(&self) -> &Value {
        &self.controller.report
    }

    pub fn report_mut(&mut self) -> &mut Value {
        &mut self.controller.report
    }

    pub fn finish(&mut self) -> bool {
        self.finish_expected_steps(self.ticks as u64)
    }

    /// Live display can finish just after its requested bound when a frame is
    /// catching up several already-owed ticks.
    pub fn finish_expected_steps(&mut self, expected_steps: u64) -> bool {
        let final_snapshot = self.runtime.world_snapshot();
        self.controller.report["final_snapshot"] = json!(final_snapshot);
        self.controller.report["fixed_step_driver"] = json!("FixedStepRuntime");
        self.controller.report["final_clock"] = json!(self.runtime.clock_snapshot());
        let passed = self.controller.policy.successful_inference_count() == expected_steps
            && final_snapshot.integration_count == expected_steps
            && self.runtime.clock_snapshot().global_step == expected_steps
            && self.controller.report["trace"]
                .as_array()
                .is_some_and(|rows| rows.len() as u64 == expected_steps)
            && self.controller.report["pose_frames"]
                .as_array()
                .is_some_and(|rows| rows.len() as u64 == expected_steps + 1);
        self.controller.report["passed"] = json!(passed);
        passed
    }

    pub fn into_report(self) -> Value {
        self.controller.report
    }
}

/// Create the same one-world diagnostic session used by both entry points.
pub fn prepare(arguments: &[String], report: &mut Value) -> Result<DiagnosticSession, String> {
    if arguments.len() < 9 {
        return Err("diagnostic session requires nine model and runtime arguments".into());
    }
    let model_bytes = fs::read(&arguments[0]).map_err(|error| error.to_string())?;
    if digest(&model_bytes) != arguments[1] {
        return Err("compiled robot definition SHA256 mismatch".into());
    }
    let definition = Arc::new(
        match Path::new(&arguments[0])
            .extension()
            .and_then(|v| v.to_str())
        {
            Some("json") => RobotDefinition::load_json(Path::new(&arguments[0]), &arguments[1]),
            Some("ron") => RobotDefinition::load_ron(Path::new(&arguments[0]), &arguments[1]),
            _ => return Err("model must be a compiled .json or .ron definition".into()),
        }
        .map_err(|error| error.to_string())?,
    );
    let qpos_bytes = fs::read(&arguments[2]).map_err(|error| error.to_string())?;
    if digest(&qpos_bytes) != arguments[3] {
        return Err("initial source qpos SHA256 mismatch".into());
    }
    let qpos: Vec<f64> = serde_json::from_slice(&qpos_bytes).map_err(|error| error.to_string())?;
    let assets = fs::canonicalize(&arguments[4]).map_err(|error| error.to_string())?;
    let scene = StationScene::load(&assets)?;
    report["station_identity"] = json!({
        "model_sha256":scene.0.model_sha256,
        "manifest_sha256":scene.0.manifest_sha256,
        "layout_sha256":scene.0.layout_sha256,
    });
    let policy = if arguments[6] == "--legacy-original" {
        let name = Path::new(&arguments[5])
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or("legacy ONNX name is invalid")?
            .to_owned();
        let expected_family = if name.starts_with("roller") {
            "roller_allcollisions"
        } else {
            "leg_allcollisions"
        };
        if definition.model().family != expected_family {
            return Err("legacy policy and compiled robot family mismatch".into());
        }
        let actor = LegacyCpuActor::load(Path::new(&arguments[7]), Path::new(&arguments[5]))
            .map_err(|error| error.to_string())?;
        report["policy_origin"] = json!("original_200hz_physics_50hz_policy_zero_shot_at_60_60");
        report["legacy_model_name"] = json!(name);
        report["legacy_deployment_profile"] = json!(
            "Unity scenario command, action scale and target filters at experimental 60Hz; BAM omitted"
        );
        report["policy_model_sha256"] = json!(digest(
            &fs::read(&arguments[5]).map_err(|error| error.to_string())?
        ));
        report["native_runtime"] = json!(actor.identity());
        DiagnosticPolicy::Legacy { actor, name }
    } else {
        let sidecar_bytes = fs::read(&arguments[6]).map_err(|error| error.to_string())?;
        let contract: PolicyContract =
            serde_json::from_slice(&sidecar_bytes).map_err(|error| error.to_string())?;
        if contract.robot_definition_sha256 != definition.file_sha256() {
            return Err("policy sidecar and robot definition differ".into());
        }
        let actor = CpuPolicy::load(
            Path::new(&arguments[7]),
            Path::new(&arguments[5]),
            &contract,
        )
        .map_err(|error| error.to_string())?;
        report["policy_origin"] = json!("versioned_source60_sidecar");
        report["policy_model_sha256"] = json!(contract.model_sha256);
        report["policy_sidecar_sha256"] = json!(digest(&sidecar_bytes));
        report["native_runtime"] = json!(actor.runtime_identity());
        DiagnosticPolicy::Native { actor, contract }
    };
    let ticks: usize = arguments[8].parse().map_err(|_| "invalid tick count")?;
    if !(1..=600).contains(&ticks) {
        return Err("diagnostic tick count must be 1..=600".into());
    }
    let mut simulation = SimulationWorld::new();
    report["world_count"] = json!(1);
    let imported = import_station(&mut simulation, &scene.0.colliders)?;
    let assembly = build_structure(&mut simulation.world, &definition, &qpos)
        .map_err(|error| error.to_string())?;
    report["station_static_colliders"] = json!(imported);
    report["station_mass_properties_prepared"] = json!(imported);
    report["deferred_prop_colliders"] = json!(scene.0.colliders.len() - imported);
    report["initial_counts"] = json!(simulation.counts());
    report["initial_snapshot"] = json!(simulation.snapshot());
    report["model_file_sha256"] = json!(definition.file_sha256());
    let previous_action = [0.0_f32; ACTION_DIMENSION];
    let previous_target: Option<[f32; ACTION_DIMENSION]> = None;
    let first_pose = assembly
        .pose_frame(&simulation.snapshot())
        .map_err(|error| error.to_string())?;
    report["pose_frames"] = json!([first_pose]);
    report["trace"] = json!([]);
    let runtime = FixedStepRuntime::<()>::new(simulation).map_err(|error| error.to_string())?;
    let controller = DiagnosticController {
        policy,
        definition: definition.clone(),
        assembly,
        previous_action,
        previous_target,
        pending: None,
        report: report.clone(),
        published_frames: Vec::new(),
    };
    Ok(DiagnosticSession {
        runtime,
        controller,
        initial_pose: Arc::new(first_pose),
        definition,
        scene,
        ticks,
    })
}

/// Retain the original exact rational-time offline replay and report fields.
pub fn run_offline(arguments: &[String], report: &mut Value) -> Result<(), String> {
    let mut session = prepare(arguments, report)?;
    let ticks = session.ticks();
    let result = (|| {
        let mut last_nanos = 0_u64;
        for tick in 1..=ticks {
            // Rational wall-time boundaries provide exactly one 60 Hz budget each.
            let target_nanos = (tick as u64 * 1_000_000_000).div_ceil(PHYSICS_HZ as u64);
            let elapsed = Duration::from_nanos(target_nanos - last_nanos);
            last_nanos = target_nanos;
            let outcome = session
                .advance_frame(elapsed)
                .map_err(|error| format!("fixed-step diagnostic tick {tick}: {error}"))?;
            if outcome.completed_steps != 1 || outcome.pending_ticks != 0 {
                return Err(format!(
                    "diagnostic tick {tick} did not complete exactly once"
                ));
            }
        }
        session.finish();
        Ok(())
    })();
    *report = session.into_report();
    result
}

pub fn initial_report() -> Value {
    json!({
        "schema":"station_robot_60hz_diagnostic_v1",
        "scope":"single_rapier_world_native_ort_each_60hz_step_p_only_diagnostic_only",
        "physics_hz":PHYSICS_HZ,
        "policy_hz":PHYSICS_HZ,
        "control_law":"diagnostic_p_only_kp_0.55_kd_0_clamp_0.6405236_not_BAM",
        "station_contact_qualified":false,
        "source_self_contact_filter_qualified":false,
        "bam_external_load_qualified":false,
        "delay_model_qualified":false,
        "skill_qualified":false,
        "passed":false,
        "world_count":0,
        "integration_count":0,
        "policy_inference_count":0,
        "error":Value::Null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_station_mass_preparation_preserves_native_properties_and_clock() {
        let vertices: Vec<[f32; 3]> = [-1.0, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(move |y| [-1.0, 1.0].into_iter().map(move |z| [x, y, z]))
            })
            .collect();
        let source = StationCollider {
            owner: "station".into(),
            tag: "immutable mass preparation regression".into(),
            shape: StationCollisionShape::Convex {
                vertices: vertices.clone(),
            },
        };
        let mut prepared = SimulationWorld::new();
        assert_eq!(import_station(&mut prepared, &[source]).unwrap(), 1);
        let mut original = SimulationWorld::new();
        let fixed = original.world.bodies.insert(RigidBodyBuilder::fixed());
        let points: Vec<Vector> = vertices.into_iter().map(Vector::from_array).collect();
        original.world.colliders.insert_with_parent(
            ColliderBuilder::convex_hull(&points).unwrap(),
            fixed,
            &mut original.world.bodies,
        );
        assert_eq!(
            serde_json::to_value(prepared.snapshot()).unwrap(),
            serde_json::to_value(original.snapshot()).unwrap()
        );
        let prepared_collider = prepared.world.colliders.iter().next().unwrap().1;
        let original_collider = original.world.colliders.iter().next().unwrap().1;
        assert_eq!(
            prepared_collider.mass_properties(),
            original_collider.mass_properties()
        );
        let prepared_step = prepared.step_with_torques(&[]).unwrap();
        let original_step = original.step_with_torques(&[]).unwrap();
        assert_eq!(
            serde_json::to_value(prepared_step).unwrap(),
            serde_json::to_value(original_step).unwrap()
        );
        assert_eq!(
            prepared.world.bodies[fixed].mass_properties().local_mprops,
            original.world.bodies[fixed].mass_properties().local_mprops
        );
    }
}
