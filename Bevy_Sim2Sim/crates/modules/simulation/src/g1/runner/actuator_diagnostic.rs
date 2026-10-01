//! Ignored, bounded actuator-backend experiment; never a game qualification.
//! Both worlds and Homie histories are cold rebuilt. The byte-bound ORT library
//! remains shared in this process. No source frequency, gain, or guard is changed.
//!
//! Opt-in entry: `cargo test -p simulation_minigame --features sim2sim_motor_row_trace
//! g1::runner::actuator_diagnostic::real_homie_actuator_backend_diagnostic --
//! --ignored --exact --test-threads=1 --nocapture`. The caller must provide frozen
//! `G1_MODEL_DIR`, `G1_DEFINITION_SHA256`, `G1_ORT`, `G1_ORT_SHA256`, a full
//! `G1_CODE_COMMIT`, and a new `G1_ACTUATOR_DIAGNOSTIC_OUTPUT` file in an existing
//! evidence directory. External preparation must bind the executable hash too.

use super::*;
use rapier3d::{
    dynamics::{MotorRowTracePhase, MultibodyObservation},
    math::{Pose, Rotation, Vector},
};
use robot_minigame::{
    basis::{source_to_engine_rotation, source_to_engine_vector},
    g1::{
        contract::{G1State, JOINT_COUNT, JOINT_NAMES},
        definition::{ARENA_COMMIT, JointKind, USD_SHA256},
        policy::{STAND_SHA256, WALK_SHA256},
    },
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    env,
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
};

const TICKS: u64 = 150;
const RAW_TICKS: u64 = 20;
const LIMIT_TOLERANCE: f64 = 0.001;
type Mapping = [(MultibodyJointHandle, usize); JOINT_COUNT];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    ExternalExplicitPd,
    NativeForceBased,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::ExternalExplicitPd => "external_explicit_pd_50hz",
            Self::NativeForceBased => "homie_v2_rapier_force_based_50hz_diagnostic",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct FirstInference {
    action: Vec<f32>,
    targets: Vec<f32>,
    action_bits: Vec<u32>,
    target_bits: Vec<u32>,
}

impl FirstInference {
    fn new(inference: &HomieResult) -> Self {
        Self {
            action: inference.action.to_vec(),
            targets: inference.targets.clone(),
            action_bits: bits(&inference.action),
            target_bits: bits(&inference.targets),
        }
    }
    fn identical(&self, other: &Self) -> bool {
        self.action_bits == other.action_bits && self.target_bits == other.target_bits
    }
}

#[derive(Serialize)]
struct BackendReceipt {
    backend: &'static str,
    qualified: bool,
    termination: String,
    error: Option<String>,
    initial_frame: Option<Value>,
    initial_state: Option<Value>,
    initial_counts: Option<G1ProgressCounts>,
    joint_audit: Vec<Value>,
    first_inference: Option<FirstInference>,
    first_tick_matches_baseline: Option<bool>,
    first_limit_violation: Option<Value>,
    completed_without_fall: bool,
    max_horizontal_drift_m: f32,
    final_counts: Option<G1ProgressCounts>,
    rows: Vec<Value>,
}

impl BackendReceipt {
    fn new(backend: Backend) -> Self {
        Self {
            backend: backend.label(),
            qualified: false,
            termination: "initializing".into(),
            error: None,
            initial_frame: None,
            initial_state: None,
            initial_counts: None,
            joint_audit: Vec::new(),
            first_inference: None,
            first_tick_matches_baseline: None,
            first_limit_violation: None,
            completed_without_fall: false,
            max_horizontal_drift_m: 0.0,
            final_counts: None,
            rows: Vec::with_capacity(TICKS as usize),
        }
    }
}

#[derive(Default, Serialize)]
struct Attempt {
    stage: &'static str,
    guard_calls: u32,
    pre_state: Option<Value>,
    inference: Option<HomieResult>,
    /// This reference is never applied on the motor backend.
    explicit_pd_reference: Option<Vec<f32>>,
    actual_external_pd_torques: Option<Vec<f32>>,
    external_body_torque_contribution_count: usize,
    completed: Option<Value>,
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn state_receipt(state: &G1State) -> Value {
    json!({"joint_positions":state.positions.to_vec(), "joint_velocities":state.velocities.to_vec(),
        "joint_position_bits":bits(&state.positions), "joint_velocity_bits":bits(&state.velocities),
        "root_rotation_wxyz":state.root_rotation_wxyz,
        "root_rotation_bits":bits(&state.root_rotation_wxyz),
        "root_angular_velocity_body":state.root_angular_velocity_body,
        "root_angular_velocity_bits":bits(&state.root_angular_velocity_body)})
}

fn source_pose(pose: &SourcePose) -> Result<Pose, RobotError> {
    Ok(Pose::from_parts(
        Vector::from_array(source_to_engine_vector(
            pose.position.map(|value| value as f32),
        )),
        Rotation::from_array(source_to_engine_rotation(
            pose.rotation_wxyz.map(|value| value as f32),
        )?),
    ))
}

fn same_pose(left: Pose, right: Pose) -> bool {
    bits(&left.translation.to_array()) == bits(&right.translation.to_array())
        && bits(&left.rotation.to_array()) == bits(&right.rotation.to_array())
}

fn configuration_guard(runner: &G1Runner) -> Result<(), RobotError> {
    let config = runner.simulation.configuration();
    let params = &runner.simulation.world.integration_parameters;
    if config.physics_hz != 50
        || config.dt.to_bits() != (1.0_f32 / 50.0).to_bits()
        || config.num_solver_iterations != 1
        || config.max_ccd_substeps != 1
        || config.additional_solver_iterations_max != 0
        || params.num_internal_pgs_iterations != 1
        || params.num_internal_stabilization_iterations != 1
    {
        return Err(error(
            "actuator diagnostic requires unchanged 50Hz, one integration, one PGS and one stabilization",
        ));
    }
    Ok(())
}

/// Names, source frames/limits, native DOFs and actuator parameters are checked
/// through generation-bound assembly handles, never inferred from body order.
fn audit_joints(
    runner: &G1Runner,
    definition: &G1Definition,
    mapping: &Mapping,
    backend: Backend,
    targets: Option<&[f32]>,
) -> Result<Vec<Value>, RobotError> {
    configuration_guard(runner)?;
    if runner.assembly.definition_sha256() != definition.file_sha256()
        || definition.driven_joints().len() != JOINT_COUNT
        || runner.assembly.body_count() != 53
    {
        return Err(error("frozen definition/assembly identity mismatch"));
    }
    let state = runner.assembly.state(&runner.simulation.world)?;
    let mut slots = HashSet::new();
    let mut names = HashSet::new();
    let mut audit = Vec::with_capacity(JOINT_COUNT);
    for (wbc, &(handle, slot)) in mapping.iter().enumerate() {
        let source = &definition.model().joints[definition.driven_joints()[wbc]];
        if source.name != JOINT_NAMES[wbc]
            || source.kind != JointKind::Revolute
            || !names.insert(source.name.as_str())
            || !(6..49).contains(&slot)
            || !slots.insert(slot)
        {
            return Err(error(format!(
                "named motor mapping mismatch: {}",
                JOINT_NAMES[wbc]
            )));
        }
        let (tree, link_id) = runner
            .simulation
            .world
            .multibody_joints
            .get(handle)
            .ok_or_else(|| error("stale generation-bound driven handle"))?;
        let link = tree
            .link(link_id)
            .ok_or_else(|| error("missing driven link"))?;
        let joint = &link.joint().data;
        if tree.ndofs() != 49
            || tree.links().count() != 53
            || tree.root().rigid_body_handle() != runner.assembly.root_handle()
            || tree.root().joint().ndofs() != 6
            || !tree.root().joint().data.locked_axes.is_empty()
            || !tree.root().joint().data.motor_axes.is_empty()
            || link.assembly_id() != slot
            || link.joint().ndofs() != 1
            || joint.locked_axes != JointAxesMask::LOCKED_REVOLUTE_AXES
            || !joint.limit_axes.contains(JointAxesMask::ANG_X)
            || joint.limits[3].min.to_bits() != (source.limits[0] as f32).to_bits()
            || joint.limits[3].max.to_bits() != (source.limits[1] as f32).to_bits()
            || link.joint().coords()[3].to_bits() != state.positions[wbc].to_bits()
            || tree.generalized_velocity()[slot].to_bits() != state.velocities[wbc].to_bits()
        {
            return Err(error(format!(
                "native/source DOF or limit mismatch: {}",
                source.name
            )));
        }
        let axis_source = match source.axis.as_str() {
            "X" => [1., 0., 0.],
            "Y" => [0., 1., 0.],
            "Z" => [0., 0., 1.],
            _ => return Err(error("unknown frozen source joint axis")),
        };
        let axis = Vector::from_array(source_to_engine_vector(axis_source));
        let alignment =
            Pose::from_parts(Vector::ZERO, Rotation::from_rotation_arc(Vector::X, axis));
        if !same_pose(
            joint.local_frame1,
            source_pose(&source.frame_parent)? * alignment,
        ) || !same_pose(
            joint.local_frame2,
            source_pose(&source.frame_child)? * alignment,
        ) {
            return Err(error(format!(
                "named source joint frame mismatch: {}",
                source.name
            )));
        }
        let p = actuator::parameters(wbc)?;
        if tree.armature()[slot].to_bits() != p.armature.to_bits()
            || tree.armature().as_slice()[..6]
                .iter()
                .any(|value| *value != 0.0)
            || tree.damping().iter().any(|value| *value != 0.0)
            || tree.frictions().iter().any(|value| *value != 0.0)
            || tree
                .links()
                .any(|item| (0..6).any(|axis| item.joint().spring(axis).0 != 0.0))
        {
            return Err(error("armature or passive-force configuration changed"));
        }
        let motor = joint.motor(JointAxis::AngX);
        match backend {
            Backend::ExternalExplicitPd if !joint.motor_axes.is_empty() => {
                return Err(error("explicit backend has an active native motor"));
            }
            Backend::ExternalExplicitPd => {}
            Backend::NativeForceBased => {
                let motor =
                    motor.ok_or_else(|| error("native backend is missing a driven motor"))?;
                if joint.motor_axes != JointAxesMask::ANG_X
                    || motor.model != MotorModel::ForceBased
                    || motor.stiffness.to_bits() != p.stiffness.to_bits()
                    || motor.damping.to_bits() != p.damping.to_bits()
                    || motor.max_force.to_bits() != p.effort_limit.to_bits()
                    || motor.target_vel != 0.0
                    || targets
                        .is_some_and(|target| motor.target_pos.to_bits() != target[wbc].to_bits())
                {
                    return Err(error(format!(
                        "native source motor parameter mismatch: {}",
                        source.name
                    )));
                }
            }
        }
        audit.push(json!({"name":source.name, "wbc_index":wbc,"backend_dof":slot,
            "handle_index_generation":handle.into_raw_parts(),"native_link_id":link_id,
            "child_body_index_generation":link.rigid_body_handle().into_raw_parts(),
            "source_child_name":definition.model().bodies[source.child].name,
            "source_axis":source.axis,"source_limits":source.limits,"parameters":p,
            "native_motor":motor.map(|motor| json!({"force_based":motor.model == MotorModel::ForceBased,
                "target":motor.target_pos,"target_velocity":motor.target_vel,
                "stiffness":motor.stiffness,"damping":motor.damping,"max_force":motor.max_force}))}));
    }
    // No unowned fixed-link motors or additional actuation axes may be present.
    let (tree, _) = runner
        .simulation
        .world
        .multibody_joints
        .get(mapping[0].0)
        .ok_or_else(|| error("missing articulation"))?;
    let enabled_axes: u32 = tree
        .links()
        .map(|link| link.joint().data.motor_axes.bits().count_ones())
        .sum();
    let expected = if backend == Backend::NativeForceBased {
        43
    } else {
        0
    };
    if enabled_axes != expected {
        return Err(error("motor count differs from the 43 named source joints"));
    }
    Ok(audit)
}

fn set_motor_targets(
    runner: &mut G1Runner,
    mapping: &Mapping,
    targets: &[f32],
) -> Result<(), RobotError> {
    if targets.len() != JOINT_COUNT || targets.iter().any(|value| !value.is_finite()) {
        return Err(error(
            "native motor target contract must be 43 finite radians",
        ));
    }
    for (wbc, &(handle, slot)) in mapping.iter().enumerate() {
        let p = actuator::parameters(wbc)?;
        let (tree, link_id) = runner
            .simulation
            .world
            .multibody_joints
            .get_mut(handle)
            .ok_or_else(|| error("stale native motor handle"))?;
        let link = tree
            .link_mut(link_id)
            .ok_or_else(|| error("missing native motor link"))?;
        if link.assembly_id() != slot {
            return Err(error("native motor slot changed"));
        }
        link.joint
            .data
            .set_motor_model(JointAxis::AngX, MotorModel::ForceBased)
            .set_motor_position(JointAxis::AngX, targets[wbc], p.stiffness, p.damping)
            .set_motor_max_force(JointAxis::AngX, p.effort_limit);
    }
    Ok(())
}

/// This test-only copy preserves the existing runner's two guard boundaries,
/// state/policy validation, torque validation, world guards and halt-on-error.
/// It exposes accepted inference even if a later state/guard rejects the Tick.
fn step_backend(
    runner: &mut G1Runner,
    definition: &G1Definition,
    mapping: &Mapping,
    backend: Backend,
    command: &G1Command,
    first_expected: Option<&FirstInference>,
    guard: &mut impl FnMut() -> Result<(), RobotError>,
    attempt: &mut Attempt,
) -> Result<(), RobotError> {
    let result = (|| {
        attempt.stage = "legacy_halt_guard";
        if runner.halted {
            return Err(error("G1 runner halted after a failed boundary"));
        }
        attempt.guard_calls += 1;
        attempt.stage = "legacy_pre_inference_guard";
        guard()?;
        attempt.stage = "clock_contract";
        configuration_guard(runner)?;
        attempt.stage = "legacy_observation_guard";
        let state = runner.assembly.state(&runner.simulation.world)?;
        attempt.pre_state = Some(state_receipt(&state));
        attempt.stage = "legacy_inference_guard";
        let inference = runner.policy.infer(&state, command)?;
        attempt.inference = Some(inference.clone());
        attempt.guard_calls += 1;
        attempt.stage = "legacy_post_inference_guard";
        guard()?;
        attempt.stage = "clock_contract";
        configuration_guard(runner)?;
        attempt.stage = "first_inference_contract";
        if inference.inference_count == 1
            && first_expected
                .is_some_and(|first| !first.identical(&FirstInference::new(&inference)))
        {
            return Err(error(
                "first raw action/targets differ from the cold explicit baseline",
            ));
        }
        let pd = actuator::torques(&inference.targets, &state.positions, &state.velocities)?;
        attempt.explicit_pd_reference = Some(pd.to_vec());
        let snapshot = match backend {
            Backend::ExternalExplicitPd => {
                attempt.stage = "actuator_contract";
                audit_joints(runner, definition, mapping, backend, None)?;
                attempt.stage = "legacy_external_torque_guard";
                let torques = runner.assembly.torques(&runner.simulation.world, &pd)?;
                attempt.actual_external_pd_torques = Some(pd.to_vec());
                attempt.external_body_torque_contribution_count = torques.len();
                attempt.stage = "legacy_world_guards";
                runner
                    .simulation
                    .step_with_torques(&torques)
                    .map_err(error)?
            }
            Backend::NativeForceBased => {
                attempt.stage = "actuator_contract";
                set_motor_targets(runner, mapping, &inference.targets)?;
                audit_joints(
                    runner,
                    definition,
                    mapping,
                    backend,
                    Some(&inference.targets),
                )?;
                // This existing boundary clears every old user torque before
                // its sole integration. No explicit PD body torque is supplied.
                attempt.stage = "legacy_world_guards";
                runner.simulation.step_with_torques(&[]).map_err(error)?
            }
        };
        attempt.stage = "counter_contract";
        if snapshot.integration_count != inference.inference_count
            || snapshot.torque_update_count != inference.inference_count
        {
            return Err(error("G1 policy/command-boundary/physics count mismatch"));
        }
        attempt.stage = "legacy_completed_state_guard";
        let completed = runner.assembly.state(&runner.simulation.world)?;
        let root = &runner.simulation.world.bodies[runner.assembly.root_handle()];
        attempt.stage = "legacy_completed_frame_guard";
        attempt.completed = Some(json!({"frame":runner.completed_frame(&snapshot)?,
            "state":state_receipt(&completed), "root_position_source":engine_to_source_vector(root.translation().to_array()),
            "root_velocity_source":engine_to_source_vector(runner.assembly.current_root_velocity(&runner.simulation.world)?.linvel.to_array()),
            "root_upright_cosine":(*root.rotation() * Vector::Y).y,
            "active_contact_pairs":snapshot.active_contact_pair_count,"step_configuration":runner.simulation.configuration()}));
        attempt.stage = "completed";
        Ok(())
    })();
    if result.is_err() {
        runner.halted = true;
    }
    result
}

fn trace_receipt(observation: &MultibodyObservation, mapping: &Mapping) -> Value {
    let trace = &observation.motor_row_trace;
    let a = &trace.aggregate;
    let raw: Vec<_> = trace.raw_rows.iter().map(|row| {
        let wbc = mapping.iter().position(|(_, slot)| *slot == row.backend_dof);
        json!({"joint_name":wbc.map(|index| JOINT_NAMES[index]),"wbc_index":wbc,"phase":row.phase.as_str(),
        "substep_id":row.substep_id,"row_index":row.row_index,"backend_dof":row.backend_dof,
        "joint_local_dof":row.joint_local_dof,"coordinate":row.coordinate,"generalized_velocity":row.generalized_velocity,
        "rhs":row.rhs,"rhs_without_bias":row.rhs_without_bias,"impulse":row.impulse,"impulse_bounds":row.impulse_bounds,
        "gamma":row.gamma,"inv_lhs":row.inv_lhs,"target_position":row.target_position,
        "stiffness":row.stiffness,"damping":row.damping,"max_force":row.max_force,"force_based":row.force_based,
        "equivalent_boundary_mean_torque":row.equivalent_mean_torque})}).collect();
    json!({"session_tick":trace.session_tick,"active":trace.active,"dt":trace.dt,"raw_rows":raw,
        "aggregate":{"rows_per_phase":a.rows_per_phase,"max_abs_impulse":a.max_abs_impulse,
            "max_abs_equivalent_mean_torque":a.max_abs_equivalent_mean_torque,"cap_violation":a.cap_violation,
            "nonfinite":a.nonfinite,"ownership_incomplete":a.ownership_incomplete,"overflow":a.overflow,
            "temporal_step_invalid":a.temporal_step_invalid},
        "observation_epoch":observation.epoch,"ordinary_observation_valid":observation.valid,
        "ordinary_contact_coverage":observation.contact_coverage,"energy_guard_evaluated":observation.energy_guard_evaluated,
        "energy_guard_fallback":observation.energy_guard_fallback,"energy_guard_acceleration_cleared":observation.energy_guard_acceleration_cleared,
        "user_force_projection":observation.user_force_projection})
}

fn validate_trace(
    observation: &MultibodyObservation,
    mapping: &Mapping,
    backend: Backend,
    tick: u64,
    inference: Option<&HomieResult>,
) -> Result<(), RobotError> {
    let t = &observation.motor_row_trace;
    let a = &t.aggregate;
    if !t.active
        || t.session_tick as u64 != tick
        || t.dt.to_bits() != (1.0_f32 / 50.0).to_bits()
        || a.nonfinite
        || a.cap_violation
        || a.ownership_incomplete
        || a.overflow
        || a.temporal_step_invalid
    {
        return Err(error("invalid bounded motor-row instrumentation receipt"));
    }
    let expected_counts = if backend == Backend::NativeForceBased {
        [43; 3]
    } else {
        [0; 3]
    };
    if a.rows_per_phase != expected_counts {
        return Err(error("motor-row phase coverage mismatch"));
    }
    if backend == Backend::ExternalExplicitPd {
        if !t.raw_rows.is_empty() {
            return Err(error("explicit backend unexpectedly has motor rows"));
        }
        return Ok(());
    }
    if observation
        .user_force_projection
        .iter()
        .any(|value| *value != 0.0)
    {
        return Err(error(
            "ForceBased world contains a projected external user force/torque",
        ));
    }
    if tick > RAW_TICKS {
        if !t.raw_rows.is_empty() {
            return Err(error("raw motor rows exceeded the 20 Tick budget"));
        }
        return Ok(());
    }
    let inference =
        inference.ok_or_else(|| error("motor rows have no accepted inference identity"))?;
    if t.raw_rows.len() != 129 {
        return Err(error("missing raw motor rows in one of the three phases"));
    }
    for phase in [
        MotorRowTracePhase::AfterBiasedSolve,
        MotorRowTracePhase::AfterPositionIntegration,
        MotorRowTracePhase::AfterUnbiasedSolve,
    ] {
        let mut seen = HashSet::new();
        for row in t.raw_rows.iter().filter(|row| row.phase == phase) {
            let wbc = mapping
                .iter()
                .position(|(_, slot)| *slot == row.backend_dof)
                .ok_or_else(|| error("motor row is outside named source DOFs"))?;
            let p = actuator::parameters(wbc)?;
            let gamma = 1.0 / (t.dt * t.dt * p.stiffness + t.dt * p.damping);
            if !seen.insert(row.backend_dof)
                || row.substep_id != 0
                || row.joint_local_dof != 0
                || !row.force_based
                || row.stiffness.to_bits() != p.stiffness.to_bits()
                || row.damping.to_bits() != p.damping.to_bits()
                || row.max_force.to_bits() != p.effort_limit.to_bits()
                || row.target_position.to_bits() != inference.targets[wbc].to_bits()
                || row.impulse_bounds != [-p.effort_limit * t.dt, p.effort_limit * t.dt]
                || row.gamma.to_bits() != gamma.to_bits()
                || row.inv_lhs <= 0.0
                || row.impulse < row.impulse_bounds[0]
                || row.impulse > row.impulse_bounds[1]
            {
                return Err(error(format!(
                    "named raw motor row mismatch: {} / {}",
                    JOINT_NAMES[wbc],
                    phase.as_str()
                )));
            }
        }
        if seen.len() != JOINT_COUNT {
            return Err(error("raw phase does not cover all 43 named motors"));
        }
    }
    Ok(())
}

fn final_native_receipt(runner: &G1Runner, definition: &G1Definition, mapping: &Mapping) -> Value {
    let root = runner
        .simulation
        .world
        .bodies
        .get(runner.assembly.root_handle());
    let max_user_torque = runner
        .simulation
        .world
        .bodies
        .iter()
        .map(|(_, body)| body.user_torque().length())
        .fold(0.0_f32, f32::max);
    // Diagnostic operands only, never fed back to inference. In particular a
    // quaternion guard rejection does not erase the actual joint-limit evidence.
    let raw_joints: Vec<_> = mapping
        .iter()
        .enumerate()
        .map(|(wbc, &(handle, slot))| {
            let native = runner.simulation.world.multibody_joints.get(handle);
            let q =
                native.and_then(|(tree, id)| tree.link(id).map(|link| link.joint().coords()[3]));
            let dq = native.and_then(|(tree, _)| tree.generalized_velocity().get(slot).copied());
            json!({"name":JOINT_NAMES[wbc],"backend_dof":slot,"position":q,"velocity":dq})
        })
        .collect();
    // Preserve the actual rejected boundary without normalizing its quaternions
    // or manufacturing a valid frame. FK operands distinguish solver storage
    // from the immutable snapshot copy; this data never enters control.
    let snapshot = runner.simulation.snapshot();
    let tree = runner.simulation.world.multibody_joints.get(mapping[0].0);
    let raw_bodies: Vec<_> = runner
        .assembly
        .body_trace_handles()
        .iter()
        .enumerate()
        .map(|(source_body, handle)| {
            let body = runner.simulation.world.bodies.get(*handle);
            let raw_handle = handle.into_raw_parts();
            let sample = snapshot.bodies.iter().find(|sample| {
                sample.handle == [raw_handle.0, raw_handle.1]
            });
            let link = tree.and_then(|(tree, _)| {
                tree.links().find(|link| link.rigid_body_handle() == *handle)
            });
            let rotation = body.map(|body| body.rotation().to_array());
            let norm_error = rotation.map(|rotation| {
                rotation.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() - 1.0
            });
            json!({"source_body":source_body,"name":definition.model().bodies[source_body].name,
                "rigid_body_rotation_xyzw":rotation,"squared_norm_error_f64":norm_error,
                "snapshot_rotation_xyzw":sample.map(|sample| sample.rotation_xyzw),
                "snapshot_matches_storage_bits":rotation.zip(sample).map(|(q,sample)| bits(&q)==bits(&sample.rotation_xyzw)),
                "link_id":link.map(|link|link.link_id()),"parent_link_id":link.and_then(|link|link.parent_id()),
                "fk_world_rotation_xyzw":link.map(|link|link.local_to_world().rotation.to_array()),
                "fk_parent_rotation_xyzw":link.map(|link|link.local_to_parent().rotation.to_array()),
                "joint_rotation_xyzw":link.map(|link|link.joint().joint_rot().to_array()),
                "joint_frame1_rotation_xyzw":link.map(|link|link.joint().data.local_frame1.rotation.to_array()),
                "joint_frame2_rotation_xyzw":link.map(|link|link.joint().data.local_frame2.rotation.to_array())})
        })
        .collect();
    json!({"actual_counts":runner.progress_counts(),"configuration":runner.simulation.configuration(),
        "root_handle_still_live":root.is_some(),
        "raw_root_position_source":root.map(|body| engine_to_source_vector(body.translation().to_array())),
        "raw_root_rotation_xyzw":root.map(|body| body.rotation().to_array()),
        "raw_root_upright_cosine":root.map(|body| (*body.rotation() * Vector::Y).y),
        "max_body_user_torque_after_step":max_user_torque,
        "raw_joint_operands_not_control_observations":raw_joints,
        "raw_body_rotation_operands_not_control_observations":raw_bodies})
}

fn run_backend(
    config: &G1RunnerConfig,
    definition: &G1Definition,
    backend: Backend,
    baseline: Option<&BackendReceipt>,
    mut checkpoint: impl FnMut(&BackendReceipt) -> std::io::Result<()>,
) -> std::io::Result<BackendReceipt> {
    let mut report = BackendReceipt::new(backend);
    let mut runner = match G1Runner::load(config) {
        Ok(runner) => runner,
        Err(e) => {
            report.termination = "load_error".into();
            report.error = Some(e.to_string());
            checkpoint(&report)?;
            return Ok(report);
        }
    };
    let mut mapping = None;
    let initialized = (|| {
        let actual = runner
            .assembly
            .driven_motor_trace_pairs()
            .ok_or_else(|| error("expected exactly 43 generation-bound mappings"))?;
        let state = runner.assembly.state(&runner.simulation.world)?;
        report.initial_frame = Some(json!(runner.initial_frame()?));
        report.initial_state = Some(state_receipt(&state));
        report.initial_counts = Some(runner.progress_counts());
        if runner.counts().bodies != 54
            || runner.counts().colliders != 53
            || runner.counts().impulse_joints != 0
            || runner.counts().multibody_joint_handles != 52
            || runner.progress_counts().integration_count != 0
            || runner.policy.inference_count() != 0
            || runner.policy.inference_attempt_count() != 0
        {
            return Err(error("world/policy was not cold initialized"));
        }
        if let Some(baseline) = baseline {
            if report.initial_frame != baseline.initial_frame
                || report.initial_state != baseline.initial_state
                || report.initial_counts != baseline.initial_counts
                || baseline.first_inference.is_none()
            {
                return Err(error(
                    "cold initial inputs or baseline first inference are unavailable/different",
                ));
            }
        }
        // Audit the inherited explicit state before enabling any motor.
        audit_joints(
            &runner,
            definition,
            &actual,
            Backend::ExternalExplicitPd,
            None,
        )?;
        if backend == Backend::NativeForceBased {
            set_motor_targets(&mut runner, &actual, &state.positions)?;
        }
        report.joint_audit = audit_joints(
            &runner,
            definition,
            &actual,
            backend,
            Some(&state.positions),
        )?;
        if !runner
            .simulation
            .world
            .multibody_joints
            .sim2sim_start_motor_row_trace(actual[0].0)
        {
            return Err(error(
                "motor instrumentation requires a supported single serial session",
            ));
        }
        mapping = Some(actual);
        Ok(())
    })();
    if let Err(e) = initialized {
        runner.halted = true;
        report.termination = "initial_contract_error".into();
        report.error = Some(e.to_string());
        report.final_counts = Some(runner.progress_counts());
        checkpoint(&report)?;
        return Ok(report);
    }
    let mapping =
        mapping.ok_or_else(|| std::io::Error::other("validated mapping was not installed"))?;
    report.termination = "running".into();
    checkpoint(&report)?;
    for tick in 1..=TICKS {
        let mut attempt = Attempt::default();
        let expected_first = baseline.and_then(|report| report.first_inference.as_ref());
        let step_result = step_backend(
            &mut runner,
            definition,
            &mapping,
            backend,
            &G1Command::default(),
            expected_first,
            &mut || Ok(()),
            &mut attempt,
        );
        if tick == 1 {
            report.first_inference = attempt.inference.as_ref().map(FirstInference::new);
            report.first_tick_matches_baseline = expected_first.map(|expected| {
                report
                    .first_inference
                    .as_ref()
                    .is_some_and(|first| first.identical(expected))
            });
        }
        let counts = runner.progress_counts();
        let observation = runner
            .simulation
            .world
            .multibody_joints
            .get(mapping[0].0)
            .and_then(|(tree, _)| tree.sim2sim_motor_trace_observation());
        let trace = observation.map(|obs| trace_receipt(obs, &mapping));
        let trace_result = if counts.integration_count == tick {
            observation
                .ok_or_else(|| error("missing motor trace session"))
                .and_then(|obs| {
                    validate_trace(obs, &mapping, backend, tick, attempt.inference.as_ref())
                })
        } else {
            Ok(())
        };
        let native = final_native_receipt(&runner, definition, &mapping);
        let max_user_torque = runner
            .simulation
            .world
            .bodies
            .iter()
            .map(|(_, body)| body.user_torque().length())
            .fold(0.0_f32, f32::max);
        let counter_error = step_result.is_ok()
            && (counts.integration_count != tick
                || counts.torque_update_count != tick
                || counts.successful_inference_count != tick
                || counts.inference_attempt_count != tick);
        let external_leak = backend == Backend::NativeForceBased && max_user_torque != 0.0;
        if report.first_limit_violation.is_none() && counts.integration_count == tick {
            if let Some(q) = native["raw_joint_operands_not_control_observations"].as_array() {
                for (wbc, &source_id) in definition.driven_joints().iter().enumerate() {
                    let source = &definition.model().joints[source_id];
                    if let Some(value) = q[wbc]["position"].as_f64() {
                        if value < source.limits[0] - LIMIT_TOLERANCE
                            || value > source.limits[1] + LIMIT_TOLERANCE
                        {
                            report.first_limit_violation = Some(
                                json!({"tick":tick,"joint":source.name,"position":value,"limits":source.limits,"tolerance_rad":LIMIT_TOLERANCE}),
                            );
                            break;
                        }
                    }
                }
            }
        }
        let root = runner
            .simulation
            .world
            .bodies
            .get(runner.assembly.root_handle());
        if let Some(root) = root {
            let root_position = engine_to_source_vector(root.translation().to_array());
            let dx = root_position[0] - config.root_pose.position[0] as f32;
            let dy = root_position[1] - config.root_pose.position[1] as f32;
            report.max_horizontal_drift_m = report.max_horizontal_drift_m.max(dx.hypot(dy));
        }
        let fell = root.is_some_and(|root| {
            engine_to_source_vector(root.translation().to_array())[2] < 0.35
                || (*root.rotation() * Vector::Y).y < 0.5
        });
        let step_error = step_result.err().map(|e| e.to_string());
        let failure_stage = attempt.stage;
        let trace_error = trace_result.err().map(|e| e.to_string());
        report.rows.push(json!({"attempted_tick":tick,"attempt":attempt,"native":native,"motor_trace":trace,
            "new_integration_occurred_this_attempt":counts.integration_count == tick,
            "boundary_error":step_error.as_deref(),"motor_trace_error":trace_error.as_deref(),"counter_error":counter_error,"external_pd_leak":external_leak}));
        report.final_counts = Some(counts);
        if let Some(message) = step_error {
            report.termination = if failure_stage.ends_with("_contract") {
                "backend_protocol_error"
            } else {
                "boundary_guard_error"
            }
            .into();
            report.error = Some(message);
        } else if let Some(message) = trace_error {
            runner.halted = true;
            report.termination = "motor_trace_contract_error".into();
            report.error = Some(message);
        } else if counter_error || external_leak {
            runner.halted = true;
            report.termination = "backend_protocol_error".into();
            report.error = Some("actual counters or external PD isolation failed".into());
        } else if fell {
            report.termination = "fell".into();
        } else if tick == TICKS {
            report.termination = "completed_150_ticks".into();
            report.completed_without_fall = true;
        }
        report.final_counts = Some(runner.progress_counts());
        // Every completed/error boundary is saved before another inference.
        checkpoint(&report)?;
        if report.termination != "running" {
            break;
        }
    }
    Ok(report)
}

fn config_from_env() -> Result<G1RunnerConfig, String> {
    let required = |name| env::var(name).map_err(|e| format!("{name}: {e}"));
    let models = PathBuf::from(required("G1_MODEL_DIR")?);
    Ok(G1RunnerConfig {
        episode_id: 0,
        definition: models.join("g1_physics.json"),
        definition_sha256: required("G1_DEFINITION_SHA256")?,
        ort_library: required("G1_ORT")?.into(),
        ort_sha256: required("G1_ORT_SHA256")?,
        stand_model: models.join("stand.onnx"),
        walk_model: models.join("walk.onnx"),
        root_pose: SourcePose {
            position: [0., 0., 0.78],
            rotation_wxyz: [1., 0., 0., 0.],
        },
        robot_contact_friction: 0.5,
        floor_contact_friction: 1.0,
    })
}

fn save(file: &mut File, receipt: &Value) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(receipt)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&bytes)?;
    file.set_len(bytes.len() as u64)?;
    file.flush()?;
    file.sync_data()
}

#[test]
#[ignore = "source-only until approved run; requires frozen G1 env, G1_CODE_COMMIT, and fresh G1_ACTUATOR_DIAGNOSTIC_OUTPUT"]
fn real_homie_actuator_backend_diagnostic() {
    let output =
        env::var("G1_ACTUATOR_DIAGNOSTIC_OUTPUT").expect("required fresh diagnostic output path");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .expect("refuse overwrite; create output before loading models");
    let mut receipt = json!({"qualification":"actuator_backend_diagnostic_only_not_game","qualified":false,
        "code_commit":env::var("G1_CODE_COMMIT").ok(),"instrument_commit":"8536f833cbe54c4400294db560cc6565978bb6fe",
        "arena_commit":ARENA_COMMIT,"usd_sha256":USD_SHA256,"stand_sha256":STAND_SHA256,"walk_sha256":WALK_SHA256,
        "nominal_physics_hz":50,"control_hz":50,"maximum_ticks_per_backend":TICKS,"raw_motor_tick_budget":RAW_TICKS,
        "fall_root_height_m":0.35,"fall_upright_cosine":0.5,"joint_limit_tolerance_rad":LIMIT_TOLERANCE,
        "command":G1Command::default(),"counter_semantics":"torque_update_count counts actual command boundaries including an empty external-torque list; it is not motor solver-iteration count",
        "cold_initialization":"new G1Runner worlds and Homie sessions/history, shared byte-bound process ORT library",
        "limitations":["diagnostic only; no standing, task, or 1x realtime qualification", "source continuous parameters retained, actuator discrete behavior changed",
            "one PGS pass, one position integration, then one stabilization; not exact backward Euler",
            "external force acceleration uses gyro/Coriolis augmented mass and energy guard; motor impulse uses constraint mass path",
            "first inference must match bitwise; later closed-loop targets may diverge with different physical observations",
            "read-only motor impulses are boundary averages, not continuous torque waveforms; contact/limit impulses are separate",
            "synchronous per-boundary receipt I/O affects wall time, never the fixed physics dt"]});
    save(&mut file, &receipt).expect("save initial diagnostic receipt");
    let setup: Result<(G1RunnerConfig, G1Definition), String> = (|| {
        let source_commit = env::var("G1_CODE_COMMIT").map_err(|e| e.to_string())?;
        if source_commit.len() != 40
            || !source_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("G1_CODE_COMMIT must bind a full source commit".into());
        }
        if !Multibody::sim2sim_observation_backend_supported() {
            return Err("parallel telemetry is unsupported".into());
        }
        let config = config_from_env()?;
        let definition = G1Definition::load(&config.definition, &config.definition_sha256)
            .map_err(|e| e.to_string())?;
        Ok((config, definition))
    })();
    let (config, definition) = match setup {
        Ok(setup) => setup,
        Err(message) => {
            receipt["setup_error"] = json!(message);
            save(&mut file, &receipt).expect("save setup failure");
            panic!("diagnostic setup failed: {message}");
        }
    };
    receipt["inputs"] = json!({"definition":config.definition,"definition_sha256":config.definition_sha256,
        "ort_library":config.ort_library,"ort_sha256":config.ort_sha256,"stand_model":config.stand_model,"walk_model":config.walk_model,
        "root_pose":{"position":config.root_pose.position,"rotation_wxyz":config.root_pose.rotation_wxyz},
        "robot_contact_friction":config.robot_contact_friction,"floor_contact_friction":config.floor_contact_friction});
    let explicit = run_backend(
        &config,
        &definition,
        Backend::ExternalExplicitPd,
        None,
        |report| {
            receipt["external_explicit_pd"] = json!(report);
            save(&mut file, &receipt)
        },
    )
    .expect("persist explicit backend before continuing");
    let motor = run_backend(
        &config,
        &definition,
        Backend::NativeForceBased,
        Some(&explicit),
        |report| {
            receipt["native_force_based"] = json!(report);
            save(&mut file, &receipt)
        },
    )
    .expect("persist native backend before continuing");
    receipt["both_backends_attempted"] = json!(true);
    receipt["first_tick_action_and_targets_identical"] = json!(motor.first_tick_matches_baseline);
    save(&mut file, &receipt).expect("save final diagnostic receipt");
    println!(
        "G1_ACTUATOR_DIAGNOSTIC explicit={} native={} first_tick_match={:?} output={output} qualified=false",
        explicit.termination, motor.termination, motor.first_tick_matches_baseline
    );
    // This asserts the experiment's comparison contract, never physical success.
    assert_eq!(
        motor.first_tick_matches_baseline,
        Some(true),
        "cold first inference comparison unavailable or unequal; receipt preserved"
    );
    for report in [&explicit, &motor] {
        assert!(
            matches!(
                report.termination.as_str(),
                "completed_150_ticks" | "fell" | "boundary_guard_error"
            ),
            "diagnostic comparison protocol failed: {} / {:?}; receipt preserved",
            report.termination,
            report.error
        );
    }
}
