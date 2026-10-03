//! Explicit traditional T1 left-palm geometry, separate from both VLA and Homie.
//! Immutable original joint frames plus self sensors only; no physical world.

use super::runner::G1Measurement;
use rapier3d::na::{
    Isometry3, Quaternion, SMatrix, SVector, Translation3, UnitQuaternion, Vector3,
};
use robot_minigame::{
    RobotError,
    g1::{
        agile::AgileCommand,
        contract::{JOINT_COUNT, JOINT_NAMES, LOWER_COUNT},
        definition::{G1Definition, JointKind, SourcePose},
    },
};
use serde::Serialize;

const LEFT_PALM: usize = 28;
const LEFT_ARM: [usize; 7] = [15, 16, 17, 18, 19, 20, 21];
struct Joint {
    parent: usize,
    child: usize,
    parent_frame: Isometry3<f64>,
    child_inverse: Isometry3<f64>,
    axis: Vector3<f64>,
    slot: Option<usize>,
}
pub struct StaticLeftPalmKinematics {
    joints: Vec<Joint>,
    limits: [[f64; 2]; JOINT_COUNT],
}
#[derive(Clone, Debug, Serialize)]
pub struct StaticCartesianReceipt {
    pub schema: &'static str,
    pub episode_id: u64,
    pub source_tick: u64,
    pub offset_root_source_m: [f64; 3],
    pub maximum_left_arm_target_change_rad: f64,
    pub position_residual_m: f64,
    pub rotation_residual: f64,
    pub iterations: usize,
    pub original_vla_output: bool,
    pub task_qualified: bool,
}
impl StaticLeftPalmKinematics {
    pub fn new(definition: &G1Definition) -> Result<Self, RobotError> {
        let model = definition.model();
        if model.bodies.len() != 53 || model.bodies[LEFT_PALM].name != "left_hand_palm_link" {
            return Err(invalid("static kinematics requires original left palm"));
        }
        let mut remaining: Vec<_> = model.joints.iter().collect();
        let mut ready = [false; 53];
        ready[0] = true;
        let mut joints = Vec::new();
        let mut limits = [[0.; 2]; JOINT_COUNT];
        while !remaining.is_empty() {
            let i = remaining
                .iter()
                .position(|j| ready[j.parent])
                .ok_or_else(|| invalid("disconnected original tree"))?;
            let j = remaining.remove(i);
            let slot = if j.kind == JointKind::Revolute {
                let n = JOINT_NAMES
                    .iter()
                    .position(|&n| n == j.name)
                    .ok_or_else(|| invalid("foreign driven joint"))?;
                limits[n] = j.limits;
                Some(n)
            } else {
                None
            };
            let axis = match j.axis.as_str() {
                "X" => Vector3::x(),
                "Y" => Vector3::y(),
                "Z" => Vector3::z(),
                _ if slot.is_none() => Vector3::zeros(),
                _ => return Err(invalid("foreign axis")),
            };
            joints.push(Joint {
                parent: j.parent,
                child: j.child,
                parent_frame: pose(&j.frame_parent),
                child_inverse: pose(&j.frame_child).inverse(),
                axis,
                slot,
            });
            ready[j.child] = true;
        }
        Ok(Self { joints, limits })
    }
    fn forward(&self, q: &[f64; JOINT_COUNT]) -> [Isometry3<f64>; 53] {
        let mut poses = [Isometry3::identity(); 53];
        for j in &self.joints {
            let angle = j.slot.map_or(0., |i| q[i]);
            poses[j.child] = poses[j.parent]
                * j.parent_frame
                * Isometry3::from_parts(
                    Translation3::identity(),
                    UnitQuaternion::from_scaled_axis(j.axis * angle),
                )
                * j.child_inverse;
        }
        poses
    }
    /// One <=1.6mm increment (<=0.08m/s at50Hz). Only seven left-arm targets
    /// change. Both hands, right arm, pelvis and navigation are preserved.
    pub fn translate(
        &self,
        state: &G1Measurement,
        previous: &AgileCommand,
        offset: [f64; 3],
    ) -> Result<(AgileCommand, StaticCartesianReceipt), RobotError> {
        previous.validate()?;
        if state.episode_id == 0
            || state.source_tick.checked_mul(20_000_000) != Some(state.sim_time_ns)
            || state.joint_positions.len() != JOINT_COUNT
            || !state.joint_positions.iter().all(|x| x.is_finite())
            || !offset.iter().all(|x| x.is_finite())
            || Vector3::from(offset).norm() > 0.001600001
            || previous.navigation.iter().any(|x| x.abs() > 0.01)
        {
            return Err(invalid("static increment/self-state admission failed"));
        }
        let mut q = [0.; JOINT_COUNT];
        for (i, x) in q.iter_mut().enumerate() {
            *x = f64::from(if i < LOWER_COUNT {
                state.joint_positions[i]
            } else {
                previous.upper_positions[i - LOWER_COUNT]
            });
            if i >= LOWER_COUNT && (*x < self.limits[i][0] || *x > self.limits[i][1]) {
                return Err(invalid("static original upper target outside joint limit"));
            }
        }
        let initial = q;
        let mut goal = self.forward(&q)[LEFT_PALM];
        goal.translation.vector += Vector3::from(offset);
        let mut iterations = 0;
        let mut converged = false;
        for n in 0..60 {
            iterations = n;
            let current = self.forward(&q)[LEFT_PALM];
            let error = pose_error(&goal, &current);
            if error.fixed_rows::<3>(0).norm() < 1e-7 && error.fixed_rows::<3>(3).norm() < 1e-7 {
                converged = true;
                break;
            }
            let mut jacobian = SMatrix::<f64, 6, 7>::zeros();
            for (col, &slot) in LEFT_ARM.iter().enumerate() {
                let mut perturbed = q;
                perturbed[slot] += 1e-5;
                jacobian.set_column(
                    col,
                    &(pose_error(&self.forward(&perturbed)[LEFT_PALM], &current) / 1e-5),
                );
            }
            let system = jacobian * jacobian.transpose() + SMatrix::<f64, 6, 6>::identity() * 1e-6;
            let delta = jacobian.transpose()
                * system
                    .lu()
                    .solve(&error)
                    .ok_or_else(|| invalid("static Jacobian solve failed"))?;
            for (&slot, &step) in LEFT_ARM.iter().zip(delta.iter()) {
                q[slot] = (q[slot] + step.clamp(-0.05, 0.05))
                    .clamp(self.limits[slot][0], self.limits[slot][1]);
            }
        }
        if !converged {
            return Err(invalid(
                "static left-palm path unreachable with frozen orientation/limits",
            ));
        }
        let maximum = LEFT_ARM
            .iter()
            .map(|&i| (q[i] - initial[i]).abs())
            .fold(0., f64::max);
        if maximum > 0.05 {
            return Err(invalid("static increment exceeds0.05rad arm target change"));
        }
        let mut command = previous.clone();
        for i in LEFT_ARM {
            command.upper_positions[i - LOWER_COUNT] = q[i] as f32;
            q[i] = f64::from(command.upper_positions[i - LOWER_COUNT]);
        }
        let actual = self.forward(&q)[LEFT_PALM];
        let position = (goal.translation.vector - actual.translation.vector).norm();
        let rotation = (goal.rotation.to_rotation_matrix().matrix()
            - actual.rotation.to_rotation_matrix().matrix())
        .abs()
        .max();
        if position > 1e-6 || rotation > 1e-6 {
            return Err(invalid("static rounded target residual exceeds1e-6"));
        }
        Ok((
            command,
            StaticCartesianReceipt {
                schema: "g1_static_agile_left_palm_cartesian_v1",
                episode_id: state.episode_id,
                source_tick: state.source_tick,
                offset_root_source_m: offset,
                maximum_left_arm_target_change_rad: maximum,
                position_residual_m: position,
                rotation_residual: rotation,
                iterations,
                original_vla_output: false,
                task_qualified: false,
            },
        ))
    }
}
/// Current RGB-derived object centers. They are relative to the measured root,
/// with no simulator world pose, contacts or accepted model-output mutation.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticVisualTransferGoal {
    pub observation: task_minigame::types::ObservationStamp,
    pub apple_root_source_m: [f64; 3],
    pub plate_root_source_m: [f64; 3],
}
impl StaticVisualTransferGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let stamp = self.observation;
        let delta =
            Vector3::from(self.plate_root_source_m) - Vector3::from(self.apple_root_source_m);
        if stamp.episode_id == 0
            || stamp.frame_id == 0
            || stamp.captured_at_unix_ms == 0
            || stamp.sim_time_ns != 140 * 20_000_000
            || !self
                .apple_root_source_m
                .iter()
                .chain(self.plate_root_source_m.iter())
                .all(|x| x.is_finite())
            || delta.norm() > 0.35
            || delta.norm() < 0.05
        {
            return Err(invalid(
                "static visual transfer requires bounded two-target140Tick observation",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct StaticVisualTransferStep {
    pub goal: StaticVisualTransferGoal,
    pub transfer_ticks: u32,
    pub completed: bool,
    pub classical_geometry: StaticCartesianReceipt,
    pub command: AgileCommand,
    pub world_or_contact_truth_input: bool,
}
pub struct StaticVisualTransfer {
    goal: StaticVisualTransferGoal,
    source_horizontal_increment: Vector3<f64>,
    ticks: u32,
    command: AgileCommand,
}
impl StaticVisualTransfer {
    pub fn new(
        goal: StaticVisualTransferGoal,
        state: &G1Measurement,
        command: AgileCommand,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        if goal.observation.episode_id != state.episode_id
            || goal.observation.sim_time_ns != state.sim_time_ns
        {
            return Err(invalid(
                "static transfer image does not match current owner boundary",
            ));
        }
        let rotation = self_rotation(state)?;
        let mut delta = rotation
            * (Vector3::from(goal.plate_root_source_m) - Vector3::from(goal.apple_root_source_m));
        delta.z = 0.;
        if delta.norm() > 0.30 {
            return Err(invalid("static horizontal path exceeds30cm"));
        }
        let source_horizontal_increment = delta / 200.;
        // Whole-path FK admission belongs before command submission while the
        // diagnostic owner is explicitly paused. Never run250geometry solves
        // inside a physical Tick; each owner update checks its one increment.
        Ok(Self {
            goal,
            source_horizontal_increment,
            ticks: 0,
            command,
        })
    }
    pub fn goal(&self) -> &StaticVisualTransferGoal {
        &self.goal
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        kinematics: &StaticLeftPalmKinematics,
    ) -> Result<StaticVisualTransferStep, RobotError> {
        if self.ticks >= 250
            || state.episode_id != self.goal.observation.episode_id
            || state.source_tick != 140 + u64::from(self.ticks)
            || state.sim_time_ns != state.source_tick * 20_000_000
        {
            return Err(invalid("static transfer repeated/foreign/excessive Tick"));
        }
        let rotation = self_rotation(state)?;
        let source = if self.ticks < 50 {
            Vector3::new(0., 0., 0.001)
        } else {
            self.source_horizontal_increment
        };
        let (command, receipt) =
            kinematics.translate(state, &self.command, (rotation.inverse() * source).into())?;
        self.command = command;
        self.ticks += 1;
        Ok(StaticVisualTransferStep {
            goal: self.goal.clone(),
            transfer_ticks: self.ticks,
            completed: self.ticks == 250,
            classical_geometry: receipt,
            command: self.command.clone(),
            world_or_contact_truth_input: false,
        })
    }
}
fn self_rotation(state: &G1Measurement) -> Result<UnitQuaternion<f64>, RobotError> {
    let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
    if ![w, x, y, z].iter().all(|q| q.is_finite())
        || (w * w + x * x + y * y + z * z - 1.).abs() > 2e-5
    {
        return Err(invalid("invalid static self orientation"));
    }
    Ok(UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)))
}

fn pose(p: &SourcePose) -> Isometry3<f64> {
    let [w, x, y, z] = p.rotation_wxyz;
    Isometry3::from_parts(
        Translation3::from(Vector3::from(p.position)),
        UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)),
    )
}
fn pose_error(goal: &Isometry3<f64>, current: &Isometry3<f64>) -> SVector<f64, 6> {
    let p = goal.translation.vector - current.translation.vector;
    let m = (goal.rotation * current.rotation.inverse())
        .to_rotation_matrix()
        .into_inner();
    SVector::<f64, 6>::new(
        p.x,
        p.y,
        p.z,
        (m[(2, 1)] - m[(1, 2)]) * 0.5,
        (m[(0, 2)] - m[(2, 0)]) * 0.5,
        (m[(1, 0)] - m[(0, 1)]) * 0.5,
    )
}
fn invalid(s: impl std::fmt::Display) -> RobotError {
    RobotError::Contract(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use serde::Deserialize;
    use std::{fs, path::Path};
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        definition: String,
        definition_sha256: String,
        observation: task_minigame::types::ObservationStamp,
        measured_joints: SelfSensors,
        command: AgileCommand,
        increments_root_source_m: Vec<[f64; 3]>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SelfSensors {
        positions: Vec<f32>,
        velocities: Vec<f32>,
        root_rotation_wxyz: [f32; 4],
        root_angular_velocity_body: [f32; 3],
        root_velocity_source: [f32; 3],
    }
    #[test]
    #[ignore = "explicit actual RGB/self-sensor geometry fixture;0physics/model calls"]
    fn preflight_current_static_visual_path() -> Result<(), RobotError> {
        let path = std::env::var("G1_STATIC_GEOMETRY_FIXTURE").map_err(invalid)?;
        let hash = std::env::var("G1_STATIC_GEOMETRY_FIXTURE_SHA256").map_err(invalid)?;
        let bytes = bound_bytes(Path::new(&path), &hash)?;
        if bytes.len() > 128 * 1024 {
            return Err(invalid("geometry fixture exceeds bound"));
        }
        let fixture: Fixture = serde_json::from_slice(&bytes).map_err(invalid)?;
        if fixture.increments_root_source_m.is_empty()
            || fixture.increments_root_source_m.len() > 350
        {
            return Err(invalid("unbounded static path"));
        }
        let definition =
            G1Definition::load(Path::new(&fixture.definition), &fixture.definition_sha256)?;
        let kinematics = StaticLeftPalmKinematics::new(&definition)?;
        let mut state = G1Measurement {
            episode_id: fixture.observation.episode_id,
            source_tick: fixture.observation.sim_time_ns / 20_000_000,
            sim_time_ns: fixture.observation.sim_time_ns,
            joint_positions: fixture.measured_joints.positions,
            joint_velocities: fixture.measured_joints.velocities,
            root_rotation_wxyz: fixture.measured_joints.root_rotation_wxyz,
            root_angular_velocity_body: fixture.measured_joints.root_angular_velocity_body,
            root_velocity_source: fixture.measured_joints.root_velocity_source,
        };
        let original = fixture.command;
        let mut command = original.clone();
        let mut receipts = Vec::new();
        let mut failed = None;
        for (index, offset) in fixture.increments_root_source_m.into_iter().enumerate() {
            match kinematics.translate(&state, &command, offset) {
                Ok((next, receipt)) => {
                    assert_eq!(next.upper_positions[7..], original.upper_positions[7..]);
                    assert_eq!(next.pelvis_height, original.pelvis_height);
                    assert_eq!(next.navigation, original.navigation);
                    command = next;
                    receipts.push(receipt);
                    state.source_tick += 1;
                    state.sim_time_ns += 20_000_000;
                }
                Err(e) => {
                    failed = Some(serde_json::json!({"path_index":index,"error":e.to_string()}));
                    break;
                }
            }
        }
        let report = serde_json::json!({"schema":"g1_static_visual_left_palm_path_preflight_v1","physics_integrations":0,"model_calls":0,"source_observation":fixture.observation,"computed_geometry_points":receipts.len(),"failure":failed,"command":command,"receipts":receipts,"world_or_contact_truth_input":false,"task_qualified":false});
        let output = std::env::var("G1_STATIC_GEOMETRY_OUTPUT").map_err(invalid)?;
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
