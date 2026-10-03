//! Source-calibrated traditional grip correction from G1 self state and an
//! already admitted mobile VLA command. No world/object/contact input is used.
//! This opt-in development component grants no task qualification.

use rapier3d::na::{
    Isometry3, Matrix3, Quaternion, SMatrix, SVector, Translation3, UnitQuaternion, Vector3,
};
use robot_minigame::{
    RobotError,
    g1::{
        contract::{G1Command, JOINT_COUNT, JOINT_NAMES, LOWER_COUNT},
        definition::{G1Definition, JointKind, SourcePose},
    },
};
use serde::Serialize;

use super::runner::G1Measurement;

/// Desired palm separation measured from the matched source grasp (case0174),
/// not a fitted friction/force parameter. Both original palm poses stay centered.
pub const SOURCE_PALM_GAP_M: f64 = 0.156_042_580_477_048_72;
const PALMS: [usize; 2] = [28, 45];
const ARMS: [[usize; 7]; 2] = [[15, 16, 17, 18, 19, 20, 21], [29, 30, 31, 32, 33, 34, 35]];

struct KinematicJoint {
    parent: usize,
    child: usize,
    frame_parent: Isometry3<f64>,
    frame_child_inverse: Isometry3<f64>,
    axis: Vector3<f64>,
    slot: Option<usize>,
}

/// Immutable original joint frames/limits; independent of the physical world.
pub struct MobileGripCalibration {
    joints: Vec<KinematicJoint>,
    limits: [[f64; 2]; JOINT_COUNT],
}

/// Explicit provenance stays alongside the corrected controller command.
#[derive(Clone, Debug, Serialize)]
pub struct MobileGripReceipt {
    pub schema: &'static str,
    pub episode_id: u64,
    pub source_tick: u64,
    pub original_vla_output: bool,
    pub task_qualified: bool,
    pub original_palm_gap_m: f64,
    pub calibrated_palm_gap_m: f64,
    pub maximum_joint_target_change_rad: f64,
    pub maximum_palm_position_residual_m: f64,
    pub maximum_palm_rotation_residual: f64,
    pub solve_iterations: [usize; 2],
}

pub struct MobileGripCorrection {
    pub command: G1Command,
    pub receipt: MobileGripReceipt,
}

impl MobileGripCalibration {
    pub fn new(definition: &G1Definition) -> Result<Self, RobotError> {
        let model = definition.model();
        if model.bodies.len() != 53
            || model.bodies[28].name != "left_hand_palm_link"
            || model.bodies[45].name != "right_hand_palm_link"
        {
            return Err(invalid(
                "grip calibration requires original G1 hand identities",
            ));
        }
        let mut remaining: Vec<_> = model.joints.iter().collect();
        let mut ready = [false; 53];
        ready[0] = true;
        let mut joints = Vec::with_capacity(52);
        let mut limits = [[0.; 2]; JOINT_COUNT];
        while !remaining.is_empty() {
            let Some(i) = remaining.iter().position(|j| ready[j.parent]) else {
                return Err(invalid("grip kinematic tree is disconnected"));
            };
            let joint = remaining.remove(i);
            let slot = if joint.kind == JointKind::Revolute {
                let slot = JOINT_NAMES
                    .iter()
                    .position(|&name| name == joint.name)
                    .ok_or_else(|| invalid("grip tree contains foreign driven joint"))?;
                limits[slot] = joint.limits;
                Some(slot)
            } else {
                None
            };
            let axis = match joint.axis.as_str() {
                "X" => Vector3::x(),
                "Y" => Vector3::y(),
                "Z" => Vector3::z(),
                _ if slot.is_none() => Vector3::zeros(),
                _ => return Err(invalid("grip tree has unknown driven axis")),
            };
            joints.push(KinematicJoint {
                parent: joint.parent,
                child: joint.child,
                frame_parent: pose(&joint.frame_parent),
                frame_child_inverse: pose(&joint.frame_child).inverse(),
                axis,
                slot,
            });
            ready[joint.child] = true;
        }
        Ok(Self { joints, limits })
    }

    /// Bound camera-to-owner palm motion from measured joints/IMU and self
    /// velocity odometry only. No physical body or object pose is consulted.
    pub(super) fn measured_palm_motion(
        &self,
        image: &G1Measurement,
        current: &G1Measurement,
        self_displacement_source_m: [f64; 3],
    ) -> Result<(f64, f64), RobotError> {
        validate_self_state(image)?;
        validate_self_state(current)?;
        if image.episode_id != current.episode_id
            || !self_displacement_source_m.iter().all(|v| v.is_finite())
        {
            return Err(invalid("palm admission has foreign/nonfinite self history"));
        }
        let joints = |s: &G1Measurement| std::array::from_fn(|i| f64::from(s.joint_positions[i]));
        let before = self.forward(&joints(image));
        let after = self.forward(&joints(current));
        let rotation = |s: &G1Measurement| {
            let [w, x, y, z] = s.root_rotation_wxyz;
            UnitQuaternion::new_normalize(Quaternion::new(
                f64::from(w),
                f64::from(x),
                f64::from(y),
                f64::from(z),
            ))
        };
        let r0 = rotation(image);
        let r1 = rotation(current);
        let displacement = Vector3::from(self_displacement_source_m);
        let maximum = PALMS
            .iter()
            .map(|&p| {
                (displacement + r1 * after[p].translation.vector
                    - r0 * before[p].translation.vector)
                    .norm()
            })
            .fold(0., f64::max);
        Ok((maximum, r0.angle_to(&r1)))
    }

    /// Pure correction. On a failed/bounded solve the owner receives an error,
    /// never clipped model bytes or an unvalidated motor command.
    pub fn correct(
        &self,
        state: &G1Measurement,
        original: &G1Command,
    ) -> Result<MobileGripCorrection, RobotError> {
        self.solve(state, original, None, None, None)
    }

    /// One bounded Cartesian increment preserves both palm rotations/gap and
    /// all original finger/lower targets. It reads self state, never the box.
    pub fn translate(
        &self,
        state: &G1Measurement,
        previous: &G1Command,
        offset_root_source_m: [f64; 3],
    ) -> Result<MobileGripCorrection, RobotError> {
        if !offset_root_source_m.iter().all(|x| x.is_finite())
            || Vector3::from(offset_root_source_m).norm() > 0.003
        {
            return Err(invalid(
                "Cartesian grip increment exceeds3mm per native Tick",
            ));
        }
        self.solve(state, previous, Some(offset_root_source_m), None, None)
    }

    /// Open both palm targets symmetrically; no prop/contact sensor is read.
    pub fn spread(
        &self,
        state: &G1Measurement,
        previous: &G1Command,
        gap_increment_m: f64,
    ) -> Result<MobileGripCorrection, RobotError> {
        if !gap_increment_m.is_finite() || !(0. ..=0.004).contains(&gap_increment_m) {
            return Err(invalid("grip opening exceeds2mm per palm per native Tick"));
        }
        self.solve(state, previous, None, Some(gap_increment_m), None)
    }

    /// Spread perpendicular to gravity while preserving each palm's height.
    /// Unlike a tilted palm-to-palm line, this adds no differential lift.
    pub fn spread_horizontal(
        &self,
        state: &G1Measurement,
        previous: &G1Command,
        horizontal_gap_increment_m: f64,
    ) -> Result<MobileGripCorrection, RobotError> {
        if !horizontal_gap_increment_m.is_finite()
            || !(0. ..=0.004).contains(&horizontal_gap_increment_m)
        {
            return Err(invalid("horizontal opening exceeds2mm per palm per Tick"));
        }
        self.solve(
            state,
            previous,
            None,
            None,
            Some(horizontal_gap_increment_m),
        )
    }

    /// Convert the existing three-dimensional final gap to a fixed horizontal
    /// increment using the named self orientation and original FK only.
    pub fn horizontal_opening_increment(
        &self,
        state: &G1Measurement,
        command: &G1Command,
        target_gap_m: f64,
        duration_ticks: u32,
    ) -> Result<f64, RobotError> {
        let delta = self.commanded_palm_separation(state, command)?;
        let up = up_in_root(state)?;
        let vertical = delta.dot(&up);
        let horizontal = (delta - up * vertical).norm();
        let final_squared = target_gap_m.powi(2) - vertical.powi(2);
        if !target_gap_m.is_finite()
            || !(0.25..=0.35).contains(&target_gap_m)
            || !(50..=150).contains(&duration_ticks)
            || horizontal < 1e-4
            || final_squared <= 0.
        {
            return Err(invalid("invalid horizontal opening geometry"));
        }
        let increment = (final_squared.sqrt() - horizontal) / f64::from(duration_ticks);
        if !(0. ..=0.004).contains(&increment) {
            return Err(invalid(
                "horizontal opening increment outside bounded envelope",
            ));
        }
        Ok(increment)
    }

    pub fn commanded_gap(
        &self,
        state: &G1Measurement,
        command: &G1Command,
    ) -> Result<f64, RobotError> {
        Ok(self.commanded_palm_separation(state, command)?.norm())
    }

    fn commanded_palm_separation(
        &self,
        state: &G1Measurement,
        command: &G1Command,
    ) -> Result<Vector3<f64>, RobotError> {
        validate_self_state(state)?;
        command.validate()?;
        let mut q = [0.; JOINT_COUNT];
        for i in 0..JOINT_COUNT {
            q[i] = f64::from(if i < LOWER_COUNT {
                state.joint_positions[i]
            } else {
                command.upper_positions[i - LOWER_COUNT]
            });
        }
        let poses = self.forward(&q);
        Ok(poses[28].translation.vector - poses[45].translation.vector)
    }

    /// Restore this episode's original calibrated transport pose through one
    /// common bounded Cartesian shift. Inputs are commands and self joints only.
    pub fn transport_restore_offset(
        &self,
        state: &G1Measurement,
        current: &G1Command,
        original: &G1Command,
    ) -> Result<[f64; 3], RobotError> {
        validate_self_state(state)?;
        current.validate()?;
        original.validate()?;
        let frames = |command: &G1Command| {
            let mut q = [0.; JOINT_COUNT];
            for i in 0..JOINT_COUNT {
                q[i] = f64::from(if i < LOWER_COUNT {
                    state.joint_positions[i]
                } else {
                    command.upper_positions[i - LOWER_COUNT]
                });
            }
            self.forward(&q)
        };
        let now = frames(current);
        let target = frames(original);
        let offsets = PALMS.map(|i| target[i].translation.vector - now[i].translation.vector);
        let common = (offsets[0] + offsets[1]) * 0.5;
        if current.navigation != [0.; 3]
            || original.navigation != [0.; 3]
            || common.norm() > 0.18
            || (offsets[0] - offsets[1]).norm() > 2e-6
            || PALMS.iter().any(|&i| {
                (target[i].rotation.to_rotation_matrix().matrix()
                    - now[i].rotation.to_rotation_matrix().matrix())
                .norm()
                    > 2e-5
            })
            || [7..14, 21..28].iter().any(|range| {
                current.upper_positions[range.clone()] != original.upper_positions[range.clone()]
            })
        {
            return Err(invalid(
                "transport restoration is not a bounded common rigid-grip translation",
            ));
        }
        Ok(common.into())
    }

    /// Keep both original shoulder-to-wrist radii from increasing during a
    /// downward increment. One common inward shift keeps the box grip rigid.
    pub fn reach_preserving_offset(
        &self,
        state: &G1Measurement,
        command: &G1Command,
        downward_root_source_m: [f64; 3],
    ) -> Result<[f64; 3], RobotError> {
        validate_self_state(state)?;
        command.validate()?;
        let mut q = [0.; JOINT_COUNT];
        for (i, value) in q.iter_mut().enumerate() {
            *value = f64::from(if i < LOWER_COUNT {
                state.joint_positions[i]
            } else {
                command.upper_positions[i - LOWER_COUNT]
            });
        }
        let poses = self.forward(&q);
        let downward = Vector3::from(downward_root_source_m);
        if !downward.iter().all(|x| x.is_finite()) || downward.norm() > 0.003 {
            return Err(invalid("reach-preserving increment exceeds3mm"));
        }
        let mut inward = 0_f64;
        for (arm, slot) in [15, 29].into_iter().enumerate() {
            let joint = self
                .joints
                .iter()
                .find(|j| j.slot == Some(slot))
                .ok_or_else(|| invalid("original shoulder pitch joint absent"))?;
            let shoulder = (poses[joint.parent] * joint.frame_parent)
                .translation
                .vector;
            let wrist_joint = self
                .joints
                .iter()
                .find(|j| j.slot == Some([21, 35][arm]))
                .ok_or_else(|| invalid("original wrist yaw joint absent"))?;
            // A fixed palm orientation fixes the wrist-center offset too.
            // Palm-center radius alone misses that offset and still reaches
            // the arm-extension singularity (physical diagnostic0192).
            let wrist = (poses[wrist_joint.parent] * wrist_joint.frame_parent)
                .translation
                .vector;
            let relative = wrist - shoulder;
            let candidate = relative + downward;
            let radicand = relative.norm_squared() - candidate.y.powi(2) - candidate.z.powi(2);
            if relative.x < 0.05 || radicand <= 0.0025 {
                return Err(invalid(
                    "lowering would leave the forward arm reach envelope",
                ));
            }
            inward = inward.min(radicand.sqrt() - candidate.x);
        }
        let offset = downward + Vector3::new(inward, 0., 0.);
        if offset.norm() > 0.003 {
            return Err(invalid("reach-preserving inward step exceeds3mm"));
        }
        Ok(offset.into())
    }

    fn solve(
        &self,
        state: &G1Measurement,
        original: &G1Command,
        translation: Option<[f64; 3]>,
        opening: Option<f64>,
        horizontal_opening: Option<f64>,
    ) -> Result<MobileGripCorrection, RobotError> {
        validate_self_state(state)?;
        original.validate()?;
        let mut q = [0.; JOINT_COUNT];
        for (i, target) in q.iter_mut().enumerate() {
            *target = f64::from(if i < LOWER_COUNT {
                state.joint_positions[i]
            } else {
                original.upper_positions[i - LOWER_COUNT]
            });
            if i >= LOWER_COUNT && (*target < self.limits[i][0] || *target > self.limits[i][1]) {
                return Err(invalid(
                    "original mobile upper target outside original joint limit",
                ));
            }
        }
        let initial_q = q;
        let initial = self.forward(&q);
        let midpoint = (initial[28].translation.vector + initial[45].translation.vector) * 0.5;
        let delta = initial[28].translation.vector - initial[45].translation.vector;
        let original_gap = delta.norm();
        if original_gap < 1e-4 {
            return Err(invalid(
                "coincident palm goals cannot define grip direction",
            ));
        }
        let direction = delta / original_gap;
        let mut goals = [initial[28], initial[45]];
        if let Some(offset) = translation {
            for goal in &mut goals {
                goal.translation.vector += Vector3::from(offset);
            }
        } else if let Some(increment) = horizontal_opening {
            let offset = horizontal_spread_offset(delta, up_in_root(state)?, increment)?;
            goals[0].translation.vector += offset;
            goals[1].translation.vector -= offset;
        } else {
            let gap = opening.map_or(SOURCE_PALM_GAP_M, |increment| original_gap + increment);
            goals[0].translation.vector = midpoint + direction * (gap * 0.5);
            goals[1].translation.vector = midpoint - direction * (gap * 0.5);
        }
        let mut iterations = [0; 2];
        for arm in 0..2 {
            let mut converged = false;
            for iteration in 0..60 {
                let current = self.forward(&q)[PALMS[arm]];
                let error = pose_error(&goals[arm], &current);
                iterations[arm] = iteration;
                if error.fixed_rows::<3>(0).norm() < 1e-7 && error.fixed_rows::<3>(3).norm() < 1e-7
                {
                    converged = true;
                    break;
                }
                let mut jacobian = SMatrix::<f64, 6, 7>::zeros();
                for (column, &slot) in ARMS[arm].iter().enumerate() {
                    let mut perturbed = q;
                    perturbed[slot] += 1e-5;
                    let derivative =
                        pose_error(&self.forward(&perturbed)[PALMS[arm]], &current) / 1e-5;
                    jacobian.set_column(column, &derivative);
                }
                let system =
                    jacobian * jacobian.transpose() + SMatrix::<f64, 6, 6>::identity() * 1e-6;
                let solved = system
                    .lu()
                    .solve(&error)
                    .ok_or_else(|| invalid("bounded grip Jacobian solve failed"))?;
                let delta = jacobian.transpose() * solved;
                for (&slot, &step) in ARMS[arm].iter().zip(delta.iter()) {
                    q[slot] = (q[slot] + step.clamp(-0.05, 0.05))
                        .clamp(self.limits[slot][0], self.limits[slot][1]);
                }
            }
            if !converged {
                return Err(invalid("bounded grip solver did not converge"));
            }
        }
        let maximum_change = q
            .iter()
            .zip(initial_q)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if maximum_change > 0.1 || !q.iter().all(|value| value.is_finite()) {
            return Err(invalid("grip correction exceeds bounded target change"));
        }
        let mut command = original.clone();
        for i in LOWER_COUNT..JOINT_COUNT {
            command.upper_positions[i - LOWER_COUNT] = q[i] as f32;
            q[i] = f64::from(command.upper_positions[i - LOWER_COUNT]);
        }
        let final_poses = self.forward(&q);
        let position_residual = (0..2)
            .map(|i| {
                (goals[i].translation.vector - final_poses[PALMS[i]].translation.vector).norm()
            })
            .fold(0., f64::max);
        let rotation_residual = (0..2)
            .map(|i| {
                (goals[i].rotation.to_rotation_matrix().matrix()
                    - final_poses[PALMS[i]].rotation.to_rotation_matrix().matrix())
                .abs()
                .max()
            })
            .fold(0., f64::max);
        if position_residual > 1e-6 || rotation_residual > 1e-6 {
            return Err(invalid(
                "rounded grip correction fails geometric residual bounds",
            ));
        }
        Ok(MobileGripCorrection {
            command,
            receipt: MobileGripReceipt {
                schema: if translation.is_some() {
                    "g1_mobile_self_state_cartesian_grip_v1"
                } else if horizontal_opening.is_some() {
                    "g1_mobile_self_state_horizontal_open_grip_v1"
                } else if opening.is_some() {
                    "g1_mobile_self_state_open_grip_v1"
                } else {
                    "g1_mobile_source_gap_assist_v1"
                },
                episode_id: state.episode_id,
                source_tick: state.source_tick,
                original_vla_output: false,
                task_qualified: false,
                original_palm_gap_m: original_gap,
                calibrated_palm_gap_m: (final_poses[28].translation.vector
                    - final_poses[45].translation.vector)
                    .norm(),
                maximum_joint_target_change_rad: maximum_change,
                maximum_palm_position_residual_m: position_residual,
                maximum_palm_rotation_residual: rotation_residual,
                solve_iterations: iterations,
            },
        })
    }

    fn forward(&self, q: &[f64; JOINT_COUNT]) -> [Isometry3<f64>; 53] {
        let mut poses = [Isometry3::identity(); 53];
        for joint in &self.joints {
            let angle = joint.slot.map_or(0., |slot| q[slot]);
            let turn = Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_scaled_axis(joint.axis * angle),
            );
            poses[joint.child] =
                poses[joint.parent] * joint.frame_parent * turn * joint.frame_child_inverse;
        }
        poses
    }
}

fn up_in_root(state: &G1Measurement) -> Result<Vector3<f64>, RobotError> {
    let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
    let norm_squared = w * w + x * x + y * y + z * z;
    if !norm_squared.is_finite() || (norm_squared - 1.).abs() > 2e-5 {
        return Err(invalid(
            "horizontal opening needs a normalized self quaternion",
        ));
    }
    Ok(UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z))
        .inverse_transform_vector(&Vector3::z()))
}

fn horizontal_spread_offset(
    delta: Vector3<f64>,
    up: Vector3<f64>,
    increment: f64,
) -> Result<Vector3<f64>, RobotError> {
    let horizontal = delta - up * delta.dot(&up);
    if horizontal.norm() < 1e-4 {
        return Err(invalid("horizontal opening has no lateral direction"));
    }
    Ok(horizontal.normalize() * (increment * 0.5))
}

fn pose(source: &SourcePose) -> Isometry3<f64> {
    let [w, x, y, z] = source.rotation_wxyz;
    Isometry3::from_parts(
        Translation3::from(Vector3::from(source.position)),
        UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)),
    )
}

fn vee(rotation: Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(
        rotation[(2, 1)] - rotation[(1, 2)],
        rotation[(0, 2)] - rotation[(2, 0)],
        rotation[(1, 0)] - rotation[(0, 1)],
    ) * 0.5
}

fn pose_error(goal: &Isometry3<f64>, current: &Isometry3<f64>) -> SVector<f64, 6> {
    let position = goal.translation.vector - current.translation.vector;
    let rotation = vee((goal.rotation * current.rotation.inverse())
        .to_rotation_matrix()
        .into_inner());
    SVector::<f64, 6>::new(
        position.x, position.y, position.z, rotation.x, rotation.y, rotation.z,
    )
}

fn validate_self_state(state: &G1Measurement) -> Result<(), RobotError> {
    if state.joint_positions.len() != JOINT_COUNT
        || !state.joint_positions.iter().all(|q| q.is_finite())
        || state.source_tick.checked_mul(20_000_000) != Some(state.sim_time_ns)
    {
        return Err(invalid(
            "grip correction requires coherent original43joint/50Hz self state",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use serde::Deserialize;
    use std::{fs, path::Path};

    #[test]
    fn horizontal_spread_preserves_height_for_tilted_palms_and_body() {
        let up = Vector3::<f64>::new(0.1, 0.2, 0.97).normalize();
        let delta = Vector3::<f64>::new(0.004, 0.24, 0.03);
        assert!(delta.normalize().dot(&up).abs() > 0.1);
        let offset = horizontal_spread_offset(delta, up, 0.003).unwrap();
        assert!(offset.dot(&up).abs() < 1e-12);
        assert!((offset.norm() - 0.0015).abs() < 1e-12);
        assert!(horizontal_spread_offset(up * 0.2, up, 0.003).is_err());
    }

    #[test]
    fn horizontal_opening_rejects_unknown_self_gravity() {
        let mut state = G1Measurement {
            episode_id: 1,
            source_tick: 0,
            sim_time_ns: 0,
            joint_positions: vec![0.; JOINT_COUNT],
            joint_velocities: vec![0.; JOINT_COUNT],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [0.; 3],
        };
        assert_eq!(up_in_root(&state).unwrap(), Vector3::z());
        for quaternion in [[0.; 4], [2., 0., 0., 0.], [f32::NAN, 0., 0., 0.]] {
            state.root_rotation_wxyz = quaternion;
            assert!(up_in_root(&state).is_err());
        }
    }

    #[derive(Deserialize)]
    struct Fixture {
        state: serde_json::Value,
        original_command: G1Command,
        expected_upper_positions: [f32; 28],
    }

    fn measurement(v: serde_json::Value) -> Result<G1Measurement, serde_json::Error> {
        Ok(G1Measurement {
            episode_id: serde_json::from_value(v["episode_id"].clone())?,
            source_tick: serde_json::from_value(v["source_tick"].clone())?,
            sim_time_ns: serde_json::from_value(v["sim_time_ns"].clone())?,
            joint_positions: serde_json::from_value(v["joint_positions"].clone())?,
            joint_velocities: serde_json::from_value(v["joint_velocities"].clone())?,
            root_rotation_wxyz: serde_json::from_value(v["root_rotation_wxyz"].clone())?,
            root_angular_velocity_body: serde_json::from_value(
                v["root_angular_velocity_body"].clone(),
            )?,
            root_velocity_source: serde_json::from_value(v["root_velocity_source"].clone())?,
        })
    }

    #[derive(Deserialize)]
    struct LowerFixture {
        state: serde_json::Value,
        command: G1Command,
        goal: super::super::mobile_lowering::MobileLowerGoal,
    }

    #[derive(Deserialize)]
    struct RestoreFixture {
        state: serde_json::Value,
        command: G1Command,
        original_command: G1Command,
        goal: super::super::mobile_restore::MobileRestoreGoal,
    }
    #[derive(Deserialize)]
    struct ReleaseFixture {
        state: serde_json::Value,
        command: G1Command,
        goal: super::super::mobile_release::MobileReleaseGoal,
    }
    #[test]
    #[ignore = "requires pinned real self-state/command; pure opening envelope,0physics/models"]
    fn real_original_transport_release_envelope() -> Result<(), RobotError> {
        let env = |name| std::env::var(name).map_err(|e| invalid(format!("{name}: {e}")));
        let definition = G1Definition::load(
            Path::new(&env("G1_GRIP_DEFINITION")?),
            &env("G1_GRIP_DEFINITION_SHA256")?,
        )?;
        let fixture: ReleaseFixture = serde_json::from_slice(&bound_bytes(
            Path::new(&env("G1_GRIP_FIXTURE")?),
            &env("G1_GRIP_FIXTURE_SHA256")?,
        )?)
        .map_err(|e| invalid(e.to_string()))?;
        let state = measurement(fixture.state).map_err(|e| invalid(e.to_string()))?;
        let calibration = MobileGripCalibration::new(&definition)?;
        let original = fixture.command.clone();
        let target_gap = fixture.goal.target_palm_gap_m;
        let mut release = super::super::mobile_release::MobileGripRelease::new(
            fixture.goal,
            &state,
            fixture.command,
            &calibration,
        )?;
        let mut steps = Vec::new();
        for index in 0..225 {
            let mut synthetic = state.clone();
            synthetic.source_tick += index;
            synthetic.sim_time_ns = synthetic.source_tick * 20_000_000;
            let step = release.update(&synthetic, &calibration)?;
            assert_eq!(step.command.navigation, [0.; 3]);
            assert_eq!(
                step.command.upper_positions[7..14],
                original.upper_positions[7..14]
            );
            assert_eq!(
                step.command.upper_positions[21..28],
                original.upper_positions[21..28]
            );
            steps.push(step);
        }
        assert!(steps.last().unwrap().completed);
        assert!((steps.last().unwrap().commanded_palm_gap_m - target_gap).abs() < 1e-6);
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(env("G1_GRIP_OUTPUT")?)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({
                "actual_integrations":0,"actual_model_inferences":0,"qualified":false,
                "synthetic_future_self_states_numeric_only":true,"completed":true,"steps":steps,
            }),
        )
        .map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }
    #[test]
    #[ignore = "requires original definition and frozen raised self-state/commands;0world/model work"]
    fn real_mobile_transport_restore_envelope() -> Result<(), RobotError> {
        let env = |name| std::env::var(name).map_err(|e| invalid(format!("{name}: {e}")));
        let definition = G1Definition::load(
            Path::new(&env("G1_GRIP_DEFINITION")?),
            &env("G1_GRIP_DEFINITION_SHA256")?,
        )?;
        let fixture: RestoreFixture = serde_json::from_slice(&bound_bytes(
            Path::new(&env("G1_GRIP_FIXTURE")?),
            &env("G1_GRIP_FIXTURE_SHA256")?,
        )?)
        .map_err(|e| invalid(e.to_string()))?;
        let state = measurement(fixture.state).map_err(|e| invalid(e.to_string()))?;
        let calibration = MobileGripCalibration::new(&definition)?;
        let mut restoring = super::super::mobile_restore::MobileGripRestoring::new(
            fixture.goal,
            &state,
            fixture.command,
            &fixture.original_command,
            &calibration,
        )?;
        let mut steps = Vec::new();
        let result = (|| -> Result<(), RobotError> {
            for index in 0..300 {
                let mut synthetic = state.clone();
                synthetic.source_tick += index;
                synthetic.sim_time_ns = synthetic.source_tick * 20_000_000;
                let step = restoring.update(&synthetic, &calibration)?;
                let completed = step.completed;
                steps.push(step);
                if completed {
                    return Ok(());
                }
            }
            Err(invalid("restore pure envelope exceeded300updates"))
        })();
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(env("G1_GRIP_OUTPUT")?)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(file,&serde_json::json!({"actual_integrations":0,"actual_model_inferences":0,"qualified":false,"synthetic_future_self_states_numeric_only":true,"completed":result.is_ok(),"steps":steps,"failure":result.as_ref().err().map(ToString::to_string)})).map_err(|e|invalid(e.to_string()))?;
        result
    }

    /// Pure planning with one frozen real self state. Future Tick identities
    /// below belong only to this numeric fixture, never a physical observation.
    #[test]
    #[ignore = "requires frozen G1/self-state lowering fixture;0physical steps/0models"]
    fn real_mobile_grip_lowering_envelope() -> Result<(), RobotError> {
        let env = |name| std::env::var(name).map_err(|e| invalid(format!("{name}: {e}")));
        let definition = G1Definition::load(
            Path::new(&env("G1_GRIP_DEFINITION")?),
            &env("G1_GRIP_DEFINITION_SHA256")?,
        )?;
        let fixture: LowerFixture = serde_json::from_slice(&bound_bytes(
            Path::new(&env("G1_GRIP_FIXTURE")?),
            &env("G1_GRIP_FIXTURE_SHA256")?,
        )?)
        .map_err(|e| invalid(e.to_string()))?;
        let original_state = measurement(fixture.state).map_err(|e| invalid(e.to_string()))?;
        let calibration = MobileGripCalibration::new(&definition)?;
        let mut lowering = super::super::mobile_lowering::MobileGripLowering::new(
            fixture.goal,
            &original_state,
            fixture.command,
        )?;
        let mut steps = Vec::new();
        let result = (|| -> Result<(), RobotError> {
            for index in 0..350 {
                let mut synthetic = original_state.clone();
                synthetic.source_tick += index;
                synthetic.sim_time_ns = synthetic.source_tick * 20_000_000;
                let step = lowering.update(&synthetic, &calibration)?;
                let completed = step.completed;
                steps.push(step);
                if completed {
                    return Ok(());
                }
            }
            Err(invalid("pure lowering envelope exceeded350fixture updates"))
        })();
        let output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(env("G1_GRIP_OUTPUT")?)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(output,&serde_json::json!({
            "pure_numeric_fixture_not_execution":true,"actual_integrations":0,"actual_model_inferences":0,
            "qualified":false,"completed":result.is_ok(),"steps":steps,
            "failure":result.as_ref().err().map(ToString::to_string),
        })).map_err(|e| invalid(e.to_string()))?;
        result
    }

    /// Cross-language numerical comparison to the independent offline FK/IK
    /// fixture. It opens no model session and integrates no physical world.
    #[test]
    #[ignore = "requires frozen G1 definition/real-grasp fixture hashes and a new output path"]
    fn real_mobile_grip_calibration_equivalence() -> Result<(), RobotError> {
        let env = |name| std::env::var(name).map_err(|e| invalid(format!("{name}: {e}")));
        let definition = G1Definition::load(
            Path::new(&env("G1_GRIP_DEFINITION")?),
            &env("G1_GRIP_DEFINITION_SHA256")?,
        )?;
        let bytes = bound_bytes(
            Path::new(&env("G1_GRIP_FIXTURE")?),
            &env("G1_GRIP_FIXTURE_SHA256")?,
        )?;
        let fixture: Fixture =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let state = measurement(fixture.state).map_err(|e| invalid(e.to_string()))?;
        let calibration = MobileGripCalibration::new(&definition)?;
        let correction = calibration.correct(&state, &fixture.original_command)?;
        let maximum_golden_difference = correction
            .command
            .upper_positions
            .iter()
            .zip(fixture.expected_upper_positions)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(maximum_golden_difference < 2e-6);
        assert_eq!(
            correction.command.upper_positions[7..14],
            fixture.original_command.upper_positions[7..14]
        );
        assert_eq!(
            correction.command.upper_positions[21..28],
            fixture.original_command.upper_positions[21..28]
        );
        assert_eq!(
            correction.command.navigation,
            fixture.original_command.navigation
        );
        assert_eq!(
            correction.command.torso_rpy,
            fixture.original_command.torso_rpy
        );
        assert_eq!(
            correction.command.pelvis_height,
            fixture.original_command.pelvis_height
        );
        let mut malformed = state.clone();
        malformed.joint_positions.pop();
        assert!(
            calibration
                .correct(&malformed, &fixture.original_command)
                .is_err()
        );
        malformed = state.clone();
        malformed.joint_positions[0] = f32::NAN;
        assert!(
            calibration
                .correct(&malformed, &fixture.original_command)
                .is_err()
        );
        malformed = state.clone();
        malformed.sim_time_ns += 1;
        assert!(
            calibration
                .correct(&malformed, &fixture.original_command)
                .is_err()
        );
        let mut unsafe_command = fixture.original_command.clone();
        unsafe_command.upper_positions[0] = f32::NAN;
        assert!(calibration.correct(&state, &unsafe_command).is_err());
        unsafe_command = fixture.original_command;
        unsafe_command.upper_positions[0] = 100.;
        assert!(calibration.correct(&state, &unsafe_command).is_err());
        let output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(env("G1_GRIP_OUTPUT")?)
            .map_err(|e| invalid(e.to_string()))?;
        serde_json::to_writer_pretty(output, &serde_json::json!({
            "receipt":correction.receipt,"corrected_command":correction.command,
            "maximum_independent_golden_target_difference_rad":maximum_golden_difference,
            "negative_input_guards_passed":5,"actual_vla_or_body_inferences":0,"actual_integrations":0,
            "qualified":false,"autonomous_execution":false,
        })).map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }
}
