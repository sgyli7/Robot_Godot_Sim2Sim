//! Source-derived articulation structure, constructed in the application's world.
//!
//! This stage verifies initialization and imported geometry. Native contact
//! filtering, complete force measurements and the control lifecycle are separate
//! gates; construction alone never qualifies a target plant or a skill.

use rapier3d::{
    math::{Pose, Rotation, Vector},
    prelude::*,
};
use robot_minigame::{
    ACTION_DIMENSION, RobotError,
    basis::{source_to_engine_rotation, source_to_engine_vector},
    body_pose::{RobotBodyPose, RobotPoseFrame},
    collision_hull,
    definition::RobotDefinition,
    joint_feedback::{JointFeedbackChannel, NativeHandleIdentity},
    kinematics::source_body_poses,
};

use crate::BodyTorque;

#[derive(Debug)]
pub struct SourceJointMapping {
    pub source_joint: usize,
    pub source_dof: usize,
    pub handle: MultibodyJointHandle,
    pub backend_dof: usize,
    pub reference: f32,
}

/// Handles refer only to the caller's sole world, including handle generations.
pub struct RobotAssembly {
    body_handles: Vec<Option<RigidBodyHandle>>,
    collision_handles: Vec<(usize, ColliderHandle)>,
    joint_mapping: Vec<SourceJointMapping>,
    actuator_joints: [usize; ACTION_DIMENSION],
    source_file_sha256: String,
}

impl RobotAssembly {
    pub fn body_handles(&self) -> &[Option<RigidBodyHandle>] {
        &self.body_handles
    }
    pub fn collision_handles(&self) -> &[(usize, ColliderHandle)] {
        &self.collision_handles
    }
    pub fn joint_mapping(&self) -> &[SourceJointMapping] {
        &self.joint_mapping
    }
    pub fn actuator_joints(&self) -> &[usize; ACTION_DIMENSION] {
        &self.actuator_joints
    }
    pub fn source_file_sha256(&self) -> &str {
        &self.source_file_sha256
    }

    /// Map this assembly's generation-bound handles into one complete snapshot.
    /// No kinematics, integration or second basis conversion occurs here.
    pub fn pose_frame(&self, snapshot: &crate::StepSnapshot) -> Result<RobotPoseFrame, RobotError> {
        let mut samples = std::collections::HashMap::new();
        for sample in &snapshot.bodies {
            if samples.insert(sample.handle, sample).is_some() {
                return Err(invalid("snapshot contains duplicate native body handles"));
            }
        }
        let mut poses = Vec::with_capacity(self.body_handles.len() - 1);
        for source_body_id in 1..self.body_handles.len() {
            let handle = self.body_handles[source_body_id]
                .ok_or_else(|| invalid("robot assembly is missing a source body"))?;
            let (index, generation) = handle.into_raw_parts();
            let backend_handle = [index, generation];
            let sample = samples
                .get(&backend_handle)
                .ok_or_else(|| invalid("robot snapshot has missing or stale body generation"))?;
            poses.push(RobotBodyPose {
                source_body_id,
                backend_handle,
                translation: sample.translation,
                rotation_xyzw: sample.rotation_xyzw,
            });
        }
        let frame = RobotPoseFrame {
            model_file_sha256: self.source_file_sha256.clone(),
            episode_id: snapshot.episode_id,
            global_step: snapshot.global_step,
            episode_step: snapshot.episode_step,
            poses,
        };
        frame.validate(&self.source_file_sha256, self.body_handles.len())?;
        Ok(frame)
    }

    /// No raw solver offset survives rebuilding an articulation.
    pub fn joint_state(&self, world: &PhysicsWorld) -> Result<Vec<(usize, f32, f32)>, RobotError> {
        self.joint_mapping
            .iter()
            .map(|mapping| {
                let (multibody, link_id) = world
                    .multibody_joints
                    .get(mapping.handle)
                    .ok_or_else(|| invalid("stale articulation handle"))?;
                let link = multibody
                    .link(link_id)
                    .ok_or_else(|| invalid("missing joint link"))?;
                if link.assembly_id() != mapping.backend_dof || link.joint().ndofs() != 1 {
                    return Err(invalid("articulation topology changed since construction"));
                }
                Ok((
                    mapping.source_joint,
                    mapping.reference + link.joint().coords()[3],
                    multibody.generalized_velocity()[mapping.backend_dof],
                ))
            })
            .collect()
    }

    /// Read each driven joint in the contract order from live native links.
    /// The caller owns the world and must separately bind this read to its
    /// instance token and snapshot tick; no inferred observation is produced.
    pub fn actuator_joint_feedback(
        &self,
        world: &PhysicsWorld,
        definition: &RobotDefinition,
    ) -> Result<[JointFeedbackChannel; ACTION_DIMENSION], RobotError> {
        if self.source_file_sha256 != definition.file_sha256()
            || self.actuator_joints != *definition.actuator_joint_ids()
        {
            return Err(invalid("joint feedback definition identity changed"));
        }
        let fields = &definition.model().fields;
        let mut channels = Vec::with_capacity(ACTION_DIMENSION);
        for source_joint in self.actuator_joints {
            let mut mappings = self
                .joint_mapping
                .iter()
                .filter(|mapping| mapping.source_joint == source_joint);
            let mapping = mappings
                .next()
                .ok_or_else(|| invalid("driven source joint mapping is missing"))?;
            if mappings.next().is_some()
                || source_joint == 0
                || source_joint >= fields.jnt_bodyid.len()
                || mapping.source_dof != fields.jnt_dofadr[source_joint]
            {
                return Err(invalid(
                    "driven source joint mapping is duplicated or changed",
                ));
            }
            let source_body = fields.jnt_bodyid[source_joint];
            let body_handle = self
                .body_handles
                .get(source_body)
                .and_then(|handle| *handle)
                .ok_or_else(|| invalid("driven source body generation is missing"))?;
            if world.bodies.get(body_handle).is_none() {
                return Err(invalid("driven source body generation is stale"));
            }
            let (multibody, link_id) = world
                .multibody_joints
                .get(mapping.handle)
                .ok_or_else(|| invalid("driven joint generation is stale"))?;
            let link = multibody
                .link(link_id)
                .ok_or_else(|| invalid("driven joint link is absent"))?;
            if link.rigid_body_handle() != body_handle
                || link.assembly_id() != mapping.backend_dof
                || link.joint().ndofs() != 1
                || mapping.backend_dof >= multibody.generalized_velocity().len()
            {
                return Err(invalid("driven native articulation topology changed"));
            }
            let position = mapping.reference + link.joint().coords()[3];
            let velocity = multibody.generalized_velocity()[mapping.backend_dof];
            if !position.is_finite() || !velocity.is_finite() {
                return Err(invalid("non-finite driven native joint feedback"));
            }
            let (joint_index, joint_generation) = mapping.handle.into_raw_parts();
            let (body_index, body_generation) = body_handle.into_raw_parts();
            channels.push(JointFeedbackChannel {
                source_joint,
                source_dof: mapping.source_dof,
                backend_dof: mapping.backend_dof,
                joint_handle: NativeHandleIdentity {
                    index: joint_index,
                    generation: joint_generation,
                },
                body_handle: NativeHandleIdentity {
                    index: body_index,
                    generation: body_generation,
                },
                position,
                velocity,
            });
        }
        channels
            .try_into()
            .map_err(|_| invalid("driven native feedback channel count changed"))
    }

    /// Map source-signed scalar actuator torques to world-space body pairs.
    ///
    /// The joint axis is expressed in the source child-body frame. This only
    /// maps already-computed torques; it does not evaluate BAM or step physics.
    /// The caller must separately guard the current world instance and tick.
    pub fn actuator_body_torques(
        &self,
        world: &PhysicsWorld,
        definition: &RobotDefinition,
        source_torques: &[f32; ACTION_DIMENSION],
    ) -> Result<Vec<BodyTorque>, RobotError> {
        if source_torques.iter().any(|torque| !torque.is_finite()) {
            return Err(invalid("non-finite source actuator torque"));
        }
        // Recheck all 14 source IDs, DOF slots, handles and live link owners
        // before constructing any contribution from a rebuilt articulation.
        let feedback = self.actuator_joint_feedback(world, definition)?;
        let fields = &definition.model().fields;
        let mut contributions = Vec::with_capacity(2 * ACTION_DIMENSION);
        for (index, channel) in feedback.iter().enumerate() {
            let source_joint = channel.source_joint;
            let child_source = *fields
                .jnt_bodyid
                .get(source_joint)
                .ok_or_else(|| invalid("driven source joint body is absent"))?;
            let parent_source = *fields
                .body_parentid
                .get(child_source)
                .ok_or_else(|| invalid("driven source joint parent is absent"))?;
            if parent_source == 0 || parent_source == child_source {
                return Err(invalid("driven source joint has no physical parent"));
            }
            let parent_handle = self
                .body_handles
                .get(parent_source)
                .and_then(|handle| *handle)
                .ok_or_else(|| invalid("driven native parent body is absent"))?;
            let child_handle = self
                .body_handles
                .get(child_source)
                .and_then(|handle| *handle)
                .ok_or_else(|| invalid("driven native child body is absent"))?;
            let parent = world
                .bodies
                .get(parent_handle)
                .ok_or_else(|| invalid("driven native parent body generation is stale"))?;
            let child = world
                .bodies
                .get(child_handle)
                .ok_or_else(|| invalid("driven native child body generation is stale"))?;
            if parent.user_data != parent_source as u128
                || child.user_data != child_source as u128
                || world
                    .multibody_joints
                    .joint_between(parent_handle, child_handle)
                    .is_none_or(|(handle, _, _)| {
                        handle.into_raw_parts()
                            != (channel.joint_handle.index, channel.joint_handle.generation)
                    })
            {
                return Err(invalid("driven native parent-child topology changed"));
            }
            for body in [parent, child] {
                let rotation = body.rotation();
                if !body.position().translation.is_finite()
                    || !rotation.is_finite()
                    || (rotation.length_squared() - 1.0).abs() > 1e-5
                {
                    return Err(invalid("driven native body pose is invalid"));
                }
            }
            let axis = engine_vector(
                *fields
                    .jnt_axis
                    .get(source_joint)
                    .ok_or_else(|| invalid("driven source joint axis is absent"))?,
            )?;
            if (axis.length_squared() - 1.0).abs() > 1e-5 {
                return Err(invalid("driven source joint axis is not unit length"));
            }
            let world_axis = *child.rotation() * axis;
            let child_torque = world_axis * source_torques[index];
            if !child_torque.is_finite() {
                return Err(invalid(
                    "world actuator torque cannot be represented in float32",
                ));
            }
            contributions.extend(BodyTorque::joint_pair(
                parent_handle,
                child_handle,
                child_torque.to_array(),
            ));
        }
        Ok(contributions)
    }

    /// Force measurements and source collision filtering are deliberately still
    /// unqualified. A gameplay/controller entry must not admit this stage alone.
    pub fn control_ready(&self) -> bool {
        false
    }
}

/// Build all physical robot bodies and every collision-enabled source geom.
/// `qpos` is explicit: the source keyframe root position is not a safe game spawn.
/// This performs kinematics and mass-property updates, with zero integrations.
pub fn build_structure(
    world: &mut PhysicsWorld,
    definition: &RobotDefinition,
    qpos: &[f64],
) -> Result<RobotAssembly, RobotError> {
    let model = definition.model();
    let fields = &model.fields;
    let source_poses = source_body_poses(definition, qpos)?;
    let poses: Vec<_> = source_poses
        .iter()
        .map(|pose| engine_pose(pose.position, pose.rotation_wxyz))
        .collect::<Result<_, _>>()?;
    // Construct potentially failing hulls before adding anything to the world.
    let mut shapes = Vec::new();
    for geom in 0..model.counts.ngeom {
        if fields.geom_bodyid[geom] == 0
            || fields.geom_contype[geom] == 0 && fields.geom_conaffinity[geom] == 0
        {
            continue;
        }
        let size = fields.geom_size[geom].map(|v| v as f32);
        let leaf = match fields.geom_type[geom] {
            2 => ColliderBuilder::ball(size[0]),
            3 => ColliderBuilder::capsule_y(size[1], size[0]),
            5 => ColliderBuilder::cylinder(size[1], size[0]),
            6 => ColliderBuilder::cuboid(size[0], size[2], size[1]),
            7 => {
                let mesh = fields.geom_dataid[geom] as usize;
                let hull = collision_hull::from_definition(definition, mesh)?;
                // Compiled vertices are already centered in the geom's frame.
                // Applying mesh_pos/mesh_quat again would double the compiler's
                // mesh-frame offset already contained in geom_pos/geom_quat.
                let points: Vec<_> = hull
                    .vertices
                    .iter()
                    .map(|v| engine_vector(*v))
                    .collect::<Result<_, _>>()?;
                ColliderBuilder::convex_mesh(points, &hull.triangles).ok_or_else(|| {
                    invalid(format!(
                        "native collision hull {mesh} is not backend-representable"
                    ))
                })?
            }
            kind => return Err(invalid(format!("unsupported collision geom type {kind}"))),
        };
        // Keep the collider's zero mass properties in their canonical identity
        // frame. Transforming a massless collider's frame otherwise forces the
        // native mass sum to re-diagonalize tiny source tensors incorrectly.
        // A single compound child retains the original shape and geom pose;
        // broader collision/manifold equivalence still needs its own gates.
        let shape = ColliderBuilder::compound(vec![(
            engine_pose(fields.geom_pos[geom], fields.geom_quat[geom])?,
            leaf.shape,
        )])
        .density(0.0)
        .friction(fields.geom_friction[geom][0] as f32)
        .restitution(0.0)
        .user_data(geom as u128);
        shapes.push((geom, shape));
    }
    let mut body_builders = Vec::new();
    for body in 1..model.counts.nbody {
        // The body and inertial local bases both change by B. Consequently the
        // principal diagonal also permutes X,Z,Y; rotating the frame alone with
        // unpermuted moments would silently change the physical tensor.
        let inertia = fields.body_inertia[body];
        let mass = fields.body_mass[body] as f32;
        let principal = Vector::new(inertia[0] as f32, inertia[2] as f32, inertia[1] as f32);
        if mass <= 0.0
            || !mass.is_finite()
            || principal.min_element() <= 0.0
            || !principal.is_finite()
        {
            return Err(invalid(
                "source mass/inertia cannot be represented in float32",
            ));
        }
        let properties = MassProperties::with_principal_inertia_frame(
            engine_vector(fields.body_ipos[body])?,
            mass,
            principal,
            engine_rotation(fields.body_iquat[body])?,
        );
        body_builders.push((
            body,
            RigidBodyBuilder::dynamic()
                .pose(poses[body])
                .additional_mass_properties(properties)
                .linear_damping(0.0)
                .angular_damping(0.0)
                .additional_solver_iterations(0)
                .can_sleep(false)
                .user_data(body as u128),
        ));
    }
    let damping = backend_scalars(&fields.dof_damping)?;
    let armature = backend_scalars(&fields.dof_armature)?;
    let friction = backend_scalars(&fields.dof_frictionloss)?;
    let mut joint_descriptions = Vec::new();
    let mut references = vec![0.0; model.counts.njnt];
    let mut displacements = vec![0.0; model.counts.njnt];
    for joint in 1..model.counts.njnt {
        let body = fields.jnt_bodyid[joint];
        let parent = fields.body_parentid[body];
        let axis = engine_vector(fields.jnt_axis[joint])?;
        if (axis.length_squared() - 1.0).abs() > 1e-5 {
            return Err(invalid("hinge axis is not a float32 unit vector"));
        }
        let local_frame2 = Pose::from_parts(
            engine_vector(fields.jnt_pos[joint])?,
            Rotation::from_rotation_arc(Vector::X, axis.normalize()),
        );
        let reference_pose = engine_pose(fields.body_pos[body], fields.body_quat[body])?;
        let mut description = GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
            .local_frame1(reference_pose * local_frame2)
            .local_frame2(local_frame2)
            .contacts_enabled(false);
        let reference = fields.qpos0[fields.jnt_qposadr[joint]];
        references[joint] = backend_scalar(reference)?;
        displacements[joint] = backend_scalar(qpos[fields.jnt_qposadr[joint]] - reference)?;
        if fields.jnt_limited[joint] {
            let range = [
                backend_scalar(fields.jnt_range[joint][0] - reference)?,
                backend_scalar(fields.jnt_range[joint][1] - reference)?,
            ];
            if range[0] >= range[1] {
                return Err(invalid("joint limit range collapses at float32 boundary"));
            }
            description = description.limits(JointAxis::AngX, range);
        }
        joint_descriptions.push((joint, parent, body, description));
    }
    // All source conversions and hull construction are checked before native
    // insertion. Unexpected backend/topology errors also remove every new body.
    let mut body_handles = vec![None; model.counts.nbody];
    let result = (|| {
        for (body, builder) in body_builders {
            body_handles[body] = Some(world.bodies.insert(builder));
        }
        let mut collision_handles = Vec::new();
        for (geom, shape) in shapes {
            let body = body_handles[fields.geom_bodyid[geom]]
                .ok_or_else(|| invalid("missing geom owner"))?;
            let handle = world
                .colliders
                .insert_with_parent(shape, body, &mut world.bodies);
            collision_handles.push((geom, handle));
        }
        for body in body_handles.iter().flatten() {
            world.bodies[*body].recompute_mass_properties_from_colliders(&world.colliders);
        }
        let mut joint_handles = Vec::new();
        for (joint, parent, body, description) in joint_descriptions {
            let handle = world
                .multibody_joints
                .insert(
                    body_handles[parent].ok_or_else(|| invalid("missing hinge parent"))?,
                    body_handles[body].ok_or_else(|| invalid("missing hinge child"))?,
                    description,
                    true,
                )
                .ok_or_else(|| invalid("source articulation is not an acyclic tree"))?;
            joint_handles.push((joint, handle));
        }
        let first = joint_handles
            .first()
            .ok_or_else(|| invalid("missing robot hinges"))?
            .1;
        let (multibody, _) = world
            .multibody_joints
            .get_mut(first)
            .ok_or_else(|| invalid("missing articulation"))?;
        multibody.forward_kinematics(&world.bodies, true);
        if multibody.ndofs() != model.counts.nv {
            return Err(invalid("backend/source degree-of-freedom count mismatch"));
        }
        multibody.damping_mut().fill(0.0);
        multibody.armature_mut().fill(0.0);
        multibody.frictions_mut().fill(0.0);
        let mut displacement = vec![0.0; model.counts.nv];
        let mut joint_mapping = Vec::new();
        // Free-root vector bases permute as X,Z,-Y; diagonal passive quantities are
        // insensitive to the sign, while hinge scalars retain their source sign.
        for (engine, source) in [0, 2, 1, 3, 5, 4].into_iter().enumerate() {
            multibody.damping_mut()[engine] = damping[source];
            multibody.armature_mut()[engine] = armature[source];
            multibody.frictions_mut()[engine] = friction[source];
        }
        // Obtain handles' link indices only after the whole tree was assembled.
        // The recorded scalar slot is checked again before every subsequent read.
        for (joint, handle) in joint_handles {
            let body = fields.jnt_bodyid[joint];
            let link = multibody
                .links()
                .find(|link| Some(link.rigid_body_handle()) == body_handles[body])
                .ok_or_else(|| invalid("missing source body link"))?;
            let slot = link.assembly_id();
            let dof = fields.jnt_dofadr[joint];
            if link.joint().ndofs() != 1 || slot >= model.counts.nv {
                return Err(invalid("source hinge is not one backend scalar DoF"));
            }
            displacement[slot] = displacements[joint];
            multibody.damping_mut()[slot] = damping[dof];
            multibody.armature_mut()[slot] = armature[dof];
            multibody.frictions_mut()[slot] = friction[dof];
            joint_mapping.push(SourceJointMapping {
                source_joint: joint,
                source_dof: dof,
                handle,
                backend_dof: slot,
                reference: references[joint],
            });
        }
        multibody.apply_displacements(&displacement);
        multibody.forward_kinematics(&world.bodies, false);
        multibody.update_rigid_bodies(&mut world.bodies, true);
        world
            .bodies
            .propagate_modified_body_positions_to_colliders(&mut world.colliders);
        Ok(RobotAssembly {
            body_handles: body_handles.clone(),
            collision_handles,
            joint_mapping,
            actuator_joints: *definition.actuator_joint_ids(),
            source_file_sha256: definition.file_sha256().into(),
        })
    })();
    if result.is_err() {
        for handle in body_handles.iter().rev().flatten() {
            world.remove_body(*handle);
        }
    }
    result
}

fn backend_scalar(value: f64) -> Result<f32, RobotError> {
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("source scalar cannot be represented in float32"))
    }
}

fn backend_scalars(values: &[f64]) -> Result<Vec<f32>, RobotError> {
    values.iter().copied().map(backend_scalar).collect()
}

fn engine_vector(source: [f64; 3]) -> Result<Vector, RobotError> {
    let result = Vector::from_array(source_to_engine_vector(source.map(|v| v as f32)));
    if !result.is_finite() {
        return Err(invalid("source vector cannot be represented in float32"));
    }
    Ok(result)
}
fn engine_rotation(source: [f64; 4]) -> Result<Rotation, RobotError> {
    Ok(Rotation::from_array(source_to_engine_rotation(
        source.map(|v| v as f32),
    )?))
}
fn engine_pose(position: [f64; 3], rotation: [f64; 4]) -> Result<Pose, RobotError> {
    Ok(Pose::from_parts(
        engine_vector(position)?,
        engine_rotation(rotation)?,
    ))
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod feedback_tests {
    use super::*;
    use std::{fs, path::Path};

    fn frozen_definitions() -> Vec<RobotDefinition> {
        let fixture_root = std::env::var("SOURCE_COLLISION_FIXTURES")
            .expect("explicit frozen metadata fixture root is required");
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(Path::new(&fixture_root).join("metadata_manifest.json")).unwrap(),
        )
        .unwrap();
        [0, 2]
            .into_iter()
            .map(|index| {
                let row = &manifest["profiles"][index]["definition"];
                RobotDefinition::load_json(
                    Path::new(row["path"].as_str().unwrap()),
                    row["sha256"].as_str().unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    fn source_quat_rotate([w, x, y, z]: [f64; 4], [vx, vy, vz]: [f64; 3]) -> [f64; 3] {
        let uv = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx];
        let uuv = [
            y * uv[2] - z * uv[1],
            z * uv[0] - x * uv[2],
            x * uv[1] - y * uv[0],
        ];
        [
            vx + 2.0 * (w * uv[0] + uuv[0]),
            vy + 2.0 * (w * uv[1] + uuv[1]),
            vz + 2.0 * (w * uv[2] + uuv[2]),
        ]
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn zero_step_actuator_torque_pairs_follow_source_axes_and_signs() {
        for definition in frozen_definitions() {
            let fields = &definition.model().fields;
            let mut qpos = fields.key_qpos[0].clone();
            let half = 0.31_f64;
            qpos[3..7].copy_from_slice(&[half.cos(), half.sin(), 0.0, 0.0]);
            let first = definition.actuator_joint_ids()[0];
            qpos[fields.jnt_qposadr[first]] += 0.17;
            let source_poses = source_body_poses(&definition, &qpos).unwrap();
            let mut simulation = crate::SimulationWorld::new();
            let assembly = build_structure(&mut simulation.world, &definition, &qpos).unwrap();
            let scalars = std::array::from_fn(|index| {
                let magnitude = 0.07 * (index + 1) as f32;
                if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            });
            let pairs = assembly
                .actuator_body_torques(&simulation.world, &definition, &scalars)
                .unwrap();
            assert_eq!(pairs.len(), 2 * ACTION_DIMENSION);
            for (index, pair) in pairs.chunks_exact(2).enumerate() {
                let joint = definition.actuator_joint_ids()[index];
                let child = fields.jnt_bodyid[joint];
                let parent = fields.body_parentid[child];
                assert_eq!(pair[0].body, assembly.body_handles[parent].unwrap());
                assert_eq!(pair[1].body, assembly.body_handles[child].unwrap());
                let source_world_axis =
                    source_quat_rotate(source_poses[child].rotation_wxyz, fields.jnt_axis[joint]);
                let expected_axis = source_to_engine_vector(source_world_axis.map(|v| v as f32));
                for component in 0..3 {
                    let expected = expected_axis[component] * scalars[index];
                    assert!(
                        (pair[1].world_torque[component] - expected).abs() < 2e-5,
                        "joint {joint}, component {component}: actual={} expected={expected}",
                        pair[1].world_torque[component]
                    );
                    assert_eq!(
                        pair[0].world_torque[component].to_bits(),
                        (-pair[1].world_torque[component]).to_bits()
                    );
                }
            }
            assert_eq!(simulation.snapshot().integration_count, 0);
            assert_eq!(simulation.snapshot().torque_update_count, 0);
            for body in assembly.body_handles.iter().flatten() {
                assert_eq!(simulation.world.bodies[*body].user_torque(), Vector::ZERO);
            }
        }
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn zero_step_actuator_torque_pairs_reject_stale_or_wrong_identity() {
        let definitions = frozen_definitions();
        let definition = &definitions[0];
        let mut simulation = crate::SimulationWorld::new();
        let mut assembly = build_structure(
            &mut simulation.world,
            definition,
            &definition.model().fields.key_qpos[0],
        )
        .unwrap();
        let scalar = [0.1; ACTION_DIMENSION];
        assert!(
            assembly
                .actuator_body_torques(&simulation.world, &definitions[1], &scalar)
                .is_err()
        );
        let mut nonfinite = scalar;
        nonfinite[11] = f32::NAN;
        assert!(
            assembly
                .actuator_body_torques(&simulation.world, definition, &nonfinite)
                .is_err()
        );
        let joint = definition.actuator_joint_ids()[0];
        let child = definition.model().fields.jnt_bodyid[joint];
        let parent = definition.model().fields.body_parentid[child];
        let original_parent = assembly.body_handles[parent].unwrap();
        let (index, generation) = original_parent.into_raw_parts();
        assembly.body_handles[parent] = Some(RigidBodyHandle::from_raw_parts(
            index,
            generation.wrapping_add(1),
        ));
        assert!(
            assembly
                .actuator_body_torques(&simulation.world, definition, &scalar)
                .is_err()
        );
        assembly.body_handles[parent] = Some(original_parent);
        let mapping_index = assembly
            .joint_mapping
            .iter()
            .position(|mapping| mapping.source_joint == joint)
            .unwrap();
        let original_joint = assembly.joint_mapping[mapping_index].handle;
        let (index, generation) = original_joint.into_raw_parts();
        assembly.joint_mapping[mapping_index].handle =
            MultibodyJointHandle::from_raw_parts(index, generation.wrapping_add(1));
        assert!(
            assembly
                .actuator_body_torques(&simulation.world, definition, &scalar)
                .is_err()
        );
        assembly.joint_mapping[mapping_index].handle = original_joint;
        assert_eq!(
            assembly
                .actuator_body_torques(&simulation.world, definition, &scalar)
                .unwrap()
                .len(),
            2 * ACTION_DIMENSION
        );
        assert_eq!(simulation.snapshot().integration_count, 0);
        assert_eq!(simulation.snapshot().torque_update_count, 0);
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn zero_step_joint_feedback_rejects_stale_and_duplicated_mappings() {
        for definition in frozen_definitions() {
            let mut simulation = crate::SimulationWorld::new();
            let mut assembly = build_structure(
                &mut simulation.world,
                &definition,
                &definition.model().fields.key_qpos[0],
            )
            .unwrap();
            assert!(
                assembly
                    .actuator_joint_feedback(&simulation.world, &definition)
                    .is_ok()
            );
            let first_joint = definition.actuator_joint_ids()[0];
            let mapping_index = assembly
                .joint_mapping
                .iter()
                .position(|mapping| mapping.source_joint == first_joint)
                .unwrap();
            let source_body = definition.model().fields.jnt_bodyid[first_joint];

            let original_joint = assembly.joint_mapping[mapping_index].handle;
            let (joint_index, generation) = original_joint.into_raw_parts();
            assembly.joint_mapping[mapping_index].handle =
                MultibodyJointHandle::from_raw_parts(joint_index, generation.wrapping_add(1));
            assert!(
                assembly
                    .actuator_joint_feedback(&simulation.world, &definition)
                    .is_err()
            );
            assembly.joint_mapping[mapping_index].handle = original_joint;

            let original_body = assembly.body_handles[source_body].unwrap();
            let (body_index, generation) = original_body.into_raw_parts();
            assembly.body_handles[source_body] = Some(RigidBodyHandle::from_raw_parts(
                body_index,
                generation.wrapping_add(1),
            ));
            assert!(
                assembly
                    .actuator_joint_feedback(&simulation.world, &definition)
                    .is_err()
            );
            assembly.body_handles[source_body] = Some(original_body);

            let duplicate_index = assembly
                .joint_mapping
                .iter()
                .position(|mapping| mapping.source_joint != first_joint)
                .unwrap();
            let original_source_joint = assembly.joint_mapping[duplicate_index].source_joint;
            assembly.joint_mapping[duplicate_index].source_joint = first_joint;
            assert!(
                assembly
                    .actuator_joint_feedback(&simulation.world, &definition)
                    .is_err()
            );
            assembly.joint_mapping[duplicate_index].source_joint = original_source_joint;
            assert!(
                assembly
                    .actuator_joint_feedback(&simulation.world, &definition)
                    .is_ok()
            );
            assert_eq!(simulation.snapshot().integration_count, 0);
            assert_eq!(simulation.snapshot().torque_update_count, 0);
        }
    }
}
