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
    kinematics::source_body_poses,
};

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
