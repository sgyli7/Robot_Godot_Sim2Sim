//! USD rigid bodies, fixed links and 43 native scalar joints; no pose playback.

use crate::{BodyTorque, StepSnapshot};
use rapier3d::{
    math::{Pose, Rotation, Vector},
    prelude::*,
};
use robot_minigame::{
    RobotError,
    basis::{
        engine_to_source_rotation, engine_to_source_vector, source_to_engine_rotation,
        source_to_engine_vector,
    },
    g1::{
        actuator,
        contract::{G1State, JOINT_COUNT},
        definition::{
            G1BodyFrame, G1BodyPose, G1Definition, JointKind, Shape, SourcePose, USD_SHA256,
        },
    },
};

struct DrivenJoint {
    handle: MultibodyJointHandle,
    slot: usize,
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    parent_axis: Vector,
}

/// Generation-bound handles remain private to the physics worker.
pub struct G1Assembly {
    bodies: Vec<RigidBodyHandle>,
    driven: Vec<DrivenJoint>,
    definition_sha256: String,
}

impl G1Assembly {
    pub fn root_handle(&self) -> RigidBodyHandle {
        self.bodies[0]
    }
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
    pub fn definition_sha256(&self) -> &str {
        &self.definition_sha256
    }

    pub fn state(&self, world: &PhysicsWorld) -> Result<G1State, RobotError> {
        let mut positions = [0.; JOINT_COUNT];
        let mut velocities = [0.; JOINT_COUNT];
        for (i, mapping) in self.driven.iter().enumerate() {
            let (tree, link_id) = world
                .multibody_joints
                .get(mapping.handle)
                .ok_or_else(|| invalid("stale G1 joint"))?;
            let link = tree
                .link(link_id)
                .ok_or_else(|| invalid("missing G1 link"))?;
            if link.rigid_body_handle() != mapping.child
                || link.assembly_id() != mapping.slot
                || link.joint().ndofs() != 1
            {
                return Err(invalid("G1 topology changed"));
            }
            positions[i] = link.joint().coords()[3];
            velocities[i] = tree.generalized_velocity()[mapping.slot];
        }
        let root = world
            .bodies
            .get(self.bodies[0])
            .ok_or_else(|| invalid("stale G1 root"))?;
        let state = G1State {
            positions,
            velocities,
            root_rotation_wxyz: engine_to_source_rotation(root.rotation().to_array())?,
            root_angular_velocity_body: engine_to_source_vector(
                (root.rotation().inverse() * root.angvel()).to_array(),
            ),
        };
        state.validate()?;
        Ok(state)
    }

    pub fn torques(
        &self,
        world: &PhysicsWorld,
        values: &[f32; JOINT_COUNT],
    ) -> Result<Vec<BodyTorque>, RobotError> {
        self.state(world)?;
        let mut torques = Vec::with_capacity(86);
        for (i, mapping) in self.driven.iter().enumerate() {
            let parent = world
                .bodies
                .get(mapping.parent)
                .ok_or_else(|| invalid("stale torque parent"))?;
            let torque = *parent.rotation() * mapping.parent_axis * values[i];
            if !torque.is_finite() {
                return Err(invalid("non-finite G1 torque"));
            }
            torques.extend(BodyTorque::joint_pair(
                mapping.parent,
                mapping.child,
                torque.to_array(),
            ));
        }
        Ok(torques)
    }

    pub fn frame(&self, snapshot: &StepSnapshot) -> Result<G1BodyFrame, RobotError> {
        let mut bodies = Vec::with_capacity(53);
        for (body, handle) in self.bodies.iter().enumerate() {
            let (index, generation) = handle.into_raw_parts();
            let sample = snapshot
                .bodies
                .iter()
                .find(|b| b.handle == [index, generation])
                .ok_or_else(|| invalid("G1 completed snapshot lacks live body"))?;
            bodies.push(G1BodyPose {
                body,
                translation: sample.translation,
                rotation_xyzw: sample.rotation_xyzw,
            });
        }
        let frame = G1BodyFrame {
            usd_sha256: USD_SHA256.into(),
            episode_id: snapshot.episode_id,
            source_tick: snapshot.episode_step,
            sim_time: snapshot.episode_seconds,
            bodies,
        };
        frame.validate()?;
        Ok(frame)
    }
}

/// Initial root position and joint state are explicit source SI values. Pose
/// installation below happens once before the first integration, never at runtime.
/// `contact_friction` is a named diagnostic parameter until material parity passes.
pub fn build(
    world: &mut PhysicsWorld,
    definition: &G1Definition,
    root: &SourcePose,
    initial_positions: &[f32; JOINT_COUNT],
    contact_friction: f32,
) -> Result<G1Assembly, RobotError> {
    if !contact_friction.is_finite()
        || contact_friction < 0.
        || initial_positions.iter().any(|q| !q.is_finite())
    {
        return Err(invalid("invalid G1 initialization"));
    }
    let model = definition.model();
    let root = engine_pose(root)?;
    let mut shapes = Vec::new();
    for c in &model.collisions {
        let (local, leaf) = match &c.shape {
            Shape::ConvexHull { points } => {
                let vertices: Vec<_> = points.iter().map(|p| engine_vector(*p)).collect();
                (
                    Pose::IDENTITY,
                    ColliderBuilder::convex_hull(&vertices)
                        .ok_or_else(|| invalid(format!("cannot cook source hull {}", c.path)))?,
                )
            }
            Shape::Sphere { radius, local_pose } => (
                engine_pose(local_pose)?,
                ColliderBuilder::ball(*radius as f32),
            ),
            Shape::Capsule {
                radius,
                half_height,
                axis,
                local_pose,
            } => {
                let v = engine_vector(axis_vector(axis)?);
                let frame = engine_pose(local_pose)?
                    * Pose::from_parts(Vector::ZERO, Rotation::from_rotation_arc(Vector::Y, v));
                (
                    frame,
                    ColliderBuilder::capsule_y(*half_height as f32, *radius as f32),
                )
            }
            Shape::Box {
                half_extents,
                local_pose,
            } => (
                engine_pose(local_pose)?,
                ColliderBuilder::cuboid(
                    half_extents[0] as f32,
                    half_extents[2] as f32,
                    half_extents[1] as f32,
                ),
            ),
        };
        // Massless collider identity avoids re-diagonalizing tiny source tensors.
        // All robot links are group 1 and exclude group 1: whole-articulation
        // self collision is disabled in Arena, including nonadjacent links.
        let collider = ColliderBuilder::compound(vec![(local, leaf.shape)])
            .density(0.)
            .friction(contact_friction)
            .restitution(0.)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                Group::ALL ^ Group::GROUP_1,
                InteractionTestMode::And,
            ));
        shapes.push((c.body, collider));
    }
    let mut bodies = Vec::new();
    let result = (|| {
        for body in &model.bodies {
            let inertia = body.principal_inertia;
            let properties = MassProperties::with_principal_inertia_frame(
                engine_vector(body.center_of_mass),
                body.mass as f32,
                Vector::new(inertia[0] as f32, inertia[2] as f32, inertia[1] as f32),
                engine_rotation(body.principal_axes_wxyz)?,
            );
            bodies.push(
                world.bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .pose(root * engine_pose(&body.rest_pose)?)
                        .additional_mass_properties(properties)
                        .can_sleep(false)
                        .linear_damping(0.)
                        .angular_damping(0.)
                        .additional_solver_iterations(0),
                ),
            );
        }
        for (body, shape) in shapes {
            world
                .colliders
                .insert_with_parent(shape, bodies[body], &mut world.bodies);
        }
        for body in &bodies {
            world.bodies[*body].recompute_mass_properties_from_colliders(&world.colliders);
        }
        let mut joints = vec![None; model.joints.len()];
        let mut inserted = vec![false; bodies.len()];
        inserted[0] = true;
        for _ in 0..model.joints.len() {
            let mut changed = false;
            for (i, joint) in model.joints.iter().enumerate() {
                if joints[i].is_some() || !inserted[joint.parent] {
                    continue;
                }
                let align = Pose::from_parts(
                    Vector::ZERO,
                    Rotation::from_rotation_arc(
                        Vector::X,
                        engine_vector(axis_vector(&joint.axis)?),
                    ),
                );
                let mask = if joint.kind == JointKind::Fixed {
                    JointAxesMask::LOCKED_FIXED_AXES
                } else {
                    JointAxesMask::LOCKED_REVOLUTE_AXES
                };
                let mut desc = GenericJointBuilder::new(mask)
                    .local_frame1(engine_pose(&joint.frame_parent)? * align)
                    .local_frame2(engine_pose(&joint.frame_child)? * align)
                    .contacts_enabled(false);
                if joint.kind == JointKind::Revolute {
                    desc = desc.limits(JointAxis::AngX, joint.limits.map(|v| v as f32));
                }
                joints[i] = Some(
                    world
                        .multibody_joints
                        .insert(bodies[joint.parent], bodies[joint.child], desc, true)
                        .ok_or_else(|| invalid("G1 tree insertion failed"))?,
                );
                inserted[joint.child] = true;
                changed = true;
            }
            if !changed {
                break;
            }
        }
        if joints.iter().any(Option::is_none) {
            return Err(invalid("G1 topology not fully inserted"));
        }
        let first = joints[0].unwrap();
        let (tree, _) = world
            .multibody_joints
            .get_mut(first)
            .ok_or_else(|| invalid("missing G1 articulation"))?;
        tree.forward_kinematics(&world.bodies, true);
        if tree.ndofs() != 49 || tree.links().count() != 53 {
            return Err(invalid(
                "G1 must have 6 free-root + 43 joint DOFs and 53 links",
            ));
        }
        tree.damping_mut().fill(0.);
        tree.armature_mut().fill(0.);
        tree.frictions_mut().fill(0.);
        let mut displacement = vec![0.; 49];
        let mut driven = Vec::with_capacity(43);
        for (wbc, &id) in definition.driven_joints().iter().enumerate() {
            let joint = &model.joints[id];
            if initial_positions[wbc] < joint.limits[0] as f32 - 1e-5
                || initial_positions[wbc] > joint.limits[1] as f32 + 1e-5
            {
                return Err(invalid(format!(
                    "initial position outside source limit: {}",
                    joint.name
                )));
            }
            let link = tree
                .links()
                .find(|link| link.rigid_body_handle() == bodies[joint.child])
                .ok_or_else(|| invalid("G1 hinge has no link"))?;
            let slot = link.assembly_id();
            if link.joint().ndofs() != 1 || slot < 6 || slot >= 49 {
                return Err(invalid("G1 hinge slot mismatch"));
            }
            tree.armature_mut()[slot] = actuator::parameters(wbc)?.armature;
            displacement[slot] = initial_positions[wbc];
            driven.push(DrivenJoint {
                handle: joints[id].unwrap(),
                slot,
                parent: bodies[joint.parent],
                child: bodies[joint.child],
                parent_axis: engine_rotation(joint.frame_parent.rotation_wxyz)?
                    * engine_vector(axis_vector(&joint.axis)?),
            });
        }
        tree.apply_displacements(&displacement);
        tree.forward_kinematics(&world.bodies, false);
        tree.update_rigid_bodies(&mut world.bodies, true);
        world
            .bodies
            .propagate_modified_body_positions_to_colliders(&mut world.colliders);
        let assembly = G1Assembly {
            bodies: bodies.clone(),
            driven,
            definition_sha256: definition.file_sha256().into(),
        };
        let state = assembly.state(world)?;
        for i in 0..43 {
            if (state.positions[i] - initial_positions[i]).abs() > 1e-5 {
                return Err(invalid(
                    "G1 native initialization does not match source joint sign/order",
                ));
            }
        }
        Ok(assembly)
    })();
    if result.is_err() {
        for body in bodies.iter().rev() {
            world.remove_body(*body);
        }
    }
    result
}

fn engine_vector(v: [f64; 3]) -> Vector {
    Vector::from_array(source_to_engine_vector(v.map(|x| x as f32)))
}
fn engine_rotation(q: [f64; 4]) -> Result<Rotation, RobotError> {
    Ok(Rotation::from_array(source_to_engine_rotation(
        q.map(|x| x as f32),
    )?))
}
fn engine_pose(p: &SourcePose) -> Result<Pose, RobotError> {
    Ok(Pose::from_parts(
        engine_vector(p.position),
        engine_rotation(p.rotation_wxyz)?,
    ))
}
fn axis_vector(axis: &str) -> Result<[f64; 3], RobotError> {
    match axis {
        "X" => Ok([1., 0., 0.]),
        "Y" => Ok([0., 1., 0.]),
        "Z" => Ok([0., 0., 1.]),
        _ => Err(invalid("unknown USD axis")),
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}
