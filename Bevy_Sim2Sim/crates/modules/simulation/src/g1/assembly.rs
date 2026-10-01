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

    /// Generation-bound driven joint handles and articulation slots.
    /// The ignored actuator diagnostic is the only caller; assembly is unchanged.
    #[cfg(feature = "sim2sim_motor_row_trace")]
    #[allow(dead_code)]
    pub(super) fn driven_motor_trace_pairs(
        &self,
    ) -> Option<[(MultibodyJointHandle, usize); JOINT_COUNT]> {
        if self.driven.len() != JOINT_COUNT {
            return None;
        }
        let mut pairs = [(MultibodyJointHandle::invalid(), 0); JOINT_COUNT];
        for (index, joint) in self.driven.iter().enumerate() {
            pairs[index] = (joint.handle, joint.slot);
        }
        Some(pairs)
    }

    /// Current free-root motion from the completed generalized state. Rapier's
    /// staged solver writes qdot after its last rigid-body velocity refresh, so
    /// rb.angvel()/linvel() are one boundary old for this articulation.
    ///
    /// The free joint stores the root-frame-origin translation: its linear
    /// coordinates are that origin's world velocity (verified against pose
    /// increments), not a fabricated COM sensor. Angular velocity is common to
    /// every point on the root body. No backend state or pose is changed here.
    pub(super) fn current_root_velocity(
        &self,
        world: &PhysicsWorld,
    ) -> Result<RigidBodyVelocity<f32>, RobotError> {
        let first = self
            .driven
            .first()
            .ok_or_else(|| invalid("missing G1 joint mapping"))?;
        let (tree, _) = world
            .multibody_joints
            .get(first.handle)
            .ok_or_else(|| invalid("stale G1 articulation"))?;
        let root = tree.root();
        if root.rigid_body_handle() != self.bodies[0]
            || root.assembly_id() != 0
            || root.joint().ndofs() != 6
            || !root.joint().data.locked_axes.is_empty()
        {
            return Err(invalid("G1 root is not the original free six-DoF link"));
        }
        let current = tree.generalized_velocity();
        let velocity = root.joint().jacobian_mul_coordinates(current.as_slice());
        if !velocity.linvel.is_finite() || !velocity.angvel.is_finite() {
            return Err(invalid("non-finite completed G1 root velocity"));
        }
        Ok(velocity)
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
                (root.rotation().inverse() * self.current_root_velocity(world)?.angvel).to_array(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, path::Path};

    #[test]
    #[ignore = "requires G1_MODEL_DIR and G1_DEFINITION_SHA256"]
    fn frozen_usd_mass_pose_freefall_and_torque_sign() {
        let path = env::var("G1_MODEL_DIR").unwrap();
        let definition = G1Definition::load(
            &Path::new(&path).join("g1_physics.json"),
            &env::var("G1_DEFINITION_SHA256").unwrap(),
        )
        .unwrap();
        let mut sim = crate::SimulationWorld::with_game_frequency(50).unwrap();
        let source_root = SourcePose {
            position: [0., 0., 3.],
            rotation_wxyz: [1., 0., 0., 0.],
        };
        let root = engine_pose(&source_root).unwrap();
        let assembly = build(&mut sim.world, &definition, &source_root, &[0.; 43], 0.5).unwrap();
        let mut max_pose_error = 0.0_f32;
        let mut max_inertia_relative_error = 0.0_f32;
        let mut max_rotation_basis_error = 0.0_f32;
        for (i, source) in definition.model().bodies.iter().enumerate() {
            let native = &sim.world.bodies[assembly.bodies[i]];
            assert!(native.is_dynamic());
            let expected_pose = root * engine_pose(&source.rest_pose).unwrap();
            let position_error = (native.translation() - expected_pose.translation).length();
            max_pose_error = max_pose_error.max(position_error);
            assert!(
                position_error < 1e-4,
                "source rest position mismatch {}: {position_error}",
                source.name
            );
            for axis in [Vector::X, Vector::Y, Vector::Z] {
                // Quat::angle_between uses acos and overstates sub-ULP identity
                // drift; compare rotated basis vectors after normalization.
                let error = (native.rotation().normalize() * axis
                    - expected_pose.rotation.normalize() * axis)
                    .length();
                max_rotation_basis_error = max_rotation_basis_error.max(error);
                // The authored shoulder-roll local joint frames themselves
                // differ ~2.08e-4 rad from USD prim rest rotations; physics uses
                // joint frames, while visual zero rests retain this discrepancy.
                assert!(
                    error < 5e-4,
                    "source rest rotation mismatch {}: {error}",
                    source.name
                );
            }
            let mass = &native.mass_properties().local_mprops;
            assert!(
                (mass.mass() / source.mass as f32 - 1.).abs() < 2e-6,
                "source mass mismatch {}",
                source.name
            );
            assert!((mass.local_com - engine_vector(source.center_of_mass)).length() < 1e-6);
            let d = source.principal_inertia;
            let expected = MassProperties::with_principal_inertia_frame(
                engine_vector(source.center_of_mass),
                source.mass as f32,
                Vector::new(d[0] as f32, d[2] as f32, d[1] as f32),
                engine_rotation(source.principal_axes_wxyz).unwrap(),
            )
            .reconstruct_inertia_matrix();
            let actual = mass.reconstruct_inertia_matrix();
            let scale = expected
                .to_cols_array()
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f32, f32::max);
            let difference = (actual - expected)
                .to_cols_array()
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f32, f32::max)
                / scale;
            max_inertia_relative_error = max_inertia_relative_error.max(difference);
            assert!(
                difference < 2e-5,
                "source inertia mismatch {}: {difference}",
                source.name
            );
        }
        let mut gravity_velocities = Vec::new();
        for _ in 0..5 {
            sim.step_with_torques(&[]).unwrap();
            gravity_velocities.push(sim.world.bodies[assembly.root_handle()].linvel().y);
        }
        println!(
            "G1_IMPORT mass_pose_checked max_pose_error={max_pose_error}; max_inertia_relative_error={max_inertia_relative_error}; max_rotation_basis_error={max_rotation_basis_error}; gravity_velocities={gravity_velocities:?}"
        );
        let state = assembly.state(&sim.world).unwrap();
        let root = &sim.world.bodies[assembly.root_handle()];
        assert!(
            (root.linvel().y + 0.981).abs() < 1e-4,
            "free root gravity velocity {:?}",
            root.linvel()
        );
        assert!(state.velocities.iter().all(|dq| dq.abs() < 1e-3));
        assert_eq!(sim.snapshot().active_contact_pair_count, 0);
        sim.world.gravity = Vector::ZERO;
        let mut torque = [0.; 43];
        torque[3] = 1.;
        let pairs = assembly.torques(&sim.world, &torque).unwrap();
        let total = pairs
            .iter()
            .fold(Vector::ZERO, |v, t| v + Vector::from_array(t.world_torque));
        assert!(total.length() < 1e-7);
        sim.step_with_torques(&pairs).unwrap();
        assert!(
            assembly.state(&sim.world).unwrap().velocities[3] > 0.,
            "positive knee torque must accelerate positive source coordinate"
        );
        println!(
            "G1_IMPORT max_pose_error={max_pose_error}; max_inertia_relative_error={max_inertia_relative_error}; max_rotation_basis_error={max_rotation_basis_error}; freefall_50Hz=passed; positive_knee_torque=passed"
        );
    }

    #[test]
    #[ignore = "requires frozen G1 definition plus G1_INERTIA_ORACLE, G1_INERTIA_ORACLE_SHA256, G1_IMPORT_OUTPUT and G1_CODE_COMMIT"]
    fn independent_usd_tensor_and_axis_oracle_without_integration() {
        use robot_minigame::g1::{contract::JOINT_NAMES, policy::bound_bytes};
        use serde_json::{Value, json};
        let definition = G1Definition::load(
            &Path::new(&env::var("G1_MODEL_DIR").unwrap()).join("g1_physics.json"),
            &env::var("G1_DEFINITION_SHA256").unwrap(),
        )
        .unwrap();
        let bytes = bound_bytes(
            Path::new(&env::var("G1_INERTIA_ORACLE").unwrap()),
            &env::var("G1_INERTIA_ORACLE_SHA256").unwrap(),
        )
        .unwrap();
        let oracle: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(oracle["usd_sha256"].as_str().unwrap(), USD_SHA256);
        assert_eq!(oracle["bodies"].as_array().unwrap().len(), 53);
        let case = &oracle["independent_joint_frame_case"];
        let positions = std::array::from_fn(|i| {
            case["joint_positions_by_name"][JOINT_NAMES[i]]
                .as_f64()
                .unwrap() as f32
        });
        let source_root = SourcePose {
            position: [0., 0., 3.],
            rotation_wxyz: [1., 0., 0., 0.],
        };
        let mut sim = crate::SimulationWorld::with_game_frequency(50).unwrap();
        let assembly = build(&mut sim.world, &definition, &source_root, &positions, 0.5).unwrap();
        let mut mass_rows = Vec::new();
        let mut pose_rows = Vec::new();
        let mut axis_rows = Vec::new();
        let mut max_tensor = 0.0_f64;
        let mut max_position = 0.0_f64;
        let mut max_rotation = 0.0_f64;
        let mut max_axis = 0.0_f64;
        for i in 0..53 {
            let expected = &oracle["bodies"][i];
            assert_eq!(
                expected["name"].as_str().unwrap(),
                definition.model().bodies[i].name
            );
            let body = &sim.world.bodies[assembly.bodies[i]];
            let actual = body
                .mass_properties()
                .local_mprops
                .reconstruct_inertia_matrix()
                .to_cols_array();
            let mut tensor = [[0_f64; 3]; 3];
            let mut norm = 0_f64;
            let mut difference = 0_f64;
            for r in 0..3 {
                for c in 0..3 {
                    let value = expected["engine_tensor_row_major"][r][c].as_f64().unwrap();
                    tensor[r][c] = f64::from(actual[c * 3 + r]);
                    norm = norm.max(value.abs());
                    difference = difference.max((tensor[r][c] - value).abs());
                }
            }
            let relative = difference / norm;
            max_tensor = max_tensor.max(relative);
            mass_rows.push(json!({"body":i,"name":definition.model().bodies[i].name,"native_tensor_row_major":tensor,"expected_tensor_row_major":expected["engine_tensor_row_major"],"relative_max_error":relative}));
            let expected_pose = &case["bodies"][i];
            let mut pe = 0_f64;
            let mut re = 0_f64;
            for k in 0..3 {
                pe = pe.max(
                    (f64::from(body.translation()[k])
                        - expected_pose["position_engine"][k].as_f64().unwrap())
                    .abs(),
                );
            }
            for (c, axis) in [Vector::X, Vector::Y, Vector::Z].into_iter().enumerate() {
                let actual = body.rotation().normalize() * axis;
                for r in 0..3 {
                    re = re.max(
                        (f64::from(actual[r])
                            - expected_pose["rotation_engine_row_major"][r][c]
                                .as_f64()
                                .unwrap())
                        .abs(),
                    );
                }
            }
            max_position = max_position.max(pe);
            max_rotation = max_rotation.max(re);
            pose_rows.push(json!({"body":i,"position_max_error":pe,"rotation_basis_max_error":re}));
        }
        for (i, mapping) in assembly.driven.iter().enumerate() {
            let expected = case["joint_axes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["name"] == JOINT_NAMES[i])
                .unwrap();
            let actual = *sim.world.bodies[mapping.parent].rotation() * mapping.parent_axis;
            let mut error = 0_f64;
            for k in 0..3 {
                error = error.max(
                    (f64::from(actual[k]) - expected["engine_world_axis"][k].as_f64().unwrap())
                        .abs(),
                );
            }
            max_axis = max_axis.max(error);
            axis_rows.push(json!({"name":JOINT_NAMES[i],"native_engine_axis":actual.to_array(),"expected_engine_axis":expected["engine_world_axis"],"max_error":error}));
        }
        assert_eq!(sim.snapshot().integration_count, 0);
        let passed =
            max_tensor < 2e-5 && max_position < 2e-5 && max_rotation < 2e-5 && max_axis < 2e-5;
        let report = json!({"check":"independent_usd_tensor_and_axis_oracle","passed":passed,"integration_count":0,
            "code_commit":env::var("G1_CODE_COMMIT").unwrap(),"definition_sha256":definition.file_sha256(),
            "oracle_sha256":env::var("G1_INERTIA_ORACLE_SHA256").unwrap(),"usd_sha256":USD_SHA256,
            "formula":oracle["formula"],"max_tensor_relative_error":max_tensor,"max_position_error_m":max_position,
            "max_rotation_basis_error":max_rotation,"max_axis_error":max_axis,"bodies":mass_rows,"poses":pose_rows,"joint_axes":axis_rows});
        std::fs::write(
            env::var("G1_IMPORT_OUTPUT").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        println!(
            "G1_INDEPENDENT_IMPORT passed={passed}; integrations=0; tensor_relative={max_tensor}; pose_m={max_position}; rotation_basis={max_rotation}; axis={max_axis}"
        );
        assert!(passed, "independent source tensor/axis oracle mismatch");
    }

    #[test]
    #[ignore = "five real 50Hz freefall boundaries; requires G1_MODEL_DIR, G1_DEFINITION_SHA256, G1_IMPORT_OUTPUT, G1_CODE_COMMIT"]
    fn five_tick_freefall_boundary_instrument() {
        use serde_json::json;
        let definition = G1Definition::load(
            &Path::new(&env::var("G1_MODEL_DIR").unwrap()).join("g1_physics.json"),
            &env::var("G1_DEFINITION_SHA256").unwrap(),
        )
        .unwrap();
        let mut sim = crate::SimulationWorld::with_game_frequency(50).unwrap();
        let root_pose = SourcePose {
            position: [0., 0., 3.],
            rotation_wxyz: [1., 0., 0., 0.],
        };
        let assembly = build(&mut sim.world, &definition, &root_pose, &[0.; 43], 0.5).unwrap();
        let sample = |sim: &crate::SimulationWorld| {
            let mut bodies = Vec::new();
            let mut total_mass = 0_f64;
            let mut weighted_position = [0_f64; 3];
            let mut weighted_cached_com = [0_f64; 3];
            let mut weighted_velocity = [0_f64; 3];
            let active: Vec<_> = sim.world.islands.active_bodies().collect();
            for (i, handle) in assembly.bodies.iter().enumerate() {
                let body = &sim.world.bodies[*handle];
                let mass = f64::from(body.mass());
                let com = body.position() * body.local_center_of_mass();
                total_mass += mass;
                for k in 0..3 {
                    weighted_position[k] += mass * f64::from(com[k]);
                    weighted_cached_com[k] += mass * f64::from(body.center_of_mass()[k]);
                    weighted_velocity[k] += mass * f64::from(body.linvel()[k]);
                }
                bodies.push(json!({"body":i,"mass":mass,"effective_mass":body.mass_properties().effective_mass().to_array(),
                    "pose_com":com.to_array(),"cached_com":body.center_of_mass().to_array(),"linvel":body.linvel().to_array(),
                    "angvel":body.angvel().to_array(),"user_force":body.user_force().to_array(),"user_torque":body.user_torque().to_array(),
                    "gravity_scale":body.gravity_scale(),"enabled":body.is_enabled(),"sleeping":body.is_sleeping(),"in_active_island":active.contains(handle),
                    // Frozen Rapier's public Debug contains its private force
                    // cache and change flags. Capture it read-only, without a
                    // vendor patch or pretending expected gravity is measured.
                    "native_debug":format!("{body:?}")}));
            }
            let root = &sim.world.bodies[assembly.root_handle()];
            let (tree, _) = sim
                .world
                .multibody_joints
                .get(assembly.driven[0].handle)
                .unwrap();
            let snapshot = sim.snapshot();
            json!({"tick":snapshot.episode_step,"integrations":snapshot.integration_count,"torque_updates":snapshot.torque_update_count,
                "total_mass":total_mass,"whole_com_from_poses":weighted_position.map(|v|v/total_mass),
                "whole_com_from_cache":weighted_cached_com.map(|v|v/total_mass),"whole_com_velocity":weighted_velocity.map(|v|v/total_mass),
                "root_position_engine":root.translation().to_array(),"root_body_com_velocity":root.linvel().to_array(),
                "root_origin_velocity":(root.linvel()+root.angvel().cross(root.translation()-root.center_of_mass())).to_array(),
                "generalized_velocity":tree.generalized_velocity().as_slice(),"generalized_acceleration":tree.generalized_acceleration().as_slice(),
                "active_body_count":active.len(),"active_contact_pairs":snapshot.active_contact_pair_count,"bodies":bodies})
        };
        let mut rows = vec![sample(&sim)];
        for _ in 0..5 {
            sim.step_with_torques(&[]).unwrap();
            rows.push(sample(&sim));
        }
        let report = json!({"experiment":"five_tick_freefall_instrument_only","code_commit":env::var("G1_CODE_COMMIT").unwrap(),
            "definition_sha256":definition.file_sha256(),"usd_sha256":USD_SHA256,"root_source_position":[0.,0.,3.],
            "initial_q":([0.;43].as_slice()),"floor":false,"controller":false,"gravity_engine":sim.world.gravity.to_array(),
            "configuration":sim.configuration(),"rows":rows});
        std::fs::write(
            env::var("G1_IMPORT_OUTPUT").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        for row in &rows {
            println!(
                "G1_FREEFALL tick={} root_vy={} whole_com_vy={} generalized_root_vy={} active={}",
                row["tick"],
                row["root_body_com_velocity"][1],
                row["whole_com_velocity"][1],
                row["generalized_velocity"][1],
                row["active_body_count"]
            );
        }
        assert_eq!(sim.snapshot().integration_count, 5);
        assert_eq!(sim.snapshot().active_contact_pair_count, 0);
    }

    #[test]
    #[ignore = "one 50Hz pulse; requires frozen G1 definition, G1_IMPORT_OUTPUT and G1_CODE_COMMIT"]
    fn completed_root_gyro_matches_current_native_velocity() {
        use serde_json::json;
        let definition = G1Definition::load(
            &Path::new(&env::var("G1_MODEL_DIR").unwrap()).join("g1_physics.json"),
            &env::var("G1_DEFINITION_SHA256").unwrap(),
        )
        .unwrap();
        let mut sim = crate::SimulationWorld::with_game_frequency(50).unwrap();
        // Interior coordinates avoid a limit/contact bias contaminating the
        // independent rotation-increment check of a single completed boundary.
        let positions = std::array::from_fn(|i| {
            let limits = definition.model().joints[definition.driven_joints()[i]].limits;
            (0.63 * limits[0] + 0.37 * limits[1]) as f32
        });
        let root_pose = SourcePose {
            position: [0., 0., 3.],
            rotation_wxyz: [
                (std::f64::consts::PI / 12.).cos(),
                0.,
                0.,
                (std::f64::consts::PI / 12.).sin(),
            ],
        };
        let assembly = build(&mut sim.world, &definition, &root_pose, &positions, 0.5).unwrap();
        let root_handle = assembly.root_handle();
        let before = *sim.world.bodies[root_handle].position();
        sim.step_with_torques(&[BodyTorque {
            body: root_handle,
            world_torque: [1., 0.2, 0.3],
        }])
        .unwrap();
        let root = &sim.world.bodies[root_handle];
        let (tree, _) = sim
            .world
            .multibody_joints
            .get(assembly.driven[0].handle)
            .unwrap();
        let link = tree.root();
        assert_eq!(link.rigid_body_handle(), root_handle);
        assert_eq!(link.joint().ndofs(), 6);
        let velocity = link.joint().jacobian_mul_coordinates(
            &tree.generalized_velocity().as_slice()[link.assembly_id()..],
        );
        let expected_body =
            engine_to_source_vector((root.rotation().inverse() * velocity.angvel).to_array());
        let measured = assembly
            .state(&sim.world)
            .unwrap()
            .root_angular_velocity_body;
        let delta = (*root.rotation() * before.rotation.inverse()).normalize();
        let [x, y, z, w] = delta.to_array();
        let imaginary = Vector::new(x, y, z);
        let angle = 2. * imaginary.length().atan2(w);
        let finite_rotation_velocity = imaginary.normalize() * angle / 0.02;
        let finite_origin_velocity = (root.translation() - before.translation) / 0.02;
        let error = (Vector::from_array(measured) - Vector::from_array(expected_body)).length();
        let rotation_fd_error = (finite_rotation_velocity - velocity.angvel).length();
        let origin_fd_error = (finite_origin_velocity - velocity.linvel).length();
        let report = json!({"experiment":"completed_root_gyro_single_pulse","code_commit":env::var("G1_CODE_COMMIT").unwrap(),
            "definition_sha256":definition.file_sha256(),"integration_count":sim.snapshot().integration_count,"configuration":sim.configuration(),
            "world_torque_engine":[1.,0.2,0.3],"cached_rigid_body_omega_engine":root.angvel().to_array(),
            "current_native_omega_engine":velocity.angvel.to_array(),"finite_rotation_velocity_engine":finite_rotation_velocity.to_array(),
            "g1_gyro_body_source":measured,"expected_gyro_body_source":expected_body,"gyro_error":error,
            "finite_rotation_velocity_error":rotation_fd_error,"current_native_linear_engine":velocity.linvel.to_array(),
            "finite_origin_velocity_engine":finite_origin_velocity.to_array(),"finite_origin_velocity_error":origin_fd_error,
            "root_origin_position_engine":root.translation().to_array(),"root_com_position_engine":root.center_of_mass().to_array()});
        std::fs::write(
            env::var("G1_IMPORT_OUTPUT").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        println!(
            "G1_ROOT_GYRO cached={:?}; current={:?}; gyro_error={error}; rotation_fd_error={rotation_fd_error}; origin_fd_error={origin_fd_error}",
            root.angvel(),
            velocity.angvel
        );
        assert_eq!(sim.snapshot().integration_count, 1);
        assert_eq!(sim.snapshot().active_contact_pair_count, 0);
        assert!(
            velocity.angvel.length() > 1e-3,
            "pulse must generate observable angular velocity"
        );
        assert!(
            rotation_fd_error < 1e-4,
            "native omega does not match independent pose increment"
        );
        assert!(
            origin_fd_error < 3e-5,
            "native linear motion is not root-origin coordinate velocity"
        );
        assert!(
            error < 1e-6,
            "G1 sensor read is from a different native boundary"
        );
    }
}
