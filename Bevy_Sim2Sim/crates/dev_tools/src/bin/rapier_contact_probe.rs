//! Bounded contact-row evidence; no BAM or source-target qualification.

#[allow(dead_code)]
#[path = "rapier_observation_probe.rs"]
mod primitives;

use rapier3d::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf};

type FrozenJacobians = Vec<(RigidBodyHandle, MultibodyIndex, Vec<Vec<f32>>)>;

fn scene(dynamic_prop: bool, reverse: bool, bouncy: bool) -> (PhysicsWorld, MultibodyJointHandle) {
    let mut world = PhysicsWorld::new();
    world.gravity = -Vector::Y * 9.81;
    world.integration_parameters.dt = 1.0 / 60.0;
    world.integration_parameters.num_solver_iterations = 1;
    world.integration_parameters.max_ccd_substeps = 1;
    let ground = |world: &mut PhysicsWorld| {
        world.insert(
            if dynamic_prop {
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(0.0, -0.1, 0.0))
                    .can_sleep(false)
            } else {
                RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0))
            },
            ColliderBuilder::cuboid(3.0, 0.1, 3.0).restitution(if bouncy { 0.6 } else { 0.0 }),
        );
    };
    if !reverse {
        ground(&mut world);
    }
    let props = MassProperties::new(Vector::ZERO, 1.0, Vector::splat(0.02));
    let root = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::Y * 0.08)
            .additional_mass_properties(props)
            .can_sleep(false),
    );
    let child = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.3, 0.08, 0.0))
            .additional_mass_properties(props)
            .can_sleep(false),
    );
    world.insert_collider(
        ColliderBuilder::cuboid(0.1, 0.1, 0.1)
            .density(0.0)
            .restitution(if bouncy { 0.6 } else { 0.0 }),
        Some(root),
    );
    world.insert_collider(ColliderBuilder::ball(0.1).density(0.0), Some(child));
    for body in [root, child] {
        world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
    }
    let handle = world
        .insert_multibody_joint(
            root,
            child,
            FixedJointBuilder::new().local_anchor1(Vector::X * 0.3),
        )
        .unwrap();
    let (multibody, _) = world.multibody_joints.get_mut(handle).unwrap();
    multibody.forward_kinematics(&world.bodies, true);
    multibody.damping_mut().fill(0.0);
    multibody.generalized_velocity_mut()[0] = 2.0;
    multibody.generalized_velocity_mut()[1] = if bouncy { -2.0 } else { 0.0 };
    multibody.generalized_velocity_mut()[2] = 0.7;
    multibody.generalized_velocity_mut()[4] = 1.0;
    multibody.update_rigid_bodies(&mut world.bodies, true);
    world
        .bodies
        .propagate_modified_body_positions_to_colliders(&mut world.colliders);
    if reverse {
        ground(&mut world);
    }
    (world, handle)
}

fn add_pair(world: &mut PhysicsWorld, position: Vector, spacing: f32) -> MultibodyJointHandle {
    let props = MassProperties::new(Vector::ZERO, 1.0, Vector::splat(0.02));
    let root = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(position)
            .additional_mass_properties(props)
            .can_sleep(false),
    );
    let child = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(position + Vector::X * spacing)
            .additional_mass_properties(props)
            .can_sleep(false),
    );
    world.insert_collider(ColliderBuilder::ball(0.1).density(0.0), Some(root));
    world.insert_collider(ColliderBuilder::ball(0.1).density(0.0), Some(child));
    for body in [root, child] {
        world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
    }
    let handle = world
        .insert_multibody_joint(
            root,
            child,
            FixedJointBuilder::new().local_anchor1(Vector::X * spacing),
        )
        .unwrap();
    let (multibody, _) = world.multibody_joints.get_mut(handle).unwrap();
    multibody.forward_kinematics(&world.bodies, true);
    multibody.damping_mut().fill(0.0);
    multibody.generalized_velocity_mut()[0] = 2.0;
    multibody.generalized_velocity_mut()[2] = 0.7;
    multibody.generalized_velocity_mut()[4] = 1.0;
    multibody.update_rigid_bodies(&mut world.bodies, true);
    world
        .bodies
        .propagate_modified_body_positions_to_colliders(&mut world.colliders);
    handle
}

fn freeze(world: &PhysicsWorld) -> FrozenJacobians {
    world
        .bodies
        .iter()
        .filter_map(|(body, _)| {
            let link = world.multibody_joints.rigid_body_link(body)?;
            let multibody = world.multibody_joints.get_multibody(link.multibody)?;
            if (link.id == 0 && !world.bodies[body].is_dynamic()) || multibody.ndofs() == 0 {
                return None;
            }
            let jacobian = multibody.body_jacobian(link.id);
            Some((
                body,
                link.multibody,
                (0..multibody.ndofs())
                    .map(|dof| (0..6).map(|axis| jacobian[(axis, dof)]).collect())
                    .collect(),
            ))
        })
        .collect()
}

#[cfg(feature = "sim2sim_observation")]
fn verify(
    world: &PhysicsWorld,
    handle: MultibodyJointHandle,
    frozen: &FrozenJacobians,
    evidence: &mut Vec<Value>,
) -> Result<(), Box<dyn Error>> {
    let (multibody, _) = world.multibody_joints.get(handle).ok_or("missing owner")?;
    let data = multibody
        .sim2sim_contact_complete_observation()
        .ok_or("missing complete native contact data")?;
    let owner = world
        .multibody_joints
        .rigid_body_link(multibody.link(0).unwrap().rigid_body_handle())
        .ok_or("missing root owner")?
        .multibody;
    let mut normal = vec![0.0_f32; multibody.ndofs()];
    let mut tangent = normal.clone();
    let mut sides = 0;
    for pair in world.narrow_phase.contact_pairs() {
        for manifold in pair.solver_manifolds() {
            for contact in &manifold.data.solver_contacts {
                let id = (contact.contact_id[0] & !rapier3d::geometry::NEW_CONTACT_BIT) as usize;
                let point = &manifold.points[id];
                for (side, body) in [manifold.data.rigid_body1, manifold.data.rigid_body2]
                    .into_iter()
                    .enumerate()
                {
                    let Some((_, _, jacobian)) = frozen
                        .iter()
                        .find(|(handle, index, _)| Some(*handle) == body && *index == owner)
                    else {
                        continue;
                    };
                    let sign = if side == 0 { 1.0 } else { -1.0 };
                    let force = -manifold.data.normal * point.data.impulse * sign;
                    // One actual temporal step: native total tangent equals the newly written
                    // current tangent warmstart vector. This is never an old-step seed.
                    let friction = point.data.warmstart_tangent_world * sign;
                    let arm = if side == 0 {
                        point.data.solver_dp1
                    } else {
                        point.data.solver_dp2
                    };
                    let torque = arm.cross(force);
                    let friction_torque = arm.cross(friction);
                    let wrench = [force.x, force.y, force.z, torque.x, torque.y, torque.z];
                    let friction_wrench = [
                        friction.x,
                        friction.y,
                        friction.z,
                        friction_torque.x,
                        friction_torque.y,
                        friction_torque.z,
                    ];
                    for (dof, j) in jacobian.iter().enumerate() {
                        for axis in 0..6 {
                            normal[dof] += j[axis] * wrench[axis];
                            tangent[dof] += j[axis] * friction_wrench[axis];
                        }
                    }
                    sides += 1;
                }
            }
        }
    }
    let normal_error = normal
        .iter()
        .zip(&data.contact_normal_impulse)
        .map(|(expected, actual)| (*expected - *actual).abs())
        .fold(0.0_f32, f32::max);
    let tangent_error = tangent
        .iter()
        .zip(&data.contact_tangent_impulse)
        .map(|(expected, actual)| (*expected - *actual).abs())
        .fold(0.0_f32, f32::max);
    if normal_error > 2e-5
        || tangent_error > 2e-5
        || !normal_error.is_finite()
        || !tangent_error.is_finite()
        || data.contact_expected_side_count != sides
        || data.contact_measured_side_count != sides
        || data.contact_normal_row_side_count != sides
        || data.contact_tangent_row_side_count != 2 * sides
    {
        return Err(format!("pre-integration signed contact projection/count mismatch: normal={normal_error} tangent={tangent_error} sides={sides} actual={}",data.contact_measured_side_count).into());
    }
    evidence.push(json!({"epoch":data.epoch,"sides":sides,"normal_max_abs":normal_error,"tangent_max_abs":tangent_error,
        "normal":data.contact_normal_impulse,"tangent":data.contact_tangent_impulse,"contact_coverage":data.contact_coverage,
        "pre_integration_jacobian":true,"full_bam_load_qualified":false}));
    Ok(())
}

#[cfg(not(feature = "sim2sim_observation"))]
fn verify(
    _: &PhysicsWorld,
    _: MultibodyJointHandle,
    _: &FrozenJacobians,
    _: &mut Vec<Value>,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut output = None;
    let mut compare = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing flag path")?;
        match flag.as_str() {
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            "--compare" if compare.is_none() => compare = Some(PathBuf::from(value)),
            _ => return Err("unknown or duplicate argument".into()),
        }
    }
    let output = output.ok_or("expected --output")?;
    let mut probe = primitives::run_gates()?;
    let mut evidence = Vec::new();
    let mut contact_step_calls = 0_usize;
    for (index, (prop, reverse, bouncy)) in [
        (false, false, false),
        (false, true, false),
        (true, false, false),
        (false, false, true),
        (false, false, false),
    ]
    .into_iter()
    .enumerate()
    {
        let (mut world, handle) = scene(prop, reverse, bouncy);
        if index == 4 {
            let fixed_root = world
                .bodies
                .iter()
                .find(|(_, body)| body.is_fixed())
                .unwrap()
                .0;
            let child = world.insert_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(20.0, -0.1, 0.0))
                    .additional_mass_properties(MassProperties::new(
                        Vector::ZERO,
                        1.0,
                        Vector::splat(0.02),
                    ))
                    .can_sleep(false),
            );
            world.bodies[child].recompute_mass_properties_from_colliders(&world.colliders);
            let fixed_handle = world
                .insert_multibody_joint(
                    fixed_root,
                    child,
                    PrismaticJointBuilder::new(Vector::X).local_anchor1(Vector::X * 20.0),
                )
                .unwrap();
            let (fixed_multibody, _) = world.multibody_joints.get_mut(fixed_handle).unwrap();
            fixed_multibody.forward_kinematics(&world.bodies, true);
            fixed_multibody.damping_mut().fill(0.0);
            fixed_multibody.update_rigid_bodies(&mut world.bodies, true);
        }
        for tick in 0..60 {
            let frozen = freeze(&world);
            world.step();
            contact_step_calls += 1;
            verify(&world, handle, &frozen, &mut evidence)?;
            probe.capture(&world, handle, &format!("contact_{index}"), tick);
        }
    }
    let (mut multiple, first) = scene(false, false, false);
    let second = add_pair(&mut multiple, Vector::new(0.15, 0.08, 0.0), 0.3);
    for tick in 0..30 {
        let frozen = freeze(&multiple);
        multiple.step();
        contact_step_calls += 1;
        for (index, handle) in [first, second].into_iter().enumerate() {
            verify(&multiple, handle, &frozen, &mut evidence)?;
            probe.capture(
                &multiple,
                handle,
                &format!("contact_two_owners_{index}"),
                tick,
            );
        }
    }
    let mut self_contact = PhysicsWorld::new();
    self_contact.gravity = Vector::ZERO;
    self_contact.integration_parameters.dt = 1.0 / 60.0;
    self_contact.integration_parameters.num_solver_iterations = 1;
    let same = add_pair(&mut self_contact, Vector::ZERO, 0.15);
    let frozen = freeze(&self_contact);
    self_contact.step();
    contact_step_calls += 1;
    verify(&self_contact, same, &frozen, &mut evidence)?;
    #[cfg(feature = "sim2sim_observation")]
    if self_contact
        .multibody_joints
        .get(same)
        .unwrap()
        .0
        .sim2sim_contact_complete_observation()
        .unwrap()
        .contact_measured_side_count
        != 2
    {
        return Err("same-articulation contact did not retain both point sides".into());
    }
    probe.capture(&self_contact, same, "same_articulation_both_sides", 0);
    let (mut reset, current) = scene(false, false, false);
    let mut current = current;
    for iteration in 0..20 {
        for _ in 0..5 {
            reset.step();
            contact_step_calls += 1;
        }
        let handles: Vec<_> = reset
            .multibody_joints
            .get(current)
            .unwrap()
            .0
            .links()
            .map(|link| link.rigid_body_handle())
            .collect();
        for body in handles.into_iter().rev() {
            reset.remove_body(body);
        }
        if reset.multibody_joints.get(current).is_some() {
            return Err("removed owner generation still live".into());
        }
        current = add_pair(&mut reset, Vector::Y * 0.08, 0.3);
        #[cfg(feature = "sim2sim_observation")]
        if reset
            .multibody_joints
            .get(current)
            .unwrap()
            .0
            .sim2sim_contact_complete_observation()
            .is_some()
        {
            return Err("new generation inherited complete contact measurement".into());
        }
        // The reset primitive uses the same two ball shapes for both worlds.
        let mut fresh = PhysicsWorld::new();
        fresh.gravity = -Vector::Y * 9.81;
        fresh.integration_parameters.dt = 1.0 / 60.0;
        fresh.integration_parameters.num_solver_iterations = 1;
        fresh.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(3.0, 0.1, 3.0),
        );
        let fresh_handle = add_pair(&mut fresh, Vector::Y * 0.08, 0.3);
        for tick in 0..60 {
            let frozen = freeze(&reset);
            reset.step();
            contact_step_calls += 1;
            fresh.step();
            contact_step_calls += 1;
            verify(&reset, current, &frozen, &mut evidence)?;
            let reset_body = reset
                .multibody_joints
                .get(current)
                .unwrap()
                .0
                .link(0)
                .unwrap()
                .rigid_body_handle();
            let fresh_body = fresh
                .multibody_joints
                .get(fresh_handle)
                .unwrap()
                .0
                .link(0)
                .unwrap()
                .rigid_body_handle();
            let error = (reset.bodies[reset_body].translation()
                - fresh.bodies[fresh_body].translation())
            .abs()
            .max_element()
            .max(
                (reset.bodies[reset_body].linvel() - fresh.bodies[fresh_body].linvel())
                    .abs()
                    .max_element(),
            )
            .max(
                (reset.bodies[reset_body].angvel() - fresh.bodies[fresh_body].angvel())
                    .abs()
                    .max_element(),
            )
            .max(
                reset.bodies[reset_body]
                    .rotation()
                    .to_array()
                    .into_iter()
                    .zip(fresh.bodies[fresh_body].rotation().to_array())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max),
            );
            if error > 1e-6 {
                return Err(format!("reset contact warmstart altered fresh state: {error}").into());
            }
            probe.capture(&reset, current, &format!("contact_reset_{iteration}"), tick);
        }
    }
    let bytes = serde_json::to_vec(&probe.trace)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let equal = if let Some(compare) = compare {
        let baseline: Value = serde_json::from_slice(&fs::read(compare)?)?;
        Some(
            baseline["observation_feature_enabled"] == false
                && baseline["physics_trace_sha256"] == hash
                && baseline["physics_trace"] == serde_json::to_value(&probe.trace)?,
        )
    } else {
        None
    };
    let passed = equal != Some(false);
    let report = json!({"scope":"rapier_observation_stage_b_primitives","passed":passed,"observation_feature_enabled":cfg!(feature="sim2sim_observation"),"trace_rows":probe.trace.len(),"physics_trace_sha256":hash,"bitwise_baseline_equal":equal,"physics_trace":probe.trace,"observations":probe.observations,"contact_evidence":evidence,"contact_scene_physics_pipeline_step_calls":contact_step_calls,"native_contact_owner_checks":evidence.len(),
        "complete_bam_external_load_qualified":false,"rolling_torsion_qualified":false,"parallel_observation_supported":false,
        "unqualified":["microduck","source_target_behavior","bam_control","performance"]});
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    if !passed {
        return Err("contact observer changed native physical bit trace".into());
    }
    println!(
        "PASS stage_b rows={} hash={hash} bitwise={equal:?}",
        report["trace_rows"]
    );
    Ok(())
}
