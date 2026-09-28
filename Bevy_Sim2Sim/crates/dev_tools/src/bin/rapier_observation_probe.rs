//! Bounded primitive gates and bitwise physics traces for the observation-only fork.

use rapier3d::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf};

const DT: f32 = 1.0 / 60.0;

#[derive(Serialize)]
pub(crate) struct TraceRow {
    scenario: String,
    tick: usize,
    body_bits: Vec<([u32; 2], Vec<u32>)>,
    joint_bits: Vec<([u32; 2], Vec<u32>, Vec<u32>)>,
    contact_bits: Vec<([u32; 2], [u32; 2], Vec<u32>)>,
}

pub(crate) struct Probe {
    pub(crate) trace: Vec<TraceRow>,
    pub(crate) observations: Vec<Value>,
    pub(crate) gates: Vec<String>,
}

impl Probe {
    pub(crate) fn capture(
        &mut self,
        world: &PhysicsWorld,
        handle: MultibodyJointHandle,
        scenario: &str,
        tick: usize,
    ) {
        let mut body_bits: Vec<_> = world
            .bodies
            .iter()
            .map(|(handle, body)| {
                let mut values = body.translation().to_array().to_vec();
                values.extend(body.rotation().to_array());
                values.extend(body.linvel().to_array());
                values.extend(body.angvel().to_array());
                let (index, generation) = handle.into_raw_parts();
                (
                    [index, generation],
                    values.into_iter().map(f32::to_bits).collect(),
                )
            })
            .collect();
        body_bits.sort_by_key(|row| row.0);
        let mut joint_bits: Vec<_> = world
            .multibody_joints
            .iter()
            .map(|(handle, _, multibody, link)| {
                let (index, generation) = handle.into_raw_parts();
                (
                    [index, generation],
                    link.joint()
                        .coords()
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect(),
                    multibody
                        .generalized_velocity()
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect(),
                )
            })
            .collect();
        joint_bits.sort_by_key(|row| row.0);
        let mut contact_bits: Vec<_> = world
            .narrow_phase
            .contact_pairs()
            .map(|pair| {
                let (first, first_generation) = pair.collider1.into_raw_parts();
                let (second, second_generation) = pair.collider2.into_raw_parts();
                let mut bits = Vec::new();
                for manifold in &pair.manifolds {
                    bits.extend(manifold.data.normal.to_array().map(f32::to_bits));
                    for point in &manifold.points {
                        bits.extend(point.local_p1.to_array().map(f32::to_bits));
                        bits.extend(point.local_p2.to_array().map(f32::to_bits));
                        bits.push(point.dist.to_bits());
                        bits.push(point.data.impulse.to_bits());
                        bits.extend(point.data.tangent_impulse.iter().copied().map(f32::to_bits));
                        bits.push(point.data.warmstart_impulse.to_bits());
                        bits.extend(
                            point
                                .data
                                .warmstart_tangent_impulse
                                .iter()
                                .copied()
                                .map(f32::to_bits),
                        );
                        bits.push(point.data.warmstart_twist_impulse.to_bits());
                        bits.extend(
                            point
                                .data
                                .warmstart_tangent_world
                                .to_array()
                                .map(f32::to_bits),
                        );
                        bits.extend(point.data.solver_dp1.to_array().map(f32::to_bits));
                        bits.extend(point.data.solver_dp2.to_array().map(f32::to_bits));
                    }
                }
                ([first, first_generation], [second, second_generation], bits)
            })
            .collect();
        contact_bits.sort_by_key(|row| (row.0, row.1));
        self.trace.push(TraceRow {
            scenario: scenario.into(),
            tick,
            body_bits,
            joint_bits,
            contact_bits,
        });
        self.observations.push(measurement(world, handle));
    }
}

fn world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = DT;
    world.integration_parameters.num_solver_iterations = 1;
    world.integration_parameters.max_ccd_substeps = 1;
    world
}

fn body(world: &mut PhysicsWorld, local_com: Vector) -> RigidBodyHandle {
    let handle = world.insert_body(
        RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .additional_solver_iterations(0)
            .additional_mass_properties(MassProperties::new(local_com, 1.0, Vector::splat(0.02))),
    );
    world.bodies[handle].recompute_mass_properties_from_colliders(&world.colliders);
    handle
}

fn hinge(
    armature: f32,
    friction: f32,
    velocity: f32,
    com: Vector,
    limits: bool,
) -> (
    PhysicsWorld,
    RigidBodyHandle,
    RigidBodyHandle,
    MultibodyJointHandle,
) {
    let mut world = world();
    let root = world.insert_body(RigidBodyBuilder::fixed());
    let child = body(&mut world, com);
    let joint = if limits {
        RevoluteJointBuilder::new(Vector::Z).limits([-0.1, 0.1])
    } else {
        RevoluteJointBuilder::new(Vector::Z)
    };
    let handle = world.insert_multibody_joint(root, child, joint).unwrap();
    let (multibody, _) = world.multibody_joints.get_mut(handle).unwrap();
    multibody.forward_kinematics(&world.bodies, true);
    multibody.armature_mut().fill(armature);
    multibody.frictions_mut().fill(friction);
    multibody.damping_mut().fill(0.0);
    multibody.generalized_velocity_mut().fill(velocity);
    multibody.update_rigid_bodies(&mut world.bodies, true);
    (world, root, child, handle)
}

fn torque(world: &mut PhysicsWorld, child: RigidBodyHandle, value: f32) {
    for (_, body) in world.bodies.iter_mut() {
        body.reset_torques(false);
    }
    world.bodies[child].add_torque(Vector::Z * value, false);
}

fn near(actual: f32, expected: f32, tolerance: f32, name: &str) -> Result<(), Box<dyn Error>> {
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        return Err(
            format!("{name}: actual={actual} expected={expected} tolerance={tolerance}").into(),
        );
    }
    Ok(())
}

#[cfg(feature = "sim2sim_observation")]
fn measurement(world: &PhysicsWorld, handle: MultibodyJointHandle) -> Value {
    let Some((multibody, _)) = world.multibody_joints.get(handle) else {
        return Value::Null;
    };
    let Some(data) = multibody.sim2sim_observation() else {
        return json!({"status":multibody.sim2sim_observation_status(),"measured":false});
    };
    json!({"epoch":data.epoch,"valid":data.valid,"contact_coverage":data.contact_coverage,"inertial_projection":data.inertial_projection,"gravity_projection":data.gravity_projection,"generic_joint_impulse":data.generic_joint_impulse,"own_dry_friction_impulse":data.own_dry_friction_impulse,"generic_joint_row_side_count":data.generic_joint_row_side_count,"own_dry_friction_row_side_count":data.own_dry_friction_row_side_count,"energy_guard_evaluated":data.energy_guard_evaluated,"energy_guard_fallback":data.energy_guard_fallback,"energy_guard_acceleration_cleared":data.energy_guard_acceleration_cleared})
}

#[cfg(not(feature = "sim2sim_observation"))]
fn measurement(_: &PhysicsWorld, _: MultibodyJointHandle) -> Value {
    Value::Null
}

#[cfg(feature = "sim2sim_observation")]
fn observation(
    world: &PhysicsWorld,
    handle: MultibodyJointHandle,
) -> Result<&rapier3d::dynamics::MultibodyObservation, Box<dyn Error>> {
    if !Multibody::sim2sim_observation_backend_supported() {
        if world
            .multibody_joints
            .get(handle)
            .is_some_and(|(multibody, _)| multibody.sim2sim_observation().is_some())
        {
            return Err("parallel backend incorrectly published a valid observation".into());
        }
        return Err("Rapier parallel backend observation unsupported in stage A".into());
    }
    let (multibody, _) = world
        .multibody_joints
        .get(handle)
        .ok_or("missing articulation")?;
    let observation = multibody
        .sim2sim_observation()
        .ok_or("missing valid phase-A measurement")?;
    if !observation.valid {
        return Err("incorrect partial observation validity".into());
    }
    Ok(observation)
}

pub(crate) fn run_gates() -> Result<Probe, Box<dyn Error>> {
    let mut probe = Probe {
        trace: Vec::new(),
        observations: Vec::new(),
        gates: Vec::new(),
    };
    for armature in [0.0, 0.1] {
        let (mut world, _, child, handle) = hinge(armature, 0.0, 0.0, Vector::ZERO, false);
        #[cfg(feature = "sim2sim_observation")]
        if world
            .multibody_joints
            .get(handle)
            .unwrap()
            .0
            .sim2sim_observation()
            .is_some()
        {
            return Err("unstepped observation falsely valid".into());
        }
        torque(&mut world, child, 0.06);
        world.step();
        let (multibody, _) = world.multibody_joints.get(handle).unwrap();
        near(
            multibody.generalized_velocity()[0],
            0.06 / (0.02 + armature) * DT,
            1e-7,
            "torque armature",
        )?;
        #[cfg(feature = "sim2sim_observation")]
        {
            let data = observation(&world, handle)?;
            near(data.inertial_projection[0], 0.0, 1e-8, "stationary inertia")?;
            near(data.gravity_projection[0], 0.0, 1e-8, "zero gravity")?;
            near(data.generic_joint_impulse[0], 0.0, 1e-8, "no constraints")?;
            if data.energy_guard_evaluated || data.epoch != 1 {
                return Err("initial epoch/guard mismatch".into());
            }
        }
        probe.capture(&world, handle, &format!("torque_armature_{armature}"), 0);
    }
    probe
        .gates
        .push("isolated_torque_with_native_armature".into());

    let (mut gravity_world, _, _, gravity_handle) = hinge(0.1, 0.0, 0.0, Vector::X, false);
    gravity_world.gravity = -Vector::Y * 9.81;
    gravity_world.step();
    near(
        gravity_world
            .multibody_joints
            .get(gravity_handle)
            .unwrap()
            .0
            .generalized_velocity()[0],
        -9.81 / 1.12 * DT,
        1e-6,
        "gravity direction",
    )?;
    #[cfg(feature = "sim2sim_observation")]
    {
        let data = observation(&gravity_world, gravity_handle)?;
        near(
            data.gravity_projection[0],
            -9.81,
            2e-6,
            "gravity projection sign",
        )?;
        near(
            data.inertial_projection[0],
            0.0,
            1e-8,
            "gravity initial inertia",
        )?;
    }
    probe.capture(&gravity_world, gravity_handle, "gravity_sign", 0);
    probe.gates.push("gravity_projection_and_sign".into());

    for (name, velocity, applied, expected) in [
        ("friction_positive_velocity", 1.0, 0.0, -0.02 * DT),
        ("friction_negative_velocity", -1.0, 0.0, 0.02 * DT),
        ("friction_positive_stick", 0.0, 0.01, -0.01 * DT),
        ("friction_negative_stick", 0.0, -0.01, 0.01 * DT),
        ("friction_positive_slip", 0.0, 0.06, -0.02 * DT),
        ("friction_negative_slip", 0.0, -0.06, 0.02 * DT),
    ] {
        let (mut world, _, child, handle) = hinge(0.1, 0.02, velocity, Vector::ZERO, false);
        torque(&mut world, child, applied);
        world.step();
        #[cfg(feature = "sim2sim_observation")]
        {
            let data = observation(&world, handle)?;
            near(data.own_dry_friction_impulse[0], expected, 1e-7, name)?;
            near(
                data.generic_joint_impulse[0],
                expected,
                1e-7,
                "friction contained once in joint sum",
            )?;
            if data.own_dry_friction_row_side_count != 1 || data.generic_joint_row_side_count != 1 {
                return Err("internal MAX friction row lost or duplicated".into());
            }
        }
        near(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .generalized_velocity()[0],
            velocity + (applied * DT + expected) / 0.12,
            2e-6,
            name,
        )?;
        probe.capture(&world, handle, name, 0);
    }
    probe
        .gates
        .push("native_dry_friction_positive_negative_stick_slip".into());

    let (mut limit_world, _, _, limit_handle) = hinge(0.1, 0.0, 1.0, Vector::ZERO, true);
    {
        let (multibody, _) = limit_world.multibody_joints.get_mut(limit_handle).unwrap();
        multibody.apply_displacements(&[0.2]);
        multibody.forward_kinematics(&limit_world.bodies, true);
        multibody.update_rigid_bodies(&mut limit_world.bodies, true);
    }
    limit_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        let data = observation(&limit_world, limit_handle)?;
        // The final unbiased pass removes the outward unit velocity. The earlier
        // position correction moved q but is not a final velocity impulse.
        let expected = -0.12;
        near(
            data.generic_joint_impulse[0],
            expected,
            1e-5,
            "limit original row impulse sign",
        )?;
        near(
            data.own_dry_friction_impulse[0],
            0.0,
            1e-8,
            "limit not own friction",
        )?;
        if data.generic_joint_row_side_count != 1 {
            return Err("internal limit row absent".into());
        }
    }
    probe.capture(&limit_world, limit_handle, "upper_limit", 0);
    probe
        .gates
        .push("internal_limit_signed_original_row".into());

    let (mut motor_world, _, _, motor_handle) = hinge(0.1, 0.0, 0.0, Vector::ZERO, false);
    {
        let (multibody, link_id) = motor_world.multibody_joints.get_mut(motor_handle).unwrap();
        multibody.link_mut(link_id).unwrap().joint.data = RevoluteJointBuilder::new(Vector::Z)
            .motor_velocity(1.0, 1.0)
            .motor_max_force(0.03)
            .build()
            .data;
    }
    motor_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        let data = observation(&motor_world, motor_handle)?;
        near(
            data.generic_joint_impulse[0],
            0.03 * DT,
            1e-7,
            "native motor original signed row",
        )?;
        near(
            data.own_dry_friction_impulse[0],
            0.0,
            1e-8,
            "motor not own friction",
        )?;
    }
    probe.capture(&motor_world, motor_handle, "native_motor_row", 0);
    probe.gates.push("native_motor_signed_original_row".into());

    let mut coupling_world = world();
    let root = coupling_world.insert_body(RigidBodyBuilder::fixed());
    let first = body(&mut coupling_world, Vector::ZERO);
    let second = body(&mut coupling_world, Vector::ZERO);
    let coupling_handle = coupling_world
        .insert_multibody_joint(root, first, RevoluteJointBuilder::new(Vector::Z))
        .unwrap();
    coupling_world
        .insert_multibody_joint(root, second, RevoluteJointBuilder::new(Vector::Z))
        .unwrap();
    {
        let (multibody, _) = coupling_world
            .multibody_joints
            .get_mut(coupling_handle)
            .unwrap();
        multibody.forward_kinematics(&coupling_world.bodies, true);
        multibody.armature_mut().fill(0.1);
        multibody.damping_mut().fill(0.0);
        multibody
            .generalized_velocity_mut()
            .copy_from_slice(&[1.0, 0.0]);
        multibody.add_dof_coupling(MultibodyDofCoupling {
            link1: 1,
            dof1: 0,
            axis1: 3,
            link2: 2,
            dof2: 0,
            axis2: 3,
            coeff: 2.0,
            offset: 0.0,
        });
        multibody.update_rigid_bodies(&mut coupling_world.bodies, true);
    }
    coupling_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        let data = observation(&coupling_world, coupling_handle)?;
        // J=[-2,1], equal diagonal mass=.12, lambda=-2/(5/.12).
        near(
            data.generic_joint_impulse[0],
            -0.096,
            1e-6,
            "coupling first original J",
        )?;
        near(
            data.generic_joint_impulse[1],
            0.048,
            1e-6,
            "coupling second original J",
        )?;
        if data.generic_joint_row_side_count != 1 {
            return Err("internal MAX coupling row omitted".into());
        }
    }
    probe.capture(&coupling_world, coupling_handle, "native_coupling_row", 0);
    probe
        .gates
        .push("internal_coupling_original_unweighted_row".into());

    let mut owners_world = world();
    let mut owners = Vec::new();
    for (index, velocity, friction) in [(0, 1.0, 0.02), (1, -1.0, 0.03)] {
        let root = owners_world
            .insert_body(RigidBodyBuilder::fixed().translation(Vector::X * index as f32 * 4.0));
        let child = body(&mut owners_world, Vector::ZERO);
        let handle = owners_world
            .insert_multibody_joint(root, child, RevoluteJointBuilder::new(Vector::Z))
            .unwrap();
        let (multibody, _) = owners_world.multibody_joints.get_mut(handle).unwrap();
        multibody.forward_kinematics(&owners_world.bodies, true);
        multibody.armature_mut().fill(0.1);
        multibody.frictions_mut().fill(friction);
        multibody.damping_mut().fill(0.0);
        multibody.generalized_velocity_mut().fill(velocity);
        multibody.update_rigid_bodies(&mut owners_world.bodies, true);
        owners.push((child, handle));
    }
    owners_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        near(
            observation(&owners_world, owners[0].1)?.own_dry_friction_impulse[0],
            -0.02 * DT,
            1e-7,
            "owner first offset",
        )?;
        near(
            observation(&owners_world, owners[1].1)?.own_dry_friction_impulse[0],
            0.03 * DT,
            1e-7,
            "owner second offset",
        )?;
    }
    probe.capture(&owners_world, owners[0].1, "two_owners_first", 0);
    probe.capture(&owners_world, owners[1].1, "two_owners_second", 0);
    let persistent_first = owners_world
        .islands
        .persistent_island_of(&owners_world.bodies, owners[0].0);
    let persistent_second = owners_world
        .islands
        .persistent_island_of(&owners_world.bodies, owners[1].0);
    if persistent_first.is_none()
        || persistent_second.is_none()
        || persistent_first == persistent_second
    {
        return Err("independent owners did not exercise distinct persistent islands".into());
    }
    owners_world.bodies[owners[0].0].set_enabled(false);
    owners_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        if owners_world
            .multibody_joints
            .get(owners[0].1)
            .unwrap()
            .0
            .sim2sim_observation()
            .is_some()
        {
            return Err("inactive articulation falsely reused a prior solver observation".into());
        }
        observation(&owners_world, owners[1].1)?;
    }
    probe.capture(&owners_world, owners[0].1, "inactive_owner_has_no_data", 1);
    probe.capture(
        &owners_world,
        owners[1].1,
        "active_other_persistent_island",
        1,
    );
    owners_world
        .multibody_joints
        .get_mut(owners[0].1)
        .unwrap()
        .0
        .frictions_mut()
        .fill(0.0);
    owners_world.bodies[owners[0].0].set_enabled(true);
    owners_world.bodies[owners[0].0].wake_up(true);
    owners_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        let data = observation(&owners_world, owners[0].1)?;
        near(
            data.generic_joint_impulse[0],
            0.0,
            1e-8,
            "fresh active owner with no generic rows",
        )?;
        near(
            data.own_dry_friction_impulse[0],
            0.0,
            1e-8,
            "removed friction not stale zero-data placeholder",
        )?;
        if data.generic_joint_row_side_count != 0 || data.own_dry_friction_row_side_count != 0 {
            return Err("old generic rows survived into no-row solve".into());
        }
        observation(&owners_world, owners[1].1)?;
    }
    probe.capture(
        &owners_world,
        owners[0].1,
        "reactivated_owner_no_generic_rows",
        2,
    );
    probe.capture(&owners_world, owners[1].1, "other_owner_still_measured", 2);
    probe
        .gates
        .push("distinct_persistent_islands_inactive_missing_data_and_fresh_no_row_solve".into());
    owners_world.remove_body(owners[0].0);
    #[cfg(feature = "sim2sim_observation")]
    if owners_world
        .multibody_joints
        .get(owners[1].1)
        .unwrap()
        .0
        .sim2sim_observation()
        .is_some()
    {
        return Err("topology change retained former owner offsets".into());
    }
    owners_world.step();
    #[cfg(feature = "sim2sim_observation")]
    near(
        observation(&owners_world, owners[1].1)?.own_dry_friction_impulse[0],
        0.03 * DT,
        1e-7,
        "owner remapped after first removal",
    )?;
    probe.capture(&owners_world, owners[1].1, "owner_remapped_after_remove", 3);
    probe
        .gates
        .push("multiple_articulation_solver_offsets_rebuilt_after_topology_change".into());

    let mut external_world = world();
    let mut external_owners = Vec::new();
    for velocity in [1.0, -1.0] {
        let root = external_world.insert_body(RigidBodyBuilder::fixed());
        let child = body(&mut external_world, Vector::ZERO);
        let handle = external_world
            .insert_multibody_joint(root, child, RevoluteJointBuilder::new(Vector::Z))
            .unwrap();
        let (multibody, _) = external_world.multibody_joints.get_mut(handle).unwrap();
        multibody.forward_kinematics(&external_world.bodies, true);
        multibody.armature_mut().fill(0.1);
        multibody.damping_mut().fill(0.0);
        multibody.generalized_velocity_mut().fill(velocity);
        multibody.update_rigid_bodies(&mut external_world.bodies, true);
        external_owners.push((child, handle));
    }
    external_world.insert_impulse_joint(
        external_owners[0].0,
        external_owners[1].0,
        GenericJointBuilder::new(JointAxesMask::ANG_Z),
    );
    external_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        near(
            observation(&external_world, external_owners[0].1)?.generic_joint_impulse[0],
            -0.12,
            1e-6,
            "external joint signed first side",
        )?;
        near(
            observation(&external_world, external_owners[1].1)?.generic_joint_impulse[0],
            0.12,
            1e-6,
            "external joint signed second side",
        )?;
        for (_, handle) in &external_owners {
            if observation(&external_world, *handle)?.generic_joint_row_side_count != 1 {
                return Err("external joint owner side lost".into());
            }
        }
    }
    probe.capture(
        &external_world,
        external_owners[0].1,
        "external_joint_side_one",
        0,
    );
    probe.capture(
        &external_world,
        external_owners[1].1,
        "external_joint_side_two",
        0,
    );
    probe
        .gates
        .push("external_generic_joint_both_original_signed_owner_rows".into());

    for (index, velocity) in [
        [0.8, -0.3],
        [3.0, -1.0],
        [10.0, -4.0],
        [50.0, -25.0],
        [120.0, 60.0],
        [-50.0, 80.0],
        [1000.0, -500.0],
    ]
    .into_iter()
    .enumerate()
    {
        let mut world = world();
        let root = world.insert_body(RigidBodyBuilder::fixed());
        let first = body(&mut world, Vector::X * 0.5);
        let second = body(&mut world, Vector::X * 0.5);
        let first_handle = world
            .insert_multibody_joint(root, first, RevoluteJointBuilder::new(Vector::Z))
            .unwrap();
        world
            .insert_multibody_joint(
                first,
                second,
                RevoluteJointBuilder::new(Vector::Z).local_anchor1(Vector::X),
            )
            .unwrap();
        {
            let (multibody, _) = world.multibody_joints.get_mut(first_handle).unwrap();
            multibody.forward_kinematics(&world.bodies, true);
            multibody.apply_displacements(&[0.0, 0.7]);
            multibody
                .generalized_velocity_mut()
                .copy_from_slice(&velocity);
            multibody.damping_mut().fill(0.0);
            multibody.armature_mut().fill(0.1);
            multibody.forward_kinematics(&world.bodies, true);
            multibody.update_rigid_bodies(&mut world.bodies, true);
        }
        world.step();
        #[cfg(feature = "sim2sim_observation")]
        {
            let data = observation(&world, first_handle)?;
            let coupling = 0.5 * 0.7_f32.sin();
            let expected = [
                -coupling * (2.0 * velocity[0] * velocity[1] + velocity[1] * velocity[1]),
                coupling * velocity[0] * velocity[0],
            ];
            for axis in 0..2 {
                near(
                    data.inertial_projection[axis],
                    expected[axis],
                    1e-6 + expected[axis].abs() * 1e-6,
                    "planar two-link Coriolis operand",
                )?;
            }
            if !data.energy_guard_evaluated {
                return Err("nonzero-velocity energy guard missing".into());
            }
        }
        probe.capture(&world, first_handle, &format!("coriolis_{index}"), 0);
    }
    probe
        .gates
        .push("nonzero_velocity_analytic_planar_coriolis_and_actual_guard_flags".into());

    let (mut reset_world, root, child, handle) = hinge(0.1, 0.02, 0.0, Vector::ZERO, false);
    reset_world.step();
    probe.capture(&reset_world, handle, "before_reset", 0);
    reset_world.remove_body(child);
    if reset_world.multibody_joints.get(handle).is_some() {
        return Err("removed joint generation still live".into());
    }
    let child = body(&mut reset_world, Vector::ZERO);
    let new_handle = reset_world
        .insert_multibody_joint(root, child, RevoluteJointBuilder::new(Vector::Z))
        .unwrap();
    if handle.into_raw_parts() == new_handle.into_raw_parts() {
        return Err("reset generation reused".into());
    }
    {
        let (multibody, _) = reset_world.multibody_joints.get_mut(new_handle).unwrap();
        multibody.forward_kinematics(&reset_world.bodies, true);
        multibody.damping_mut().fill(0.0);
        multibody.update_rigid_bodies(&mut reset_world.bodies, true);
        #[cfg(feature = "sim2sim_observation")]
        if multibody.sim2sim_observation().is_some() {
            return Err("new generation inherited observations".into());
        }
    }
    reset_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        let data = observation(&reset_world, new_handle)?;
        if data.epoch != 2 || data.generic_joint_row_side_count != 0 {
            return Err("reset stale epoch/row cache".into());
        }
        reset_world
            .multibody_joints
            .invalidate_sim2sim_observations();
        if reset_world
            .multibody_joints
            .get(new_handle)
            .unwrap()
            .0
            .sim2sim_observation()
            .is_some()
        {
            return Err("explicit cold reset did not invalidate".into());
        }
    }
    probe.capture(&reset_world, new_handle, "after_reset_invalidated", 1);
    reset_world.integration_parameters.num_solver_iterations = 4;
    reset_world.step();
    #[cfg(feature = "sim2sim_observation")]
    if reset_world
        .multibody_joints
        .get(new_handle)
        .unwrap()
        .0
        .sim2sim_observation()
        .is_some()
    {
        return Err("subdivided time falsely qualified".into());
    }
    probe.capture(
        &reset_world,
        new_handle,
        "unsupported_temporal_subdivision",
        2,
    );
    reset_world.integration_parameters.num_solver_iterations = 1;
    reset_world.step();
    #[cfg(feature = "sim2sim_observation")]
    {
        observation(&reset_world, new_handle)?;
    }
    probe.capture(&reset_world, new_handle, "single_step_after_reject", 3);
    probe
        .gates
        .push("reset_generation_epoch_invalidity_and_temporal_rejection".into());

    let (mut contact_world, _, child, handle) = hinge(0.1, 0.0, 0.0, Vector::X, false);
    contact_world.gravity = -Vector::Y * 9.81;
    contact_world.insert_collider(
        ColliderBuilder::ball(0.2)
            .translation(Vector::X)
            .density(0.0),
        Some(child),
    );
    contact_world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.7, 0.0)),
        ColliderBuilder::cuboid(3.0, 0.1, 3.0),
    );
    for tick in 0..120 {
        contact_world.step();
        probe.capture(&contact_world, handle, "contact_trace_only", tick);
    }
    if !probe
        .trace
        .iter()
        .filter(|row| row.scenario == "contact_trace_only")
        .any(|row| row.contact_bits.iter().any(|(_, _, bits)| !bits.is_empty()))
    {
        return Err("bitwise baseline has no actual contact rows".into());
    }
    probe
        .gates
        .push("contact_physics_trace_present_without_contact_observation_qualification".into());
    Ok(probe)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut output = None;
    let mut compare = None;
    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or("expected path after argument")?;
        match flag.as_str() {
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            "--compare" if compare.is_none() => compare = Some(PathBuf::from(value)),
            _ => return Err("unknown or duplicate probe argument".into()),
        }
    }
    let output = output.ok_or("expected --output report path")?;
    let probe = run_gates()?;
    let trace_bytes = serde_json::to_vec(&probe.trace)?;
    let hash = format!("{:x}", Sha256::digest(&trace_bytes));
    let mut bitwise_baseline_equal = None;
    if let Some(compare) = compare {
        let baseline: Value = serde_json::from_slice(&fs::read(compare)?)?;
        let equal = baseline["observation_feature_enabled"] == false
            && baseline["physics_trace_sha256"] == hash
            && baseline["physics_trace"] == serde_json::to_value(&probe.trace)?;
        bitwise_baseline_equal = Some(equal);
    }
    let fallback_count = probe
        .observations
        .iter()
        .filter(|row| row["energy_guard_fallback"] == true)
        .count();
    if cfg!(feature = "sim2sim_observation") && fallback_count == 0 {
        return Err("original energy guard fallback branch not exercised".into());
    }
    let passed = bitwise_baseline_equal != Some(false);
    let report = json!({"scope":"rapier_observation_stage_a_primitives","passed":passed,"observation_feature_enabled":cfg!(feature="sim2sim_observation"),"observation_gates_executed":cfg!(feature="sim2sim_observation"),"contact_measurement_exercised":false,"complete_bam_external_load_qualified":false,"unqualified":["contact_generalized_impulse","microduck","bam_complete_load","source_target_behavior","energy_guard_last_chance_acceleration_clear"],"physics_trace_sha256":hash,"bitwise_baseline_equal":bitwise_baseline_equal,"trace_rows":probe.trace.len(),"energy_guard_fallback_observed_rows":fallback_count,"gates":probe.gates,"physics_trace":probe.trace,"observations":probe.observations});
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    if !passed {
        return Err("observation feature changed bitwise q/qd/body/contact trace".into());
    }
    println!(
        "PASS stage_a rows={} observation_feature={} contact_measurement_exercised=false bitwise_baseline_equal={bitwise_baseline_equal:?} guard_fallback_rows={fallback_count}",
        report["trace_rows"],
        cfg!(feature = "sim2sim_observation")
    );
    Ok(())
}
