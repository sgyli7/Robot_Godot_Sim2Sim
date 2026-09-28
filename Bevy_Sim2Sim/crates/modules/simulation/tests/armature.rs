//! Primitive API evidence only: no MicroDuck model or actuator is qualified here.

use rapier3d::prelude::*;
use serde::Serialize;
use simulation_minigame::{BodyTorque, PHYSICS_DT, SimulationWorld};

#[derive(Debug, Serialize)]
struct ArmatureProbe {
    armature: f32,
    spatial_inertia_before: [f32; 3],
    spatial_inertia_after: [f32; 3],
    assembly_id: usize,
    dt: f32,
    torque: f32,
    expected_velocity: f32,
    actual_velocity: f32,
    expected_position: f32,
    actual_position: f32,
    integration_count: u64,
}

#[derive(Debug, Serialize)]
struct MassTensorProbe {
    mass: f32,
    local_com: [f32; 3],
    supplied_inertia_columns: [f32; 9],
    actual_inertia_columns: [f32; 9],
    collider_density: f32,
    integration_count: u64,
}

fn one_step(armature: f32) -> ArmatureProbe {
    let mut simulation = SimulationWorld::new();
    simulation.world.gravity = Vector::ZERO;
    let parent = simulation.world.bodies.insert(RigidBodyBuilder::fixed());
    let child = simulation.world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .additional_solver_iterations(0)
            .additional_mass_properties(MassProperties::new(
                Vector::ZERO,
                1.0,
                Vector::new(0.02, 0.02, 0.02),
            )),
    );
    // Builder additional mass properties are deferred until step/recompute.
    // Establish actual initial inertia without any integration/warmup.
    simulation.world.bodies[child]
        .recompute_mass_properties_from_colliders(&simulation.world.colliders);
    let handle = simulation
        .world
        .multibody_joints
        .insert(parent, child, RevoluteJointBuilder::new(Vector::Z), true)
        .unwrap();
    let assembly_id;
    {
        let (multibody, link_id) = simulation.world.multibody_joints.get_mut(handle).unwrap();
        // FK establishes the fixed root before obtaining an index; no warmup step.
        multibody.forward_kinematics(&simulation.world.bodies, true);
        assembly_id = multibody.link(link_id).unwrap().assembly_id();
        multibody.armature_mut()[assembly_id] = armature;
        // Rapier initializes angular generalized damping to 0.1 by default.
        multibody.damping_mut().fill(0.0);
        multibody.generalized_velocity_mut().fill(0.0);
        multibody.update_rigid_bodies(&mut simulation.world.bodies, true);
    }
    let before = simulation.world.bodies[child]
        .mass_properties()
        .local_mprops
        .principal_inertia()
        .to_array();
    let torque = 0.06;
    let state = simulation
        .step_with_torques(&BodyTorque::joint_pair(parent, child, [0.0, 0.0, torque]))
        .unwrap();
    let (multibody, link_id) = simulation.world.multibody_joints.get(handle).unwrap();
    let velocity = multibody.generalized_velocity()[assembly_id];
    // RevoluteJoint aligns free AngX with the supplied body-local axis.
    let position = multibody.link(link_id).unwrap().joint().coords()[3];
    let after = simulation.world.bodies[child]
        .mass_properties()
        .local_mprops
        .principal_inertia()
        .to_array();
    ArmatureProbe {
        armature,
        spatial_inertia_before: before,
        spatial_inertia_after: after,
        assembly_id,
        dt: PHYSICS_DT,
        torque,
        expected_velocity: torque / (before[2] + armature) * PHYSICS_DT,
        actual_velocity: velocity,
        expected_position: torque / (before[2] + armature) * PHYSICS_DT * PHYSICS_DT,
        actual_position: position,
        integration_count: state.integration_count,
    }
}

#[test]
fn native_armature_and_one_solver_time_step_follow_the_primitive_formula() {
    let probes = [one_step(0.0), one_step(0.1)];
    for probe in &probes {
        assert_eq!(probe.assembly_id, 0);
        assert_eq!(probe.integration_count, 1);
        assert_eq!(probe.spatial_inertia_before, probe.spatial_inertia_after);
        assert!((probe.spatial_inertia_before[2] - 0.02).abs() < 1e-7);
        assert!(
            (probe.actual_velocity - probe.expected_velocity).abs() < 1e-7,
            "{probe:?}"
        );
        assert!(
            (probe.actual_position - probe.expected_position).abs() < 1e-7,
            "{probe:?}"
        );
    }
    let evidence = serde_json::to_string_pretty(&probes).unwrap();
    eprintln!("{evidence}");
    if let Ok(path) = std::env::var("BEVY_ARMATURE_PROBE_OUTPUT") {
        let path = std::path::PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, evidence).unwrap();
    }
}

#[test]
fn explicit_mass_tensor_and_zero_density_collider_preserve_com_and_full_inertia() {
    let mut simulation = SimulationWorld::new();
    let local_com = Vector::new(0.1, -0.2, 0.3);
    let principal_frame = Rotation::from_rotation_z(0.4);
    let rotation_matrix = Matrix::from_quat(principal_frame);
    let supplied_inertia = rotation_matrix
        * Matrix::from_diagonal(Vector::new(0.02, 0.03, 0.04))
        * rotation_matrix.transpose();
    let body =
        simulation
            .world
            .bodies
            .insert(RigidBodyBuilder::dynamic().additional_mass_properties(
                MassProperties::with_inertia_matrix(local_com, 2.0, supplied_inertia),
            ));
    simulation.world.colliders.insert_with_parent(
        ColliderBuilder::ball(0.25).density(0.0),
        body,
        &mut simulation.world.bodies,
    );
    simulation.world.bodies[body]
        .recompute_mass_properties_from_colliders(&simulation.world.colliders);
    let actual = &simulation.world.bodies[body].mass_properties().local_mprops;
    let actual_inertia = actual.reconstruct_inertia_matrix();
    assert!((actual.mass() - 2.0).abs() < 1e-7);
    assert!(actual.local_com.abs_diff_eq(local_com, 1e-7));
    assert!(actual_inertia.abs_diff_eq(supplied_inertia, 1e-7));
    let probe = MassTensorProbe {
        mass: actual.mass(),
        local_com: actual.local_com.to_array(),
        supplied_inertia_columns: supplied_inertia.to_cols_array(),
        actual_inertia_columns: actual_inertia.to_cols_array(),
        collider_density: 0.0,
        integration_count: simulation.snapshot().integration_count,
    };
    assert_eq!(probe.integration_count, 0);
    let evidence = serde_json::to_string_pretty(&probe).unwrap();
    eprintln!("{evidence}");
    if let Ok(path) = std::env::var("BEVY_ARMATURE_PROBE_OUTPUT") {
        let path = std::path::PathBuf::from(path).with_file_name("mass_tensor_probe.json");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, evidence).unwrap();
    }
}
