//! Native stop evidence at 50 Hz. These primitives do not qualify a Goose plant.

use rapier3d::prelude::*;
use simulation_minigame::{BodyTorque, PhysicsClockProfile, SimulationWorld};

#[test]
fn unsupported_joints_and_conflicting_limit_modes_are_refused() {
    for (description, kinematic) in [
        (GenericJoint::default(), false),
        (RevoluteJointBuilder::new(Vector::Z).build().into(), false),
        (
            RevoluteJointBuilder::new(Vector::Z)
                .limits([0.0, 0.55])
                .build()
                .into(),
            true,
        ),
        (
            RevoluteJointBuilder::new(Vector::Z)
                .limits([0.55, 0.0])
                .build()
                .into(),
            false,
        ),
    ] {
        let mut joint = MultibodyJoint::new(description, kinematic);
        assert!(!joint.set_predictive_limits_enabled(true));
        assert!(!joint.predictive_limits_enabled());
    }
    #[cfg(feature = "sim2sim_source_limit_probe")]
    {
        let mut joint = MultibodyJoint::new(
            RevoluteJointBuilder::new(Vector::Z)
                .limits([0.0, 0.55])
                .build()
                .into(),
            false,
        );
        assert!(joint.set_predictive_limits_enabled(true));
        assert!(!joint.sim2sim_set_source_limit_probe(
            [0.002, 1.0],
            [0.9, 0.95, 0.001, 0.5, 2.0],
            0.0,
            50.0
        ));
        assert!(joint.set_predictive_limits_enabled(false));
        assert!(joint.sim2sim_set_source_limit_probe(
            [0.002, 1.0],
            [0.9, 0.95, 0.001, 0.5, 2.0],
            0.0,
            50.0
        ));
        assert!(!joint.set_predictive_limits_enabled(true));
    }
}

fn fixture(
    slide: bool,
    initial: f32,
    velocity: f32,
    predictive: bool,
) -> (
    SimulationWorld,
    MultibodyJointHandle,
    RigidBodyHandle,
    RigidBodyHandle,
) {
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    let world = &mut simulation.world;
    world.gravity = Vector::ZERO;
    let parent = world.bodies.insert(RigidBodyBuilder::fixed());
    let child = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .additional_mass_properties(MassProperties::new(
                Vector::ZERO,
                0.02,
                Vector::splat(0.02),
            )),
    );
    world.bodies[child].recompute_mass_properties_from_colliders(&world.colliders);
    let description: GenericJoint = if slide {
        GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES)
            .limits(JointAxis::LinX, [0.0, 0.55])
            .build()
    } else {
        RevoluteJointBuilder::new(Vector::Z)
            .limits([0.0, 0.55])
            .build()
            .into()
    };
    let handle = world
        .multibody_joints
        .insert(parent, child, description, true)
        .unwrap();
    let (multibody, link_id) = world.multibody_joints.get_mut(handle).unwrap();
    multibody.forward_kinematics(&world.bodies, true);
    multibody.damping_mut().fill(0.0);
    assert!(
        !multibody
            .link(link_id)
            .unwrap()
            .joint
            .predictive_limits_enabled()
    );
    assert!(
        multibody
            .link_mut(link_id)
            .unwrap()
            .joint
            .set_predictive_limits_enabled(predictive)
    );
    multibody
        .link_mut(link_id)
        .unwrap()
        .joint
        .apply_displacement(&[initial]);
    multibody.generalized_velocity_mut().fill(velocity);
    multibody.forward_kinematics(&world.bodies, true);
    multibody.update_rigid_bodies(&mut world.bodies, true);
    (simulation, handle, parent, child)
}

fn state(simulation: &SimulationWorld, handle: MultibodyJointHandle, slide: bool) -> (f32, f32) {
    let (multibody, link_id) = simulation.world.multibody_joints.get(handle).unwrap();
    let link = multibody.link(link_id).unwrap();
    (
        link.joint.coords()[if slide { 0 } else { 3 }],
        multibody.joint_velocity(link)[0],
    )
}

fn step(
    simulation: &mut SimulationWorld,
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    slide: bool,
    effort: f32,
) {
    simulation.world.bodies[child].reset_forces(true);
    let torques = if slide {
        simulation.world.bodies[child].add_force(Vector::X * effort, true);
        Vec::new()
    } else {
        BodyTorque::joint_pair(parent, child, [0.0, 0.0, effort]).to_vec()
    };
    simulation.step_with_torques(&torques).unwrap();
}

#[test]
fn predictive_stop_prevents_this_step_crossing_and_default_keeps_original_behavior() {
    for slide in [false, true] {
        for direction in [-1.0, 1.0] {
            let initial = if direction > 0.0 { 0.50 } else { 0.05 };
            let (mut original, handle, parent, child) = fixture(slide, initial, 0.0, false);
            step(&mut original, parent, child, slide, 4.4 * direction);
            let (q, _) = state(&original, handle, slide);
            let expected = initial + direction * 4.4 / 0.02 * 0.02 * 0.02;
            assert!(
                (q - expected).abs() < 1e-6,
                "original q={q}, expected={expected}"
            );
            assert!(q < 0.0 || q > 0.55);
            let (mut predictive, handle, parent, child) = fixture(slide, initial, 0.0, true);
            step(&mut predictive, parent, child, slide, 4.4 * direction);
            let (q, v) = state(&predictive, handle, slide);
            let bound = if direction > 0.0 { 0.55 } else { 0.0 };
            assert!((q - bound).abs() < 1e-6, "q={q}, bound={bound}");
            assert!((v - direction * 2.5).abs() < 1e-5, "v={v}");
            assert_eq!(predictive.snapshot().integration_count, 1);
            assert_eq!(predictive.configuration().num_solver_iterations, 1);
            assert_eq!(predictive.configuration().max_ccd_substeps, 1);
        }
    }
}

#[test]
fn interior_motion_is_unconstrained_and_both_physical_endpoints_remain_reachable() {
    for slide in [false, true] {
        let (mut simulation, handle, parent, child) = fixture(slide, 0.275, 0.4, true);
        step(&mut simulation, parent, child, slide, 0.0);
        let (q, v) = state(&simulation, handle, slide);
        assert!((q - 0.283).abs() < 1e-6);
        assert!((v - 0.4).abs() < 1e-6);
        for direction in [-1.0, 1.0] {
            let (mut simulation, handle, parent, child) = fixture(slide, 0.275, 0.0, true);
            for _ in 0..100 {
                step(&mut simulation, parent, child, slide, 4.4 * direction);
                let (q, _) = state(&simulation, handle, slide);
                assert!((-1e-5..=0.55001).contains(&q), "q={q}");
            }
            let (q, _) = state(&simulation, handle, slide);
            let bound = if direction > 0.0 { 0.55 } else { 0.0 };
            assert!((q - bound).abs() < 1e-5);
            assert_eq!(simulation.snapshot().integration_count, 100);
            assert_eq!(simulation.snapshot().torque_update_count, 100);
        }
    }
}

#[test]
fn unforced_stop_does_not_increase_physical_kinetic_energy() {
    for slide in [false, true] {
        for direction in [-1.0, 1.0] {
            let (mut simulation, handle, parent, child) =
                fixture(slide, 0.275, 10.0 * direction, true);
            let mut previous_energy = 0.5 * 0.02 * 100.0;
            for _ in 0..100 {
                step(&mut simulation, parent, child, slide, 0.0);
                let (q, v) = state(&simulation, handle, slide);
                let energy = 0.5 * 0.02 * v * v;
                assert!(
                    energy <= previous_energy + 2e-6,
                    "energy {previous_energy} -> {energy}"
                );
                assert!((-1e-5..=0.55001).contains(&q), "q={q}");
                previous_energy = energy;
            }
            assert!(previous_energy < 1e-8);
        }
    }
}

#[test]
#[cfg(feature = "sim2sim_limit_row_trace")]
fn signed_limit_trace_records_the_impulse_used_before_position_integration() {
    use rapier3d::dynamics::LimitRowTracePhase;
    for direction in [-1.0_f32, 1.0] {
        let initial = if direction > 0.0 { 0.50 } else { 0.05 };
        let (mut simulation, handle, parent, child) = fixture(false, initial, 0.0, true);
        step(&mut simulation, parent, child, false, 4.4 * direction);
        let (multibody, _) = simulation.world.multibody_joints.get(handle).unwrap();
        let observation = multibody.sim2sim_observation().unwrap();
        assert_eq!(observation.limit_row_timing.len(), 6);
        let integrated: Vec<_> = observation
            .limit_row_timing
            .iter()
            .filter(|s| s.phase == LimitRowTracePhase::AfterBiasedSolve)
            .collect();
        assert_eq!(integrated.len(), 2);
        assert!(integrated.iter().any(|s| s.jacobian_sign == 1.0));
        assert!(integrated.iter().any(|s| s.jacobian_sign == -1.0));
        for sample in &integrated {
            assert_eq!(sample.substep_id, 0);
            assert!((sample.coordinate - initial).abs() < 1e-6);
            assert!((sample.generalized_velocity - direction * 2.5).abs() < 1e-5);
        }
        let impulse = integrated
            .iter()
            .map(|s| -s.jacobian_sign * s.impulse)
            .sum::<f32>();
        assert!((impulse - direction * (0.02 * 2.5 - 4.4 * 0.02)).abs() < 1e-6);
        assert_eq!(simulation.snapshot().integration_count, 1);
    }
}
