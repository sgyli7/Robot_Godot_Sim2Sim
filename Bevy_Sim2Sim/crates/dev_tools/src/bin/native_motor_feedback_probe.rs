//! Actual native motor fixture; observations never alter the solver.

use rapier3d::prelude::*;
use serde_json::json;
use simulation_minigame::{PhysicsClockProfile, SimulationWorld};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os().nth(1).ok_or("output path required")?;
    if std::path::Path::new(&path).exists() {
        return Err("output exists".into());
    }
    let mut cases = Vec::new();
    for (target, cap, bias) in [
        (1.0, 0.3, 0.0),
        (-1.0, 0.3, 0.0),
        (0.2, 10.0, 0.0),
        (-0.2, 10.0, 0.0),
        (0.4, 0.3, 0.1),
    ] {
        let mut world = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
        world.world.gravity = Vector::ZERO;
        world
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 4;
        let root = world.world.bodies.insert(RigidBodyBuilder::fixed());
        let child = world.world.bodies.insert(
            RigidBodyBuilder::dynamic()
                .can_sleep(false)
                .additional_mass_properties(MassProperties::new(
                    Vector::ZERO,
                    1.0,
                    Vector::splat(0.1),
                )),
        );
        world.world.bodies[child].recompute_mass_properties_from_colliders(&world.world.colliders);
        let handle = world
            .world
            .multibody_joints
            .insert(
                root,
                child,
                PrismaticJointBuilder::new(Vector::X)
                    .motor_model(MotorModel::ForceBased)
                    .motor_position(target - bias / 10.0, 10.0, 2.0)
                    .motor_max_force(cap),
                true,
            )
            .ok_or("joint insertion failed")?;
        let mut rows = Vec::new();
        for _ in 0..100 {
            world.world.bodies[child].reset_forces(false);
            world.world.bodies[child].add_force(Vector::X * bias, false);
            world.step_with_torques(&[])?;
            let (mb, id) = world
                .world
                .multibody_joints
                .get(handle)
                .ok_or("joint absent")?;
            let link = mb.link(id).ok_or("link absent")?;
            let obs = mb.sim2sim_observation().ok_or("observation invalid")?;
            let slot = link.assembly_id();
            rows.push(json!({"q":link.joint.coords()[0],"v":mb.generalized_velocity()[slot],
                "generic_impulse":obs.generic_joint_impulse[slot],"generic_rows":obs.generic_joint_row_side_count,
                "motor_impulse":obs.own_motor_impulse[slot],"motor_rows":obs.own_motor_row_count[slot],
                "body_position":world.world.bodies[child].translation().to_array(),"snapshot":world.snapshot()}));
        }
        cases.push(json!({"target":target,"cap":cap,"bias":bias,"trace":rows}));
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&json!({"schema":"native_motor_feedback_fixture_v1",
        "physics_integrals":500,"dt_s":0.02,"cases":cases}))?,
    )?;
    println!("STATUS: success; actual_integrals=500");
    Ok(())
}

#[cfg(test)]
mod unit_rotation_tests {
    use rapier3d::prelude::*;

    #[test]
    fn free_joint_rotation_stays_unit_after_repeated_integration() {
        let mut joint = MultibodyJoint::new(GenericJoint::new(JointAxesMask::empty()), false);
        let velocity = [0.0, 0.0, 0.0, 0.13, 0.21, 0.17];
        for tick in 0..5000 {
            joint.integrate(0.02, &velocity);
            assert!(
                (joint.joint_rot().length_squared() - 1.0).abs() <= 4.0 * f32::EPSILON,
                "unit Tick {}",
                tick + 1
            );
        }
        let expected = Rotation::from_scaled_axis(Vector::new(0.13, 0.21, 0.17) * 100.0);
        assert!(joint.joint_rot().dot(expected).abs() > 1.0 - 1e-5);
        assert_eq!(joint.coords()[0], 0.0);
        assert_eq!(joint.coords()[1], 0.0);
        assert_eq!(joint.coords()[2], 0.0);
    }
}
