//! Isolate zero-mass geometry effects on source principal inertia import.

use rapier3d::{math::Rotation, prelude::*};
use robot_minigame::{
    basis::{source_to_engine_rotation, source_to_engine_vector},
    definition::RobotDefinition,
};
use serde_json::json;
use std::{error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[2] != "--output" {
        return Err("expected compiled JSON, SHA256, --output report.json".into());
    }
    let definition = RobotDefinition::load_json(Path::new(&args[0]), &args[1])?;
    let model = definition.model();
    let f = &model.fields;
    let mut rows = Vec::new();
    for body in 1..model.counts.nbody {
        let inertia = f.body_inertia[body];
        let properties = MassProperties::with_principal_inertia_frame(
            Vector::from_array(source_to_engine_vector(f.body_ipos[body].map(|v| v as f32))),
            f.body_mass[body] as f32,
            Vector::new(inertia[0] as f32, inertia[2] as f32, inertia[1] as f32),
            Rotation::from_array(source_to_engine_rotation(
                f.body_iquat[body].map(|v| v as f32),
            )?),
        );
        let expected = properties.reconstruct_inertia_matrix().to_cols_array();
        let geom = f.geom_bodyid.iter().position(|id| *id == body);
        for mode in ["no_collider", "identity_zero_mass", "geom_pose_zero_mass"] {
            let mut world = PhysicsWorld::new();
            world.integration_parameters.dt = 1.0 / 60.0;
            world.integration_parameters.num_solver_iterations = 1;
            world.integration_parameters.max_ccd_substeps = 1;
            let handle = world.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .additional_mass_properties(properties)
                    .can_sleep(false),
            );
            if mode != "no_collider" {
                let mut shape = ColliderBuilder::ball(0.02).density(0.0);
                if mode == "geom_pose_zero_mass" {
                    if let Some(geom) = geom {
                        shape = shape.position(Pose::from_parts(
                            Vector::from_array(source_to_engine_vector(
                                f.geom_pos[geom].map(|v| v as f32),
                            )),
                            Rotation::from_array(source_to_engine_rotation(
                                f.geom_quat[geom].map(|v| v as f32),
                            )?),
                        ));
                    }
                }
                world
                    .colliders
                    .insert_with_parent(shape, handle, &mut world.bodies);
            }
            world.bodies[handle].recompute_mass_properties_from_colliders(&world.colliders);
            let initial = world.bodies[handle]
                .mass_properties()
                .local_mprops
                .reconstruct_inertia_matrix()
                .to_cols_array();
            world.step();
            let after = world.bodies[handle]
                .mass_properties()
                .local_mprops
                .reconstruct_inertia_matrix()
                .to_cols_array();
            let error = |values: &[f32; 9]| {
                values
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (*a - b).abs())
                    .fold(0.0_f32, f32::max)
            };
            rows.push(json!({"source_body": body,"name": model.names.body[body],"mode":mode,
                "expected_tensor_cols":expected,"initial_tensor_cols":initial,"after_one_step_tensor_cols":after,
                "initial_max_abs":error(&initial),"after_one_step_max_abs":error(&after),
                "physics_integrations":1}));
        }
    }
    let report = json!({"scope":"native_zero_mass_geometry_inertia_diagnostic",
        "model_sha256":definition.file_sha256(),"rows":rows,"policy_inferences":0,
        "robot_control_qualified":false});
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("{}", json!({"output":args[3]}));
    Ok(())
}
