//! Initialization and actual single-step diagnostic for complete source inertia.

use rapier3d::prelude::*;
use serde_json::json;
use std::{error::Error, fs};

const TOLERANCE: f64 = 2e-10;

fn tensor_error(first: &MassProperties, second: &MassProperties) -> f64 {
    first
        .reconstruct_inertia_matrix()
        .to_cols_array()
        .iter()
        .zip(second.reconstruct_inertia_matrix().to_cols_array())
        .map(|(first, second)| f64::from((*first - second).abs()))
        .fold(0.0_f64, f64::max)
}

fn inspect_native(
    name: &str,
    properties: MassProperties,
    collider: Option<ColliderBuilder>,
) -> serde_json::Value {
    let mut world = PhysicsWorld::new();
    let body =
        world.insert_body(RigidBodyBuilder::dynamic().additional_mass_properties(properties));
    if let Some(collider) = collider {
        world
            .colliders
            .insert_with_parent(collider, body, &mut world.bodies);
    }
    world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
    let actual = &world.bodies[body].mass_properties().local_mprops;
    let error = tensor_error(&properties, actual);
    let initial_mass_error = (properties.mass() - actual.mass()).abs();
    let initial_com_error = (properties.local_com - actual.local_com)
        .abs()
        .max_element();
    let initial_tensor = actual.reconstruct_inertia_matrix().to_cols_array();
    world.integration_parameters.dt = 1.0 / 60.0;
    world.integration_parameters.num_solver_iterations = 1;
    world.step();
    let after = &world.bodies[body].mass_properties().local_mprops;
    let after_error = tensor_error(&properties, after);
    json!({
        "name":name,"tensor_max_abs":error,"passed":error<=TOLERANCE && after_error<=TOLERANCE,
        "mass_error":initial_mass_error,"com_error":initial_com_error,
        "tensor_cols":initial_tensor,"after_one_step_tensor_max_abs":after_error,
        "colliders":world.colliders.len(),"physics_integrations":1,
    })
}

fn geometry_control() -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let body_pose = Pose::from_parts(
        Vector::new(0.01, 0.06, -0.02),
        Rotation::from_rotation_z(0.1),
    );
    let geom_pose = Pose::from_parts(
        Vector::new(0.02, -0.05, 0.01),
        Rotation::from_rotation_y(0.3),
    );
    let ground_pose = Pose::translation(0.0, -0.02, 0.0);
    let ground = SharedShape::cuboid(2.0, 0.02, 2.0);
    let mut cases = Vec::new();
    for (name, leaf) in [
        ("ball", SharedShape::ball(0.05)),
        ("capsule", SharedShape::capsule_y(0.03, 0.03)),
        (
            "convex_box",
            SharedShape::convex_hull(&[
                Vector::new(-0.03, -0.02, -0.04),
                Vector::new(-0.03, -0.02, 0.04),
                Vector::new(-0.03, 0.02, -0.04),
                Vector::new(-0.03, 0.02, 0.04),
                Vector::new(0.03, -0.02, -0.04),
                Vector::new(0.03, -0.02, 0.04),
                Vector::new(0.03, 0.02, -0.04),
                Vector::new(0.03, 0.02, 0.04),
            ])
            .ok_or("invalid convex control")?,
        ),
    ] {
        let compound = SharedShape::compound(vec![(geom_pose, leaf.clone())]);
        let direct = rapier3d::parry::query::contact(
            &(body_pose * geom_pose),
            &*leaf,
            &ground_pose,
            &*ground,
            0.1,
        )?
        .ok_or("direct control has no contact")?;
        let wrapped =
            rapier3d::parry::query::contact(&body_pose, &*compound, &ground_pose, &*ground, 0.1)?
                .ok_or("compound control has no contact")?;
        let error = (direct.point1 - wrapped.point1)
            .abs()
            .max_element()
            .max((direct.point2 - wrapped.point2).abs().max_element())
            .max((direct.normal1 - wrapped.normal1).abs().max_element())
            .max((direct.normal2 - wrapped.normal2).abs().max_element())
            .max((direct.dist - wrapped.dist).abs());
        let direct_aabb = leaf.compute_aabb(&(body_pose * geom_pose));
        let compound_aabb = compound.compute_aabb(&body_pose);
        let aabb_contained = (direct_aabb.mins - compound_aabb.mins).min_element() >= -1e-6
            && (compound_aabb.maxs - direct_aabb.maxs).min_element() >= -1e-6;
        cases.push(json!({"name":name,"contact_geometry_max_abs":error,
            "contact_geometry_tolerance":1e-6,"passed":error<=1e-6 && aabb_contained,
            "compound_aabb_contains_direct_aabb":aabb_contained,
            "direct_aabb":[direct_aabb.mins.to_array(),direct_aabb.maxs.to_array()],
            "compound_aabb":[compound_aabb.mins.to_array(),compound_aabb.maxs.to_array()],
            "physics_integrations":0,"contact_query_count":2,
        }));
    }
    Ok(cases)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || args[0] != "--output" {
        return Err("expected --output report.json".into());
    }
    let properties = MassProperties::with_principal_inertia_frame(
        Vector::new(0.01, -0.02, 0.03),
        0.03,
        Vector::new(6e-6, 2e-6, 4e-6),
        Rotation::from_rotation_z(0.5),
    );
    let pose = Pose::from_parts(Vector::new(0.02, 0.01, -0.03), Rotation::IDENTITY);
    let rotated_pose = Pose::from_parts(Vector::ZERO, Rotation::from_rotation_y(0.3));
    let inverse_compensated_zero = MassProperties::default().transform_by(&pose.inverse());
    let cases = [
        inspect_native("no_collider", properties, None),
        inspect_native(
            "density_zero_identity_pose",
            properties,
            Some(ColliderBuilder::ball(0.01).density(0.0)),
        ),
        inspect_native(
            "density_zero_translation_only",
            properties,
            Some(ColliderBuilder::ball(0.01).density(0.0).position(pose)),
        ),
        inspect_native(
            "density_zero_rotation_only",
            properties,
            Some(
                ColliderBuilder::ball(0.01)
                    .density(0.0)
                    .position(rotated_pose),
            ),
        ),
        inspect_native(
            "identity_collider_single_child_compound_translation",
            properties,
            Some(ColliderBuilder::compound(vec![(pose, SharedShape::ball(0.01))]).density(0.0)),
        ),
        inspect_native(
            "identity_collider_single_child_compound_rotation",
            properties,
            Some(
                ColliderBuilder::compound(vec![(rotated_pose, SharedShape::ball(0.01))])
                    .density(0.0),
            ),
        ),
        inspect_native(
            "explicit_mass_zero_translation",
            properties,
            Some(ColliderBuilder::ball(0.01).mass(0.0).position(pose)),
        ),
        inspect_native(
            "custom_inverse_transformed_zero_translation",
            properties,
            Some(
                ColliderBuilder::ball(0.01)
                    .mass_properties(inverse_compensated_zero)
                    .position(pose),
            ),
        ),
    ];
    let mut direct_eigen = Vec::new();
    for scale in [1.0_f32, 100.0, 10_000.0] {
        let scaled = MassProperties::with_principal_inertia_frame(
            properties.local_com,
            properties.mass(),
            Vector::new(6e-6, 2e-6, 4e-6) * scale,
            properties.principal_inertia_local_frame,
        );
        let diagonalized = MassProperties::with_inertia_matrix(
            scaled.local_com,
            scaled.mass(),
            scaled.reconstruct_inertia_matrix(),
        );
        direct_eigen.push(json!({
            "scale":scale,"tensor_error_divided_by_scale":tensor_error(&scaled,&diagonalized)/f64::from(scale),
            "diagonalized_frame":diagonalized.principal_inertia_local_frame.to_array(),
        }));
    }
    let original_failed = cases[2]["passed"] == false;
    let controls_passed = cases[0]["passed"] == true && cases[1]["passed"] == true;
    let compound_mass_import_passed = cases[4]["passed"] == true && cases[5]["passed"] == true;
    let geometry = geometry_control()?;
    let geometry_passed = geometry.iter().all(|row| row["passed"] == true);
    let diagnostic_passed =
        original_failed && controls_passed && compound_mass_import_passed && geometry_passed;
    fs::write(
        &args[1],
        serde_json::to_string_pretty(&json!({
            "scope":"massless_collider_import_and_single_step_diagnostic",
            "physics_integrations":cases.len(),"integrations_per_native_case":1,
            "expected_tensor_cols":properties.reconstruct_inertia_matrix().to_cols_array(),
            "fixed_tolerance":TOLERANCE,"original_configuration_passed":!original_failed,
            "diagnostic_passed":diagnostic_passed,"cases":cases,"direct_eigen":direct_eigen,
            "default_zero_transformed_is_canonical_zero":MassProperties::default().transform_by(&pose)==MassProperties::default(),
            "zero_transformed_tensor_cols":MassProperties::default().transform_by(&pose).reconstruct_inertia_matrix().to_cols_array(),
            "compound_mass_import_passed":compound_mass_import_passed,
            "compound_primitive_contact_queries_passed":geometry_passed,"geometry_controls":geometry,
            "compound_full_collision_dispatch_qualified":false,"robot_assembly_qualified":false,
            "target_plant_accepted":false,
        }))?,
    )?;
    println!(
        "original_configuration_passed={} diagnostic_passed={diagnostic_passed}",
        !original_failed
    );
    if !diagnostic_passed {
        return Err("massless collider diagnosis/control gates did not match".into());
    }
    Ok(())
}
