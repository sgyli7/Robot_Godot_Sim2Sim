//! Inspect the actual native articulation without integrating or admitting control.

use robot_minigame::{
    basis::{source_to_engine_rotation, source_to_engine_vector},
    definition::RobotDefinition,
    kinematics::source_body_poses,
};
use serde_json::{Value, json};
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};
use std::{error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[2] != "--output" {
        return Err("expected compiled model path, SHA256, --output result.json".into());
    }
    let path = Path::new(&args[0]);
    let definition = match path.extension().and_then(|extension| extension.to_str()) {
        Some("ron") => RobotDefinition::load_ron(path, &args[1])?,
        Some("json") => RobotDefinition::load_json(path, &args[1])?,
        _ => return Err("expected .ron or .json compiled model".into()),
    };
    let model = definition.model();
    let mut perturbation = model.fields.key_qpos[0].clone();
    perturbation[..3].copy_from_slice(&[0.3, -0.2, 0.45]);
    let half: f64 = 0.13;
    perturbation[3] = half.cos();
    perturbation[4..7].fill(half.sin() / 3.0_f64.sqrt());
    for joint in 1..model.counts.njnt {
        perturbation[model.fields.jnt_qposadr[joint]] += 0.05 * (joint as f64).sin();
    }
    let mut cases = Vec::new();
    for (name, qpos) in [
        ("source_home", model.fields.key_qpos[0].as_slice()),
        (
            "translated_tilted_joint_perturbation",
            perturbation.as_slice(),
        ),
    ] {
        cases.push(inspect(&definition, name, qpos)?);
    }
    // Bad state must fail before inserting native objects in an existing world.
    let mut simulation = SimulationWorld::foundation();
    let before = simulation.counts();
    let mut invalid_qpos = model.fields.key_qpos[0].clone();
    invalid_qpos[7] = f64::INFINITY;
    let rejected = build_structure(&mut simulation.world, &definition, &invalid_qpos).is_err();
    let unchanged = before == simulation.counts();
    let passed = cases.iter().all(|case| case["passed"] == true) && rejected && unchanged;
    let report = json!({
        "scope": "native_articulation_initialization_only",
        "family": model.family, "model_file_sha256": definition.file_sha256(),
        "cases": cases,
        "invalid_qpos_rejected": rejected, "preexisting_world_counts_preserved": unchanged,
        "physics_integrations": 0, "policy_inferences": 0,
        "contact_filter_qualified": false, "complete_force_observation": false,
        "target_plant_accepted": false, "passed": passed,
    });
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!(
        "{}",
        serde_json::to_string(&json!({"output": args[3], "passed": passed}))?
    );
    if !passed {
        return Err("native articulation initialization comparison failed".into());
    }
    Ok(())
}

fn inspect(
    definition: &RobotDefinition,
    name: &str,
    qpos: &[f64],
) -> Result<Value, Box<dyn Error>> {
    let mut simulation = SimulationWorld::new();
    let assembly = build_structure(&mut simulation.world, definition, qpos)?;
    let model = definition.model();
    let fields = &model.fields;
    let expected = source_body_poses(definition, qpos)?;
    let mut rows = Vec::new();
    let mut position_error = 0.0_f64;
    let mut rotation_error = 0.0_f64;
    let mut mass_error = 0.0_f64;
    let mut com_error = 0.0_f64;
    let mut inertia_error = 0.0_f64;
    for body_id in 1..model.counts.nbody {
        let body =
            &simulation.world.bodies[assembly.body_handles()[body_id].ok_or("missing body")?];
        let position = body.translation().to_array();
        let rotation = body.rotation().to_array();
        let expected_position =
            source_to_engine_vector(expected[body_id].position.map(|v| v as f32));
        let expected_rotation =
            source_to_engine_rotation(expected[body_id].rotation_wxyz.map(|v| v as f32))?;
        position_error = position_error.max(max_error(&position, &expected_position));
        let positive = max_error(&rotation, &expected_rotation);
        let negative = max_error(&rotation, &expected_rotation.map(|v| -v));
        rotation_error = rotation_error.max(positive.min(negative));
        mass_error = mass_error.max((f64::from(body.mass()) - fields.body_mass[body_id]).abs());
        let properties = &body.mass_properties().local_mprops;
        com_error = com_error.max(max_error(
            &properties.local_com.to_array(),
            &source_to_engine_vector(fields.body_ipos[body_id].map(|v| v as f32)),
        ));
        let tensor = properties.reconstruct_inertia_matrix().to_cols_array();
        let expected_tensor =
            expected_tensor(fields.body_iquat[body_id], fields.body_inertia[body_id]);
        for (actual, expected) in tensor.iter().zip(expected_tensor) {
            inertia_error = inertia_error.max((f64::from(*actual) - expected).abs());
        }
        rows.push(json!({"source_body": body_id, "name": model.names.body[body_id],
            "translation": position, "rotation_xyzw": rotation, "mass": body.mass(),
            "local_com": properties.local_com.to_array(), "local_inertia_cols": tensor,
            "linear_velocity": body.linvel().to_array(), "angular_velocity": body.angvel().to_array(),
        }));
    }
    let states = assembly.joint_state(&simulation.world)?;
    let mut joint_error = 0.0_f64;
    let mut passive_error = 0.0_f64;
    let mut joint_rows = Vec::new();
    for mapping in assembly.joint_mapping() {
        let (_, q, qd) = states
            .iter()
            .find(|state| state.0 == mapping.source_joint)
            .ok_or("missing state")?;
        joint_error =
            joint_error.max((f64::from(*q) - qpos[fields.jnt_qposadr[mapping.source_joint]]).abs());
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get(mapping.handle)
            .ok_or("missing articulation")?;
        let slot = mapping.backend_dof;
        for (actual, expected) in [
            (
                multibody.damping()[slot],
                fields.dof_damping[mapping.source_dof],
            ),
            (
                multibody.armature()[slot],
                fields.dof_armature[mapping.source_dof],
            ),
            (
                multibody.frictions()[slot],
                fields.dof_frictionloss[mapping.source_dof],
            ),
        ] {
            passive_error = passive_error.max((f64::from(actual) - expected).abs());
        }
        joint_rows.push(
            json!({"source_joint": mapping.source_joint, "source_dof": mapping.source_dof,
            "backend_dof": slot, "q": q, "qd": qd, "damping": multibody.damping()[slot],
            "armature": multibody.armature()[slot], "frictionloss": multibody.frictions()[slot]}),
        );
    }
    let expected_colliders = (0..model.counts.ngeom)
        .filter(|geom| {
            fields.geom_bodyid[*geom] != 0
                && (fields.geom_contype[*geom] != 0 || fields.geom_conaffinity[*geom] != 0)
        })
        .count();
    let mut geometry_rows = Vec::new();
    for (geom_id, handle) in assembly.collision_handles() {
        let collider = &simulation.world.colliders[*handle];
        let compound = collider
            .shape()
            .as_compound()
            .ok_or("missing source geom child")?;
        if compound.shapes().len() != 1 {
            return Err("source collider must retain exactly one geom child".into());
        }
        let (local_pose, leaf) = &compound.shapes()[0];
        let world_pose = *collider.position() * *local_pose;
        let polyhedron = leaf
            .as_convex_polyhedron()
            .ok_or("expected source mesh hull")?;
        let world_points: Vec<_> = polyhedron
            .points()
            .iter()
            .map(|point| world_pose.transform_point(*point).to_array())
            .collect();
        geometry_rows.push(json!({"source_geom": geom_id,"source_body":fields.geom_bodyid[*geom_id],
            "name":model.names.geom[*geom_id],"source_mesh":fields.geom_dataid[*geom_id],
            "leaf_world_translation":world_pose.translation.to_array(),
            "leaf_world_rotation_xyzw":world_pose.rotation.to_array(),
            "hull_world_vertices":world_points,
            "collider_local_identity":collider.position_wrt_parent().is_some_and(|pose| *pose == Default::default()),
            "density":collider.density()}));
    }
    let counts = simulation.counts();
    let snapshot = simulation.snapshot();
    let frame = assembly.pose_frame(&snapshot)?;
    let mut refusal_checks = Vec::new();
    let mut missing = snapshot.clone();
    missing.bodies.pop();
    refusal_checks.push(("missing_body", assembly.pose_frame(&missing).is_err()));
    let mut duplicate = snapshot.clone();
    duplicate.bodies.push(duplicate.bodies[0].clone());
    refusal_checks.push(("duplicate_handle", assembly.pose_frame(&duplicate).is_err()));
    let mut nonfinite = frame.clone();
    nonfinite.poses[0].translation[0] = f32::NAN;
    refusal_checks.push((
        "nonfinite_pose",
        nonfinite
            .validate(definition.file_sha256(), model.counts.nbody)
            .is_err(),
    ));
    let mut wrong_identity = frame.clone();
    wrong_identity.model_file_sha256 = "0".repeat(64);
    refusal_checks.push((
        "wrong_model_identity",
        wrong_identity
            .validate(definition.file_sha256(), model.counts.nbody)
            .is_err(),
    ));
    let mut wrong_quaternion = frame.clone();
    wrong_quaternion.poses[0].rotation_xyzw = [0.0; 4];
    refusal_checks.push((
        "nonunit_quaternion",
        wrong_quaternion
            .validate(definition.file_sha256(), model.counts.nbody)
            .is_err(),
    ));
    // The initialization evidence above remains unchanged. Exercise a genuine
    // native removal/reinsertion afterward to reject the old handle generation.
    let removed = assembly
        .body_handles()
        .last()
        .copied()
        .flatten()
        .ok_or("missing generation-refusal body")?;
    simulation.remove_body(removed)?;
    let replacement = simulation
        .world
        .bodies
        .insert(rapier3d::prelude::RigidBodyBuilder::dynamic().additional_solver_iterations(0));
    let removed_parts = removed.into_raw_parts();
    let replacement_parts = replacement.into_raw_parts();
    refusal_checks.push((
        "actual_removed_reinserted_generation",
        removed_parts != replacement_parts && assembly.pose_frame(&simulation.snapshot()).is_err(),
    ));
    let passed = position_error <= 1e-6
        && rotation_error <= 1e-6
        && joint_error <= 1e-6
        && mass_error <= 1e-7
        && com_error <= 1e-7
        && inertia_error <= 2e-10
        && passive_error <= 1e-8
        && counts.bodies == model.counts.nbody - 1
        && counts.colliders == expected_colliders
        && counts.multibody_joint_handles == model.counts.njnt - 1
        && states.iter().all(|state| state.2 == 0.0)
        && !assembly.control_ready()
        && refusal_checks.iter().all(|(_, refused)| *refused);
    Ok(
        json!({"name": name, "qpos_source": qpos, "native_bodies": rows, "native_joints": joint_rows,
        "native_collision_geoms":geometry_rows,
        "body_pose_frame":frame, "pose_frame_refusal_checks":refusal_checks,
        "actual_removed_handle":removed_parts, "actual_replacement_handle":replacement_parts,
        "world_counts": counts, "expected_collision_geoms": expected_colliders,
        "position_max_abs": position_error, "rotation_max_abs": rotation_error,
        "joint_max_abs": joint_error, "mass_max_abs": mass_error,
        "local_com_max_abs": com_error, "inertia_tensor_max_abs": inertia_error,
        "passive_scalar_max_abs": passive_error, "passed": passed}),
    )
}

fn max_error<const N: usize>(actual: &[f32; N], expected: &[f32; N]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| f64::from((*a - *b).abs()))
        .fold(0.0, f64::max)
}

/// Independent f64 tensor construction and signed basis permutation.
fn expected_tensor([w, x, y, z]: [f64; 4], diagonal: [f64; 3]) -> [f64; 9] {
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let axes = [0, 2, 1];
    let signs = [1.0, 1.0, -1.0];
    std::array::from_fn(|index| {
        let row = index % 3;
        let col = index / 3;
        signs[row]
            * signs[col]
            * (0..3)
                .map(|k| r[axes[row]][k] * diagonal[k] * r[axes[col]][k])
                .sum::<f64>()
    })
}
