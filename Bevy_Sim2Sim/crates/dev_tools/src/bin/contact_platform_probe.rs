//! One original MicroDuck against a SHA-bound static platform, for contact diagnosis.
//! This direct-world experiment is outside SourceCollisionWorld and cannot qualify
//! the source contact filter, the BAM plant, or policy behavior.

use std::{env, fs, path::Path};

use rapier3d::prelude::*;
use robot_minigame::{basis::source_to_engine_vector, definition::RobotDefinition};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};

const FROZEN_LEG_MODEL_SHA256: &str =
    "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb";

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_checked(path: &str, expected: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if digest(&bytes) != expected {
        return Err(format!("SHA256 mismatch: {path}"));
    }
    Ok(bytes)
}

fn triple(value: &Value, key: &str) -> Result<[f32; 3], String> {
    let numbers = value[key]
        .as_array()
        .ok_or_else(|| format!("{key} must be an array"))?;
    if numbers.len() != 3 {
        return Err(format!("{key} must have three entries"));
    }
    let result: [f32; 3] = numbers
        .iter()
        .map(|number| {
            number
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| format!("{key} must contain numbers"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| format!("{key} must have three entries"))?;
    if result.iter().any(|number| !number.is_finite()) {
        return Err(format!("{key} exceeds float32 range"));
    }
    Ok(result)
}

fn run(args: &[String]) -> Result<Value, String> {
    let steps: usize = if args.len() >= 8 {
        args[7].parse().map_err(|_| "invalid step count")?
    } else {
        1
    };
    if !(1..=8).contains(&steps) {
        return Err("platform probe supports one through eight 60 Hz steps".into());
    }
    let pgs_iterations: usize = if args.len() >= 9 {
        args[8].parse().map_err(|_| "invalid internal PGS count")?
    } else {
        1
    };
    if !(1..=16).contains(&pgs_iterations) {
        return Err("internal PGS count must be 1..=16".into());
    }
    let foot_friction_override: Option<f32> = if args.len() == 10 {
        let friction: f32 = args[9]
            .parse()
            .map_err(|_| "invalid diagnostic foot friction")?;
        if !friction.is_finite() || !(0.0..=2.0).contains(&friction) {
            return Err("diagnostic foot friction must be finite and in 0..=2".into());
        }
        Some(friction)
    } else {
        None
    };
    let definition_bytes = read_checked(&args[0], &args[1])?;
    if digest(&definition_bytes) != FROZEN_LEG_MODEL_SHA256 {
        return Err("platform probe requires the frozen original leg definition".into());
    }
    let definition = RobotDefinition::load_json(Path::new(&args[0]), &args[1])
        .map_err(|error| error.to_string())?;
    if definition.model().family != "leg_allcollisions" {
        return Err("platform probe accepts only the original leg family".into());
    }
    let qpos_bytes = read_checked(&args[2], &args[3])?;
    let qpos: Vec<f64> = serde_json::from_slice(&qpos_bytes).map_err(|error| error.to_string())?;
    let platform_bytes = read_checked(&args[4], &args[5])?;
    let platform: Value =
        serde_json::from_slice(&platform_bytes).map_err(|error| error.to_string())?;
    let center = triple(&platform, "source_center")?;
    let half_extents = triple(&platform, "source_half_extents")?;
    if half_extents.iter().any(|extent| *extent <= 0.0) {
        return Err("platform half-extents must be positive".into());
    }
    let friction = platform["sliding_friction"]
        .as_f64()
        .ok_or("sliding_friction must be a number")? as f32;
    if !friction.is_finite() || friction < 0.0 {
        return Err("sliding_friction must be nonnegative and finite".into());
    }
    let mut simulation = SimulationWorld::new();
    simulation
        .world
        .integration_parameters
        .num_internal_pgs_iterations = pgs_iterations;
    let params = &simulation.world.integration_parameters;
    if (params.dt - 1.0 / 60.0).abs() > 1.0e-8
        || params.num_solver_iterations != 1
        || params.max_ccd_substeps != 1
    {
        return Err("platform probe requires one 1/60 s physics step".into());
    }
    let platform_body = simulation.world.bodies.insert(
        RigidBodyBuilder::fixed()
            .translation(Vector::from_array(source_to_engine_vector(center)))
            .user_data(0),
    );
    let platform_collider = simulation.world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(half_extents[0], half_extents[2], half_extents[1])
            .friction(friction)
            .restitution(0.0)
            .user_data(u128::MAX),
        platform_body,
        &mut simulation.world.bodies,
    );
    let assembly = build_structure(&mut simulation.world, &definition, &qpos)
        .map_err(|error| error.to_string())?;
    if let Some(friction) = foot_friction_override {
        // Scratch-only contact-law isolation. The original definition remains
        // SHA-bound; only this copied world's one left-foot collider changes.
        let foot: Vec<_> = assembly
            .collision_handles()
            .iter()
            .filter(|(source_geom, _)| *source_geom == 28)
            .map(|(_, handle)| *handle)
            .collect();
        if foot.len() != 1 {
            return Err("frozen left-foot collider identity is not unique".into());
        }
        simulation
            .world
            .colliders
            .get_mut(foot[0])
            .ok_or("frozen left-foot collider is missing")?
            .set_friction(friction);
    }
    let mut step_reports = Vec::with_capacity(steps);
    for step in 1..=steps {
        let before_snapshot = simulation.snapshot();
        let before_pose = assembly
            .pose_frame(&before_snapshot)
            .map_err(|error| error.to_string())?;
        let before_joints = assembly
            .actuator_joint_feedback(&simulation.world, &definition)
            .map_err(|error| error.to_string())?;
        let snapshot = simulation
            .step_with_torques(&[])
            .map_err(|error| error.to_string())?;
        let after_pose = assembly
            .pose_frame(&snapshot)
            .map_err(|error| error.to_string())?;
        let after_joints = assembly
            .actuator_joint_feedback(&simulation.world, &definition)
            .map_err(|error| error.to_string())?;
        let mut contact_pairs = Vec::new();
        for pair in simulation.world.narrow_phase.contact_pairs() {
            let first = &simulation.world.colliders[pair.collider1];
            let second = &simulation.world.colliders[pair.collider2];
            let first_geom = if pair.collider1 == platform_collider {
                json!("platform")
            } else {
                json!(first.user_data as u64)
            };
            let second_geom = if pair.collider2 == platform_collider {
                json!("platform")
            } else {
                json!(second.user_data as u64)
            };
            contact_pairs.push(json!({
            "source_geom_ids_or_platform": [first_geom, second_geom],
            "platform_pair": pair.collider1 == platform_collider || pair.collider2 == platform_collider,
            "active": pair.has_any_active_contact(),
            "solver_contact_count": pair.manifolds.iter().map(|manifold| manifold.data.solver_contacts.len()).sum::<usize>(),
            "normal_impulse_sum_ns": pair.manifolds.iter().flat_map(|manifold| &manifold.points).map(|point| point.data.impulse).sum::<f32>(),
            "tangent_impulse_sum_abs_ns": pair.manifolds.iter().flat_map(|manifold| &manifold.points).map(|point| point.data.tangent_impulse.norm()).sum::<f32>(),
            "manifolds": pair.manifolds.iter().map(|manifold| json!({
                "normal_engine": manifold.data.normal.to_array(),
                "effective_sliding_friction": manifold.data.friction,
                "contacts": manifold.data.solver_contacts.iter().map(|contact| {
                    let id = (contact.contact_id[0] & !rapier3d::geometry::NEW_CONTACT_BIT) as usize;
                    let point = &manifold.points[id];
                    json!({
                        "anchor1_engine": contact.anchor1.to_array(),
                        "anchor2_engine": contact.anchor2.to_array(),
                        "initial_distance_m": contact.dist,
                        "normal_impulse_ns": point.data.impulse,
                        "tangent_impulse_basis_nms": [point.data.tangent_impulse.x, point.data.tangent_impulse.y],
                        "next_step_warmstart_tangent_world_engine_nms": point.data.warmstart_tangent_world.to_array(),
                    })
                }).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }));
        }
        let first_mapping = assembly
            .joint_mapping()
            .first()
            .ok_or("robot has no mapped joints")?;
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get(first_mapping.handle)
            .ok_or("robot articulation is absent")?;
        let observation = multibody
            .sim2sim_contact_complete_observation()
            .ok_or("native contact observation is incomplete")?;
        if observation.epoch != step as u64
            || !observation.contact_coverage
            || snapshot.integration_count != step as u64
        {
            return Err("native contact observation is not one complete step".into());
        }
        let actuator_rows: Vec<_> = before_joints
            .iter()
            .zip(after_joints.iter())
            .map(|(before, after)| {
                let dof = before.backend_dof;
                json!({
                    "source_joint": before.source_joint,
                    "source_dof": before.source_dof,
                    "backend_dof": dof,
                    "initial_position_rad": before.position,
                    "initial_velocity_rad_s": before.velocity,
                    "post_position_rad": after.position,
                    "post_velocity_rad_s": after.velocity,
                    "contact_normal_impulse_nms": observation.contact_normal_impulse[dof],
                    "contact_tangent_impulse_nms": observation.contact_tangent_impulse[dof],
                    "generic_joint_impulse_nms": observation.generic_joint_impulse[dof],
                    "own_dry_friction_impulse_nms": observation.own_dry_friction_impulse[dof],
                    "inertial_projection_nm": observation.inertial_projection[dof],
                    "gravity_projection_nm": observation.gravity_projection[dof],
                    "user_force_projection_nm": observation.user_force_projection[dof],
                })
            })
            .collect();
        step_reports.push(json!({
            "step_index": step,
            "physics_integrations": snapshot.integration_count,
            "policy_inferences": 0,
            "initial_pose": before_pose,
            "post_pose": after_pose,
            "actuator_rows": actuator_rows,
            "contact_pair_count": snapshot.contact_pair_count,
            "active_contact_pair_count": snapshot.active_contact_pair_count,
            "contact_pairs": contact_pairs,
            "all_dof_contact_normal_impulse_nms": observation.contact_normal_impulse,
            "all_dof_contact_tangent_impulse_nms": observation.contact_tangent_impulse,
            "post_generalized_velocity": multibody.generalized_velocity().as_slice(),
            "post_generalized_acceleration": multibody.generalized_acceleration().as_slice(),
            "contact_expected_side_count": observation.contact_expected_side_count,
            "contact_measured_side_count": observation.contact_measured_side_count,
        }));
    }
    let mut report = json!({
        "schema": if steps == 1 { "original_leg_static_platform_rapier_contact_probe_v1" } else { "original_leg_static_platform_rapier_contact_trajectory_v1" },
        "scope": "direct_world_contact_diagnostic_not_source_collision_or_BAM_qualification",
        "model_path": args[0],
        "model_sha256": digest(&definition_bytes),
        "qpos_sha256": digest(&qpos_bytes),
        "platform_sha256": digest(&platform_bytes),
        "platform": platform,
        "world_count": 1,
        "requested_steps": steps,
        "internal_pgs_iterations": pgs_iterations,
        "step_configuration": simulation.configuration(),
        "source_contact_qualified": false,
        "bam_external_load_qualified": false,
        "skill_qualified": false,
    });
    if let Some(friction) = foot_friction_override {
        report["diagnostic_foot_friction_override"] = json!(friction);
    }
    if steps == 1 {
        for (key, value) in step_reports
            .pop()
            .ok_or("missing single-step report")?
            .as_object()
            .ok_or("single-step report is not an object")?
        {
            report[key] = value.clone();
        }
    } else {
        report["trace"] = json!(step_reports);
        report["physics_integrations"] = json!(steps);
        report["policy_inferences"] = json!(0);
    }
    Ok(report)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if !(7..=10).contains(&args.len()) {
        return Err("usage: contact_platform_probe MODEL.json SHA QPOS.json SHA PLATFORM.json SHA NEW_REPORT.json [STEPS 1..8 [INTERNAL_PGS 1..16 [DIAGNOSTIC_FOOT_FRICTION 0..=2]]]".into());
    }
    let output = Path::new(&args[6]);
    if output.exists() {
        return Err("report output must be new".into());
    }
    let report = run(&args)?;
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!("STATUS: success; report={}", output.display());
    Ok(())
}
