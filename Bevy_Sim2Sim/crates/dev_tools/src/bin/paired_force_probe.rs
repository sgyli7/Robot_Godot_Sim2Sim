//! One-step, robot-only Rapier force evidence for a separate MuJoCo comparison.
//!
//! This is deliberately outside the guarded source-collision world. It cannot
//! qualify BAM external load or source contacts; it exposes the raw terms and
//! source DOF identity needed to decide whether that bridge is even plausible.

use std::{env, fs, path::Path};

use rapier3d::prelude::Multibody;
use robot_minigame::definition::RobotDefinition;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_checked(path: &str, expected_sha256: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if sha256(&bytes) != expected_sha256 {
        return Err(format!("SHA256 mismatch: {path}"));
    }
    Ok(bytes)
}

fn run(args: &[String]) -> Result<Value, String> {
    if !Multibody::sim2sim_observation_backend_supported() {
        return Err("native Rapier observation backend is unavailable".into());
    }
    let definition_bytes = read_checked(&args[0], &args[1])?;
    let definition = match Path::new(&args[0]).extension().and_then(|v| v.to_str()) {
        Some("json") => RobotDefinition::load_json(Path::new(&args[0]), &args[1]),
        Some("ron") => RobotDefinition::load_ron(Path::new(&args[0]), &args[1]),
        _ => return Err("definition must be JSON or RON".into()),
    }
    .map_err(|error| error.to_string())?;
    let qpos_bytes = read_checked(&args[2], &args[3])?;
    let qpos: Vec<f64> = serde_json::from_slice(&qpos_bytes).map_err(|error| error.to_string())?;
    let mut simulation = SimulationWorld::new();
    let assembly = build_structure(&mut simulation.world, &definition, &qpos)
        .map_err(|error| error.to_string())?;
    let qvel_sha256 = if args.len() == 7 {
        let bytes = read_checked(&args[4], &args[5])?;
        let qvel: Vec<f64> = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if qvel.len() != definition.model().counts.nv
            || qvel.len() < 6
            || qvel.iter().any(|value| !value.is_finite())
            || qvel[..6].iter().any(|value| *value != 0.0)
        {
            return Err("qvel must have source nv finite entries and a stationary root".into());
        }
        let world = &mut simulation.world;
        let first = assembly
            .joint_mapping()
            .first()
            .ok_or("source articulation has no mapped joints")?;
        let (multibody, _) = world
            .multibody_joints
            .get_mut(first.handle)
            .ok_or("source articulation disappeared")?;
        for mapping in assembly.joint_mapping() {
            let velocity = qvel[mapping.source_dof] as f32;
            if !velocity.is_finite() {
                return Err("source qvel exceeds the float32 backend range".into());
            }
            multibody.generalized_velocity_mut()[mapping.backend_dof] = velocity;
        }
        multibody.update_rigid_bodies(&mut world.bodies, true);
        world
            .bodies
            .propagate_modified_body_positions_to_colliders(&mut world.colliders);
        Some(sha256(&bytes))
    } else {
        None
    };
    let before = assembly
        .actuator_joint_feedback(&simulation.world, &definition)
        .map_err(|error| error.to_string())?;
    let snapshot = simulation
        .step_with_torques(&[])
        .map_err(|error| error.to_string())?;
    let after = assembly
        .actuator_joint_feedback(&simulation.world, &definition)
        .map_err(|error| error.to_string())?;
    let mut rows = Vec::with_capacity(before.len());
    let mut common_epoch = None;
    let mut common_topology = None;
    for (initial, final_state) in before.iter().zip(&after) {
        if initial.source_joint != final_state.source_joint
            || initial.source_dof != final_state.source_dof
            || initial.backend_dof != final_state.backend_dof
            || initial.joint_handle != final_state.joint_handle
            || initial.body_handle != final_state.body_handle
        {
            return Err("driven joint identity changed over one step".into());
        }
        let mapping = assembly
            .joint_mapping()
            .iter()
            .find(|mapping| mapping.source_joint == initial.source_joint)
            .ok_or("source joint mapping disappeared")?;
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get(mapping.handle)
            .ok_or("native articulation disappeared")?;
        let (epoch, valid, coverage) = multibody.sim2sim_observation_status();
        if !valid || !coverage || epoch != snapshot.integration_count {
            return Err(format!(
                "native observation incomplete: source joint {} epoch {epoch} valid {valid} contact coverage {coverage}",
                initial.source_joint
            ));
        }
        let observation = multibody
            .sim2sim_contact_complete_observation()
            .ok_or("native observation getter disagrees with status")?;
        if observation.full_step_dt().to_bits()
            != simulation.world.integration_parameters.dt.to_bits()
            || common_epoch.is_some_and(|first| first != epoch)
            || common_topology.is_some_and(|first| first != observation.topology_epoch)
        {
            return Err(
                "native observation tick, dt or topology differs across driven DOFs".into(),
            );
        }
        common_epoch = Some(epoch);
        common_topology = Some(observation.topology_epoch);
        let dof = initial.backend_dof;
        let inertial = *observation
            .inertial_projection
            .get(dof)
            .ok_or("inertial DOF missing")?;
        let gravity = *observation
            .gravity_projection
            .get(dof)
            .ok_or("gravity DOF missing")?;
        let generic = *observation
            .generic_joint_impulse
            .get(dof)
            .ok_or("generic joint DOF missing")?;
        let own_friction = *observation
            .own_dry_friction_impulse
            .get(dof)
            .ok_or("own dry-friction DOF missing")?;
        let contact_normal = *observation
            .contact_normal_impulse
            .get(dof)
            .ok_or("contact normal DOF missing")?;
        let contact_tangent = *observation
            .contact_tangent_impulse
            .get(dof)
            .ok_or("contact tangent DOF missing")?;
        let dt = observation.full_step_dt();
        let residual =
            gravity - inertial + (generic - own_friction + contact_normal + contact_tangent) / dt;
        if [
            inertial,
            gravity,
            generic,
            own_friction,
            contact_normal,
            contact_tangent,
            residual,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("native force term is non-finite".into());
        }
        rows.push(json!({
            "source_joint":initial.source_joint,
            "source_dof":initial.source_dof,
            "backend_dof":dof,
            "initial_position":initial.position,
            "initial_velocity":initial.velocity,
            "post_position":final_state.position,
            "post_velocity":final_state.velocity,
            "inertial_projection_nm":inertial,
            "gravity_projection_nm":gravity,
            "generic_joint_impulse_nms":generic,
            "own_dry_friction_impulse_nms":own_friction,
            "contact_normal_impulse_nms":contact_normal,
            "contact_tangent_impulse_nms":contact_tangent,
            "candidate_residual_nm":residual,
        }));
    }
    let active_contact_pairs: Vec<Value> = simulation
        .world
        .narrow_phase
        .contact_pairs()
        .filter(|pair| pair.has_any_active_contact())
        .map(|pair| {
            let collider1 = &simulation.world.colliders[pair.collider1];
            let collider2 = &simulation.world.colliders[pair.collider2];
            let body1 = collider1.parent().map(|body| simulation.world.bodies[body].user_data);
            let body2 = collider2.parent().map(|body| simulation.world.bodies[body].user_data);
            json!({
                "source_geom_ids":[collider1.user_data,collider2.user_data],
                "source_body_ids":[body1,body2],
                "solver_contact_count":pair.manifolds.iter().map(|m| m.data.solver_contacts.len()).sum::<usize>(),
                "normal_impulse_sum":pair.manifolds.iter().flat_map(|m| &m.points).map(|p| p.data.impulse).sum::<f32>(),
            })
        })
        .collect();
    Ok(json!({
        "scope":"robot_only_rapier_one_step_raw_force_diagnostic",
        "model_file_sha256":sha256(&definition_bytes),
        "qpos_file_sha256":sha256(&qpos_bytes),
        "qvel_file_sha256":qvel_sha256,
        "family":definition.model().family,
        "physics_integrations":snapshot.integration_count,
        "policy_inferences":0,
        "actual_step_dt_seconds":simulation.world.integration_parameters.dt,
        "native_observation_epoch":common_epoch,
        "native_topology_epoch":common_topology,
        "active_contact_pair_count":snapshot.active_contact_pair_count,
        "active_contact_pairs":active_contact_pairs,
        "rows":rows,
        "bam_external_load_qualified":false,
        "source_target_equivalent":false,
    }))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 5 && args.len() != 7 {
        eprintln!(
            "usage: paired_force_probe MODEL MODEL_SHA QPOS QPOS_SHA [QVEL QVEL_SHA] OUTPUT.json"
        );
        std::process::exit(2);
    }
    let report = match run(&args) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("paired force probe: {error}");
            std::process::exit(1);
        }
    };
    let content = serde_json::to_vec_pretty(&report).expect("force report serializes");
    if let Err(error) = fs::write(args.last().expect("output path argument exists"), content) {
        eprintln!("paired force probe: {error}");
        std::process::exit(1);
    }
}
