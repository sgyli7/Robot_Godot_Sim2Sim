//! Verify a task-proxy model/control/plant binding without creating a world.

use std::{env, fs, path::PathBuf};

use robot_minigame::goose::plant::GoosePlant;
use serde_json::json;
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.len() != 3 {
        return Err("usage: goose_task_proxy_intake MODEL.xml CONTROL.json PLANT.json".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&paths[2])?;
    if !plant.is_task_proxy() {
        return Err("task-proxy intake requires its explicit schema and candidate".into());
    }
    let model_hash = format!("{:x}", Sha256::digest(fs::read(&paths[0])?));
    if model_hash != plant.model_sha256 {
        return Err("task-proxy model bytes differ from the plant binding".into());
    }
    let control_bytes = fs::read(&paths[1])?;
    let control = plant.bound_control_contract(&control_bytes)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "goose_lab_task_proxy_intake_v1",
            "candidate": plant.candidate_id,
            "plant_sha256": plant_hash,
            "contract_sha256": plant.derived_contract_sha256,
            "model_sha256": model_hash,
            "declared_robot_bodies": plant.bodies.len(),
            "declared_tree_joints": plant.joints.len(),
            "declared_active_axes": plant.joints.iter().filter(|joint| joint.driven).count(),
            "declared_actual_convex_leaves": plant.colliders.len(),
            "declared_explicit_exclusions": plant.exclusions.len(),
            "mass_kg": plant.robot_mass_kg,
            "runtime_revision": control.runtime_revision,
            "controller_revision": control.controller_revision,
            "physics_dt_s": control.physics_dt_s,
            "torque_dt_s": control.torque_dt_s,
            "policy_dt_s": control.policy_dt_s,
            "observation_size": control.observation_size,
            "action_size": control.action_size,
            "world_created": false,
            "integrations": 0,
            "GPU_calls": 0,
            "physical_admission": false,
            "Bevy_qualified": false,
            "next_requirement": "reuse and independently admit the versioned sole contact runtime before target stepping"
        }))?
    );
    Ok(())
}
