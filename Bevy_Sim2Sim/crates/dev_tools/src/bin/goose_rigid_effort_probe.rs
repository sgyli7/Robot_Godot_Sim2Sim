//! Common external-effort physics diagnostic, never a policy/controller test.

use std::{env, fs, path::PathBuf, time::Instant};

use dev_tools_minigame::goose_rigid_diagnostic::boundary;
use rapier3d::prelude::*;
use robot_minigame::goose::plant::GoosePlant;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{PhysicsClockProfile, SimulationWorld, goose::builder::GooseAssembly};

#[derive(Deserialize)]
struct Input {
    schema: String,
    model_sha256: String,
    contract_sha256: String,
    initial_state_sha256: String,
    dt_s: f64,
    gravity_source_m_s2: [f32; 3],
    floor: bool,
    ground_friction: f32,
    ground_contype: u32,
    ground_conaffinity: u32,
    efforts_nm: Vec<[f64; 18]>,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.len() != 4 {
        return Err(
            "usage: goose_rigid_effort_probe PLANT.json CONTROL.json INPUT.json OUTPUT.json".into(),
        );
    }
    if paths[3].exists() {
        return Err("Preserve existing evidence; output already exists".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&paths[0])?;
    if !plant.is_rigid_native() {
        return Err("Rigid native plant required".into());
    }
    let contract = plant.bound_control_contract(&fs::read(&paths[1])?)?;
    let bytes = fs::read(&paths[2])?;
    let input: Input = serde_json::from_slice(&bytes)?;
    if input.schema != "goose_common_external_effort_v1"
        || input.model_sha256 != plant.model_sha256
        || input.contract_sha256 != plant.derived_contract_sha256
        || input.initial_state_sha256
            != plant
                .native_initialization
                .as_ref()
                .unwrap()
                .initial_state_sha256
        || input.dt_s != 0.02
        || input.efforts_nm.is_empty()
        || input.efforts_nm.len() > 1000
        || !input.gravity_source_m_s2.iter().all(|v| v.is_finite())
        || !input.ground_friction.is_finite()
        || input.ground_friction < 0.0
        || input.efforts_nm.iter().any(|row| {
            row.iter().enumerate().any(|(i, v)| {
                !v.is_finite() || v.abs() > contract.joints[i].torque_peak_limit_nm + 1e-9
            })
        })
    {
        return Err("Common effort input identity/timing/finite/peak bound mismatch".into());
    }
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    // Convergence passes inside one real 20 ms step; never time subdivision.
    simulation
        .world
        .integration_parameters
        .num_internal_pgs_iterations = 4;
    simulation.world.gravity = Vector::from_array(robot_minigame::basis::source_to_engine_vector(
        input.gravity_source_m_s2,
    ));
    if input.floor {
        let floor = simulation.world.bodies.insert(RigidBodyBuilder::fixed());
        simulation.world.colliders.insert_with_parent(
            ColliderBuilder::halfspace(rapier3d::na::Unit::new_unchecked(Vector::Y))
                .friction(input.ground_friction)
                .friction_combine_rule(CoefficientCombineRule::Max)
                .collision_groups(InteractionGroups::new(
                    Group::from_bits_retain(input.ground_contype),
                    Group::from_bits_retain(input.ground_conaffinity),
                    InteractionTestMode::Or,
                )),
            floor,
            &mut simulation.world.bodies,
        );
    }
    let assembly = GooseAssembly::build(&mut simulation, &plant)?;
    let support = dev_tools_minigame::goose_rigid_diagnostic::support_vertices(
        &simulation,
        &assembly,
        &plant,
    )?;
    let initial = boundary(&simulation, &assembly, &plant, &contract, &support)?;
    let initial_snapshot = simulation.snapshot();
    let mut rows = Vec::new();
    let mut failure: Option<String> = None;
    for torque in &input.efforts_nm {
        let start = Instant::now();
        let step = assembly.step(&mut simulation, *torque);
        let physics_ms = start.elapsed().as_secs_f64() * 1000.0;
        let snapshot = simulation.snapshot();
        let measured = boundary(&simulation, &assembly, &plant, &contract, &support);
        let mut row =
            json!({"snapshot": snapshot, "requested_torque_nm": torque, "physics_ms": physics_ms});
        match measured {
            Ok(value) => row["boundary"] = value,
            Err(error) => failure = Some(error.to_string()),
        }
        if let Err(error) = step {
            failure = Some(error.to_string());
        }
        rows.push(row);
        if failure.is_some() {
            break;
        }
    }
    let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let code = [
        "crates/modules/robot/src/goose/plant.rs",
        "crates/modules/robot/src/goose/contract.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/dev_tools/src/bin/goose_rigid_effort_probe.rs",
        "crates/dev_tools/src/goose_rigid_diagnostic.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/unit_multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/sim2sim_observation.rs",
        "Cargo.lock",
    ];
    let hashes: serde_json::Map<String, Value> = code
        .into_iter()
        .map(|name| fs::read(project.join(name)).map(|b| (name.to_owned(), json!(digest(&b)))))
        .collect::<Result<_, _>>()?;
    let executable = env::current_exe()?;
    let receipt = json!({"schema":"goose_rigid_common_effort_receipt_v1", "engine":"Rapier 0.35.3",
        "candidate":plant.candidate_id, "plant_sha256":plant_hash, "input_sha256":digest(&bytes),
        "model_sha256":plant.model_sha256, "contract_sha256":plant.derived_contract_sha256,
        "runtime_code_sha256":hashes, "executable_sha256":digest(&fs::read(executable)?),
        "scope":"same exogenous motor efforts; neither implicit drive nor Actor migration qualification",
        "inference_calls":0, "optimizer_updates":0, "qualified":false, "target_qualified":false,
        "cold_root_lift_m":0, "configuration":simulation.configuration(), "counts":simulation.counts(),
        "source_leaves":assembly.source_geometry_count, "native_collider_groups":assembly.collider_source_groups,
        "body_handles":assembly.body_handles.iter().map(|(name,handle)| {
            let (index,generation)=handle.into_raw_parts(); (name.clone(),json!([index,generation]))
        }).collect::<serde_json::Map<String,Value>>(),
        "mass_and_full_inertia":assembly.body_measurements, "failure":failure, "initial":initial,
        "initial_snapshot":initial_snapshot,
        "actual_integrals":simulation.snapshot().integration_count,
        "actual_torque_updates":simulation.snapshot().torque_update_count, "trace":rows});
    fs::write(&paths[3], serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "{}",
        json!({"output":paths[3],"actual_integrals":receipt["actual_integrals"], "failure":receipt["failure"]})
    );
    Ok(())
}
