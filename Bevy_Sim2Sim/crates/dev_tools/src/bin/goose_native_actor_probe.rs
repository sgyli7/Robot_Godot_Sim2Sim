//! Deterministic source Actor on the named Rapier native motor revision.

use dev_tools_minigame::goose_rigid_diagnostic::{boundary, support_vertices};
use rapier3d::prelude::*;
use robot_minigame::{
    goose::{
        gravity::GooseNominalGravity,
        native_drive::{GOOSE_RAPIER_DRIVE_REVISION, GooseNativeDrive},
        plant::GoosePlant,
    },
    policy::CpuPolicy,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{PhysicsClockProfile, SimulationWorld, goose::builder::GooseAssembly};
use std::{env, fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Input {
    schema: String,
    model_sha256: String,
    contract_sha256: String,
    initial_state_sha256: String,
    actor_sha256: String,
    dt_s: f64,
    commands: Vec<[f64; 3]>,
    #[serde(default)]
    replay_actions: Option<Vec<[f64; 18]>>,
}

fn digest(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.len() != 6 {
        return Err(
            "usage: goose_native_actor_probe PLANT CONTROL ORT_LIB ACTOR INPUT OUTPUT".into(),
        );
    }
    if paths[5].exists() {
        return Err("Preserve evidence: output exists".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&paths[0])?;
    let contract_bytes = fs::read(&paths[1])?;
    let contract = plant.bound_control_contract(&contract_bytes)?;
    let input_bytes = fs::read(&paths[4])?;
    let input: Input = serde_json::from_slice(&input_bytes)?;
    let replay = input.schema == "goose_native_motor_replay_v1";
    if !plant.is_rigid_native()
        || !matches!(
            input.schema.as_str(),
            "goose_native_motor_actor_v1" | "goose_native_motor_replay_v1"
        )
        || replay != input.replay_actions.is_some()
        || input.replay_actions.as_ref().is_some_and(|rows| {
            rows.len() != input.commands.len() || rows.iter().flatten().any(|v| !v.is_finite())
        })
        || input.model_sha256 != plant.model_sha256
        || input.contract_sha256 != plant.derived_contract_sha256
        || input.initial_state_sha256
            != plant
                .native_initialization
                .as_ref()
                .ok_or("Cold state absent")?
                .initial_state_sha256
        || input.dt_s != 0.02
        || input.commands.is_empty()
        || input.commands.len() > 3000
        || input.commands.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("Actor protocol identity/clock/commands mismatch".into());
    }
    let mut actor =
        CpuPolicy::<65, 18>::load_checked_actor(&paths[2], &paths[3], &input.actor_sha256)?;
    let gravity = GooseNominalGravity::from_contract_bytes(&contract_bytes)?;
    let mut drive = GooseNativeDrive::new(contract.clone(), gravity, 1.0, false)?;
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    simulation
        .world
        .integration_parameters
        .num_internal_pgs_iterations = 4;
    let floor = simulation.world.bodies.insert(RigidBodyBuilder::fixed());
    simulation.world.colliders.insert_with_parent(
        ColliderBuilder::halfspace(rapier3d::na::Unit::new_unchecked(Vector::Y))
            .friction(0.65)
            .friction_combine_rule(CoefficientCombineRule::Max)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                Group::GROUP_1,
                InteractionTestMode::Or,
            )),
        floor,
        &mut simulation.world.bodies,
    );
    let assembly = GooseAssembly::build(&mut simulation, &plant)?;
    let support = support_vertices(&simulation, &assembly, &plant)?;
    let initial = boundary(&simulation, &assembly, &plant, &contract, &support)?;
    let initial_snapshot = simulation.snapshot();
    let mut trace = Vec::new();
    let mut failure = None;
    for (tick, command) in input.commands.iter().enumerate() {
        let mut row = json!({"command":command});
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let state = assembly.state(&simulation)?;
            let observation =
                contract.observation(&state, *command, drive.previous_action(), drive.phase())?;
            row["observation_65"] = json!(observation.to_vec());
            let start = Instant::now();
            let action = if let Some(actions) = &input.replay_actions {
                actions[tick]
            } else {
                actor.infer(&observation)?.map(f64::from)
            };
            row["inference_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
            row["action_18"] = json!(action.to_vec());
            let packet =
                drive.prepare(&state, assembly.motor_velocity_rad_s(&simulation)?, action)?;
            row["drive_command"] = serde_json::to_value(&packet)?;
            let start = Instant::now();
            let (_, effort) = assembly.step_native_motors(&mut simulation, &contract, &packet)?;
            row["physics_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
            row["actual_torque_nm"] = json!(effort.to_vec());
            drive.finish(&packet, effort)?;
            row["heat_nm2"] = json!(drive.squared_torque_ewma().to_vec());
            let measured = boundary(&simulation, &assembly, &plant, &contract, &support)?;
            let fallen = measured["upright"].as_f64().ok_or("Upright absent")? < 0.65
                || measured["com_engine_m"][1].as_f64().ok_or("COM absent")? < 0.18;
            row["boundary"] = measured;
            if fallen {
                return Err("Operational fall: preserve failed state".into());
            }
            Ok(())
        })();
        row["snapshot"] = serde_json::to_value(simulation.snapshot())?;
        if let Err(error) = result {
            failure = Some(error.to_string());
            row["failure"] = json!(failure);
        }
        trace.push(row);
        if failure.is_some() {
            break;
        }
    }
    let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let names = [
        "crates/modules/robot/src/goose/gravity.rs",
        "crates/modules/robot/src/goose/native_drive.rs",
        "crates/modules/robot/src/goose/contract.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "crates/modules/robot/src/policy.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/dev_tools/src/goose_rigid_diagnostic.rs",
        "crates/dev_tools/src/bin/goose_native_actor_probe.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/unit_multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/sim2sim_observation.rs",
        "Cargo.lock",
    ];
    let hashes = names
        .into_iter()
        .map(|name| fs::read(project.join(name)).map(|b| (name.to_string(), json!(digest(&b)))))
        .collect::<Result<serde_json::Map<String, Value>, _>>()?;
    let receipt = json!({"schema":"goose_native_actor_receipt_v1","driver_revision":GOOSE_RAPIER_DRIVE_REVISION,
        "input_mode":if replay {"diagnostic action replay; no Actor inference"} else {"deterministic CPU Actor"},
        "effort_measurement":"final Rapier internal motor generalized impulse / real full-step dt plus external bias; axis5 true rotor effort; not interchangeable with source reported actuator_force",
        "source_driver_revision":contract.runtime_revision,"model_sha256":plant.model_sha256,
        "native_joint_friction_contract":plant.native_joint_friction,
        "contract_sha256":plant.derived_contract_sha256,"plant_sha256":plant_hash,"actor_sha256":input.actor_sha256,
        "input_sha256":digest(&input_bytes),"runtime_code_sha256":hashes,"executable_sha256":digest(&fs::read(env::current_exe()?)?),
        "native_onnx_runtime":actor.runtime_identity(),"inference_calls":actor.inference_count(),
        "successful_inferences":actor.successful_inference_count(),"drive_updates":drive.control_count(),
        "actual_integrals":simulation.snapshot().integration_count,"actual_torque_updates":simulation.snapshot().torque_update_count,
        "configuration":simulation.configuration(),"mass_and_full_inertia":assembly.body_measurements,
        "source_leaves":assembly.source_geometry_count,"collider_source_groups":assembly.collider_source_groups,
        "cold_root_lift_m":0,"post_birth_pose_writes":0,"qualified":false,"target_qualified":false,"optimizer_updates":0,
        "body_handles":assembly.body_handles.iter().map(|(name,handle)|{let(i,g)=handle.into_raw_parts();(name.clone(),json!([i,g]))}).collect::<serde_json::Map<String,Value>>(),
        "initial":initial,"initial_snapshot":initial_snapshot,"failure":failure,"trace":trace});
    fs::write(&paths[5], serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "{}",
        json!({"output":paths[5],"actual_integrals":receipt["actual_integrals"],"inferences":receipt["inference_calls"],"failure":receipt["failure"]})
    );
    Ok(())
}
