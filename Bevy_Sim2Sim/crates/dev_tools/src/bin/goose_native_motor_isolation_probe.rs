//! Same absolute motor targets in an ungrounded, zero-gravity real Goose tree.
//! This removes task controllers deliberately and never qualifies a policy.

use dev_tools_minigame::goose_rigid_diagnostic::{boundary, support_vertices};
use rapier3d::prelude::*;
use robot_minigame::goose::{native_drive::GooseNativeMotorCommand, plant::GoosePlant};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use simulation_minigame::{PhysicsClockProfile, SimulationWorld, goose::builder::GooseAssembly};
use std::{env, fs, path::PathBuf};

#[derive(Deserialize)]
struct Input {
    schema: String,
    model_sha256: String,
    contract_sha256: String,
    initial_state_sha256: String,
    dt_s: f64,
    absolute_targets_rad: Vec<[f64; 18]>,
    #[serde(default)]
    disable_joint_friction_for_diagnosis: bool,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.len() != 4 || paths[3].exists() {
        return Err(
            "usage: goose_native_motor_isolation_probe PLANT CONTRACT INPUT NEW_OUTPUT".into(),
        );
    }
    let (plant, plant_hash) = GoosePlant::read(&paths[0])?;
    let contract_bytes = fs::read(&paths[1])?;
    let contract = plant.bound_control_contract(&contract_bytes)?;
    let bytes = fs::read(&paths[2])?;
    let input: Input = serde_json::from_slice(&bytes)?;
    if !plant.is_rigid_native()
        || input.schema != "goose_native_motor_isolation_v1"
        || input.model_sha256 != plant.model_sha256
        || input.contract_sha256 != plant.derived_contract_sha256
        || input.initial_state_sha256
            != plant
                .native_initialization
                .as_ref()
                .ok_or("Cold state absent")?
                .initial_state_sha256
        || input.dt_s != 0.02
        || input.absolute_targets_rad.is_empty()
        || input.absolute_targets_rad.len() > 100
        || input.absolute_targets_rad.iter().any(|row| {
            row.iter().enumerate().any(|(axis, value)| {
                !value.is_finite()
                    || *value < contract.joints[axis].range_rad[0]
                    || *value > contract.joints[axis].range_rad[1]
            })
        })
    {
        return Err("Isolation protocol identity, clock, targets or count mismatch".into());
    }
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    simulation.world.gravity = Vector::ZERO;
    simulation
        .world
        .integration_parameters
        .num_internal_pgs_iterations = 4;
    let assembly = GooseAssembly::build(&mut simulation, &plant)?;
    if input.disable_joint_friction_for_diagnosis {
        let handle = simulation
            .world
            .multibody_joints
            .iter()
            .next()
            .ok_or("Joint absent")?
            .0;
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get_mut(handle)
            .ok_or("Joint stale")?;
        multibody.frictions_mut().fill(0.0);
    }
    let support = support_vertices(&simulation, &assembly, &plant)?;
    let initial = boundary(&simulation, &assembly, &plant, &contract, &support)?;
    let initial_snapshot = simulation.snapshot();
    let mut trace = Vec::new();
    let mut failure = None;
    for (tick, target) in input.absolute_targets_rad.iter().enumerate() {
        let peak = std::array::from_fn(|axis| {
            if axis == 5 {
                0.0
            } else {
                contract.joints[axis].torque_peak_limit_nm
            }
        });
        let packet = GooseNativeMotorCommand {
            update: tick as u64 + 1,
            target_rad: *target,
            motor_target_rad: *target,
            symmetric_limit_nm: peak,
            external_effort_nm: [0.0; 18],
            lower_limit_nm: peak.map(|v| -v),
            upper_limit_nm: peak,
            pre_tick_motor_velocity_rad_s: assembly.motor_velocity_rad_s(&simulation)?,
            motoring_power_bound_w: 0.0,
        };
        let result = assembly.step_native_motors(&mut simulation, &contract, &packet);
        let mut row = json!({"absolute_target_rad":target});
        match result {
            Ok((_, effort)) => {
                row["actual_solved_torque_nm"] = json!(effort.to_vec());
                match boundary(&simulation, &assembly, &plant, &contract, &support) {
                    Ok(measured) => row["boundary"] = measured,
                    Err(error) => failure = Some(error.to_string()),
                }
            }
            Err(error) => failure = Some(error.to_string()),
        }
        row["snapshot"] = serde_json::to_value(simulation.snapshot())?;
        trace.push(row);
        if failure.is_some() {
            break;
        }
    }
    let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let names = [
        "crates/dev_tools/src/bin/goose_native_motor_isolation_probe.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "third_party/rapier3d/src/dynamics/joint/motor_model.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/unit_multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
        "Cargo.lock",
    ];
    let hashes = names
        .into_iter()
        .map(|name| fs::read(project.join(name)).map(|b| (name.to_string(), json!(digest(&b)))))
        .collect::<Result<serde_json::Map<String, serde_json::Value>, _>>()?;
    let receipt = json!({"schema":"goose_native_motor_isolation_receipt_v1",
        "model_sha256":plant.model_sha256,"contract_sha256":plant.derived_contract_sha256,
        "plant_sha256":plant_hash,"input_sha256":digest(&bytes),"code_sha256":hashes,
        "executable_sha256":digest(&fs::read(env::current_exe()?)?),"configuration":simulation.configuration(),
        "world_gravity_engine":simulation.world.gravity.to_array(),"floor_present":false,
        "joint_friction_disabled_for_diagnosis":input.disable_joint_friction_for_diagnosis,
        "actual_integrals":simulation.snapshot().integration_count,"actual_torque_updates":simulation.snapshot().torque_update_count,
        "inferences":0,"optimizer_updates":0,"post_birth_pose_writes":0,"qualified":false,
        "scope":"native position-drive isolation; no Actor/task driver, 17 raw same position targets, true rotor torque0; no policy/driver qualification",
        "initial":initial,"initial_snapshot":initial_snapshot,"failure":failure,"trace":trace});
    fs::write(&paths[3], serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "{}",
        json!({"output":paths[3],"actual_integrals":receipt["actual_integrals"],"failure":failure})
    );
    Ok(())
}
