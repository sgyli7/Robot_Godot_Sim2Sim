//! Bounded Goose Rapier evidence. No policy inference or M0 promotion is implied.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use rapier3d::prelude::*;
use robot_minigame::goose::plant::GoosePlant;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{PhysicsClockProfile, SimulationWorld, goose::builder::GooseAssembly};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Goose M0 probe: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    let option = |name: &str| {
        arguments
            .iter()
            .position(|value| value == name)
            .and_then(|index| arguments.get(index + 1))
    };
    let plant_path = PathBuf::from(option("--plant").ok_or("--plant is required")?);
    let output = PathBuf::from(option("--output").ok_or("--output is required")?);
    if output.exists() {
        return Err("output receipt already exists; use a fresh experiment path".into());
    }
    let ticks: usize = option("--ticks")
        .map(String::as_str)
        .unwrap_or("100")
        .parse()?;
    let resets: usize = option("--cold-resets")
        .map(String::as_str)
        .unwrap_or("20")
        .parse()?;
    let pgs: usize = option("--pgs").map(String::as_str).unwrap_or("4").parse()?;
    if ticks == 0 || ticks > 100 || resets == 0 || resets > 20 || ![4, 8, 16, 32].contains(&pgs) {
        return Err("bounded M0 requires ticks1..100, cold-resets1..20, PGS4/8/16/32".into());
    }
    let (plant, plant_sha256) = GoosePlant::read(&plant_path)?;
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let code_paths = [
        "crates/modules/robot/src/goose/plant.rs",
        "crates/modules/robot/src/goose/contract.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/modules/simulation/src/fixed_step_runtime.rs",
        "crates/dev_tools/src/bin/goose_m0_probe.rs",
        "Cargo.lock",
    ];
    let code_hashes: serde_json::Map<String, Value> = code_paths
        .into_iter()
        .map(|path| {
            fs::read(project.join(path)).map(|bytes| {
                (
                    path.to_owned(),
                    json!(format!("{:x}", Sha256::digest(bytes))),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let executable = env::current_exe()?;
    let executable_sha256 = format!("{:x}", Sha256::digest(fs::read(&executable)?));
    let mut receipt = json!({"schema":"goose_m0_target_receipt_v1","engine":"rapier","engine_version":"0.35.3",
    "candidate":plant.candidate_id,"contract_sha256":plant.derived_contract_sha256,"model_sha256":plant.model_sha256,
    "neutral_contract_sha256":plant.neutral_contract_sha256,"plant_sha256":plant_sha256,
    "timing":{"physics_dt_s":0.02,"torque_dt_s":0.02,"policy_dt_s":0.02,"substeps":1},
    "runtime_code_sha256":code_hashes,"qualified":false,"target_complete":false,"optimizer_updates":0,"inference_count":0,
    "executable":executable,"executable_sha256":executable_sha256,
    "debug_assertions":cfg!(debug_assertions),"target_arch":std::env::consts::ARCH,
    "actuation_mode":"native prescribed effort: beak input rotor0.24Nm, other motors0; no learned actor",
    "cold_reset_initial_root_lift_m":0.002,
    "checks":{
        "mass_com_full_inertia":{"status":"not_checked"},"cold_resets_single_integrations":{"status":"not_checked"},
        "named_axes":{"status":"not_checked"},"jaw_constraints":{"status":"not_checked"},
        "pad_load_curve":{"status":"not_checked"},"energy_growth":{"status":"not_checked"},
        "penetration_escape":{"status":"not_checked"},"foot_release_dissipation":{"status":"not_checked"},
        "foot_impact":{"status":"not_checked"},"collision_proxy":{"status":"not_checked"},
        "ground_reach_payload":{"status":"not_checked"},"source_target_pad_comparison":{"status":"not_checked"},
        "controller_gravity_feedforward":{"status":"not_checked"},
        "same_world_reset_cleanup":{"status":"not_checked"}
    }});
    let mut trace = Vec::new();
    let mut times = Vec::new();
    let mut failures = Vec::new();
    for reset in 0..resets {
        let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
        simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = pgs;
        let assembly = match GooseAssembly::build(&mut simulation, &plant) {
            Ok(assembly) => assembly,
            Err(error) => {
                failures.push(error.to_string());
                break;
            }
        };
        if reset == 0 {
            receipt["configuration"] = serde_json::to_value(simulation.configuration())?;
            receipt["body_measurements"] = serde_json::to_value(&assembly.body_measurements)?;
            receipt["native_robot_body_count"] = json!(assembly.body_handles.len());
            receipt["native_robot_collider_count"] = json!(assembly.collider_count);
            receipt["source_geometry_count"] = json!(assembly.source_geometry_count);
            receipt["native_collider_source_groups"] = json!(assembly.collider_source_groups);
            receipt["geometry_representation"] = json!(
                "unchanged convex hulls packed by rigid body, mask and friction; no convex filling of hollow structures"
            );
            receipt["native_passive_spring_count"] = json!(assembly.passive_spring_count);
            receipt["checks"]["mass_com_full_inertia"] = json!({"status":"passed",
                "scope":"native per-body mass, COM, reconstructed full tensor at initialization; no dynamic equivalence"});
        }
        let floor = simulation.world.bodies.insert(
            RigidBodyBuilder::fixed()
                .translation(Vector::new(0.0, -0.1, 0.0))
                .additional_solver_iterations(0),
        );
        simulation.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(2.0, 0.1, 2.0)
                .density(0.0)
                .friction(0.65)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_1,
                    Group::GROUP_2,
                    InteractionTestMode::Or,
                )),
            floor,
            &mut simulation.world.bodies,
        );
        let reset_started = Instant::now();
        let mut completed = 0;
        let mut reason = None;
        let mut maximum_pin_error = 0.0_f32;
        for tick in 0..ticks {
            let started = Instant::now();
            let mut torque = [0.0; 18];
            torque[5] = if tick < 25 { 0.24 } else { 0.0 };
            let snapshot = match assembly.step(&mut simulation, torque) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    reason = Some(error.to_string());
                    break;
                }
            };
            let physics_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            times.push(physics_wall_ms);
            completed += 1;
            let state = assembly.state(&simulation)?;
            let pin_error = assembly.jaw_pin_error_m(&simulation);
            maximum_pin_error = maximum_pin_error.max(pin_error);
            trace.push(json!({"reset":reset,"tick":tick,"snapshot":snapshot,"q_rad":state.joint_position_rad,
                "qd_rad_s":state.joint_velocity_rad_s,"jaw_pin_error_m":pin_error,
                "physics_wall_ms":physics_wall_ms}));
            if !pin_error.is_finite()
                || state
                    .joint_position_rad
                    .iter()
                    .chain(&state.joint_velocity_rad_s)
                    .any(|value| !value.is_finite())
            {
                reason = Some("nonfinite_native_joint_state".into());
                break;
            }
            if reset_started.elapsed().as_secs_f64() >= 5.0 {
                reason = Some("bounded_probe_wall_budget_5s_per_reset".into());
                break;
            }
        }
        if let Some(reason) = &reason {
            failures.push(reason.clone());
        }
        receipt["resets"].as_array_mut().map(|rows| {
            rows.push(json!({"reset":reset,"ticks_completed":completed,
            "failure_reason":reason,"max_jaw_pin_error_m":maximum_pin_error}))
        });
        if !receipt["resets"].is_array() {
            receipt["resets"] = json!([{"reset":reset,"ticks_completed":completed,
            "failure_reason":reason,"max_jaw_pin_error_m":maximum_pin_error}]);
        }
    }
    times.sort_by(f64::total_cmp);
    let actual_resets = receipt["resets"].as_array().map_or(0, Vec::len);
    let finished = failures.is_empty() && actual_resets == resets && trace.len() == resets * ticks;
    receipt["checks"]["cold_resets_single_integrations"] = json!({"status":if finished && resets==20 && ticks==100 {"passed"} else if finished {"partial"} else {"failed"},
        "measurement_passed":finished,"scope":"fresh_world_reinitialization; same-world scene retention/handle cleanup not tested",
        "reset_count":actual_resets,"requested_resets":resets,"requested_ticks_per_reset":ticks,
        "completed_integrations":trace.len(),"failures":failures,
        "qualification_limits":"prescribed effort diagnostic; energy, penetration, actual controller, full M0 not qualified"});
    receipt["physics_only_p95_ms"] = if times.is_empty() {
        Value::Null
    } else {
        json!(times[(times.len() * 95 / 100).min(times.len() - 1)])
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let trace_path = output.with_file_name(format!(
        "{}_trace.json",
        output
            .file_stem()
            .ok_or("invalid output name")?
            .to_string_lossy()
    ));
    let trace_bytes = serde_json::to_vec(&json!({"schema":"goose_target_trace_v1","rows":trace}))?;
    receipt["trace_path"] = json!(trace_path);
    receipt["trace_sha256"] = json!(format!("{:x}", Sha256::digest(&trace_bytes)));
    fs::write(&trace_path, trace_bytes)?;
    fs::write(&output, serde_json::to_vec_pretty(&receipt)?)?;
    println!("STATUS M0_UNQUALIFIED {}", output.display());
    Ok(())
}
