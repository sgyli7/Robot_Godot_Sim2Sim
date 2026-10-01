//! Offline oracle diagnostic only: original source targets, real native physics.
//! No production entry point, policy inference, pose playback or collision mask.

use super::*;
use robot_minigame::g1::policy::bound_bytes;
use serde_json::{Value, json};
use std::{env, fs::OpenOptions, io::Write, path::Path};

#[test]
#[ignore = "requires frozen source standing targets, original G1 and ORT assets"]
fn real_original_source_target_stand_replay() {
    let output = env::var("G1_AGILE_SOURCE_REPLAY_OUTPUT").unwrap();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    let mut report = json!({"schema":"g1_original_source_target_native_replay_v1",
        "qualified":false,"task_success":false,"oracle_diagnostic":true,
        "policy_inferences":0,"native_physics_hz":50,"native_control_hz":50,
        "integrations_per_tick":1,"source_code_commit":env::var("G1_CODE_COMMIT").ok(),
        "scope":"fixed recorded source controller targets through native motors; no policy feedback",
        "steps":[],"completed_standing_budget":false});
    let result = run(&mut report);
    report["error"] = result
        .as_ref()
        .err()
        .map_or(Value::Null, |e| json!(e.to_string()));
    report["completed_standing_budget"] = json!(result.is_ok());
    serde_json::to_writer_pretty(&mut file, &report).unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
    println!(
        "G1_SOURCE_TARGET_REPLAY ticks={} passed={} qualified=false",
        report["steps"].as_array().unwrap().len(),
        result.is_ok()
    );
    assert!(
        result.is_ok(),
        "native replay failed; evidence preserved: {result:?}"
    );
}

fn run(report: &mut Value) -> Result<(), RobotError> {
    let source_path = env::var("G1_AGILE_SOURCE_REPLAY").map_err(error)?;
    let source_sha = env::var("G1_AGILE_SOURCE_REPLAY_SHA256").map_err(error)?;
    let bytes = bound_bytes(Path::new(&source_path), &source_sha)?;
    if bytes.len() > 8_000_000 {
        return Err(error("source sequence exceeds bounded diagnostic size"));
    }
    let input: Value = serde_json::from_slice(&bytes).map_err(error)?;
    if input["schema"] != "g1_t1_source_standing_target_sequence_v1"
        || input["source_arena_commit"] != "8b4a3a47fc53de23e8205089d71109a2e2348acd"
        || input["usd_sha256"] != robot_minigame::g1::definition::USD_SHA256
        || input["agile_sha256"] != agile::MODEL_SHA256
        || input["source_physics_hz"] != 200
        || input["source_control_hz"] != 50
        || input["shelf_enabled"] != true
        || input["source_standing_budget_passed"] != true
        || input["joint_names"] != json!(JOINT_NAMES)
    {
        return Err(error("unmatched original source standing sequence"));
    }
    let rows = input["rows"]
        .as_array()
        .ok_or_else(|| error("missing source rows"))?;
    if rows.len() != 1500 {
        return Err(error(
            "source replay requires the frozen 1500 tick sequence",
        ));
    }
    let config_path = env::var("G1_AGILE_CONFIG").map_err(error)?;
    let config_sha = env::var("G1_AGILE_CONFIG_SHA256").map_err(error)?;
    let config: AgileRunnerConfig =
        serde_json::from_slice(&bound_bytes(Path::new(&config_path), &config_sha)?)
            .map_err(error)?;
    let initial_root: [f64; 3] =
        serde_json::from_value(input["initial_root_position_source"].clone()).map_err(error)?;
    let initial_q: Vec<f32> =
        serde_json::from_value(input["initial_joint_positions"].clone()).map_err(error)?;
    if config
        .task_objects
        .as_ref()
        .and_then(|scene| scene.source_t1_shelf.as_ref())
        .is_none()
        || initial_root
            .iter()
            .zip(config.root_pose.position)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 1e-6)
        || config.root_pose.rotation_wxyz != [1., 0., 0., 0.]
        || initial_q != config.default_positions
    {
        return Err(error("source/native initial shelf configuration changed"));
    }
    report["source_sequence_sha256"] = json!(source_sha);
    report["source_sequence_identity"] = json!({"trace_sha256":input["source_trace_sha256"],
        "receipt_sha256":input["source_receipt_sha256"],"source_head":input["source_head"]});
    report["config_sha256"] = json!(config_sha);
    report["original_config"] = json!(config);
    let mut runner = AgileRunner::load(&config)?;
    validate_clock(runner.configuration())?;
    report["initial_frame"] = json!(runner.initial_frame()?);
    report["initial_measurement"] = json!(runner.measurement()?);
    report["actuator_parameters"] = json!(runner.parameters.to_vec());
    report["step_configuration"] = json!(runner.configuration());
    for (index, row) in rows.iter().enumerate() {
        let tick = index as u64 + 1;
        if row["control_tick"] != tick {
            return Err(error("non-contiguous source target ticks"));
        }
        let targets: Vec<f32> = serde_json::from_value(row["targets"].clone()).map_err(error)?;
        let targets: [f32; JOINT_COUNT] = targets
            .try_into()
            .map_err(|_| error("source target length changed"))?;
        if !targets.iter().all(|x| x.is_finite()) {
            return Err(error("nonfinite source target"));
        }
        runner.check_source_positions(&runner.state()?.positions)?;
        runner.assembly.set_force_based_targets_with_parameters(
            &mut runner.simulation.world,
            &targets,
            &runner.parameters,
        )?;
        runner.motor_update_count += 1;
        let snapshot = runner.simulation.step_with_torques(&[]).map_err(error)?;
        if snapshot.integration_count != tick
            || snapshot.torque_update_count != tick
            || runner.motor_update_count != tick
            || runner.policy.inference_count() != 0
        {
            return Err(error("source replay native count mismatch"));
        }
        let measurement = runner.measurement()?;
        let root = &runner.simulation.world.bodies[runner.assembly.root_handle()];
        let position = engine_to_source_vector(root.translation().to_array());
        let upright = (*root.rotation() * Vector::Y).y;
        let guard = runner.check_source_positions(&measurement.joint_positions);
        let valid = guard.is_ok() && position[2] >= 0.35 && upright >= 0.5;
        report["steps"].as_array_mut().unwrap().push(json!({"tick":tick,
            "state_after_integration":measurement,"reference_joint_positions":row["joint_positions"],
            "source_targets":targets.to_vec(),"root_position_source":position,"upright":upright,
            "valid_for_runtime_admission":valid,"integration_count":snapshot.integration_count,
            "motor_update_count":runner.motor_update_count,"policy_inferences":0}));
        guard?;
        if !valid {
            return Err(error("source target replay fall guard"));
        }
    }
    report["final_counts"] = json!(runner.progress_counts());
    Ok(())
}
