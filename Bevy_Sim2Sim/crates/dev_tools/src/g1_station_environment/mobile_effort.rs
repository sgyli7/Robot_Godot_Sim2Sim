//! Frozen recorded commands replayed in a cold owner for read-only efforts.
//! Exact physical-prefix equality is mandatory before using new measurements.

use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::{contract::G1Command, policy::bound_bytes};
use serde::Deserialize;
use serde_json::{Value, json};
use simulation_minigame::g1::{
    runner::G1Runner,
    task_policy::{ArenaControllerCommand, controller_command},
    task_runner::{ArenaTaskBodyConfig, ArenaTaskRunnerConfig},
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use task_minigame::{policy::PolicyActionChunk, types::TaskProfile};

mod foreaft;
mod insertion;
mod retention;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema: String,
    config: Value,
    chunks: Vec<PolicyActionChunk>,
    trace: PathBuf,
    trace_sha256: String,
    window_start_tick: u64,
    stop_tick: u64,
}

fn read(path: &Path, hash: &str, maximum: usize) -> Result<Vec<u8>, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > maximum as u64 {
        return Err("recorded motor fixture exceeds finite input budget".into());
    }
    bound_bytes(path, hash).map_err(|e| e.to_string())
}

fn remove_point_diagnostics(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("diagnostic_solver_contacts");
            for value in map.values_mut() {
                remove_point_diagnostics(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                remove_point_diagnostics(value);
            }
        }
        _ => {}
    }
}

fn journal_body_value<T: serde::Serialize>(body: &T) -> Result<Value, String> {
    let value = serde_json::to_value(body).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

#[test]
fn expanded_f32_recorded_journal_comparison_uses_same_reader() {
    // Real first-Tick velocity from20708. The native journal first builds a
    // Value, expands f32 exactly, writes it, then the replay reads its bytes.
    let native: Value = serde_json::from_str("{\"joint_velocity\":-0.030285218730568886}").unwrap();
    let actual =
        journal_body_value(&json!({"joint_velocity":-0.030285218730568886_f64 as f32})).unwrap();
    assert_eq!(
        actual, native,
        "recorded f32 journal value falsely differs after one JSON reader pass"
    );
}

fn command(row: &Value) -> Result<G1Command, String> {
    let execution = row["execution"]
        .as_object()
        .ok_or("recorded execution absent")?;
    if execution.len() != 1 {
        return Err("recorded execution phase is not unique".into());
    }
    let (phase, record) = execution.iter().next().unwrap();
    let value = match phase.as_str() {
        "classical_grip_settle" | "classical_hold" => &record["holding"]["command"],
        "classical_raise" => &record["raising"]["command"],
        "classical_scan" | "classical_carry" => &record["command"],
        _ => return Err(format!("unsupported recorded mechanics phase {phase}")),
    };
    let command: G1Command = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    command.validate().map_err(|e| e.to_string())?;
    Ok(command)
}

#[test]
#[ignore = "hash-bound original1971Tick failure trace/chunks; <=2000 real CPU50Hz steps,0VLA/Qwen; late20raw motor Ticks and contact points; exact prefix required"]
fn recorded_station_grip_motor_effort_window() -> Result<(), String> {
    let fixture_path =
        PathBuf::from(std::env::var("G1_MOTOR_REPLAY_FIXTURE").map_err(|e| e.to_string())?);
    let fixture_hash =
        std::env::var("G1_MOTOR_REPLAY_FIXTURE_SHA256").map_err(|e| e.to_string())?;
    let fixture: Fixture = serde_json::from_slice(&read(&fixture_path, &fixture_hash, 1_048_576)?)
        .map_err(|e| e.to_string())?;
    if fixture.schema != "g1_recorded_station_motor_effort_fixture_v1"
        || !(200..=1980).contains(&fixture.window_start_tick)
        || fixture.stop_tick != fixture.window_start_tick + 30
        || fixture.stop_tick > 2000
        || fixture.chunks.len() != 4
    {
        return Err("recorded motor fixture changed bounded scope".into());
    }
    let trace_bytes = read(&fixture.trace, &fixture.trace_sha256, 64 * 1024 * 1024)?;
    let rows: Vec<Value> = trace_bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(serde_json::from_slice)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    if rows.len() < fixture.stop_tick as usize {
        return Err("recorded motor trace lacks frozen suffix".into());
    }
    let config: ArenaTaskRunnerConfig =
        serde_json::from_value(fixture.config["runner"].clone()).map_err(|e| e.to_string())?;
    let station: G1StationConfiguration =
        serde_json::from_value(fixture.config["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::MobileHomieV2(mut body) = config.body else {
        return Err("recorded effort requires original mobile body".into());
    };
    let mut commands = Vec::with_capacity(fixture.stop_tick as usize);
    for (index, chunk) in fixture.chunks.iter().enumerate() {
        if chunk.profile != TaskProfile::MobileBox
            || chunk.sequence_id != index as u64 + 1
            || chunk.frames.len() != 50
            || chunk.action_period_ns != 20_000_000
            || chunk.observation.episode_id != body.episode_id
            || chunk.observation.sim_time_ns != index as u64 * 1_000_000_000
        {
            return Err("recorded effort fixture changed source chunk identity or time".into());
        }
        for frame in &chunk.frames {
            let ArenaControllerCommand::MobileHomieV2(command) =
                controller_command(TaskProfile::MobileBox, frame, &config.limits)
                    .map_err(|e| format!("{e:?}"))?
                    .controller
            else {
                return Err("recorded effort decoded wrong controller".into());
            };
            commands.push(command);
        }
    }
    for (index, row) in rows.iter().take(fixture.stop_tick as usize).enumerate() {
        if row["episode_id"] != body.episode_id
            || row["owner_episode_integrations"] != index as u64 + 1
        {
            return Err("recorded effort trace reset, gap or foreign episode".into());
        }
        if index >= 200 {
            commands.push(command(row)?);
        }
    }
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, environment, preparation) = prepare(&scene, &station, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let output = PathBuf::from(std::env::var("G1_MOTOR_REPLAY_OUTPUT").map_err(|e| e.to_string())?);
    let mut evidence = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|e| e.to_string())?;
    let mut owner =
        G1Runner::load_mobile_station_fixture_diagnostic(&body).map_err(|e| e.to_string())?;
    let mut token = None;
    let mut exact = 0;
    writeln!(
        evidence,
        "{}",
        json!({"schema":"g1_recorded_motor_replay_start_v1", "preparation":preparation,
        "fixture_sha256":fixture_hash, "source_trace_sha256":fixture.trace_sha256,
        "new_VLA_Qwen_RGB":0, "new_WBC_and_physics_expected":fixture.stop_tick,
        "task_qualified":false, "real_time_qualified":false})
    )
    .map_err(|e| e.to_string())?;
    evidence.flush().map_err(|e| e.to_string())?;
    for (index, command) in commands.iter().enumerate() {
        let tick = index as u64 + 1;
        if index as u64 == fixture.window_start_tick {
            let started = owner
                .start_motor_effort_window()
                .map_err(|e| e.to_string())?;
            if owner.start_motor_effort_window().is_ok() {
                return Err("duplicate motor trace registration accepted".into());
            }
            if owner.motor_effort_window_snapshot(&started).is_ok() {
                return Err("unintegrated motor trace accepted".into());
            }
            token = Some(started);
        }
        let step = owner.step(command).map_err(|e| e.to_string())?;
        // The owner journal builds a serde_json::Value before writing JSON.
        // Direct f32 byte serialization uses shorter decimal numbers and would
        // report false differences against its exact expanded f32 values.
        let mut actual = journal_body_value(&step)?;
        let mut expected = rows[index]["body"]["mobile_homie_v2"].clone();
        remove_point_diagnostics(&mut actual);
        remove_point_diagnostics(&mut expected);
        if actual != expected {
            writeln!(
                evidence,
                "{}",
                json!({"event":"first_physical_prefix_difference", "tick":tick,
                "actual":actual, "expected":expected, "task_qualified":false})
            )
            .map_err(|e| e.to_string())?;
            evidence.flush().map_err(|e| e.to_string())?;
            return Err(format!(
                "instrumented headless physical prefix differs atTick{tick}; no causal attribution permitted"
            ));
        }
        exact += 1;
        if let Some(token) = &token {
            let counts = owner.progress_counts();
            let snapshot = owner
                .motor_effort_window_snapshot(token)
                .map_err(|e| e.to_string())?;
            let repeated = owner
                .motor_effort_window_snapshot(token)
                .map_err(|e| e.to_string())?;
            if snapshot != repeated || owner.progress_counts() != counts {
                return Err("read-only motor snapshot advanced owner or changed evidence".into());
            }
            writeln!(
                evidence,
                "{}",
                json!({"event":"current_motor_effort_and_contact_window", "tick":tick,
                "motors":snapshot, "task_objects":step.task_objects, "body_counts":counts,
                "exact_physical_prefix_rows":exact, "new_VLA_Qwen_RGB":0, "task_qualified":false})
            )
            .map_err(|e| e.to_string())?;
            evidence.flush().map_err(|e| e.to_string())?;
        }
    }
    writeln!(evidence, "{}", json!({"event":"bounded_motor_replay_complete", "actual_physics_body_torque_updates":exact,
        "exact_physical_prefix_rows":exact, "actual_motor_window_rows":30, "raw_motor_window_ticks":20,
        "fresh_VLA_Qwen_RGB":0, "task_success":false, "real_time_qualified":false})).map_err(|e| e.to_string())?;
    evidence.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}
