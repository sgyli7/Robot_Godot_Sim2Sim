//! One unchanged saved chunk through the real 50 Hz native task owner.
//!
//! Offline action/transport diagnosis only: no current native RGB inference,
//! perception, Qwen decision or task-success qualification is manufactured.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use simulation_minigame::g1::{
    task_runner::{ArenaTaskCommand, ArenaTaskRunnerConfig},
    worker::{ArenaTaskWorker, G1WorkerPhase, TimedArenaTaskCommand},
};
use task_minigame::policy::{ARENA_ACTION_PERIOD_NS, PolicyActionChunk};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedSequence {
    schema: String,
    chunks: Vec<PolicyActionChunk>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SavedInput {
    Chunk(PolicyActionChunk),
    Sequence(SavedSequence),
}

fn bounded_read(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > max {
        return Err("diagnostic input exceeds byte bound".into());
    }
    fs::read(path).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !matches!(args.len(), 4 | 5 | 6)
        || args[3] != "--offline-diagnostic"
        || (args.len() == 5 && args[4] != "--predictive-limits")
        || (args.len() == 6 && (args[4] != "--constraint-sweeps" || args[5] != "16"))
    {
        return Err("usage: g1_task_chunk_probe CONFIG.json UNCHANGED_CHUNK.json NEW_RECEIPT.json --offline-diagnostic".into());
    }
    let output = Path::new(&args[2]);
    let mut receipt_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let trace_path = output.with_extension("jsonl");
    let mut trace = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&trace_path)
        .map_err(|e| e.to_string())?;
    let mut receipt = json!({
        "schema": "g1_native_task_chunk_probe_v1", "qualified": false,
        "scope": "unchanged saved action chunk; offline input; real background physical owner",
        "task_success_verified": false, "current_native_rgb_inference": false,
        "physics_hz": 50, "control_hz": 50, "integrations_per_tick": 1,
        "completed": false, "probe_passed": false,
    });
    let result = (|| -> Result<(), String> {
        let config_bytes = bounded_read(Path::new(&args[0]), 262_144)?;
        let chunk_bytes = bounded_read(Path::new(&args[1]), 1_048_576)?;
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
        let input: SavedInput = serde_json::from_slice(&chunk_bytes).map_err(|e| e.to_string())?;
        let chunks = match input {
            SavedInput::Chunk(chunk) => vec![chunk],
            SavedInput::Sequence(sequence) if sequence.schema == "g1_saved_action_sequence_v1" => {
                sequence.chunks
            }
            _ => return Err("unrecognized saved sequence schema".into()),
        };
        if chunks.is_empty() || chunks.len() > 8 {
            return Err("offline comparison requires 1..=8 original chunks".into());
        }
        let mut expected_ticks = 0;
        let mut expected_frames = Vec::new();
        let mut previous_sequence = 0;
        for chunk in &chunks {
            if chunk.observation.episode_id != config.body.episode_id()
                || chunk.profile != config.body.profile()
                || chunk.observation.sim_time_ns != expected_ticks * ARENA_ACTION_PERIOD_NS
                || chunk.sequence_id <= previous_sequence
            {
                return Err("original chunks must match the same world, profile and successive physical boundaries".into());
            }
            expected_frames.extend((0..chunk.frames.len()).map(|i| (chunk.sequence_id, i)));
            expected_ticks += chunk.frames.len() as u64;
            previous_sequence = chunk.sequence_id;
        }
        receipt["config_sha256"] = json!(format!("{:x}", Sha256::digest(&config_bytes)));
        receipt["chunk_sha256"] = json!(format!("{:x}", Sha256::digest(&chunk_bytes)));
        receipt["input_observations"] =
            json!(chunks.iter().map(|c| c.observation).collect::<Vec<_>>());
        receipt["expected_ticks"] = json!(expected_ticks);
        let owner = if args.len() == 5 {
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                let owner = ArenaTaskWorker::spawn_static_predictive_limit_diagnostic(config)
                    .map_err(|e| e.to_string())?;
                receipt["factory_verified_predictive_limit_joints"] = json!(43);
                receipt["predictive_free_root"] = json!(false);
                receipt["coordinate_projection"] = json!(false);
                owner
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            {
                return Err(
                    "predictive limit comparison requires its explicit development feature".into(),
                );
            }
        } else if args.len() == 6 {
            receipt["isolated_nonintegrating_constraint_sweeps"] = json!(16);
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                ArenaTaskWorker::spawn_static_constraint_diagnostic(config)
                    .map_err(|e| e.to_string())?
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            {
                return Err(
                    "constraint comparison requires its explicit development feature".into(),
                );
            }
        } else {
            ArenaTaskWorker::spawn(config).map_err(|e| e.to_string())?
        };
        let start = Instant::now();
        let deadline = start + Duration::from_secs(30);
        let mut submitted_chunks = 0;
        let mut next_boundary_tick = 0;
        let mut observed_frames = Vec::new();
        while Instant::now() < deadline {
            let Some(snapshot) = owner.take_latest() else {
                if owner.is_finished() {
                    return Err("task owner exited without final evidence".into());
                }
                thread::sleep(Duration::from_millis(1));
                continue;
            };
            receipt["last_phase"] = json!(format!("{:?}", snapshot.phase));
            receipt["last_reason"] = json!(snapshot.reason);
            receipt["integrations"] = json!(snapshot.timing.episode_integrations);
            receipt["torque_updates"] = json!(snapshot.timing.episode_torque_updates);
            receipt["body_inference_attempts"] = json!(snapshot.timing.episode_inference_attempts);
            receipt["body_successful_inferences"] =
                json!(snapshot.timing.episode_successful_inferences);
            receipt["pending_ticks"] = json!(snapshot.timing.pending_ticks);
            receipt["active_wall_seconds"] = json!(snapshot.timing.episode_active_wall_seconds);
            receipt["active_sim_seconds"] = json!(snapshot.timing.episode_active_sim_seconds);
            receipt["display_updates_dropped"] = json!(snapshot.timing.display_updates_dropped);
            if snapshot.phase == G1WorkerPhase::Failed {
                return Err(snapshot
                    .reason
                    .clone()
                    .unwrap_or("task owner failed".into()));
            }
            if let Some(step) = &snapshot.step {
                let index = (step.execution.sequence_id, step.execution.frame_index);
                if observed_frames.last() != Some(&index) {
                    observed_frames.push(index);
                    serde_json::to_writer(&mut trace, step.as_ref()).map_err(|e| e.to_string())?;
                    writeln!(trace).map_err(|e| e.to_string())?;
                    trace.flush().map_err(|e| e.to_string())?;
                }
            }
            if snapshot.phase == G1WorkerPhase::Paused
                && snapshot.timing.episode_integrations == next_boundary_tick
            {
                if submitted_chunks < chunks.len() {
                    let chunk = &chunks[submitted_chunks];
                    next_boundary_tick += chunk.frames.len() as u64;
                    owner
                        .submit(TimedArenaTaskCommand {
                            episode_id: chunk.observation.episode_id,
                            valid_until_sim_ns: next_boundary_tick * ARENA_ACTION_PERIOD_NS,
                            valid_until_wall: Instant::now() + Duration::from_secs(5),
                            command: ArenaTaskCommand {
                                chunk: Arc::new(chunk.clone()),
                            },
                        })
                        .map_err(|e| e.to_string())?;
                    submitted_chunks += 1;
                    continue;
                }
                if snapshot.timing.episode_integrations != expected_ticks
                    || snapshot.timing.episode_torque_updates != expected_ticks
                    || snapshot.timing.episode_successful_inferences != expected_ticks
                    || observed_frames != expected_frames
                {
                    return Err(
                        "native chunk missed/restarted frames or changed physical update counts"
                            .into(),
                    );
                }
                receipt["observed_action_frames"] = json!(observed_frames);
                receipt["paused_at_original_action_expiry"] = json!(true);
                receipt["elapsed_wall_seconds"] = json!(start.elapsed().as_secs_f64());
                owner.shutdown().map_err(|e| e.to_string())?;
                receipt["owned_thread_closed"] = json!(true);
                return Ok(());
            }
        }
        Err("bounded native chunk probe timed out".into())
    })();
    receipt["completed"] = json!(true);
    receipt["probe_passed"] = json!(result.is_ok());
    receipt["error"] = json!(result.as_ref().err());
    receipt["trace_sha256"] = json!(format!(
        "{:x}",
        Sha256::digest(fs::read(&trace_path).map_err(|e| e.to_string())?)
    ));
    serde_json::to_writer_pretty(&mut receipt_file, &receipt).map_err(|e| e.to_string())?;
    writeln!(receipt_file).map_err(|e| e.to_string())?;
    println!(
        "STATUS: {}; receipt={}",
        if result.is_ok() { "success" } else { "failed" },
        output.display()
    );
    result
}
