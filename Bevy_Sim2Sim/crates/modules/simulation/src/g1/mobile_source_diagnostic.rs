//! Finite mechanical comparison of original T2 commands in the native owner.
//! This is an ignored development test, never a policy or runtime fallback.

use super::*;
use crate::g1::task_objects::TaskObjectKind;
use robot_minigame::g1::policy::bound_bytes;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSequence {
    schema: String,
    chunks: Vec<PolicyActionChunk>,
}

fn read_bound(name: &str) -> Result<Vec<u8>, RobotError> {
    let hash = std::env::var(format!("{name}_SHA256")).map_err(error)?;
    let path = std::env::var(name).map_err(error)?;
    let bytes = bound_bytes(Path::new(&path), &hash)?;
    if bytes.len() > 1_048_576 {
        return Err(error("mobile replay fixture exceeds bounded input"));
    }
    Ok(bytes)
}

/// Same 200 source commands, real Homie inference and one native step each.
/// Only an offline copy of the old wall stamp is rebound for queue admission;
/// original stamps and action bytes remain in the hash-bound input/receipt.
#[test]
#[ignore = "requires frozen original T2 chunks/assets; 200 real native steps and no VLA"]
fn real_mobile_source_grasp_window_diagnostic() -> Result<(), RobotError> {
    let config: ArenaTaskRunnerConfig =
        serde_json::from_slice(&read_bound("G1_MOBILE_REPLAY_CONFIG")?).map_err(error)?;
    let sequence: SourceSequence =
        serde_json::from_slice(&read_bound("G1_MOBILE_REPLAY_ACTIONS")?).map_err(error)?;
    if config.body.profile() != TaskProfile::MobileBox
        || sequence.schema != "g1_original_mobile_action_sequence_v1"
        || sequence.chunks.len() != 4
    {
        return Err(error(
            "mobile replay requires exactly four original T2 chunks",
        ));
    }
    for (index, chunk) in sequence.chunks.iter().enumerate() {
        ArenaTaskCommand {
            chunk: Arc::new(chunk.clone()),
            scheduled_start_sim_ns: None,
        }
        .validate()?;
        if chunk.profile != TaskProfile::MobileBox
            || chunk.observation.episode_id != config.body.episode_id()
            || chunk.observation.frame_id != index as u64 * 50
            || chunk.observation.sim_time_ns != index as u64 * 50 * ARENA_ACTION_PERIOD_NS
            || chunk.sequence_id != index as u64 + 1
        {
            return Err(error(
                "original T2 fixture profile/episode/timeline mismatch",
            ));
        }
        for frame in &chunk.frames {
            controller_command(TaskProfile::MobileBox, frame, &config.limits).map_err(error)?;
        }
    }
    let output = PathBuf::from(std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(error)?);
    let mut receipt_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(error)?;
    let mut trace = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("jsonl"))
        .map_err(error)?;
    let mut owner = ArenaTaskRunner::load_mobile_constraint_diagnostic(&config)?;
    let initial_frame = owner.initial_frame()?;
    let initial_objects = owner
        .task_object_frame()?
        .ok_or_else(|| error("mobile replay missing original task objects"))?;
    let initial_box = initial_objects
        .objects
        .iter()
        .find(|object| object.kind == TaskObjectKind::BrownBox)
        .ok_or_else(|| error("mobile replay missing original box"))?;
    let initial_box_height = initial_box.position_source[2];
    let mut maximum_box_height = initial_box_height;
    let mut minimum_upright = 1.0_f32;
    let mut maximum_held_lift_suffix = 0_u64;
    let mut held_lift_suffix = 0_u64;
    let mut hand_impulse_ticks = 0_u64;
    let mut shelf_impulse_ticks = 0_u64;
    let mut maximum_hand_impulse = 0.0_f32;
    let mut last_objects = initial_objects.clone();
    let mut rebound_observations = Vec::new();
    let start = Instant::now();
    let result = (|| -> Result<(), RobotError> {
        for (chunk_index, original) in sequence.chunks.iter().enumerate() {
            let mut admitted = original.clone();
            admitted.observation.captured_at_unix_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(error)?
                .as_millis()
                .try_into()
                .map_err(error)?;
            if admitted.frames != original.frames
                || admitted.observation.sim_time_ns != original.observation.sim_time_ns
                || admitted.observation.episode_id != original.observation.episode_id
            {
                return Err(error(
                    "offline wall admission changed physical commands/timeline",
                ));
            }
            rebound_observations.push(serde_json::json!({
                "original": original.observation, "offline_admission_copy": admitted.observation,
            }));
            let command = ArenaTaskCommand {
                chunk: Arc::new(admitted),
                scheduled_start_sim_ns: None,
            };
            for frame_index in 0..50 {
                let tick = chunk_index as u64 * 50 + frame_index as u64 + 1;
                let step = owner.step_with_guard(&command, &mut || Ok(()))?;
                let ArenaBodyStep::MobileHomieV2(body) = &step.body else {
                    return Err(error("mobile replay changed controller"));
                };
                if body.integration_count != tick
                    || body.torque_update_count != tick
                    || body.inference.inference_count != tick
                    || body.step_configuration.dt != 0.02
                    || body.step_configuration.num_solver_iterations != 1
                    || body.step_configuration.num_internal_pgs_iterations != 4
                    || step.execution.frame_index != frame_index
                    || step.execution.admitted_chunks != chunk_index as u64 + 1
                {
                    return Err(error("mobile replay changed timing/counts/solver budget"));
                }
                let objects = body
                    .task_objects
                    .as_ref()
                    .ok_or_else(|| error("mobile replay lost task owner"))?;
                let box_object = objects
                    .objects
                    .iter()
                    .find(|object| object.kind == TaskObjectKind::BrownBox)
                    .ok_or_else(|| error("mobile replay lost box"))?;
                let hand_impulse = box_object
                    .last_solve_contacts
                    .iter()
                    .filter(|contact| contact.other_robot_body_index.is_some())
                    .map(|contact| contact.normal_impulse_n_s)
                    .sum::<f32>();
                let shelf_impulse = box_object.last_solve_contacts.iter().any(|contact| {
                    contact.normal_impulse_n_s > 1e-6
                        && contact
                            .other_background_collider_path
                            .as_ref()
                            .is_some_and(|path| path.contains("/TaskAssets/shelf/"))
                });
                maximum_box_height = maximum_box_height.max(box_object.position_source[2]);
                minimum_upright = minimum_upright.min(body.root_upright_cosine);
                hand_impulse_ticks += u64::from(hand_impulse > 1e-6);
                shelf_impulse_ticks += u64::from(shelf_impulse);
                maximum_hand_impulse = maximum_hand_impulse.max(hand_impulse);
                let held_lift = box_object.position_source[2] - initial_box_height >= 0.05
                    && hand_impulse > 1e-6
                    && !shelf_impulse
                    && body.root_upright_cosine > 0.95;
                held_lift_suffix = if held_lift { held_lift_suffix + 1 } else { 0 };
                maximum_held_lift_suffix = maximum_held_lift_suffix.max(held_lift_suffix);
                last_objects = objects.clone();
                serde_json::to_writer(&mut trace, &serde_json::json!({
                    "diagnostic_saved_source_commands": true, "step": step,
                    "box_hand_impulse_n_s": hand_impulse, "box_shelf_support_impulse": shelf_impulse,
                    "held_lift_rule": held_lift, "held_lift_suffix_ticks": held_lift_suffix,
                })).map_err(error)?;
                writeln!(trace).map_err(error)?;
                if body.root_upright_cosine < 0.7 || body.root_position_source[2] < 0.35 {
                    return Err(error("mobile source replay crossed native upright guard"));
                }
            }
        }
        Ok(())
    })();
    trace.flush().map_err(error)?;
    let counts = owner.progress_counts();
    let reproduced_held_lift = result.is_ok() && maximum_held_lift_suffix >= 5;
    serde_json::to_writer_pretty(&mut receipt_file, &serde_json::json!({
        "schema": "g1_native_mobile_source_grasp_window_diagnostic_v1",
        "code_commit": std::env::var("G1_CODE_COMMIT").map_err(error)?,
        "config_sha256": std::env::var("G1_MOBILE_REPLAY_CONFIG_SHA256").map_err(error)?,
        "actions_sha256": std::env::var("G1_MOBILE_REPLAY_ACTIONS_SHA256").map_err(error)?,
        "original_observations_and_explicit_offline_wall_rebinding": rebound_observations,
        "original_action_values_unchanged": true, "original_sim_timeline_unchanged": true,
        "qualified": false, "task_success_verified": false, "autonomous_execution": false,
        "completed": result.is_ok(), "error": result.as_ref().err().map(ToString::to_string),
        "actual_vla_inferences": 0, "physical_tick_budget": 200,
        "physics_hz": 50, "integrations_per_tick": 1, "counts": counts,
        "free_running_wall_seconds": start.elapsed().as_secs_f64(), "one_x_qualified": false,
        "initial_robot_frame": initial_frame, "initial_objects": initial_objects,
        "last_objects": last_objects, "minimum_upright": minimum_upright,
        "maximum_box_rise_m": maximum_box_height - initial_box_height,
        "hand_impulse_ticks": hand_impulse_ticks, "shelf_impulse_ticks": shelf_impulse_ticks,
        "maximum_box_hand_impulse_n_s": maximum_hand_impulse,
        "held_lift_rule_frozen_before_run": {
            "box_rise_m_at_least": 0.05, "hand_impulse_n_s_above": 1e-6,
            "no_shelf_impulse_n_s_above": 1e-6, "upright_cosine_above": 0.95,
            "consecutive_ticks_at_least": 5,
        },
        "maximum_held_lift_suffix_ticks": maximum_held_lift_suffix,
        "source_grasp_window_reproduced": reproduced_held_lift,
        "scope": "fresh complete T2 native scene; four saved original source chunks; no images/model requests/pose writes/hold/extended actions; mechanical diagnosis only",
    })).map_err(error)?;
    writeln!(receipt_file).map_err(error)?;
    receipt_file.flush().map_err(error)?;
    result?;
    if !reproduced_held_lift {
        return Err(error(
            "same source commands did not reproduce held lift in native physics",
        ));
    }
    Ok(())
}
