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

/// A different causal question from the VLA rollout: after the same proven
/// source grasp, can the body keep its last upper targets during a fixed turn,
/// clear-aisle walk and stop? Saved commands are diagnostic fixtures, not live
/// decisions. No object truth changes any command or physical pose.
#[test]
#[ignore = "requires four frozen source chunks/assets; up to1050 real native steps,0 VLA"]
fn real_mobile_fixed_grip_body_carry_diagnostic() -> Result<(), RobotError> {
    let config: ArenaTaskRunnerConfig =
        serde_json::from_slice(&read_bound("G1_MOBILE_REPLAY_CONFIG")?).map_err(error)?;
    let sequence: SourceSequence =
        serde_json::from_slice(&read_bound("G1_MOBILE_REPLAY_ACTIONS")?).map_err(error)?;
    if config.body.profile() != TaskProfile::MobileBox
        || !matches!(
            sequence.schema.as_str(),
            "g1_original_mobile_action_sequence_v1" | "g1_saved_native_mobile_action_sequence_v1"
        )
        || sequence.chunks.len() != 4
    {
        return Err(error(
            "fixed grip body diagnostic requires four matched source chunks",
        ));
    }
    for (i, chunk) in sequence.chunks.iter().enumerate() {
        ArenaTaskCommand {
            chunk: Arc::new(chunk.clone()),
            scheduled_start_sim_ns: None,
        }
        .validate()?;
        if chunk.observation.episode_id != config.body.episode_id()
            || chunk.observation.frame_id
                != if sequence.schema == "g1_original_mobile_action_sequence_v1" {
                    i as u64 * 50
                } else {
                    i as u64 + 1
                }
            || chunk.observation.sim_time_ns != i as u64 * 1_000_000_000
            || chunk.sequence_id != i as u64 + 1
        {
            return Err(error("fixed grip source fixture timeline mismatch"));
        }
        for frame in &chunk.frames {
            controller_command(TaskProfile::MobileBox, frame, &config.limits).map_err(error)?;
        }
    }
    let navigation_mode = std::env::var("G1_MOBILE_FIXED_GRIP_NAVIGATION").unwrap_or_default();
    let straight_back = navigation_mode == "proprioceptive_straight_back_2m";
    let feedback_navigation = match navigation_mode.as_str() {
        "proprioceptive_heading_odometry_2m" | "proprioceptive_straight_back_2m" => true,
        "" => false,
        _ => return Err(error("unsupported fixed grip navigation comparison")),
    };
    let tick_budget = if feedback_navigation { 2050 } else { 1050 };
    let mut navigation_stage = "turn";
    let mut navigation_stage_ticks = 0_u64;
    let mut heading_ready_ticks = 0_u64;
    let mut own_velocity_odometry = [0_f32; 2];
    let mut feedback_completed = false;
    let mut held_heading = None;
    let last_frame = sequence.chunks.last().unwrap().frames.last().unwrap();
    let admitted =
        controller_command(TaskProfile::MobileBox, last_frame, &config.limits).map_err(error)?;
    let ArenaControllerCommand::MobileHomieV2(mut held_command) = admitted.controller else {
        return Err(error("fixed grip body profile mismatch"));
    };
    let output = PathBuf::from(std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(error)?);
    let mut file = fs::OpenOptions::new()
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
    let initial = owner
        .task_object_frame()?
        .ok_or_else(|| error("missing task scene"))?;
    let initial_box = initial
        .objects
        .iter()
        .find(|o| o.kind == TaskObjectKind::BrownBox)
        .unwrap()
        .position_source;
    let mut final_position = initial_box;
    let mut held_suffix = 0_u64;
    let mut minimum_upright = 1_f32;
    let mut root_walk_start = None;
    let mut maximum_walk_displacement = 0_f32;
    let mut maximum_box_displacement = 0_f32;
    let mut actual_ticks = 0_u64;
    let start = Instant::now();
    let mut current_command = None;
    let result = (|| -> Result<(), RobotError> {
        for tick in 1..=tick_budget {
            let (phase, body) = if tick <= 200 {
                if (tick - 1) % 50 == 0 {
                    let index = ((tick - 1) / 50) as usize;
                    let mut chunk = sequence.chunks[index].clone();
                    chunk.observation.captured_at_unix_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_err(error)?
                        .as_millis()
                        .try_into()
                        .map_err(error)?;
                    current_command = Some(ArenaTaskCommand {
                        chunk: Arc::new(chunk),
                        scheduled_start_sim_ns: None,
                    });
                }
                let step =
                    owner.step_with_guard(current_command.as_ref().unwrap(), &mut || Ok(()))?;
                let ArenaBodyStep::MobileHomieV2(body) = step.body else {
                    return Err(error("foreign body"));
                };
                ("saved_grasp_fixture", body)
            } else {
                let ArenaBodyRunner::MobileHomieV2(body) = &mut owner.body else {
                    return Err(error("fixed grip lost mobile owner"));
                };
                let phase = if feedback_navigation {
                    let own_state = body.measurement()?;
                    let [w, x, y, z] = own_state.root_rotation_wxyz;
                    let heading = (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z));
                    let target_heading = if straight_back {
                        *held_heading.get_or_insert(heading)
                    } else {
                        std::f32::consts::FRAC_PI_2
                    };
                    let heading_error = (target_heading - heading)
                        .sin()
                        .atan2((target_heading - heading).cos());
                    navigation_stage_ticks += 1;
                    match navigation_stage {
                        "turn" => {
                            heading_ready_ticks = if heading_error.abs() < 0.06 {
                                heading_ready_ticks + 1
                            } else {
                                0
                            };
                            if heading_ready_ticks >= 20 {
                                navigation_stage = "walk";
                                navigation_stage_ticks = 0;
                            } else if navigation_stage_ticks >= 750 {
                                return Err(error("proprioceptive turn deadline missed"));
                            }
                            held_command.navigation =
                                [0., 0., (0.8 * heading_error).clamp(-0.4, 0.4)];
                            "feedback_fixed_grip_turn"
                        }
                        "walk" => {
                            // Self-state velocity only. Neither root/object scene poses nor
                            // contacts/acceptance values choose a navigation command.
                            for k in 0..2 {
                                own_velocity_odometry[k] +=
                                    own_state.root_velocity_source[k] * 0.02;
                            }
                            let distance = own_velocity_odometry[0].hypot(own_velocity_odometry[1]);
                            if distance >= 2. {
                                navigation_stage = "stop";
                                navigation_stage_ticks = 0;
                                held_command.navigation = [0.; 3];
                            } else if navigation_stage_ticks >= 1000 {
                                return Err(error("proprioceptive two-meter walk deadline missed"));
                            } else {
                                held_command.navigation = [
                                    if straight_back { -0.3 } else { 0.3 },
                                    0.,
                                    (0.8 * heading_error).clamp(-0.25, 0.25),
                                ];
                            }
                            "feedback_fixed_grip_walk"
                        }
                        "stop" => {
                            held_command.navigation = [0.; 3];
                            feedback_completed = navigation_stage_ticks >= 100;
                            "feedback_fixed_grip_stop"
                        }
                        _ => return Err(error("unknown navigation stage")),
                    }
                } else {
                    let phase = if tick <= 600 {
                        "fixed_grip_turn"
                    } else if tick <= 950 {
                        "fixed_grip_walk"
                    } else {
                        "fixed_grip_stop"
                    };
                    held_command.navigation = match phase {
                        "fixed_grip_turn" => [0., 0., 0.4],
                        "fixed_grip_walk" => [0.3, 0., 0.],
                        _ => [0.; 3],
                    };
                    phase
                };
                (phase, Box::new(body.step(&held_command)?))
            };
            actual_ticks = tick;
            if body.integration_count != tick
                || body.torque_update_count != tick
                || body.inference.inference_count != tick
                || body.step_configuration.dt != 0.02
                || body.step_configuration.physics_hz != 50
                || body.step_configuration.num_solver_iterations != 1
                || body.step_configuration.num_internal_pgs_iterations != 4
                || body.step_configuration.max_ccd_substeps != 1
            {
                return Err(error("fixed grip physical counts/config changed"));
            }
            let object = body
                .task_objects
                .as_ref()
                .unwrap()
                .objects
                .iter()
                .find(|o| o.kind == TaskObjectKind::BrownBox)
                .unwrap();
            final_position = object.position_source;
            minimum_upright = minimum_upright.min(body.root_upright_cosine);
            let hand_impulse = object
                .last_solve_contacts
                .iter()
                .filter(|c| c.other_robot_body_index.is_some())
                .map(|c| c.normal_impulse_n_s)
                .sum::<f32>();
            let other_support = object
                .last_solve_contacts
                .iter()
                .any(|c| c.other_robot_body_index.is_none() && c.normal_impulse_n_s > 1e-6);
            let held = hand_impulse > 1e-6 && !other_support && body.root_upright_cosine > 0.95;
            if tick > 200 {
                held_suffix = if held { held_suffix + 1 } else { 0 };
            }
            if phase.ends_with("fixed_grip_walk") {
                let origin = root_walk_start.get_or_insert(body.root_position_source);
                maximum_walk_displacement = maximum_walk_displacement.max(
                    ((body.root_position_source[0] - origin[0]).powi(2)
                        + (body.root_position_source[1] - origin[1]).powi(2))
                    .sqrt(),
                );
            }
            maximum_box_displacement = maximum_box_displacement.max(
                object
                    .position_source
                    .iter()
                    .zip(initial_box)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt(),
            );
            serde_json::to_writer(
                &mut trace,
                &serde_json::json!({
                    "phase":phase,"diagnostic_saved_grasp_fixture":true,"grasp_fixture_schema":sequence.schema,
                    "held_rule":held,"held_suffix_ticks":held_suffix,
                    "manual_body_command":if tick > 200 {Some(&held_command)} else {None},
                "own_velocity_odometry_xy_m":own_velocity_odometry,
                    "body":{"mobile_homie_v2":body},
                }),
            )
            .map_err(error)?;
            writeln!(trace).map_err(error)?;
            // Independent test abortion only, never a hidden recovery controller.
            if minimum_upright < 0.95 || final_position[2] < initial_box[2] - 0.2 {
                return Err(error("fixed grip diagnostic aborted after fall/drop"));
            }
            if feedback_completed {
                break;
            }
            if tick == 200 && (!held || final_position[2] < initial_box[2] + 0.05) {
                return Err(error(
                    "fixed grip precondition: source grasp not reproduced",
                ));
            }
        }
        Ok(())
    })();
    trace.flush().map_err(error)?;
    let passed = result.is_ok()
        && maximum_walk_displacement >= 2.
        && if feedback_navigation {
            feedback_completed && held_suffix >= 350
        } else {
            held_suffix == 850
        };
    serde_json::to_writer_pretty(&mut file,&serde_json::json!({
        "schema":"g1_fixed_grip_body_carry_diagnostic_v1","qualified":false,
        "autonomous_execution":false,"task_success_verified":false,"one_x_qualified":false,
        "actual_vla_inferences":0,"code_commit":std::env::var("G1_CODE_COMMIT").map_err(error)?,
        "source_actions_sha256":std::env::var("G1_MOBILE_REPLAY_ACTIONS_SHA256").map_err(error)?,
        "config_sha256":std::env::var("G1_MOBILE_REPLAY_CONFIG_SHA256").map_err(error)?,
        "original_upper_targets_unchanged_after_grasp":true,"no_runtime_pose_or_object_writes":true,
        "feedback_navigation":feedback_navigation,"navigation_mode":navigation_mode,"maximum_physical_tick_budget":tick_budget,
        "own_velocity_odometry_xy_m":own_velocity_odometry,"feedback_completed":feedback_completed,
        "fixed_phases_ticks":{"saved_source_grasp":200,"turn":if feedback_navigation {750} else {400},"walk":if feedback_navigation {1000} else {350},"stop":100},
        "fixed_commands":{"turn_max_yaw_rate":0.4,"walk_forward_speed":0.3,"walk_max_yaw_rate":0.25,"stop":[0.,0.,0.]},
        "physics_hz":50,"integrations_per_tick":1,"actual_ticks":actual_ticks,
        "counts":owner.progress_counts(),"completed":result.is_ok(),
        "error":result.as_ref().err().map(ToString::to_string),"minimum_upright":minimum_upright,
        "maximum_box_displacement_m":maximum_box_displacement,"final_box_position_source":final_position,
        "maximum_root_walk_phase_displacement_m":maximum_walk_displacement,
        "final_manual_phase_held_suffix_ticks":held_suffix,"fixed_grip_2m_walk_passed":passed,
        "wall_seconds":start.elapsed().as_secs_f64(),
        "scope":"one finite source-grasp/manual-body mechanical comparison; no RGB/VLA/Qwen/task release qualification",
    })).map_err(error)?;
    writeln!(file).map_err(error)?;
    file.flush().map_err(error)?;
    result?;
    if !passed {
        return Err(error(
            "fixed grip body test did not preserve grasp over2m walk/stop",
        ));
    }
    Ok(())
}
