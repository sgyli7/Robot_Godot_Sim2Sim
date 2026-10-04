//! New actual RGB after the independently measured native station grip hold.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_navigation::{MobileCarryGoal, MobileScanGoal},
    worker::TimedCommand,
};

pub(super) fn carry_boundary_completed(
    execution: &MobileAssistExecution,
    submitted: Option<ObservationStamp>,
) -> bool {
    // A queued fine carry may leave its completed predecessor visible until
    // the unique owner consumes the command. Await this goal's own receipt.
    matches!(execution, MobileAssistExecution::ClassicalCarry { goal, navigation, .. }
        if navigation.completed && submitted.is_none_or(|stamp| goal.observation == stamp))
}

fn completed_hold_ticks(
    execution: &MobileAssistExecution,
    tick: u64,
) -> Result<Option<u32>, String> {
    // Submission is asynchronous: until the owner consumes the hold command,
    // its paused snapshot still describes the completed fourth grasp chunk.
    // Only that exact predecessor may wait; other receipts remain errors.
    if tick == 200
        && matches!(execution, MobileAssistExecution::OriginalVla(v)
            if v.profile == TaskProfile::MobileBox
                && v.sequence_id == 4
                && v.frame_index == 49
                && v.admitted_chunks == 4
                && v.execution_start_sim_ns == 3_000_000_000
                && v.observation.sim_time_ns == 3_000_000_000)
    {
        return Ok(None);
    }
    let MobileAssistExecution::ClassicalGripSettle { holding, .. } = execution else {
        return Err("station carry handoff lost its completed hold".into());
    };
    if !holding.completed
        || !(100..=250).contains(&holding.holding_ticks)
        || !(300..=450).contains(&tick)
    {
        return Err("station carry requires completed100..250Tick self-state hold".into());
    }
    Ok(Some(holding.holding_ticks))
}

pub(super) fn continue_after_hold(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("station hold owner absent")?
        .clone();
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let Some(step) = &latest.assist_step else {
        return Err("station hold has no native execution receipt".into());
    };
    let Some(holding_ticks) =
        completed_hold_ticks(&step.execution, latest.timing.episode_integrations)?
    else {
        return Ok(false);
    };
    let config = &runtime
        .mobile_assist
        .as_ref()
        .ok_or("station carry configuration absent")?
        .configuration;
    let (heading, distance) = match config {
        MobileAssistStage::Carry(config) => (
            config.heading_yaw_source_rad,
            Some(config.relative_distance_m),
        ),
        MobileAssistStage::Scan(config) => (config.heading_yaw_source_rad, None),
    };
    let Some((observation, _)) =
        capture_current_marker_frame(runtime, port, "station_held_transport")?
    else {
        return Ok(false);
    };
    let (command, goal, receipt_key, origin, limit_ns, wall_seconds) = match distance {
        Some(relative_distance_m) => {
            let goal = MobileCarryGoal {
                observation,
                heading_yaw_source_rad: heading,
                relative_distance_m,
            };
            goal.validate().map_err(|e| e.to_string())?;
            (
                MobileAssistCommand::ClassicalCarry(goal.clone()),
                serde_json::to_value(goal).map_err(|e| e.to_string())?,
                "station_public_aisle_carry_after_hold",
                "explicit_public_court_clear_aisle_not_target_bin_localization",
                46_000_000_000,
                40,
            )
        }
        None => {
            let goal = MobileScanGoal {
                observation,
                heading_yaw_source_rad: heading,
            };
            goal.validate().map_err(|e| e.to_string())?;
            (
                MobileAssistCommand::ClassicalScan(goal.clone()),
                serde_json::to_value(goal).map_err(|e| e.to_string())?,
                "station_public_bin_scan_after_hold",
                "explicit_public_map_search_heading_target_pose_requires_new_actual_rgb",
                21_000_000_000,
                20,
            )
        }
    };
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("station carry lost the unique assisted physics owner".into());
    };
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: limit_ns,
            valid_until_wall: Instant::now() + Duration::from_secs(wall_seconds),
            command,
        })
        .map_err(|e| e.to_string())?;
    let mut receipt = outcome.0.lock().unwrap();
    let handoff = receipt
        .mobile_assist_handoff
        .as_mut()
        .ok_or("station grip provenance absent")?;
    handoff[receipt_key] = serde_json::json!({
        "goal":goal,"minimum_holding_ticks":100,"actual_holding_ticks":holding_ticks,
        "actual_current_rgb":true,"source_tick":latest.timing.episode_integrations,
        "goal_origin":origin,
        "original_task_profile":"mobile_box_homie_v2_n1_6",
        "world_or_contact_truth_input":false,"Qwen_calls":0,"task_qualified":false,
    });
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.submitted = true;
    assist.submitted_carry_observation = distance.map(|_| observation);
    runtime.requested = false;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulation_minigame::g1::task_runner::ArenaTaskExecution;

    #[test]
    fn newly_submitted_fine_carry_cannot_reuse_previous_completed_boundary() {
        use robot_minigame::g1::contract::G1Command;
        use simulation_minigame::g1::{
            mobile_grip::MobileGripReceipt,
            mobile_navigation::{MobileNavigationPhase, MobileNavigationStep},
        };
        let old = ObservationStamp {
            episode_id: 20632,
            frame_id: 7,
            sim_time_ns: 17_560_000_000,
            captured_at_unix_ms: 1,
        };
        let submitted = ObservationStamp {
            frame_id: 8,
            sim_time_ns: 26_680_000_000,
            ..old
        };
        let mut execution = MobileAssistExecution::ClassicalCarry {
            goal: MobileCarryGoal {
                observation: old,
                heading_yaw_source_rad: 0.,
                relative_distance_m: 0.1,
            },
            grip: MobileGripReceipt {
                schema: "unused_boundary_fixture",
                episode_id: 20632,
                source_tick: 200,
                original_vla_output: false,
                task_qualified: false,
                original_palm_gap_m: 0.,
                calibrated_palm_gap_m: 0.,
                maximum_joint_target_change_rad: 0.,
                maximum_palm_position_residual_m: 0.,
                maximum_palm_rotation_residual: 0.,
                solve_iterations: [0; 2],
            },
            navigation: MobileNavigationStep {
                phase: MobileNavigationPhase::Stop,
                navigation: [0.; 3],
                own_velocity_odometry_xy_m: [0.; 2],
                completed: true,
                execution_heading_yaw_source_rad: 0.,
                execution_relative_distance_m: 0.1,
            },
            command: G1Command::default(),
        };
        // The owner still publishes the old completed coarse carry immediately
        // after queuing a fine carry; requesting RGB here captured Tick1334.
        assert!(!carry_boundary_completed(&execution, Some(submitted)));
        let MobileAssistExecution::ClassicalCarry {
            goal, navigation, ..
        } = &mut execution
        else {
            unreachable!()
        };
        goal.observation = submitted;
        navigation.completed = false;
        assert!(!carry_boundary_completed(&execution, Some(submitted)));
        let MobileAssistExecution::ClassicalCarry { navigation, .. } = &mut execution else {
            unreachable!()
        };
        navigation.completed = true;
        assert!(carry_boundary_completed(&execution, Some(submitted)));
        assert!(!carry_boundary_completed(
            &execution,
            Some(ObservationStamp {
                episode_id: 20633,
                ..submitted
            })
        ));
        assert!(carry_boundary_completed(&execution, None));
    }

    #[test]
    fn submitted_hold_waits_for_owner_acknowledgment_of_prior_completed_grasp() {
        let execution = MobileAssistExecution::OriginalVla(ArenaTaskExecution {
            profile: TaskProfile::MobileBox,
            sequence_id: 4,
            frame_index: 49,
            observation: ObservationStamp {
                episode_id: 20625,
                frame_id: 4,
                sim_time_ns: 3_000_000_000,
                captured_at_unix_ms: 1,
            },
            execution_start_sim_ns: 3_000_000_000,
            observation_age_ns: 980_000_000,
            observation_wall_age_ms: 0,
            decoded_waist_targets_rad: [0.; 3],
            admitted_chunks: 4,
        });
        assert_eq!(completed_hold_ticks(&execution, 200), Ok(None));
        assert!(completed_hold_ticks(&execution, 201).is_err());
    }

    #[test]
    fn prior_incomplete_or_different_grasp_is_rejected_at_hold_boundary() {
        let valid = ArenaTaskExecution {
            profile: TaskProfile::MobileBox,
            sequence_id: 4,
            frame_index: 49,
            observation: ObservationStamp {
                episode_id: 20626,
                frame_id: 4,
                sim_time_ns: 3_000_000_000,
                captured_at_unix_ms: 1,
            },
            execution_start_sim_ns: 3_000_000_000,
            observation_age_ns: 980_000_000,
            observation_wall_age_ms: 0,
            decoded_waist_targets_rad: [0.; 3],
            admitted_chunks: 4,
        };
        let mut wrong = valid.clone();
        wrong.frame_index = 48;
        assert!(completed_hold_ticks(&MobileAssistExecution::OriginalVla(wrong), 200).is_err());
        let mut wrong = valid.clone();
        wrong.sequence_id = 3;
        assert!(completed_hold_ticks(&MobileAssistExecution::OriginalVla(wrong), 200).is_err());
        let mut wrong = valid.clone();
        wrong.admitted_chunks = 3;
        assert!(completed_hold_ticks(&MobileAssistExecution::OriginalVla(wrong), 200).is_err());
        let mut wrong = valid;
        wrong.execution_start_sim_ns = 2_000_000_000;
        assert!(completed_hold_ticks(&MobileAssistExecution::OriginalVla(wrong), 200).is_err());
    }
}
