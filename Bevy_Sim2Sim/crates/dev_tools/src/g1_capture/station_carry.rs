//! New actual RGB after the independently measured native station grip hold.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_navigation::{MobileCarryGoal, MobileScanGoal},
    worker::TimedCommand,
};

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
    runtime.mobile_assist.as_mut().unwrap().submitted = true;
    runtime.requested = false;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulation_minigame::g1::task_runner::ArenaTaskExecution;

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
