//! One actual paused observation before any calibrated grip command.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_open::MobileOpenGoal,
    worker::TimedCommand,
};

pub(super) fn validate_predecessor(
    execution: &MobileAssistExecution,
    tick: u64,
) -> Result<(), String> {
    if !matches!(execution, MobileAssistExecution::OriginalVla(previous)
        if tick == 200 && previous.profile == TaskProfile::MobileBox
            && previous.sequence_id == 4 && previous.frame_index == 49
            && previous.admitted_chunks == 4
            && previous.execution_start_sim_ns == 3_000_000_000)
    {
        return Err(
            "pregrasp diagnostic requires exactly four completed original chunks at200Tick".into(),
        );
    }
    Ok(())
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("pregrasp owner absent")?
        .clone();
    if runtime.mobile_assist.as_ref().unwrap().regrasp.is_some() {
        return super::mobile_regrasp::drive(runtime, outcome, port);
    }
    if runtime
        .mobile_assist
        .as_ref()
        .unwrap()
        .pregrasp_close_goal
        .is_some()
    {
        return super::mobile_pregrasp_close::drive(runtime, outcome, port);
    }
    if runtime
        .mobile_assist
        .as_ref()
        .unwrap()
        .pregrasp_open_goal
        .is_some()
    {
        return after_opening(runtime, outcome, port);
    }
    if latest.timing.episode_integrations > 200 {
        return Err("pregrasp diagnostic advanced beyond its200Tick observation budget".into());
    }
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let execution = &latest
        .assist_step
        .as_ref()
        .ok_or("pregrasp execution absent")?
        .execution;
    validate_predecessor(execution, latest.timing.episode_integrations)?;
    if runtime.mobile_assist.as_ref().unwrap().vision_job.is_none() {
        start_marker_job(runtime, port, "pregrasp_geometry")?;
        return Ok(false);
    }
    let job = runtime
        .mobile_assist
        .as_ref()
        .unwrap()
        .vision_job
        .as_ref()
        .unwrap();
    let Some(reply) = job.try_take() else {
        return Ok(false);
    };
    let reply = reply?;
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&reply)?;
    if stamp.episode_id != runtime.episode_id
        || stamp.sim_time_ns != 4_000_000_000
        || stamp != job.observation
    {
        return Err("pregrasp report changed its actual200Tick image identity".into());
    }
    let receipt = serde_json::json!({
        "mode":"actual_pregrasp_paired_RGB_original_hand_geometry_diagnostic",
        "observation":stamp,"actual_integrations":200,"original_vla_chunks":4,
        "owner_paused_at_same_tick_during_both_views":true,
        "calibrated_grip_insertion_hold_or_transport_submitted":false,
        "Qwen_results":0,"task_qualified":false,"holding_proven":false,
        "owner_execution_admitted":false,"geometry":reply["held_contact_geometry"],
    });
    fs::write(
        runtime
            .options
            .output
            .join("pregrasp_geometry_receipt.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    outcome.0.lock().unwrap().mobile_assist_handoff = Some(receipt);
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.vision_job = None;
    if matches!(&assist.configuration, MobileAssistStage::Scan(c) if c.pregrasp_source_close_once) {
        let goal = simulation_minigame::g1::mobile_hold::MobileHoldGoal { observation: stamp };
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("source-close diagnostic lost its sole owner".into());
        };
        let submitted_at = Instant::now();
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: super::mobile_pregrasp_close::SOURCE_CLOSE_END_SIM_NS,
                valid_until_wall: submitted_at + Duration::from_secs(4),
                command: MobileAssistCommand::ClassicalGripSettle(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        assist.pregrasp_close_goal = Some(goal);
        assist.pregrasp_close_submitted_at = Some(submitted_at);
        runtime.requested = false;
        return Ok(false);
    }
    if matches!(&assist.configuration, MobileAssistStage::Scan(c) if c.pregrasp_open_once) {
        let goal = MobileOpenGoal { observation: stamp };
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("pregrasp opening lost its sole owner".into());
        };
        let submitted_at = Instant::now();
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: 5_000_000_000,
                valid_until_wall: submitted_at + Duration::from_secs(4),
                command: MobileAssistCommand::ClassicalGraspOpen(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        assist.pregrasp_open_goal = Some(goal);
        assist.pregrasp_open_submitted_at = Some(submitted_at);
        runtime.requested = false;
        return Ok(false);
    }
    assist.completed = true;
    Ok(true)
}

fn opening_boundary(
    execution: &MobileAssistExecution,
    tick: u64,
    goal: &MobileOpenGoal,
) -> Result<bool, String> {
    goal.validate().map_err(|e| e.to_string())?;
    if tick == 200 {
        validate_predecessor(execution, tick)?;
        if !matches!(execution, MobileAssistExecution::OriginalVla(previous)
            if previous.observation.episode_id == goal.observation.episode_id)
        {
            return Err("queued opening predecessor belongs to another episode".into());
        }
        // The display can still hold the old paused snapshot after submission.
        // Only the matching owner completion at250 admits the next camera pair.
        return Ok(false);
    }
    if !matches!(execution,
        MobileAssistExecution::ClassicalGraspOpen {goal:executed,opening}
            if executed==goal && opening.completed && opening.opening_ticks==50
                && opening.settling_ticks==0 && tick==250)
    {
        return Err("post-opening RGB lacks its own completed50Tick opening".into());
    }
    Ok(true)
}

fn after_opening(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("opened owner absent")?
        .clone();
    let goal = runtime
        .mobile_assist
        .as_ref()
        .unwrap()
        .pregrasp_open_goal
        .as_ref()
        .unwrap()
        .clone();
    if latest.timing.episode_integrations > 250 {
        return Err("pregrasp opening exceeded its250Tick budget".into());
    }
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    if !opening_boundary(
        &latest
            .assist_step
            .as_ref()
            .ok_or("opening execution absent")?
            .execution,
        latest.timing.episode_integrations,
        &goal,
    )? {
        let submitted_at = runtime
            .mobile_assist
            .as_ref()
            .unwrap()
            .pregrasp_open_submitted_at
            .ok_or("queued opening lacks its submission deadline")?;
        if submitted_at.elapsed() >= Duration::from_secs(4) {
            return Err("queued opening did not advance within its original4s lifetime".into());
        }
        return Ok(false);
    }
    if runtime.mobile_assist.as_ref().unwrap().vision_job.is_none() {
        start_marker_job(runtime, port, "pregrasp_opened_geometry")?;
        return Ok(false);
    }
    let job = runtime
        .mobile_assist
        .as_ref()
        .unwrap()
        .vision_job
        .as_ref()
        .unwrap();
    let Some(reply) = job.try_take() else {
        return Ok(false);
    };
    let reply = reply?;
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&reply)?;
    if stamp != job.observation
        || stamp.episode_id != goal.observation.episode_id
        || stamp.sim_time_ns != 5_000_000_000
        || stamp.frame_id <= goal.observation.frame_id
        || stamp.captured_at_unix_ms <= goal.observation.captured_at_unix_ms
    {
        return Err("post-opening report is foreign or lacks fresh250Tick RGB".into());
    }
    let receipt = serde_json::json!({
        "mode":"actual_opened_pregrasp_paired_RGB_geometry_diagnostic","origin_observation":goal.observation,
        "observation":stamp,"actual_integrations":250,"original_vla_chunks":4,
        "original_release_horizontal_opening_ticks":50,"target_palm_gap_m":0.30,
        "settling_insertion_close_hold_or_transport_submitted":false,
        "owner_paused_at_same_tick_during_both_views":true,"Qwen_results":0,
        "task_qualified":false,"holding_proven":false,"insertion_execution_admitted":false,
        "geometry":reply["held_contact_geometry"],
    });
    fs::write(
        runtime
            .options
            .output
            .join("pregrasp_opened_geometry_receipt.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .unwrap()["post_opening"] = receipt;
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.vision_job = None;
    assist.completed = true;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulation_minigame::g1::task_runner::ArenaTaskExecution;

    #[test]
    fn queued_opening_keeps_completed_vla_snapshot_pending() {
        let previous = ArenaTaskExecution {
            profile: TaskProfile::MobileBox,
            sequence_id: 4,
            frame_index: 49,
            observation: ObservationStamp {
                episode_id: 20711,
                frame_id: 4,
                sim_time_ns: 3_000_000_000,
                captured_at_unix_ms: 1,
            },
            execution_start_sim_ns: 3_000_000_000,
            observation_age_ns: 0,
            observation_wall_age_ms: 0,
            decoded_waist_targets_rad: [0.; 3],
            admitted_chunks: 4,
        };
        let goal = MobileOpenGoal {
            observation: ObservationStamp {
                episode_id: 20711,
                frame_id: 5,
                sim_time_ns: 4_000_000_000,
                captured_at_unix_ms: 2,
            },
        };
        let execution = MobileAssistExecution::OriginalVla(previous);
        assert_eq!(opening_boundary(&execution, 200, &goal), Ok(false));
        for tick in [199, 201, 250] {
            assert!(opening_boundary(&execution, tick, &goal).is_err());
        }
        let mut foreign = goal.clone();
        foreign.observation.episode_id += 1;
        assert!(opening_boundary(&execution, 200, &foreign).is_err());
        let mut stale = goal;
        stale.observation.sim_time_ns = 3_000_000_000;
        assert!(opening_boundary(&execution, 200, &stale).is_err());
    }

    #[test]
    fn a_later_hold_or_incomplete_chunk_cannot_be_called_pregrasp() {
        let previous = ArenaTaskExecution {
            profile: TaskProfile::MobileBox,
            sequence_id: 4,
            frame_index: 49,
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 4,
                sim_time_ns: 3_000_000_000,
                captured_at_unix_ms: 1,
            },
            execution_start_sim_ns: 3_000_000_000,
            observation_age_ns: 0,
            observation_wall_age_ms: 0,
            decoded_waist_targets_rad: [0.; 3],
            admitted_chunks: 4,
        };
        let execution = MobileAssistExecution::OriginalVla(previous.clone());
        assert!(validate_predecessor(&execution, 200).is_ok());
        for tick in [199, 201, 250, 300] {
            assert!(validate_predecessor(&execution, tick).is_err());
        }
        let mut incomplete = previous;
        incomplete.frame_index = 48;
        assert!(
            validate_predecessor(&MobileAssistExecution::OriginalVla(incomplete), 200).is_err()
        );
    }
}
