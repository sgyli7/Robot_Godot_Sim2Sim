//! One actual paused observation before any calibrated grip command.

use super::*;
use simulation_minigame::g1::mobile_assist::MobileAssistExecution;

fn validate_predecessor(execution: &MobileAssistExecution, tick: u64) -> Result<(), String> {
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
    assist.completed = true;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulation_minigame::g1::task_runner::ArenaTaskExecution;

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
