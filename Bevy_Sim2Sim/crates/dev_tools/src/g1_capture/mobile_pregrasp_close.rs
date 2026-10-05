//! Current300Tick observation after the original source-gap settling primitive.

use super::*;
use simulation_minigame::g1::{mobile_assist::MobileAssistExecution, mobile_hold::MobileHoldGoal};

// The owner validates the completed integration time, including the last Tick.
pub(super) const SOURCE_CLOSE_END_SIM_NS: u64 = 6_000_000_000;

fn completed_boundary(
    execution: &MobileAssistExecution,
    tick: u64,
    goal: &MobileHoldGoal,
) -> Result<bool, String> {
    goal.validate().map_err(|e| e.to_string())?;
    if goal.observation.sim_time_ns != 4_000_000_000 {
        return Err("source-close diagnostic requires its current200Tick origin".into());
    }
    if tick == 200 {
        super::mobile_pregrasp_geometry::validate_predecessor(execution, tick)?;
        if !matches!(execution, MobileAssistExecution::OriginalVla(p)
            if p.observation.episode_id == goal.observation.episode_id)
        {
            return Err("source-close predecessor belongs to another episode".into());
        }
        return Ok(false);
    }
    if matches!(execution, MobileAssistExecution::ClassicalGripSettle {goal:executed,holding,..}
        if executed==goal && tick==300 && holding.completed && holding.holding_ticks==100
            && holding.stable_velocity_ticks>=25)
    {
        return Ok(true);
    }
    Err("current closed RGB lacks its own completed original100Tick settle".into())
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("source-close owner absent")?
        .clone();
    let assist = runtime.mobile_assist.as_ref().unwrap();
    let goal = assist.pregrasp_close_goal.as_ref().unwrap().clone();
    let tick = latest.timing.episode_integrations;
    if tick > 300 {
        return Err("source-close diagnostic exceeded its300Tick budget".into());
    }
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let execution = &latest
        .assist_step
        .as_ref()
        .ok_or("source-close execution absent")?
        .execution;
    if !completed_boundary(execution, tick, &goal)? {
        let submitted = assist
            .pregrasp_close_submitted_at
            .ok_or("source-close deadline absent")?;
        if submitted.elapsed() >= Duration::from_secs(4) {
            return Err("source-close queue exceeded its original4s lifetime".into());
        }
        return Ok(false);
    }
    if assist.vision_job.is_none() {
        start_marker_job(runtime, port, "pregrasp_source_closed_geometry")?;
        return Ok(false);
    }
    let job = assist.vision_job.as_ref().unwrap();
    let Some(reply) = job.try_take() else {
        return Ok(false);
    };
    let reply = reply?;
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&reply)?;
    if stamp != job.observation
        || stamp.episode_id != goal.observation.episode_id
        || stamp.sim_time_ns != 6_000_000_000
        || stamp.frame_id <= goal.observation.frame_id
        || stamp.captured_at_unix_ms <= goal.observation.captured_at_unix_ms
    {
        return Err("source-closed report changed its fresh300Tick image identity".into());
    }
    let receipt = serde_json::json!({
        "mode":"actual_original_source_closed300_paired_RGB_geometry_diagnostic",
        "origin_observation":goal.observation,"observation":stamp,"actual_integrations":300,
        "original_vla_chunks":4,"original_source_grip_settle_ticks":100,"Qwen_results":0,
        "original_source_gap_gains_fingers_and_materials_unchanged":true,
        "owner_paused_at_same_tick_during_both_views":true,
        "opening_insertion_lift_carry_release_submitted":false,
        "task_qualified":false,"holding_proven":false,"regrasp_owner_admitted":false,
        "geometry":reply["held_contact_geometry"],
    });
    fs::write(
        runtime
            .options
            .output
            .join("pregrasp_source_closed_geometry_receipt.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .unwrap()["post_source_closing"] = receipt;
    if matches!(&runtime.mobile_assist.as_ref().unwrap().configuration,MobileAssistStage::Scan(c) if c.current_closed_regrasp)
    {
        super::mobile_regrasp::begin(runtime, &reply)?;
        return Ok(false);
    }
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
    fn source_close_deadline_covers_the_last_integration_but_not_an_extra_tick() {
        // Worker::check_command compares the completed integration time.
        assert!(5_980_000_000 + 20_000_000 <= SOURCE_CLOSE_END_SIM_NS);
        assert!(6_000_000_000 + 20_000_000 > SOURCE_CLOSE_END_SIM_NS);
    }

    #[test]
    fn queued_source_close_cannot_consume_foreign_or_later_original_snapshot() {
        let goal = MobileHoldGoal {
            observation: ObservationStamp {
                episode_id: 20714,
                frame_id: 5,
                sim_time_ns: 4_000_000_000,
                captured_at_unix_ms: 2,
            },
        };
        let execution = MobileAssistExecution::OriginalVla(ArenaTaskExecution {
            profile: TaskProfile::MobileBox,
            sequence_id: 4,
            frame_index: 49,
            observation: ObservationStamp {
                episode_id: 20714,
                frame_id: 4,
                sim_time_ns: 3_000_000_000,
                captured_at_unix_ms: 1,
            },
            execution_start_sim_ns: 3_000_000_000,
            observation_age_ns: 0,
            observation_wall_age_ms: 0,
            decoded_waist_targets_rad: [0.; 3],
            admitted_chunks: 4,
        });
        assert_eq!(completed_boundary(&execution, 200, &goal), Ok(false));
        assert!(completed_boundary(&execution, 300, &goal).is_err());
        let mut foreign = goal;
        foreign.observation.episode_id += 1;
        assert!(completed_boundary(&execution, 200, &foreign).is_err());
    }
}
