//! New actual RGB after the independently measured native station grip hold.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_navigation::MobileCarryGoal,
    worker::TimedCommand,
};

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
    let MobileAssistExecution::ClassicalGripSettle { holding, .. } = &step.execution else {
        return Err("station carry handoff lost its completed hold".into());
    };
    if !holding.completed
        || !(100..=250).contains(&holding.holding_ticks)
        || !(300..=450).contains(&latest.timing.episode_integrations)
    {
        return Err("station carry requires completed100..250Tick self-state hold".into());
    }
    let config = match &runtime
        .mobile_assist
        .as_ref()
        .ok_or("station carry configuration absent")?
        .configuration
    {
        MobileAssistStage::Carry(config) => config.clone(),
        _ => return Err("station public-aisle carry cannot select a source scan".into()),
    };
    let Some((observation, _)) =
        capture_current_marker_frame(runtime, port, "station_held_transport")?
    else {
        return Ok(false);
    };
    let goal = MobileCarryGoal {
        observation,
        heading_yaw_source_rad: config.heading_yaw_source_rad,
        relative_distance_m: config.relative_distance_m,
    };
    goal.validate().map_err(|e| e.to_string())?;
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("station carry lost the unique assisted physics owner".into());
    };
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: 46_000_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(40),
            command: MobileAssistCommand::ClassicalCarry(goal.clone()),
        })
        .map_err(|e| e.to_string())?;
    let mut receipt = outcome.0.lock().unwrap();
    let handoff = receipt
        .mobile_assist_handoff
        .as_mut()
        .ok_or("station grip provenance absent")?;
    handoff["station_public_aisle_carry_after_hold"] = serde_json::json!({
        "goal":goal,"minimum_holding_ticks":100,"actual_holding_ticks":holding.holding_ticks,
        "actual_current_rgb":true,"source_tick":latest.timing.episode_integrations,
        "goal_origin":"explicit_public_court_clear_aisle_not_target_bin_localization",
        "original_task_profile":"mobile_box_homie_v2_n1_6",
        "world_or_contact_truth_input":false,"Qwen_calls":0,"task_qualified":false,
    });
    runtime.mobile_assist.as_mut().unwrap().submitted = true;
    runtime.requested = false;
    Ok(false)
}
