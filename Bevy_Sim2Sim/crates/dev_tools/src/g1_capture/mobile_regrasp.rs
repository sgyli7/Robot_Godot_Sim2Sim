//! Current closed RGB drives at most two finite reacquisitions, never truth.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_regrasp::{MobileRegraspGoal, MobileRegraspPhase},
    worker::TimedCommand,
};

pub(super) struct Recovery {
    goal: MobileRegraspGoal,
    submitted_at: Instant,
    closed_marker_root_m: [f64; 3],
}

fn marker_position(reply: &serde_json::Value) -> Result<[f64; 3], String> {
    if reply["same_tick_box_pair_pose"]["marker_id"] != 22 {
        return Err("regrasp lost visible box22".into());
    }
    let pose = &reply["same_tick_box_pair_pose"]["root_from_marker"];
    let mut p = [0.; 3];
    for i in 0..3 {
        p[i] = pose[i][3]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or("regrasp lacks current marker pose")?;
    }
    Ok(p)
}

fn goal_from_closed(
    reply: &serde_json::Value,
    attempt: u32,
) -> Result<Option<MobileRegraspGoal>, String> {
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(reply)?;
    let p = &reply["closed_regrasp_proposal"];
    if p["geometry_candidate_found"] != true {
        return Err("measured closed thumbs have no bounded reacquisition candidate".into());
    }
    let depth = p["common_forward_translation_m"]
        .as_f64()
        .ok_or("closed depth absent")?;
    if !depth.is_finite() || !(0. ..=0.05).contains(&depth) {
        return Err("closed depth outside original5cm bound".into());
    }
    if depth < 0.000_001 {
        return Ok(None);
    }
    let offset: [f64; 3] = serde_json::from_value(p["common_offset_root_source_m"].clone())
        .map_err(|e| e.to_string())?;
    let goal = MobileRegraspGoal {
        attempt,
        phase: MobileRegraspPhase::Open,
        closed_observation: stamp,
        observation: stamp,
        common_offset_root_source_m: offset,
    };
    goal.validate().map_err(|e| e.to_string())?;
    Ok(Some(goal))
}

fn submit(
    runtime: &mut CaptureRuntime,
    goal: MobileRegraspGoal,
    closed_marker_root_m: [f64; 3],
) -> Result<(), String> {
    goal.validate().map_err(|e| e.to_string())?;
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("regrasp lost sole physical owner".into());
    };
    let submitted_at = Instant::now();
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: goal.end_tick() * 20_000_000,
            valid_until_wall: submitted_at + Duration::from_secs(4),
            command: MobileAssistCommand::ClassicalRegrasp(goal.clone()),
        })
        .map_err(|e| e.to_string())?;
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.regrasp = Some(Recovery {
        goal,
        submitted_at,
        closed_marker_root_m,
    });
    assist.vision_job = None;
    runtime.requested = false;
    Ok(())
}

pub(super) fn begin(runtime: &mut CaptureRuntime, reply: &serde_json::Value) -> Result<(), String> {
    let goal = goal_from_closed(reply, 1)?.ok_or(
        "initial closed depth needs no reacquisition; separate holding validation required",
    )?;
    submit(runtime, goal, marker_position(reply)?)
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("regrasp owner absent")?
        .clone();
    let assist = runtime.mobile_assist.as_ref().unwrap();
    let r = assist.regrasp.as_ref().unwrap();
    let goal = r.goal.clone();
    let marker = r.closed_marker_root_m;
    let tick = latest.timing.episode_integrations;
    if tick > goal.end_tick() {
        return Err("regrasp exceeded finite phase integration budget".into());
    }
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let execution = &latest
        .assist_step
        .as_ref()
        .ok_or("regrasp execution absent")?
        .execution;
    if tick == goal.start_tick() {
        // Only a queued command may still expose its previous paused image.
        let own = match execution {
            MobileAssistExecution::ClassicalGripSettle {
                goal: g, holding, ..
            } => {
                goal.attempt == 1
                    && goal.phase == MobileRegraspPhase::Open
                    && holding.completed
                    && g.observation.episode_id == goal.observation.episode_id
                    && tick == 300
            }
            MobileAssistExecution::ClassicalRegrasp {
                goal: previous,
                regrasp,
            } => regrasp.completed && goal.follows(previous),
            _ => false,
        };
        if !own || r.submitted_at.elapsed() >= Duration::from_secs(4) {
            return Err("regrasp queued predecessor is foreign or expired".into());
        }
        return Ok(false);
    }
    if !matches!(execution,MobileAssistExecution::ClassicalRegrasp {goal:g,regrasp}
        if g==&goal && tick==goal.end_tick() && regrasp.completed && regrasp.phase_ticks==goal.phase.duration_ticks())
    {
        return Err("fresh regrasp camera lacks its own completed physical phase".into());
    }
    let directory = format!(
        "regrasp_{}_{}",
        goal.attempt,
        match goal.phase {
            MobileRegraspPhase::Open => "opened",
            MobileRegraspPhase::Settle => "settled",
            MobileRegraspPhase::Insert => "inserted",
            MobileRegraspPhase::Close => "closed",
            MobileRegraspPhase::Hold => "held",
        }
    );
    if assist.vision_job.is_none() {
        start_marker_job(runtime, port, &directory)?;
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
        || stamp.sim_time_ns != goal.end_tick() * 20_000_000
        || stamp.frame_id <= goal.observation.frame_id
        || stamp.captured_at_unix_ms <= goal.observation.captured_at_unix_ms
    {
        return Err("regrasp changed current phase RGB identity".into());
    }
    let receipt = serde_json::json!({"mode":"bounded_current_closed_RGB_reacquisition","goal":goal,"observation":stamp,
        "actual_integrations":tick,"execution":execution,"geometry":reply["held_contact_geometry"],
        "depth_proposal":reply["closed_regrasp_proposal"],"open_closed_reference_forecast":reply["open_regrasp_forecast"],"task_qualified":false,"holding_proven":false,
        "world_or_contact_truth_input":false,"lift_carry_release_submitted":false});
    fs::write(
        runtime
            .options
            .output
            .join(format!("{directory}_receipt.json")),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .unwrap()[&directory] = receipt;
    let phase = match goal.phase {
        MobileRegraspPhase::Open => {
            // The box can resettle when released from a partial source grip.
            // Keep the original opened target for100 real standing Ticks,
            // then require fresh RGB and an explicitly disclosed forecast.
            let current = marker_position(&reply)?;
            submit(
                runtime,
                MobileRegraspGoal {
                    phase: MobileRegraspPhase::Settle,
                    observation: stamp,
                    ..goal
                },
                current,
            )?;
            return Ok(false);
        }
        MobileRegraspPhase::Settle => {
            let current = marker_position(&reply)?;
            let motion = current
                .iter()
                .zip(marker)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if motion > 0.02 {
                return Err("opened target still moves over existing20mm bound after finite settling; pause required".into());
            }
            let forecast = &reply["open_regrasp_forecast"];
            if forecast["geometry_candidate_found"] != true {
                return Err("fresh settled box lacks bounded closed-reference forecast".into());
            }
            let offset: [f64; 3] =
                serde_json::from_value(forecast["common_offset_root_source_m"].clone())
                    .map_err(|e| e.to_string())?;
            submit(
                runtime,
                MobileRegraspGoal {
                    phase: MobileRegraspPhase::Insert,
                    observation: stamp,
                    common_offset_root_source_m: offset,
                    ..goal
                },
                current,
            )?;
            return Ok(false);
        }
        MobileRegraspPhase::Insert => {
            let current = marker_position(&reply)?;
            let motion = current
                .iter()
                .zip(marker)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if motion > 0.02 {
                return Err(
                    "open insertion moved visible target over20mm; source reclose rejected".into(),
                );
            }
            MobileRegraspPhase::Close
        }
        MobileRegraspPhase::Close => MobileRegraspPhase::Hold,
        MobileRegraspPhase::Hold => {
            if matches!(&runtime.mobile_assist.as_ref().unwrap().configuration,MobileAssistStage::Scan(c) if c.current_regrasp_lift_probe)
            {
                if goal.attempt != 1 {
                    return Err("lift probe cannot repeat regrasp".into());
                }
                let assist = runtime.mobile_assist.as_mut().unwrap();
                assist.vision_job = None;
                assist.regrasp_lift = Some(super::mobile_regrasp_lift::Probe::default());
                runtime.requested = false;
                return Ok(false);
            }
            let p = &reply["closed_regrasp_proposal"];
            if p["geometry_candidate_found"] != true {
                return Err("post-regrasp measured thumbs lack bounded depth candidate".into());
            }
            let depth = p["common_forward_translation_m"]
                .as_f64()
                .ok_or("post-regrasp depth absent")?;
            if depth < 0.000_001 {
                let assist = runtime.mobile_assist.as_mut().unwrap();
                assist.vision_job = None;
                assist.completed = true;
                return Ok(true);
            }
            if goal.attempt == 2 {
                return Err(
                    "two visual reacquisitions exhausted; measured thumbs still require insertion"
                        .into(),
                );
            }
            let next = goal_from_closed(&reply, goal.attempt + 1)?
                .ok_or("closed geometry changed during recovery")?;
            submit(runtime, next, marker_position(&reply)?)?;
            return Ok(false);
        }
    };
    submit(
        runtime,
        MobileRegraspGoal {
            phase,
            observation: stamp,
            ..goal
        },
        marker,
    )?;
    Ok(false)
}
