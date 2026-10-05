//! One actual lift test; wholehand geometry is only a candidate, never holding proof.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_raise::MobileRaiseGoal,
    mobile_regrasp::MobileRegraspPhase,
    worker::TimedCommand,
};

#[derive(Default)]
pub(super) struct Probe {
    origin: Option<serde_json::Value>,
    goal: Option<MobileRaiseGoal>,
    submitted_at: Option<Instant>,
}

pub(super) fn bilateral_candidate(reply: &serde_json::Value) -> bool {
    let Some(bodies) = reply["held_contact_geometry"]["bodies"].as_array() else {
        return false;
    };
    ["left", "right"].into_iter().all(|side| {
        bodies.iter().any(|b| {
            b["side"] == side
                && b["linf_distance_m"]
                    .as_f64()
                    .is_some_and(|d| d.is_finite() && (0. ..=1e-7).contains(&d))
        })
    })
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("regrasp lift owner absent")?
        .clone();
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let tick = latest.timing.episode_integrations;
    let execution = &latest
        .assist_step
        .as_ref()
        .ok_or("lift execution absent")?
        .execution;
    let assist = runtime.mobile_assist.as_ref().unwrap();
    let probe = assist.regrasp_lift.as_ref().unwrap();
    if probe.goal.is_none() {
        if !matches!(execution,MobileAssistExecution::ClassicalRegrasp {goal,regrasp}
            if goal.attempt==1 && goal.phase==MobileRegraspPhase::Hold && goal.end_tick()==tick
                && tick==650 && regrasp.completed && regrasp.phase_ticks==100 && regrasp.stable_velocity_ticks>=20)
        {
            return Err("lift probe lacks its own actual first regrasp hold".into());
        }
        if assist.vision_job.is_none() {
            start_marker_job(runtime, port, "regrasp_lift_before")?;
            return Ok(false);
        }
        let job = assist.vision_job.as_ref().unwrap();
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&reply)?;
        if stamp != job.observation
            || stamp.episode_id != runtime.episode_id
            || stamp.sim_time_ns != 13_000_000_000
            || !bilateral_candidate(&reply)
        {
            return Err("lift probe has no current bilateral wholehand candidate".into());
        }
        let goal = MobileRaiseGoal {
            observation: stamp,
            distance_m: 0.1,
            duration_ticks: 100,
        };
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("lift lost sole owner".into());
        };
        let submitted_at = Instant::now();
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: 17_000_000_000,
                valid_until_wall: submitted_at + Duration::from_secs(6),
                command: MobileAssistCommand::ClassicalRegraspPickupRaise(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        let assist = runtime.mobile_assist.as_mut().unwrap();
        let p = assist.regrasp_lift.as_mut().unwrap();
        p.origin = Some(reply);
        p.goal = Some(goal);
        p.submitted_at = Some(submitted_at);
        assist.vision_job = None;
        runtime.requested = false;
        return Ok(false);
    }
    let goal = probe.goal.as_ref().unwrap();
    if tick == 650 {
        if probe.submitted_at.unwrap().elapsed() >= Duration::from_secs(6)
            || !matches!(execution,MobileAssistExecution::ClassicalRegrasp {goal:g,regrasp}
            if g.observation.episode_id==goal.observation.episode_id && g.phase==MobileRegraspPhase::Hold && regrasp.completed)
        {
            return Err("queued lift lacks its own unexpired hold predecessor".into());
        }
        return Ok(false);
    }
    let MobileAssistExecution::ClassicalRaise {
        goal: executed,
        raising,
    } = execution
    else {
        return Err("regrasp lift execution changed phase".into());
    };
    if executed != goal
        || tick != 850
        || !raising.completed
        || raising.raising_ticks != 100
        || raising.settling_ticks != 100
    {
        return Err("lift exceeded its original100raise+100settle budget".into());
    }
    if assist.vision_job.is_none() {
        start_marker_job(runtime, port, "regrasp_lift_after")?;
        return Ok(false);
    }
    let job = assist.vision_job.as_ref().unwrap();
    let Some(reply) = job.try_take() else {
        return Ok(false);
    };
    let reply = reply?;
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&reply)?;
    if stamp != job.observation {
        return Err("lift changed fresh after-image identity".into());
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64;
    let verdict = super::mobile_pickup::verify_geometry_lift(
        probe.origin.as_ref().unwrap(),
        &reply,
        raising.self_velocity_displacement_source_m,
        tick,
        now,
    )?;
    let receipt = serde_json::json!({"mode":"actual_regrasp_lift_current_paired_RGB_diagnostic","goal":goal,"verdict":verdict,
        "actual_integrations":850,"raising_ticks":100,"settling_ticks":100,
        "wholehand_geometry_only_pre_lift_candidate":true,"thumb0_contact_not_required_or_sufficient":true,
        "hand_only_support_requires_independent_truth_audit":true,"Qwen_results":0,"carry_release_submitted":false,"task_qualified":false});
    fs::write(
        runtime.options.output.join("regrasp_lift_receipt.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .unwrap()["regrasp_lift"] = receipt;
    if matches!(&runtime.mobile_assist.as_ref().unwrap().configuration,
        MobileAssistStage::Scan(c) if c.current_regrasp_stability_probe)
    {
        if verdict["box_source_center_lift_m"]
            .as_f64()
            .is_none_or(|h| h < 0.025)
            || verdict["midpalm_source_lift_m"]
                .as_f64()
                .is_none_or(|h| h < 0.025)
            || !bilateral_candidate(&reply)
        {
            return Err(
                "postload observation lacks a lifted bilateral candidate; no hold admitted".into(),
            );
        }
        // The failed entry verdict remains in the journal. Only two finite
        // unchanged-target holds may distinguish settling from continuing slip.
        super::mobile_regrasp_loaded::begin(runtime, reply)?;
        return Ok(false);
    }
    if verdict["accepted"] != true {
        return Err(
            "actual regrasp lift failed original pairedRGB25mm rise/20mm relative guard".into(),
        );
    }
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.vision_job = None;
    assist.completed = true;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_hand_or_nonfinite_geometry_cannot_admit_a_lift_candidate() {
        let left = serde_json::json!({"side":"left","linf_distance_m":0.});
        let right = serde_json::json!({"side":"right","linf_distance_m":0.});
        assert!(!bilateral_candidate(
            &serde_json::json!({"held_contact_geometry":{"bodies":[left]}})
        ));
        assert!(bilateral_candidate(
            &serde_json::json!({"held_contact_geometry":{"bodies":[left,right]}})
        ));
        assert!(!bilateral_candidate(
            &serde_json::json!({"held_contact_geometry":{"bodies":[left,{"side":"right","linf_distance_m":null}]}})
        ));
    }
}
