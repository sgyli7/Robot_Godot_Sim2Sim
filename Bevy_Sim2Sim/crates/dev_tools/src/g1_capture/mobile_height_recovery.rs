//! One opt-in image-derived raise followed by a fresh same-Tick RGB pair.
//! This never admits forward motion or relaxes the original 100 mm rim gate.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_raise::MobileRaiseGoal,
    worker::TimedCommand,
};

#[derive(Default)]
pub(super) struct Recovery {
    goal: Option<MobileRaiseGoal>,
    completed: bool,
}

pub(super) fn reobservation_directory(fine_goals_submitted: u32) -> String {
    format!(
        "visual_fine_approach_after_height_recovery_{:02}",
        fine_goals_submitted + 1
    )
}

impl Recovery {
    pub fn pending(&self) -> bool {
        self.goal.is_some() && !self.completed
    }
    pub fn used(&self) -> bool {
        self.goal.is_some()
    }
    pub fn completed_for(&self, execution: &MobileAssistExecution) -> bool {
        self.completed
            && self.goal.as_ref().is_some_and(|submitted| {
                matches!(execution, MobileAssistExecution::ClassicalRaise { goal, raising }
                if submitted == goal && raising.completed)
            })
    }
}

fn admit_goal(
    reply: &serde_json::Value,
    observation: ObservationStamp,
    native_tick: u64,
    used: bool,
) -> Result<MobileRaiseGoal, String> {
    if used {
        return Err("one RGB height recovery was already used; owner remains paused".into());
    }
    let clearance = &reply["clearance_proposal"];
    let required = clearance["required_raise_m"]
        .as_f64()
        .filter(|r| r.is_finite() && *r > 0.)
        .ok_or("height recovery requires positive current image clearance")?;
    let goal: MobileRaiseGoal = serde_json::from_value(clearance["raise_goal"].clone())
        .map_err(|e| format!("current RGB has no bounded recovery raise goal: {e}"))?;
    goal.validate().map_err(|e| e.to_string())?;
    if observation.sim_time_ns != native_tick * 20_000_000
        || goal.observation != observation
        || clearance["observation"] != serde_json::to_value(observation).unwrap()
        || (f64::from(goal.distance_m) - required).abs() > 1e-7
        || clearance["minimum_desired_rim_clearance_m"] != 0.1
        || clearance["world_or_contact_truth_input"] != false
        || reply["same_tick_box_pair_pose"]["schema"] != "g1_actual_rgb_same_tick_box_pair_pose_v1"
    {
        return Err("height recovery has stale, unpaired or changed current RGB geometry".into());
    }
    Ok(goal)
}

pub(super) fn submit_from_current_rgb(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    reply: serde_json::Value,
    observation: ObservationStamp,
    localization_wall_ms: f64,
) -> Result<bool, String> {
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("height recovery absent")?;
    if !matches!(&assist.configuration, MobileAssistStage::Scan(c)
        if c.current_rgb_height_recovery && c.vision.as_ref().is_some_and(|v| v.same_tick_box_pair))
    {
        return Err(
            "fine approach lacks current auxiliary RGB/rim clearance; owner remains paused".into(),
        );
    }
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("height recovery owner absent")?;
    if latest.phase != G1WorkerPhase::Paused || observation.episode_id != runtime.episode_id {
        return Err("height recovery requires the current paused sole owner".into());
    }
    let goal = admit_goal(
        &reply,
        observation,
        latest.timing.episode_integrations,
        assist.height_recovery.used(),
    )?;
    if !latest.assist_step.as_ref().is_some_and(|step| {
        matches!(&step.execution, MobileAssistExecution::ClassicalCarry { navigation, .. }
            if navigation.completed && !navigation.blocked_stop_completed)
    }) {
        return Err("height recovery requires a normally completed stationary carry".into());
    }
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("height recovery lost sole native owner".into());
    };
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: observation.sim_time_ns + 10_000_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(12),
            command: MobileAssistCommand::ClassicalRaise(goal.clone()),
        })
        .map_err(|e| e.to_string())?;
    let mut receipt = outcome.0.lock().unwrap();
    let handoff = receipt
        .mobile_assist_handoff
        .as_mut()
        .ok_or("height recovery provenance absent")?;
    handoff["current_rgb_height_recovery"] = serde_json::json!({
        "schema":"g1_one_current_rgb_height_recovery_v1", "actual_localization":reply,
        "executed_raise_goal":goal,"source_tick":latest.timing.episode_integrations,
        "maximum_attempts":1,"localization_wall_ms":localization_wall_ms,
        "original_minimum_rim_clearance_m":0.1,"new_image_required_after_raise":true,
        "world_or_contact_truth_input":false,"Qwen_calls":0,"task_qualified":false,
    });
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.vision_job.take();
    assist.height_recovery.goal = Some(goal);
    assist.height_recovery.completed = false;
    runtime.requested = false;
    Ok(false)
}

pub(super) fn await_completion(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("height recovery owner absent")?;
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("height recovery absent")?;
    let submitted = assist
        .height_recovery
        .goal
        .as_ref()
        .ok_or("height recovery goal absent")?;
    let step = latest
        .assist_step
        .as_ref()
        .ok_or("height recovery execution absent")?;
    if latest.timing.episode_integrations * 20_000_000 == submitted.observation.sim_time_ns
        && matches!(&step.execution, MobileAssistExecution::ClassicalCarry { navigation, .. }
            if navigation.completed && !navigation.blocked_stop_completed)
    {
        // A command can be queued while its completed predecessor is still
        // visible. Never request an image from that old boundary.
        return Ok(false);
    }
    let MobileAssistExecution::ClassicalRaise { goal, raising } = &step.execution else {
        return Err("height recovery owner acknowledged a different skill".into());
    };
    if goal != submitted
        || !raising.completed
        || raising.raising_ticks != submitted.duration_ticks
        || raising.settling_ticks != 100
        || latest.timing.episode_integrations
            != submitted.observation.sim_time_ns / 20_000_000
                + u64::from(submitted.duration_ticks)
                + 100
    {
        return Err("height recovery lacks its exact completed raise receipt".into());
    }
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .ok_or("height recovery provenance absent")?["current_rgb_height_recovery"]["completed_source_tick"] =
        latest.timing.episode_integrations.into();
    runtime
        .mobile_assist
        .as_mut()
        .unwrap()
        .height_recovery
        .completed = true;
    runtime.requested = false;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_height_recovery_routes_to_the_existing_paired_capture_stage() {
        for segment in 0..=5 {
            let path = reobservation_directory(segment);
            assert!(super::super::allowed_mobile_box_pair_stage(&path));
            assert_ne!(path, "visual_fine_approach");
            assert!(!path.contains('/'));
        }
        assert!(!super::super::allowed_mobile_box_pair_stage(
            "visual_fine_after_height_recovery_01"
        ));
        assert!(!super::super::allowed_mobile_box_pair_stage(
            "station_held_transport"
        ));
    }
    fn fixture() -> (serde_json::Value, ObservationStamp) {
        let observation = ObservationStamp {
            episode_id: 1,
            frame_id: 8,
            sim_time_ns: 25_780_000_000,
            captured_at_unix_ms: 1,
        };
        let goal = MobileRaiseGoal {
            observation,
            distance_m: 0.021034081,
            duration_ticks: 50,
        };
        (
            serde_json::json!({"clearance_proposal":{"observation":observation,
            "required_raise_m":0.021034080928143684,"raise_goal":goal,
            "minimum_desired_rim_clearance_m":0.1,"world_or_contact_truth_input":false},
            "same_tick_box_pair_pose":{"schema":"g1_actual_rgb_same_tick_box_pair_pose_v1"}}),
            observation,
        )
    }
    #[test]
    fn only_current_paired_bounded_image_can_admit_one_raise() {
        let (reply, stamp) = fixture();
        assert!(admit_goal(&reply, stamp, 1289, false).is_ok());
        assert!(admit_goal(&reply, stamp, 1288, false).is_err());
        assert!(admit_goal(&reply, stamp, 1289, true).is_err());
        for (pointer, value) in [
            (
                "/clearance_proposal/raise_goal/observation/episode_id",
                serde_json::json!(2),
            ),
            (
                "/clearance_proposal/raise_goal/distance_m",
                serde_json::json!(0.18),
            ),
            (
                "/clearance_proposal/minimum_desired_rim_clearance_m",
                serde_json::json!(0.05),
            ),
            (
                "/clearance_proposal/world_or_contact_truth_input",
                serde_json::json!(true),
            ),
            (
                "/clearance_proposal/required_raise_m",
                serde_json::json!(-0.01),
            ),
            (
                "/same_tick_box_pair_pose/schema",
                serde_json::json!("single_view"),
            ),
        ] {
            let mut bad = reply.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(admit_goal(&bad, stamp, 1289, false).is_err(), "{pointer}");
        }
    }
    #[test]
    fn completed_raise_from_another_frame_cannot_admit_reobservation() {
        use robot_minigame::g1::contract::G1Command;
        use simulation_minigame::g1::mobile_raise::MobileRaiseStep;
        let (reply, stamp) = fixture();
        let goal = admit_goal(&reply, stamp, 1289, false).unwrap();
        let recovery = Recovery {
            goal: Some(goal.clone()),
            completed: true,
        };
        let mut execution = MobileAssistExecution::ClassicalRaise {
            goal,
            raising: MobileRaiseStep {
                commanded_rise_m: 0.021034081,
                cartesian_increment_root_source_m: [0.; 3],
                raising_ticks: 50,
                settling_ticks: 100,
                completed: true,
                self_velocity_displacement_source_m: [0.; 3],
                correction: None,
                command: G1Command::default(),
            },
        };
        assert!(recovery.completed_for(&execution));
        let MobileAssistExecution::ClassicalRaise { goal, .. } = &mut execution else {
            unreachable!()
        };
        goal.observation.frame_id += 1;
        assert!(!recovery.completed_for(&execution));
        assert!(recovery.used() && !recovery.pending());
        let pending = Recovery {
            goal: recovery.goal.clone(),
            completed: false,
        };
        assert!(pending.pending() && !pending.completed_for(&execution));
        assert!(!Recovery::default().used());
    }
}
