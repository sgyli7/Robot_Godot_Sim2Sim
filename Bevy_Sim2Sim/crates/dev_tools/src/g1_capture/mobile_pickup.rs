//! One declared lift with paired RGB/self feedback before any station turn.
//! Contact and world-object state remain independent acceptance inputs only.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_raise::MobileRaiseGoal,
    worker::TimedCommand,
};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default)]
pub(super) struct Pickup {
    origin: Option<serde_json::Value>,
    goal: Option<MobileRaiseGoal>,
    holding_ticks: Option<u32>,
    accepted: bool,
    transport_baseline: Option<serde_json::Value>,
}

impl Pickup {
    pub fn completed(&self) -> bool {
        self.accepted
    }

    pub(super) fn transport_baseline(&self) -> Option<&serde_json::Value> {
        self.accepted
            .then_some(self.transport_baseline.as_ref())
            .flatten()
    }

    pub fn completed_hold_ticks(
        &self,
        execution: &MobileAssistExecution,
        tick: u64,
    ) -> Result<u32, String> {
        if self.accepted && self.own_completed_raise(execution, tick) {
            self.holding_ticks
                .ok_or("pickup lacks original grip hold".into())
        } else {
            Err("station turn lacks its own verified completed pickup".into())
        }
    }

    fn own_completed_raise(&self, execution: &MobileAssistExecution, tick: u64) -> bool {
        self.goal.as_ref().is_some_and(|submitted| {
            matches!(execution, MobileAssistExecution::ClassicalRaise { goal, raising }
                if goal == submitted && raising.completed
                    && raising.raising_ticks == 100 && raising.settling_ticks == 100
                    && tick == submitted.observation.sim_time_ns / 20_000_000 + 200)
        })
    }
}

fn feedback_vector(reply: &serde_json::Value, key: &str) -> Result<[f64; 3], String> {
    let vector: [f64; 3] = serde_json::from_value(reply[key].clone()).map_err(|e| e.to_string())?;
    if vector.iter().any(|v| !v.is_finite() || v.abs() > 8.) {
        return Err("pickup RGB/self feedback vector is nonfinite".into());
    }
    Ok(vector)
}

fn norm(v: [f64; 3]) -> f64 {
    v.into_iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn validate_feedback_pose(reply: &serde_json::Value) -> Result<(), String> {
    if reply["world_or_contact_truth_input"] != false
        || reply["box_view_only"] != true
        || reply["same_tick_box_pair_pose"]["schema"] != "g1_actual_rgb_same_tick_box_pair_pose_v1"
        || reply["same_tick_box_pair_pose"]["observations"][0] != reply["observation"]
        || !reply["navigation_proposal"].is_null()
        || !reply["clearance_proposal"].is_null()
    {
        return Err("pickup feedback lacks its bound current paired box-only RGB".into());
    }
    for key in [
        "current_box_gravity_center_m",
        "current_midpalm_gravity_center_m",
        "current_box_relative_left_palm_m",
    ] {
        feedback_vector(reply, key)?;
    }
    Ok(())
}

fn verify(
    origin: &serde_json::Value,
    current: &serde_json::Value,
    displacement: [f64; 3],
    native_tick: u64,
    now_ms: u64,
) -> Result<serde_json::Value, String> {
    verify_with_validation(
        origin,
        current,
        displacement,
        native_tick,
        now_ms,
        validate_feedback_pose,
    )
}

pub(super) fn verify_geometry_lift(
    origin: &serde_json::Value,
    current: &serde_json::Value,
    displacement: [f64; 3],
    native_tick: u64,
    now_ms: u64,
) -> Result<serde_json::Value, String> {
    fn validate(reply: &serde_json::Value) -> Result<(), String> {
        if reply["geometry_lift_feedback"] != true
            || reply["world_or_contact_truth_input"] != false
            || reply["held_contact_geometry"].is_null()
            || reply["same_tick_box_pair_pose"]["schema"]
                != "g1_actual_rgb_same_tick_box_pair_pose_v1"
            || !reply["navigation_proposal"].is_null()
            || !reply["clearance_proposal"].is_null()
        {
            return Err("lift feedback lacks current paired RGB/wholehand geometry".into());
        }
        for key in [
            "current_box_gravity_center_m",
            "current_midpalm_gravity_center_m",
            "current_box_relative_left_palm_m",
        ] {
            feedback_vector(reply, key)?;
        }
        Ok(())
    }
    verify_with_validation(origin, current, displacement, native_tick, now_ms, validate)
}

fn verify_with_validation(
    origin: &serde_json::Value,
    current: &serde_json::Value,
    displacement: [f64; 3],
    native_tick: u64,
    now_ms: u64,
    validate: fn(&serde_json::Value) -> Result<(), String>,
) -> Result<serde_json::Value, String> {
    let a: ObservationStamp =
        serde_json::from_value(origin["observation"].clone()).map_err(|e| e.to_string())?;
    let b: ObservationStamp =
        serde_json::from_value(current["observation"].clone()).map_err(|e| e.to_string())?;
    if a.episode_id == 0
        || a.episode_id != b.episode_id
        || a.frame_id == 0
        || b.frame_id <= a.frame_id
        || a.sim_time_ns % 20_000_000 != 0
        || b.sim_time_ns != a.sim_time_ns + 4_000_000_000
        || b.sim_time_ns != native_tick * 20_000_000
        || a.captured_at_unix_ms == 0
        || b.captured_at_unix_ms <= a.captured_at_unix_ms
        || b.captured_at_unix_ms > now_ms
        || now_ms - b.captured_at_unix_ms > 2000
        || now_ms - a.captured_at_unix_ms > 12_000
        || displacement.iter().any(|v| !v.is_finite())
        || norm(displacement) > 0.03
    {
        return Err(
            "pickup feedback has stale/reset/repeated frames or unbounded self displacement".into(),
        );
    }
    for reply in [origin, current] {
        validate(reply)?;
    }
    let difference = |key: &str| -> Result<[f64; 3], String> {
        let before = feedback_vector(origin, key)?;
        let after = feedback_vector(current, key)?;
        Ok(std::array::from_fn(|i| after[i] - before[i]))
    };
    let box_motion = difference("current_box_gravity_center_m")?;
    let palm_motion = difference("current_midpalm_gravity_center_m")?;
    let drift = norm(difference("current_box_relative_left_palm_m")?);
    let box_lift = box_motion[2] + displacement[2];
    let palm_lift = palm_motion[2] + displacement[2];
    let accepted = [drift, box_lift, palm_lift].iter().all(|v| v.is_finite())
        && drift <= 0.02
        && box_lift >= 0.025
        && palm_lift >= 0.025;
    Ok(serde_json::json!({
        "schema":"g1_station_two_paired_RGB_pickup_feedback_v1",
        "origin_observation":a,"current_observation":b,
        "box_source_center_lift_m":box_lift,"midpalm_source_lift_m":palm_lift,
        "box_relative_left_palm_displacement_m":drift,
        "minimum_observed_lift_m":0.025,"maximum_relative_displacement_m":0.02,
        "root_translation_source":"original_named_velocity_integral_during_200Tick_raise_and_settle",
        "self_velocity_displacement_source_m":displacement,"accepted":accepted,
        "hand_only_support_proven":false,"contact_and_full_geometry_require_independent_acceptance":true,
        "world_or_contact_truth_input":false,"task_qualified":false,
    }))
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("pickup owner absent")?
        .clone();
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let step = latest
        .assist_step
        .as_ref()
        .ok_or("pickup execution receipt absent")?;
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("pickup stage absent")?;
    if assist.pickup.goal.is_none() {
        let Some(holding_ticks) = station_carry::completed_hold_ticks(
            &step.execution,
            latest.timing.episode_integrations,
        )?
        else {
            return Ok(false);
        };
        let Some(job) = &assist.vision_job else {
            start_marker_job(runtime, port, "station_pickup_before")?;
            return Ok(false);
        };
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        let observation = job.observation;
        validate_feedback_pose(&reply)?;
        if observation.episode_id != runtime.episode_id
            || observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
            || reply["observation"] != serde_json::to_value(observation).unwrap()
        {
            return Err("pickup origin RGB differs from its completed grip hold".into());
        }
        let goal = MobileRaiseGoal {
            observation,
            distance_m: 0.1,
            duration_ticks: 100,
        };
        goal.validate().map_err(|e| e.to_string())?;
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("pickup lost sole native owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: observation.sim_time_ns + 10_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(12),
                command: MobileAssistCommand::ClassicalPickupRaise(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        outcome
            .0
            .lock()
            .unwrap()
            .mobile_assist_handoff
            .as_mut()
            .ok_or("pickup provenance absent")?["verified_station_pickup"] = serde_json::json!({
            "origin_actual_localization":reply,"executed_goal":goal,
            "traditional_declared_lift_m":0.1,"maximum_attempts":1,
            "fresh_paired_RGB_required_before_any_turn":true,"world_or_contact_truth_input":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.pickup.origin = Some(reply);
        assist.pickup.goal = Some(goal);
        assist.pickup.holding_ticks = Some(holding_ticks);
        runtime.requested = false;
        return Ok(false);
    }
    let submitted = assist.pickup.goal.as_ref().unwrap();
    if latest.timing.episode_integrations * 20_000_000 == submitted.observation.sim_time_ns {
        let Some(ticks) = station_carry::completed_hold_ticks(
            &step.execution,
            latest.timing.episode_integrations,
        )?
        else {
            return Err("queued pickup lost its completed original hold".into());
        };
        if Some(ticks) != assist.pickup.holding_ticks {
            return Err("queued pickup changed its grip hold".into());
        }
        return Ok(false);
    }
    if !assist
        .pickup
        .own_completed_raise(&step.execution, latest.timing.episode_integrations)
    {
        return Err("pickup lacks its own exact100raise+100settle completion".into());
    }
    let MobileAssistExecution::ClassicalRaise { raising, .. } = &step.execution else {
        unreachable!();
    };
    let Some(job) = &assist.vision_job else {
        start_marker_job(runtime, port, "station_pickup_after")?;
        return Ok(false);
    };
    let Some(reply) = job.try_take() else {
        return Ok(false);
    };
    let reply = reply?;
    if job.observation.episode_id != runtime.episode_id
        || job.observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
    {
        return Err("pickup reobservation differs from completed lift".into());
    }
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64;
    let feedback = verify(
        assist.pickup.origin.as_ref().unwrap(),
        &reply,
        raising.self_velocity_displacement_source_m,
        latest.timing.episode_integrations,
        now_ms,
    )?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .ok_or("pickup provenance absent")?["verified_station_pickup"]["feedback"] =
        serde_json::json!({"actual_current_localization":reply,"verification":feedback});
    if feedback["accepted"] != true {
        return Err("current paired RGB/self did not verify pickup; no turn admitted".into());
    }
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.vision_job.take();
    assist.pickup.accepted = true;
    assist.pickup.transport_baseline = Some(reply);
    runtime.requested = false;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (serde_json::Value, serde_json::Value) {
        let make = |tick: u64, frame, wall, lift| {
            let stamp = ObservationStamp {
                episode_id: 1,
                frame_id: frame,
                sim_time_ns: tick * 20_000_000,
                captured_at_unix_ms: wall,
            };
            serde_json::json!({"observation":stamp,"box_view_only":true,"world_or_contact_truth_input":false,
                "same_tick_box_pair_pose":{"schema":"g1_actual_rgb_same_tick_box_pair_pose_v1","observations":[stamp]},
                "current_box_gravity_center_m":[0.,0.,lift],"current_midpalm_gravity_center_m":[0.,0.,lift],
                "current_box_relative_left_palm_m":[0.,0.,0.]})
        };
        (make(300, 4, 1000, 0.), make(500, 6, 6000, 0.04))
    }
    #[test]
    fn stationary_slipping_and_palm_only_lifts_cannot_admit_a_turn() {
        let (a, b) = fixture();
        assert_eq!(
            verify(&a, &b, [0.; 3], 500, 6000).unwrap()["accepted"],
            true
        );
        for (key, value) in [
            (
                "current_box_gravity_center_m",
                serde_json::json!([0., 0., 0.]),
            ),
            (
                "current_midpalm_gravity_center_m",
                serde_json::json!([0., 0., 0.]),
            ),
            (
                "current_box_relative_left_palm_m",
                serde_json::json!([0.021, 0., 0.]),
            ),
        ] {
            let mut bad = b.clone();
            bad[key] = value;
            assert_eq!(
                verify(&a, &bad, [0.; 3], 500, 6000).unwrap()["accepted"],
                false
            );
        }
        assert!(verify(&a, &b, [f64::NAN, 0., 0.], 500, 6000).is_err());
        assert!(verify(&a, &b, [0.031, 0., 0.], 500, 6000).is_err());
    }
    #[test]
    fn old_foreign_missing_and_unpaired_rgb_cannot_verify_pickup() {
        let (a, b) = fixture();
        assert!(verify(&a, &b, [0.; 3], 499, 6000).is_err());
        assert!(verify(&a, &b, [0.; 3], 500, 8001).is_err());
        assert!(verify(&a, &b, [0.; 3], 500, 5999).is_err());
        for (pointer, value) in [
            ("/observation/episode_id", serde_json::json!(2)),
            ("/observation/frame_id", serde_json::json!(4)),
            (
                "/observation/sim_time_ns",
                serde_json::json!(501 * 20_000_000_u64),
            ),
            (
                "/same_tick_box_pair_pose/schema",
                serde_json::json!("foreign"),
            ),
            ("/world_or_contact_truth_input", serde_json::json!(true)),
            ("/current_box_relative_left_palm_m", serde_json::Value::Null),
        ] {
            let mut bad = b.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(verify(&a, &bad, [0.; 3], 500, 6000).is_err());
        }
    }

    #[test]
    fn only_own_exact_completed_raise_can_release_the_station_turn() {
        use robot_minigame::g1::contract::G1Command;
        use simulation_minigame::g1::mobile_raise::MobileRaiseStep;
        let (origin, _) = fixture();
        let goal = MobileRaiseGoal {
            observation: serde_json::from_value(origin["observation"].clone()).unwrap(),
            distance_m: 0.1,
            duration_ticks: 100,
        };
        let pickup = Pickup {
            goal: Some(goal.clone()),
            holding_ticks: Some(100),
            accepted: true,
            ..default()
        };
        let mut execution = MobileAssistExecution::ClassicalRaise {
            goal,
            raising: MobileRaiseStep {
                commanded_rise_m: 0.1,
                cartesian_increment_root_source_m: [0.; 3],
                raising_ticks: 100,
                settling_ticks: 100,
                completed: true,
                self_velocity_displacement_source_m: [0.; 3],
                correction: None,
                command: G1Command::default(),
            },
        };
        assert_eq!(pickup.completed_hold_ticks(&execution, 500).unwrap(), 100);
        assert!(pickup.completed_hold_ticks(&execution, 499).is_err());
        assert!(pickup.completed_hold_ticks(&execution, 501).is_err());
        let MobileAssistExecution::ClassicalRaise { raising, .. } = &mut execution else {
            unreachable!();
        };
        raising.completed = false;
        assert!(pickup.completed_hold_ticks(&execution, 500).is_err());
        let MobileAssistExecution::ClassicalRaise { raising, goal } = &mut execution else {
            unreachable!();
        };
        raising.completed = true;
        goal.observation.episode_id = 2;
        assert!(pickup.completed_hold_ticks(&execution, 500).is_err());
    }
}
