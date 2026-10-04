//! Reject another carry when current paired RGB drifts from accepted pickup.
//! This conservative point-estimate guard does not prove grip or task success.

use super::*;
use crate::g1_marker_vision::mobile_held;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

fn evaluate(
    baseline: &Value,
    current: &Value,
    previous_carry: Option<ObservationStamp>,
    native_tick: u64,
    now_ms: u64,
    max_current_age_ms: u64,
) -> Result<Value, String> {
    let a = mobile_held::paired_stamp(baseline)?;
    let b = mobile_held::paired_stamp(current)?;
    let secondary_wall =
        current["same_tick_box_pair_pose"]["observations"][1]["captured_at_unix_ms"]
            .as_u64()
            .ok_or("held pair wall time absent")?;
    if baseline["box_view_only"] != true
        || !baseline["navigation_proposal"].is_null()
        || !baseline["clearance_proposal"].is_null()
        || a.episode_id != b.episode_id
        || b.frame_id <= a.frame_id
        || b.sim_time_ns <= a.sim_time_ns
        || b.sim_time_ns
            != native_tick
                .checked_mul(20_000_000)
                .ok_or("held tick overflow")?
        || b.sim_time_ns - a.sim_time_ns > 70_000_000_000
        || b.captured_at_unix_ms <= a.captured_at_unix_ms
        || secondary_wall > now_ms
        || now_ms - b.captured_at_unix_ms > max_current_age_ms
        || now_ms - a.captured_at_unix_ms > 120_000
        || !matches!(max_current_age_ms, 2000 | 20_000)
        || previous_carry.is_some_and(|p| {
            p.episode_id != b.episode_id
                || b.frame_id <= p.frame_id
                || b.sim_time_ns <= p.sim_time_ns
        })
    {
        return Err(
            "held-box feedback is expired, reset, repeated or not the current owner boundary"
                .into(),
        );
    }
    let origin = mobile_held::vector(&baseline["current_box_relative_left_palm_m"])?;
    let present = mobile_held::current_relative(current)?;
    let drift = origin
        .iter()
        .zip(present)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(serde_json::json!({
        "schema":"g1_current_paired_rgb_carry_grip_guard_v1",
        "pickup_observation":a,"current_observation":b,
        "box_relative_left_palm_displacement_m":drift,
        "maximum_relative_displacement_m":0.02,"accepted":drift <= 0.02,
        "maximum_current_wall_age_ms":max_current_age_ms,
        "hand_only_support_proven":false,"measurement_uncertainty_bound_proven":false,
        "world_or_contact_truth_input":false,"task_qualified":false,
    }))
}

pub(super) fn check(
    runtime: &CaptureRuntime,
    outcome: &CaptureOutcome,
    current: &Value,
    phase: &str,
    max_current_age_ms: u64,
) -> Result<(), String> {
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("held-box stage absent")?;
    if !assist.pickup.completed() {
        return Ok(());
    }
    let baseline = assist
        .pickup
        .transport_baseline()
        .ok_or("accepted pickup lost its paired RGB baseline")?;
    let latest = runtime.latest.as_ref().ok_or("held-box owner absent")?;
    if latest.phase != G1WorkerPhase::Paused || latest.episode_id != runtime.episode_id {
        return Err("held-box feedback requires its own paused native owner".into());
    }
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64;
    let mut receipt = outcome.0.lock().map_err(|_| "held-box receipt poisoned")?;
    let handoff = receipt
        .mobile_assist_handoff
        .as_mut()
        .ok_or("held-box provenance absent")?;
    let records = handoff
        .as_object_mut()
        .ok_or("held-box provenance not an object")?
        .entry("held_box_feedback_checks")
        .or_insert_with(|| serde_json::json!([]))
        .as_array_mut()
        .ok_or("held-box provenance not an array")?;
    if records.len() >= 16 {
        return Err("held-box feedback exhausted its bounded checks".into());
    }
    let previous = records
        .iter()
        .rev()
        .find(|r| r["carry_admission_boundary"] == true && r["feedback"]["accepted"] == true)
        .map(|r| {
            serde_json::from_value(r["feedback"]["current_observation"].clone())
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    let feedback = evaluate(
        baseline,
        current,
        previous,
        latest.timing.episode_integrations,
        now_ms,
        max_current_age_ms,
    );
    let record = match &feedback {
        Ok(feedback) => {
            serde_json::json!({"phase":phase,"carry_admission_boundary":phase != "before_qwen_request",
            "actual_current_localization":current,"feedback":feedback,"task_qualified":false})
        }
        Err(reason) => {
            serde_json::json!({"phase":phase,"carry_admission_boundary":false,"accepted":false,"reason":reason,"task_qualified":false})
        }
    };
    records.push(record);
    if feedback?["accepted"] != true {
        return Err(
            "current paired RGB exceeded pickup-relative slip limit; no further carry admitted"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(tick: u64, frame: u64, wall: u64, box_only: bool) -> Value {
        let stamp = ObservationStamp {
            episode_id: 1,
            frame_id: frame,
            sim_time_ns: tick * 20_000_000,
            captured_at_unix_ms: wall,
        };
        let second = ObservationStamp {
            frame_id: frame + 1,
            captured_at_unix_ms: wall + 10,
            ..stamp
        };
        let mut reply = serde_json::json!({"schema":"g1_mobile_actual_marker_localization_v1","observation":stamp,
            "actual_rgb_only_object_measurement":true,"world_or_contact_truth_input":false,"task_qualified":false,
            "same_tick_box_pair_pose":{"schema":"g1_actual_rgb_same_tick_box_pair_pose_v1","marker_id":22,
            "observations":[stamp,second],"world_or_contact_truth_input":false,"task_qualified":false}});
        if box_only {
            reply["box_view_only"] = serde_json::json!(true);
            reply["current_box_relative_left_palm_m"] = serde_json::json!([0., 0., 0.]);
        } else {
            reply["held_box_feedback"] = serde_json::json!({"schema":"g1_current_paired_rgb_relative_grip_v1","observation":stamp,
            "current_box_relative_left_palm_m":[0.,0.,0.],"world_or_contact_truth_input":false,"task_qualified":false});
        }
        reply
    }
    #[test]
    fn fresh_small_drift_and_original_limit_are_distinguished_before_carry() {
        let a = fixture(500, 6, 1000, true);
        for (drift, accepted) in [
            (0., true),
            (0.019, true),
            (0.02, true),
            (0.02006455409322831, false),
            (0.04407636539212186, false),
        ] {
            let mut b = fixture(1002, 10, 12_000, false);
            b["held_box_feedback"]["current_box_relative_left_palm_m"] =
                serde_json::json!([drift, 0., 0.]);
            assert_eq!(
                evaluate(&a, &b, None, 1002, 12_010, 2000).unwrap()["accepted"],
                accepted
            );
        }
    }
    #[test]
    fn reset_stale_future_repeated_and_foreign_feedback_cannot_admit_carry() {
        let a = fixture(500, 6, 1000, true);
        let b = fixture(1002, 10, 12_000, false);
        for (tick, now, age) in [
            (1001, 12_010, 2000),
            (1002, 11_999, 2000),
            (1002, 14_001, 2000),
            (1002, 32_001, 20_000),
            (1002, 121_001, 20_000),
            (1002, 12_010, 2001),
        ] {
            assert!(evaluate(&a, &b, None, tick, now, age).is_err());
        }
        let previous = serde_json::from_value(b["observation"].clone()).unwrap();
        assert!(evaluate(&a, &b, Some(previous), 1002, 12_010, 2000).is_err());
        for (pointer, value) in [
            ("/observation/episode_id", serde_json::json!(2)),
            (
                "/same_tick_box_pair_pose/observations/1/sim_time_ns",
                serde_json::json!(1),
            ),
            (
                "/same_tick_box_pair_pose/task_qualified",
                serde_json::json!(true),
            ),
            (
                "/held_box_feedback/world_or_contact_truth_input",
                serde_json::json!(true),
            ),
            (
                "/held_box_feedback/observation/frame_id",
                serde_json::json!(999),
            ),
            (
                "/held_box_feedback/current_box_relative_left_palm_m",
                serde_json::Value::Null,
            ),
            (
                "/held_box_feedback/current_box_relative_left_palm_m",
                serde_json::json!([9., 0., 0.]),
            ),
        ] {
            let mut bad = b.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(
                evaluate(&a, &bad, None, 1002, 12_010, 2000).is_err(),
                "{pointer}"
            );
        }
        let mut bad = b.clone();
        bad["held_box_feedback"]["foreign_truth"] = serde_json::json!([1, 2, 3]);
        assert!(evaluate(&a, &bad, None, 1002, 12_010, 2000).is_err());
        assert!(mobile_held::vector(&serde_json::json!([f64::NAN, 0., 0.])).is_err());
        assert!(mobile_held::validate_requested_feedback(&b, previous, false).is_err());
    }
    #[test]
    #[ignore = "requires saved actual worker replies; CPU only, no physics/models or restamping"]
    fn actual_rgb_replay_cannot_admit_slipped_carry() -> Result<(), String> {
        let path = std::env::var("G1_HELD_FEEDBACK_FIXTURE").map_err(|e| e.to_string())?;
        let input: Value = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let read = |key: &str| -> Result<Value, String> {
            serde_json::from_slice(
                &fs::read(input[key].as_str().ok_or("reply path absent")?)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())
        };
        let baseline = read("baseline")?;
        let current = read("current")?;
        let now = current["same_tick_box_pair_pose"]["observations"][1]["captured_at_unix_ms"]
            .as_u64()
            .ok_or("time absent")?
            + 100;
        let result = evaluate(
            &baseline,
            &current,
            None,
            input["native_tick"].as_u64().ok_or("tick absent")?,
            now,
            2000,
        )?;
        if result["accepted"] != input["expected_accepted"] {
            return Err("actual RGB carry rejection differs".into());
        }
        fs::write(
            input["output"].as_str().ok_or("output absent")?,
            serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}
