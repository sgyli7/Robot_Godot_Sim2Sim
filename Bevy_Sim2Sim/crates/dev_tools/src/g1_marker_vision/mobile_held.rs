//! Whitelisted current RGB/self feedback; never contact or object-world truth.

use serde::Deserialize;
use serde_json::Value;
use task_minigame::types::ObservationStamp;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldFeedback {
    schema: String,
    observation: ObservationStamp,
    current_box_relative_left_palm_m: [f64; 3],
    world_or_contact_truth_input: bool,
    task_qualified: bool,
}

pub(crate) fn vector(value: &Value) -> Result<[f64; 3], String> {
    let vector: [f64; 3] = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    if vector.iter().any(|v| !v.is_finite() || v.abs() > 8.) {
        return Err("held-box feedback vector is outside its finite bound".into());
    }
    Ok(vector)
}

/// The worker also binds every proof hash/pose to its actual two input files.
pub(crate) fn paired_stamp(reply: &Value) -> Result<ObservationStamp, String> {
    let stamp: ObservationStamp =
        serde_json::from_value(reply["observation"].clone()).map_err(|e| e.to_string())?;
    let proof = &reply["same_tick_box_pair_pose"];
    let pair: [ObservationStamp; 2] =
        serde_json::from_value(proof["observations"].clone()).map_err(|e| e.to_string())?;
    if reply["schema"] != "g1_mobile_actual_marker_localization_v1"
        || reply["actual_rgb_only_object_measurement"] != true
        || reply["world_or_contact_truth_input"] != false
        || reply["task_qualified"] != false
        || proof["schema"] != "g1_actual_rgb_same_tick_box_pair_pose_v1"
        || proof["marker_id"] != 22
        || proof["world_or_contact_truth_input"] != false
        || proof["task_qualified"] != false
        || pair[0] != stamp
        || stamp.episode_id == 0
        || stamp.frame_id == 0
        || stamp.sim_time_ns == 0
        || stamp.sim_time_ns % 20_000_000 != 0
        || stamp.captured_at_unix_ms == 0
        || pair[1].episode_id != stamp.episode_id
        || pair[1].sim_time_ns != stamp.sim_time_ns
        || pair[1].frame_id <= stamp.frame_id
        || pair[1].captured_at_unix_ms <= stamp.captured_at_unix_ms
        || pair[1].captured_at_unix_ms - stamp.captured_at_unix_ms > 2000
    {
        return Err("held-box feedback lacks its current same-Tick RGB pair".into());
    }
    Ok(stamp)
}

pub(crate) fn current_relative(reply: &Value) -> Result<[f64; 3], String> {
    let stamp = paired_stamp(reply)?;
    let feedback: HeldFeedback =
        serde_json::from_value(reply["held_box_feedback"].clone()).map_err(|e| e.to_string())?;
    if feedback.schema != "g1_current_paired_rgb_relative_grip_v1"
        || feedback.observation != stamp
        || feedback.world_or_contact_truth_input
        || feedback.task_qualified
        || reply["box_view_only"] == true
        || reply["placement_view_only"] == true
        || reply["target_memory_used"] == true
    {
        return Err("held-box feedback has foreign identity, memory or mode".into());
    }
    vector(&serde_json::json!(
        feedback.current_box_relative_left_palm_m
    ))
}

pub(crate) fn validate_requested_feedback(
    reply: &Value,
    observation: ObservationStamp,
    requested: bool,
) -> Result<(), String> {
    if requested {
        current_relative(reply)?;
        if paired_stamp(reply)? != observation {
            return Err("held-box feedback differs from the worker's requested frame".into());
        }
    } else if !reply["held_box_feedback"].is_null() {
        return Err("unrequested held-box feedback".into());
    }
    Ok(())
}
