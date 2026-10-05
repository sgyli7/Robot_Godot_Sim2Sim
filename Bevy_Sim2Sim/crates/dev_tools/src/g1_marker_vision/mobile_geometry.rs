//! Paired-RGB geometry diagnostics. This port never grants motor admission.

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use task_minigame::types::ObservationStamp;

const HELPER: &[u8] = include_bytes!("../../python/scripts/unitree_g1_mobile_hand_geometry.py");
const VERTICES: [usize; 8] = [21392, 4761, 15221, 4761, 15221, 94, 8, 15221];
const PARTS: [&str; 8] = [
    "palm", "index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2",
];

pub(super) fn helper_hash() -> String {
    format!("{:x}", Sha256::digest(HELPER))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema: String,
    observation: ObservationStamp,
    robot_definition_sha256: String,
    task_geometry_sha256: String,
    geometry_source_sha256: String,
    source: String,
    bodies: [Body; 16],
    world_or_contact_truth_input: bool,
    rigid_point_20mm_guard_changed: bool,
    holding_proven: bool,
    physical_contact_forces_proven: bool,
    owner_execution_admitted: bool,
    task_qualified: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    body_index: usize,
    body_name: String,
    side: String,
    linf_distance_m: f64,
    certified_separation_m: f64,
    complete_original_vertex_count: usize,
    plane_normal_box: Option<[f64; 3]>,
    hand_witness_box_m: [f64; 3],
    box_witness_box_m: [f64; 3],
    physical_contact_force_measured: bool,
}

/// File hashes and both image proofs are checked by the enclosing worker.
pub(super) fn validate_reply(
    reply: &Value,
    observation: ObservationStamp,
    image_hash: &str,
    input_hash: &str,
    definition_hash: &str,
    geometry_hash: &str,
) -> Result<(), String> {
    super::mobile_held::current_relative(reply)?;
    if super::mobile_held::paired_stamp(reply)? != observation
        || reply["image_sha256"] != image_hash
        || reply["input_sha256"] != input_hash
        || reply["robot_definition_sha256"] != definition_hash
        || reply["target_memory_used"] == true
        || [
            "navigation_proposal",
            "clearance_proposal",
            "fine_approach_proposal",
            "release_proposal",
            "thumb_preparation_proposal",
            "target_memory_estimate",
            "grasp_centering_proposal",
            "public_bin_board_pose",
        ]
        .iter()
        .any(|key| !reply[*key].is_null())
    {
        return Err("hand geometry cannot admit navigation, memory or controller goals".into());
    }
    validate_report(
        &reply["held_contact_geometry"],
        observation,
        definition_hash,
        geometry_hash,
    )
}

fn validate_report(
    value: &Value,
    observation: ObservationStamp,
    definition_hash: &str,
    geometry_hash: &str,
) -> Result<(), String> {
    let report: Report = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    if report.schema != "g1_current_rgb_original_hand_separation_v1"
        || report.observation != observation
        || report.robot_definition_sha256 != definition_hash
        || report.task_geometry_sha256 != geometry_hash
        || report.geometry_source_sha256 != helper_hash()
        || report.source != "current_paired_RGB_original_self_FK_and_complete_hand_hulls"
        || report.world_or_contact_truth_input
        || report.rigid_point_20mm_guard_changed
        || report.holding_proven
        || report.physical_contact_forces_proven
        || report.owner_execution_admitted
        || report.task_qualified
    {
        return Err(
            "hand geometry has foreign identity or unproven force/holding/admission claims".into(),
        );
    }
    for (slot, body) in report.bodies.iter().enumerate() {
        let side = if slot < 8 { "left" } else { "right" };
        let part = slot % 8;
        let index = if slot < 8 { 28 + part } else { 45 + part };
        if body.body_index != index
            || body.body_name != format!("{side}_hand_{}_link", PARTS[part])
            || body.side != side
            || body.complete_original_vertex_count != VERTICES[part]
            || body.physical_contact_force_measured
            || !body.linf_distance_m.is_finite()
            || !(0. ..=8.).contains(&body.linf_distance_m)
            || !body.certified_separation_m.is_finite()
            || body.certified_separation_m < 0.
            || body.certified_separation_m > 3_f64.sqrt() * body.linf_distance_m + 2e-7
            || body
                .hand_witness_box_m
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 8.)
            || body
                .box_witness_box_m
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 0.1 + 2e-7)
            || (0..3).any(|i| {
                (body.hand_witness_box_m[i] - body.box_witness_box_m[i]).abs()
                    > body.linf_distance_m + 2e-7
            })
            || body.plane_normal_box.is_some_and(|normal| {
                normal.iter().any(|v| !v.is_finite())
                    || (normal.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.).abs() > 1e-6
            })
            || (body.linf_distance_m > 2e-7
                && (body.certified_separation_m < 1e-8 || body.plane_normal_box.is_none()))
        {
            return Err(
                "hand geometry changed original hulls or its finite distance certificate".into(),
            );
        }
    }
    Ok(())
}

pub(super) fn reject_unrequested(reply: &Value) -> Result<(), String> {
    if reply["held_contact_geometry"].is_null() {
        Ok(())
    } else {
        Err("unrequested original hand geometry".into())
    }
}

#[cfg(test)]
mod tests;

const GRASP_HELPER: &[u8] = include_bytes!("../../python/scripts/unitree_g1_mobile_grasp.py");
pub(super) fn grasp_helper_hash() -> String {
    format!("{:x}", Sha256::digest(GRASP_HELPER))
}
pub(super) fn validate_closed_regrasp(
    reply: &Value,
    observation: ObservationStamp,
    requested: bool,
) -> Result<(), String> {
    let p = &reply["closed_regrasp_proposal"];
    if !requested {
        return if p.is_null() {
            Ok(())
        } else {
            Err("unrequested closed regrasp proposal".into())
        };
    }
    let stamp: ObservationStamp =
        serde_json::from_value(p["observation"].clone()).map_err(|e| e.to_string())?;
    if stamp != observation
        || ![300, 650, 1000].contains(&(stamp.sim_time_ns / 20_000_000))
        || p["schema"] != "g1_current_rgb_original_hand_grasp_centering_v1"
        || p["closed_regrasp_only"] != true
        || p["planner_source_sha256"] != grasp_helper_hash()
        || p["uses_measured_hands_not_commanded_targets"] != true
        || p["world_or_contact_truth_input"] != false
        || p["physical_contact_forces_proven"] != false
        || p["owner_execution_admitted"] != false
        || p["task_qualified"] != false
        || p["robot_definition_sha256"] != reply["robot_definition_sha256"]
        || p["task_geometry_sha256"] != reply["held_contact_geometry"]["task_geometry_sha256"]
    {
        return Err("closed regrasp proposal changed current identity or authority".into());
    }
    Ok(())
}

pub(super) fn validate_open_regrasp(
    reply: &Value,
    observation: ObservationStamp,
    reference: Option<&std::path::Path>,
) -> Result<(), String> {
    let p = &reply["open_regrasp_forecast"];
    let Some(reference) = reference else {
        return if p.is_null() {
            Ok(())
        } else {
            Err("unrequested measured closed-reference forecast".into())
        };
    };
    let bytes = std::fs::read(reference).map_err(|e| e.to_string())?;
    if bytes.len() > 128 * 1024 {
        return Err("closed reference exceeds budget".into());
    }
    let source: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let source_stamp: ObservationStamp =
        serde_json::from_value(source["stamp"].clone()).map_err(|e| e.to_string())?;
    let stamp: ObservationStamp =
        serde_json::from_value(p["observation"].clone()).map_err(|e| e.to_string())?;
    if stamp != observation
        || ![450, 800].contains(&(stamp.sim_time_ns / 20_000_000))
        || source_stamp.episode_id != stamp.episode_id
        || source_stamp.sim_time_ns + 3_000_000_000 != stamp.sim_time_ns
        || p["closed_reference_observation"] != source["stamp"]
        || p["closed_reference_input_sha256"] != format!("{:x}", Sha256::digest(&bytes))
        || p["schema"] != "g1_current_open_rgb_measured_closed_reference_forecast_v1"
        || p["planner_source_sha256"] != grasp_helper_hash()
        || p["actual_current_closed_self"] != false
        || p["uses_own_measured_closed_reference_for_forecast"] != true
        || p["world_or_contact_truth_input"] != false
        || p["physical_contact_forces_proven"] != false
        || p["owner_execution_admitted"] != false
        || p["task_qualified"] != false
    {
        return Err("opened forecast changed current image/reference identity or authority".into());
    }
    Ok(())
}
