use super::*;
use serde_json::json;

fn fixture() -> (Value, ObservationStamp) {
    let stamp = ObservationStamp {
        episode_id: 7,
        frame_id: 9,
        sim_time_ns: 10_000_000_000,
        captured_at_unix_ms: 123,
    };
    let bodies: Vec<_> = (0..16)
        .map(|slot| {
            let side = if slot < 8 { "left" } else { "right" };
            let part = slot % 8;
            json!({"body_index":if slot<8 {28+part} else {45+part},
            "body_name":format!("{side}_hand_{}_link",PARTS[part]),"side":side,
            "linf_distance_m":0.,"certified_separation_m":0.,
            "complete_original_vertex_count":VERTICES[part],"plane_normal_box":null,
            "hand_witness_box_m":[0.,0.,0.],"box_witness_box_m":[0.,0.,0.],
            "physical_contact_force_measured":false})
        })
        .collect();
    (
        json!({"schema":"g1_current_rgb_original_hand_separation_v1","observation":stamp,
        "robot_definition_sha256":"definition","task_geometry_sha256":"geometry",
        "geometry_source_sha256":helper_hash(),
        "source":"current_paired_RGB_original_self_FK_and_complete_hand_hulls","bodies":bodies,
        "world_or_contact_truth_input":false,"rigid_point_20mm_guard_changed":false,
        "holding_proven":false,"physical_contact_forces_proven":false,
        "owner_execution_admitted":false,"task_qualified":false}),
        stamp,
    )
}

#[test]
fn geometry_cannot_assert_force_holding_or_relax_the_guard() {
    let (original, stamp) = fixture();
    assert!(validate_report(&original, stamp, "definition", "geometry").is_ok());
    for key in [
        "world_or_contact_truth_input",
        "rigid_point_20mm_guard_changed",
        "holding_proven",
        "physical_contact_forces_proven",
        "owner_execution_admitted",
        "task_qualified",
    ] {
        let mut changed = original.clone();
        changed[key] = true.into();
        assert!(
            validate_report(&changed, stamp, "definition", "geometry").is_err(),
            "{key}"
        );
    }
    let mut changed = original;
    changed["bodies"][0]["physical_contact_force_measured"] = true.into();
    assert!(validate_report(&changed, stamp, "definition", "geometry").is_err());
}

#[test]
fn reset_source_swap_and_mesh_decimation_cannot_pass() {
    let (original, stamp) = fixture();
    for (key, value) in [
        ("episode_id", 8_u64),
        ("frame_id", 10),
        ("sim_time_ns", 10_020_000_000),
    ] {
        let mut changed = original.clone();
        changed["observation"][key] = value.into();
        assert!(validate_report(&changed, stamp, "definition", "geometry").is_err());
    }
    for key in [
        "robot_definition_sha256",
        "task_geometry_sha256",
        "geometry_source_sha256",
    ] {
        let mut changed = original.clone();
        changed[key] = "foreign".into();
        assert!(validate_report(&changed, stamp, "definition", "geometry").is_err());
    }
    for (key, value) in [("body_index", 45), ("complete_original_vertex_count", 94)] {
        let mut changed = original.clone();
        changed["bodies"][0][key] = value.into();
        assert!(validate_report(&changed, stamp, "definition", "geometry").is_err());
    }
    let mut changed = original;
    changed["oracle_world_pose"] = json!([0., 0., 0.]);
    assert!(validate_report(&changed, stamp, "definition", "geometry").is_err());
}

#[test]
fn separated_hulls_need_a_finite_bounded_witness_and_plane() {
    let (mut report, stamp) = fixture();
    let body = &mut report["bodies"][0];
    body["linf_distance_m"] = 0.01.into();
    body["certified_separation_m"] = 0.01.into();
    body["plane_normal_box"] = json!([1., 0., 0.]);
    body["hand_witness_box_m"] = json!([0.11, 0., 0.]);
    body["box_witness_box_m"] = json!([0.1, 0., 0.]);
    assert!(validate_report(&report, stamp, "definition", "geometry").is_ok());
    for (key, value) in [
        ("plane_normal_box", json!(null)),
        ("plane_normal_box", json!([2., 0., 0.])),
        ("hand_witness_box_m", json!([0.12, 0., 0.])),
        ("box_witness_box_m", json!([0.2, 0., 0.])),
        ("linf_distance_m", json!(-1.)),
        ("certified_separation_m", json!(0.02)),
    ] {
        let mut changed = report.clone();
        changed["bodies"][0][key] = value;
        assert!(
            validate_report(&changed, stamp, "definition", "geometry").is_err(),
            "{key}"
        );
    }
}

#[test]
fn ordinary_requests_reject_a_diagnostic_payload() {
    assert!(reject_unrequested(&json!({})).is_ok());
    assert!(reject_unrequested(&json!({"held_contact_geometry":{}})).is_err());
}

#[test]
fn geometry_only_reply_cannot_smuggle_a_navigation_goal() {
    let (report, stamp) = fixture();
    let secondary = ObservationStamp {
        frame_id: stamp.frame_id + 1,
        captured_at_unix_ms: stamp.captured_at_unix_ms + 1,
        ..stamp
    };
    let original = json!({
        "schema":"g1_mobile_actual_marker_localization_v1","observation":stamp,
        "image_sha256":"image","input_sha256":"input","robot_definition_sha256":"definition",
        "actual_rgb_only_object_measurement":true,"world_or_contact_truth_input":false,"task_qualified":false,
        "held_contact_geometry":report,
        "same_tick_box_pair_pose":{"schema":"g1_actual_rgb_same_tick_box_pair_pose_v1",
            "marker_id":22,"observations":[stamp,secondary],"world_or_contact_truth_input":false,"task_qualified":false},
        "held_box_feedback":{"schema":"g1_current_paired_rgb_relative_grip_v1","observation":stamp,
            "current_box_relative_left_palm_m":[0.1,0.,0.],"world_or_contact_truth_input":false,"task_qualified":false}
    });
    let valid = |r: &Value| validate_reply(r, stamp, "image", "input", "definition", "geometry");
    assert!(valid(&original).is_ok());
    for key in [
        "navigation_proposal",
        "clearance_proposal",
        "release_proposal",
        "grasp_centering_proposal",
    ] {
        let mut changed = original.clone();
        changed[key] = json!({"command":"go"});
        assert!(valid(&changed).is_err(), "{key}");
    }
    for key in ["image_sha256", "input_sha256"] {
        let mut changed = original.clone();
        changed[key] = "foreign".into();
        assert!(valid(&changed).is_err(), "{key}");
    }
    let mut changed = original;
    changed["target_memory_used"] = true.into();
    assert!(valid(&changed).is_err());
}
