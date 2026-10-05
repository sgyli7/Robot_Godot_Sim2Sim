//! Source binding for the explicit same-Tick auxiliary RGB box-pose experiment.
use rendering_minigame::g1_camera::G1CameraJointState;
use serde_json::Value;
use std::{fs, path::Path};
use task_minigame::types::ObservationStamp;

pub(super) struct SecondaryBinding {
    observation: ObservationStamp,
    profile: String,
    image_sha256: String,
    input_sha256: String,
}

/// Compare exact native sensor bits; reject reset, motion and stale images.
pub(crate) fn validate_mobile_pair_inputs(first: &Value, second: &Value) -> Result<(), String> {
    let keys = [
        "schema",
        "stamp",
        "camera",
        "measured_joints",
        "camera_mount_profile",
    ];
    for input in [first, second] {
        let map = input
            .as_object()
            .ok_or("box pair observation is not an object")?;
        if map.len() != keys.len()
            || keys.iter().any(|key| !map.contains_key(*key))
            || input["schema"] != "g1_mobile_marker_observation_v1"
        {
            return Err("box pair has foreign observation fields/schema".into());
        }
    }
    let a: G1CameraJointState =
        serde_json::from_value(first["measured_joints"].clone()).map_err(|e| e.to_string())?;
    let b: G1CameraJointState =
        serde_json::from_value(second["measured_joints"].clone()).map_err(|e| e.to_string())?;
    let x: ObservationStamp =
        serde_json::from_value(first["stamp"].clone()).map_err(|e| e.to_string())?;
    let y: ObservationStamp =
        serde_json::from_value(second["stamp"].clone()).map_err(|e| e.to_string())?;
    let profiles = (
        first["camera_mount_profile"].as_str(),
        second["camera_mount_profile"].as_str(),
    );
    let sensor_equal = |a: &[f32], b: &[f32]| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
    };
    if !matches!(
        profiles,
        (
            Some("auxiliary_grip_overview"),
            Some("auxiliary_bin_placement")
        ) | (
            Some("auxiliary_bin_placement"),
            Some("auxiliary_grip_overview")
        ) | (
            Some("auxiliary_pregrasp_overview"),
            Some("auxiliary_bin_placement")
        ) | (
            Some("auxiliary_bin_placement"),
            Some("auxiliary_pregrasp_overview")
        )
    ) || first["camera"] != second["camera"]
        || x.episode_id == 0
        || x.episode_id != y.episode_id
        || x.sim_time_ns == 0
        || x.sim_time_ns != y.sim_time_ns
        || x.sim_time_ns % 20_000_000 != 0
        || x.frame_id == 0
        || x.frame_id >= y.frame_id
        || x.captured_at_unix_ms == 0
        || y.captured_at_unix_ms <= x.captured_at_unix_ms
        || y.captured_at_unix_ms - x.captured_at_unix_ms > 2000
        || a.positions.len() != 43
        || a.velocities.len() != 43
        || !sensor_equal(&a.positions, &b.positions)
        || !sensor_equal(&a.velocities, &b.velocities)
        || !sensor_equal(&a.root_rotation_wxyz, &b.root_rotation_wxyz)
        || !sensor_equal(&a.root_angular_velocity_body, &b.root_angular_velocity_body)
        || !sensor_equal(&a.root_velocity_source, &b.root_velocity_source)
    {
        return Err(
            "box pair changed its native Tick/self bits, camera mount or capture budget".into(),
        );
    }
    Ok(())
}

pub(super) fn read_secondary(directory: &Path, first: &Value) -> Result<SecondaryBinding, String> {
    let secondary = directory.join("secondary");
    let image = secondary.join("ego.png");
    let input = secondary.join("observation.json");
    for path in [&secondary, &image, &input] {
        if fs::symlink_metadata(path)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("box pair input cannot follow a symlink".into());
        }
    }
    let image_sha256 = super::digest(&image, 16 * 1024 * 1024)?;
    let input_sha256 = super::digest(&input, 128 * 1024)?;
    let second: Value = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    validate_mobile_pair_inputs(first, &second)?;
    Ok(SecondaryBinding {
        observation: serde_json::from_value(second["stamp"].clone()).map_err(|e| e.to_string())?,
        profile: second["camera_mount_profile"].as_str().unwrap().to_owned(),
        image_sha256,
        input_sha256,
    })
}

pub(super) fn validate_reply(
    reply: &Value,
    first: &Value,
    binding: Option<&SecondaryBinding>,
) -> Result<(), String> {
    let proof = &reply["same_tick_box_pair_pose"];
    let Some(binding) = binding else {
        return if proof.is_null() {
            Ok(())
        } else {
            Err("unrequested box pair pose".into())
        };
    };
    let errors: [f64; 2] = serde_json::from_value(proof["per_view_reprojection_rms_px"].clone())
        .map_err(|e| e.to_string())?;
    let edges: [f64; 2] =
        serde_json::from_value(proof["minimum_edges_px"].clone()).map_err(|e| e.to_string())?;
    let corners: [[[f64; 2]; 4]; 2] =
        serde_json::from_value(proof["corners_px"].clone()).map_err(|e| e.to_string())?;
    let pose: [[f64; 4]; 4] =
        serde_json::from_value(proof["root_from_marker"].clone()).map_err(|e| e.to_string())?;
    let evaluations: Vec<u64> = serde_json::from_value(proof["candidate_fit_evaluations"].clone())
        .map_err(|e| e.to_string())?;
    let detection = reply["detections"]
        .as_array()
        .and_then(|ds| ds.iter().find(|d| d["marker_id"] == 22))
        .ok_or("box pair current canonical22 absent")?;
    if proof["schema"] != "g1_actual_rgb_same_tick_box_pair_pose_v1"
        || proof["marker_id"] != 22
        || [
            "observations",
            "image_sha256",
            "input_sha256",
            "camera_mount_profiles",
        ]
        .iter()
        .any(|key| {
            proof[*key]
                .as_array()
                .is_none_or(|values| values.len() != 2)
        })
        || proof["observations"][0] != first["stamp"]
        || proof["observations"][1]
            != serde_json::to_value(binding.observation).map_err(|e| e.to_string())?
        || proof["image_sha256"][0] != reply["image_sha256"]
        || proof["input_sha256"][0] != reply["input_sha256"]
        || proof["image_sha256"][1] != binding.image_sha256
        || proof["input_sha256"][1] != binding.input_sha256
        || proof["camera_mount_profiles"][0] != first["camera_mount_profile"]
        || proof["camera_mount_profiles"][1] != binding.profile
        || proof["camera_baseline_m"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || (v - 0.08).abs() > 1e-7)
        || proof["solver"] != "two_IPPE_seeds_bounded50_joint_camera_fit"
        || proof["upright_prior_used"] != false
        || proof["world_or_contact_truth_input"] != false
        || proof["task_qualified"] != false
        || evaluations.is_empty()
        || evaluations.len() > 2
        || evaluations.iter().any(|v| !(1..=50).contains(v))
        || proof["selected_candidate"]
            .as_u64()
            .is_none_or(|v| v as usize >= evaluations.len())
        || errors
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
        || edges.iter().any(|v| !v.is_finite() || *v < 8.)
        || pose.iter().flatten().any(|v| !v.is_finite())
        || pose[3] != [0., 0., 0., 1.]
        || detection["root_from_marker"] != proof["root_from_marker"]
        || detection["corners_px"] != proof["corners_px"][0]
        || detection["reprojection_rms_px"].as_f64() != Some(errors[0])
        || detection["minimum_edge_px"].as_f64() != Some(edges[0])
    {
        return Err("box pair has foreign source, quality, budget or canonical pose proof".into());
    }
    for view in 0..2 {
        if corners[view].iter().any(|p| {
            p.iter().any(|v| !v.is_finite())
                || p[0] < 0.
                || p[0] >= 640.
                || p[1] < 0.
                || p[1] >= 480.
        }) {
            return Err("box pair corners outside current image".into());
        }
        let edge = (0..4)
            .map(|j| {
                let a = corners[view][j];
                let b = corners[view][(j + 1) % 4];
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        if (edge - edges[view]).abs() > 1e-7 {
            return Err("box pair edge proof mismatch".into());
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let value: f64 = (0..3).map(|k| pose[k][i] * pose[k][j]).sum();
            if (value - if i == j { 1. } else { 0. }).abs() > 1e-6 {
                return Err("box pair pose is not a rigid rotation".into());
            }
        }
    }
    let determinant = pose[0][0] * (pose[1][1] * pose[2][2] - pose[1][2] * pose[2][1])
        - pose[0][1] * (pose[1][0] * pose[2][2] - pose[1][2] * pose[2][0])
        + pose[0][2] * (pose[1][0] * pose[2][1] - pose[1][1] * pose[2][0]);
    if (determinant - 1.).abs() > 1e-6 {
        return Err("box pair cannot contain a reflected rotation".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pair_proof_binds_both_images_and_rejects_foreign_quality_budget_pose() {
        let binding = SecondaryBinding {
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 4,
                sim_time_ns: 20_000_000,
                captured_at_unix_ms: 10010,
            },
            profile: "auxiliary_bin_placement".into(),
            image_sha256: "b".repeat(64),
            input_sha256: "c".repeat(64),
        };
        let first = json!({"camera_mount_profile":"auxiliary_grip_overview", "stamp":{
            "episode_id":7,"frame_id":3,"sim_time_ns":20000000,"captured_at_unix_ms":10000}});
        let pose = json!([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.]
        ]);
        let corners = json!([[100., 100.], [110., 100.], [110., 110.], [100., 110.]]);
        let proof = json!({"schema":"g1_actual_rgb_same_tick_box_pair_pose_v1", "marker_id":22,
            "observations":[first["stamp"],binding.observation], "image_sha256":["a".repeat(64),binding.image_sha256],
            "input_sha256":["d".repeat(64),binding.input_sha256], "camera_mount_profiles":["auxiliary_grip_overview",binding.profile],
            "corners_px":[corners,corners], "minimum_edges_px":[10.,10.], "per_view_reprojection_rms_px":[0.5,0.5],
            "root_from_marker":pose, "camera_baseline_m":0.08, "candidate_fit_evaluations":[5,5], "selected_candidate":1,
            "solver":"two_IPPE_seeds_bounded50_joint_camera_fit", "upright_prior_used":false,
            "world_or_contact_truth_input":false,"task_qualified":false});
        let reply = json!({"same_tick_box_pair_pose":proof, "image_sha256":"a".repeat(64),"input_sha256":"d".repeat(64),
            "detections":[{"marker_id":22,"root_from_marker":pose,"corners_px":corners,
                "minimum_edge_px":10.,"reprojection_rms_px":0.5}]});
        assert!(validate_reply(&reply, &first, Some(&binding)).is_ok());
        for mutation in 0..8 {
            let mut bad = reply.clone();
            match mutation {
                0 => bad["same_tick_box_pair_pose"]["observations"][1]["episode_id"] = 8.into(),
                1 => bad["same_tick_box_pair_pose"]["image_sha256"][1] = "a".repeat(64).into(),
                2 => bad["same_tick_box_pair_pose"]["minimum_edges_px"][1] = 7.9.into(),
                3 => bad["same_tick_box_pair_pose"]["per_view_reprojection_rms_px"][1] = 1.1.into(),
                4 => bad["same_tick_box_pair_pose"]["candidate_fit_evaluations"][1] = 51.into(),
                5 => bad["same_tick_box_pair_pose"]["root_from_marker"][0][3] = 0.1.into(),
                6 => bad["same_tick_box_pair_pose"]["upright_prior_used"] = true.into(),
                7 => bad["same_tick_box_pair_pose"]["corners_px"][1][0][0] = 101.0.into(),
                _ => unreachable!(),
            }
            assert!(validate_reply(&bad, &first, Some(&binding)).is_err());
        }
    }

    #[test]
    fn rejects_reset_stale_camera_foreign_fields_and_one_sensor_ulp() {
        let a = json!({"schema":"g1_mobile_marker_observation_v1", "camera":{},
            "camera_mount_profile":"auxiliary_grip_overview", "stamp":{"episode_id":7,"frame_id":3,
            "sim_time_ns":20000000,"captured_at_unix_ms":10000}, "measured_joints": G1CameraJointState {
                positions: vec![-0.25085914;43], velocities: vec![0.;43], root_rotation_wxyz:[1.,0.,0.,0.],
                root_angular_velocity_body:[0.;3], root_velocity_source:[0.;3]}});
        let a: Value = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        let mut b = a.clone();
        b["camera_mount_profile"] = "auxiliary_bin_placement".into();
        b["stamp"]["frame_id"] = 4.into();
        b["stamp"]["captured_at_unix_ms"] = 10010.into();
        assert!(validate_mobile_pair_inputs(&a, &b).is_ok());
        let mut pregrasp = a.clone();
        pregrasp["camera_mount_profile"] = "auxiliary_pregrasp_overview".into();
        assert!(validate_mobile_pair_inputs(&pregrasp, &b).is_ok());
        let mut duplicate = b.clone();
        duplicate["camera_mount_profile"] = "auxiliary_pregrasp_overview".into();
        assert!(validate_mobile_pair_inputs(&pregrasp, &duplicate).is_err());
        for (key, v) in [
            ("episode_id", 8),
            ("frame_id", 3),
            ("sim_time_ns", 40000000),
            ("captured_at_unix_ms", 12001),
        ] {
            let mut c = b.clone();
            c["stamp"][key] = v.into();
            assert!(validate_mobile_pair_inputs(&a, &c).is_err());
        }
        let mut c = b.clone();
        c["camera_mount_profile"] = "auxiliary_grip_overview".into();
        assert!(validate_mobile_pair_inputs(&a, &c).is_err());
        let mut c = b.clone();
        c["measured_joints"]["positions"][0] =
            json!(f32::from_bits((-0.25085914_f32).to_bits() + 1));
        assert!(validate_mobile_pair_inputs(&a, &c).is_err());
        b["box_pose"] = json!([0., 0., 0.]);
        assert!(validate_mobile_pair_inputs(&a, &b).is_err());
        assert!(validate_reply(&json!({"same_tick_box_pair_pose":{}}), &a, None).is_err());
    }
}
