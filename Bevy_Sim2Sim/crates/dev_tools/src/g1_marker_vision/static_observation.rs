//! Static RGB/self schema shared by transport and passive paired capture.
use rendering_minigame::g1_camera::{G1CameraJointState, G1CameraMountProfile, G1CaptureStamp};
use serde_json::{Value, json};
use task_minigame::types::ObservationStamp;

pub(crate) fn static_observation(stamp: &G1CaptureStamp) -> Result<Value, String> {
    let state = stamp
        .native_state
        .as_ref()
        .ok_or("static RGB self state absent")?;
    state.validate()?;
    if state.body_frame.episode_id != stamp.episode_id
        || state.body_frame.source_tick * 20_000_000 != stamp.sim_time_ns
        || stamp.source_ticks != [state.body_frame.source_tick; 2]
    {
        return Err("static RGB and self frame identities differ".into());
    }
    Ok(json!({
        "schema":"g1_static_marker_observation_v1",
        "stamp":ObservationStamp {episode_id:stamp.episode_id,frame_id:stamp.capture_sequence,
            sim_time_ns:stamp.sim_time_ns,captured_at_unix_ms:stamp.captured_at_unix_ms},
        "camera":{"fx":458.1245526,"fy":458.1245526,"cx":320.,"cy":240.,"near_m":0.1,"far_m":5.,
            "vertical_fov_radians":2. * (240_f64 / 458.1245526).atan(),"mount_profile":stamp.mount_profile},
        "measured_joints":state.measured_joints,
    }))
}

/// Compare in the original f32 sensor representation, with no tolerance.
/// Parsing serialized JSON into f64 can differ by a decimal conversion ULP.
pub(crate) fn validate_fixed_pair(first: &Value, second: &Value) -> Result<(), String> {
    let a: G1CameraJointState =
        serde_json::from_value(first["measured_joints"].clone()).map_err(|e| e.to_string())?;
    let b: G1CameraJointState =
        serde_json::from_value(second["measured_joints"].clone()).map_err(|e| e.to_string())?;
    let x: ObservationStamp =
        serde_json::from_value(first["stamp"].clone()).map_err(|e| e.to_string())?;
    let y: ObservationStamp =
        serde_json::from_value(second["stamp"].clone()).map_err(|e| e.to_string())?;
    let first_mount: G1CameraMountProfile =
        serde_json::from_value(first["camera"]["mount_profile"].clone())
            .map_err(|e| e.to_string())?;
    let second_mount: G1CameraMountProfile =
        serde_json::from_value(second["camera"]["mount_profile"].clone())
            .map_err(|e| e.to_string())?;
    if first["schema"] != "g1_static_marker_observation_v1"
        || second["schema"] != first["schema"]
        || x.episode_id == 0
        || x.episode_id != y.episode_id
        || x.sim_time_ns != y.sim_time_ns
        || x.frame_id >= y.frame_id
        || x.captured_at_unix_ms >= y.captured_at_unix_ms
        || first_mount != G1CameraMountProfile::StaticPlacementOverview
        || second_mount != G1CameraMountProfile::ArenaEgo
        || a.positions != b.positions
        || a.velocities != b.velocities
        || a.root_rotation_wxyz != b.root_rotation_wxyz
        || a.root_angular_velocity_body != b.root_angular_velocity_body
        || a.root_velocity_source != b.root_velocity_source
    {
        return Err(
            "fixed camera pair changed its native Tick/self state or mount identity".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pair() -> (Value, Value) {
        let sensors = G1CameraJointState {
            positions: vec![-0.25085914; 43],
            velocities: vec![0.0034756537; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
            root_velocity_source: [0.; 3],
        };
        let mut a = json!({"schema":"g1_static_marker_observation_v1","stamp":{"episode_id":7,"frame_id":3,"sim_time_ns":6920000000_u64,"captured_at_unix_ms":10000},"camera":{"mount_profile":"static_placement_overview"},"measured_joints":sensors});
        // The first image has crossed the actual JSON file boundary.
        a = serde_json::from_slice(&serde_json::to_vec_pretty(&a).unwrap()).unwrap();
        let b = json!({"schema":"g1_static_marker_observation_v1","stamp":{"episode_id":7,"frame_id":4,"sim_time_ns":6920000000_u64,"captured_at_unix_ms":10010},"camera":{"mount_profile":"arena_ego"},"measured_joints":sensors});
        (a, b)
    }
    #[test]
    fn preserves_original_sensor_bits_across_file_boundary() {
        let (a, b) = pair();
        assert!(validate_fixed_pair(&a, &b).is_ok());
    }
    #[test]
    fn rejects_one_sensor_ulp_and_foreign_time_schema_mount() {
        let (a, b) = pair();
        let mut changed = b.clone();
        let v = -0.25085914_f32;
        changed["measured_joints"]["positions"][0] = json!(f32::from_bits(v.to_bits() + 1));
        assert!(validate_fixed_pair(&a, &changed).is_err());
        for (key, value) in [
            ("episode_id", 8_u64),
            ("sim_time_ns", 6940000000),
            ("frame_id", 3),
            ("captured_at_unix_ms", 10000),
        ] {
            let mut changed = b.clone();
            changed["stamp"][key] = json!(value);
            assert!(validate_fixed_pair(&a, &changed).is_err());
        }
        let mut changed = b.clone();
        changed["schema"] = "g1_mobile_marker_observation_v1".into();
        assert!(validate_fixed_pair(&a, &changed).is_err());
        let mut changed = b;
        changed["camera"]["mount_profile"] = "static_placement_overview".into();
        assert!(validate_fixed_pair(&a, &changed).is_err());
    }
}
