//! Two real RGB poses and named self sensors verify the finite lift.
//! No contact, object body pose, or independent acceptance result is input.
use rapier3d::na::{Quaternion, UnitQuaternion, Vector3};
use serde_json::{Value, json};
use simulation_minigame::g1::{runner::G1Measurement, static_transfer::StaticLeftPalmKinematics};
use task_minigame::types::ObservationStamp;

pub(super) fn verify(
    origin: &Value,
    current: &Value,
    samples: &[G1Measurement],
    kinematics: &StaticLeftPalmKinematics,
    now_ms: u64,
) -> Result<Value, String> {
    let a: ObservationStamp =
        serde_json::from_value(origin["observation"].clone()).map_err(|e| e.to_string())?;
    let b: ObservationStamp =
        serde_json::from_value(current["observation"].clone()).map_err(|e| e.to_string())?;
    validate_stamps(a, b, now_ms)?;
    if samples.len() != 51
        || samples.iter().enumerate().any(|(i, s)| {
            s.episode_id != a.episode_id
                || s.source_tick != 140 + i as u64
                || s.sim_time_ns != s.source_tick * 20_000_000
                || s.root_velocity_source.iter().any(|v| !v.is_finite())
        })
    {
        return Err("grip verification self stream has gap/reset/nonfinite velocity".into());
    }
    let first = &samples[0];
    let last = &samples[50];
    let apple_a = apple_position(origin)?;
    let apple_b = apple_position(current)?;
    let palm_a = kinematics
        .measured_left_palm(first)
        .map_err(|e| e.to_string())?;
    let palm_b = kinematics
        .measured_left_palm(last)
        .map_err(|e| e.to_string())?;
    let relative_a = palm_a.inverse_transform_point(&apple_a.into());
    let relative_b = palm_b.inverse_transform_point(&apple_b.into());
    let drift = (relative_b - relative_a).norm();
    let rotation = |s: &G1Measurement| -> Result<UnitQuaternion<f64>, String> {
        let [w, x, y, z] = s.root_rotation_wxyz.map(f64::from);
        if ![w, x, y, z].iter().all(|x| x.is_finite())
            || (w * w + x * x + y * y + z * z - 1.).abs() > 1e-4
        {
            return Err("grip verification IMU is nonfinite or not normalized".into());
        }
        Ok(UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)))
    };
    let ra = rotation(first)?;
    let rb = rotation(last)?;
    let translation: Vector3<f64> = samples[1..]
        .iter()
        .map(|s| Vector3::from(s.root_velocity_source.map(f64::from)) * 0.02)
        .sum();
    let apple_lift = (rb * apple_b - ra * apple_a + translation).z;
    let palm_lift =
        (rb * palm_b.translation.vector - ra * palm_a.translation.vector + translation).z;
    let accepted = movement_admitted(drift, apple_lift, palm_lift);
    Ok(
        json!({"schema":"g1_static_actual_two_rgb_lift_verification_v1",
        "origin_observation":a,"current_observation":b,"actual_self_samples":51,
        "apple_relative_palm_displacement_m":drift,"apple_source_lift_m":apple_lift,
        "palm_source_lift_m":palm_lift,"maximum_relative_displacement_m":0.02,
        "minimum_lift_m":0.025,"accepted":accepted,
        "root_translation_source":"integrated_named_self_velocity_50Ticks",
        "world_or_contact_truth_input":false,"task_qualified":false}),
    )
}

fn validate_stamps(a: ObservationStamp, b: ObservationStamp, now: u64) -> Result<(), String> {
    if a.episode_id == 0
        || a.episode_id != b.episode_id
        || a.frame_id == 0
        || b.frame_id <= a.frame_id
        || a.sim_time_ns != 140 * 20_000_000
        || b.sim_time_ns != 190 * 20_000_000
        || a.captured_at_unix_ms == 0
        || b.captured_at_unix_ms <= a.captured_at_unix_ms
        || b.captured_at_unix_ms > now
        || now - b.captured_at_unix_ms > 2000
        || now - a.captured_at_unix_ms > 8000
    {
        return Err("grip verification images are stale, repeated or from another episode".into());
    }
    Ok(())
}

fn apple_position(localization: &Value) -> Result<Vector3<f64>, String> {
    if localization["world_or_contact_truth_input"] != false {
        return Err("grip verification requires actual RGB localization".into());
    }
    let targets = localization["detections"]
        .as_array()
        .ok_or("grip targets absent")?;
    let target = targets
        .iter()
        .find(|d| d["marker_id"] == 31 && d["object_kind"] == "t1_apple")
        .ok_or("actual grip image did not detect apple31")?;
    let matrix: [[f64; 4]; 4] =
        serde_json::from_value(target["root_from_object"].clone()).map_err(|e| e.to_string())?;
    if !matrix.iter().flatten().all(|v| v.is_finite()) || matrix[3] != [0., 0., 0., 1.] {
        return Err("grip RGB pose is invalid".into());
    }
    Ok(Vector3::new(matrix[0][3], matrix[1][3], matrix[2][3]))
}

fn movement_admitted(drift: f64, apple_lift: f64, palm_lift: f64) -> bool {
    [drift, apple_lift, palm_lift].iter().all(|v| v.is_finite())
        && (0. ..=0.02).contains(&drift)
        && apple_lift >= 0.025
        && palm_lift >= 0.025
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stationary_object_slip_and_nonfinite_motion_block_horizontal_transfer() {
        assert!(movement_admitted(0.008, 0.04, 0.045));
        assert!(!movement_admitted(0.008, 0., 0.045));
        assert!(!movement_admitted(0.021, 0.04, 0.045));
        assert!(!movement_admitted(0.008, 0.04, 0.));
        assert!(!movement_admitted(f64::NAN, 0.04, 0.045));
    }
    #[test]
    fn old_reset_and_repeated_images_cannot_verify_grip() {
        let a = ObservationStamp {
            episode_id: 1,
            frame_id: 3,
            sim_time_ns: 140 * 20_000_000,
            captured_at_unix_ms: 10000,
        };
        let b = ObservationStamp {
            frame_id: 4,
            sim_time_ns: 190 * 20_000_000,
            captured_at_unix_ms: 11000,
            ..a
        };
        assert!(validate_stamps(a, b, 12000).is_ok());
        assert!(validate_stamps(a, b, 14000).is_err());
        assert!(validate_stamps(a, ObservationStamp { episode_id: 2, ..b }, 12000).is_err());
        assert!(validate_stamps(a, ObservationStamp { frame_id: 3, ..b }, 12000).is_err());
        assert!(
            validate_stamps(
                a,
                ObservationStamp {
                    sim_time_ns: 189 * 20_000_000,
                    ..b
                },
                12000
            )
            .is_err()
        );
    }
}
