//! A proper rotation from MuJoCo Z-up to the Bevy/Rapier Y-up basis.
//!
//! The same change of basis applies to each body's local frame. Quaternions
//! therefore transform by B R B^-1, not by premultiplying B alone.

use crate::RobotError;

pub fn source_to_engine_vector([x, y, z]: [f32; 3]) -> [f32; 3] {
    [x, z, -y]
}
pub fn engine_to_source_vector([x, y, z]: [f32; 3]) -> [f32; 3] {
    [x, -z, y]
}

/// Source wxyz to engine xyzw, with no accumulated Euler-angle conversion.
pub fn source_to_engine_rotation([w, x, y, z]: [f32; 4]) -> Result<[f32; 4], RobotError> {
    validate_rotation([w, x, y, z])?;
    Ok([x, z, -y, w])
}

pub fn engine_to_source_rotation([x, y, z, w]: [f32; 4]) -> Result<[f32; 4], RobotError> {
    validate_rotation([w, x, y, z])?;
    Ok([w, x, -z, y])
}

fn validate_rotation(values: [f32; 4]) -> Result<(), RobotError> {
    if !values.iter().all(|value| value.is_finite()) {
        return Err(RobotError::NonFinite("rotation"));
    }
    if (values.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() > 2e-6 {
        return Err(RobotError::Contract("non-unit boundary quaternion".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    #[test]
    fn gravity_and_heading_preserve_right_handed_axes() {
        assert_eq!(
            source_to_engine_vector([0.0, 0.0, -9.81]),
            [0.0, -9.81, -0.0]
        );
        let forward = source_to_engine_vector([1.0, 0.0, 0.0]);
        let left = source_to_engine_vector([0.0, 1.0, 0.0]);
        assert_eq!(
            cross(forward, left),
            source_to_engine_vector([0.0, 0.0, 1.0])
        );
        let native = [0.123, -0.456, 0.789];
        assert_eq!(
            engine_to_source_vector(source_to_engine_vector(native)),
            native
        );
    }

    #[test]
    fn source_yaw_becomes_positive_engine_y_rotation() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let source = [half, 0.0, 0.0, half];
        assert_eq!(
            source_to_engine_rotation(source).unwrap(),
            [0.0, half, -0.0, half]
        );
        assert_eq!(
            engine_to_source_rotation([0.0, half, 0.0, half]).unwrap(),
            source
        );
        assert!(source_to_engine_rotation([1.0, 1.0, 0.0, 0.0]).is_err());
        assert!(engine_to_source_rotation([f32::NAN, 0.0, 0.0, 1.0]).is_err());
    }
}
