//! Arena G1_CFG explicit IdealPDActuator values, independent of USD drive gains.
//!
//! Source joint friction is a PhysX coefficient, not Rapier Coulomb torque.
//! It is recorded here and must not be silently applied with different units.

use super::contract::{JOINT_COUNT, JOINT_NAMES, finite};
use crate::RobotError;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct IdealPd {
    pub stiffness: f32,
    pub damping: f32,
    pub effort_limit: f32,
    pub velocity_limit: f32,
    pub armature: f32,
    pub physx_friction_coefficient: Option<f32>,
}

pub fn parameters(index: usize) -> Result<IdealPd, RobotError> {
    let name = JOINT_NAMES
        .get(index)
        .ok_or_else(|| RobotError::Contract("G1 actuator index out of range".into()))?;
    let (stiffness, damping, effort_limit, velocity_limit) = if name.contains("hip_") {
        (150., 2., 88., 32.)
    } else if name.contains("knee_") {
        (300., 4., 139., 20.)
    } else if name.contains("ankle_") {
        (40., 2., 50., 37.)
    } else if name.contains("waist_yaw") {
        (250., 5., 88., 32.)
    } else if name.contains("waist_") {
        (250., 5., 50., 37.)
    } else if name.contains("shoulder_pitch") || name.contains("shoulder_roll") {
        (100., 5., 25., 37.)
    } else if name.contains("shoulder_yaw") || name.contains("elbow_") {
        (40., 2., 25., 37.)
    } else if name.contains("wrist_roll") {
        (20., 2., 25., 37.)
    } else if name.contains("wrist_") {
        (20., 2., 5., 22.)
    } else if name.contains("hand_") {
        (4., 0.5, 5., 10.)
    } else {
        return Err(RobotError::Contract(
            "G1 joint lacks source actuator".into(),
        ));
    };
    Ok(IdealPd {
        stiffness,
        damping,
        effort_limit,
        velocity_limit,
        armature: 0.03,
        physx_friction_coefficient: if name.contains("hip_") || name.contains("knee_") {
            None
        } else {
            Some(0.03)
        },
    })
}

/// Source explicit PD evaluated exactly once per target physics boundary.
/// Velocity_limit is metadata: IdealPD does not implement a motor speed curve.
pub fn torques(
    targets: &[f32],
    positions: &[f32; JOINT_COUNT],
    velocities: &[f32; JOINT_COUNT],
) -> Result<[f32; JOINT_COUNT], RobotError> {
    if targets.len() != JOINT_COUNT {
        return Err(RobotError::Contract("G1 target width must be 43".into()));
    }
    finite(targets)?;
    finite(positions)?;
    finite(velocities)?;
    let mut output = [0.; JOINT_COUNT];
    for i in 0..JOINT_COUNT {
        let p = parameters(i)?;
        output[i] = (p.stiffness * (targets[i] - positions[i]) - p.damping * velocities[i])
            .clamp(-p.effort_limit, p.effort_limit);
    }
    finite(&output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_pd_clips_effort_but_has_no_velocity_curve() {
        let mut q = [0.; 43];
        let mut dq = [0.; 43];
        let mut target = [0.; 43];
        target[3] = 1.;
        target[21] = -1.;
        target[22] = 0.1;
        dq[0] = 40.;
        q[0] = -1.;
        let t = torques(&target, &q, &dq).unwrap();
        assert_eq!(t[3], 139.);
        assert_eq!(t[21], -5.);
        assert_eq!(t[22], 0.4);
        assert_eq!(t[0], 70.);
        for i in 0..43 {
            assert_eq!(parameters(i).unwrap().armature, 0.03);
        }
        assert!(parameters(43).is_err());
    }
}
