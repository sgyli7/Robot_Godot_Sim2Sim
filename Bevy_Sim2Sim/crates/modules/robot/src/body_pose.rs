//! Read-only visual boundary in engine Y-up coordinates; no renderer or physics.

use crate::RobotError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotBodyPose {
    pub source_body_id: usize,
    /// Native index and generation, inseparable across cold rebuilds.
    pub backend_handle: [u32; 2],
    pub translation: [f32; 3],
    pub rotation_xyzw: [f32; 4],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotPoseFrame {
    pub model_file_sha256: String,
    pub episode_id: u64,
    pub global_step: u64,
    pub episode_step: u64,
    /// All robot bodies, excluding the source world's body zero.
    pub poses: Vec<RobotBodyPose>,
}

impl RobotPoseFrame {
    /// Validate a complete frame before displaying any part of it.
    /// This checks data consistency, not the origin or qualification of a plant.
    pub fn validate(
        &self,
        expected_model_sha256: &str,
        source_body_count: usize,
    ) -> Result<(), RobotError> {
        if self.model_file_sha256 != expected_model_sha256
            || source_body_count < 2
            || self.poses.len() != source_body_count - 1
            || self.episode_step > self.global_step
        {
            return Err(invalid(
                "robot pose identity, body coverage or step mismatch",
            ));
        }
        let mut bodies = HashSet::new();
        let mut handles = HashSet::new();
        for pose in &self.poses {
            if pose.source_body_id == 0
                || pose.source_body_id >= source_body_count
                || !bodies.insert(pose.source_body_id)
                || !handles.insert(pose.backend_handle)
            {
                return Err(invalid("duplicate or out-of-range robot body/handle"));
            }
            if !pose
                .translation
                .iter()
                .chain(&pose.rotation_xyzw)
                .all(|value| value.is_finite())
            {
                return Err(invalid("non-finite robot display pose"));
            }
            let norm_squared: f64 = pose
                .rotation_xyzw
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum();
            if (norm_squared - 1.0).abs() > 1e-5 {
                return Err(invalid("robot display quaternion is not unit length"));
            }
        }
        Ok(())
    }
}

fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}
