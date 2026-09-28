//! Versioned policy boundaries; native sensor values are supplied by the plant.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const ACTION_DIMENSION: usize = 14;
pub const OBSERVATION_DIMENSION: usize = 61;
pub const ACTUATOR_ORDER: [&str; ACTION_DIMENSION] = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
];

#[derive(Debug, Error)]
pub enum RobotError {
    #[error("invalid robot contract: {0}")]
    Contract(String),
    #[error("non-finite robot data: {0}")]
    NonFinite(&'static str),
    #[error("policy initialization or inference failed: {0}")]
    Policy(String),
}

/// Immutable timing provenance and latency values in physical seconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaySeconds {
    pub motor_source_samples: [f64; 4],
    pub joint_velocity: f64,
    pub imu_source_samples: [f64; 2],
    pub imu_resample_period: f64,
}

/// A policy exported for this exact plant and observation interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyContract {
    pub schema_version: String,
    pub model_sha256: String,
    pub robot_definition_sha256: String,
    pub actuator_order: [String; ACTION_DIMENSION],
    pub home: [f32; ACTION_DIMENSION],
    pub action_scale: [f32; ACTION_DIMENSION],
    pub physics_hz: u32,
    pub policy_hz: u32,
    pub normalizer_embedded: bool,
    pub delay_seconds: DelaySeconds,
}

impl PolicyContract {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.schema_version != "microduck_pollen_bam60_v1"
            || self.physics_hz != 60
            || self.policy_hz != 60
            || !self.normalizer_embedded
        {
            return Err(RobotError::Contract(
                "schema, 60 Hz clocks or embedded normalizer mismatch".into(),
            ));
        }
        for hash in [&self.model_sha256, &self.robot_definition_sha256] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            {
                return Err(RobotError::Contract("expected lowercase SHA256".into()));
            }
        }
        if self
            .actuator_order
            .iter()
            .map(String::as_str)
            .ne(ACTUATOR_ORDER)
        {
            return Err(RobotError::Contract("actuator order mismatch".into()));
        }
        finite(&self.home, "home")?;
        finite(&self.action_scale, "action scale")?;
        if self.action_scale.iter().any(|value| *value <= 0.0) {
            return Err(RobotError::Contract("non-positive action scale".into()));
        }
        let timing = &self.delay_seconds;
        for value in timing
            .motor_source_samples
            .iter()
            .chain(timing.imu_source_samples.iter())
            .chain([&timing.joint_velocity, &timing.imu_resample_period])
        {
            if !value.is_finite() || *value < 0.0 || *value > 10.0 {
                return Err(RobotError::Contract("invalid latency seconds".into()));
            }
        }
        Ok(())
    }

    /// Retains the upstream multiplication then addition, without fused FMA.
    pub fn targets(
        &self,
        action: &[f32; ACTION_DIMENSION],
    ) -> Result<[f32; ACTION_DIMENSION], RobotError> {
        finite(action, "action")?;
        let mut result = [0.0; ACTION_DIMENSION];
        for index in 0..ACTION_DIMENSION {
            let scaled = action[index] * self.action_scale[index];
            result[index] = scaled + self.home[index];
        }
        finite(&result, "target")?;
        Ok(result)
    }

    pub fn observation(
        &self,
        state: &NativeState,
        command: &Command,
        previous_action: &[f32; ACTION_DIMENSION],
    ) -> Result<[f32; OBSERVATION_DIMENSION], RobotError> {
        let mut result = [0.0; OBSERVATION_DIMENSION];
        result[..3].copy_from_slice(&state.gyro);
        result[3..6].copy_from_slice(&state.projected_gravity);
        for index in 0..ACTION_DIMENSION {
            result[6 + index] = state.joint_position[index] - self.home[index];
        }
        result[20..34].copy_from_slice(&state.joint_velocity);
        result[34..48].copy_from_slice(previous_action);
        result[48..51].copy_from_slice(&command.locomotion);
        result[51..55].copy_from_slice(&command.head);
        result[55..61].copy_from_slice(&command.body);
        finite(&result, "observation")?;
        Ok(result)
    }
}

/// Native source-equivalent sensors, after the explicit latency queue.
#[derive(Debug, Clone, Default)]
pub struct NativeState {
    pub gyro: [f32; 3],
    pub projected_gravity: [f32; 3],
    pub joint_position: [f32; ACTION_DIMENSION],
    pub joint_velocity: [f32; ACTION_DIMENSION],
}

/// Already interpreted upstream command terms in their actor order.
#[derive(Debug, Clone, Default)]
pub struct Command {
    pub locomotion: [f32; 3],
    pub head: [f32; 4],
    pub body: [f32; 6],
}

pub(crate) fn finite(values: &[f32], name: &'static str) -> Result<(), RobotError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(RobotError::NonFinite(name))
    }
}
