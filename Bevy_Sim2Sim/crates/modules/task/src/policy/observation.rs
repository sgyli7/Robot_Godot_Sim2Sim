use super::{ACTION_JOINT_NAMES, PolicyActionFrame};
use crate::types::ObservationStamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A real camera frame and measured proprioception from one simulation tick.
/// Task-object coordinates, object labels and desired poses are deliberately
/// absent. RGB data is contiguous HWC in the original camera's 0..255 range.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyObservation {
    pub stamp: ObservationStamp,
    pub camera_rgb: Vec<u8>,
    pub camera_width: u32,
    pub camera_height: u32,
    /// Order is ACTION_JOINT_NAMES, populated by name, not simulator indices.
    pub measured_joint_positions_rad: [f32; 31],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyObservationError {
    CameraShape,
    JointCount,
    DuplicateJoint(String),
    MissingJoint(&'static str),
    NonFinite,
}

impl PolicyObservation {
    pub fn validate(&self) -> Result<(), PolicyObservationError> {
        if self.camera_width != 640
            || self.camera_height != 480
            || self.camera_rgb.len() != 640 * 480 * 3
        {
            return Err(PolicyObservationError::CameraShape);
        }
        if !self
            .measured_joint_positions_rad
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(PolicyObservationError::NonFinite);
        }
        Ok(())
    }

    /// The numerical static preprocess graph uses the original training hand
    /// order. Its graph.yaml input element_names incorrectly advertise thumb
    /// first; the percentile normalization probe verifies this discrepancy.
    pub fn static_onnx_state(&self) -> Result<StaticOnnxState, PolicyObservationError> {
        self.validate()?;
        let q = &self.measured_joint_positions_rad;
        Ok(StaticOnnxState {
            left_arm: q[..7].try_into().expect("fixed range"),
            right_arm: q[7..14].try_into().expect("fixed range"),
            left_hand: q[14..21].try_into().expect("fixed range"),
            right_hand: q[21..28].try_into().expect("fixed range"),
            waist: q[28..31].try_into().expect("fixed range"),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticOnnxState {
    pub left_arm: [f32; 7],
    pub right_arm: [f32; 7],
    pub left_hand: [f32; 7],
    pub right_hand: [f32; 7],
    pub waist: [f32; 3],
}

/// Accept a simulator's complete joint array and extract the frozen policy order.
/// Extra lower-body joints are allowed; missing or duplicated names are not.
pub fn map_measured_joints(
    names: &[String],
    positions: &[f32],
) -> Result<[f32; 31], PolicyObservationError> {
    if names.len() != positions.len() {
        return Err(PolicyObservationError::JointCount);
    }
    let mut map = HashMap::with_capacity(names.len());
    for (name, value) in names.iter().zip(positions) {
        if !value.is_finite() {
            return Err(PolicyObservationError::NonFinite);
        }
        if map.insert(name.as_str(), *value).is_some() {
            return Err(PolicyObservationError::DuplicateJoint(name.clone()));
        }
    }
    let mut ordered = [0.0; 31];
    for (index, name) in ACTION_JOINT_NAMES.iter().enumerate() {
        ordered[index] = *map
            .get(name)
            .ok_or(PolicyObservationError::MissingJoint(name))?;
    }
    Ok(ordered)
}

/// Recorded after the physical engine steps. This is measurement, not an
/// acknowledgement that a command was submitted or that the task succeeded.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExecutionFeedback {
    pub stamp: ObservationStamp,
    pub sequence_id: u64,
    pub measured_joint_positions_rad: [f32; 31],
    pub measured_base_velocity_mps: [f32; 3],
    pub measured_projected_gravity: [f32; 3],
    pub controller_unstable: bool,
}

impl PhysicalExecutionFeedback {
    pub fn max_joint_error_rad(
        &self,
        submitted: &PolicyActionFrame,
    ) -> Result<f32, PolicyObservationError> {
        if !self
            .measured_joint_positions_rad
            .iter()
            .chain(&self.measured_base_velocity_mps)
            .chain(&self.measured_projected_gravity)
            .all(|x| x.is_finite())
        {
            return Err(PolicyObservationError::NonFinite);
        }
        let mut max_error = 0.0_f32;
        for (measured, target) in self
            .measured_joint_positions_rad
            .iter()
            .zip(submitted.joint_targets())
        {
            if !target.is_finite() {
                return Err(PolicyObservationError::NonFinite);
            }
            max_error = max_error.max((measured - target).abs());
        }
        Ok(max_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_simulator_order_maps_to_the_numerically_verified_training_order() {
        let names: Vec<String> = ACTION_JOINT_NAMES
            .iter()
            .rev()
            .map(|x| (*x).into())
            .collect();
        let values: Vec<f32> = (0..31).rev().map(|x| x as f32).collect();
        let observation = PolicyObservation {
            stamp: ObservationStamp {
                episode_id: 1,
                frame_id: 1,
                sim_time_ns: 0,
                captured_at_unix_ms: 0,
            },
            camera_rgb: vec![0; 640 * 480 * 3],
            camera_width: 640,
            camera_height: 480,
            measured_joint_positions_rad: map_measured_joints(&names, &values).unwrap(),
        };
        let state = observation.static_onnx_state().unwrap();
        assert_eq!(state.left_hand, [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]);
        assert_eq!(state.right_hand, [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]);
        assert_eq!(state.waist, [28.0, 29.0, 30.0]);
    }

    #[test]
    fn missing_or_duplicated_sensor_joint_is_rejected() {
        assert!(matches!(
            map_measured_joints(&[], &[]),
            Err(PolicyObservationError::MissingJoint(_))
        ));
        assert_eq!(
            map_measured_joints(&["joint".into(), "joint".into()], &[0.0, 0.0]),
            Err(PolicyObservationError::DuplicateJoint("joint".into()))
        );
    }
}
