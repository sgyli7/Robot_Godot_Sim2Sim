use crate::types::{ObservationStamp, TaskProfile};
use serde::{Deserialize, Serialize};

/// The action spacing of the matched Arena control interface.
pub const ARENA_ACTION_PERIOD_NS: u64 = 20_000_000;

/// State/order contracts must remain attached to their own checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyProfileContract {
    pub repository: &'static str,
    pub revision: &'static str,
    pub arena_revision: &'static str,
    pub body_backend: PolicyBodyBackend,
    pub action_horizon: usize,
    /// False when published checkpoint bytes do not match this Arena interface.
    pub artifact_matches_interface: bool,
    pub camera_width: u32,
    pub camera_height: u32,
    pub reference_instruction: &'static str,
    pub instruction_is_fixed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyBodyBackend {
    Agile,
    HomieV2,
}

pub fn profile_contract(profile: TaskProfile) -> PolicyProfileContract {
    match profile {
        TaskProfile::StaticApple => PolicyProfileContract {
            repository: "nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace",
            revision: "7f78bebf1a90131e7304beacfcd47eb27bad16ab",
            arena_revision: "8b4a3a47fc53de23e8205089d71109a2e2348acd",
            body_backend: PolicyBodyBackend::Agile,
            action_horizon: 40,
            artifact_matches_interface: true,
            camera_width: 640,
            camera_height: 480,
            reference_instruction: "move the apple to the plate",
            instruction_is_fixed: true,
        },
        TaskProfile::MobileBox => PolicyProfileContract {
            repository: "nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation",
            revision: "dfe74af855007f26093f362cd2d7a2f404b64b93",
            arena_revision: "7d75c95934c51a0318c957a8831e862ca43c53b5",
            body_backend: PolicyBodyBackend::HomieV2,
            action_horizon: 50,
            // The gn1_6 processor and tensor dimensions match. This does not
            // grant task capability; source and physical qualification still apply.
            artifact_matches_interface: true,
            camera_width: 640,
            camera_height: 480,
            reference_instruction: "Pick up the brown box from the shelf, and place it into the blue bin on the table located at the right of the shelf.",
            instruction_is_fixed: false,
        },
    }
}

/// Decoded absolute joint targets in radians and base commands in SI units.
///
/// Each hand uses index(2), middle(2), thumb(3), matching the *output* groups
/// of the matched static export. Its ONNX state input has a different hand order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyActionFrame {
    pub left_arm: [f32; 7],
    pub right_arm: [f32; 7],
    pub left_hand: [f32; 7],
    pub right_hand: [f32; 7],
    pub waist: [f32; 3],
    pub base_height_m: f32,
    pub navigate_mps_rps: [f32; 3],
}

/// Upper-body joint order of the decoded Arena action interface.
pub const ACTION_JOINT_NAMES: [&str; 31] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
];

impl PolicyActionFrame {
    pub fn joint_targets(&self) -> [f32; 31] {
        let mut values = [0.0; 31];
        values[..7].copy_from_slice(&self.left_arm);
        values[7..14].copy_from_slice(&self.right_arm);
        values[14..21].copy_from_slice(&self.left_hand);
        values[21..28].copy_from_slice(&self.right_hand);
        values[28..].copy_from_slice(&self.waist);
        values
    }

    pub fn validate(&self, limits: &ActionLimits) -> Result<(), PolicyActionError> {
        limits.validate()?;
        for (index, value) in self.joint_targets().into_iter().enumerate() {
            if !value.is_finite() {
                return Err(PolicyActionError::NonFinite);
            }
            if value < limits.joint_min_rad[index] || value > limits.joint_max_rad[index] {
                return Err(PolicyActionError::JointLimit { index });
            }
        }
        if !self.base_height_m.is_finite() || !self.navigate_mps_rps.iter().all(|x| x.is_finite()) {
            return Err(PolicyActionError::NonFinite);
        }
        if self.base_height_m < limits.base_height_min_m
            || self.base_height_m > limits.base_height_max_m
        {
            return Err(PolicyActionError::BaseHeightLimit);
        }
        if self
            .navigate_mps_rps
            .iter()
            .zip(limits.navigation_abs_max)
            .any(|(value, max)| value.abs() > max)
        {
            return Err(PolicyActionError::NavigationLimit);
        }
        Ok(())
    }
}

/// Limits supplied from the frozen embodiment and the verified controller.
/// No permissive default silently substitutes for the robot's actual limits.
#[derive(Clone, Debug)]
pub struct ActionLimits {
    pub joint_min_rad: [f32; 31],
    pub joint_max_rad: [f32; 31],
    pub base_height_min_m: f32,
    pub base_height_max_m: f32,
    pub navigation_abs_max: [f32; 3],
}

impl ActionLimits {
    pub fn validate(&self) -> Result<(), PolicyActionError> {
        let joints_valid = self
            .joint_min_rad
            .iter()
            .zip(self.joint_max_rad)
            .all(|(min, max)| min.is_finite() && max.is_finite() && *min <= max);
        let height_valid = self.base_height_min_m.is_finite()
            && self.base_height_max_m.is_finite()
            && self.base_height_min_m > 0.0
            && self.base_height_min_m <= self.base_height_max_m;
        if !joints_valid
            || !height_valid
            || !self
                .navigation_abs_max
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
        {
            return Err(PolicyActionError::InvalidLimits);
        }
        Ok(())
    }
}

/// An inference reply bound to the observation and episode that produced it.
/// The observation stamp remains unchanged during admission. The matched Arena
/// interface executes frame zero after inference returns; the physical owner
/// separately records that execution start and enforces observation age limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyActionChunk {
    pub profile: TaskProfile,
    pub observation: ObservationStamp,
    pub sequence_id: u64,
    pub model_revision: String,
    pub action_period_ns: u64,
    pub frames: Vec<PolicyActionFrame>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyActionError {
    InvalidLimits,
    UnmatchedArtifact,
    Stopped,
    NonFinite,
    JointLimit { index: usize },
    BaseHeightLimit,
    NavigationLimit,
    WrongEpisode,
    StaleEpisode,
    WrongProfile,
    WrongRevision,
    WrongPeriod,
    WrongHorizon,
    StaleSequence,
    StaleObservation,
    FutureObservation,
    Expired,
    TimestampOverflow,
    TimeWentBackwards,
}
