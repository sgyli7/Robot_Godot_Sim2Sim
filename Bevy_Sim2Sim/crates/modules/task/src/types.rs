//! Task-level commands; these are not joint commands or world-state observations.

use serde::{Deserialize, Serialize};

/// A monotonically increasing scene-reset generation, owned by the runtime.
pub type EpisodeId = u64;

/// Each profile must use its own qualified body and task-policy contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskProfile {
    StaticApple,
    MobileBox,
}

/// Acquisition times supplied by the camera, never by the language model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationStamp {
    pub episode_id: EpisodeId,
    pub frame_id: u64,
    pub sim_time_ns: u64,
    pub captured_at_unix_ms: u64,
}

/// Proprioception only; no task-object locations or hidden object identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotSelfState {
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub base_velocity_mps: [f32; 3],
    pub projected_gravity: [f32; 3],
}

/// The complete allowlist exposed to visual decisions. Runtime safety checks
/// remain required when the physical executor consumes an accepted request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "skill", rename_all = "snake_case", deny_unknown_fields)]
pub enum SkillRequest {
    Observe,
    ExecuteTask {
        task_profile: TaskProfile,
        target_id: String,
        destination_id: String,
    },
    NavigateAdjustment {
        forward_mps: f32,
        lateral_mps: f32,
        yaw_rps: f32,
        duration_ms: u32,
    },
    Stop {
        reason: String,
    },
}

/// Capabilities are enabled by the executor after qualification, not by Qwen.
/// The safe default permits only observation and stopping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillAvailability {
    pub static_apple: bool,
    pub mobile_box: bool,
    pub navigate_adjustment: bool,
}

impl SkillAvailability {
    pub fn supports(self, profile: TaskProfile) -> bool {
        match profile {
            TaskProfile::StaticApple => self.static_apple,
            TaskProfile::MobileBox => self.mobile_box,
        }
    }
}
