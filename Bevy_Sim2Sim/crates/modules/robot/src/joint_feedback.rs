//! Source-ordered, read-only joint feedback from one registered native plant.

use serde::Serialize;

use crate::ACTION_DIMENSION;

/// A native handle is meaningful only together with its generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct NativeHandleIdentity {
    pub index: u32,
    pub generation: u32,
}

/// One driven source joint, measured directly from its current Rapier link.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct JointFeedbackChannel {
    pub source_joint: usize,
    pub source_dof: usize,
    pub backend_dof: usize,
    pub joint_handle: NativeHandleIdentity,
    pub body_handle: NativeHandleIdentity,
    pub position: f32,
    pub velocity: f32,
}

/// No previous-solve load is qualified for the current plant yet. In
/// particular, an initial state's absent solve must never become zero load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PreviousSolveLoad {
    Unavailable,
}

/// An immutable value frame bound to one robot instance and completed tick.
/// This is joint feedback, not a 61D policy observation or control readiness.
#[derive(Clone, Debug, Serialize)]
pub struct JointFeedbackFrame {
    pub model_file_sha256: String,
    pub collision_profile_file_sha256: String,
    pub world_epoch: u64,
    pub robot_instance: u64,
    pub episode_id: u64,
    pub global_step: u64,
    pub episode_step: u64,
    pub channels: [JointFeedbackChannel; ACTION_DIMENSION],
    pub previous_solve_load: PreviousSolveLoad,
}
