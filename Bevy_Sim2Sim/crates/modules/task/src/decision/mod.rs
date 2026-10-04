//! Image-grounded local decisions, generation isolation and bounded feedback.

mod client;
mod observation;
mod selection;
mod session;
mod wire;
mod worker;

pub use client::{LocalQwenClient, LocalQwenConfig};
pub use observation::{CameraRgb, ObservationSnapshot, TaskGoal};
pub use selection::{
    RgbTarget, RgbVerifiedTargets, SelectionDecision, SelectionInput, SelectionReply,
    SelectionRequest,
};
pub use session::{DecisionLimits, DecisionSession, RegisteredTarget, ValidatedDecision};
pub use wire::{
    DecisionInput, DecisionMemory, DecisionReply, ExecutionFeedback, FeedbackStatus, ModelDecision,
    TargetKind, VisualTarget, decision_json_schema,
};
pub use worker::DecisionWorker;

/// Failures are surfaced to the runtime; no alternative model is selected.
#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    #[error("invalid local decision configuration: {0}")]
    Configuration(String),
    #[error("invalid observation: {0}")]
    Observation(String),
    #[error("decision service failed: {0}")]
    Service(String),
    #[error("invalid model response: {0}")]
    Response(String),
    #[error("decision rejected: {0}")]
    Rejected(String),
    #[error("one decision is already pending")]
    Busy,
}
