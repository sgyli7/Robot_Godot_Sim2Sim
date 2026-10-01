//! The model sees an allowlisted schema and returns untrusted visual claims.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::types::{EpisodeId, ObservationStamp, SkillAvailability, SkillRequest};

use super::{DecisionError, ObservationSnapshot, RegisteredTarget, TaskGoal};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    Apple,
    Box,
    PlacementZone,
    Obstacle,
}

/// A visual claim from the RGB image, never a world-space grasp coordinate.
/// IDs name a visible marker or the explicitly described visual detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualTarget {
    pub id: String,
    pub kind: TargetKind,
    pub visible_description: String,
    pub bbox_xyxy: [f32; 4],
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDecision {
    pub episode_id: EpisodeId,
    pub frame_id: u64,
    pub observed_targets: Vec<VisualTarget>,
    pub request: SkillRequest,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackStatus {
    Running,
    Succeeded,
    Failed,
    Rejected,
    SafelyStopped,
}

/// Feedback comes from the skill executor's ordinary sensors/status. Independent
/// oracle success labels must stay in the evaluator and are not feedback input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionFeedback {
    pub episode_id: EpisodeId,
    pub decision_id: u64,
    pub status: FeedbackStatus,
    pub summary: String,
    pub sim_time_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionMemory {
    pub decision_id: u64,
    pub frame_id: u64,
    pub request: SkillRequest,
    pub reason: String,
}

/// Prepared by DecisionSession. The HTTP transport treats the response as
/// untrusted until DecisionSession accepts it against the current generation.
#[derive(Debug, Clone)]
pub struct DecisionInput {
    pub request_id: u64,
    pub observation: ObservationSnapshot,
    pub goal: TaskGoal,
    pub available_skills: SkillAvailability,
    pub remembered_targets: Vec<RegisteredTarget>,
    pub feedback: Vec<ExecutionFeedback>,
    pub recent_decisions: Vec<DecisionMemory>,
}

#[derive(Debug)]
pub struct DecisionReply {
    pub request_id: u64,
    pub stamp: ObservationStamp,
    pub elapsed_ms: u64,
    pub result: Result<ModelDecision, DecisionError>,
}

/// Strict schema sent using OpenAI-compatible response_format.json_schema.
pub fn decision_json_schema() -> Value {
    let text = |max| json!({"type":"string","minLength":1,"maxLength":max});
    let id = json!({"type":"string","minLength":1,"maxLength":64,"pattern":"^[A-Za-z0-9_-]+$"});
    let variants = vec![
        object(
            json!({"skill":{"type":"string","const":"observe"}}),
            &["skill"],
        ),
        object(
            json!({
                "skill":{"type":"string","const":"execute_task"},
                "task_profile":{"type":"string","enum":["static_apple","mobile_box"]},
                "target_id":id,"destination_id":id,
            }),
            &["skill", "task_profile", "target_id", "destination_id"],
        ),
        object(
            json!({
                "skill":{"type":"string","const":"navigate_adjustment"},
                "forward_mps":{"type":"number","minimum":-0.25,"maximum":0.25},
                "lateral_mps":{"type":"number","minimum":-0.25,"maximum":0.25},
                "yaw_rps":{"type":"number","minimum":-0.6,"maximum":0.6},
                "duration_ms":{"type":"integer","minimum":100,"maximum":1000},
            }),
            &[
                "skill",
                "forward_mps",
                "lateral_mps",
                "yaw_rps",
                "duration_ms",
            ],
        ),
        object(
            json!({"skill":{"type":"string","const":"stop"},"reason":text(512)}),
            &["skill", "reason"],
        ),
    ];
    let target = object(
        json!({
            "id":id,
            "kind":{"type":"string","enum":["apple","box","placement_zone","obstacle"]},
            "visible_description":text(128),
            "bbox_xyxy":{"type":"array","items":{"type":"number","minimum":0.0,"maximum":1.0},"minItems":4,"maxItems":4},
            "confidence":{"type":"number","minimum":0.7,"maximum":1.0},
        }),
        &[
            "id",
            "kind",
            "visible_description",
            "bbox_xyxy",
            "confidence",
        ],
    );
    object(
        json!({
            "episode_id":{"type":"integer","minimum":0},
            "frame_id":{"type":"integer","minimum":0},
            "observed_targets":{"type":"array","items":target,"maxItems":16},
            "request":{"anyOf":variants},
            "reason":text(512),
        }),
        &[
            "episode_id",
            "frame_id",
            "observed_targets",
            "request",
            "reason",
        ],
    )
}

fn object(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})
}
