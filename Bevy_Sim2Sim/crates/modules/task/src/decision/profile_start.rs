//! Admission of a declared fixed task profile before its original policy runs.
//!
//! Starting a profile is not target localization or permission to release an
//! object. The executor must acquire and validate its later RGB targets.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{DecisionError, DecisionInput};
use crate::types::ObservationStamp;

#[derive(Debug, Clone)]
pub struct ProfileStartInput {
    pub context: DecisionInput,
}

/// No target ID, bounding box or coordinate can enter through this wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProfileStartRequest {
    BeginFixedProfile {},
    Observe {},
    Stop {},
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileStartDecision {
    pub episode_id: u64,
    pub frame_id: u64,
    pub request: ProfileStartRequest,
    pub reason: String,
}

#[derive(Debug)]
pub struct ProfileStartReply {
    pub request_id: u64,
    pub stamp: ObservationStamp,
    pub elapsed_ms: u64,
    pub result: Result<ProfileStartDecision, DecisionError>,
}

#[derive(Debug, Clone)]
pub struct ValidatedProfileStart {
    pub decision_id: u64,
    pub stamp: ObservationStamp,
    pub decision: ProfileStartDecision,
    pub image_to_decision_ms: u64,
    pub service_elapsed_ms: u64,
}

pub(super) fn profile_start_schema(input: &ProfileStartInput) -> Value {
    let mut actions = vec!["observe", "stop"];
    if input
        .context
        .available_skills
        .supports(input.context.goal.profile)
    {
        actions.push("begin_fixed_profile");
    }
    json!({
        "type":"object","additionalProperties":false,
        "required":["episode_id","frame_id","request","reason"],
        "properties":{
            "episode_id":{"type":"integer","const":input.context.observation.stamp.episode_id},
            "frame_id":{"type":"integer","const":input.context.observation.stamp.frame_id},
            "request":{"type":"object","additionalProperties":false,"required":["action"],
                "properties":{"action":{"type":"string","enum":actions}}},
            "reason":{"type":"string","minLength":1,"maxLength":32},
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_start_wire_cannot_smuggle_target_identity_or_actuator_values() {
        let good = json!({"episode_id":7,"frame_id":3,"request":{"action":"begin_fixed_profile"},"reason":"supported fixed task"});
        assert!(serde_json::from_value::<ProfileStartDecision>(good.clone()).is_ok());
        for field in [
            "target_id",
            "destination_id",
            "task_profile",
            "joint_positions",
            "world_position",
        ] {
            let mut foreign = good.clone();
            foreign["request"][field] = json!("foreign");
            assert!(serde_json::from_value::<ProfileStartDecision>(foreign).is_err());
        }
    }
}
