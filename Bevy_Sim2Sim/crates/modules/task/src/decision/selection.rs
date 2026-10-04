//! Compact skill selection over caller-supplied, current RGB detections.
//!
//! The detector owns visual measurements; Qwen owns semantic target/skill
//! selection. Detector evidence is never represented as a model visual claim.
//! This module has no world coordinates, physics queries or executor access.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use super::{DecisionError, DecisionInput, ObservationSnapshot, TargetKind, wire::request_schema};
use crate::types::{EpisodeId, ObservationStamp, SkillRequest, TaskProfile};

/// A public printed marker measured by the actual RGB detector. The caller must
/// verify its detection provenance before binding it; numeric validation alone
/// is not a claim that this module detects objects in pixels.
#[derive(Debug, Clone, Serialize)]
pub struct RgbTarget {
    pub id: String,
    pub kind: TargetKind,
    pub public_description: String,
    pub marker_id: u32,
    pub marker_bbox_xyxy: [f32; 4],
    pub minimum_edge_px: f32,
    pub reprojection_rms_px: f32,
}

/// Immutable current detector measurements bound to one canonical camera PNG.
/// Cannot be deserialized from an untrusted model response.
#[derive(Debug, Clone, Serialize)]
pub struct RgbVerifiedTargets {
    stamp: ObservationStamp,
    image_sha256: String,
    targets: Vec<RgbTarget>,
}

impl RgbVerifiedTargets {
    pub fn bind_current_rgb(
        observation: &ObservationSnapshot,
        targets: Vec<RgbTarget>,
    ) -> Result<Self, DecisionError> {
        observation.validate()?;
        if targets.len() > 16 {
            return Err(DecisionError::Observation("too many RGB targets".into()));
        }
        let mut ids = BTreeSet::new();
        let mut markers = BTreeSet::new();
        for target in &targets {
            let [x0, y0, x1, y1] = target.marker_bbox_xyxy;
            let image_diagonal = ((x1 - x0) * observation.camera.width() as f32)
                .hypot((y1 - y0) * observation.camera.height() as f32);
            if target.id.is_empty()
                || target.id.len() > 64
                || !target
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                || !ids.insert(&target.id)
                || !markers.insert(target.marker_id)
                || target.public_description.trim().is_empty()
                || target.public_description.len() > 128
                || !target
                    .marker_bbox_xyxy
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                || x0 >= x1
                || y0 >= y1
                || !target.minimum_edge_px.is_finite()
                || target.minimum_edge_px < 8.0
                || target.minimum_edge_px > image_diagonal
                || !target.reprojection_rms_px.is_finite()
                || !(0.0..=1.0).contains(&target.reprojection_rms_px)
            {
                return Err(DecisionError::Observation(
                    "invalid or duplicated RGB marker measurement".into(),
                ));
            }
        }
        Ok(Self {
            stamp: observation.stamp,
            image_sha256: png_hash(observation),
            targets,
        })
    }

    pub fn stamp(&self) -> ObservationStamp {
        self.stamp
    }
    pub fn targets(&self) -> &[RgbTarget] {
        &self.targets
    }
    pub fn image_sha256(&self) -> &str {
        &self.image_sha256
    }

    pub(super) fn validate_binding(
        &self,
        observation: &ObservationSnapshot,
    ) -> Result<(), DecisionError> {
        if self.stamp != observation.stamp || self.image_sha256 != png_hash(observation) {
            return Err(DecisionError::Observation(
                "RGB candidates belong to another image or acquisition stamp".into(),
            ));
        }
        Ok(())
    }
}

fn png_hash(observation: &ObservationSnapshot) -> String {
    format!("{:x}", Sha256::digest(observation.camera.png()))
}

/// The profile is bound by the prepared goal, so the model does not repeat or
/// override it. The admitted executor request still contains its TaskProfile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SelectionRequest {
    Observe,
    ExecuteTask {
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

impl SelectionRequest {
    pub(super) fn resolve(self, profile: TaskProfile) -> SkillRequest {
        match self {
            Self::Observe => SkillRequest::Observe,
            Self::ExecuteTask {
                target_id,
                destination_id,
            } => SkillRequest::ExecuteTask {
                task_profile: profile,
                target_id,
                destination_id,
            },
            Self::NavigateAdjustment {
                forward_mps,
                lateral_mps,
                yaw_rps,
                duration_ms,
            } => SkillRequest::NavigateAdjustment {
                forward_mps,
                lateral_mps,
                yaw_rps,
                duration_ms,
            },
            Self::Stop { reason } => SkillRequest::Stop { reason },
        }
    }
}

/// Model-authored selection only. It contains no model/detector visual claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionDecision {
    pub episode_id: EpisodeId,
    pub frame_id: u64,
    pub request: SelectionRequest,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct SelectionInput {
    pub context: DecisionInput,
    pub verified_targets: RgbVerifiedTargets,
}

#[derive(Debug)]
pub struct SelectionReply {
    pub request_id: u64,
    pub stamp: ObservationStamp,
    pub elapsed_ms: u64,
    pub result: Result<SelectionDecision, DecisionError>,
}

pub(super) fn selection_schema(input: &SelectionInput) -> Value {
    let mut schema = request_schema(input.context.available_skills, input.context.goal.profile);
    schema["properties"]
        .as_object_mut()
        .unwrap()
        .remove("observed_targets");
    schema["required"]
        .as_array_mut()
        .unwrap()
        .retain(|v| v != "observed_targets");
    schema["properties"]["episode_id"] =
        json!({"type":"integer","const":input.context.observation.stamp.episode_id});
    schema["properties"]["frame_id"] =
        json!({"type":"integer","const":input.context.observation.stamp.frame_id});
    schema["properties"]["reason"]["maxLength"] = json!(32);
    let expected = match input.context.goal.profile {
        TaskProfile::StaticApple => TargetKind::Apple,
        TaskProfile::MobileBox => TargetKind::Box,
    };
    let objects: Vec<_> = input
        .verified_targets
        .targets
        .iter()
        .filter(|t| t.kind == expected)
        .map(|t| t.id.as_str())
        .collect();
    let zones: Vec<_> = input
        .verified_targets
        .targets
        .iter()
        .filter(|t| t.kind == TargetKind::PlacementZone)
        .map(|t| t.id.as_str())
        .collect();
    let variants = schema["properties"]["request"]["anyOf"]
        .as_array_mut()
        .unwrap();
    variants.retain(|v| {
        v["properties"]["skill"]["const"] != "execute_task"
            || (!objects.is_empty() && !zones.is_empty())
    });
    for variant in variants {
        match variant["properties"]["skill"]["const"].as_str() {
            Some("execute_task") => {
                variant["properties"]
                    .as_object_mut()
                    .unwrap()
                    .remove("task_profile");
                variant["required"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|v| v != "task_profile");
                variant["properties"]["target_id"] = json!({"type":"string","enum":objects});
                variant["properties"]["destination_id"] = json!({"type":"string","enum":zones});
            }
            Some("stop") => variant["properties"]["reason"]["maxLength"] = json!(32),
            _ => {}
        }
        // XGrammar emits schema properties in map order. A shared first key
        // keeps the action choice open: an initial "skill" key in the legacy
        // sorted union could only enter Observe (Execute began with destination
        // and Stop with reason). This wire discriminator sorts before all
        // action arguments, without enabling or forcing any action.
        let discriminator = variant["properties"]
            .as_object_mut()
            .unwrap()
            .remove("skill")
            .unwrap();
        variant["properties"]["action"] = discriminator;
        for required in variant["required"].as_array_mut().unwrap() {
            if *required == "skill" {
                *required = json!("action");
            }
        }
    }
    schema
}
