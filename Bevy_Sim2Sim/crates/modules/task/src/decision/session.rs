//! Generation-safe task memory and skill admission; no scripted task sequence.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;

use crate::types::{EpisodeId, ObservationStamp, SkillAvailability, SkillRequest, TaskProfile};

use super::{
    DecisionError, DecisionInput, DecisionMemory, DecisionReply, ExecutionFeedback, FeedbackStatus,
    ModelDecision, ObservationSnapshot, RgbVerifiedTargets, SelectionInput, SelectionReply,
    TargetKind, TaskGoal, VisualTarget,
};

#[derive(Debug, Clone, Copy)]
pub struct DecisionLimits {
    pub max_frame_age_ms: u64,
    pub max_sim_age_ns: u64,
    pub target_ttl_ms: u64,
}

impl Default for DecisionLimits {
    fn default() -> Self {
        Self {
            max_frame_age_ms: 30_000,
            max_sim_age_ns: 30_000_000_000,
            target_ttl_ms: 60_000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisteredTarget {
    pub visual: VisualTarget,
    pub last_seen: ObservationStamp,
}

/// Only this admitted form should enter the physical skill executor.
#[derive(Debug, Clone)]
pub struct ValidatedDecision {
    pub decision_id: u64,
    pub stamp: ObservationStamp,
    pub request: SkillRequest,
    pub reason: String,
    pub image_to_decision_ms: u64,
    pub service_elapsed_ms: u64,
}

pub struct DecisionSession {
    episode_id: EpisodeId,
    goal: TaskGoal,
    availability: SkillAvailability,
    limits: DecisionLimits,
    targets: BTreeMap<String, RegisteredTarget>,
    feedback: VecDeque<ExecutionFeedback>,
    recent_decisions: VecDeque<DecisionMemory>,
    next_request: u64,
    pending: Option<(u64, ObservationStamp)>,
    pending_rgb_targets: Option<RgbVerifiedTargets>,
    latest_decision: Option<u64>,
    last_frame: Option<u64>,
    safe_stop: Option<String>,
}

impl DecisionSession {
    pub fn new(
        episode_id: EpisodeId,
        goal: TaskGoal,
        availability: SkillAvailability,
        limits: DecisionLimits,
    ) -> Result<Self, DecisionError> {
        goal.validate()?;
        if limits.max_frame_age_ms == 0
            || limits.max_sim_age_ns == 0
            || limits.target_ttl_ms < limits.max_frame_age_ms
        {
            return Err(DecisionError::Configuration(
                "invalid freshness limits".into(),
            ));
        }
        Ok(Self {
            episode_id,
            goal,
            availability,
            limits,
            targets: BTreeMap::new(),
            feedback: VecDeque::new(),
            recent_decisions: VecDeque::new(),
            next_request: 1,
            pending: None,
            pending_rgb_targets: None,
            latest_decision: None,
            last_frame: None,
            safe_stop: None,
        })
    }

    pub fn episode_id(&self) -> EpisodeId {
        self.episode_id
    }
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
    pub fn targets(&self) -> impl Iterator<Item = &RegisteredTarget> {
        self.targets.values()
    }

    /// Withdraw physical capabilities while retaining visual/task feedback.
    /// The executor also stops or pauses its own active control. A pending
    /// decision is cancelled so a previously prepared request cannot restore
    /// withdrawn capabilities when its delayed HTTP response arrives.
    pub fn revoke_physical_skills(&mut self) {
        self.availability = SkillAvailability::default();
        if self.pending.is_some() {
            self.cancel_pending("executor withdrew physical capabilities");
        }
    }
    pub fn feedback(&self) -> &VecDeque<ExecutionFeedback> {
        &self.feedback
    }

    /// Reset clears all observation identities and task memory. Request numbers
    /// continue increasing, so delayed responses cannot alias a new task.
    pub fn reset(&mut self, episode_id: EpisodeId, goal: TaskGoal) -> Result<(), DecisionError> {
        goal.validate()?;
        if episode_id <= self.episode_id {
            return Err(DecisionError::Rejected(
                "episode must increase on reset".into(),
            ));
        }
        self.episode_id = episode_id;
        self.goal = goal;
        self.targets.clear();
        self.feedback.clear();
        self.recent_decisions.clear();
        self.pending = None;
        self.pending_rgb_targets = None;
        self.latest_decision = None;
        self.last_frame = None;
        self.safe_stop = Some("scene reset; discard prior action history".into());
        Ok(())
    }

    pub fn prepare(
        &mut self,
        observation: ObservationSnapshot,
        now_unix_ms: u64,
        current_sim_time_ns: u64,
    ) -> Result<DecisionInput, DecisionError> {
        if self.pending.is_some() {
            return Err(DecisionError::Busy);
        }
        observation.validate()?;
        self.check_stamp(observation.stamp, now_unix_ms, current_sim_time_ns)?;
        if self
            .last_frame
            .is_some_and(|frame| observation.stamp.frame_id <= frame)
        {
            return Err(DecisionError::Rejected("camera frame must increase".into()));
        }
        self.targets.retain(|_, target| {
            now_unix_ms.saturating_sub(target.last_seen.captured_at_unix_ms)
                <= self.limits.target_ttl_ms
        });
        let request_id = self.next_request;
        self.next_request = self
            .next_request
            .checked_add(1)
            .ok_or_else(|| DecisionError::Rejected("request sequence exhausted".into()))?;
        self.pending = Some((request_id, observation.stamp));
        self.pending_rgb_targets = None;
        self.last_frame = Some(observation.stamp.frame_id);
        Ok(DecisionInput {
            request_id,
            observation,
            goal: self.goal.clone(),
            available_skills: self.availability,
            remembered_targets: self.targets.values().cloned().collect(),
            feedback: self.feedback.iter().cloned().collect(),
            recent_decisions: self.recent_decisions.iter().cloned().collect(),
        })
    }

    /// Bind detector measurements to the exact submitted image. Unlike the
    /// visual-claim path, the model selects among these IDs without reporting
    /// detections. Runtime freshness, capabilities and geometry stay separate.
    pub fn prepare_selection(
        &mut self,
        observation: ObservationSnapshot,
        verified_targets: RgbVerifiedTargets,
        now_unix_ms: u64,
        current_sim_time_ns: u64,
    ) -> Result<SelectionInput, DecisionError> {
        verified_targets.validate_binding(&observation)?;
        let context = self.prepare(observation, now_unix_ms, current_sim_time_ns)?;
        self.pending_rgb_targets = Some(verified_targets.clone());
        Ok(SelectionInput {
            context,
            verified_targets,
        })
    }

    /// Clear a failed submission/transport and request a local safe stop. The
    /// runtime must consume take_safe_stop even if it chooses to request a retry.
    pub fn service_failed(&mut self, request_id: u64, reason: impl Into<String>) {
        if let Some((pending, stamp)) = self.pending.filter(|(pending, _)| *pending == request_id) {
            let reason = bounded_text(reason.into(), 512);
            self.pending = None;
            self.pending_rgb_targets = None;
            self.remember_failure(pending, stamp.sim_time_ns, reason);
        }
    }

    /// Call while a request is pending, including when no HTTP result arrived.
    /// Expiration invalidates the result immediately and asks the executor to
    /// stop; it does not wait for the network timeout to finish.
    pub fn expire_pending(&mut self, now_unix_ms: u64, sim_time_ns: u64) -> bool {
        if let Some((request_id, stamp)) = self.pending
            && let Err(error) = self.check_stamp(stamp, now_unix_ms, sim_time_ns)
        {
            self.service_failed(request_id, error.to_string());
            return true;
        }
        false
    }

    pub fn cancel_pending(&mut self, reason: impl Into<String>) {
        let reason = bounded_text(reason.into(), 512);
        if let Some((request_id, stamp)) = self.pending.take() {
            self.pending_rgb_targets = None;
            self.remember_failure(request_id, stamp.sim_time_ns, reason);
        } else {
            self.safe_stop = Some(reason);
        }
    }

    pub fn take_safe_stop(&mut self) -> Option<SkillRequest> {
        self.safe_stop
            .take()
            .map(|reason| SkillRequest::Stop { reason })
    }

    pub fn accept(
        &mut self,
        reply: DecisionReply,
        now_unix_ms: u64,
        current_sim_time_ns: u64,
    ) -> Result<ValidatedDecision, DecisionError> {
        if self.pending != Some((reply.request_id, reply.stamp)) {
            return Err(DecisionError::Rejected(
                "unmatched request or previous episode".into(),
            ));
        }
        self.pending = None;
        let request_id = reply.request_id;
        let sim_time_ns = reply.stamp.sim_time_ns;
        let result = if self.pending_rgb_targets.take().is_some() {
            Err(DecisionError::Rejected(
                "RGB selection cannot be admitted as a model visual-claim reply".into(),
            ))
        } else {
            self.accept_current(reply, now_unix_ms, current_sim_time_ns)
        };
        if let Err(error) = &result {
            self.remember_failure(
                request_id,
                sim_time_ns,
                bounded_text(error.to_string(), 512),
            );
        }
        result
    }

    pub fn accept_selection(
        &mut self,
        reply: SelectionReply,
        now_unix_ms: u64,
        current_sim_time_ns: u64,
    ) -> Result<ValidatedDecision, DecisionError> {
        if self.pending != Some((reply.request_id, reply.stamp)) {
            return Err(DecisionError::Rejected(
                "unmatched selection request or previous episode".into(),
            ));
        }
        self.pending = None;
        let request_id = reply.request_id;
        let sim_time_ns = reply.stamp.sim_time_ns;
        let result = (|| {
            let targets = self.pending_rgb_targets.take().ok_or_else(|| {
                DecisionError::Rejected(
                    "selection reply lacks prepared current RGB evidence".into(),
                )
            })?;
            self.check_stamp(reply.stamp, now_unix_ms, current_sim_time_ns)?;
            let decision = reply.result?;
            if decision.episode_id != self.episode_id
                || decision.frame_id != reply.stamp.frame_id
                || decision.reason.trim().is_empty()
                || decision.reason.len() > 128
            {
                return Err(DecisionError::Rejected(
                    "selection has wrong frame, episode or rationale".into(),
                ));
            }
            let visible = targets
                .targets()
                .iter()
                .map(|target| (target.id.as_str(), target.kind))
                .collect();
            let request = decision.request.resolve(self.goal.profile);
            self.validate_skill(&request, &visible)?;
            Ok(self.remember_admission(
                request_id,
                reply.stamp,
                request,
                decision.reason,
                reply.elapsed_ms,
                now_unix_ms,
            ))
        })();
        if let Err(error) = &result {
            self.remember_failure(
                request_id,
                sim_time_ns,
                bounded_text(error.to_string(), 512),
            );
        }
        result
    }

    fn accept_current(
        &mut self,
        reply: DecisionReply,
        now_unix_ms: u64,
        current_sim_time_ns: u64,
    ) -> Result<ValidatedDecision, DecisionError> {
        self.check_stamp(reply.stamp, now_unix_ms, current_sim_time_ns)?;
        let decision = reply.result?;
        if decision.episode_id != self.episode_id || decision.frame_id != reply.stamp.frame_id {
            return Err(DecisionError::Rejected(
                "model echoed the wrong episode or frame".into(),
            ));
        }
        validate_visual_claims(&decision)?;
        let visible: BTreeMap<_, _> = decision
            .observed_targets
            .iter()
            .map(|target| (target.id.as_str(), target.kind))
            .collect();
        self.validate_skill(&decision.request, &visible)?;
        for visual in decision.observed_targets {
            self.targets.insert(
                visual.id.clone(),
                RegisteredTarget {
                    visual,
                    last_seen: reply.stamp,
                },
            );
        }
        // Bound remembered detections even if the model invents fresh names each
        // frame; only the most recent 64 visual records can survive.
        while self.targets.len() > 64 {
            let oldest = self
                .targets
                .iter()
                .min_by_key(|(_, target)| target.last_seen.frame_id)
                .map(|(id, _)| id.clone())
                .expect("nonempty registry");
            self.targets.remove(&oldest);
        }
        Ok(self.remember_admission(
            reply.request_id,
            reply.stamp,
            decision.request,
            decision.reason,
            reply.elapsed_ms,
            now_unix_ms,
        ))
    }

    fn remember_admission(
        &mut self,
        request_id: u64,
        stamp: ObservationStamp,
        request: SkillRequest,
        reason: String,
        service_elapsed_ms: u64,
        now_unix_ms: u64,
    ) -> ValidatedDecision {
        self.latest_decision = Some(request_id);
        if self.recent_decisions.len() == 8 {
            self.recent_decisions.pop_front();
        }
        self.recent_decisions.push_back(DecisionMemory {
            decision_id: request_id,
            frame_id: stamp.frame_id,
            request: request.clone(),
            reason: reason.clone(),
        });
        ValidatedDecision {
            decision_id: request_id,
            stamp,
            request,
            reason,
            image_to_decision_ms: now_unix_ms - stamp.captured_at_unix_ms,
            service_elapsed_ms,
        }
    }

    fn validate_skill(
        &self,
        request: &SkillRequest,
        visible: &BTreeMap<&str, TargetKind>,
    ) -> Result<(), DecisionError> {
        let reject = |message: &str| DecisionError::Rejected(message.into());
        match request {
            SkillRequest::Observe => Ok(()),
            SkillRequest::Stop { reason } if !reason.trim().is_empty() && reason.len() <= 512 => {
                Ok(())
            }
            SkillRequest::Stop { .. } => Err(reject("invalid stop reason")),
            SkillRequest::ExecuteTask {
                task_profile,
                target_id,
                destination_id,
            } => {
                if *task_profile != self.goal.profile || !self.availability.supports(*task_profile)
                {
                    return Err(reject(
                        "task profile is unsupported or not qualified by the executor",
                    ));
                }
                let target = visible
                    .get(target_id.as_str())
                    .ok_or_else(|| reject("target has no current visual observation"))?;
                let destination = visible
                    .get(destination_id.as_str())
                    .ok_or_else(|| reject("destination has no current visual observation"))?;
                let expected = match task_profile {
                    TaskProfile::StaticApple => TargetKind::Apple,
                    TaskProfile::MobileBox => TargetKind::Box,
                };
                if *target != expected
                    || *destination != TargetKind::PlacementZone
                    || target_id == destination_id
                {
                    return Err(reject(
                        "object or destination kind is incompatible with the selected task profile",
                    ));
                }
                Ok(())
            }
            SkillRequest::NavigateAdjustment {
                forward_mps,
                lateral_mps,
                yaw_rps,
                duration_ms,
            } => {
                if self.goal.profile != TaskProfile::MobileBox
                    || !self.availability.mobile_box
                    || !self.availability.navigate_adjustment
                {
                    return Err(reject(
                        "navigation adjustment is unavailable for this task profile",
                    ));
                }
                if ![forward_mps, lateral_mps, yaw_rps]
                    .iter()
                    .all(|value| value.is_finite())
                    || forward_mps.abs() > 0.25
                    || lateral_mps.abs() > 0.25
                    || yaw_rps.abs() > 0.6
                    || !(100..=1000).contains(duration_ms)
                {
                    return Err(reject(
                        "navigation adjustment exceeds bounded velocity or duration",
                    ));
                }
                Ok(())
            }
        }
    }

    fn check_stamp(
        &self,
        stamp: ObservationStamp,
        now_unix_ms: u64,
        sim_time_ns: u64,
    ) -> Result<(), DecisionError> {
        if stamp.episode_id != self.episode_id
            || now_unix_ms
                .checked_sub(stamp.captured_at_unix_ms)
                .is_none_or(|age| age > self.limits.max_frame_age_ms)
            || sim_time_ns
                .checked_sub(stamp.sim_time_ns)
                .is_none_or(|age| age > self.limits.max_sim_age_ns)
        {
            return Err(DecisionError::Rejected(
                "observation is from another episode, expired or in the future".into(),
            ));
        }
        Ok(())
    }

    pub fn record_feedback(&mut self, feedback: ExecutionFeedback) -> Result<(), DecisionError> {
        if feedback.episode_id != self.episode_id
            || self.latest_decision != Some(feedback.decision_id)
            || feedback.summary.trim().is_empty()
            || feedback.summary.len() > 1024
        {
            return Err(DecisionError::Rejected(
                "invalid feedback or noncurrent execution".into(),
            ));
        }
        if matches!(
            feedback.status,
            FeedbackStatus::Failed | FeedbackStatus::Rejected
        ) {
            self.safe_stop = Some(feedback.summary.clone());
        }
        if self.feedback.len() == 16 {
            self.feedback.pop_front();
        }
        self.feedback.push_back(feedback);
        Ok(())
    }

    fn remember_failure(&mut self, request_id: u64, sim_time_ns: u64, reason: String) {
        self.safe_stop = Some(reason.clone());
        if self.feedback.len() == 16 {
            self.feedback.pop_front();
        }
        self.feedback.push_back(ExecutionFeedback {
            episode_id: self.episode_id,
            decision_id: request_id,
            status: FeedbackStatus::Rejected,
            summary: reason,
            sim_time_ns,
        });
    }
}

fn validate_visual_claims(decision: &ModelDecision) -> Result<(), DecisionError> {
    if decision.reason.trim().is_empty()
        || decision.reason.len() > 512
        || decision.observed_targets.len() > 16
    {
        return Err(DecisionError::Rejected(
            "invalid rationale or too many detections".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    for target in &decision.observed_targets {
        let [x0, y0, x1, y1] = target.bbox_xyxy;
        if target.id.is_empty()
            || target.id.len() > 64
            || !target
                .id
                .bytes()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'-')
            || !ids.insert(&target.id)
            || target.visible_description.trim().is_empty()
            || target.visible_description.len() > 128
            || !target.confidence.is_finite()
            || !(0.7..=1.0).contains(&target.confidence)
            || !target
                .bbox_xyxy
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            || x0 >= x1
            || y0 >= y1
        {
            return Err(DecisionError::Rejected(
                "invalid or duplicated visual detection".into(),
            ));
        }
    }
    Ok(())
}

fn bounded_text(mut value: String, maximum: usize) -> String {
    if value.len() > maximum {
        let mut end = maximum;
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        value.truncate(end);
    }
    value
}
