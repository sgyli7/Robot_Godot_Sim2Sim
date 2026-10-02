use super::{
    ARENA_ACTION_PERIOD_NS, ActionLimits, PolicyActionChunk, PolicyActionError, PolicyActionFrame,
    profile_contract,
};
use crate::types::ObservationStamp;
use crate::types::TaskProfile;

/// Owner-side admission metadata; the model never supplies its execution clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyActionAdmission {
    pub observation: ObservationStamp,
    pub execution_start_sim_ns: u64,
    pub execution_end_sim_ns: u64,
    pub observation_age_at_admission_ns: u64,
}

struct AdmittedChunk {
    chunk: PolicyActionChunk,
    admission: PolicyActionAdmission,
}

/// A single latest chunk: inference replies never form an accumulating backlog.
pub struct PolicyActionQueue {
    episode_id: u64,
    profile: TaskProfile,
    limits: ActionLimits,
    max_observation_age_ns: u64,
    last_sequence: Option<u64>,
    last_observation: Option<(u64, u64)>,
    last_tick_ns: Option<u64>,
    chunk: Option<AdmittedChunk>,
    last_emitted_index: Option<usize>,
    stopped: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyTick {
    /// Apply through the physical controller, never as a rigid-body pose write.
    Action {
        sequence_id: u64,
        frame_index: usize,
        action: PolicyActionFrame,
        observation: ObservationStamp,
        execution_start_sim_ns: u64,
        observation_age_ns: u64,
    },
    /// This frame was already emitted; no new target should be submitted.
    AlreadyEmitted,
    /// The caller must use its validated wait/stop behavior.
    Unavailable,
}

impl PolicyActionQueue {
    pub fn new(
        episode_id: u64,
        profile: TaskProfile,
        limits: ActionLimits,
        max_observation_age_ns: u64,
    ) -> Result<Self, PolicyActionError> {
        limits.validate()?;
        Ok(Self {
            episode_id,
            profile,
            limits,
            max_observation_age_ns,
            last_sequence: None,
            last_observation: None,
            last_tick_ns: None,
            chunk: None,
            last_emitted_index: None,
            stopped: false,
        })
    }

    /// Clear all old requests/actions before any scene reset or task switch.
    pub fn reset(
        &mut self,
        episode_id: u64,
        profile: TaskProfile,
    ) -> Result<(), PolicyActionError> {
        if episode_id <= self.episode_id {
            self.stop();
            return Err(PolicyActionError::StaleEpisode);
        }
        self.episode_id = episode_id;
        self.profile = profile;
        self.last_sequence = None;
        self.last_observation = None;
        self.last_tick_ns = None;
        self.chunk = None;
        self.last_emitted_index = None;
        self.stopped = false;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.chunk = None;
        self.last_emitted_index = None;
        self.stopped = true;
    }

    /// Admit a whole chunk at a physical-owner control boundary. Frame zero
    /// starts here, matching Arena's post-inference execution order. This does
    /// not refresh the observation stamp or relax its independent age limit.
    pub fn accept(
        &mut self,
        chunk: PolicyActionChunk,
        now_sim_ns: u64,
    ) -> Result<PolicyActionAdmission, PolicyActionError> {
        let admission = self.validate_admission(&chunk, now_sim_ns)?;
        self.last_sequence = Some(chunk.sequence_id);
        self.last_observation = Some((chunk.observation.frame_id, chunk.observation.sim_time_ns));
        self.chunk = Some(AdmittedChunk { chunk, admission });
        self.last_emitted_index = None;
        self.last_tick_ns = Some(now_sim_ns);
        Ok(admission)
    }

    /// Check a bounded future replacement without changing the current action
    /// timeline, sequence or emission index. Actual admission must recheck.
    pub fn validate_admission(
        &self,
        chunk: &PolicyActionChunk,
        now_sim_ns: u64,
    ) -> Result<PolicyActionAdmission, PolicyActionError> {
        if self.stopped {
            return Err(PolicyActionError::Stopped);
        }
        if chunk.observation.episode_id != self.episode_id {
            return Err(PolicyActionError::WrongEpisode);
        }
        if chunk.profile != self.profile {
            return Err(PolicyActionError::WrongProfile);
        }
        let contract = profile_contract(self.profile);
        if !contract.artifact_matches_interface {
            return Err(PolicyActionError::UnmatchedArtifact);
        }
        if chunk.model_revision != contract.revision {
            return Err(PolicyActionError::WrongRevision);
        }
        if chunk.action_period_ns != ARENA_ACTION_PERIOD_NS {
            return Err(PolicyActionError::WrongPeriod);
        }
        if chunk.frames.len() != contract.action_horizon {
            return Err(PolicyActionError::WrongHorizon);
        }
        if self
            .last_sequence
            .is_some_and(|previous| chunk.sequence_id <= previous)
        {
            return Err(PolicyActionError::StaleSequence);
        }
        if self.last_observation.is_some_and(|(frame, time)| {
            chunk.observation.frame_id <= frame || chunk.observation.sim_time_ns < time
        }) {
            return Err(PolicyActionError::StaleObservation);
        }
        let age = now_sim_ns
            .checked_sub(chunk.observation.sim_time_ns)
            .ok_or(PolicyActionError::FutureObservation)?;
        if age > self.max_observation_age_ns {
            return Err(PolicyActionError::StaleObservation);
        }
        let duration = chunk
            .action_period_ns
            .checked_mul(chunk.frames.len() as u64)
            .ok_or(PolicyActionError::TimestampOverflow)?;
        let execution_end_sim_ns = now_sim_ns
            .checked_add(duration)
            .ok_or(PolicyActionError::TimestampOverflow)?;
        if self
            .last_tick_ns
            .is_some_and(|previous| now_sim_ns < previous)
        {
            return Err(PolicyActionError::TimeWentBackwards);
        }
        for action in &chunk.frames {
            action.validate(&self.limits)?;
        }
        Ok(PolicyActionAdmission {
            observation: chunk.observation,
            execution_start_sim_ns: now_sim_ns,
            execution_end_sim_ns,
            observation_age_at_admission_ns: age,
        })
    }

    /// Advance using simulation time; skipped old frames are discarded, not replayed.
    pub fn tick(&mut self, now_sim_ns: u64) -> Result<PolicyTick, PolicyActionError> {
        if self
            .last_tick_ns
            .is_some_and(|previous| now_sim_ns < previous)
        {
            self.stop();
            return Err(PolicyActionError::TimeWentBackwards);
        }
        self.last_tick_ns = Some(now_sim_ns);
        let Some(admitted) = &self.chunk else {
            return Ok(PolicyTick::Unavailable);
        };
        let Some(elapsed) = now_sim_ns.checked_sub(admitted.admission.execution_start_sim_ns)
        else {
            self.stop();
            return Err(PolicyActionError::TimeWentBackwards);
        };
        let chunk = &admitted.chunk;
        let index = (elapsed / chunk.action_period_ns) as usize;
        if index >= chunk.frames.len() {
            self.chunk = None;
            self.last_emitted_index = None;
            return Ok(PolicyTick::Unavailable);
        }
        if self.last_emitted_index == Some(index) {
            return Ok(PolicyTick::AlreadyEmitted);
        }
        self.last_emitted_index = Some(index);
        Ok(PolicyTick::Action {
            sequence_id: chunk.sequence_id,
            frame_index: index,
            action: chunk.frames[index].clone(),
            observation: chunk.observation,
            execution_start_sim_ns: admitted.admission.execution_start_sim_ns,
            observation_age_ns: now_sim_ns - chunk.observation.sim_time_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ObservationStamp;

    fn queue() -> PolicyActionQueue {
        PolicyActionQueue::new(
            1,
            TaskProfile::StaticApple,
            ActionLimits {
                joint_min_rad: [-3.0; 31],
                joint_max_rad: [3.0; 31],
                base_height_min_m: 0.4,
                base_height_max_m: 0.9,
                navigation_abs_max: [1.0; 3],
            },
            1_000_000_000,
        )
        .unwrap()
    }

    fn chunk() -> PolicyActionChunk {
        PolicyActionChunk {
            profile: TaskProfile::StaticApple,
            observation: ObservationStamp {
                episode_id: 1,
                frame_id: 1,
                sim_time_ns: 0,
                captured_at_unix_ms: 0,
            },
            sequence_id: 1,
            model_revision: profile_contract(TaskProfile::StaticApple).revision.into(),
            action_period_ns: ARENA_ACTION_PERIOD_NS,
            frames: vec![
                PolicyActionFrame {
                    left_arm: [0.0; 7],
                    right_arm: [0.0; 7],
                    left_hand: [0.0; 7],
                    right_hand: [0.0; 7],
                    waist: [0.0; 3],
                    base_height_m: 0.75,
                    navigate_mps_rps: [0.0; 3],
                };
                40
            ],
        }
    }

    #[test]
    fn delayed_inference_starts_at_zero_and_preserves_actual_observation_age() {
        let mut q = queue();
        let admission = q.accept(chunk(), 604_000_000).unwrap();
        assert_eq!(admission.observation.sim_time_ns, 0);
        assert_eq!(admission.observation_age_at_admission_ns, 604_000_000);
        assert_eq!(admission.execution_end_sim_ns, 1_404_000_000);
        assert!(matches!(
            q.tick(604_000_000).unwrap(),
            PolicyTick::Action {
                frame_index: 0,
                observation_age_ns: 604_000_000,
                ..
            }
        ));
        assert_eq!(q.tick(604_000_001).unwrap(), PolicyTick::AlreadyEmitted);
        assert!(matches!(
            q.tick(1_384_000_000).unwrap(),
            PolicyTick::Action {
                frame_index: 39,
                ..
            }
        ));
        assert_eq!(q.tick(1_404_000_000).unwrap(), PolicyTick::Unavailable);
    }

    #[test]
    fn complete_horizon_survives_inference_longer_than_the_horizon() {
        let mut q = queue();
        let mut reply = chunk();
        for (i, frame) in reply.frames.iter_mut().enumerate() {
            frame.left_arm[0] = i as f32 * 0.01;
        }
        q.accept(reply, 900_000_000).unwrap();
        for i in 0..40 {
            let tick = q.tick(900_000_000 + i * ARENA_ACTION_PERIOD_NS).unwrap();
            match tick {
                PolicyTick::Action {
                    frame_index,
                    action,
                    observation,
                    ..
                } => {
                    assert_eq!(frame_index, i as usize);
                    assert_eq!(action.left_arm[0], i as f32 * 0.01);
                    assert_eq!(observation.sim_time_ns, 0);
                }
                other => panic!("Missing source action {i}: {other:?}"),
            }
        }
        assert_eq!(q.tick(1_700_000_000).unwrap(), PolicyTick::Unavailable);
    }

    #[test]
    fn owner_stall_discards_elapsed_execution_frames() {
        let mut q = queue();
        q.accept(chunk(), 100_000_000).unwrap();
        q.tick(100_000_000).unwrap();
        assert!(matches!(
            q.tick(180_000_000).unwrap(),
            PolicyTick::Action { frame_index: 4, .. }
        ));
        assert_eq!(q.tick(900_000_000).unwrap(), PolicyTick::Unavailable);
    }

    #[test]
    fn admission_clock_and_execution_end_are_checked_before_replacement() {
        let mut q = queue();
        q.accept(chunk(), 100_000_000).unwrap();
        let mut newer = chunk();
        newer.sequence_id = 2;
        newer.observation.frame_id = 2;
        assert_eq!(
            q.accept(newer, 80_000_000),
            Err(PolicyActionError::TimeWentBackwards)
        );
        assert!(matches!(
            q.tick(100_000_000).unwrap(),
            PolicyTick::Action { sequence_id: 1, .. }
        ));
        q.reset(2, TaskProfile::StaticApple).unwrap();
        let mut overflow = chunk();
        overflow.observation.episode_id = 2;
        overflow.observation.sim_time_ns = u64::MAX - 10;
        assert_eq!(
            q.accept(overflow, u64::MAX - 10),
            Err(PolicyActionError::TimestampOverflow)
        );
        assert_eq!(q.tick(u64::MAX - 10).unwrap(), PolicyTick::Unavailable);
    }

    #[test]
    fn reset_invalidates_inflight_reply_and_history() {
        let mut q = queue();
        q.accept(chunk(), 0).unwrap();
        q.reset(2, TaskProfile::StaticApple).unwrap();
        assert_eq!(q.accept(chunk(), 0), Err(PolicyActionError::WrongEpisode));
        assert_eq!(q.tick(0).unwrap(), PolicyTick::Unavailable);
        let mut fresh = chunk();
        fresh.observation.episode_id = 2;
        q.accept(fresh, 0).unwrap();
    }

    #[test]
    fn rejects_invalid_reply_without_losing_valid_chunk() {
        let mut q = queue();
        q.accept(chunk(), 0).unwrap();
        let mut invalid = chunk();
        invalid.sequence_id = 2;
        invalid.observation.frame_id = 2;
        invalid.frames[39].left_hand[6] = f32::NAN;
        assert_eq!(q.accept(invalid, 0), Err(PolicyActionError::NonFinite));
        assert!(matches!(
            q.tick(0).unwrap(),
            PolicyTick::Action { sequence_id: 1, .. }
        ));
    }

    #[test]
    fn profile_revision_and_time_are_not_interchangeable() {
        let mut q = queue();
        let mut wrong = chunk();
        wrong.profile = TaskProfile::MobileBox;
        assert_eq!(q.accept(wrong, 0), Err(PolicyActionError::WrongProfile));
        let mut wrong = chunk();
        wrong.model_revision = "main".into();
        assert_eq!(q.accept(wrong, 0), Err(PolicyActionError::WrongRevision));
        let mut wrong = chunk();
        wrong.action_period_ns = 33_333_333;
        assert_eq!(q.accept(wrong, 0), Err(PolicyActionError::WrongPeriod));
        assert_eq!(
            q.accept(chunk(), 1_000_000_001),
            Err(PolicyActionError::StaleObservation)
        );
    }

    #[test]
    fn delayed_old_inference_cannot_replace_newer_observation() {
        let mut q = queue();
        let mut newer = chunk();
        newer.observation.frame_id = 2;
        newer.observation.sim_time_ns = 20_000_000;
        q.accept(newer, 20_000_000).unwrap();
        let mut old = chunk();
        old.sequence_id = 3;
        assert_eq!(
            q.accept(old, 20_000_000),
            Err(PolicyActionError::StaleObservation)
        );
    }

    #[test]
    fn clock_rewind_clears_action() {
        let mut q = queue();
        q.accept(chunk(), 0).unwrap();
        q.tick(40_000_000).unwrap();
        assert_eq!(q.tick(0), Err(PolicyActionError::TimeWentBackwards));
        assert_eq!(q.tick(60_000_000).unwrap(), PolicyTick::Unavailable);
    }

    #[test]
    fn stop_latches_until_episode_reset() {
        let mut q = queue();
        q.stop();
        assert_eq!(q.accept(chunk(), 0), Err(PolicyActionError::Stopped));
        q.reset(2, TaskProfile::StaticApple).unwrap();
        let mut fresh = chunk();
        fresh.observation.episode_id = 2;
        q.accept(fresh, 0).unwrap();
    }

    #[test]
    fn old_mobile_main_checkpoint_cannot_submit_actions() {
        let mut q = queue();
        q.reset(2, TaskProfile::MobileBox).unwrap();
        let mut mobile = chunk();
        mobile.observation.episode_id = 2;
        mobile.profile = TaskProfile::MobileBox;
        mobile.model_revision = "629479fedb1cf97c2f11ddc49eed951c5b750139".into();
        assert_eq!(q.accept(mobile, 0), Err(PolicyActionError::WrongRevision));
    }
}
