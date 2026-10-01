use super::{
    ARENA_ACTION_PERIOD_NS, ActionLimits, PolicyActionChunk, PolicyActionError, PolicyActionFrame,
    profile_contract,
};
use crate::types::TaskProfile;

/// A single latest chunk: inference replies never form an accumulating backlog.
pub struct PolicyActionQueue {
    episode_id: u64,
    profile: TaskProfile,
    limits: ActionLimits,
    max_observation_age_ns: u64,
    last_sequence: Option<u64>,
    last_observation: Option<(u64, u64)>,
    last_tick_ns: Option<u64>,
    chunk: Option<PolicyActionChunk>,
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

    /// Validate the complete chunk before replacing the currently active one.
    pub fn accept(
        &mut self,
        chunk: PolicyActionChunk,
        now_sim_ns: u64,
    ) -> Result<(), PolicyActionError> {
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
        chunk
            .observation
            .sim_time_ns
            .checked_add(duration)
            .ok_or(PolicyActionError::TimestampOverflow)?;
        if age >= duration {
            return Err(PolicyActionError::Expired);
        }
        if self
            .last_tick_ns
            .is_some_and(|previous| now_sim_ns < previous)
        {
            return Err(PolicyActionError::TimeWentBackwards);
        }
        for action in &chunk.frames {
            action.validate(&self.limits)?;
        }
        self.last_sequence = Some(chunk.sequence_id);
        self.last_observation = Some((chunk.observation.frame_id, chunk.observation.sim_time_ns));
        self.chunk = Some(chunk);
        self.last_emitted_index = None;
        Ok(())
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
        let Some(chunk) = &self.chunk else {
            return Ok(PolicyTick::Unavailable);
        };
        let Some(age) = now_sim_ns.checked_sub(chunk.observation.sim_time_ns) else {
            self.stop();
            return Err(PolicyActionError::FutureObservation);
        };
        let index = (age / chunk.action_period_ns) as usize;
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
    fn skips_late_frames_and_expires_without_replay() {
        let mut q = queue();
        q.accept(chunk(), 60_000_000).unwrap();
        assert!(matches!(
            q.tick(60_000_000).unwrap(),
            PolicyTick::Action { frame_index: 3, .. }
        ));
        assert_eq!(q.tick(60_000_001).unwrap(), PolicyTick::AlreadyEmitted);
        assert!(matches!(
            q.tick(780_000_000).unwrap(),
            PolicyTick::Action {
                frame_index: 39,
                ..
            }
        ));
        assert_eq!(q.tick(800_000_000).unwrap(), PolicyTick::Unavailable);
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
            q.accept(chunk(), 800_000_000),
            Err(PolicyActionError::Expired)
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
