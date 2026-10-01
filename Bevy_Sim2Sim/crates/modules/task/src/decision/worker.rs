//! A single bounded worker keeps HTTP waits off the physics/render thread.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread,
    time::Instant,
};

use crate::types::EpisodeId;

use super::{DecisionError, DecisionInput, DecisionReply, LocalQwenClient};

pub struct DecisionWorker {
    requests: Option<SyncSender<DecisionInput>>,
    replies: Receiver<DecisionReply>,
    busy: Arc<AtomicBool>,
    episode: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
}

impl DecisionWorker {
    pub fn spawn(client: LocalQwenClient, episode_id: EpisodeId) -> Result<Self, DecisionError> {
        let (requests, input) = mpsc::sync_channel::<DecisionInput>(1);
        let (output, replies) = mpsc::sync_channel(1);
        let busy = Arc::new(AtomicBool::new(false));
        let episode = Arc::new(AtomicU64::new(episode_id));
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);
        let worker_episode = Arc::clone(&episode);
        let worker_shutdown = Arc::clone(&shutdown);
        thread::Builder::new()
            .name("local_qwen_decision".into())
            .spawn(move || {
                while let Ok(request) = input.recv() {
                    if worker_shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    if worker_episode.load(Ordering::Acquire)
                        != request.observation.stamp.episode_id
                    {
                        worker_busy.store(false, Ordering::Release);
                        continue;
                    }
                    let started = Instant::now();
                    let result = client.decide(&request);
                    if worker_episode.load(Ordering::Acquire)
                        != request.observation.stamp.episode_id
                    {
                        worker_busy.store(false, Ordering::Release);
                        continue;
                    }
                    let reply = DecisionReply {
                        request_id: request.request_id,
                        stamp: request.observation.stamp,
                        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        result,
                    };
                    if output.try_send(reply).is_err() {
                        worker_busy.store(false, Ordering::Release);
                    }
                }
            })
            .map_err(|error| DecisionError::Service(error.to_string()))?;
        Ok(Self {
            requests: Some(requests),
            replies,
            busy,
            episode,
            shutdown,
        })
    }

    /// At most one submitted, running, or unconsumed request exists. Busy is
    /// returned immediately instead of accumulating old images or decisions.
    pub fn try_submit(&self, input: DecisionInput) -> Result<(), DecisionError> {
        if input.observation.stamp.episode_id != self.episode.load(Ordering::Acquire) {
            return Err(DecisionError::Rejected("worker generation mismatch".into()));
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(DecisionError::Busy);
        }
        if self
            .requests
            .as_ref()
            .expect("sender exists until drop")
            .try_send(input)
            .is_err()
        {
            self.busy.store(false, Ordering::Release);
            return Err(DecisionError::Service(
                "decision worker disconnected or queue full".into(),
            ));
        }
        Ok(())
    }

    pub fn try_recv(&self) -> Result<Option<DecisionReply>, DecisionError> {
        match self.replies.try_recv() {
            Ok(reply) => {
                self.busy.store(false, Ordering::Release);
                if reply.stamp.episode_id == self.episode.load(Ordering::Acquire) {
                    Ok(Some(reply))
                } else {
                    Ok(None)
                }
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(DecisionError::Service(
                "decision worker disconnected".into(),
            )),
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    /// An already running HTTP operation ends at its configured timeout. Its
    /// output is discarded; no second model request starts in parallel.
    pub fn reset_episode(&self, episode_id: EpisodeId) -> Result<(), DecisionError> {
        if episode_id <= self.episode.load(Ordering::Acquire) {
            return Err(DecisionError::Rejected(
                "worker episode must increase".into(),
            ));
        }
        self.episode.store(episode_id, Ordering::Release);
        if self.replies.try_recv().is_ok() {
            self.busy.store(false, Ordering::Release);
        }
        Ok(())
    }
}

impl Drop for DecisionWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.requests.take();
        // Never join an HTTP operation from a real-time caller; timeout bounds
        // the detached thread's lifetime and the dropped receiver discards it.
    }
}
