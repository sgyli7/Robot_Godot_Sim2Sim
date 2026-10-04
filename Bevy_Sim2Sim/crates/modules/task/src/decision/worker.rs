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

use super::{
    DecisionError, DecisionInput, DecisionReply, LocalQwenClient, ProfileStartInput,
    ProfileStartReply, SelectionInput, SelectionReply,
};
use crate::types::ObservationStamp;

enum Input {
    Visual(DecisionInput),
    Selection(SelectionInput),
    ProfileStart(ProfileStartInput),
}
impl Input {
    fn stamp(&self) -> ObservationStamp {
        match self {
            Self::Visual(i) => i.observation.stamp,
            Self::Selection(i) => i.context.observation.stamp,
            Self::ProfileStart(i) => i.context.observation.stamp,
        }
    }
}
enum Reply {
    Visual(DecisionReply),
    Selection(SelectionReply),
    ProfileStart(ProfileStartReply),
}
impl Reply {
    fn stamp(&self) -> ObservationStamp {
        match self {
            Self::Visual(r) => r.stamp,
            Self::Selection(r) => r.stamp,
            Self::ProfileStart(r) => r.stamp,
        }
    }
}

pub struct DecisionWorker {
    requests: Option<SyncSender<Input>>,
    replies: Receiver<Reply>,
    busy: Arc<AtomicBool>,
    episode: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
    http_attempts: Arc<AtomicU64>,
    http_results: Arc<AtomicU64>,
    discarded_results: Arc<AtomicU64>,
}

impl DecisionWorker {
    pub fn spawn(client: LocalQwenClient, episode_id: EpisodeId) -> Result<Self, DecisionError> {
        let (requests, input) = mpsc::sync_channel::<Input>(1);
        let (output, replies) = mpsc::sync_channel(1);
        let busy = Arc::new(AtomicBool::new(false));
        let episode = Arc::new(AtomicU64::new(episode_id));
        let shutdown = Arc::new(AtomicBool::new(false));
        let http_attempts = Arc::new(AtomicU64::new(0));
        let http_results = Arc::new(AtomicU64::new(0));
        let discarded_results = Arc::new(AtomicU64::new(0));
        let worker_attempts = http_attempts.clone();
        let worker_results = http_results.clone();
        let worker_discarded = discarded_results.clone();
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
                    if worker_episode.load(Ordering::Acquire) != request.stamp().episode_id {
                        worker_busy.store(false, Ordering::Release);
                        continue;
                    }
                    let started = Instant::now();
                    worker_attempts.fetch_add(1, Ordering::AcqRel);
                    let stamp = request.stamp();
                    let reply = match request {
                        Input::Visual(request) => Reply::Visual(DecisionReply {
                            request_id: request.request_id,
                            stamp,
                            result: client.decide(&request),
                            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        }),
                        Input::Selection(request) => Reply::Selection(SelectionReply {
                            request_id: request.context.request_id,
                            stamp,
                            result: client.select_verified(&request),
                            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        }),
                        Input::ProfileStart(request) => Reply::ProfileStart(ProfileStartReply {
                            request_id: request.context.request_id,
                            stamp,
                            result: client.start_fixed_profile(&request),
                            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                        }),
                    };
                    worker_results.fetch_add(1, Ordering::AcqRel);
                    if worker_episode.load(Ordering::Acquire) != stamp.episode_id {
                        worker_discarded.fetch_add(1, Ordering::AcqRel);
                        worker_busy.store(false, Ordering::Release);
                        continue;
                    }
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
            http_attempts,
            http_results,
            discarded_results,
        })
    }

    /// At most one submitted, running, or unconsumed request exists. Busy is
    /// returned immediately instead of accumulating old images or decisions.
    pub fn try_submit(&self, input: DecisionInput) -> Result<(), DecisionError> {
        self.submit(Input::Visual(input))
    }

    pub fn try_submit_selection(&self, input: SelectionInput) -> Result<(), DecisionError> {
        self.submit(Input::Selection(input))
    }

    pub fn try_submit_profile_start(&self, input: ProfileStartInput) -> Result<(), DecisionError> {
        self.submit(Input::ProfileStart(input))
    }

    fn submit(&self, input: Input) -> Result<(), DecisionError> {
        if input.stamp().episode_id != self.episode.load(Ordering::Acquire) {
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
        match self.receive()? {
            Some(Reply::Visual(reply)) => Ok(Some(reply)),
            Some(Reply::Selection(_) | Reply::ProfileStart(_)) => Err(DecisionError::Response(
                "selection reply received by visual-claim consumer".into(),
            )),
            None => Ok(None),
        }
    }

    pub fn try_recv_selection(&self) -> Result<Option<SelectionReply>, DecisionError> {
        match self.receive()? {
            Some(Reply::Selection(reply)) => Ok(Some(reply)),
            Some(Reply::Visual(_) | Reply::ProfileStart(_)) => Err(DecisionError::Response(
                "visual-claim reply received by selection consumer".into(),
            )),
            None => Ok(None),
        }
    }

    pub fn try_recv_profile_start(&self) -> Result<Option<ProfileStartReply>, DecisionError> {
        match self.receive()? {
            Some(Reply::ProfileStart(reply)) => Ok(Some(reply)),
            Some(Reply::Visual(_) | Reply::Selection(_)) => Err(DecisionError::Response(
                "foreign reply received by fixed profile consumer".into(),
            )),
            None => Ok(None),
        }
    }

    fn receive(&self) -> Result<Option<Reply>, DecisionError> {
        match self.replies.try_recv() {
            Ok(reply) => {
                self.busy.store(false, Ordering::Release);
                if reply.stamp().episode_id == self.episode.load(Ordering::Acquire) {
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

    /// Actual client invocation/results, including transport failures. These
    /// are not counts of successful model inference or accepted skills.
    pub fn transport_counts(&self) -> (u64, u64, u64) {
        (
            self.http_attempts.load(Ordering::Acquire),
            self.http_results.load(Ordering::Acquire),
            self.discarded_results.load(Ordering::Acquire),
        )
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
