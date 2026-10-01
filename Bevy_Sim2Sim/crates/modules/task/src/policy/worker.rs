use super::{PolicyActionChunk, PolicyInferenceError, PolicyInferenceRequest, StaticPolicyClient};
use crate::types::{ObservationStamp, TaskProfile};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Both success and failure retain the identity and latency of their request.
#[derive(Debug)]
pub struct PolicyInferenceReply {
    pub profile: TaskProfile,
    pub observation: ObservationStamp,
    pub sequence_id: u64,
    pub elapsed: Duration,
    pub result: Result<PolicyActionChunk, PolicyInferenceError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyWorkerError {
    WrongEpisode,
    StaleEpisode,
    Stopped,
}

struct WorkerState {
    episode_id: u64,
    pending: Option<PolicyInferenceRequest>,
    completed: Option<PolicyInferenceReply>,
    stopping: bool,
}

/// One in-flight query, one replaceable pending request and one latest reply.
/// Reset invalidates even in-flight errors within the worker. Action timing and
/// physical limits must additionally pass PolicyActionQueue.
pub struct PolicyWorker {
    state: Arc<(Mutex<WorkerState>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl PolicyWorker {
    pub fn spawn(client: StaticPolicyClient, episode_id: u64) -> std::io::Result<Self> {
        Self::spawn_with(episode_id, move |request| client.infer(request))
    }

    fn spawn_with(
        episode_id: u64,
        mut infer: impl FnMut(
            &PolicyInferenceRequest,
        ) -> Result<PolicyActionChunk, PolicyInferenceError>
        + Send
        + 'static,
    ) -> std::io::Result<Self> {
        let state = Arc::new((
            Mutex::new(WorkerState {
                episode_id,
                pending: None,
                completed: None,
                stopping: false,
            }),
            Condvar::new(),
        ));
        let shared = state.clone();
        let thread = thread::Builder::new()
            .name("g1_task_policy".into())
            .spawn(move || {
                loop {
                    let request = {
                        let (mutex, wake) = &*shared;
                        let mut state = mutex.lock().unwrap_or_else(|error| error.into_inner());
                        while state.pending.is_none() && !state.stopping {
                            state = wake.wait(state).unwrap_or_else(|error| error.into_inner());
                        }
                        if state.stopping {
                            break;
                        }
                        state.pending.take().expect("pending request after wake")
                    };
                    let started = Instant::now();
                    let result = infer(&request);
                    let reply = PolicyInferenceReply {
                        profile: request.profile,
                        observation: request.observation.stamp,
                        sequence_id: request.sequence_id,
                        elapsed: started.elapsed(),
                        result,
                    };
                    let mut state = shared.0.lock().unwrap_or_else(|error| error.into_inner());
                    if state.stopping {
                        break;
                    }
                    if reply.observation.episode_id == state.episode_id {
                        state.completed = Some(reply);
                    }
                }
            })?;
        Ok(Self {
            state,
            thread: Some(thread),
        })
    }

    /// Replaces pending work; a stale episode cannot evict current work.
    pub fn submit_latest(
        &self,
        request: PolicyInferenceRequest,
    ) -> Result<Option<u64>, PolicyWorkerError> {
        let mut state = self
            .state
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.stopping {
            return Err(PolicyWorkerError::Stopped);
        }
        if request.observation.stamp.episode_id != state.episode_id {
            return Err(PolicyWorkerError::WrongEpisode);
        }
        let replaced = state
            .pending
            .replace(request)
            .map(|request| request.sequence_id);
        self.state.1.notify_one();
        Ok(replaced)
    }

    pub fn try_take_reply(&self) -> Option<PolicyInferenceReply> {
        self.state
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .completed
            .take()
    }

    /// Clear pending requests and replies before resetting scene/action history.
    /// Any in-flight result from an older episode is discarded on completion.
    pub fn reset_episode(&self, episode_id: u64) -> Result<(), PolicyWorkerError> {
        let mut state = self
            .state
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.stopping {
            return Err(PolicyWorkerError::Stopped);
        }
        if episode_id <= state.episode_id {
            return Err(PolicyWorkerError::StaleEpisode);
        }
        state.episode_id = episode_id;
        state.pending = None;
        state.completed = None;
        Ok(())
    }
}

impl Drop for PolicyWorker {
    fn drop(&mut self) {
        {
            let mut state = self
                .state
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.stopping = true;
            state.pending = None;
            state.completed = None;
            self.state.1.notify_one();
        }
        // Dropping JoinHandle detaches. The mandatory <=20s HTTP timeout lets the
        // one in-flight request finish without blocking UI/physics teardown.
        drop(self.thread.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyObservation;
    use std::sync::mpsc::{self, Receiver, Sender};

    fn request(episode_id: u64, sequence_id: u64) -> PolicyInferenceRequest {
        PolicyInferenceRequest {
            profile: TaskProfile::StaticApple,
            sequence_id,
            observation: PolicyObservation {
                stamp: ObservationStamp {
                    episode_id,
                    frame_id: sequence_id,
                    sim_time_ns: sequence_id * 20_000_000,
                    captured_at_unix_ms: sequence_id,
                },
                camera_rgb: vec![0; 640 * 480 * 3],
                camera_width: 640,
                camera_height: 480,
                measured_joint_positions_rad: [0.0; 31],
            },
        }
    }

    fn event(receiver: &Receiver<u64>, sequence_id: u64) {
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(2)).unwrap(),
            sequence_id
        );
    }

    // Every inference enters an explicit gate. Receiving its next entry proves
    // the previous result has passed through the worker's completion section.
    fn gated_worker() -> (PolicyWorker, Receiver<u64>, Sender<()>) {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = PolicyWorker::spawn_with(1, move |request| {
            entered_tx.send(request.sequence_id).unwrap();
            release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            Err(PolicyInferenceError::HttpStatus(503))
        })
        .unwrap();
        (worker, entered_rx, release_tx)
    }

    #[test]
    fn reset_discards_old_inflight_error_and_pending_work() {
        let (worker, entered, release) = gated_worker();
        worker.submit_latest(request(1, 1)).unwrap();
        event(&entered, 1);
        worker.submit_latest(request(1, 2)).unwrap();
        worker.reset_episode(2).unwrap();
        assert_eq!(
            worker.submit_latest(request(1, 3)),
            Err(PolicyWorkerError::WrongEpisode)
        );
        assert_eq!(
            worker.reset_episode(2),
            Err(PolicyWorkerError::StaleEpisode)
        );
        worker.submit_latest(request(2, 4)).unwrap();
        release.send(()).unwrap();
        event(&entered, 4);
        assert!(worker.try_take_reply().is_none());
        worker.submit_latest(request(2, 5)).unwrap();
        release.send(()).unwrap();
        event(&entered, 5);
        let reply = worker.try_take_reply().unwrap();
        assert_eq!(reply.profile, TaskProfile::StaticApple);
        assert_eq!(reply.observation, request(2, 4).observation.stamp);
        assert_eq!(reply.sequence_id, 4);
        assert_eq!(
            reply.result.unwrap_err(),
            PolicyInferenceError::HttpStatus(503)
        );
        assert!(reply.elapsed > Duration::ZERO);
        drop(worker);
        release.send(()).unwrap();
    }

    #[test]
    fn reset_discards_old_inflight_success_without_consumer_filtering() {
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let worker = PolicyWorker::spawn_with(1, move |request| {
            entered_tx.send(request.sequence_id).unwrap();
            release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            Ok(PolicyActionChunk {
                profile: request.profile,
                observation: request.observation.stamp,
                sequence_id: request.sequence_id,
                model_revision: super::super::profile_contract(request.profile)
                    .revision
                    .into(),
                action_period_ns: super::super::ARENA_ACTION_PERIOD_NS,
                frames: vec![
                    super::super::PolicyActionFrame {
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
            })
        })
        .unwrap();
        worker.submit_latest(request(1, 1)).unwrap();
        event(&entered, 1);
        worker.reset_episode(2).unwrap();
        worker.submit_latest(request(2, 2)).unwrap();
        release.send(()).unwrap();
        event(&entered, 2);
        assert!(worker.try_take_reply().is_none());
        drop(worker);
        release.send(()).unwrap();
        assert!(matches!(
            entered.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
    }

    #[test]
    fn overrun_keeps_only_latest_pending_and_latest_completed() {
        let (worker, entered, release) = gated_worker();
        worker.submit_latest(request(1, 1)).unwrap();
        event(&entered, 1);
        assert_eq!(worker.submit_latest(request(1, 2)).unwrap(), None);
        for sequence in 3..=20 {
            assert_eq!(
                worker.submit_latest(request(1, sequence)).unwrap(),
                Some(sequence - 1)
            );
        }
        release.send(()).unwrap();
        event(&entered, 20);
        worker.submit_latest(request(1, 21)).unwrap();
        release.send(()).unwrap();
        event(&entered, 21);
        assert_eq!(worker.try_take_reply().unwrap().sequence_id, 20);
        assert!(worker.try_take_reply().is_none());
        // Reset also clears the slot after it already holds a completed reply.
        worker.submit_latest(request(1, 22)).unwrap();
        release.send(()).unwrap();
        event(&entered, 22);
        worker.reset_episode(2).unwrap();
        assert!(worker.try_take_reply().is_none());
        drop(worker);
        release.send(()).unwrap();
    }

    #[test]
    fn drop_returns_while_inference_is_blocked_and_suppresses_pending_work() {
        let (worker, entered, release) = gated_worker();
        worker.submit_latest(request(1, 1)).unwrap();
        event(&entered, 1);
        worker.submit_latest(request(1, 2)).unwrap();
        let (dropped_tx, dropped_rx) = mpsc::channel();
        let drop_thread = thread::spawn(move || {
            drop(worker);
            dropped_tx.send(()).unwrap();
        });
        // The inference gate is still closed; joining in Drop would time out.
        dropped_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        drop_thread.join().unwrap();
        release.send(()).unwrap();
        assert!(matches!(
            entered.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
    }
}
