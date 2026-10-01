//! Bounded transport around a thread-owned candidate G1 world.
//!
//! The owner alone accepts commands and advances physics. Pause invalidates all
//! commands submitted before it; reset also invalidates published observations.
//! A cancelled in-flight inference requires reset because its history may have
//! advanced. Pausing is not a qualified physical standing stop.

use super::runner::{G1Measurement, G1ProgressCounts, G1Runner, G1RunnerConfig, G1Step};
use common_minigame::clock::FixedStepClock;
use robot_minigame::{
    RobotError,
    g1::{contract::G1Command, definition::G1BodyFrame},
};
use std::{
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, TryLockError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const PERIOD_NS: u64 = 20_000_000;
const MAX_PENDING_TICKS: u128 = 5;

#[derive(Clone, Debug)]
pub struct TimedG1Command {
    pub episode_id: u64,
    /// Exclusive command interval end, on the 20 ms simulation grid. The entire
    /// next integration interval must fit before or at this endpoint.
    pub valid_until_sim_ns: u64,
    /// Checked again after inference, immediately before applying its result.
    pub valid_until_wall: Instant,
    pub command: G1Command,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum G1WorkerPhase {
    Loading,
    Paused,
    Running,
    Failed,
    Stopped,
}

#[derive(Clone, Debug, Default)]
pub struct G1WorkerTiming {
    /// Lifetime counters include real integrations even when the boundary fails
    /// after integration, and include worlds discarded by an in-flight reset.
    pub total_integrations: u64,
    pub total_torque_updates: u64,
    pub total_inference_attempts: u64,
    pub total_successful_inferences: u64,
    /// Current episode counters come directly from the runner, not published frames.
    pub episode_integrations: u64,
    pub episode_torque_updates: u64,
    pub episode_inference_attempts: u64,
    pub episode_successful_inferences: u64,
    pub episode_halted: bool,
    /// Owner-active wall time and integrated time over the worker lifetime.
    pub active_wall_seconds: f64,
    pub active_sim_seconds: f64,
    pub episode_active_wall_seconds: f64,
    pub episode_active_sim_seconds: f64,
    /// Wall time since this episode loaded, including pauses.
    pub last_boundary_wall_ms: f64,
    pub last_boundary_duration_ms: f64,
    pub control_deadlines_missed: u64,
    pub pending_ticks: u64,
    /// Lifetime count of unread snapshots replaced in the display slot.
    pub display_updates_dropped: u64,
}

impl G1WorkerTiming {
    fn begin_episode(&mut self) {
        self.episode_integrations = 0;
        self.episode_torque_updates = 0;
        self.episode_inference_attempts = 0;
        self.episode_successful_inferences = 0;
        self.episode_halted = false;
        self.episode_active_wall_seconds = 0.;
        self.episode_active_sim_seconds = 0.;
        self.last_boundary_wall_ms = 0.;
        self.last_boundary_duration_ms = 0.;
        self.pending_ticks = 0;
    }

    fn record_progress(&mut self, counts: G1ProgressCounts) {
        self.total_integrations += counts.integration_count - self.episode_integrations;
        self.total_torque_updates += counts.torque_update_count - self.episode_torque_updates;
        self.total_inference_attempts +=
            counts.inference_attempt_count - self.episode_inference_attempts;
        self.total_successful_inferences +=
            counts.successful_inference_count - self.episode_successful_inferences;
        self.episode_integrations = counts.integration_count;
        self.episode_torque_updates = counts.torque_update_count;
        self.episode_inference_attempts = counts.inference_attempt_count;
        self.episode_successful_inferences = counts.successful_inference_count;
        self.episode_halted = counts.halted;
        self.active_sim_seconds = self.total_integrations as f64 * 0.02;
        self.episode_active_sim_seconds = self.episode_integrations as f64 * 0.02;
    }

    fn add_active_wall(&mut self, elapsed: Duration) {
        self.active_wall_seconds += elapsed.as_secs_f64();
        self.episode_active_wall_seconds += elapsed.as_secs_f64();
    }
}

#[derive(Clone, Debug)]
pub struct G1WorkerSnapshot {
    pub generation: u64,
    pub episode_id: u64,
    /// The pause barrier and command that produced a Running snapshot. Ordinary
    /// command replacement is applied at the next boundary, not mid-inference.
    pub pause_revision: u64,
    pub command_revision: u64,
    pub phase: G1WorkerPhase,
    pub reason: Option<String>,
    pub frame: Option<Arc<G1BodyFrame>>,
    pub measurement: Option<Arc<G1Measurement>>,
    pub step: Option<Arc<G1Step>>,
    pub timing: G1WorkerTiming,
}

#[derive(Clone)]
struct SubmittedCommand {
    revision: u64,
    generation: u64,
    pause_revision: u64,
    timed: TimedG1Command,
}

#[derive(Default)]
struct CommandSlot {
    latest: Option<SubmittedCommand>,
}

struct Shared {
    initial_episode: u64,
    shutdown: AtomicBool,
    pause_revision: AtomicU64,
    reset_count: AtomicU64,
    command_revision: AtomicU64,
    command: Mutex<CommandSlot>,
    output: Mutex<Option<Arc<G1WorkerSnapshot>>>,
    /// Owner termination must remain observable even if the display mutex was
    /// held at shutdown. This one-write latch never blocks the physics owner.
    stopped: OnceLock<Arc<G1WorkerSnapshot>>,
    stopped_consumed: AtomicBool,
}

impl Shared {
    fn new(initial_episode: u64) -> Self {
        Self {
            initial_episode,
            shutdown: AtomicBool::new(false),
            pause_revision: AtomicU64::new(0),
            reset_count: AtomicU64::new(0),
            command_revision: AtomicU64::new(0),
            command: Mutex::new(CommandSlot::default()),
            output: Mutex::new(None),
            stopped: OnceLock::new(),
            stopped_consumed: AtomicBool::new(false),
        }
    }

    fn pause(&self) {
        if self
            .pause_revision
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
            .is_err()
        {
            self.shutdown.store(true, Ordering::Release);
        }
    }

    fn reset(&self) -> Result<(), RobotError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(error("G1 worker is already stopping"));
        }
        self.pause();
        self.reset_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
                v.checked_add(1)
                    .filter(|&next| self.initial_episode.checked_add(next).is_some())
            })
            .map_err(|_| error("G1 reset generation or episode overflow"))?;
        Ok(())
    }

    fn submit(&self, command: TimedG1Command, now: Instant) -> Result<(), RobotError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(error("G1 worker is already stopping"));
        }
        command.command.validate()?;
        if now
            .checked_add(Duration::from_nanos(PERIOD_NS))
            .is_none_or(|end| end > command.valid_until_wall)
        {
            return Err(error("G1 wall deadline cannot cover a complete 20 ms tick"));
        }
        if command.valid_until_sim_ns == 0 || command.valid_until_sim_ns % PERIOD_NS != 0 {
            return Err(error("G1 simulation deadline must be on the 20 ms grid"));
        }
        let mut slot = self.command.lock().map_err(error)?;
        let generation = self.reset_count.load(Ordering::Acquire);
        if self.initial_episode.checked_add(generation) != Some(command.episode_id) {
            return Err(error("G1 command belongs to another episode"));
        }
        let revision = self
            .command_revision
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or_else(|| error("G1 command revision overflow"))?;
        slot.latest = Some(SubmittedCommand {
            revision,
            generation,
            pause_revision: self.pause_revision.load(Ordering::Acquire),
            timed: command,
        });
        // The release is the submission boundary. A simultaneous reset or pause
        // is detected by the owner; submission never clears their barriers.
        self.command_revision.store(revision, Ordering::Release);
        Ok(())
    }
}

/// The handle cannot access the Rapier world or ONNX session.
pub struct G1Worker {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl G1Worker {
    /// Load the real world and sessions on their owner, initially paused.
    pub fn spawn(config: G1RunnerConfig) -> Result<Self, RobotError> {
        let episode = config.episode_id;
        Self::spawn_owner(
            episode,
            move |episode_id| {
                G1Runner::load(&G1RunnerConfig {
                    episode_id,
                    ..config.clone()
                })
            },
            SystemClock,
        )
    }

    fn spawn_owner<R, F, C>(episode: u64, load: F, clock: C) -> Result<Self, RobotError>
    where
        R: BoundaryRunner + 'static,
        F: FnMut(u64) -> Result<R, RobotError> + Send + 'static,
        C: OwnerClock + 'static,
    {
        let shared = Arc::new(Shared::new(episode));
        let owner = shared.clone();
        let thread = thread::Builder::new()
            .name("g1-physics-50hz".into())
            .spawn(move || run_owner(episode, owner, load, clock))
            .map_err(error)?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Replace one pending command. This never directly resumes the owner.
    pub fn submit(&self, command: TimedG1Command) -> Result<(), RobotError> {
        self.shared.submit(command, Instant::now())
    }

    /// Invalidate all earlier commands. Only a subsequently submitted, valid
    /// current-episode command may resume at an owner boundary.
    pub fn pause(&self) {
        self.shared.pause();
    }

    /// Rebuild the world/history. Readers immediately reject older generations.
    pub fn reset(&self) -> Result<(), RobotError> {
        self.shared.reset()
    }

    pub fn take_latest(&self) -> Option<Arc<G1WorkerSnapshot>> {
        let generation = self.shared.reset_count.load(Ordering::Acquire);
        if let Some(stopped) = self.shared.stopped.get() {
            return (stopped.generation == generation
                && snapshot_is_current(&self.shared, stopped)
                && !self.shared.stopped_consumed.swap(true, Ordering::AcqRel))
            .then(|| stopped.clone());
        }
        let mut output = try_output(&self.shared)?;
        if generation != self.shared.reset_count.load(Ordering::Acquire) {
            return None;
        }
        let snapshot = output.take()?;
        snapshot_is_current(&self.shared, &snapshot).then_some(snapshot)
    }

    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    pub fn shutdown(mut self) -> Result<(), RobotError> {
        self.shared.shutdown.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| error("G1 physics worker panicked"))?;
        }
        Ok(())
    }
}

impl Drop for G1Worker {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The production implementation always returns real frames/steps. The private
/// test implementation returns None, so transport tests never fabricate physics.
trait BoundaryRunner: Send {
    fn initial_frame(&self) -> Result<Option<G1BodyFrame>, RobotError>;
    fn initial_measurement(&self) -> Result<Option<G1Measurement>, RobotError>;
    fn step(
        &mut self,
        command: &G1Command,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<Option<G1Step>, RobotError>;
    fn progress_counts(&self) -> G1ProgressCounts;
}

impl BoundaryRunner for G1Runner {
    fn initial_frame(&self) -> Result<Option<G1BodyFrame>, RobotError> {
        G1Runner::initial_frame(self).map(Some)
    }

    fn initial_measurement(&self) -> Result<Option<G1Measurement>, RobotError> {
        self.measurement().map(Some)
    }

    fn step(
        &mut self,
        command: &G1Command,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<Option<G1Step>, RobotError> {
        self.step_with_guard(command, guard).map(Some)
    }

    fn progress_counts(&self) -> G1ProgressCounts {
        G1Runner::progress_counts(self)
    }
}

trait OwnerClock: Send {
    fn now(&self) -> Instant;
    fn wait(&self, duration: Duration);
}

struct SystemClock;
impl OwnerClock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
    fn wait(&self, duration: Duration) {
        thread::sleep(duration);
    }
}

/// Nonblocking publication; the caller retains the latest snapshot for retries.
fn publish(shared: &Shared, snapshot: &mut G1WorkerSnapshot) -> bool {
    if !snapshot_is_current(shared, snapshot) {
        return false;
    }
    let Some(mut output) = try_output(shared) else {
        return false;
    };
    if !snapshot_is_current(shared, snapshot) {
        return false;
    }
    if output.is_some() {
        snapshot.timing.display_updates_dropped += 1;
    }
    *output = Some(Arc::new(snapshot.clone()));
    true
}

fn try_output(shared: &Shared) -> Option<MutexGuard<'_, Option<Arc<G1WorkerSnapshot>>>> {
    match shared.output.try_lock() {
        Ok(output) => Some(output),
        Err(TryLockError::WouldBlock) => None,
        // This mutex protects only an optional immutable display snapshot. A
        // reader panic cannot corrupt the owner world; recover the slot instead
        // of making all future failure notifications permanently invisible.
        Err(TryLockError::Poisoned(error)) => Some(error.into_inner()),
    }
}

fn snapshot_is_current(shared: &Shared, snapshot: &G1WorkerSnapshot) -> bool {
    snapshot.generation == shared.reset_count.load(Ordering::Acquire)
        && (snapshot.phase != G1WorkerPhase::Running
            || snapshot.pause_revision == shared.pause_revision.load(Ordering::Acquire))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cancellation {
    Shutdown,
    Reset,
    Pause,
    WallExpired,
    SimulationExpired,
}

fn check_command(
    shared: &Shared,
    command: &SubmittedCommand,
    generation: u64,
    episode: u64,
    sim_ns: u64,
    now: Instant,
) -> Result<(), Cancellation> {
    check_command_identity(shared, command, generation, episode)?;
    if now
        .checked_add(Duration::from_nanos(PERIOD_NS))
        .is_none_or(|end| end > command.timed.valid_until_wall)
    {
        return Err(Cancellation::WallExpired);
    }
    if sim_ns
        .checked_add(PERIOD_NS)
        .is_none_or(|end| end > command.timed.valid_until_sim_ns)
    {
        return Err(Cancellation::SimulationExpired);
    }
    Ok(())
}

fn check_command_identity(
    shared: &Shared,
    command: &SubmittedCommand,
    generation: u64,
    episode: u64,
) -> Result<(), Cancellation> {
    if shared.shutdown.load(Ordering::Acquire) {
        return Err(Cancellation::Shutdown);
    }
    if generation != shared.reset_count.load(Ordering::Acquire)
        || command.generation != generation
        || command.timed.episode_id != episode
    {
        return Err(Cancellation::Reset);
    }
    if command.pause_revision != shared.pause_revision.load(Ordering::Acquire) {
        return Err(Cancellation::Pause);
    }
    Ok(())
}

fn select_command(
    shared: &Shared,
    after_revision: u64,
    generation: u64,
    episode: u64,
    sim_ns: u64,
    now: Instant,
) -> Result<Option<SubmittedCommand>, RobotError> {
    let slot = match shared.command.try_lock() {
        Ok(slot) => slot,
        Err(TryLockError::WouldBlock) => return Ok(None),
        Err(TryLockError::Poisoned(e)) => return Err(error(e)),
    };
    Ok(slot
        .latest
        .as_ref()
        .filter(|command| {
            command.revision > after_revision
                && check_command(shared, command, generation, episode, sim_ns, now).is_ok()
        })
        .cloned())
}

fn fail<C: OwnerClock>(
    shared: &Shared,
    snapshot: &mut G1WorkerSnapshot,
    reason: String,
    clock: &C,
) {
    snapshot.phase = G1WorkerPhase::Failed;
    snapshot.reason = Some(reason);
    let mut delivered = false;
    while !shared.shutdown.load(Ordering::Acquire)
        && snapshot.generation == shared.reset_count.load(Ordering::Acquire)
    {
        if !delivered {
            delivered = publish(shared, snapshot);
        }
        clock.wait(Duration::from_millis(1));
    }
}

fn run_owner<R, F, C>(initial_episode: u64, shared: Arc<Shared>, mut load: F, wall: C)
where
    R: BoundaryRunner,
    F: FnMut(u64) -> Result<R, RobotError>,
    C: OwnerClock,
{
    let mut snapshot = G1WorkerSnapshot {
        generation: 0,
        episode_id: initial_episode,
        pause_revision: 0,
        command_revision: 0,
        phase: G1WorkerPhase::Loading,
        reason: None,
        frame: None,
        measurement: None,
        step: None,
        timing: G1WorkerTiming::default(),
    };
    'reload: loop {
        if shared.shutdown.load(Ordering::Acquire) {
            break;
        }
        let generation = shared.reset_count.load(Ordering::Acquire);
        let Some(episode) = initial_episode.checked_add(generation) else {
            fail(
                &shared,
                &mut snapshot,
                "G1 episode counter overflow".into(),
                &wall,
            );
            break;
        };
        snapshot.generation = generation;
        snapshot.episode_id = episode;
        snapshot.pause_revision = shared.pause_revision.load(Ordering::Acquire);
        snapshot.command_revision = 0;
        snapshot.phase = G1WorkerPhase::Loading;
        snapshot.reason = None;
        snapshot.frame = None;
        snapshot.measurement = None;
        snapshot.step = None;
        snapshot.timing.begin_episode();
        publish(&shared, &mut snapshot);
        let loaded = load(episode);
        if generation != shared.reset_count.load(Ordering::Acquire) {
            continue;
        }
        let mut runner = match loaded {
            Ok(runner) => runner,
            Err(e) => {
                fail(&shared, &mut snapshot, e.to_string(), &wall);
                continue;
            }
        };
        match runner.initial_frame() {
            Ok(frame) => snapshot.frame = frame.map(Arc::new),
            Err(e) => {
                fail(&shared, &mut snapshot, e.to_string(), &wall);
                continue;
            }
        }
        match runner.initial_measurement() {
            Ok(measurement) => snapshot.measurement = measurement.map(Arc::new),
            Err(e) => {
                fail(&shared, &mut snapshot, e.to_string(), &wall);
                continue;
            }
        }
        snapshot.timing.record_progress(runner.progress_counts());
        // Submissions linearized before this loaded-world barrier are discarded.
        let after_revision = shared.command_revision.load(Ordering::Acquire);
        let mut clock = FixedStepClock::new(50, 8).expect("constant G1 clock is valid");
        let mut last_wall = wall.now();
        // Scheduling and evidence have separate anchors: recording boundary
        // work must not remove inference/PD/integration time from clock debt.
        let mut last_clock_wall = last_wall;
        let epoch_wall = last_wall;
        let mut was_active = false;
        snapshot.phase = G1WorkerPhase::Paused;
        snapshot.reason = Some("candidate loaded; awaiting a fresh current-episode command".into());
        let mut pending_publication = !publish(&shared, &mut snapshot);
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                break 'reload;
            }
            if generation != shared.reset_count.load(Ordering::Acquire) {
                continue 'reload;
            }
            if pending_publication {
                pending_publication = !publish(&shared, &mut snapshot);
            }
            let now = wall.now();
            let elapsed = now.saturating_duration_since(last_wall);
            let clock_elapsed = now.saturating_duration_since(last_clock_wall);
            last_wall = now;
            last_clock_wall = now;
            if was_active {
                snapshot.timing.add_active_wall(elapsed);
            }
            let sim_ns = snapshot
                .timing
                .episode_integrations
                .saturating_mul(PERIOD_NS);
            let selected =
                match select_command(&shared, after_revision, generation, episode, sim_ns, now) {
                    Ok(selected) => selected,
                    Err(e) => {
                        fail(&shared, &mut snapshot, e.to_string(), &wall);
                        continue 'reload;
                    }
                };
            if selected.is_none() {
                was_active = false;
                clock.set_paused(true);
                snapshot.pause_revision = shared.pause_revision.load(Ordering::Acquire);
                if snapshot.phase != G1WorkerPhase::Paused {
                    snapshot.phase = G1WorkerPhase::Paused;
                    snapshot.reason = Some("paused or no valid current-episode command".into());
                    pending_publication = !publish(&shared, &mut snapshot);
                }
                wall.wait(Duration::from_millis(1));
                continue;
            }
            clock.set_paused(false);
            // Resuming adds none of the elapsed paused interval. Pre-pause debt
            // remains explicit, as required by FixedStepClock's contract.
            let budget = clock.begin_frame(if was_active {
                clock_elapsed
            } else {
                Duration::ZERO
            });
            was_active = true;
            snapshot.timing.pending_ticks = clock.pending_ticks() as u64;
            if clock.pending_ticks() > MAX_PENDING_TICKS {
                fail(
                    &shared,
                    &mut snapshot,
                    "physics fell more than 100 ms behind; reset required".into(),
                    &wall,
                );
                continue 'reload;
            }
            for _ in 0..budget {
                // Refresh the one-slot command at every integration boundary,
                // including each catch-up tick; never reuse an outer-loop clone.
                let sim_ns = snapshot
                    .timing
                    .episode_integrations
                    .saturating_mul(PERIOD_NS);
                let command = match select_command(
                    &shared,
                    after_revision,
                    generation,
                    episode,
                    sim_ns,
                    wall.now(),
                ) {
                    Ok(Some(command)) => command,
                    Ok(None) => break,
                    Err(e) => {
                        fail(&shared, &mut snapshot, e.to_string(), &wall);
                        continue 'reload;
                    }
                };
                let started = wall.now();
                let before_integrations = snapshot.timing.episode_integrations;
                let mut cancelled = None;
                let mut guard = || {
                    check_command(&shared, &command, generation, episode, sim_ns, wall.now())
                        .map_err(|reason| {
                            cancelled = Some(reason);
                            error(format!("G1 boundary cancelled: {reason:?}"))
                        })
                };
                let result = runner.step(&command.timed.command, &mut guard);
                let ended = wall.now();
                snapshot.timing.record_progress(runner.progress_counts());
                let integration_delta = snapshot.timing.episode_integrations - before_integrations;
                if integration_delta == 1 {
                    // A completed integration is real work even if assembling
                    // its observation subsequently failed. Never derive this
                    // ledger entry from the existence of a returned frame.
                    clock
                        .commit_tick()
                        .expect("one commit per actual integration");
                    snapshot.timing.pending_ticks = clock.pending_ticks() as u64;
                }
                snapshot.timing.last_boundary_duration_ms =
                    ended.saturating_duration_since(started).as_secs_f64() * 1000.;
                snapshot.timing.last_boundary_wall_ms =
                    ended.saturating_duration_since(epoch_wall).as_secs_f64() * 1000.;
                if ended.saturating_duration_since(started) > Duration::from_millis(20) {
                    snapshot.timing.control_deadlines_missed += 1;
                }
                // Account this real owner-active interval even on a failed or
                // discarded boundary; the next poll starts at the new anchor.
                snapshot
                    .timing
                    .add_active_wall(ended.saturating_duration_since(last_wall));
                last_wall = ended;
                if generation != shared.reset_count.load(Ordering::Acquire) {
                    continue 'reload;
                }
                if shared.shutdown.load(Ordering::Acquire) {
                    break 'reload;
                }
                if integration_delta > 1 || (result.is_ok() && integration_delta != 1) {
                    fail(
                        &shared,
                        &mut snapshot,
                        "boundary violated one-integration contract; reset required".into(),
                        &wall,
                    );
                    continue 'reload;
                }
                match result {
                    Ok(step) => {
                        // A completed same-episode frame remains authoritative
                        // even if pause arrived during its integration. Capture
                        // it and diagnose falls before the pause branch; reset
                        // still prevents any old-episode publication below.
                        snapshot.command_revision = command.revision;
                        if let Some(step) = step {
                            let fell = step.root_position_source[2] < 0.35
                                || step.root_upright_cosine < 0.5;
                            snapshot.frame = Some(Arc::new(step.frame.clone()));
                            snapshot.measurement = Some(Arc::new(step.measurement.clone()));
                            snapshot.step = Some(Arc::new(step));
                            if fell {
                                fail(
                                    &shared,
                                    &mut snapshot,
                                    "candidate fell; reset required".into(),
                                    &wall,
                                );
                                continue 'reload;
                            }
                        }
                        if ended > command.timed.valid_until_wall {
                            fail(
                                &shared,
                                &mut snapshot,
                                "completed boundary exceeded wall deadline; reset required".into(),
                                &wall,
                            );
                            continue 'reload;
                        }
                        // A reset after the previous check is rejected again by
                        // publication/consumption. Pause during non-cancellable
                        // integration never emits Running for that old barrier,
                        // but its real work stays counted. A normal replacement
                        // does not invalidate this completed transaction.
                        match check_command_identity(&shared, &command, generation, episode) {
                            Ok(()) => {}
                            Err(Cancellation::Reset) => continue 'reload,
                            Err(Cancellation::Shutdown) => break 'reload,
                            Err(Cancellation::Pause) => {
                                was_active = false;
                                clock.set_paused(true);
                                snapshot.phase = G1WorkerPhase::Paused;
                                snapshot.reason = Some("paused during completed boundary".into());
                                snapshot.pause_revision =
                                    shared.pause_revision.load(Ordering::Acquire);
                                pending_publication = !publish(&shared, &mut snapshot);
                                break;
                            }
                            Err(Cancellation::WallExpired | Cancellation::SimulationExpired) => {
                                unreachable!("identity check has no deadline branch")
                            }
                        }
                        snapshot.phase = G1WorkerPhase::Running;
                        snapshot.reason = None;
                        snapshot.pause_revision = command.pause_revision;
                        snapshot.command_revision = command.revision;
                        pending_publication = !publish(&shared, &mut snapshot);
                    }
                    Err(e) => {
                        let reason = cancelled.map_or_else(|| e.to_string(), |cause| {
                            format!("boundary cancelled ({cause:?}); inference history discarded; reset required")
                        });
                        fail(&shared, &mut snapshot, reason, &wall);
                        continue 'reload;
                    }
                }
            }
            if budget == 0 {
                wall.wait(Duration::from_micros(500));
            }
        }
    }
    let generation = shared.reset_count.load(Ordering::Acquire);
    if snapshot.generation != generation {
        snapshot.generation = generation;
        snapshot.episode_id = initial_episode
            .checked_add(generation)
            .expect("reset validates episode overflow");
        snapshot.frame = None;
        snapshot.measurement = None;
        snapshot.step = None;
        snapshot.timing.begin_episode();
    }
    snapshot.phase = G1WorkerPhase::Stopped;
    let _ = shared.stopped.set(Arc::new(snapshot));
}

fn error(e: impl std::fmt::Display) -> RobotError {
    RobotError::Contract(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::VecDeque,
        sync::mpsc::{self, Receiver, RecvTimeoutError, Sender},
    };

    /// Each owner wait is an acknowledged boundary. Tests advance this clock
    /// explicitly; no sleep duration or host scheduling decides a test result.
    #[derive(Clone)]
    struct ManualTime(Arc<Mutex<Instant>>);

    impl ManualTime {
        fn now(&self) -> Instant {
            *self.0.lock().unwrap()
        }

        fn advance(&self, duration: Duration) {
            let mut now = self.0.lock().unwrap();
            *now += duration;
        }
    }

    struct TestClock {
        time: ManualTime,
        shared: Arc<Shared>,
        waiting: Sender<()>,
        resume: Receiver<()>,
    }

    impl OwnerClock for TestClock {
        fn now(&self) -> Instant {
            self.time.now()
        }

        fn wait(&self, _duration: Duration) {
            if self.waiting.send(()).is_err() {
                return;
            }
            while !self.shared.shutdown.load(Ordering::Acquire) {
                match self.resume.recv_timeout(Duration::from_millis(10)) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        }
    }

    #[derive(Default)]
    struct BoundaryPlan {
        during_inference: Option<Box<dyn FnOnce() + Send>>,
        after_integration: Option<Box<dyn FnOnce() + Send>>,
        fail_after_integration: bool,
    }

    #[derive(Default)]
    struct TestControl {
        plans: Mutex<VecDeque<BoundaryPlan>>,
        loads: Mutex<Vec<u64>>,
        commands: Mutex<Vec<(u64, f32)>>,
    }

    /// This runner exercises the production owner and guard call order while
    /// returning no physical frames. Its counts are transport fixtures only.
    struct TestRunner {
        episode: u64,
        control: Arc<TestControl>,
        counts: G1ProgressCounts,
    }

    impl BoundaryRunner for TestRunner {
        fn initial_frame(&self) -> Result<Option<G1BodyFrame>, RobotError> {
            Ok(None)
        }

        fn initial_measurement(&self) -> Result<Option<G1Measurement>, RobotError> {
            Ok(None)
        }

        fn step(
            &mut self,
            command: &G1Command,
            guard: &mut dyn FnMut() -> Result<(), RobotError>,
        ) -> Result<Option<G1Step>, RobotError> {
            let mut plan = self
                .control
                .plans
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_default();
            let result = (|| {
                if self.counts.halted {
                    return Err(error("test runner halted"));
                }
                guard()?;
                self.control
                    .commands
                    .lock()
                    .unwrap()
                    .push((self.episode, command.pelvis_height));
                self.counts.inference_attempt_count += 1;
                self.counts.successful_inference_count += 1;
                if let Some(hook) = plan.during_inference.take() {
                    hook();
                }
                guard()?;
                self.counts.torque_update_count += 1;
                self.counts.integration_count += 1;
                if let Some(hook) = plan.after_integration.take() {
                    hook();
                }
                if plan.fail_after_integration {
                    return Err(error("fixture failed after actual boundary work"));
                }
                Ok(None)
            })();
            if result.is_err() {
                self.counts.halted = true;
            }
            result
        }

        fn progress_counts(&self) -> G1ProgressCounts {
            self.counts
        }
    }

    struct Harness {
        worker: G1Worker,
        time: ManualTime,
        control: Arc<TestControl>,
        waiting: Receiver<()>,
        resume: Sender<()>,
    }

    impl Harness {
        fn new() -> Self {
            Self::start(Arc::new(Shared::new(7)), false)
        }

        fn start(shared: Arc<Shared>, fail_first_load: bool) -> Self {
            let time = ManualTime(Arc::new(Mutex::new(Instant::now())));
            let control = Arc::new(TestControl::default());
            let (waiting_tx, waiting) = mpsc::channel();
            let (resume, resume_rx) = mpsc::channel();
            let clock = TestClock {
                time: time.clone(),
                shared: shared.clone(),
                waiting: waiting_tx,
                resume: resume_rx,
            };
            let owner = shared.clone();
            let factory_control = control.clone();
            let mut fail_load = fail_first_load;
            let thread = thread::spawn(move || {
                run_owner(
                    7,
                    owner,
                    move |episode| {
                        factory_control.loads.lock().unwrap().push(episode);
                        if std::mem::take(&mut fail_load) {
                            return Err(error("fixture load failure"));
                        }
                        Ok(TestRunner {
                            episode,
                            control: factory_control.clone(),
                            counts: G1ProgressCounts {
                                integration_count: 0,
                                torque_update_count: 0,
                                inference_attempt_count: 0,
                                successful_inference_count: 0,
                                halted: false,
                            },
                        })
                    },
                    clock,
                );
            });
            let harness = Self {
                worker: G1Worker {
                    shared,
                    thread: Some(thread),
                },
                time,
                control,
                waiting,
                resume,
            };
            harness.await_wait();
            harness
        }

        fn await_wait(&self) {
            self.waiting
                .recv_timeout(Duration::from_secs(5))
                .expect("owner must reach an acknowledged wait");
        }

        fn poll(&self, elapsed: Duration) {
            self.time.advance(elapsed);
            self.resume.send(()).unwrap();
            self.await_wait();
        }

        fn timed_command(
            &self,
            episode: u64,
            sim_deadline: u64,
            wall_lifetime: Duration,
            height: f32,
        ) -> TimedG1Command {
            TimedG1Command {
                episode_id: episode,
                valid_until_sim_ns: sim_deadline,
                valid_until_wall: self.time.now() + wall_lifetime,
                command: G1Command {
                    pelvis_height: height,
                    ..G1Command::default()
                },
            }
        }

        fn submit(
            &self,
            episode: u64,
            sim_deadline: u64,
            wall_lifetime: Duration,
            height: f32,
        ) -> Result<(), RobotError> {
            self.worker.shared.submit(
                self.timed_command(episode, sim_deadline, wall_lifetime, height),
                self.time.now(),
            )
        }

        fn resume_current(&self) {
            self.submit(7, 1_000_000_000, Duration::from_secs(2), 0.75)
                .unwrap();
            self.poll(Duration::ZERO);
        }

        fn plan(&self, plan: BoundaryPlan) {
            self.control.plans.lock().unwrap().push_back(plan);
        }

        fn latest(&self) -> Arc<G1WorkerSnapshot> {
            let snapshot = self
                .worker
                .take_latest()
                .expect("current snapshot expected");
            assert!(
                snapshot.frame.is_none()
                    && snapshot.measurement.is_none()
                    && snapshot.step.is_none(),
                "transport fixture cannot certify physical behavior"
            );
            snapshot
        }
    }

    #[test]
    fn terminal_failure_retries_nonblocking_publication_and_requires_reset() {
        let shared = Arc::new(Shared::new(7));
        let output = shared.output.lock().unwrap();
        let harness = Harness::start(shared.clone(), true);
        assert!(
            output.is_none(),
            "owner did not block behind the display lock"
        );
        drop(output);
        harness.poll(Duration::ZERO);
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert_eq!(failed.timing.total_integrations, 0);
        harness
            .submit(7, PERIOD_NS, Duration::from_secs(1), 0.75)
            .unwrap();
        harness.poll(Duration::from_millis(20));
        assert!(harness.control.commands.lock().unwrap().is_empty());
        harness.worker.reset().unwrap();
        harness.poll(Duration::ZERO);
        let reset = harness.latest();
        assert_eq!(
            (reset.generation, reset.episode_id, reset.phase.clone()),
            (1, 8, G1WorkerPhase::Paused)
        );
    }

    #[test]
    fn reset_return_invalidates_unconsumed_snapshot_immediately() {
        let harness = Harness::new();
        harness.worker.reset().unwrap();
        assert!(harness.worker.take_latest().is_none());
        harness.poll(Duration::ZERO);
        assert_eq!(harness.latest().episode_id, 8);
    }

    #[test]
    fn reset_after_integration_discards_output_but_preserves_lifetime_work() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        harness.plan(BoundaryPlan {
            after_integration: Some(Box::new(move || shared.reset().unwrap())),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let reset = harness.latest();
        assert_eq!((reset.generation, reset.episode_id), (1, 8));
        assert_eq!(reset.phase, G1WorkerPhase::Paused);
        assert_eq!(reset.timing.total_integrations, 1);
        assert_eq!(reset.timing.total_torque_updates, 1);
        assert_eq!(reset.timing.total_inference_attempts, 1);
        assert_eq!(reset.timing.episode_integrations, 0);
        assert_eq!(reset.timing.active_sim_seconds, 0.02);
        assert_eq!(reset.timing.episode_active_sim_seconds, 0.);
    }

    #[test]
    fn reset_during_inference_cancels_old_world_before_pd() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || shared.reset().unwrap())),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let reset = harness.latest();
        assert_eq!((reset.generation, reset.episode_id), (1, 8));
        assert_eq!(reset.phase, G1WorkerPhase::Paused);
        assert_eq!(reset.timing.total_inference_attempts, 1);
        assert_eq!(reset.timing.total_successful_inferences, 1);
        assert_eq!(reset.timing.total_torque_updates, 0);
        assert_eq!(reset.timing.total_integrations, 0);
        assert_eq!(reset.timing.episode_inference_attempts, 0);
    }

    #[test]
    fn pause_during_inference_rejects_pd_and_cannot_resume_mutated_history() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || shared.pause())),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert!(failed.reason.as_ref().unwrap().contains("Pause"));
        assert_eq!(failed.timing.total_inference_attempts, 1);
        assert_eq!(failed.timing.total_successful_inferences, 1);
        assert_eq!(failed.timing.total_torque_updates, 0);
        assert_eq!(failed.timing.total_integrations, 0);
        assert!(failed.timing.episode_halted);
        harness
            .submit(7, 1_000_000_000, Duration::from_secs(2), 0.8)
            .unwrap();
        harness.poll(Duration::from_secs(1));
        assert_eq!(harness.control.commands.lock().unwrap().len(), 1);
        harness.worker.reset().unwrap();
        harness.poll(Duration::ZERO);
        let reset = harness.latest();
        assert_eq!(reset.timing.total_inference_attempts, 1);
        assert_eq!(reset.timing.episode_inference_attempts, 0);
        assert!(!reset.timing.episode_halted);
    }

    #[test]
    fn ttl_expiring_during_inference_prevents_torque_and_integration() {
        let harness = Harness::new();
        let time = harness.time.clone();
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || time.advance(Duration::from_millis(11)))),
            ..BoundaryPlan::default()
        });
        harness
            .submit(7, PERIOD_NS, Duration::from_millis(50), 0.75)
            .unwrap();
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_millis(20));
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert!(failed.reason.as_ref().unwrap().contains("WallExpired"));
        assert_eq!(failed.timing.total_successful_inferences, 1);
        assert_eq!(failed.timing.total_torque_updates, 0);
        assert_eq!(failed.timing.total_integrations, 0);
        assert_eq!(failed.timing.last_boundary_duration_ms, 11.);
    }

    #[test]
    fn fresh_replacement_does_not_extend_inflight_command_deadline() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        let time = harness.time.clone();
        let replacement = harness.timed_command(7, 1_000_000_000, Duration::from_secs(2), 0.8);
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || {
                time.advance(Duration::from_millis(11));
                shared.submit(replacement, time.now()).unwrap();
            })),
            ..BoundaryPlan::default()
        });
        harness
            .submit(7, PERIOD_NS, Duration::from_millis(50), 0.75)
            .unwrap();
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_millis(20));
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert_eq!(failed.timing.total_integrations, 0);
        assert!(failed.reason.as_ref().unwrap().contains("WallExpired"));
    }

    #[test]
    fn complete_tick_must_fit_simulation_and_wall_deadlines() {
        let harness = Harness::new();
        assert!(
            harness
                .submit(7, PERIOD_NS - 1, Duration::from_secs(1), 0.75)
                .is_err()
        );
        assert!(
            harness
                .submit(7, PERIOD_NS, Duration::from_nanos(1), 0.75)
                .is_err()
        );
        harness
            .submit(7, PERIOD_NS, Duration::from_secs(1), 0.75)
            .unwrap();
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_millis(20));
        // The completed tick consumes this command's whole interval. The owner
        // publishes Paused before its next wait, replacing the Running update.
        let expired = harness.latest();
        assert_eq!(expired.phase, G1WorkerPhase::Paused);
        assert_eq!(expired.timing.total_integrations, 1);
        harness.poll(Duration::from_millis(20));
        assert_eq!(harness.control.commands.lock().unwrap().len(), 1);
        assert!(harness.worker.take_latest().is_none());
    }

    #[test]
    fn one_nanosecond_wall_remainder_never_enters_runner() {
        let harness = Harness::new();
        harness
            .submit(7, PERIOD_NS, Duration::from_millis(40), 0.75)
            .unwrap();
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_nanos(40_000_000 - 1));
        assert!(harness.control.commands.lock().unwrap().is_empty());
        assert_eq!(harness.latest().phase, G1WorkerPhase::Paused);
    }

    #[test]
    fn old_episode_submit_cannot_replace_or_resume_pre_pause_command() {
        let harness = Harness::new();
        harness
            .submit(7, 1_000_000_000, Duration::from_secs(2), 0.75)
            .unwrap();
        harness.worker.pause();
        let revision = harness
            .worker
            .shared
            .command_revision
            .load(Ordering::Acquire);
        assert!(
            harness
                .submit(6, 1_000_000_000, Duration::from_secs(2), 0.8)
                .is_err()
        );
        assert_eq!(
            harness
                .worker
                .shared
                .command_revision
                .load(Ordering::Acquire),
            revision
        );
        harness.poll(Duration::from_millis(20));
        assert!(harness.control.commands.lock().unwrap().is_empty());
        harness.worker.reset().unwrap();
        assert!(
            harness
                .submit(7, 1_000_000_000, Duration::from_secs(2), 0.8)
                .is_err()
        );
        harness.poll(Duration::ZERO);
        assert_eq!(harness.latest().phase, G1WorkerPhase::Paused);
    }

    #[test]
    fn pause_invalidates_running_output_and_one_second_adds_no_ticks() {
        let harness = Harness::new();
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        harness.worker.pause();
        assert!(
            harness.worker.take_latest().is_none(),
            "a pre-pause Running output must not be consumed"
        );
        harness.poll(Duration::ZERO);
        assert_eq!(harness.latest().phase, G1WorkerPhase::Paused);
        harness.poll(Duration::from_secs(1));
        harness
            .submit(7, 1_000_000_000, Duration::from_secs(2), 0.8)
            .unwrap();
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_millis(20));
        let resumed = harness.latest();
        assert_eq!(resumed.phase, G1WorkerPhase::Running);
        assert_eq!(resumed.timing.total_integrations, 2);
        assert_eq!(resumed.timing.pending_ticks, 0);
        assert!((resumed.timing.active_wall_seconds - 0.04).abs() < 1e-12);
    }

    #[test]
    fn command_mutex_contention_never_blocks_owner_or_reuses_old_command() {
        let harness = Harness::new();
        harness.resume_current();
        let slot = harness.worker.shared.command.lock().unwrap();
        harness.poll(Duration::from_millis(40));
        assert!(harness.control.commands.lock().unwrap().is_empty());
        drop(slot);
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_millis(20));
        let resumed = harness.latest();
        assert_eq!(resumed.timing.total_integrations, 1);
        assert_eq!(resumed.timing.pending_ticks, 0);
    }

    #[test]
    fn every_catch_up_boundary_refreshes_latest_command() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        let time = harness.time.clone();
        let replacement = harness.timed_command(7, 1_000_000_000, Duration::from_secs(2), 0.8);
        harness.plan(BoundaryPlan {
            after_integration: Some(Box::new(move || {
                shared.submit(replacement, time.now()).unwrap()
            })),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(60));
        assert_eq!(
            *harness.control.commands.lock().unwrap(),
            [(7, 0.75), (7, 0.8), (7, 0.8)]
        );
        assert_eq!(harness.latest().timing.total_integrations, 3);
    }

    #[test]
    fn ordinary_replacement_during_inference_applies_only_next_boundary() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        let time = harness.time.clone();
        let replacement = harness.timed_command(7, 1_000_000_000, Duration::from_secs(2), 0.8);
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || {
                shared.submit(replacement, time.now()).unwrap()
            })),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let first = harness.latest();
        assert_eq!(first.phase, G1WorkerPhase::Running);
        assert_eq!(first.command_revision, 1);
        assert_eq!(first.timing.total_integrations, 1);
        assert!(!first.timing.episode_halted);
        harness.poll(Duration::from_millis(20));
        let second = harness.latest();
        assert_eq!(second.command_revision, 2);
        assert_eq!(second.timing.total_integrations, 2);
        assert_eq!(
            *harness.control.commands.lock().unwrap(),
            [(7, 0.75), (7, 0.8)]
        );
    }

    #[test]
    fn inference_wall_time_remains_in_fixed_clock_debt() {
        let harness = Harness::new();
        let time = harness.time.clone();
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || time.advance(Duration::from_millis(20)))),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let current = harness.latest();
        assert_eq!(current.phase, G1WorkerPhase::Running);
        assert_eq!(current.timing.total_integrations, 2);
        assert_eq!(current.timing.active_sim_seconds, 0.04);
        assert!((current.timing.active_wall_seconds - 0.04).abs() < 1e-12);
        assert_eq!(current.timing.pending_ticks, 0);
    }

    #[test]
    fn inference_stall_retains_debt_and_fails_instead_of_silent_time_loss() {
        let harness = Harness::new();
        let time = harness.time.clone();
        harness.plan(BoundaryPlan {
            during_inference: Some(Box::new(move || time.advance(Duration::from_millis(120)))),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert_eq!(failed.timing.total_integrations, 1);
        assert_eq!(failed.timing.pending_ticks, 6);
        assert_eq!(failed.timing.control_deadlines_missed, 1);
        assert!((failed.timing.active_wall_seconds - 0.14).abs() < 1e-12);
    }

    #[test]
    fn completed_integration_that_overruns_wall_ttl_is_counted_and_halts_owner() {
        let harness = Harness::new();
        let time = harness.time.clone();
        harness.plan(BoundaryPlan {
            after_integration: Some(Box::new(move || time.advance(Duration::from_millis(31)))),
            ..BoundaryPlan::default()
        });
        harness
            .submit(7, PERIOD_NS, Duration::from_millis(50), 0.75)
            .unwrap();
        harness.poll(Duration::ZERO);
        harness.poll(Duration::from_millis(20));
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert!(
            failed
                .reason
                .as_ref()
                .unwrap()
                .contains("exceeded wall deadline")
        );
        assert_eq!(failed.timing.total_integrations, 1);
        assert_eq!(failed.timing.total_torque_updates, 1);
        assert_eq!(failed.timing.control_deadlines_missed, 1);
        assert_eq!(failed.timing.pending_ticks, 0);
    }

    #[test]
    fn pause_during_completed_boundary_never_publishes_running() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        harness.plan(BoundaryPlan {
            after_integration: Some(Box::new(move || shared.pause())),
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let paused = harness.latest();
        assert_eq!(paused.phase, G1WorkerPhase::Paused);
        assert_eq!(paused.timing.total_integrations, 1);
        assert_eq!(paused.command_revision, 1);
        harness.poll(Duration::from_secs(1));
        assert_eq!(harness.control.commands.lock().unwrap().len(), 1);
    }

    #[test]
    fn actual_work_after_partial_failure_is_not_inferred_from_successful_frames() {
        let harness = Harness::new();
        harness.plan(BoundaryPlan {
            fail_after_integration: true,
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        let failed = harness.latest();
        assert_eq!(failed.phase, G1WorkerPhase::Failed);
        assert_eq!(failed.timing.total_integrations, 1);
        assert_eq!(failed.timing.total_torque_updates, 1);
        assert_eq!(failed.timing.episode_integrations, 1);
        assert_eq!(failed.timing.pending_ticks, 0);
        assert!(failed.timing.episode_halted);
        harness.worker.reset().unwrap();
        harness.poll(Duration::ZERO);
        let reset = harness.latest();
        assert_eq!(reset.timing.total_integrations, 1);
        assert_eq!(reset.timing.episode_integrations, 0);
    }

    #[test]
    fn stopped_latch_survives_display_lock_contention_without_blocking_shutdown() {
        let mut harness = Harness::new();
        let shared = harness.worker.shared.clone();
        let output = shared.output.lock().unwrap();
        shared.shutdown.store(true, Ordering::Release);
        harness.worker.thread.take().unwrap().join().unwrap();
        let stopped = harness.latest();
        assert_eq!(stopped.phase, G1WorkerPhase::Stopped);
        assert!(harness.worker.take_latest().is_none());
        assert!(harness.worker.reset().is_err());
        assert!(
            harness
                .submit(7, PERIOD_NS, Duration::from_secs(1), 0.75)
                .is_err()
        );
        drop(output);
    }

    #[test]
    fn poisoned_display_slot_cannot_permanently_hide_failure() {
        let harness = Harness::new();
        let shared = harness.worker.shared.clone();
        assert!(
            thread::spawn(move || {
                let _output = shared.output.lock().unwrap();
                panic!("fixture display reader panic");
            })
            .join()
            .is_err()
        );
        assert_eq!(harness.latest().phase, G1WorkerPhase::Paused);
        harness.plan(BoundaryPlan {
            fail_after_integration: true,
            ..BoundaryPlan::default()
        });
        harness.resume_current();
        harness.poll(Duration::from_millis(20));
        assert_eq!(harness.latest().phase, G1WorkerPhase::Failed);
    }
}
