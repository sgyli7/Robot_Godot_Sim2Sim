//! Development-only live window over the actual 60 Hz diagnostic world.
//!
//! The robot is driven by the existing P-only diagnostic controller. Neither
//! this window nor its report qualifies contact, BAM, props, or a skill.

use bevy::{
    app::AppExit,
    diagnostic::{DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        diagnostic::{MeshAllocatorDiagnosticPlugin, RenderDiagnosticsPlugin},
        render_resource::{CachedPipelineState, PipelineCache, PollType, WgpuFeatures},
        renderer::{RenderAdapterInfo, RenderDevice},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    shader::{Shader, ShaderCacheError},
};
use dev_tools_minigame::station_robot_diagnostic::{DiagnosticSession, initial_report, prepare};
use rendering_minigame::{
    RobotVisualInput, RobotVisualModel, RobotVisualPhase, RobotVisualPlugin, RobotVisualStatus,
    RobotVisualSystems, StationCameraControl, StationLabelBakeStatus, StationScene, StationView,
    StationVisualPlugin, robot_mesh::VerifiedRobotAppearance,
};
use robot_minigame::body_pose::RobotPoseFrame;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{RapierCounterSample, fixed_step_runtime::FrameOutcome};
use std::{
    env::{self, VarError},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const LIVE_SCHEMA: &str = "station_robot_live_preview_v3";
const PROFILE_ENV: &str = "SAI_LAB_LIVE_PROFILE";
const PROFILE_RESOLUTION_ENV: &str = "SAI_LAB_LIVE_PROFILE_RESOLUTION";
const PROFILE_PRESENT_MODE_ENV: &str = "SAI_LAB_LIVE_PROFILE_PRESENT_MODE";
const PROFILE_ABLATION_ENV: &str = "SAI_LAB_LIVE_PROFILE_ABLATION";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RenderAblation {
    #[default]
    None,
    SignageCameras,
    Msaa,
    Shadows,
}

impl RenderAblation {
    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SignageCameras => "signage_cameras",
            Self::Msaa => "msaa",
            Self::Shadows => "shadows",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveProfileConfig {
    width: u32,
    height: u32,
    present_mode: bevy::window::PresentMode,
    ablation: RenderAblation,
}

impl LiveProfileConfig {
    fn requested_config(self) -> Value {
        let present_mode = match self.present_mode {
            bevy::window::PresentMode::AutoVsync => "auto_vsync",
            bevy::window::PresentMode::AutoNoVsync => "auto_no_vsync",
            bevy::window::PresentMode::Fifo => "fifo",
            bevy::window::PresentMode::Mailbox => "mailbox",
            _ => unreachable!("profile parser admits only the supported requested modes"),
        };
        json!({
            "requested_resolution": [self.width, self.height],
            "requested_present_mode": present_mode,
            "render_ablation": self.ablation.label(),
        })
    }
}

fn parse_live_profile_config_with_ablation(
    enabled: Option<&str>,
    resolution: Option<&str>,
    present_mode: Option<&str>,
    ablation: Option<&str>,
) -> Result<Option<LiveProfileConfig>, String> {
    match enabled {
        None if resolution.is_none() && present_mode.is_none() && ablation.is_none() => {
            return Ok(None);
        }
        None => return Err(format!("{PROFILE_ENV}=1 is required for profile settings")),
        Some("1") => {}
        Some(_) => return Err(format!("{PROFILE_ENV} must be exactly 1")),
    }
    let (width, height) = match resolution.unwrap_or("1920x1080") {
        "1920x1080" => (1920, 1080),
        "960x540" => (960, 540),
        _ => {
            return Err(format!(
                "{PROFILE_RESOLUTION_ENV} must be 1920x1080 or 960x540"
            ));
        }
    };
    let present_mode = match present_mode.unwrap_or("fifo") {
        "auto_vsync" => bevy::window::PresentMode::AutoVsync,
        "auto_no_vsync" => bevy::window::PresentMode::AutoNoVsync,
        "fifo" => bevy::window::PresentMode::Fifo,
        "mailbox" => bevy::window::PresentMode::Mailbox,
        _ => {
            return Err(format!(
                "{PROFILE_PRESENT_MODE_ENV} must be auto_vsync, auto_no_vsync, fifo, or mailbox"
            ));
        }
    };
    let ablation = match ablation.unwrap_or("none") {
        "none" => RenderAblation::None,
        "signage_cameras" => RenderAblation::SignageCameras,
        "msaa" => RenderAblation::Msaa,
        "shadows" => RenderAblation::Shadows,
        _ => {
            return Err(format!(
                "{PROFILE_ABLATION_ENV} must be none, signage_cameras, msaa, or shadows"
            ));
        }
    };
    Ok(Some(LiveProfileConfig {
        width,
        height,
        present_mode,
        ablation,
    }))
}

#[cfg(test)]
fn parse_live_profile_config(
    enabled: Option<&str>,
    resolution: Option<&str>,
    present_mode: Option<&str>,
) -> Result<Option<LiveProfileConfig>, String> {
    parse_live_profile_config_with_ablation(enabled, resolution, present_mode, None)
}

fn profile_env_value(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(VarError::NotUnicode(_)) => Err(format!("{name} must be valid Unicode")),
    }
}

fn live_profile_config_from_env() -> Result<Option<LiveProfileConfig>, String> {
    let enabled = profile_env_value(PROFILE_ENV)?;
    let resolution = profile_env_value(PROFILE_RESOLUTION_ENV)?;
    let present_mode = profile_env_value(PROFILE_PRESENT_MODE_ENV)?;
    let ablation = profile_env_value(PROFILE_ABLATION_ENV)?;
    let main_budget = profile_env_value("SAI_LAB_LIVE_PROFILE_MAIN_BUDGET_MS")?;
    if main_budget.is_some()
        && (enabled.as_deref() != Some("1") || !cfg!(feature = "live_render_profile"))
    {
        return Err(
            "main budget probe requires SAI_LAB_LIVE_PROFILE=1 and live_render_profile".into(),
        );
    }
    parse_live_profile_config_with_ablation(
        enabled.as_deref(),
        resolution.as_deref(),
        present_mode.as_deref(),
        ablation.as_deref(),
    )
}

fn disable_signage_cameras(mut cameras: Query<&mut Camera, With<Camera2d>>) {
    for mut camera in &mut cameras {
        camera.is_active = false;
    }
}

fn disable_msaa(mut cameras: Query<&mut Msaa, With<Camera3d>>) {
    for mut msaa in &mut cameras {
        *msaa = Msaa::Off;
    }
}

fn disable_shadows(mut lights: Query<&mut DirectionalLight>) {
    for mut light in &mut lights {
        light.shadow_maps_enabled = false;
    }
}

#[derive(Default)]
struct TimingSummary {
    count: u64,
    sum_ns: u64,
    max_ns: u64,
}

impl TimingSummary {
    fn record(&mut self, duration: Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.count = self.count.saturating_add(1);
        self.sum_ns = self.sum_ns.saturating_add(nanos);
        self.max_ns = self.max_ns.max(nanos);
    }

    fn report(&self) -> Value {
        json!({"count": self.count, "sum_ns": self.sum_ns, "max_ns": self.max_ns})
    }
}

struct LiveProfile {
    config: LiveProfileConfig,
    adapter: Option<GpuAdapterReceipt>,
    frame_start_to_start: TimingSummary,
    advance_frame_wall: TimingSummary,
    worker_deadline_lag: TimingSummary,
    deadline_missed_ticks: u64,
    rapier_counters_per_tick: Vec<RapierCounterSample>,
    frame_samples: Vec<Value>,
    worker_tick_samples: Vec<Value>,
    #[cfg(feature = "live_render_profile")]
    render_cpu_profile: Option<dev_tools_minigame::render_profile::RenderCpuProfile>,
}

#[derive(Clone, serde::Serialize)]
struct GpuAdapterReceipt {
    name: String,
    backend: String,
    timestamp_query_supported: bool,
    pipeline_statistics_query_supported: bool,
}

impl LiveProfile {
    fn new(config: LiveProfileConfig) -> Self {
        Self {
            config,
            adapter: None,
            frame_start_to_start: TimingSummary::default(),
            advance_frame_wall: TimingSummary::default(),
            worker_deadline_lag: TimingSummary::default(),
            deadline_missed_ticks: 0,
            rapier_counters_per_tick: Vec::new(),
            frame_samples: Vec::new(),
            worker_tick_samples: Vec::new(),
            #[cfg(feature = "live_render_profile")]
            render_cpu_profile: None,
        }
    }

    fn report(&self) -> Value {
        let mut report = self.config.requested_config();
        report["render_adapter"] = json!(self.adapter);
        report["frame_start_to_start"] = self.frame_start_to_start.report();
        report["advance_frame_cpu"] = self.advance_frame_wall.report();
        report["worker_deadline_lag"] = self.worker_deadline_lag.report();
        report["deadline_missed_ticks"] = json!(self.deadline_missed_ticks);
        report["rapier_counters_per_tick"] = json!(self.rapier_counters_per_tick);
        report["rapier_profiler_compiled"] = json!(cfg!(feature = "live_physics_profile"));
        report["simulation_thread"] = json!("dedicated_60hz");
        report["frame_samples"] = json!(self.frame_samples);
        report["worker_tick_samples"] = json!(self.worker_tick_samples);
        report["worker_advance_timing_semantics"] = json!(
            "wall_time_including_descheduling; legacy_fields_advance_frame_cpu_and_advance_cpu_ns"
        );
        #[cfg(feature = "live_render_profile")]
        if let Some(profile) = &self.render_cpu_profile {
            report["render_thread_cpu"] = profile.report();
        }
        report["gpu_sample_semantics"] = json!(
            "latest_completed_render_diagnostic_at_main_frame_start; pass_times_are_not_whole_frame_gpu_time"
        );
        report
    }
}

#[derive(Default)]
struct GpuReadiness {
    ready: bool,
    error: Option<String>,
    render_schedule_cleanup_count: u64,
    collect_adapter: bool,
    adapter: Option<GpuAdapterReceipt>,
}

#[derive(Resource, Clone)]
struct SharedGpuReadiness(Arc<Mutex<GpuReadiness>>);

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
struct FinalPngReceipt {
    path: String,
    sha256: String,
    bytes: u64,
    width: u32,
    height: u32,
    global_step: u64,
}

#[derive(Default)]
struct ScreenshotState {
    outcome: Option<Result<FinalPngReceipt, String>>,
}

#[derive(Resource, Clone)]
struct SharedScreenshotState(Arc<Mutex<ScreenshotState>>);

#[derive(Resource)]
struct RequiredShaders {
    enamel: Handle<Shader>,
    ink: Handle<Shader>,
}

fn check_gpu_pipelines(
    cache: Res<PipelineCache>,
    shaders: Res<RequiredShaders>,
    readiness: Res<SharedGpuReadiness>,
    device: Res<RenderDevice>,
    adapter: Res<RenderAdapterInfo>,
) {
    let mut status = readiness.0.lock().unwrap();
    if status.collect_adapter && status.adapter.is_none() {
        status.adapter = Some(GpuAdapterReceipt {
            name: adapter.name.clone(),
            backend: format!("{:?}", adapter.backend),
            timestamp_query_supported: device.features().contains(WgpuFeatures::TIMESTAMP_QUERY),
            pipeline_statistics_query_supported: device
                .features()
                .contains(WgpuFeatures::PIPELINE_STATISTICS_QUERY),
        });
    }
    let mut enamel_ready = false;
    let mut ink_ready = false;
    let mut pending = 0;
    for pipeline in cache.pipelines() {
        match &pipeline.state {
            CachedPipelineState::Ok(_) => {
                if let bevy::material::descriptor::PipelineDescriptor::RenderPipelineDescriptor(
                    descriptor,
                ) = &pipeline.descriptor
                {
                    if let Some(fragment) = &descriptor.fragment {
                        enamel_ready |= fragment.shader == shaders.enamel;
                        ink_ready |= fragment.shader == shaders.ink;
                    }
                }
            }
            CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            ) => pending += 1,
            CachedPipelineState::Err(error) => {
                status.error = Some(format!("live preview shader/pipeline failed: {error:?}"));
            }
            CachedPipelineState::Queued | CachedPipelineState::Creating(_) => pending += 1,
        }
    }
    status.ready = enamel_ready && ink_ready && pending == 0 && status.error.is_none();
    status.render_schedule_cleanup_count += 1;
    if status.error.is_some() && pending == 0 {
        if let Err(error) = device.poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(2)),
        }) {
            status.error = Some(format!("failed to drain GPU after shader error: {error}"));
        }
    }
}

struct LiveRun {
    session: Arc<Mutex<DiagnosticSession>>,
    minimum_ticks: u64,
    worker: Option<PhysicsWorker>,
    worker_finished: bool,
    worker_wall_ns: Option<u64>,
    worker_deadline_missed_ticks: u64,
    worker_max_deadline_lag_ns: u64,
    baked_label_camera_count: usize,
    output: PathBuf,
    profile: Option<LiveProfile>,
    started: Instant,
    last_running_frame: Option<Instant>,
    display_frames: u64,
    completed_publishes: u64,
    last_displayed_step: u64,
    awaiting_final_visual: bool,
    final_pose_render_schedule_cleanup_count: Option<u64>,
    observed_render_schedule_cleanup_count: u64,
    final_png_path: Option<PathBuf>,
    screenshot_started: Option<Instant>,
    final_png_receipt: Option<FinalPngReceipt>,
    report_written: bool,
}

struct PhysicsWorker {
    updates: Receiver<WorkerUpdate>,
    cancel: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

enum WorkerUpdate {
    Tick {
        pose: Arc<RobotPoseFrame>,
        advance_wall: Duration,
        deadline_lag: Duration,
        rapier_counters: Option<RapierCounterSample>,
    },
    Finished(Duration),
    Failed(String),
}

/// One exact 60 Hz physics boundary per call. Rendering may skip poses, but
/// the worker never skips, repeats, or coalesces an inference/physics tick.
fn spawn_physics_worker(
    session: Arc<Mutex<DiagnosticSession>>,
    ticks: u64,
    profile_rapier: bool,
) -> PhysicsWorker {
    let (sender, updates) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    let handle = thread::Builder::new()
        .name("sim2sim_physics_60hz".into())
        .spawn(move || {
            let epoch = Instant::now();
            let mut last_nanos = 0_u64;
            let mut session = session.lock().unwrap();
            for tick in 1..=ticks {
                if worker_cancel.load(Ordering::Acquire) {
                    return;
                }
                let target_nanos = (tick * 1_000_000_000).div_ceil(60);
                let scheduled_start = epoch + Duration::from_nanos(last_nanos);
                if let Some(remaining) = scheduled_start.checked_duration_since(Instant::now()) {
                    thread::sleep(remaining);
                }
                if worker_cancel.load(Ordering::Acquire) {
                    return;
                }
                let started = Instant::now();
                let _tick_span = tracing::info_span!("sim2sim_worker_tick", tick).entered();
                let outcome =
                    session.advance_frame(Duration::from_nanos(target_nanos - last_nanos));
                let advance_wall = started.elapsed();
                last_nanos = target_nanos;
                let deadline = epoch + Duration::from_nanos(target_nanos);
                let outcome = match outcome {
                    Ok(outcome) if outcome.completed_steps == 1 && outcome.pending_ticks == 0 => {
                        outcome
                    }
                    Ok(outcome) => {
                        let _ = sender.send(WorkerUpdate::Failed(format!(
                            "worker tick {tick} completed {} steps with {} pending",
                            outcome.completed_steps, outcome.pending_ticks
                        )));
                        return;
                    }
                    Err(error) => {
                        let _ = sender
                            .send(WorkerUpdate::Failed(format!("worker tick {tick}: {error}")));
                        return;
                    }
                };
                let poses = session.take_published_frames();
                let pose = match display_pose_for_frame(
                    &poses,
                    outcome,
                    tick - 1,
                    session.global_step(),
                ) {
                    Ok(Some(pose)) => pose,
                    Ok(None) => {
                        let _ = sender.send(WorkerUpdate::Failed(format!(
                            "worker tick {tick} published no robot pose"
                        )));
                        return;
                    }
                    Err(error) => {
                        let _ = sender
                            .send(WorkerUpdate::Failed(format!("worker tick {tick}: {error}")));
                        return;
                    }
                };
                let deadline_lag = Instant::now().saturating_duration_since(deadline);
                let rapier_counters = profile_rapier.then(|| session.rapier_counter_sample());
                if sender
                    .send(WorkerUpdate::Tick {
                        pose,
                        advance_wall,
                        deadline_lag,
                        rapier_counters,
                    })
                    .is_err()
                {
                    return;
                }
            }
            let _ = sender.send(WorkerUpdate::Finished(epoch.elapsed()));
        })
        .expect("could not create physics worker");
    PhysicsWorker {
        updates,
        cancel,
        handle,
    }
}

impl LiveRun {
    fn join_worker(&mut self) -> Result<(), String> {
        if let Some(worker) = self.worker.take() {
            worker.cancel.store(true, Ordering::Release);
            worker
                .handle
                .join()
                .map_err(|_| "physics worker panicked".to_string())?;
        }
        Ok(())
    }

    fn report(
        &mut self,
        passed: bool,
        error: Option<&str>,
        visual: &RobotVisualPhase,
    ) -> Result<(), String> {
        let worker_join = self.join_worker();
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let actual_steps = session.global_step();
        let minimum_ticks = self.minimum_ticks;
        let pending_ticks = session.pending_ticks();
        let counts_passed = session.finish_expected_steps(actual_steps);
        let deadline_budget_ns = (minimum_ticks * 1_000_000_000).div_ceil(60);
        let worker_60hz_deadline_met = worker_join.is_ok()
            && self.worker_finished
            && actual_steps == minimum_ticks
            && self.worker_deadline_missed_ticks == 0
            && self
                .worker_wall_ns
                .is_some_and(|wall_ns| wall_ns <= deadline_budget_ns);
        let profile_report = self.profile.as_ref().map(LiveProfile::report);
        let report = session.report_mut();
        report["schema"] = json!(LIVE_SCHEMA);
        report["scope"] =
            json!("dev_only_live_bevy_single_rapier_world_native_ort_p_only_diagnostic");
        report["minimum_ticks"] = json!(minimum_ticks);
        report["actual_completed_ticks"] = json!(actual_steps);
        report["overshoot_ticks"] = json!(actual_steps.saturating_sub(minimum_ticks as u64));
        report["display_frames"] = json!(self.display_frames);
        report["completed_publishes"] = json!(self.completed_publishes);
        report["last_displayed_step"] = json!(self.last_displayed_step);
        report["pending_ticks"] = json!(pending_ticks);
        report["runtime_schedule"] = json!("dedicated_rational_60hz_worker");
        report["worker_finished"] = json!(self.worker_finished);
        report["worker_wall_ns"] = json!(self.worker_wall_ns);
        report["worker_deadline_missed_ticks"] = json!(self.worker_deadline_missed_ticks);
        report["worker_max_deadline_lag_ns"] = json!(self.worker_max_deadline_lag_ns);
        report["station_label_cameras_baked"] = json!(self.baked_label_camera_count);
        report["worker_60hz_deadline_met"] = json!(worker_60hz_deadline_met);
        report["robot_visual_status"] = json!(visual);
        report["props_physical"] = json!(false);
        report["live_visual_qualified"] = json!(false);
        report["performance_qualified"] = json!(false);
        if let Some(profile_report) = profile_report {
            report["live_profile"] = profile_report;
        }
        report["render_schedule_cleanup_count"] =
            json!(self.observed_render_schedule_cleanup_count);
        report["render_schedule_cleanups_after_final_publish"] = json!(
            self.observed_render_schedule_cleanup_count
                .saturating_sub(self.final_pose_render_schedule_cleanup_count.unwrap_or(0))
        );
        report["final_png_required"] = json!(self.final_png_path.is_some());
        report["final_png"] = json!(self.final_png_receipt);
        let accepted = passed
            && worker_join.is_ok()
            && self.worker_finished
            && self.worker_wall_ns.is_some()
            && counts_passed
            && actual_steps == minimum_ticks
            && self.completed_publishes == actual_steps
            && self.last_displayed_step == actual_steps
            && (!cfg!(feature = "live_physics_profile")
                || self.profile.as_ref().is_none_or(|profile| {
                    profile.rapier_counters_per_tick.len() as u64 == actual_steps
                }))
            && (self.final_png_path.is_none() || self.final_png_receipt.is_some());
        report["passed"] = json!(accepted);
        report["error"] = match error {
            Some(reason) => json!(reason),
            None if let Err(reason) = worker_join => json!(reason),
            None if !accepted => {
                json!("live completion counts did not match the minimum tick bound")
            }
            None => Value::Null,
        };
        fs::write(
            &self.output,
            serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("write live report {}: {e}", self.output.display()))?;
        self.report_written = true;
        Ok(())
    }
}

fn validate_final_png(
    requested_path: &Path,
    receipt: &FinalPngReceipt,
    expected_step: u64,
) -> Result<(), String> {
    if receipt.global_step != expected_step || receipt.width == 0 || receipt.height == 0 {
        return Err("final screenshot does not describe the frozen visual frame".into());
    }
    let canonical = requested_path
        .canonicalize()
        .map_err(|error| format!("final screenshot path: {error}"))?;
    if receipt.path != canonical.to_string_lossy() {
        return Err("final screenshot receipt names another file".into());
    }
    let bytes = fs::read(&canonical).map_err(|error| format!("read final screenshot: {error}"))?;
    if bytes.is_empty()
        || bytes.len() as u64 != receipt.bytes
        || format!("{:x}", Sha256::digest(&bytes)) != receipt.sha256
    {
        return Err("final screenshot bytes do not match their SHA256 receipt".into());
    }
    Ok(())
}

impl Drop for LiveRun {
    fn drop(&mut self) {
        if !self.report_written {
            let phase = RobotVisualPhase::Failed {
                reason: "live window closed before its final visual frame was verified".into(),
            };
            let _ = self.report(false, Some("live window closed before completion"), &phase);
        }
    }
}

fn fail(
    run: &mut LiveRun,
    input: &mut RobotVisualInput,
    exit: &mut MessageWriter<AppExit>,
    reason: String,
) {
    input.withdraw();
    let phase = RobotVisualPhase::Failed {
        reason: reason.clone(),
    };
    if let Err(write_error) = run.report(false, Some(&reason), &phase) {
        eprintln!("{write_error}");
    }
    eprintln!("Live diagnostic failed: {reason}");
    exit.write(AppExit::error());
}

fn display_pose_for_frame(
    poses: &[Arc<RobotPoseFrame>],
    outcome: FrameOutcome,
    previous_step: u64,
    current_step: u64,
) -> Result<Option<Arc<RobotPoseFrame>>, String> {
    if poses.len() as u64 != outcome.completed_steps {
        return Err("completed-pose count differs from fixed-step outcome".into());
    }
    if poses.is_empty() {
        if previous_step != current_step {
            return Err("physics advanced without a completed display pose".into());
        }
        return Ok(None);
    }
    if previous_step.checked_add(1) != Some(poses[0].global_step)
        || poses
            .last()
            .is_none_or(|pose| pose.global_step != current_step)
        || poses
            .windows(2)
            .any(|pair| pair[1].global_step != pair[0].global_step + 1)
    {
        return Err("published pose sequence is not contiguous".into());
    }
    Ok(poses.last().cloned())
}

fn advance_live(
    mut run: NonSendMut<LiveRun>,
    mut input: ResMut<RobotVisualInput>,
    mut camera: ResMut<StationCameraControl>,
    visual: Res<RobotVisualStatus>,
    gpu: Res<SharedGpuReadiness>,
    diagnostics: Res<DiagnosticsStore>,
    labels: Res<StationLabelBakeStatus>,
    mut exit: MessageWriter<AppExit>,
) {
    if run.awaiting_final_visual || run.report_written {
        return;
    }
    let gpu_state = gpu.0.lock().unwrap();
    if let Some(error) = &gpu_state.error {
        fail(&mut run, &mut input, &mut exit, error.clone());
        return;
    }
    if let RobotVisualPhase::Failed { reason } = visual.phase() {
        fail(&mut run, &mut input, &mut exit, reason.clone());
        return;
    }
    // Renderer startup and GPU shader compilation do not accrue simulation debt.
    if !gpu_state.ready
        || !labels.is_ready()
        || !matches!(visual.phase(), RobotVisualPhase::Ready { .. })
    {
        run.last_running_frame = None;
        if run.started.elapsed() > Duration::from_secs(120) {
            fail(
                &mut run,
                &mut input,
                &mut exit,
                "initial robot/GPU frame timed out".into(),
            );
        }
        return;
    }
    if let Some(profile) = run.profile.as_mut() {
        if profile.adapter.is_none() {
            profile.adapter = gpu_state.adapter.clone();
        }
    }
    drop(gpu_state);
    run.baked_label_camera_count = labels.baked_camera_count().unwrap_or(0);
    #[cfg(feature = "live_render_profile")]
    if let Some(profile) = run
        .profile
        .as_ref()
        .and_then(|p| p.render_cpu_profile.as_ref())
    {
        profile.mark_frame(run.display_frames);
    }
    let now = Instant::now();
    let _frame_span = tracing::info_span!(
        "sim2sim_live_frame",
        display_frame = run.display_frames,
        last_displayed_step = run.last_displayed_step
    )
    .entered();
    let previous_start = run.last_running_frame.replace(now);
    let elapsed = previous_start.map_or(Duration::ZERO, |last| now.duration_since(last));
    if let (Some(_), Some(profile)) = (previous_start, run.profile.as_mut()) {
        profile.frame_start_to_start.record(elapsed);
    }
    let display_frame = run.display_frames;
    let last_displayed_step_at_start = run.last_displayed_step;
    let start_since_run_ns =
        u64::try_from(now.duration_since(run.started).as_nanos()).unwrap_or(u64::MAX);
    if let Some(profile) = run.profile.as_mut() {
        profile.frame_samples.push(json!({
            "display_frame": display_frame,
            "start_since_run_ns": start_since_run_ns,
            "previous_interval_ns": previous_start.map(|_| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)),
            "last_displayed_step_at_start": last_displayed_step_at_start,
            "render_diagnostics": diagnostics.iter().filter(|d| d.path().as_str().starts_with("render/")).map(|d| json!({"path":d.path().as_str(),"latest":d.value()})).collect::<Vec<_>>(),
        }));
    }
    if run.worker.is_none() && !run.worker_finished {
        run.worker = Some(spawn_physics_worker(
            run.session.clone(),
            run.minimum_ticks,
            run.profile.is_some() && cfg!(feature = "live_physics_profile"),
        ));
    }
    let mut latest_pose = None;
    let mut failure = None;
    loop {
        let update = run.worker.as_ref().map(|worker| worker.updates.try_recv());
        match update {
            Some(Ok(WorkerUpdate::Tick {
                pose,
                advance_wall,
                deadline_lag,
                rapier_counters,
            })) => {
                if pose.global_step != run.completed_publishes + 1 {
                    failure = Some("physics worker pose sequence is not contiguous".into());
                    break;
                }
                if let Some(profile) = run.profile.as_mut() {
                    profile.worker_tick_samples.push(json!({
                        "global_step": pose.global_step,
                        "advance_cpu_ns": u64::try_from(advance_wall.as_nanos()).unwrap_or(u64::MAX),
                        "deadline_lag_ns": u64::try_from(deadline_lag.as_nanos()).unwrap_or(u64::MAX),
                    }));
                    profile.advance_frame_wall.record(advance_wall);
                    profile.worker_deadline_lag.record(deadline_lag);
                    if !deadline_lag.is_zero() {
                        profile.deadline_missed_ticks += 1;
                    }
                    if let Some(counters) = rapier_counters {
                        profile.rapier_counters_per_tick.push(counters);
                    }
                }
                run.completed_publishes += 1;
                let lag_ns = u64::try_from(deadline_lag.as_nanos()).unwrap_or(u64::MAX);
                if lag_ns > 0 {
                    run.worker_deadline_missed_ticks += 1;
                    run.worker_max_deadline_lag_ns = run.worker_max_deadline_lag_ns.max(lag_ns);
                }
                latest_pose = Some(pose);
            }
            Some(Ok(WorkerUpdate::Finished(wall))) => {
                run.worker_finished = true;
                run.worker_wall_ns = Some(u64::try_from(wall.as_nanos()).unwrap_or(u64::MAX));
                if run.completed_publishes != run.minimum_ticks {
                    failure = Some("physics worker finished before all poses were received".into());
                }
                break;
            }
            Some(Ok(WorkerUpdate::Failed(error))) => {
                failure = Some(error);
                break;
            }
            Some(Err(TryRecvError::Disconnected)) => {
                failure = Some("physics worker disconnected before completion".into());
                break;
            }
            Some(Err(TryRecvError::Empty)) | None => break,
        }
    }
    if let Some(error) = failure {
        fail(&mut run, &mut input, &mut exit, error);
        return;
    }
    if let Some(last) = latest_pose {
        let Some(root) = last.poses.iter().find(|pose| pose.source_body_id == 1) else {
            fail(
                &mut run,
                &mut input,
                &mut exit,
                "published robot root pose is missing".into(),
            );
            return;
        };
        input.publish(last.clone());
        camera.target = Vec3::from_array(root.translation);
        run.last_displayed_step = last.global_step;
    }
    run.display_frames += 1;
    if run.worker_finished {
        if let Err(error) = run.join_worker() {
            fail(&mut run, &mut input, &mut exit, error);
            return;
        }
        if run.final_png_path.is_some() {
            // Preserve the benchmark camera while the completed pose is captured.
            camera.input_enabled = false;
        }
        #[cfg(feature = "live_render_profile")]
        if let Some(profile) = run
            .profile
            .as_ref()
            .and_then(|p| p.render_cpu_profile.as_ref())
        {
            profile.stop();
        }
        run.awaiting_final_visual = true;
        run.final_pose_render_schedule_cleanup_count =
            Some(gpu.0.lock().unwrap().render_schedule_cleanup_count);
    }
}

fn verify_final_visual(
    mut commands: Commands,
    mut run: NonSendMut<LiveRun>,
    visual: Res<RobotVisualStatus>,
    gpu: Res<SharedGpuReadiness>,
    screenshot: Res<SharedScreenshotState>,
    mut exit: MessageWriter<AppExit>,
    mut input: ResMut<RobotVisualInput>,
    diagnostics: Res<DiagnosticsStore>,
) {
    if !run.awaiting_final_visual || run.report_written {
        return;
    }
    if run
        .screenshot_started
        .is_some_and(|started| started.elapsed() > Duration::from_secs(90))
    {
        fail(
            &mut run,
            &mut input,
            &mut exit,
            "final screenshot readback timed out".into(),
        );
        return;
    }
    let gpu = gpu.0.lock().unwrap();
    if let Some(error) = &gpu.error {
        fail(&mut run, &mut input, &mut exit, error.clone());
        return;
    }
    if !gpu.ready {
        return;
    }
    run.observed_render_schedule_cleanup_count = gpu.render_schedule_cleanup_count;
    let required_cleanups = if run.final_png_path.is_some() { 2 } else { 1 };
    if run
        .final_pose_render_schedule_cleanup_count
        .is_none_or(|before| {
            gpu.render_schedule_cleanup_count < before.saturating_add(required_cleanups)
        })
    {
        if run.started.elapsed() > Duration::from_secs(120) {
            drop(gpu);
            fail(
                &mut run,
                &mut input,
                &mut exit,
                "final pose was never followed by a render-schedule Cleanup run".into(),
            );
        }
        return;
    }
    drop(gpu);
    let expected = run.session.lock().unwrap().global_step();
    match visual.phase() {
        RobotVisualPhase::Ready { global_step, .. } if *global_step == expected => {
            if let Some(path) = run.final_png_path.clone() {
                if run.screenshot_started.is_none() {
                    commands.spawn(Screenshot::primary_window()).observe(
                        move |captured: On<ScreenshotCaptured>,
                              state: Res<SharedScreenshotState>| {
                            let width = captured.image.texture_descriptor.size.width;
                            let height = captured.image.texture_descriptor.size.height;
                            let result = (|| {
                                if path.exists() {
                                    return Err(format!(
                                        "final screenshot output already exists: {}",
                                        path.display()
                                    ));
                                }
                                captured
                                    .image
                                    .clone()
                                    .try_into_dynamic()
                                    .map_err(|error| {
                                        format!("final screenshot conversion: {error}")
                                    })?
                                    .to_rgb8()
                                    .save(&path)
                                    .map_err(|error| {
                                        format!("save final screenshot {}: {error}", path.display())
                                    })?;
                                let bytes = fs::read(&path)
                                    .map_err(|error| format!("read final screenshot: {error}"))?;
                                let canonical = path
                                    .canonicalize()
                                    .map_err(|error| format!("final screenshot path: {error}"))?;
                                Ok(FinalPngReceipt {
                                    path: canonical.to_string_lossy().into_owned(),
                                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                                    bytes: bytes.len() as u64,
                                    width,
                                    height,
                                    global_step: expected,
                                })
                            })();
                            state.0.lock().unwrap().outcome = Some(result);
                        },
                    );
                    run.screenshot_started = Some(Instant::now());
                    return;
                }
                let captured = screenshot.0.lock().unwrap().outcome.take();
                match captured {
                    None => return,
                    Some(Err(error)) => {
                        fail(&mut run, &mut input, &mut exit, error);
                        return;
                    }
                    Some(Ok(receipt)) => {
                        if let Err(error) = validate_final_png(&path, &receipt, expected) {
                            fail(&mut run, &mut input, &mut exit, error);
                            return;
                        }
                        run.final_png_receipt = Some(receipt);
                    }
                }
            }
            let phase = visual.phase().clone();
            if run.profile.is_some() {
                let mut rows: Vec<_> = diagnostics
                    .iter()
                    .map(|diagnostic| {
                        json!({
                            "path": diagnostic.path().as_str(),
                            "latest": diagnostic.value(),
                            "average": diagnostic.average(),
                        })
                    })
                    .collect();
                rows.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
                run.session.lock().unwrap().report_mut()["bevy_builtin_diagnostics"] = json!(rows);
            }
            match run.report(true, None, &phase) {
                Ok(()) if run.session.lock().unwrap().report()["passed"] == true => {
                    println!("STATUS: success; live report={}", run.output.display());
                    exit.write(AppExit::Success);
                }
                Ok(()) => {
                    eprintln!("Live diagnostic counts did not match the completed frame");
                    exit.write(AppExit::error());
                }
                Err(error) => {
                    eprintln!("{error}");
                    exit.write(AppExit::error());
                }
            }
        }
        RobotVisualPhase::Failed { reason } => {
            fail(&mut run, &mut input, &mut exit, reason.clone())
        }
        RobotVisualPhase::Ready { .. }
        | RobotVisualPhase::AwaitingModel
        | RobotVisualPhase::AwaitingFrame => {
            fail(
                &mut run,
                &mut input,
                &mut exit,
                "final visual frame did not match the completed physics tick".into(),
            );
        }
    }
}

fn required_assets(root: &Path) -> Result<PathBuf, String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    for shader in ["station_enamel.wgsl", "station_ink.wgsl"] {
        fs::read_to_string(root.join("game/shaders").join(shader)).map_err(|e| e.to_string())?;
    }
    fs::read(root.join("third_party/fonts/noto_sans_cjk_regular.otf"))
        .map_err(|e| e.to_string())?;
    Ok(root)
}

fn validate_live_exit(exit: AppExit, report: &Value) -> Result<(), String> {
    if exit.is_error() {
        return Err("live diagnostic window exited with an error; see report".into());
    }
    if report["passed"] != true {
        return Err(
            "live diagnostic window closed before successful completion; see report".into(),
        );
    }
    if report["final_png_required"] == true {
        let receipt: FinalPngReceipt = serde_json::from_value(report["final_png"].clone())
            .map_err(|error| format!("final screenshot receipt is missing: {error}"))?;
        let expected_step = report["actual_completed_ticks"]
            .as_u64()
            .ok_or("final screenshot receipt has no completed tick")?;
        validate_final_png(Path::new(&receipt.path), &receipt, expected_step)?;
    }
    Ok(())
}

fn run(
    args: &[String],
    output: PathBuf,
    final_png_path: Option<PathBuf>,
    profile_config: Option<LiveProfileConfig>,
) -> Result<(), String> {
    let mut report = initial_report();
    report["schema"] = json!(LIVE_SCHEMA);
    report["scope"] = json!("dev_only_live_bevy_single_rapier_world_native_ort_p_only_diagnostic");
    report["final_png_required"] = json!(final_png_path.is_some());
    report["final_png"] = Value::Null;
    if let Some(config) = profile_config {
        report["live_profile"] = config.requested_config();
    }
    let mut session = match prepare(&args[..9], &mut report) {
        Ok(session) => session,
        Err(error) => {
            report["error"] = json!(error);
            fs::write(
                &output,
                serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            return Err(error);
        }
    };
    if profile_config.is_some() && cfg!(feature = "live_physics_profile") {
        session.enable_rapier_counters();
    }
    let definition = session.definition();
    let appearance_path = Path::new(&args[9]);
    let appearance = Arc::new(
        match appearance_path.extension().and_then(|value| value.to_str()) {
            Some("json") => {
                VerifiedRobotAppearance::load_json(appearance_path, &args[10], &definition)
            }
            Some("ron") => {
                VerifiedRobotAppearance::load_ron(appearance_path, &args[10], &definition)
            }
            _ => return Err("appearance must be .json or .ron".into()),
        }
        .map_err(|e| e.to_string())?,
    );
    let model = RobotVisualModel::new(definition.clone(), appearance).map_err(|e| e.to_string())?;
    let input =
        RobotVisualInput::new(&definition, session.initial_pose()).map_err(|e| e.to_string())?;
    let minimum_ticks = session.ticks() as u64;
    let scene: StationScene = session.scene();
    let asset_root = required_assets(Path::new(&args[4]))?;
    let initial_pose = session.initial_pose();
    let first_root = initial_pose
        .poses
        .iter()
        .find(|pose| pose.source_body_id == 1)
        .ok_or("initial robot root pose is missing")?;
    let camera = StationCameraControl {
        view: StationView::Overview,
        target: Vec3::from_array(first_root.translation),
        input_enabled: profile_config.is_none(),
        ..default()
    };
    let completion_report = output.clone();
    let live = LiveRun {
        session: Arc::new(Mutex::new(session)),
        minimum_ticks,
        worker: None,
        worker_finished: false,
        worker_wall_ns: None,
        worker_deadline_missed_ticks: 0,
        worker_max_deadline_lag_ns: 0,
        baked_label_camera_count: 0,
        output,
        profile: profile_config.map(LiveProfile::new),
        started: Instant::now(),
        last_running_frame: None,
        display_frames: 0,
        completed_publishes: 0,
        last_displayed_step: 0,
        awaiting_final_visual: false,
        final_pose_render_schedule_cleanup_count: None,
        observed_render_schedule_cleanup_count: 0,
        final_png_path,
        screenshot_started: None,
        final_png_receipt: None,
        report_written: false,
    };
    let readiness = SharedGpuReadiness(Arc::new(Mutex::new(GpuReadiness {
        collect_adapter: profile_config.is_some(),
        ..default()
    })));
    let screenshot = SharedScreenshotState(Arc::new(Mutex::new(ScreenshotState::default())));
    let (width, height, present_mode) = profile_config
        .map_or((1920, 1080, bevy::window::PresentMode::Fifo), |config| {
            (config.width, config.height, config.present_mode)
        });
    #[cfg(feature = "live_render_profile")]
    let profile_epoch = live.started;
    let mut app = App::new();
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .insert_resource(scene)
        .insert_resource(camera)
        .insert_resource(model)
        .insert_resource(input)
        .insert_resource(readiness.clone())
        .insert_resource(screenshot)
        .insert_non_send(live)
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "DEV ONLY — MicroDuck 60 Hz P-only live diagnostic".into(),
                        resolution: (width, height).into(),
                        present_mode,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(StationVisualPlugin)
        .add_plugins(RobotVisualPlugin)
        .add_systems(Update, advance_live)
        .add_systems(
            PostUpdate,
            verify_final_visual.after(RobotVisualSystems::ApplyPoses),
        );
    if profile_config.is_some() {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
            MeshAllocatorDiagnosticPlugin,
        ));
    }
    if let Some(config) = profile_config {
        match config.ablation {
            RenderAblation::None => {}
            RenderAblation::SignageCameras => {
                app.add_systems(PostStartup, disable_signage_cameras);
            }
            RenderAblation::Msaa => {
                app.add_systems(PostStartup, disable_msaa);
            }
            RenderAblation::Shadows => {
                app.add_systems(PostStartup, disable_shadows);
            }
        }
    }
    let shaders = {
        let server = app.world().resource::<AssetServer>();
        RequiredShaders {
            enamel: server.load("game/shaders/station_enamel.wgsl"),
            ink: server.load("game/shaders/station_ink.wgsl"),
        }
    };
    let render_app = app
        .get_sub_app_mut(RenderApp)
        .ok_or("GPU render application unavailable")?;
    render_app.world_mut().insert_resource(readiness);
    render_app.world_mut().insert_resource(shaders);
    render_app.add_systems(Render, check_gpu_pipelines.in_set(RenderSystems::Cleanup));
    #[cfg(feature = "live_render_profile")]
    if profile_config.is_some() {
        let cpu_profile = dev_tools_minigame::render_profile::install(&mut app, profile_epoch)?;
        app.world_mut()
            .non_send_mut::<LiveRun>()
            .profile
            .as_mut()
            .unwrap()
            .render_cpu_profile = Some(cpu_profile);
    }
    let result = app.run();
    let completed_report: Value = serde_json::from_slice(
        &fs::read(&completion_report)
            .map_err(|error| format!("read live completion report: {error}"))?,
    )
    .map_err(|error| format!("parse live completion report: {error}"))?;
    validate_live_exit(result, &completed_report)
}

fn parse_final_png_arg(args: &[String]) -> Result<Option<PathBuf>, String> {
    let usage = "usage: station_robot_live_preview MODEL MODEL_SHA QPOS QPOS_SHA ASSETS ONNX POLICY_CONTRACT|--legacy-original NATIVE_ORT_LIB MIN_TICKS APPEARANCE APPEARANCE_SHA NEW_REPORT.json [--final-png NEW_IMAGE.png]";
    match args.len() {
        12 => Ok(None),
        14 if args[12] == "--final-png" => {
            let path = PathBuf::from(&args[13]);
            if path.extension().and_then(|extension| extension.to_str()) != Some("png") {
                return Err("final screenshot output must have a .png extension".into());
            }
            Ok(Some(path))
        }
        _ => Err(usage.into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let final_png_path = parse_final_png_arg(&args)?;
    let output = PathBuf::from(&args[11]);
    if output.exists() {
        return Err("live report output must be new".into());
    }
    if let Some(path) = &final_png_path {
        if path == &output || path.exists() {
            return Err("final screenshot output must be new and distinct from the report".into());
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
    }
    let profile_config = live_profile_config_from_env();
    let result = match &profile_config {
        Ok(config) => run(&args, output.clone(), final_png_path.clone(), *config),
        Err(error) => Err(error.clone()),
    };
    if let Err(error) = &result {
        if !output.exists() {
            let mut report = initial_report();
            report["schema"] = json!(LIVE_SCHEMA);
            report["scope"] =
                json!("dev_only_live_bevy_single_rapier_world_native_ort_p_only_diagnostic");
            report["final_png_required"] = json!(final_png_path.is_some());
            if let Ok(Some(config)) = profile_config {
                report["live_profile"] = config.requested_config();
            }
            report["error"] = json!(error);
            fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        }
    }
    result.map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "live_physics_profile")]
    #[test]
    fn rapier_native_counters_measure_a_real_step() {
        let mut world = simulation_minigame::SimulationWorld::foundation();
        world.enable_rapier_counters();
        world.step_with_torques(&[]).unwrap();
        let counters = world.rapier_counter_sample();
        assert!(counters.step_ns > 0);
        assert!(counters.broad_phase_ns > 0);
    }

    #[test]
    fn profile_requires_explicit_enable_and_accepts_only_supported_settings() {
        assert!(
            parse_live_profile_config(None, None, None)
                .unwrap()
                .is_none()
        );
        assert!(parse_live_profile_config(None, Some("960x540"), None).is_err());
        assert!(parse_live_profile_config(None, None, Some("auto_no_vsync")).is_err());
        assert!(parse_live_profile_config(Some("0"), None, None).is_err());
        assert!(parse_live_profile_config(Some("true"), None, None).is_err());
        assert!(parse_live_profile_config(Some("1"), Some("960X540"), None).is_err());
        assert!(parse_live_profile_config(Some("1"), Some("960x540 "), None).is_err());
        assert!(parse_live_profile_config(Some("1"), None, Some("immediate")).is_err());

        let default = parse_live_profile_config(Some("1"), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(default.width, 1920);
        assert_eq!(default.height, 1080);
        assert_eq!(default.present_mode, bevy::window::PresentMode::Fifo);
        assert_eq!(default.requested_config()["requested_present_mode"], "fifo");

        let alternate =
            parse_live_profile_config(Some("1"), Some("960x540"), Some("auto_no_vsync"))
                .unwrap()
                .unwrap();
        assert_eq!(alternate.width, 960);
        assert_eq!(alternate.height, 540);
        assert_eq!(
            alternate.present_mode,
            bevy::window::PresentMode::AutoNoVsync
        );
        assert_eq!(
            alternate.requested_config()["requested_resolution"],
            json!([960, 540])
        );
        assert_eq!(
            alternate.requested_config()["requested_present_mode"],
            "auto_no_vsync"
        );

        for (requested, expected) in [
            ("fifo", bevy::window::PresentMode::Fifo),
            ("mailbox", bevy::window::PresentMode::Mailbox),
        ] {
            let config = parse_live_profile_config(Some("1"), None, Some(requested))
                .unwrap()
                .unwrap();
            assert_eq!(config.present_mode, expected);
            assert_eq!(
                config.requested_config()["requested_present_mode"],
                requested
            );
        }
        assert!(parse_live_profile_config(Some("1"), None, Some("Fifo")).is_err());
        assert!(parse_live_profile_config(Some("1"), None, Some("mailbox ")).is_err());
        assert_eq!(
            parse_live_profile_config(Some("1"), None, Some("immediate")).unwrap_err(),
            format!(
                "{PROFILE_PRESENT_MODE_ENV} must be auto_vsync, auto_no_vsync, fifo, or mailbox"
            )
        );
        assert!(parse_live_profile_config_with_ablation(None, None, None, Some("msaa")).is_err());
        for requested in ["none", "signage_cameras", "msaa", "shadows"] {
            let config =
                parse_live_profile_config_with_ablation(Some("1"), None, None, Some(requested))
                    .unwrap()
                    .unwrap();
            assert_eq!(config.requested_config()["render_ablation"], requested);
        }
        assert!(
            parse_live_profile_config_with_ablation(Some("1"), None, None, Some("all")).is_err()
        );
    }

    fn pose(step: u64) -> Arc<RobotPoseFrame> {
        Arc::new(RobotPoseFrame {
            model_file_sha256: "boundary-only synthetic frame".into(),
            episode_id: 0,
            global_step: step,
            episode_step: step,
            poses: Vec::new(),
        })
    }

    #[test]
    fn catch_up_presents_only_last_completed_tick() {
        let frames = vec![pose(8), pose(9), pose(10)];
        let selected = display_pose_for_frame(
            &frames,
            FrameOutcome {
                completed_steps: 3,
                pending_ticks: 4,
            },
            7,
            10,
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected.global_step, 10);
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn idle_frame_keeps_the_last_pose_without_inventing_a_tick() {
        assert!(
            display_pose_for_frame(
                &[],
                FrameOutcome {
                    completed_steps: 0,
                    pending_ticks: 0,
                },
                9,
                9,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            display_pose_for_frame(
                &[],
                FrameOutcome {
                    completed_steps: 0,
                    pending_ticks: 0,
                },
                9,
                10,
            )
            .is_err()
        );
    }

    #[test]
    fn missing_or_miscounted_tick_is_rejected_before_visual_publish() {
        let missing_middle = vec![pose(8), pose(10)];
        assert!(
            display_pose_for_frame(
                &missing_middle,
                FrameOutcome {
                    completed_steps: 2,
                    pending_ticks: 0,
                },
                7,
                10,
            )
            .is_err()
        );
        assert!(
            display_pose_for_frame(
                &[pose(8)],
                FrameOutcome {
                    completed_steps: 2,
                    pending_ticks: 0,
                },
                7,
                8,
            )
            .is_err()
        );
    }

    #[test]
    fn window_close_without_a_passed_receipt_is_an_error() {
        assert!(validate_live_exit(AppExit::Success, &json!({"passed": false})).is_err());
        assert!(validate_live_exit(AppExit::Success, &json!({})).is_err());
        assert!(validate_live_exit(AppExit::Success, &json!({"passed": true})).is_ok());
        assert!(
            validate_live_exit(
                AppExit::Success,
                &json!({"passed": true, "final_png_required": true})
            )
            .is_err()
        );
        assert!(validate_live_exit(AppExit::error(), &json!({"passed": true})).is_err());
    }

    #[test]
    fn final_screenshot_is_an_explicit_optional_png_argument() {
        let mut arguments = vec!["input".to_owned(); 12];
        assert!(parse_final_png_arg(&arguments).unwrap().is_none());
        arguments.push("--final-png".into());
        arguments.push("capture.png".into());
        assert_eq!(
            parse_final_png_arg(&arguments).unwrap(),
            Some(PathBuf::from("capture.png"))
        );
        arguments[13] = "capture.jpg".into();
        assert!(parse_final_png_arg(&arguments).is_err());
        arguments[12] = "--not-final-png".into();
        assert!(parse_final_png_arg(&arguments).is_err());
    }
}
