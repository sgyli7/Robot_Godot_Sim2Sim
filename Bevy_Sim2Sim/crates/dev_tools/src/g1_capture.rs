//! Bounded native G1 camera evidence in the runner's actual simple floor world.
//!
//! This development scene does not load scientific-station geometry. Its sole
//! environment mesh is the same 40 x 0.5 x 40 metre floor as G1Runner. Real RGB
//! and measured self state can therefore be tested before station task physics
//! is integrated. Neither initialization nor these captures qualify standing,
//! manipulation, locomotion or a visual decision loop.

use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    camera::{Exposure, Hdr},
    core_pipeline::tonemapping::Tonemapping,
    light::cluster::GlobalClusterSettings,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    winit::WinitPlugin,
};
use rendering_minigame::{
    StationRenderHealth, StationScene, default_asset_root,
    g1_background_visual::{
        BackgroundBodyPose, BackgroundOwnerFrame, G1BackgroundVisualInput, G1BackgroundVisualModel,
        G1BackgroundVisualPlugin, G1BackgroundVisualStatus,
    },
    g1_camera::{
        CameraPoseSource, G1ActiveCameraMount, G1BodyObservationInput, G1CameraJointState,
        G1CameraMountProfile, G1CameraNativeState, G1CameraPlugin, G1CameraPort, G1CaptureStamp,
        G1CapturedRgb, G1EgoCamera, G1ObservationPlugin,
    },
    g1_task_fiducial::{G1TaskFiducialModel, G1TaskFiducialPlugin, G1TaskFiducialReceipt},
    g1_task_visual::{
        G1TaskVisualFrame, G1TaskVisualInput, G1TaskVisualModel, G1TaskVisualPlugin,
        G1TaskVisualPose, G1TaskVisualStatus,
    },
    g1_visual::{G1VisualModel, G1VisualPlugin, G1VisualStatus},
    install_station_render_health, validate_render_asset_root,
};
use robot_minigame::g1::{agile::AgileCommand, contract::G1Command};
use serde::{Deserialize, Serialize};
use simulation_minigame::g1::{
    agile_runner::AgileRunnerConfig,
    runner::{G1ActuatorBackend, G1RunnerConfig},
    task_objects::{TaskObjectFrame, TaskObjectKind, TaskObjectSceneConfig},
    task_runner::{ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskRunnerConfig, ArenaTaskStep},
    worker::{
        AgileWorker, ArenaTaskWorker, G1Worker, G1WorkerPhase, G1WorkerTiming, OwnerStepTrace,
        TimedAgileCommand, TimedArenaTaskCommand, TimedG1Command, WorkerSnapshot,
    },
};
use task_minigame::{
    policy::{
        ARENA_ACTION_PERIOD_NS, MobilePolicyClient, PolicyInferenceRequest, PolicyObservation,
        PolicyWorker, StaticPolicyClient, map_measured_joints, profile_contract,
    },
    types::{ObservationStamp, TaskProfile},
};

use super::g1_marker_vision::MarkerVisionConfiguration;
#[cfg(feature = "g1_constraint_diagnostic")]
use super::g1_marker_vision::{MarkerTargetMemory, MarkerVisionJob};
use super::g1_source_lighting::{SourceLightingReceipt, SourceRectLighting};

/// Explicit evidence settings. Zero ticks captures the native initialized world
/// without inference or integration. Nonzero budgets are bounded by the
/// explicitly selected development scene and retain actual failures.
pub struct G1CaptureOptions {
    pub output: PathBuf,
    pub ticks: u32,
    pub timeout: Duration,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureConfiguration {
    runner: CaptureRunnerConfig,
    visual_path: PathBuf,
    visual_sha256: String,
    #[serde(default)]
    task_visual_path: Option<PathBuf>,
    #[serde(default)]
    task_visual_sha256: Option<String>,
    #[serde(default)]
    policy: Option<LivePolicyConfiguration>,
    #[serde(default)]
    exposure_ev100: Option<f32>,
    #[serde(default)]
    directional_shadow_maps: Option<bool>,
    /// Development-only renderer/compute contention comparison. Physics stays
    /// on its independent 50 Hz owner clock; no asset/quality setting changes.
    #[serde(default)]
    diagnostic_vsync: bool,
    /// One finite application-side contention control, never a physical clock.
    #[serde(default)]
    diagnostic_render_hz: Option<u32>,
    #[serde(default)]
    diagnostic_ambient_brightness: Option<f32>,
    #[serde(default)]
    diagnostic_directional_illuminance: Option<f32>,
    /// Source T2 uses ACES. This opt-in comparison uses Bevy's fitted ACES
    /// and an HDR intermediate; it does not claim RTX shader equivalence.
    #[serde(default)]
    diagnostic_aces_fitted: bool,
    /// One fixed quadrature of the original eight ceiling panels; no tuning.
    #[serde(default)]
    diagnostic_source_rect_lighting: Option<BackgroundVisualConfiguration>,
    #[serde(default)]
    predictive_limit_diagnostic: bool,
    #[serde(default)]
    diagnostic_constraint_sweeps: Option<u32>,
    #[serde(default)]
    background_visual: Option<BackgroundVisualConfiguration>,
    /// Renderer comparison only: a fresh zero-Tick world may be initialized
    /// at measured external poses. This never admits actions or steps physics.
    #[serde(default)]
    render_only_environment_translation: Option<[f64; 3]>,
    #[serde(default)]
    task_lab: Option<super::g1_task_lab::G1TaskLabConfiguration>,
    #[serde(default)]
    mobile_assist: Option<MobileAssistCaptureConfiguration>,
    #[serde(default)]
    mobile_scan: Option<MobileScanCaptureConfiguration>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MobileAssistCaptureConfiguration {
    heading_yaw_source_rad: f32,
    relative_distance_m: f32,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MobileScanCaptureConfiguration {
    heading_yaw_source_rad: f32,
    fiducial_assets: BackgroundVisualConfiguration,
    #[serde(default)]
    lowering: Option<MobileLowerCaptureConfiguration>,
    #[serde(default)]
    vision: Option<MarkerVisionConfiguration>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MobileLowerCaptureConfiguration {
    distance_m: f32,
    duration_ticks: u32,
}

enum MobileAssistStage {
    Carry(MobileAssistCaptureConfiguration),
    Scan(MobileScanCaptureConfiguration),
}

struct MobileAssistCaptureRuntime {
    scan_only: bool,
    view_with_lowering: bool,
    visual_approach: bool,
    raising_view: bool,
    memory_view: bool,
    restored_view: bool,
    auxiliary_view: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    auxiliary_view_completed: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    restore_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    reobserve_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    restored_view_completed: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    restored_view_rotation: Option<[f32; 4]>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    target_memory: Option<MarkerTargetMemory>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    memory_completed: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    vision_job: Option<MarkerVisionJob>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    visual_goal_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    raise_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    lower_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    configuration: MobileAssistStage,
    #[cfg(feature = "g1_constraint_diagnostic")]
    submitted: bool,
    completed: bool,
}

impl MobileAssistCaptureRuntime {
    fn new(
        configuration: MobileAssistStage,
        raising_view: bool,
        memory_view: bool,
        restored_view: bool,
        auxiliary_view: bool,
    ) -> Self {
        let scan_only = matches!(&configuration, MobileAssistStage::Scan(_));
        let view_with_lowering =
            matches!(&configuration, MobileAssistStage::Scan(c) if c.lowering.is_some());
        let visual_approach =
            matches!(&configuration, MobileAssistStage::Scan(c) if c.vision.is_some());
        #[cfg(not(feature = "g1_constraint_diagnostic"))]
        let _ = configuration;
        Self {
            scan_only,
            view_with_lowering,
            visual_approach,
            raising_view,
            memory_view,
            restored_view,
            auxiliary_view,
            #[cfg(feature = "g1_constraint_diagnostic")]
            auxiliary_view_completed: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            restore_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            reobserve_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            restored_view_completed: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            restored_view_rotation: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            target_memory: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            memory_completed: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            vision_job: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            visual_goal_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            raise_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            lower_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            configuration,
            #[cfg(feature = "g1_constraint_diagnostic")]
            submitted: false,
            completed: false,
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackgroundVisualConfiguration {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LivePolicyConfiguration {
    endpoint: String,
    max_calls: u32,
    timeout_ms: u64,
    #[serde(default)]
    prefetch_after_ticks: Option<u32>,
}

struct LivePolicyRuntime {
    worker: PolicyWorker,
    profile: TaskProfile,
    next_sequence: u64,
    pending: Option<ObservationStamp>,
    submitted_chunks: u32,
    max_calls: u32,
    next_boundary_tick: u64,
    prefetch_after_ticks: Option<u32>,
}

// Distinct complete startup schemas and typed workers; no action conversion.
#[derive(Deserialize)]
#[serde(untagged)]
enum CaptureRunnerConfig {
    Task(ArenaTaskRunnerConfig),
    Mobile(G1RunnerConfig),
    Static(AgileRunnerConfig),
}

impl CaptureRunnerConfig {
    fn task_objects(&self) -> Option<&TaskObjectSceneConfig> {
        match self {
            Self::Task(c) => match &c.body {
                ArenaTaskBodyConfig::StaticAgile(c) => c.task_objects.as_ref(),
                ArenaTaskBodyConfig::MobileHomieV2(c) => c.task_objects.as_ref(),
            },
            Self::Mobile(c) => c.task_objects.as_ref(),
            Self::Static(c) => c.task_objects.as_ref(),
        }
    }
    fn episode(&self) -> u64 {
        match self {
            Self::Task(c) => c.body.episode_id(),
            Self::Mobile(c) => c.episode_id,
            Self::Static(c) => c.episode_id,
        }
    }
    fn spawn(
        self,
        predictive_limit_diagnostic: bool,
        diagnostic_constraint_sweeps: Option<u32>,
        assisted_carry: bool,
    ) -> Result<CaptureWorker, String> {
        if assisted_carry {
            let Self::Task(config) = self else {
                return Err("assisted carry requires a matched mobile task owner".into());
            };
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                return simulation_minigame::g1::worker::MobileAssistWorker::spawn_mobile_assist(
                    config,
                )
                .map(CaptureWorker::AssistedMobile)
                .map_err(|e| e.to_string());
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            {
                let _ = config;
                return Err(
                    "assisted carry requires the explicit g1_constraint_diagnostic build feature"
                        .into(),
                );
            }
        }
        if diagnostic_constraint_sweeps == Some(4) && !predictive_limit_diagnostic {
            let Self::Task(config) = self else {
                return Err(
                    "mobile convergence candidate requires a matched mobile task owner".into(),
                );
            };
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                return ArenaTaskWorker::spawn_mobile_constraint_diagnostic(config)
                    .map(CaptureWorker::Task)
                    .map_err(|e| e.to_string());
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            {
                let _ = config;
                return Err(
                    "mobile convergence candidate requires its explicit development feature".into(),
                );
            }
        }
        if diagnostic_constraint_sweeps.is_some()
            && (diagnostic_constraint_sweeps != Some(16) || !predictive_limit_diagnostic)
        {
            return Err(
                "contact comparison permits only 16 sweeps with explicit predictive limits".into(),
            );
        }
        if predictive_limit_diagnostic {
            let Self::Task(config) = self else {
                return Err("predictive limit comparison requires the static task owner".into());
            };
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                let worker = if diagnostic_constraint_sweeps.is_some() {
                    ArenaTaskWorker::spawn_static_predictive_constraint_diagnostic(config)
                } else {
                    ArenaTaskWorker::spawn_static_predictive_limit_diagnostic(config)
                };
                return worker.map(CaptureWorker::Task).map_err(|e| e.to_string());
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            {
                let _ = config;
                return Err(
                    "predictive limit comparison requires its explicit development feature".into(),
                );
            }
        }
        match self {
            Self::Task(c) => ArenaTaskWorker::spawn(c)
                .map(CaptureWorker::Task)
                .map_err(|e| e.to_string()),
            Self::Mobile(c) => G1Worker::spawn(c)
                .map(CaptureWorker::Mobile)
                .map_err(|e| e.to_string()),
            Self::Static(c) => {
                c.validate().map_err(|e| e.to_string())?;
                let mut upper = [0.; 28];
                upper.copy_from_slice(&c.default_positions[15..]);
                let command = AgileCommand {
                    navigation: [0.; 3],
                    pelvis_height: 0.75,
                    upper_positions: upper,
                };
                AgileWorker::spawn(c)
                    .map(|worker| CaptureWorker::Static { worker, command })
                    .map_err(|e| e.to_string())
            }
        }
    }
}

enum CaptureWorker {
    Task(ArenaTaskWorker),
    #[cfg(feature = "g1_constraint_diagnostic")]
    AssistedMobile(simulation_minigame::g1::worker::MobileAssistWorker),
    Mobile(G1Worker),
    Static {
        worker: AgileWorker,
        command: AgileCommand,
    },
}

struct CaptureSnapshot {
    episode_id: u64,
    phase: G1WorkerPhase,
    reason: Option<String>,
    frame: Option<Arc<robot_minigame::g1::definition::G1BodyFrame>>,
    measurement: Option<Arc<simulation_minigame::g1::runner::G1Measurement>>,
    task_objects: Option<Arc<TaskObjectFrame>>,
    timing: G1WorkerTiming,
    task_step: Option<Arc<ArenaTaskStep>>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    assist_step: Option<Arc<simulation_minigame::g1::mobile_assist::MobileAssistStep>>,
}
impl CaptureSnapshot {
    fn from_worker<S>(s: &WorkerSnapshot<S>) -> Self {
        Self {
            episode_id: s.episode_id,
            phase: s.phase.clone(),
            reason: s.reason.clone(),
            frame: s.frame.clone(),
            measurement: s.measurement.clone(),
            task_objects: s.task_objects.clone(),
            timing: s.timing.clone(),
            task_step: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            assist_step: None,
        }
    }
}
impl CaptureWorker {
    fn take_latest(&self) -> Option<Arc<CaptureSnapshot>> {
        match self {
            #[cfg(feature = "g1_constraint_diagnostic")]
            Self::AssistedMobile(w) => w.take_latest().map(|s| {
                let mut snapshot=CaptureSnapshot::from_worker(&s);
                snapshot.assist_step=s.step.clone();
                if let Some(step)=&s.step {
                    if let simulation_minigame::g1::mobile_assist::MobileAssistExecution::OriginalVla(execution)=&step.execution {
                        snapshot.task_step=Some(Arc::new(ArenaTaskStep {execution:execution.clone(),body:step.body.clone()}));
                    }
                }
                Arc::new(snapshot)
            }),
            Self::Task(w) => w.take_latest().map(|s| {
                let mut snapshot = CaptureSnapshot::from_worker(&s);
                snapshot.task_step = s.step.clone();
                Arc::new(snapshot)
            }),
            Self::Mobile(w) => w
                .take_latest()
                .map(|s| Arc::new(CaptureSnapshot::from_worker(&s))),
            Self::Static { worker, .. } => worker
                .take_latest()
                .map(|s| Arc::new(CaptureSnapshot::from_worker(&s))),
        }
    }
    fn pause(&self) {
        match self {
            #[cfg(feature = "g1_constraint_diagnostic")]
            Self::AssistedMobile(w) => w.pause(),
            Self::Task(w) => w.pause(),
            Self::Mobile(w) => w.pause(),
            Self::Static { worker, .. } => worker.pause(),
        }
    }
    fn submit_stand(&self, episode_id: u64, ticks: u32) -> Result<(), String> {
        let valid_until_sim_ns = u64::from(ticks) * 20_000_000;
        let valid_until_wall = Instant::now() + Duration::from_secs(8);
        match self {
            #[cfg(feature = "g1_constraint_diagnostic")]
            Self::AssistedMobile(_) => {
                return Err("assisted mobile owner requires typed VLA/carry requests".into());
            }
            Self::Task(_) => return Err("task owner requires actual matched policy output".into()),
            Self::Mobile(w) => w.submit(TimedG1Command {
                episode_id,
                valid_until_sim_ns,
                valid_until_wall,
                command: G1Command::default(),
            }),
            Self::Static { worker, command } => worker.submit(TimedAgileCommand {
                episode_id,
                valid_until_sim_ns,
                valid_until_wall,
                command: command.clone(),
            }),
        }
        .map_err(|e| e.to_string())
    }
}

impl CaptureWorker {
    fn submit_live_chunk(&self, timed: TimedArenaTaskCommand) -> Result<(), String> {
        match self {
            Self::Task(worker) => worker.submit(timed).map_err(|e| e.to_string()),
            #[cfg(feature = "g1_constraint_diagnostic")]
            Self::AssistedMobile(worker) => worker
                .submit(simulation_minigame::g1::worker::TimedCommand {
                    episode_id: timed.episode_id,
                    valid_until_sim_ns: timed.valid_until_sim_ns,
                    valid_until_wall: timed.valid_until_wall,
                    command:
                        simulation_minigame::g1::mobile_assist::MobileAssistCommand::OriginalVla(
                            timed.command,
                        ),
                })
                .map_err(|e| e.to_string()),
            Self::Mobile(_) | Self::Static { .. } => {
                Err("live task lost its unique typed owner".into())
            }
        }
    }
}

/// Read the caller's frozen configuration and run this diagnostic explicitly.
pub fn run_capture_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::Camera)
}

/// Two fixed1000/1500Tick original mobile carry-stage budgets. This does
/// not widen the ordinary camera/static-task budgets or qualify task execution.
pub fn run_mobile_carry_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileCarry)
}

/// Four fresh native RGB/VLA grasp chunks, then one explicitly traditional
/// clear-aisle carry goal. No replay input or qualified task claim is admitted.
pub fn run_mobile_assist_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileAssist)
}

/// Four fresh unmarked grasp images, then a finite marked turn-only observation.
pub fn run_mobile_scan_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileScan)
}

/// Matched live grasp, public-map scan, bounded lowering, then actual marker RGB.
pub fn run_mobile_target_view_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileTargetView)
}

/// Four fresh original RGB grasp chunks, scan/lower, then actual RGB coarse approach.
/// Development evidence only; no release, arbitrary target, or task qualification.
pub fn run_mobile_target_approach_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileTargetApproach)
}

/// Fresh near RGB/public geometry raises the closed grip, then captures a new
/// actual image. This grants no retention, release or task qualification.
pub fn run_mobile_target_raise_view_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileTargetRaiseView)
}

/// Reobserve the held box with recent static-target memory and original self
/// velocity. The proposed final approach is not executed by this development entry.
pub fn run_mobile_target_memory_view_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileTargetMemoryView)
}

/// Restore the episode's calibrated grip, capture a new current image, then
/// perform one disclosed standing view turn and localize a new actual RGB.
pub fn run_mobile_target_restored_view_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileTargetRestoredView)
}

/// Original unmarked grasp and calibrated standing scan, followed by a fixed
/// auxiliary sensor/public-label view. No visibility-driven arm motion.
pub fn run_mobile_auxiliary_view_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileAuxiliaryView)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureMode {
    Camera,
    TaskLab,
    MobileCarry,
    MobileAssist,
    MobileScan,
    MobileTargetView,
    MobileTargetApproach,
    MobileTargetRaiseView,
    MobileTargetMemoryView,
    MobileTargetRestoredView,
    MobileAuxiliaryView,
}

impl CaptureMode {
    fn assisted_tick_limit(self) -> Option<u32> {
        match self {
            Self::MobileAssist => Some(2050),
            Self::MobileScan | Self::MobileAuxiliaryView => Some(1050),
            Self::MobileTargetView => Some(1300),
            Self::MobileTargetApproach
            | Self::MobileTargetRaiseView
            | Self::MobileTargetMemoryView
            | Self::MobileTargetRestoredView => Some(3150),
            Self::Camera | Self::TaskLab | Self::MobileCarry => None,
        }
    }
}

/// Interactive development entry; it never advertises unqualified execution.
pub fn run_task_lab_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::TaskLab)
}

fn run_from_file(
    path: &Path,
    options: G1CaptureOptions,
    mode: CaptureMode,
) -> Result<G1CaptureReceipt, String> {
    let interactive = mode == CaptureMode::TaskLab;
    let config: CaptureConfiguration = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?,
    )
    .map_err(|error| format!("G1 diagnostic configuration: {error}"))?;
    if config.task_visual_path.is_some() != config.task_visual_sha256.is_some() {
        return Err("task visual path/hash must both be supplied".into());
    }
    if matches!(config.runner, CaptureRunnerConfig::Task(_)) != config.policy.is_some() {
        return Err("task owner and live policy configuration must be supplied together".into());
    }
    if config.task_lab.is_some() != interactive
        || (interactive
            && (options.ticks != 0 || !matches!(config.runner, CaptureRunnerConfig::Task(_))))
    {
        return Err("task_lab configuration requires the explicit g1_task_lab scene, matched task owner and zero automatic Ticks".into());
    }
    run_capture_owner(
        config.runner,
        &config.visual_path,
        &config.visual_sha256,
        config
            .task_visual_path
            .as_deref()
            .zip(config.task_visual_sha256.as_deref()),
        options,
        config.policy,
        config.exposure_ev100,
        config.directional_shadow_maps,
        config.diagnostic_vsync,
        config.diagnostic_render_hz,
        config.diagnostic_ambient_brightness,
        config.diagnostic_directional_illuminance,
        config.diagnostic_aces_fitted,
        config.diagnostic_source_rect_lighting,
        config.predictive_limit_diagnostic,
        config.diagnostic_constraint_sweeps,
        config.background_visual,
        config.render_only_environment_translation,
        config.task_lab,
        config.mobile_assist,
        config.mobile_scan,
        mode,
    )
}

impl Default for G1CaptureOptions {
    fn default() -> Self {
        Self {
            output: ".scratch/g1_native_camera".into(),
            ticks: 0,
            timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct G1CaptureReceipt {
    pub schema_version: u32,
    pub scope: &'static str,
    pub capture_succeeded: bool,
    pub task_qualified: bool,
    pub environment: &'static str,
    pub floor_center_engine: [f32; 3],
    pub floor_full_extents_m: [f32; 3],
    pub floor_contact_friction: f32,
    pub robot_contact_friction: f32,
    pub definition_sha256: String,
    pub ort_sha256: String,
    pub body_profile: &'static str,
    pub stand_sha256: Option<&'static str>,
    pub walk_sha256: Option<&'static str>,
    pub agile_sha256: Option<&'static str>,
    pub task_visual_sha256: Option<String>,
    pub last_task_object_frame: Option<TaskObjectFrame>,
    pub task_visual_status: Option<G1TaskVisualStatus>,
    pub visual_sha256: String,
    pub body_bound_material_mesh_count: usize,
    pub actuator_backend: G1ActuatorBackend,
    pub physics_hz: u32,
    pub integrations_per_tick: u32,
    pub requested_ticks: u32,
    pub actual_integrations: u64,
    pub actual_torque_updates: u64,
    pub actual_model_attempts: u64,
    pub actual_model_successes: u64,
    pub owner_step_records: u64,
    pub owner_step_trace_dropped: u64,
    pub owner_step_trace_complete: bool,
    pub diagnostic_prefetch_after_ticks: Option<u32>,
    pub prefetch_discarded_image_stamps: Vec<serde_json::Value>,
    pub continuous_boundary_sim_seconds: f64,
    pub continuous_boundary_wall_seconds: f64,
    pub active_wall_seconds: f64,
    pub active_sim_seconds: f64,
    pub control_deadlines_missed: u64,
    pub pending_ticks: u64,
    pub physics_outcome: String,
    pub failure_reason: Option<String>,
    pub ego_stamp: Option<G1CaptureStamp>,
    pub main_resolution: [u32; 2],
    pub render_present_mode: &'static str,
    pub diagnostic_render_hz: Option<u32>,
    pub live_policy_inference_calls: u32,
    pub live_policy_successes: u32,
    pub live_action_chunks: Vec<serde_json::Value>,
    pub pauses_for_camera_and_policy: bool,
    pub mobile_assist_handoff: Option<serde_json::Value>,
    pub disclosed_fiducials: Option<G1TaskFiducialReceipt>,
    pub camera_exposure_ev100: f32,
    pub diagnostic_directional_shadow_maps: bool,
    pub camera_ambient_brightness: f32,
    pub camera_directional_illuminance: f32,
    pub diagnostic_aces_fitted: bool,
    pub ego_camera_hdr: Option<bool>,
    pub ego_camera_tonemapping: Option<&'static str>,
    source_rect_lighting: Option<SourceLightingReceipt>,
    source_light_masks_gpu_supported: Option<bool>,
    pub factory_verified_predictive_limit_joints: usize,
    pub factory_verified_diagnostic_constraint_sweeps: Option<u32>,
    pub background_visual_status: Option<G1BackgroundVisualStatus>,
    pub external_pose_render_only: bool,
    pub unqualified: Vec<&'static str>,
}

impl G1CaptureReceipt {
    fn initial(ticks: u32, config: &CaptureRunnerConfig, visual_sha256: &str) -> Self {
        let (
            episode_profile,
            floor_friction,
            robot_friction,
            definition_sha256,
            ort_sha256,
            backend,
            stand,
            walk,
            agile,
        ) = match config {
            CaptureRunnerConfig::Task(c) => match &c.body {
                ArenaTaskBodyConfig::StaticAgile(c) => (
                    "static_agile",
                    c.floor_contact_friction,
                    c.robot_contact_friction,
                    c.definition_sha256.clone(),
                    c.ort_sha256.clone(),
                    G1ActuatorBackend::NativeForceBased,
                    None,
                    None,
                    Some(robot_minigame::g1::agile::MODEL_SHA256),
                ),
                ArenaTaskBodyConfig::MobileHomieV2(c) => (
                    "mobile_homie",
                    c.floor_contact_friction,
                    c.robot_contact_friction,
                    c.definition_sha256.clone(),
                    c.ort_sha256.clone(),
                    c.actuator_backend,
                    Some(robot_minigame::g1::policy::STAND_SHA256),
                    Some(robot_minigame::g1::policy::WALK_SHA256),
                    None,
                ),
            },
            CaptureRunnerConfig::Mobile(c) => (
                "mobile_homie",
                c.floor_contact_friction,
                c.robot_contact_friction,
                c.definition_sha256.clone(),
                c.ort_sha256.clone(),
                c.actuator_backend,
                Some(robot_minigame::g1::policy::STAND_SHA256),
                Some(robot_minigame::g1::policy::WALK_SHA256),
                None,
            ),
            CaptureRunnerConfig::Static(c) => (
                "static_agile",
                c.floor_contact_friction,
                c.robot_contact_friction,
                c.definition_sha256.clone(),
                c.ort_sha256.clone(),
                G1ActuatorBackend::NativeForceBased,
                None,
                None,
                Some(robot_minigame::g1::agile::MODEL_SHA256),
            ),
        };
        Self {
            schema_version: 2,
            scope: "native_g1_camera_pairing_diagnostic",
            capture_succeeded: false,
            task_qualified: false,
            environment: "source_near_simple_floor_not_science_station",
            floor_center_engine: [0., -0.25, 0.],
            floor_full_extents_m: [40., 0.5, 40.],
            floor_contact_friction: floor_friction,
            robot_contact_friction: robot_friction,
            definition_sha256,
            ort_sha256,
            body_profile: episode_profile,
            stand_sha256: stand,
            walk_sha256: walk,
            agile_sha256: agile,
            task_visual_sha256: None,
            last_task_object_frame: None,
            task_visual_status: None,
            visual_sha256: visual_sha256.into(),
            body_bound_material_mesh_count: 0,
            actuator_backend: backend,
            physics_hz: 50,
            integrations_per_tick: 1,
            requested_ticks: ticks,
            actual_integrations: 0,
            actual_torque_updates: 0,
            actual_model_attempts: 0,
            actual_model_successes: 0,
            owner_step_records: 0,
            owner_step_trace_dropped: 0,
            owner_step_trace_complete: false,
            diagnostic_prefetch_after_ticks: None,
            prefetch_discarded_image_stamps: Vec::new(),
            continuous_boundary_sim_seconds: 0.,
            continuous_boundary_wall_seconds: 0.,
            active_wall_seconds: 0.,
            active_sim_seconds: 0.,
            control_deadlines_missed: 0,
            pending_ticks: 0,
            physics_outcome: "loading".into(),
            failure_reason: None,
            ego_stamp: None,
            main_resolution: [1920, 1080],
            render_present_mode: "auto_no_vsync",
            diagnostic_render_hz: None,
            live_policy_inference_calls: 0,
            live_policy_successes: 0,
            live_action_chunks: Vec::new(),
            pauses_for_camera_and_policy: false,
            mobile_assist_handoff: None,
            disclosed_fiducials: None,
            camera_exposure_ev100: Exposure::default().ev100,
            diagnostic_directional_shadow_maps: true,
            camera_ambient_brightness: 450.,
            camera_directional_illuminance: 15_000.,
            diagnostic_aces_fitted: false,
            ego_camera_hdr: None,
            ego_camera_tonemapping: None,
            source_rect_lighting: None,
            source_light_masks_gpu_supported: None,
            factory_verified_predictive_limit_joints: 0,
            factory_verified_diagnostic_constraint_sweeps: None,
            background_visual_status: None,
            external_pose_render_only: false,
            unqualified: vec![
                "standing_stability",
                "science_station_contact_geometry",
                "qwen_visual_decision",
                "t1_pick_place",
                "t2_walking_carry",
                "real_time_task_qualification",
            ],
        }
    }

    fn observe(&mut self, snapshot: &CaptureSnapshot) {
        self.last_task_object_frame = snapshot.task_objects.as_deref().cloned();
        self.actual_integrations = snapshot.timing.total_integrations;
        self.actual_torque_updates = snapshot.timing.total_torque_updates;
        self.actual_model_attempts = snapshot.timing.total_inference_attempts;
        self.actual_model_successes = snapshot.timing.total_successful_inferences;
        self.active_wall_seconds = snapshot.timing.active_wall_seconds;
        self.active_sim_seconds = snapshot.timing.active_sim_seconds;
        self.control_deadlines_missed = snapshot.timing.control_deadlines_missed;
        self.pending_ticks = snapshot.timing.pending_ticks;
        if snapshot.phase == G1WorkerPhase::Failed {
            self.physics_outcome = "candidate_failed".into();
            self.failure_reason = snapshot.reason.clone();
        }
    }
}

#[derive(Resource, Clone)]
struct CaptureOutcome(Arc<Mutex<G1CaptureReceipt>>);

#[derive(Component)]
struct SourceShelfVisual;

#[derive(Resource)]
struct DiagnosticPhotometry {
    exposure_ev100: f32,
    aces_fitted: bool,
}
#[derive(Resource)]
struct DiagnosticIllumination {
    ambient_brightness: f32,
    directional_illuminance: f32,
    source_rect_lighting: Option<SourceRectLighting>,
}

#[derive(Resource)]
struct DiagnosticDirectionalShadows(bool);

fn configure_diagnostic_exposure(
    mut commands: Commands,
    setting: Res<DiagnosticPhotometry>,
    outcome: Res<CaptureOutcome>,
    mut cameras: Query<
        (
            Entity,
            Option<&mut Exposure>,
            Option<&Hdr>,
            Option<&Tonemapping>,
            Option<&G1EgoCamera>,
        ),
        With<Camera3d>,
    >,
) {
    for (entity, camera, hdr, tonemapping, ego) in &mut cameras {
        if let Some(mut camera) = camera {
            camera.ev100 = setting.exposure_ev100;
        } else {
            commands.entity(entity).insert(Exposure {
                ev100: setting.exposure_ev100,
            });
        }
        if setting.aces_fitted && hdr.is_none() {
            commands
                .entity(entity)
                .insert((Hdr, Tonemapping::AcesFitted));
        }
        if ego.is_some() {
            let mut receipt = outcome.0.lock().unwrap();
            receipt.ego_camera_hdr = Some(hdr.is_some());
            receipt.ego_camera_tonemapping = Some(match tonemapping {
                Some(Tonemapping::None) => "none",
                Some(Tonemapping::AcesFitted) => "aces_fitted",
                _ => "unexpected",
            });
        }
    }
}

fn reserve_source_light_clusters(
    illumination: Res<DiagnosticIllumination>,
    mut settings: ResMut<GlobalClusterSettings>,
    outcome: Res<CaptureOutcome>,
    mut exit: MessageWriter<AppExit>,
) {
    if illumination.source_rect_lighting.is_some() {
        outcome.0.lock().unwrap().source_light_masks_gpu_supported =
            Some(settings.clustered_decals_are_usable);
        if !settings.clustered_decals_are_usable {
            outcome.0.lock().unwrap().failure_reason =
                Some("GPU lacks source-light texture masks; no isotropic fallback admitted".into());
            exit.write(AppExit::error());
            return;
        }
        if let Some(gpu) = &mut settings.gpu_clustering {
            // Capacity measured by the zero-step32light diagnostic. Reserve it
            // before extraction so initial model images do not see overflow.
            gpu.initial_index_list_capacity = gpu.initial_index_list_capacity.max(131_072);
        }
    }
}

#[derive(Resource)]
pub(super) struct CaptureRuntime {
    worker: CaptureWorker,
    options: G1CaptureOptions,
    episode_id: u64,
    started: Instant,
    render_frames: u32,
    latest: Option<Arc<CaptureSnapshot>>,
    command_submitted: bool,
    requested: bool,
    ego_saved: bool,
    main_saved: Arc<Mutex<Result<bool, String>>>,
    live_policy: Option<LivePolicyRuntime>,
    owner_evidence: Option<OwnerEvidence>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    assist_evidence:
        Option<OwnerEvidence<simulation_minigame::g1::mobile_assist::MobileAssistStep>>,
    mobile_assist: Option<MobileAssistCaptureRuntime>,
    interactive: bool,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CaptureSystems {
    OwnerSnapshot,
}

impl CaptureRuntime {
    pub(super) fn lab_render_frames(&self) -> u64 {
        self.render_frames.into()
    }

    pub(super) fn lab_pause(&self) {
        self.worker.pause();
    }

    pub(super) fn lab_reset(&mut self) -> Result<u64, String> {
        self.worker.pause();
        let episode_id = self.episode_id.checked_add(1).ok_or("episode exhausted")?;
        match &self.worker {
            #[cfg(feature = "g1_constraint_diagnostic")]
            CaptureWorker::AssistedMobile(worker) => worker.reset(),
            CaptureWorker::Task(worker) => worker.reset(),
            CaptureWorker::Mobile(worker) => worker.reset(),
            CaptureWorker::Static { worker, .. } => worker.reset(),
        }
        .map_err(|error| error.to_string())?;
        if let Some(live) = &mut self.live_policy {
            live.worker
                .reset_episode(episode_id)
                .map_err(|error| format!("{error:?}"))?;
            live.pending = None;
            live.next_sequence = 1;
            live.submitted_chunks = 0;
            live.next_boundary_tick = 0;
        }
        self.episode_id = episode_id;
        self.latest = None;
        self.requested = false;
        Ok(episode_id)
    }

    pub(super) fn lab_counts(&self) -> serde_json::Value {
        let timing = self.latest.as_ref().map(|latest| &latest.timing);
        serde_json::json!({
            "episode_id": self.episode_id,
            "actual_total_integrations": timing.map_or(0, |timing| timing.total_integrations),
            "actual_episode_integrations": timing.map_or(0, |timing| timing.episode_integrations),
            "render_frames": self.render_frames,
            "task_qualified": false,
        })
    }

    pub(super) fn lab_clock(&self) -> Option<task_minigame::interactive::TaskOwnerClock> {
        self.latest
            .as_ref()
            .map(|latest| task_minigame::interactive::TaskOwnerClock {
                episode_id: latest.episode_id,
                sim_time_ns: latest.timing.episode_integrations * ARENA_ACTION_PERIOD_NS,
                ready: latest.frame.is_some() && latest.measurement.is_some(),
                failure: (latest.phase == G1WorkerPhase::Failed).then(|| {
                    latest
                        .reason
                        .clone()
                        .unwrap_or("physical owner failed".into())
                }),
            })
    }
}

/// Development-only file output. No consumer I/O runs on the physics thread.
trait CapturedTaskStep {
    type Execution: Serialize;
    fn execution(&self) -> &Self::Execution;
    fn body(&self) -> &simulation_minigame::g1::task_runner::ArenaBodyStep;
}
impl CapturedTaskStep for ArenaTaskStep {
    type Execution = simulation_minigame::g1::task_runner::ArenaTaskExecution;
    fn execution(&self) -> &Self::Execution {
        &self.execution
    }
    fn body(&self) -> &simulation_minigame::g1::task_runner::ArenaBodyStep {
        &self.body
    }
}
#[cfg(feature = "g1_constraint_diagnostic")]
impl CapturedTaskStep for simulation_minigame::g1::mobile_assist::MobileAssistStep {
    type Execution = simulation_minigame::g1::mobile_assist::MobileAssistExecution;
    fn execution(&self) -> &Self::Execution {
        &self.execution
    }
    fn body(&self) -> &simulation_minigame::g1::task_runner::ArenaBodyStep {
        &self.body
    }
}

struct OwnerEvidence<S = ArenaTaskStep> {
    trace: OwnerStepTrace<S>,
    output: BufWriter<fs::File>,
    records: u64,
    first_boundary_wall_ms: Option<f64>,
}

impl<S: CapturedTaskStep> OwnerEvidence<S> {
    fn drain(&mut self, receipt: &mut G1CaptureReceipt) -> Result<(), String> {
        for record in self.trace.drain() {
            let value = serde_json::json!({
                "schema": "g1_owner_completed_step_evidence_v1",
                "episode_id": record.episode_id,
                "generation": record.generation,
                "owner_total_integrations": record.timing.total_integrations,
                "owner_episode_integrations": record.timing.episode_integrations,
                "owner_total_torque_updates": record.timing.total_torque_updates,
                "owner_total_model_attempts": record.timing.total_inference_attempts,
                "owner_total_model_successes": record.timing.total_successful_inferences,
                "active_wall_seconds": record.timing.active_wall_seconds,
                "active_sim_seconds": record.timing.active_sim_seconds,
                "pending_ticks": record.timing.pending_ticks,
                "control_deadlines_missed": record.timing.control_deadlines_missed,
                "last_boundary_wall_ms": record.timing.last_boundary_wall_ms,
                "last_boundary_duration_ms": record.timing.last_boundary_duration_ms,
                "execution": record.step.execution(),
                "body": record.step.body(),
                "scope": "independent_actual_owner_step_truth_never_model_input",
            });
            serde_json::to_writer(&mut self.output, &value).map_err(|e| e.to_string())?;
            self.output.write_all(b"\n").map_err(|e| e.to_string())?;
            self.records += 1;
            let first = *self
                .first_boundary_wall_ms
                .get_or_insert(record.timing.last_boundary_wall_ms);
            receipt.continuous_boundary_sim_seconds = self.records.saturating_sub(1) as f64 * 0.02;
            receipt.continuous_boundary_wall_seconds =
                (record.timing.last_boundary_wall_ms - first) / 1000.;
        }
        self.output.flush().map_err(|e| e.to_string())?;
        receipt.owner_step_records = self.records;
        receipt.owner_step_trace_dropped = self.trace.dropped_records();
        receipt.owner_step_trace_complete =
            receipt.owner_step_trace_dropped == 0 && self.records == receipt.actual_integrations;
        if receipt.owner_step_trace_dropped > 0 {
            return Err(
                "bounded owner evidence overflow/disconnection; physical trace incomplete".into(),
            );
        }
        Ok(())
    }
}

/// Open a real 1080p window and save one real head-camera RGB frame with native
/// measurements bound at scene extraction. The caller supplies frozen model
/// identities; the worker is the sole owner of world, control and integration.
pub fn run_capture(
    config: G1RunnerConfig,
    visual_path: &Path,
    visual_sha256: &str,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_capture_owner(
        CaptureRunnerConfig::Mobile(config),
        visual_path,
        visual_sha256,
        None,
        options,
        None,
        None,
        None,
        false,
        None,
        None,
        None,
        false,
        None,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        CaptureMode::Camera,
    )
}

fn run_capture_owner(
    config: CaptureRunnerConfig,
    visual_path: &Path,
    visual_sha256: &str,
    task_visual: Option<(&Path, &str)>,
    options: G1CaptureOptions,
    policy: Option<LivePolicyConfiguration>,
    exposure_ev100: Option<f32>,
    directional_shadow_maps: Option<bool>,
    diagnostic_vsync: bool,
    diagnostic_render_hz: Option<u32>,
    diagnostic_ambient_brightness: Option<f32>,
    diagnostic_directional_illuminance: Option<f32>,
    diagnostic_aces_fitted: bool,
    diagnostic_source_rect_lighting: Option<BackgroundVisualConfiguration>,
    predictive_limit_diagnostic: bool,
    diagnostic_constraint_sweeps: Option<u32>,
    background_visual: Option<BackgroundVisualConfiguration>,
    render_only_environment_translation: Option<[f64; 3]>,
    task_lab: Option<super::g1_task_lab::G1TaskLabConfiguration>,
    mobile_assist: Option<MobileAssistCaptureConfiguration>,
    mobile_scan: Option<MobileScanCaptureConfiguration>,
    mode: CaptureMode,
) -> Result<G1CaptureReceipt, String> {
    let interactive = mode == CaptureMode::TaskLab;
    let mobile_carry = mode == CaptureMode::MobileCarry;
    let memory_view = mode == CaptureMode::MobileTargetMemoryView;
    let restored_view = mode == CaptureMode::MobileTargetRestoredView;
    let auxiliary_view = mode == CaptureMode::MobileAuxiliaryView;
    let raising_view = mode == CaptureMode::MobileTargetRaiseView || memory_view || restored_view;
    let visual_approach =
        mode == CaptureMode::MobileTargetApproach || raising_view || auxiliary_view;
    let target_view = mode == CaptureMode::MobileTargetView || (visual_approach && !auxiliary_view);
    let scan_only = mode == CaptureMode::MobileScan || target_view || auxiliary_view;
    let assisted_carry = mode == CaptureMode::MobileAssist || scan_only;
    if (mode == CaptureMode::MobileAssist) != mobile_assist.is_some()
        || scan_only != mobile_scan.is_some()
    {
        return Err("mobile_assist configuration requires its explicit development scene".into());
    }
    if assisted_carry
        && (!cfg!(feature = "g1_constraint_diagnostic")
            || options.ticks != mode.assisted_tick_limit().unwrap_or(0)
            || !matches!(&config,CaptureRunnerConfig::Task(c) if c.body.profile()==TaskProfile::MobileBox)
            || config
                .task_objects()
                .is_none_or(|c| c.source_t2_background.is_none())
            || policy
                .as_ref()
                .is_none_or(|p| p.max_calls != 4 || p.prefetch_after_ticks.is_some())
            || diagnostic_constraint_sweeps != Some(4)
            || predictive_limit_diagnostic
            || diagnostic_source_rect_lighting.is_none()
            || !diagnostic_aces_fitted
            || task_lab.is_some()
            || (scan_only && task_visual.is_none())
            || mobile_assist.as_ref().is_some_and(|c| {
                !c.heading_yaw_source_rad.is_finite()
                    || c.heading_yaw_source_rad.abs() > std::f32::consts::PI
                    || c.relative_distance_m != 2.
            })
            || mobile_scan.as_ref().is_some_and(|c| {
                !c.heading_yaw_source_rad.is_finite()
                    || c.heading_yaw_source_rad.abs() > std::f32::consts::PI
                    || c.lowering.is_some() != target_view
                    || c.vision.is_some() != visual_approach
                    || c.vision
                        .as_ref()
                        .is_some_and(|v| v.task_geometry.is_some() != raising_view)
                    || c.lowering
                        .as_ref()
                        .is_some_and(|l| l.distance_m != 0.18 || l.duration_ticks != 150)
            }))
    {
        return Err("mobile development entry requires its exact 2050/1050/1300/3150 Tick budget, four fresh original grasp chunks and matched source scene/light/4PGS profile; manual carry is exactly 2m".into());
    }
    if let Some(vision) = mobile_scan.as_ref().and_then(|c| c.vision.as_ref()) {
        vision.validate()?;
    }
    let fiducial_model = mobile_scan
        .as_ref()
        .map(|c| G1TaskFiducialModel::load(&c.fiducial_assets.path, &c.fiducial_assets.sha256))
        .transpose()?;
    if let Some(model) = &fiducial_model {
        let auxiliary_labels = model.receipt.layout_profile
            == rendering_minigame::g1_task_fiducial::G1FiducialLayoutProfile::AuxiliaryGripTargets;
        if auxiliary_labels != auxiliary_view {
            return Err(
                "auxiliary printed labels require their separate fixed-sensor view entry".into(),
            );
        }
    }
    if let Some(scan) = &mobile_scan {
        if auxiliary_view
            && scan
                .vision
                .as_ref()
                .is_none_or(|v| v.fiducial_calibration.is_none())
        {
            return Err("auxiliary view requires the bound rendered marker calibration".into());
        }
        if scan
            .vision
            .as_ref()
            .and_then(|v| v.fiducial_calibration.as_ref())
            .is_some_and(|c| {
                c.path != scan.fiducial_assets.path || c.sha256 != scan.fiducial_assets.sha256
            })
        {
            return Err(
                "perception calibration must identify the rendered printed-marker document".into(),
            );
        }
    }
    let mobile_stage = mobile_assist
        .map(MobileAssistStage::Carry)
        .or_else(|| mobile_scan.map(MobileAssistStage::Scan));
    if mobile_carry
        && (!matches!(options.ticks, 1000 | 1500)
            || !matches!(&config, CaptureRunnerConfig::Task(c) if c.body.profile() == TaskProfile::MobileBox)
            || config
                .task_objects()
                .is_none_or(|c| c.source_t2_background.is_none())
            || policy.as_ref().is_none_or(|p| {
                p.max_calls != options.ticks / 50 || p.prefetch_after_ticks.is_some()
            })
            || diagnostic_constraint_sweeps != Some(4)
            || predictive_limit_diagnostic
            || diagnostic_source_rect_lighting.is_none()
            || !diagnostic_aces_fitted
            || task_lab.is_some())
    {
        return Err("mobile carry entry requires1000/1500Ticks and20/30original chunks, matched T2 scene/source-light profile and the existing4PGS candidate; static/UI/prefetch modes are forbidden".into());
    }
    if diagnostic_render_hz.is_some_and(|hz| hz != 60) {
        return Err(
            "renderer contention comparison permits only an explicit 60 Hz display limit".into(),
        );
    }
    let task_profile = match &config {
        CaptureRunnerConfig::Task(config) => Some(config.body.profile()),
        CaptureRunnerConfig::Mobile(_) | CaptureRunnerConfig::Static(_) => None,
    };
    if diagnostic_aces_fitted
        && (background_visual.is_none()
            || !matches!(
                &config,
                CaptureRunnerConfig::Mobile(_)
                    | CaptureRunnerConfig::Task(ArenaTaskRunnerConfig {
                        body: ArenaTaskBodyConfig::MobileHomieV2(_),
                        ..
                    })
            )
            || interactive)
    {
        return Err("ACES comparison requires the explicit original T2 background and mobile body diagnostic".into());
    }
    if diagnostic_source_rect_lighting.is_some() && !diagnostic_aces_fitted {
        return Err("source panel comparison requires the explicit mobile ACES diagnostic".into());
    }
    if diagnostic_source_rect_lighting.is_some() && !cfg!(feature = "g1_source_lighting") {
        return Err(
            "source panel masks require the explicit development g1_source_lighting build feature"
                .into(),
        );
    }
    let source_rect_lighting = diagnostic_source_rect_lighting
        .map(|input| SourceRectLighting::load(&input.path, &input.sha256))
        .transpose()?;
    let exposure_ev100 = exposure_ev100.unwrap_or(Exposure::default().ev100);
    let directional_shadow_maps = directional_shadow_maps.unwrap_or(true);
    let ambient_brightness = diagnostic_ambient_brightness.unwrap_or(450.);
    let directional_illuminance = diagnostic_directional_illuminance.unwrap_or(15_000.);
    if !ambient_brightness.is_finite()
        || !(0.0..=20_000.).contains(&ambient_brightness)
        || !directional_illuminance.is_finite()
        || !(0.0..=50_000.).contains(&directional_illuminance)
        || ((diagnostic_ambient_brightness.is_some()
            || diagnostic_directional_illuminance.is_some())
            && background_visual.is_none())
    {
        return Err("diffuse-light calibration needs the original background and finite bounded ambient/directional energies".into());
    }
    if !exposure_ev100.is_finite() || !(0.0..=20.0).contains(&exposure_ev100) {
        return Err("diagnostic exposure must be finite EV100 in 0..=20".into());
    }
    if let Some(translation) = render_only_environment_translation {
        let valid_static_scene = match &config {
            CaptureRunnerConfig::Static(c) => c.task_objects.as_ref().is_some_and(|objects| {
                objects.source_t1_shelf.is_none()
                    && objects.placements.len() == 2
                    && objects
                        .placements
                        .iter()
                        .any(|p| p.kind == TaskObjectKind::Apple)
                    && objects
                        .placements
                        .iter()
                        .any(|p| p.kind == TaskObjectKind::Plate)
            }),
            _ => false,
        };
        if translation != [0., 0., 0.795]
            || options.ticks != 0
            || policy.is_some()
            || predictive_limit_diagnostic
            || diagnostic_constraint_sweeps.is_some()
            || !valid_static_scene
            || background_visual.is_none()
        {
            return Err("external-pose renderer comparison requires zero ticks, the static apple/plate scene without a physical shelf, original background and translation [0,0,0.795]; actions and solver diagnostics are forbidden".into());
        }
    }
    let background_model = background_visual
        .map(|background| {
            if let Some(t2) = config
                .task_objects()
                .and_then(|c| c.source_t2_background.as_ref())
            {
                if render_only_environment_translation.is_some() {
                    return Err("mobile background requires actual completed physical poses".into());
                }
                G1BackgroundVisualModel::load_mobile(
                    &background.path,
                    &background.sha256,
                    t2.environment_translation_source,
                    &t2.definition_sha256,
                )
            } else {
                let translation = match render_only_environment_translation {
                    Some(translation) => translation,
                    None => {
                        config
                            .task_objects()
                            .and_then(|c| c.source_t1_shelf.as_ref())
                            .ok_or("original background needs the frozen source scene profile")?
                            .environment_translation_source
                    }
                };
                G1BackgroundVisualModel::load(&background.path, &background.sha256, translation)
            }
        })
        .transpose()?;
    let tick_limit = if let Some(limit) = mode.assisted_tick_limit() {
        limit
    } else if mobile_carry {
        1500
    } else if matches!(config, CaptureRunnerConfig::Task(_)) {
        400
    } else {
        150
    };
    if options.ticks > tick_limit
        || options.timeout.is_zero()
        || options.timeout > Duration::from_secs(180)
    {
        return Err(
            "capture exceeds its bounded tick budget or finite timeout <=180 seconds".into(),
        );
    }
    let live_policy = match (&config, policy) {
        (CaptureRunnerConfig::Task(config), Some(policy)) => {
            let profile = config.body.profile();
            let horizon = profile_contract(profile).action_horizon as u32;
            if policy.max_calls == 0
                || policy.max_calls > if mobile_carry { 30 } else { 8 }
                || (!interactive && !assisted_carry && options.ticks != policy.max_calls * horizon)
                || policy.timeout_ms == 0
                || policy.timeout_ms > 20_000
                || config.max_observation_wall_age_ms == 0
                || config.max_observation_wall_age_ms > 20_000
                || policy.prefetch_after_ticks.is_some_and(|tick| {
                    profile != TaskProfile::StaticApple || tick != 10 || policy.max_calls < 2
                })
            {
                return Err("live diagnostic requires its bounded whole-chunk count, exact tick budget and image/inference age <=20s".into());
            }
            let timeout = Duration::from_millis(policy.timeout_ms);
            let worker = match profile {
                TaskProfile::StaticApple => PolicyWorker::spawn(
                    StaticPolicyClient::new(&policy.endpoint, timeout)
                        .map_err(|e| format!("{e:?}"))?,
                    config.body.episode_id(),
                ),
                TaskProfile::MobileBox => PolicyWorker::spawn_mobile(
                    MobilePolicyClient::new(&policy.endpoint, timeout)
                        .map_err(|e| format!("{e:?}"))?,
                    config.body.episode_id(),
                ),
            }
            .map_err(|e| e.to_string())?;
            Some(LivePolicyRuntime {
                worker,
                profile,
                next_sequence: 1,
                pending: None,
                submitted_chunks: 0,
                max_calls: policy.max_calls,
                next_boundary_tick: 0,
                prefetch_after_ticks: policy.prefetch_after_ticks,
            })
        }
        (_, None) => None,
        _ => return Err("live policy requires the matched task owner".into()),
    };
    let assets = validate_render_asset_root(&default_asset_root())?;
    let scene = StationScene::load(&assets)?;
    let model = G1VisualModel::load(visual_path, visual_sha256)?;
    let source_pbr_body = model.source_material_mesh_count() == model.mesh_count();
    let task_model = match (config.task_objects(), task_visual) {
        (Some(objects), Some((path, sha))) => Some(G1TaskVisualModel::load(
            path,
            sha,
            &objects.definition_sha256,
        )?),
        (None, None) => None,
        _ => return Err("task physics and task visuals must be configured together".into()),
    };
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::create_dir(&options.output)
        .map_err(|error| format!("capture output must be new: {error}"))?;
    let output = options.output.clone();
    let episode_id = config.episode();
    let outcome = CaptureOutcome(Arc::new(Mutex::new(G1CaptureReceipt::initial(
        options.ticks,
        &config,
        visual_sha256,
    ))));
    outcome.0.lock().unwrap().task_visual_sha256 =
        task_model.as_ref().map(|m| m.file_sha256.clone());
    outcome.0.lock().unwrap().body_bound_material_mesh_count = model.source_material_mesh_count();
    outcome.0.lock().unwrap().camera_exposure_ev100 = exposure_ev100;
    outcome.0.lock().unwrap().diagnostic_directional_shadow_maps = directional_shadow_maps;
    outcome.0.lock().unwrap().render_present_mode = if diagnostic_vsync {
        "auto_vsync"
    } else {
        "auto_no_vsync"
    };
    outcome.0.lock().unwrap().diagnostic_render_hz = diagnostic_render_hz;
    outcome.0.lock().unwrap().camera_ambient_brightness = ambient_brightness;
    outcome.0.lock().unwrap().camera_directional_illuminance = directional_illuminance;
    outcome.0.lock().unwrap().diagnostic_aces_fitted = diagnostic_aces_fitted;
    outcome.0.lock().unwrap().source_rect_lighting = source_rect_lighting
        .as_ref()
        .map(|light| light.receipt.clone());
    if source_rect_lighting.is_some() {
        outcome.0.lock().unwrap().camera_directional_illuminance = 0.;
    }
    if render_only_environment_translation.is_some() {
        let mut receipt = outcome.0.lock().unwrap();
        receipt.scope = "external_measured_pose_initialized_zero_tick_renderer_comparison";
        receipt.external_pose_render_only = true;
    }
    if live_policy.is_some() {
        let mut receipt = outcome.0.lock().unwrap();
        receipt.scope = "native_live_rgb_to_matched_policy_to_same_owner_diagnostic";
        receipt.diagnostic_prefetch_after_ticks = live_policy
            .as_ref()
            .and_then(|live| live.prefetch_after_ticks);
        receipt.pauses_for_camera_and_policy = receipt.diagnostic_prefetch_after_ticks.is_none();
    }
    if mobile_carry {
        outcome.0.lock().unwrap().scope =
            "native_mobile_bounded_carry_stage_diagnostic_not_qualified";
    }
    if assisted_carry {
        outcome.0.lock().unwrap().scope = if scan_only {
            "native_actual_rgb_vla_grasp_then_public_map_turn_marked_observation_not_qualified"
        } else {
            "native_actual_rgb_vla_grasp_then_explicit_classical_carry_diagnostic_not_qualified"
        };
    }
    if interactive {
        outcome.0.lock().unwrap().scope = "native_local_qwen_interactive_lab_no_qualified_executor";
    }
    let worker = config.spawn(
        predictive_limit_diagnostic,
        diagnostic_constraint_sweeps,
        assisted_carry,
    )?;
    let owner_evidence = match &worker {
        CaptureWorker::Task(worker) => Some(OwnerEvidence {
            trace: worker.subscribe_steps(512).map_err(|e| e.to_string())?,
            output: BufWriter::new(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output.join("owner_steps.jsonl"))
                    .map_err(|e| e.to_string())?,
            ),
            records: 0,
            first_boundary_wall_ms: None,
        }),
        CaptureWorker::Mobile(_) | CaptureWorker::Static { .. } => None,
        #[cfg(feature = "g1_constraint_diagnostic")]
        CaptureWorker::AssistedMobile(_) => None,
    };
    #[cfg(feature = "g1_constraint_diagnostic")]
    let assist_evidence = match &worker {
        CaptureWorker::AssistedMobile(worker) => Some(OwnerEvidence {
            trace: worker.subscribe_steps(512).map_err(|e| e.to_string())?,
            output: BufWriter::new(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output.join("owner_steps.jsonl"))
                    .map_err(|e| e.to_string())?,
            ),
            records: 0,
            first_boundary_wall_ms: None,
        }),
        _ => None,
    };
    if predictive_limit_diagnostic {
        outcome
            .0
            .lock()
            .unwrap()
            .factory_verified_predictive_limit_joints = 43;
    }
    outcome
        .0
        .lock()
        .unwrap()
        .factory_verified_diagnostic_constraint_sweeps = diagnostic_constraint_sweeps;
    let mut app = App::new();
    // Scene supplies the existing enamel configuration only. Its meshes, props,
    // fixtures and camera shots are not spawned in this floor diagnostic.
    app.insert_resource(scene)
        .insert_resource(DiagnosticPhotometry {
            exposure_ev100,
            aces_fitted: diagnostic_aces_fitted,
        })
        .insert_resource(DiagnosticDirectionalShadows(directional_shadow_maps))
        .insert_resource(DiagnosticIllumination {
            ambient_brightness,
            directional_illuminance,
            source_rect_lighting,
        })
        .insert_resource(model)
        .insert_resource(outcome.clone())
        .init_resource::<G1TaskVisualInput>()
        .init_resource::<G1BackgroundVisualInput>()
        .insert_resource(CaptureRuntime {
            worker,
            options,
            episode_id,
            started: Instant::now(),
            render_frames: 0,
            latest: None,
            command_submitted: false,
            requested: false,
            ego_saved: false,
            main_saved: Arc::new(Mutex::new(Ok(false))),
            live_policy,
            owner_evidence,
            #[cfg(feature = "g1_constraint_diagnostic")]
            assist_evidence,
            mobile_assist: mobile_stage.map(|c| {
                MobileAssistCaptureRuntime::new(
                    c,
                    raising_view,
                    memory_view,
                    restored_view,
                    auxiliary_view,
                )
            }),
            interactive,
        })
        .insert_resource(bevy::winit::WinitSettings::continuous())
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: assets.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "G1 native camera diagnostic — unqualified floor candidate".into(),
                        resolution: (1920, 1080).into(),
                        present_mode: if diagnostic_vsync {
                            bevy::window::PresentMode::AutoVsync
                        } else {
                            bevy::window::PresentMode::AutoNoVsync
                        },
                        ..default()
                    }),
                    ..default()
                })
                .set(WinitPlugin {
                    run_on_any_thread: true,
                    ..default()
                }),
        )
        .add_plugins((G1VisualPlugin, G1CameraPlugin, G1ObservationPlugin))
        .add_systems(
            Startup,
            (reserve_source_light_clusters, setup_floor_scene).chain(),
        )
        .add_systems(
            Update,
            (configure_diagnostic_exposure, drive_capture)
                .chain()
                .in_set(CaptureSystems::OwnerSnapshot),
        );
    if let Some(model) = task_model {
        app.insert_resource(model).add_plugins(G1TaskVisualPlugin);
    }
    if let Some(model) = fiducial_model {
        outcome.0.lock().unwrap().disclosed_fiducials = Some(model.receipt.clone());
        app.insert_resource(model).add_plugins(G1TaskFiducialPlugin);
    }
    if let Some(model) = background_model {
        app.insert_resource(model)
            .add_plugins(G1BackgroundVisualPlugin);
    }
    let lab_outcome = task_lab
        .map(|configuration| {
            super::g1_task_lab::install(
                &mut app,
                configuration,
                episode_id,
                task_profile.ok_or("task lab requires task profile")?,
                &output,
            )
        })
        .transpose()?;
    if diagnostic_render_hz.is_some() {
        app.insert_resource(DiagnosticRenderPace {
            last_frame: Instant::now(),
        })
        .add_systems(Last, pace_diagnostic_render);
    }
    if episode_id > 0 {
        app.world()
            .resource::<G1CameraPort>()
            .reset(episode_id)
            .map_err(|error| format!("camera episode: {error}"))?;
    }
    if source_pbr_body {
        rendering_minigame::install_pbr_render_health(&mut app)?;
    } else {
        install_station_render_health(&mut app)?;
    }
    let mut exit = app.run();
    if let Some(outcome) = lab_outcome {
        let mut lab = outcome.0.lock().map_err(|_| "lab outcome poisoned")?;
        lab.window_closed = true;
        if lab.smoke_required && !lab.smoke_succeeded {
            lab.failure.get_or_insert(
                "window closed before the required decision/reset smoke completed".into(),
            );
            exit = AppExit::error();
        }
        fs::write(
            output.join("task_lab_receipt.json"),
            serde_json::to_vec_pretty(&*lab).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    }
    let mut receipt = outcome
        .0
        .lock()
        .map_err(|_| "capture result poisoned")?
        .clone();
    if !matches!(exit, AppExit::Success) && receipt.failure_reason.is_none() {
        receipt.failure_reason = Some("render application exited with error".into());
    }
    fs::write(
        output.join("capture_receipt.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if (!receipt.capture_succeeded && !interactive)
        || (interactive && !matches!(exit, AppExit::Success))
    {
        return Err(receipt
            .failure_reason
            .unwrap_or_else(|| "native camera capture did not finish".into()));
    }
    Ok(receipt)
}

#[cfg(test)]
mod budget_tests {
    use super::CaptureMode;
    #[test]
    fn both_startup_guards_share_the_explicit_stage_budget() {
        for (mode, budget) in [
            (CaptureMode::MobileAssist, 2050),
            (CaptureMode::MobileScan, 1050),
            (CaptureMode::MobileTargetView, 1300),
            (CaptureMode::MobileTargetApproach, 3150),
            (CaptureMode::MobileTargetRaiseView, 3150),
            (CaptureMode::MobileTargetMemoryView, 3150),
            (CaptureMode::MobileTargetRestoredView, 3150),
            (CaptureMode::MobileAuxiliaryView, 1050),
        ] {
            assert_eq!(mode.assisted_tick_limit(), Some(budget));
        }
        for mode in [
            CaptureMode::Camera,
            CaptureMode::TaskLab,
            CaptureMode::MobileCarry,
        ] {
            assert_eq!(mode.assisted_tick_limit(), None);
        }
    }
}

#[derive(Resource)]
struct DiagnosticRenderPace {
    last_frame: Instant,
}

/// Rendering may wait; the independent physical owner never calls this system.
fn pace_diagnostic_render(mut pace: ResMut<DiagnosticRenderPace>) {
    let period = Duration::from_nanos(1_000_000_000 / 60);
    if let Some(remaining) = period.checked_sub(pace.last_frame.elapsed()) {
        std::thread::sleep(remaining);
    }
    pace.last_frame = Instant::now();
}

fn setup_floor_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    background: Option<Res<G1BackgroundVisualModel>>,
    shadows: Res<DiagnosticDirectionalShadows>,
    illumination: Res<DiagnosticIllumination>,
) {
    commands.spawn((
        Name::new("g1_runner_actual_floor"),
        Mesh3d(meshes.add(Cuboid::new(40., 0.5, 40.))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.48, 0.5),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0., -0.25, 0.),
        if background.is_some() {
            Visibility::Hidden
        } else {
            Visibility::Visible
        },
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: illumination.ambient_brightness,
        ..default()
    });
    if let Some(source) = &illumination.source_rect_lighting {
        source.spawn(&mut commands, &mut images);
    } else {
        commands.spawn((
            DirectionalLight {
                illuminance: illumination.directional_illuminance,
                shadow_maps_enabled: shadows.0,
                ..default()
            },
            Transform::from_xyz(3., 5., 2.).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }
    commands.spawn((
        Camera3d::default(),
        Msaa::Sample8,
        Tonemapping::None,
        Transform::from_xyz(2.1, 1.45, 2.2).looking_at(Vec3::new(0., 0.7, 0.), Vec3::Y),
    ));
}

fn observation(snapshot: &CaptureSnapshot) -> Result<Option<Arc<G1CameraNativeState>>, String> {
    let (Some(frame), Some(measurement)) = (&snapshot.frame, &snapshot.measurement) else {
        if snapshot.frame.is_some() || snapshot.measurement.is_some() {
            return Err("worker published a partial native frame/measurement pair".into());
        }
        return Ok(None);
    };
    if snapshot.episode_id != frame.episode_id
        || measurement.episode_id != frame.episode_id
        || measurement.source_tick != frame.source_tick
        || measurement.sim_time_ns != (frame.sim_time * 1_000_000_000.0).round() as u64
    {
        return Err("worker frame and native measurement stamps disagree".into());
    }
    let native = Arc::new(G1CameraNativeState {
        body_frame: frame.as_ref().clone(),
        measured_joints: G1CameraJointState {
            positions: measurement.joint_positions.clone(),
            velocities: measurement.joint_velocities.clone(),
            root_rotation_wxyz: measurement.root_rotation_wxyz,
            root_angular_velocity_body: measurement.root_angular_velocity_body,
            root_velocity_source: measurement.root_velocity_source,
        },
    });
    native.validate()?;
    Ok(Some(native))
}

/// Common real RGB/self-state preparation. No admission clock is changed here.
fn submit_live_observation(
    live: &mut LivePolicyRuntime,
    frame: G1CapturedRgb,
    output: &Path,
) -> Result<(), String> {
    if frame.stamp.mount_profile != rendering_minigame::g1_camera::G1CameraMountProfile::ArenaEgo {
        return Err("original learned policy requires its unchanged Arena ego camera".into());
    }
    let native = frame
        .stamp
        .native_state
        .as_ref()
        .ok_or("live RGB lacks measured self state")?;
    native.validate()?;
    let names = robot_minigame::g1::contract::JOINT_NAMES.map(str::to_owned);
    let observation = PolicyObservation {
        stamp: ObservationStamp {
            episode_id: frame.stamp.episode_id,
            frame_id: frame.stamp.capture_sequence,
            sim_time_ns: frame.stamp.sim_time_ns,
            captured_at_unix_ms: frame.stamp.captured_at_unix_ms,
        },
        measured_joint_positions_rad: map_measured_joints(
            &names,
            &native.measured_joints.positions,
        )
        .map_err(|e| format!("live measured joints: {e:?}"))?,
        camera_width: frame.width,
        camera_height: frame.height,
        camera_rgb: frame.rgb,
    };
    observation
        .validate()
        .map_err(|e| format!("live RGB/self state: {e:?}"))?;
    let camera = task_minigame::decision::CameraRgb::from_rgb(
        "native_ego",
        observation.camera_width,
        observation.camera_height,
        observation.camera_rgb.clone(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        output.join(format!("live_ego_{:04}.png", live.next_sequence)),
        camera.png(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        output.join(format!("live_stamp_{:04}.json", live.next_sequence)),
        serde_json::to_vec(&frame.stamp).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let stamp = observation.stamp;
    live.worker
        .submit_latest(PolicyInferenceRequest {
            profile: live.profile,
            sequence_id: live.next_sequence,
            observation,
        })
        .map_err(|e| format!("live policy submission: {e:?}"))?;
    live.pending = Some(stamp);
    live.next_sequence += 1;
    Ok(())
}

fn drive_bounded_task(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    if runtime.mobile_assist.is_some() {
        let grasp_done = runtime
            .live_policy
            .as_ref()
            .is_some_and(|live| live.submitted_chunks == 4)
            && runtime
                .latest
                .as_ref()
                .is_some_and(|latest| latest.timing.episode_integrations >= 200);
        if grasp_done {
            drive_mobile_assist(runtime, outcome, port)
        } else {
            drive_live_policy(runtime, outcome, port)?;
            Ok(false)
        }
    } else {
        drive_live_policy(runtime, outcome, port)
    }
}

#[cfg(not(feature = "g1_constraint_diagnostic"))]
fn drive_mobile_assist(
    _: &mut CaptureRuntime,
    _: &CaptureOutcome,
    _: &G1CameraPort,
) -> Result<bool, String> {
    Err("mobile assist requires its explicit build feature".into())
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_mobile_assist(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::{MobileAssistCommand, MobileAssistExecution},
        mobile_navigation::{MobileCarryGoal, MobileScanGoal},
        worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("missing assisted owner snapshot")?
        .clone();
    if latest.phase == G1WorkerPhase::Failed {
        return Err(latest
            .reason
            .clone()
            .unwrap_or("assisted owner failed".into()));
    }
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("missing bounded assist configuration")?;
    if assist.completed {
        return Ok(true);
    }
    if assist.submitted {
        let scan_complete = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref()
            .is_some_and(|step| matches!(&step.execution, MobileAssistExecution::ClassicalScan { navigation, .. } if navigation.completed));
        if assist.auxiliary_view && scan_complete && !assist.auxiliary_view_completed {
            return drive_auxiliary_marker_view(runtime, outcome, port);
        }
        if assist.view_with_lowering && !assist.lower_submitted && scan_complete {
            return drive_lower_view(runtime, outcome, port);
        }
        let lower_complete = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref()
            .is_some_and(|step| matches!(&step.execution, MobileAssistExecution::ClassicalLower { lowering, .. } if lowering.completed));
        if assist.visual_approach && !assist.visual_goal_submitted && lower_complete {
            return drive_visual_approach(runtime, outcome, port);
        }
        let carry_complete = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref()
            .is_some_and(|step| matches!(&step.execution, MobileAssistExecution::ClassicalCarry { navigation, .. } if navigation.completed));
        if assist.raising_view
            && assist.visual_goal_submitted
            && !assist.raise_submitted
            && carry_complete
        {
            return drive_visual_raise_view(runtime, outcome, port);
        }
        let raise_complete = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref()
            .is_some_and(|step| matches!(&step.execution, MobileAssistExecution::ClassicalRaise { raising, .. } if raising.completed));
        if assist.restored_view && assist.raise_submitted && !assist.restored_view_completed {
            return drive_restored_reobserve_view(runtime, outcome, port);
        }
        if assist.memory_view
            && assist.raise_submitted
            && !assist.memory_completed
            && raise_complete
        {
            return drive_visual_memory_view(runtime, outcome, port);
        }
        let completed = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref().is_some_and(|step| {
            if assist.auxiliary_view {
                assist.auxiliary_view_completed
                    && matches!(&step.execution, MobileAssistExecution::ClassicalScan { navigation, .. } if navigation.completed)
            } else if assist.restored_view {
                assist.restored_view_completed
                    && matches!(&step.execution, MobileAssistExecution::ClassicalReobserve { navigation, .. } if navigation.completed)
            } else if assist.raising_view {
                assist.raise_submitted && (!assist.memory_view || assist.memory_completed)
                    && matches!(&step.execution, MobileAssistExecution::ClassicalRaise { raising, .. } if raising.completed)
            } else if assist.visual_approach {
                assist.visual_goal_submitted && matches!(&step.execution, MobileAssistExecution::ClassicalCarry { navigation, .. } if navigation.completed)
            } else if assist.view_with_lowering {
                matches!(&step.execution, MobileAssistExecution::ClassicalLower { lowering, .. } if lowering.completed)
            } else {
                matches!(&step.execution,MobileAssistExecution::ClassicalCarry {navigation,..} | MobileAssistExecution::ClassicalScan {navigation,..} if navigation.completed)
            }
        });
        if completed {
            runtime.mobile_assist.as_mut().unwrap().completed = true;
            let mut receipt = outcome.0.lock().unwrap();
            if let Some(handoff) = &mut receipt.mobile_assist_handoff {
                handoff["completed_at_actual_tick"] = latest.timing.episode_integrations.into();
                handoff["task_qualified"] = false.into();
            }
        }
        return Ok(completed);
    }
    if latest.phase != G1WorkerPhase::Paused || latest.timing.episode_integrations != 200 {
        return Ok(false);
    }
    if !runtime.requested {
        port.request()?;
        runtime.requested = true;
        return Ok(false);
    }
    let Some(frame) = port.take() else {
        return Ok(false);
    };
    let frame = frame?;
    if frame.stamp.source != CameraPoseSource::PhysicsBody
        || frame.stamp.mount_profile
            != if runtime
                .mobile_assist
                .as_ref()
                .is_some_and(|a| a.auxiliary_view)
            {
                G1CameraMountProfile::AuxiliaryGripOverview
            } else {
                G1CameraMountProfile::ArenaEgo
            }
        || frame.stamp.episode_id != runtime.episode_id
        || frame.stamp.source_ticks != [200; 2]
        || frame.stamp.sim_time_ns != 4_000_000_000
    {
        return Err("assisted handoff RGB is foreign to the exact native grasp boundary".into());
    }
    frame
        .stamp
        .native_state
        .as_ref()
        .ok_or("handoff RGB lacks native self state")?
        .validate()?;
    let camera = task_minigame::decision::CameraRgb::from_rgb(
        "native_ego",
        frame.width,
        frame.height,
        frame.rgb,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        runtime.options.output.join("carry_handoff_ego.png"),
        camera.png(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        runtime.options.output.join("carry_handoff_stamp.json"),
        serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let observation = ObservationStamp {
        episode_id: runtime.episode_id,
        frame_id: frame.stamp.capture_sequence,
        sim_time_ns: frame.stamp.sim_time_ns,
        captured_at_unix_ms: frame.stamp.captured_at_unix_ms,
    };
    let (command, goal, origin, limit_ns, wall_seconds) = match &assist.configuration {
        MobileAssistStage::Carry(config) => {
            let goal = MobileCarryGoal {
                observation,
                heading_yaw_source_rad: config.heading_yaw_source_rad,
                relative_distance_m: config.relative_distance_m,
            };
            (
                MobileAssistCommand::ClassicalCarry(goal.clone()),
                serde_json::to_value(goal).map_err(|e| e.to_string())?,
                "explicit_manual_clear_aisle_diagnostic",
                41_000_000_000,
                40,
            )
        }
        MobileAssistStage::Scan(config) => {
            let goal = MobileScanGoal {
                observation,
                heading_yaw_source_rad: config.heading_yaw_source_rad,
            };
            (
                MobileAssistCommand::ClassicalScan(goal.clone()),
                serde_json::to_value(goal).map_err(|e| e.to_string())?,
                "explicit_public_map_search_heading_only_not_target_pose",
                21_000_000_000,
                20,
            )
        }
    };
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("handoff lost its unique typed assisted owner".into());
    };
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: limit_ns,
            valid_until_wall: Instant::now() + Duration::from_secs(wall_seconds),
            command,
        })
        .map_err(|e| e.to_string())?;
    outcome.0.lock().unwrap().mobile_assist_handoff = Some(serde_json::json!({
        "goal":goal,"goal_origin":origin,
        "original_vla_grasp_chunks":4,"actual_rgb_handoff":true,"traditional_grip_navigation":true,
        "object_truth_in_command":false,"task_qualified":false,"autonomous_goal_selection":false,
    }));
    runtime.mobile_assist.as_mut().unwrap().submitted = true;
    runtime.requested = false;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_auxiliary_marker_view(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("auxiliary view lacks owner")?;
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("auxiliary view stage absent")?;
    if latest.phase != G1WorkerPhase::Paused || assist.auxiliary_view_completed {
        return Err("auxiliary view requires its completed standing native scan".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        if job.observation.episode_id != runtime.episode_id
            || job.observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
            || reply["camera_mount_profile"] != "auxiliary_grip_overview"
            || reply["marker_layout_profile"] != "auxiliary_grip_targets"
        {
            return Err("auxiliary view has a foreign camera/label/native boundary".into());
        }
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("auxiliary view provenance absent")?;
        handoff["actual_auxiliary_view"] = serde_json::json!({
            "actual_current_localization":reply,"same_owner_boundary_tick":latest.timing.episode_integrations,
            "localization_wall_ms":job.started.elapsed().as_secs_f64()*1000.,"physics_paused_during_localization":true,
            "original_learned_camera_unchanged":true,"static_target_memory_used":false,
            "lower_raise_restore_or_postcarry_view_turn_executed":false,"proposal_executed":false,
            "object_truth_in_command":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.auxiliary_view_completed = true;
        runtime.requested = false;
        return Ok(false);
    }
    start_marker_job(runtime, port, "visual_auxiliary_view")?;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_visual_raise_view(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::MobileAssistCommand, mobile_raise::MobileRaiseGoal, worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("raise view lacks owner")?
        .clone();
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("raise view stage absent")?;
    if latest.phase != G1WorkerPhase::Paused || assist.raise_submitted {
        return Err("raise view requires completed stationary coarse carry".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        let observation = job.observation;
        if observation.episode_id != runtime.episode_id
            || observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
        {
            return Err("clearance belongs to old episode/boundary".into());
        }
        let goal: MobileRaiseGoal =
            serde_json::from_value(reply["clearance_proposal"]["raise_goal"].clone())
                .map_err(|e| format!("current RGB has no bounded raise goal: {e}"))?;
        goal.validate().map_err(|e| e.to_string())?;
        if goal.observation != observation {
            return Err("raise goal has foreign camera identity".into());
        }
        let target_memory = if assist.memory_view {
            Some(MarkerTargetMemory::from_localization(&reply)?)
        } else {
            None
        };
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("raise view lost sole native owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: observation.sim_time_ns + 10_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(12),
                command: MobileAssistCommand::ClassicalRaise(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        let wall_ms = job.started.elapsed().as_secs_f64() * 1000.;
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("raise provenance absent")?;
        handoff["visual_raise_view"] = serde_json::json!({
            "actual_localization":reply,"executed_raise_goal":goal,
            "localization_wall_ms":wall_ms,"physics_paused_during_localization":true,
            "same_owner_boundary_tick":latest.timing.episode_integrations,
            "object_truth_in_command":false,"synthetic_camera_stamp":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.raise_submitted = true;
        assist.target_memory = target_memory;
        runtime.requested = false;
        return Ok(false);
    }
    start_marker_job(runtime, port, "visual_raise_view")?;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_lower_view(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::MobileAssistCommand, mobile_lowering::MobileLowerGoal, worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("lower view lacks current owner")?
        .clone();
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("lower view lacks admitted stage")?;
    let MobileAssistStage::Scan(config) = &assist.configuration else {
        return Err("lower view requires its public-map scan predecessor".into());
    };
    let lowering = config
        .lowering
        .clone()
        .ok_or("lower view lacks bounded pose configuration")?;
    if assist.lower_submitted {
        return Err("lower view handoff cannot repeat".into());
    }
    if !runtime.requested {
        port.request()?;
        runtime.requested = true;
        return Ok(false);
    }
    let Some(frame) = port.take() else {
        return Ok(false);
    };
    let frame = frame?;
    let tick = latest.timing.episode_integrations;
    if latest.phase != G1WorkerPhase::Paused
        || frame.stamp.source != CameraPoseSource::PhysicsBody
        || frame.stamp.episode_id != runtime.episode_id
        || frame.stamp.source_ticks != [tick; 2]
        || frame.stamp.sim_time_ns != tick * 20_000_000
    {
        return Err("lower view RGB is not from the exact completed stationary scan".into());
    }
    frame
        .stamp
        .native_state
        .as_ref()
        .ok_or("lower view lacks original self sensors")?
        .validate()?;
    let rgb = task_minigame::decision::CameraRgb::from_rgb(
        "native_ego",
        frame.width,
        frame.height,
        frame.rgb,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        runtime.options.output.join("lower_handoff_ego.png"),
        rgb.png(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        runtime.options.output.join("lower_handoff_stamp.json"),
        serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let goal = MobileLowerGoal {
        observation: ObservationStamp {
            episode_id: runtime.episode_id,
            frame_id: frame.stamp.capture_sequence,
            sim_time_ns: frame.stamp.sim_time_ns,
            captured_at_unix_ms: frame.stamp.captured_at_unix_ms,
        },
        distance_m: lowering.distance_m,
        duration_ticks: lowering.duration_ticks,
    };
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("lower view lost the sole typed native owner".into());
    };
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: frame.stamp.sim_time_ns + 7_000_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(8),
            command: MobileAssistCommand::ClassicalLower(goal.clone()),
        })
        .map_err(|e| e.to_string())?;
    let mut receipt = outcome.0.lock().unwrap();
    let handoff = receipt
        .mobile_assist_handoff
        .as_mut()
        .ok_or("lower view lost provenance")?;
    handoff["lowering_goal"] = serde_json::to_value(goal).map_err(|e| e.to_string())?;
    handoff["lowering_goal_origin"] = "finite_self_state_visibility_pose_not_bin_height".into();
    handoff["scan_completed_at_actual_tick"] = tick.into();
    handoff["object_truth_in_command"] = false.into();
    runtime.mobile_assist.as_mut().unwrap().lower_submitted = true;
    runtime.requested = false;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_restored_reobserve_view(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::{MobileAssistCommand, MobileAssistExecution},
        mobile_navigation::MobileScanGoal,
        mobile_restore::MobileRestoreGoal,
        worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("restored view lacks owner")?
        .clone();
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("restored view stage absent")?;
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    if !assist.restore_submitted {
        if !latest.assist_step.as_ref().is_some_and(|s| {
            matches!(&s.execution,
            MobileAssistExecution::ClassicalRaise { raising, .. } if raising.completed)
        }) {
            return Err("restore requires completed native raised carry".into());
        }
        let Some((observation, _)) =
            capture_current_marker_frame(runtime, port, "restore_posture")?
        else {
            return Ok(false);
        };
        let goal = MobileRestoreGoal {
            observation,
            duration_ticks: 200,
        };
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("restore lost sole native owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: observation.sim_time_ns + 8_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(10),
                command: MobileAssistCommand::ClassicalRestore(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("restore provenance absent")?;
        handoff["transport_posture_restore"] = serde_json::json!({
            "executed_goal":goal,"same_owner_boundary_tick":latest.timing.episode_integrations,
            "original_episode_command_and_self_fk_only":true,"synthetic_camera_stamp":false,
            "object_truth_in_command":false,"task_qualified":false,
        });
        runtime.mobile_assist.as_mut().unwrap().restore_submitted = true;
        runtime.requested = false;
        return Ok(false);
    }
    if !assist.reobserve_submitted {
        if !latest.assist_step.as_ref().is_some_and(|s| {
            matches!(&s.execution,
            MobileAssistExecution::ClassicalRestore { restoring, .. } if restoring.completed)
        }) {
            return Err("view turn requires completed native transport restoration".into());
        }
        let Some(job) = &assist.vision_job else {
            start_marker_job(runtime, port, "restored_view_turn")?;
            return Ok(false);
        };
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        // An unseen or visually distant box leaves the owner paused. A view
        // turn is never admitted solely because the posture command finished.
        let reply = reply?;
        let observation = job.observation;
        if observation.episode_id != runtime.episode_id
            || observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
        {
            return Err("grip visibility belongs to an old episode/boundary".into());
        }
        let [w, x, y, z] = assist
            .restored_view_rotation
            .ok_or("current view self rotation absent")?;
        let yaw = (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z));
        // One disclosed, fixed diagnostic view action; this is not model navigation.
        let heading = (yaw - 0.33 + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        let goal = MobileScanGoal {
            observation,
            heading_yaw_source_rad: heading,
        };
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("view turn lost sole native owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: observation.sim_time_ns + 18_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(20),
                command: MobileAssistCommand::ClassicalReobserve(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("view turn provenance absent")?;
        handoff["restored_view_turn"] = serde_json::json!({
            "executed_goal":goal,"current_self_yaw_rad":yaw,"fixed_view_offset_rad":-0.33,
            "actual_current_box_visibility":reply,
            "same_owner_boundary_tick":latest.timing.episode_integrations,
            "traditional_fixed_diagnostic_view_action":true,"synthetic_camera_stamp":false,
            "object_truth_in_command":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.reobserve_submitted = true;
        runtime.requested = false;
        return Ok(false);
    }
    if !latest.assist_step.as_ref().is_some_and(|s| {
        matches!(&s.execution,
        MobileAssistExecution::ClassicalReobserve { navigation, .. } if navigation.completed)
    }) {
        return Err("restored localization requires completed standing view turn".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        if job.observation.episode_id != runtime.episode_id
            || job.observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
        {
            return Err("restored localization belongs to an old episode/boundary".into());
        }
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("restored localization provenance absent")?;
        handoff["actual_restored_view"] = serde_json::json!({
            "actual_current_localization":reply,"localization_wall_ms":job.started.elapsed().as_secs_f64()*1000.,
            "same_owner_boundary_tick":latest.timing.episode_integrations,"physics_paused_during_localization":true,
            "static_target_memory_used":false,"proposal_executed":false,"object_truth_in_command":false,
            "synthetic_camera_stamp":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.restored_view_completed = true;
        runtime.requested = false;
        return Ok(false);
    }
    start_marker_job(runtime, port, "visual_restored_view")?;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_visual_memory_view(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime.latest.as_ref().ok_or("memory view lacks owner")?;
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("memory view stage absent")?;
    if latest.phase != G1WorkerPhase::Paused || assist.memory_completed {
        return Err("target memory view requires its completed native raise boundary".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        if job.observation.episode_id != runtime.episode_id
            || job.observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
        {
            return Err("memory observation belongs to old episode/boundary".into());
        }
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("memory provenance absent")?;
        handoff["visual_target_memory"] = serde_json::json!({
            "actual_current_localization":reply,"localization_wall_ms":job.started.elapsed().as_secs_f64()*1000.,
            "same_owner_boundary_tick":latest.timing.episode_integrations,"physics_paused_during_localization":true,
            "static_target_assumption":true,"object_truth_in_command":false,"proposal_executed":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.memory_completed = true;
        runtime.requested = false;
        return Ok(false);
    }
    start_marker_job(runtime, port, "visual_memory_view")?;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_visual_approach(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::MobileAssistCommand, mobile_navigation::MobileCarryGoal,
        worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("visual approach lacks owner")?
        .clone();
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("visual approach stage absent")?;
    if latest.phase != G1WorkerPhase::Paused || assist.visual_goal_submitted {
        return Err("visual approach requires its completed stationary lower boundary".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        let observation = job.observation;
        let wall_ms = job.started.elapsed().as_secs_f64() * 1000.;
        if observation.episode_id != runtime.episode_id
            || observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
        {
            return Err("visual goal belongs to an old episode/boundary".into());
        }
        let proposal = &reply["navigation_proposal"];
        // Reobserve from a clear intermediate waypoint before final placement.
        // This margin is explicitly traditional geometry, not a VLA action.
        let distance = proposal["relative_distance_m"]
            .as_f64()
            .ok_or("missing approach distance")?
            - 0.65;
        if !(0.1..=1.85).contains(&distance) {
            return Err("visual coarse waypoint outside bounded range".into());
        }
        let goal = MobileCarryGoal {
            observation,
            heading_yaw_source_rad: proposal["heading_yaw_source_rad"]
                .as_f64()
                .ok_or("missing approach heading")? as f32,
            relative_distance_m: distance as f32,
        };
        goal.validate().map_err(|e| e.to_string())?;
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("visual approach lost sole native owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: observation.sim_time_ns + 41_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(42),
                command: MobileAssistCommand::ClassicalCarry(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("visual approach provenance absent")?;
        handoff["visual_approach"] = serde_json::json!({
            "actual_localization":reply,"executed_coarse_goal":goal,"reobservation_margin_m":0.65,
            "localization_wall_ms":wall_ms,"physics_paused_during_localization":true,
            "same_owner_boundary_tick":latest.timing.episode_integrations,
            "object_truth_in_command":false,"qwen_target_selection":false,"task_qualified":false,
        });
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.visual_goal_submitted = true;
        runtime.requested = false;
        return Ok(false);
    }
    start_marker_job(runtime, port, "visual_approach")?;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn capture_current_marker_frame(
    runtime: &mut CaptureRuntime,
    port: &G1CameraPort,
    stage_directory: &str,
) -> Result<Option<(ObservationStamp, [f32; 4])>, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("current RGB lacks owner")?
        .clone();
    if latest.phase != G1WorkerPhase::Paused {
        return Err("current RGB requires a paused completed native boundary".into());
    }
    if !runtime.requested {
        port.request()?;
        runtime.requested = true;
        return Ok(None);
    }
    let Some(frame) = port.take() else {
        return Ok(None);
    };
    let frame = frame?;
    let tick = latest.timing.episode_integrations;
    if frame.stamp.source != CameraPoseSource::PhysicsBody
        || frame.stamp.mount_profile
            != if runtime
                .mobile_assist
                .as_ref()
                .is_some_and(|a| a.auxiliary_view)
            {
                G1CameraMountProfile::AuxiliaryGripOverview
            } else {
                G1CameraMountProfile::ArenaEgo
            }
        || frame.stamp.episode_id != runtime.episode_id
        || frame.stamp.source_ticks != [tick; 2]
        || frame.stamp.sim_time_ns != tick * 20_000_000
    {
        return Err("marker RGB mismatches completed native skill boundary".into());
    }
    frame
        .stamp
        .native_state
        .as_ref()
        .ok_or("visual approach self state absent")?
        .validate()?;
    let input = marker_observation(&frame.stamp)?;
    let observation: ObservationStamp =
        serde_json::from_value(input["stamp"].clone()).map_err(|e| e.to_string())?;
    let directory = runtime.options.output.join(stage_directory);
    fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let rgb = task_minigame::decision::CameraRgb::from_rgb(
        "native_ego",
        frame.width,
        frame.height,
        frame.rgb,
    )
    .map_err(|e| e.to_string())?;
    fs::write(directory.join("ego.png"), rgb.png()).map_err(|e| e.to_string())?;
    fs::write(
        directory.join("observation.json"),
        serde_json::to_vec_pretty(&input).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        directory.join("audit_stamp.json"),
        serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let rotation = frame
        .stamp
        .native_state
        .as_ref()
        .unwrap()
        .measured_joints
        .root_rotation_wxyz;
    Ok(Some((observation, rotation)))
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn start_marker_job(
    runtime: &mut CaptureRuntime,
    port: &G1CameraPort,
    stage_directory: &str,
) -> Result<(), String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("marker job lacks native owner")?
        .clone();
    if latest.phase != G1WorkerPhase::Paused {
        return Err("marker job requires a stationary completed skill boundary".into());
    }
    let Some((observation, rotation)) =
        capture_current_marker_frame(runtime, port, stage_directory)?
    else {
        return Ok(());
    };
    let directory = runtime.options.output.join(stage_directory);
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("marker job stage absent")?;
    let MobileAssistStage::Scan(config) = &assist.configuration else {
        return Err("visual approach lacks scan profile".into());
    };
    let mut config = config
        .vision
        .clone()
        .ok_or("visual localization worker config absent")?;
    if stage_directory == "restored_view_turn" {
        config.task_geometry = None;
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.restored_view_rotation = Some(rotation);
        assist.vision_job = Some(MarkerVisionJob::start_box_view(
            config,
            directory,
            observation,
        )?);
        return Ok(());
    }
    let memory = if stage_directory == "visual_memory_view" {
        use simulation_minigame::g1::mobile_assist::MobileAssistExecution;
        let mut memory = assist
            .target_memory
            .clone()
            .ok_or("admitted recent visible target absent")?;
        let Some(step) = latest.assist_step.as_ref() else {
            return Err("raise self odometry absent".into());
        };
        let MobileAssistExecution::ClassicalRaise { raising, .. } = &step.execution else {
            return Err("memory may only use original self velocity during completed raise".into());
        };
        if !raising.completed {
            return Err("memory view requires completed raise".into());
        }
        memory.current_observation = observation;
        memory.self_velocity_displacement_source_m = raising.self_velocity_displacement_source_m;
        memory.validate(observation)?;
        Some(memory)
    } else {
        None
    };
    runtime.mobile_assist.as_mut().unwrap().vision_job = Some(if memory.is_some() {
        MarkerVisionJob::start_with_memory(config, directory, observation, memory)?
    } else {
        MarkerVisionJob::start(config, directory, observation)?
    });
    Ok(())
}

fn marker_observation(stamp: &G1CaptureStamp) -> Result<serde_json::Value, String> {
    let state = stamp
        .native_state
        .as_ref()
        .ok_or("marker image self state absent")?;
    state.validate()?;
    let mut observation = serde_json::json!({
        "schema":"g1_mobile_marker_observation_v1",
        "stamp":ObservationStamp {episode_id:stamp.episode_id,frame_id:stamp.capture_sequence,
            sim_time_ns:stamp.sim_time_ns,captured_at_unix_ms:stamp.captured_at_unix_ms},
        "camera":rendering_minigame::g1_camera::G1CameraCalibration::default(),
        "measured_joints":state.measured_joints,
    });
    if stamp.mount_profile == G1CameraMountProfile::AuxiliaryGripOverview {
        observation["camera_mount_profile"] = "auxiliary_grip_overview".into();
    }
    Ok(observation)
}

fn drive_live_policy(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    if runtime
        .live_policy
        .as_ref()
        .is_some_and(|live| live.prefetch_after_ticks.is_some())
    {
        return drive_prefetched_policy(runtime, outcome, port);
    }
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("missing live owner snapshot")?;
    let live = runtime
        .live_policy
        .as_mut()
        .ok_or("missing live policy worker")?;
    if latest.phase == G1WorkerPhase::Failed {
        return Err(latest
            .reason
            .clone()
            .unwrap_or("live physical owner failed".into()));
    }
    if let Some(reply) = live.worker.try_take_reply() {
        if live.pending.take() != Some(reply.observation)
            || reply.profile != live.profile
            || reply.sequence_id + 1 != live.next_sequence
            || latest.phase != G1WorkerPhase::Paused
            || latest.timing.episode_integrations != live.next_boundary_tick
        {
            return Err("live policy reply and paused physical boundary disagree".into());
        }
        let chunk = reply.result.map_err(|e| format!("live policy: {e:?}"))?;
        let horizon = chunk.frames.len() as u64;
        let end_tick = live
            .next_boundary_tick
            .checked_add(horizon)
            .ok_or("live tick overflow")?;
        fs::write(
            runtime
                .options
                .output
                .join(format!("live_reply_{:04}.json", reply.sequence_id)),
            serde_json::to_vec(&chunk).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        runtime
            .worker
            .submit_live_chunk(TimedArenaTaskCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: end_tick
                    .checked_mul(ARENA_ACTION_PERIOD_NS)
                    .ok_or("live time overflow")?,
                valid_until_wall: Instant::now()
                    + Duration::from_nanos(horizon * ARENA_ACTION_PERIOD_NS)
                    + Duration::from_secs(1),
                command: ArenaTaskCommand {
                    scheduled_start_sim_ns: None,
                    chunk: Arc::new(chunk),
                },
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        receipt.live_policy_successes += 1;
        receipt.live_action_chunks.push(serde_json::json!({
            "sequence_id": reply.sequence_id, "observation": reply.observation,
            "inference_ms": reply.elapsed.as_millis(), "start_tick": live.next_boundary_tick,
            "end_tick_exclusive": end_tick, "frame_count": horizon,
            "same_owner_world": true, "task_qualified": false,
        }));
        live.next_boundary_tick = end_tick;
        live.submitted_chunks += 1;
        return Ok(false);
    }
    if live.pending.is_some() || latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    if latest.timing.episode_integrations != live.next_boundary_tick {
        // Submission resumes on the next owner boundary. An unchanged paused
        // display snapshot can arrive first; it must not be treated as a reset.
        return Ok(false);
    }
    if !runtime.requested {
        if let Some(step) = &latest.task_step {
            if step.execution.admitted_chunks != u64::from(live.submitted_chunks)
                || step.execution.frame_index + 1 != profile_contract(live.profile).action_horizon
            {
                return Err(
                    "owner admission/frame counters disagree with completed live chunk".into(),
                );
            }
            fs::write(
                runtime
                    .options
                    .output
                    .join(format!("live_boundary_{:04}.json", live.submitted_chunks)),
                serde_json::to_vec(step.as_ref()).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        }
        if live.submitted_chunks == live.max_calls {
            return Ok(true);
        }
        port.request()?;
        runtime.requested = true;
        return Ok(false);
    }
    if live.submitted_chunks == live.max_calls {
        return Ok(true);
    }
    if let Some(frame) = port.take() {
        let frame = frame?;
        if frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.source_ticks != [live.next_boundary_tick; 2]
            || frame.stamp.sim_time_ns != live.next_boundary_tick * ARENA_ACTION_PERIOD_NS
        {
            return Err("live RGB was rendered from a different physical boundary".into());
        }
        submit_live_observation(live, frame, &runtime.options.output)?;
        runtime.requested = false;
        outcome.0.lock().unwrap().live_policy_inference_calls += 1;
    }
    Ok(false)
}

/// One pending model request/replacement while all current original frames run.
/// Initial loading is paused; after the first real step a missed replacement
/// fails explicitly instead of keeping an expired target or rebasing a chunk.
fn drive_prefetched_policy(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("missing prefetch owner snapshot")?;
    let live = runtime
        .live_policy
        .as_mut()
        .ok_or("missing prefetch model worker")?;
    let tick = latest.timing.episode_integrations;
    let horizon = profile_contract(live.profile).action_horizon as u64;
    if latest.phase == G1WorkerPhase::Failed {
        return Err(latest
            .reason
            .clone()
            .unwrap_or("prefetch physical owner failed".into()));
    }
    if tick > 0 && latest.phase == G1WorkerPhase::Paused {
        if live.submitted_chunks == live.max_calls && tick == u64::from(runtime.options.ticks) {
            let step = latest
                .task_step
                .as_ref()
                .ok_or("completed prefetch budget has no actual step")?;
            if step.execution.admitted_chunks != u64::from(live.max_calls)
                || step.execution.frame_index + 1 != horizon as usize
            {
                return Err(
                    "prefetch admitted/frame counters disagree with actual final boundary".into(),
                );
            }
            return Ok(true);
        }
        return Err("prefetch replacement deadline missed; owner explicitly paused without expired actuation".into());
    }
    if let Some(reply) = live.worker.try_take_reply() {
        let start_tick = live.next_boundary_tick;
        let initial = live.submitted_chunks == 0;
        if live.pending.take() != Some(reply.observation)
            || reply.profile != live.profile
            || reply.sequence_id + 1 != live.next_sequence
            || (initial && (tick != 0 || latest.phase != G1WorkerPhase::Paused))
            || (!initial && tick >= start_tick)
        {
            return Err(
                "prefetch reply is stale, foreign or missed its fixed start; no rebasing".into(),
            );
        }
        let chunk = reply
            .result
            .map_err(|e| format!("prefetch policy: {e:?}"))?;
        if chunk.frames.len() as u64 != horizon {
            return Err("prefetch changed original model horizon".into());
        }
        let end_tick = start_tick
            .checked_add(horizon)
            .ok_or("prefetch tick overflow")?;
        let CaptureWorker::Task(owner) = &runtime.worker else {
            return Err("prefetch lost its sole physical owner".into());
        };
        fs::write(
            runtime
                .options
                .output
                .join(format!("live_reply_{:04}.json", reply.sequence_id)),
            serde_json::to_vec(&chunk).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        owner
            .submit(TimedArenaTaskCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: end_tick
                    .checked_mul(ARENA_ACTION_PERIOD_NS)
                    .ok_or("prefetch time overflow")?,
                valid_until_wall: Instant::now()
                    + Duration::from_nanos((end_tick - tick) * ARENA_ACTION_PERIOD_NS)
                    + Duration::from_secs(1),
                command: ArenaTaskCommand {
                    chunk: Arc::new(chunk),
                    scheduled_start_sim_ns: Some(start_tick * ARENA_ACTION_PERIOD_NS),
                },
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        receipt.live_policy_successes += 1;
        receipt.live_action_chunks.push(serde_json::json!({
            "sequence_id": reply.sequence_id, "observation": reply.observation,
            "inference_ms": reply.elapsed.as_millis(), "start_tick": start_tick,
            "end_tick_exclusive": end_tick, "frame_count": horizon,
            "arrival_display_tick": tick, "scheduled_exact_start": true,
            "observation_restamped": false, "same_owner_world": true,
            "task_qualified": false,
        }));
        live.next_boundary_tick = end_tick;
        live.submitted_chunks += 1;
        return Ok(false);
    }
    if live.pending.is_some() || live.submitted_chunks == live.max_calls {
        return Ok(false);
    }
    let initial = live.submitted_chunks == 0;
    let trigger_tick = if initial {
        0
    } else {
        live.next_boundary_tick - horizon + u64::from(live.prefetch_after_ticks.unwrap())
    };
    if tick < trigger_tick {
        return Ok(false);
    }
    if !initial && tick >= live.next_boundary_tick {
        return Err("prefetch image missed replacement window; explicit pause required".into());
    }
    if !runtime.requested {
        port.request()?;
        runtime.requested = true;
        return Ok(false);
    }
    if let Some(frame) = port.take() {
        let frame = frame?;
        let frame_tick = frame.stamp.source_ticks[0];
        if frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.source_ticks != [frame_tick; 2]
            || frame.stamp.sim_time_ns != frame_tick * ARENA_ACTION_PERIOD_NS
            || (initial && frame_tick != 0)
            || (!initial && frame_tick >= live.next_boundary_tick)
        {
            return Err(
                "prefetch image/self-state is outside the actual fixed scheduling window".into(),
            );
        }
        if frame_tick < trigger_tick {
            // Main/render pipelining can finish an older, correctly stamped
            // image after a request. Discard it, never restamp it or infer on it.
            let mut receipt = outcome.0.lock().unwrap();
            if receipt.prefetch_discarded_image_stamps.len() >= 16 {
                return Err(
                    "bounded prefetch image retries exhausted; explicit pause required".into(),
                );
            }
            receipt
                .prefetch_discarded_image_stamps
                .push(serde_json::json!({
                    "episode_id": frame.stamp.episode_id,
                    "capture_sequence": frame.stamp.capture_sequence,
                    "actual_image_tick": frame_tick, "minimum_prefetch_tick": trigger_tick,
                    "captured_at_unix_ms": frame.stamp.captured_at_unix_ms,
                    "submitted_to_model": false,
                }));
            runtime.requested = false;
            return Ok(false);
        }
        submit_live_observation(live, frame, &runtime.options.output)?;
        runtime.requested = false;
        outcome.0.lock().unwrap().live_policy_inference_calls += 1;
    }
    Ok(false)
}

fn drive_capture(
    mut commands: Commands,
    mut runtime: ResMut<CaptureRuntime>,
    outcome: Res<CaptureOutcome>,
    health: Res<StationRenderHealth>,
    visual: Res<G1VisualStatus>,
    task_model: Option<Res<G1TaskVisualModel>>,
    task_status: Option<Res<G1TaskVisualStatus>>,
    mut task_input: ResMut<G1TaskVisualInput>,
    mut input: ResMut<G1BodyObservationInput>,
    mut camera_mount: ResMut<G1ActiveCameraMount>,
    port: Res<G1CameraPort>,
    mut exit: MessageWriter<AppExit>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shelf_visual: Query<&mut Transform, With<SourceShelfVisual>>,
    (background_model, background_status, mut background_input): (
        Option<Res<G1BackgroundVisualModel>>,
        Option<Res<G1BackgroundVisualStatus>>,
        ResMut<G1BackgroundVisualInput>,
    ),
) {
    runtime.render_frames += 1;
    let result = (|| -> Result<(), String> {
        if let Some(evidence) = &mut runtime.owner_evidence {
            evidence.drain(&mut outcome.0.lock().unwrap())?;
        }
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(evidence) = &mut runtime.assist_evidence {
            evidence.drain(&mut outcome.0.lock().unwrap())?;
        }
        if let Some(error) = health.snapshot().error {
            return Err(error);
        }
        if let Some(error) = &visual.error {
            return Err(error.clone());
        }
        if let Some(error) = task_status.as_ref().and_then(|s| s.error.as_ref()) {
            return Err(error.clone());
        }
        if let Some(error) = background_status.as_ref().and_then(|s| s.error.as_ref()) {
            return Err(error.clone());
        }
        if let Some(snapshot) = runtime.worker.take_latest() {
            input.0 = observation(&snapshot)?;
            background_input.0 = snapshot
                .task_objects
                .as_ref()
                .and_then(|f| f.source_t2_background.as_ref())
                .map(|background| BackgroundOwnerFrame {
                    physics_definition_sha256: background.definition_sha256.clone(),
                    bodies: background
                        .bodies
                        .iter()
                        .filter(|b| b.dynamic)
                        .map(|b| BackgroundBodyPose {
                            path: b.path.clone(),
                            translation_engine: b.translation_engine,
                            rotation_engine_xyzw: b.rotation_engine_xyzw,
                        })
                        .collect(),
                });
            if background_model.is_none()
                && let Some(shelf) = snapshot
                    .task_objects
                    .as_ref()
                    .and_then(|frame| frame.source_t1_shelf.as_ref())
            {
                let transform = Transform {
                    translation: Vec3::from_array(shelf.translation_engine),
                    rotation: Quat::from_array(shelf.rotation_engine_xyzw),
                    ..default()
                };
                if let Some(mut shown) = shelf_visual.iter_mut().next() {
                    *shown = transform;
                } else {
                    commands.spawn((
                        Name::new("actual_source_procedural_support_geometry"),
                        SourceShelfVisual,
                        Mesh3d(meshes.add(Cuboid::new(
                            shelf.size_engine[0],
                            shelf.size_engine[1],
                            shelf.size_engine[2],
                        ))),
                        MeshMaterial3d(materials.add(StandardMaterial {
                            base_color: Color::srgb(0.5, 0.45, 0.36),
                            perceptual_roughness: 0.85,
                            ..default()
                        })),
                        transform,
                    ));
                }
            }
            task_input.0 = match (&task_model, &snapshot.task_objects) {
                (Some(model), Some(frame)) => {
                    let robot = snapshot
                        .frame
                        .as_ref()
                        .ok_or("task object frame has no paired robot")?;
                    if frame.episode_id != robot.episode_id
                        || frame.source_tick != robot.source_tick
                        || frame.sim_time != robot.sim_time
                    {
                        return Err("task object and robot capture boundaries disagree".into());
                    }
                    Some(G1TaskVisualFrame {
                        physics_definition_sha256: frame.definition_sha256.clone(),
                        episode_id: frame.episode_id,
                        source_tick: frame.source_tick,
                        sim_time: frame.sim_time,
                        poses: frame
                            .objects
                            .iter()
                            .map(|o| {
                                let kind = match o.kind {
                                    TaskObjectKind::Apple => "t1_apple",
                                    TaskObjectKind::Plate => "t1_plate",
                                    TaskObjectKind::BrownBox => "t2_box",
                                    TaskObjectKind::BlueBin => "t2_bin",
                                };
                                Ok(G1TaskVisualPose {
                                    object_index: model.object_index(kind)?,
                                    translation_engine: o.translation_engine,
                                    rotation_engine_xyzw: o.rotation_engine_xyzw,
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    })
                }
                (None, None) | (Some(_), None) if snapshot.frame.is_none() => None,
                (None, None) => None,
                _ => return Err("task capture visual/physics pairing incomplete".into()),
            };
            outcome.0.lock().unwrap().observe(&snapshot);
            runtime.latest = Some(snapshot);
        }
        let Some(latest) = runtime.latest.clone() else {
            return Ok(());
        };
        if runtime.interactive {
            return Ok(());
        }
        if latest.phase == G1WorkerPhase::Stopped {
            return Err("worker stopped before capture".into());
        }
        if latest.phase == G1WorkerPhase::Failed && input.0.is_none() {
            return Err(latest
                .reason
                .clone()
                .unwrap_or_else(|| "worker failed before publishing a native snapshot".into()));
        }
        if runtime
            .mobile_assist
            .as_ref()
            .is_some_and(|a| a.auxiliary_view)
            && latest.timing.episode_integrations >= 200
            && runtime
                .live_policy
                .as_ref()
                .is_some_and(|p| p.submitted_chunks == 4)
            && camera_mount.0 == G1CameraMountProfile::ArenaEgo
        {
            camera_mount.0 = G1CameraMountProfile::AuxiliaryGripOverview;
            return Ok(());
        }
        let render_ready = runtime.render_frames > 30
            && health.snapshot().ready
            && input.0.is_some()
            && visual.source_tick == latest.frame.as_ref().map(|frame| frame.source_tick);
        let render_ready = render_ready
            && task_model.as_ref().is_none_or(|_| {
                task_status.as_ref().is_some_and(|s| {
                    s.source_tick == latest.frame.as_ref().map(|f| f.source_tick)
                        && s.episode_id == Some(latest.episode_id)
                        && s.visible_objects
                            == latest.task_objects.as_ref().map_or(0, |f| f.objects.len())
                        && s.visible_objects > 0
                })
            });
        // Continuous extraction independently checks actual robot, prop and
        // camera Tick identities. A moving display may update before this
        // Update-stage readiness read; do not wait for a stationary snapshot.
        let prefetch_running = runtime.live_policy.as_ref().is_some_and(|live| {
            live.prefetch_after_ticks.is_some() && latest.timing.episode_integrations > 0
        });
        if !render_ready && !prefetch_running {
            return Ok(());
        }
        if runtime.live_policy.is_some() && !drive_bounded_task(&mut runtime, &outcome, &port)? {
            return Ok(());
        }
        if runtime.live_policy.is_none() && !runtime.command_submitted && runtime.options.ticks > 0
        {
            runtime
                .worker
                .submit_stand(runtime.episode_id, runtime.options.ticks)?;
            runtime.command_submitted = true;
            return Ok(());
        }
        let capture_boundary = if runtime
            .mobile_assist
            .as_ref()
            .is_some_and(|assist| assist.completed)
        {
            latest.phase == G1WorkerPhase::Paused
        } else if runtime.options.ticks == 0 {
            latest.phase == G1WorkerPhase::Paused && latest.timing.total_integrations == 0
        } else {
            latest.phase == G1WorkerPhase::Failed
                || (latest.phase == G1WorkerPhase::Paused
                    && latest.timing.total_integrations == u64::from(runtime.options.ticks))
        };
        if capture_boundary && !runtime.requested {
            if latest.timing.total_integrations > u64::from(runtime.options.ticks) {
                return Err("worker exceeded the capture integration budget".into());
            }
            port.request()?;
            runtime.requested = true;
            let path = runtime.options.output.join("main_1920x1080.png");
            let saved = runtime.main_saved.clone();
            commands.spawn(Screenshot::primary_window()).observe(
                move |event: On<ScreenshotCaptured>| {
                    let result = if event.image.width() != 1920 || event.image.height() != 1080 {
                        Err("main window resolution changed".into())
                    } else {
                        event
                            .image
                            .clone()
                            .try_into_dynamic()
                            .map_err(|error| error.to_string())
                            .and_then(|image| image.save(&path).map_err(|error| error.to_string()))
                            .map(|()| true)
                    };
                    *saved.lock().unwrap() = result;
                },
            );
            let mut receipt = outcome.0.lock().unwrap();
            receipt.task_visual_status = task_status.as_deref().cloned();
            receipt.background_visual_status = background_status.as_deref().cloned();
            if latest.phase != G1WorkerPhase::Failed {
                receipt.physics_outcome = if runtime
                    .mobile_assist
                    .as_ref()
                    .is_some_and(|assist| assist.completed)
                {
                    if runtime.mobile_assist.as_ref().is_some_and(|a| a.scan_only) {
                        if runtime.mobile_assist.as_ref().is_some_and(|a| a.auxiliary_view) { "bounded_actual_auxiliary_rgb_two_marker_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.restored_view) { "bounded_actual_restored_grip_standing_turn_rgb_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.memory_view) { "bounded_actual_raised_rgb_static_target_memory_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.raising_view) { "bounded_actual_near_rgb_public_geometry_raise_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.visual_approach) { "bounded_actual_rgb_visual_coarse_approach_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.view_with_lowering) { "bounded_classical_scan_and_lower_visibility_pose_complete_not_task_qualified" } else { "bounded_classical_scan_complete_not_task_qualified" }
                    } else {
                        "bounded_classical_carry_skill_complete_not_task_qualified"
                    }
                } else if runtime.options.ticks == 0 {
                    "native_initialization_zero_integrations"
                } else {
                    "bounded_candidate_ticks_complete"
                }
                .into();
            }
        }
        if let Some(frame) = port.take() {
            let frame = frame?;
            if frame.stamp.source == CameraPoseSource::PhysicsBody
                && frame.stamp.episode_id == runtime.episode_id
                && frame.stamp.source_ticks[0] == frame.stamp.source_ticks[1]
                && frame.stamp.source_ticks[0] < latest.timing.episode_integrations
                && frame.stamp.native_state.as_ref().is_some_and(|native| {
                    native.validate().is_ok()
                        && native.body_frame.source_tick == frame.stamp.source_ticks[0]
                        && native.body_frame.episode_id == runtime.episode_id
                })
            {
                let mut receipt = outcome.0.lock().unwrap();
                if receipt.prefetch_discarded_image_stamps.len() >= 16 {
                    return Err("bounded final-image retries exhausted".into());
                }
                receipt
                    .prefetch_discarded_image_stamps
                    .push(serde_json::json!({
                        "phase": "final_capture", "episode_id": frame.stamp.episode_id,
                        "capture_sequence": frame.stamp.capture_sequence,
                        "actual_image_tick": frame.stamp.source_ticks[0],
                        "required_final_tick": latest.timing.episode_integrations,
                        "submitted_to_model": false,
                    }));
                runtime.requested = false;
                return Ok(());
            }
            if frame.stamp.source != CameraPoseSource::PhysicsBody
                || frame.stamp.mount_profile
                    != if runtime
                        .mobile_assist
                        .as_ref()
                        .is_some_and(|a| a.auxiliary_view)
                    {
                        G1CameraMountProfile::AuxiliaryGripOverview
                    } else {
                        G1CameraMountProfile::ArenaEgo
                    }
                || frame.stamp.native_state.is_none()
                || frame.stamp.episode_id != runtime.episode_id
                || frame.stamp.source_ticks[0] != latest.timing.episode_integrations
                || frame.width != 640
                || frame.height != 480
                || frame.rgb.len() != 640 * 480 * 3
            {
                return Err("RGB and completed native measurements do not match".into());
            }
            frame.stamp.native_state.as_ref().unwrap().validate()?;
            if runtime.mobile_assist.as_ref().is_some_and(|a| a.scan_only) {
                // Deliberately omit world camera/body poses and all task objects.
                // Perception receives actual RGB plus original self sensors only.
                let input = marker_observation(&frame.stamp)?;
                fs::write(
                    runtime.options.output.join("vision_observation.json"),
                    serde_json::to_vec_pretty(&input).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
            let rgba = frame
                .rgb
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                .collect();
            Image::new(
                Extent3d {
                    width: 640,
                    height: 480,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                rgba,
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD,
            )
            .try_into_dynamic()
            .map_err(|error| error.to_string())?
            .save(runtime.options.output.join("ego_640x480.png"))
            .map_err(|error| error.to_string())?;
            fs::write(
                runtime.options.output.join("ego_stamp.json"),
                serde_json::to_vec_pretty(&frame.stamp).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            outcome.0.lock().unwrap().ego_stamp = Some(frame.stamp);
            runtime.ego_saved = true;
        }
        if runtime.ego_saved
            && *runtime
                .main_saved
                .lock()
                .unwrap()
                .as_ref()
                .map_err(Clone::clone)?
        {
            if let Some(evidence) = &mut runtime.owner_evidence {
                evidence.drain(&mut outcome.0.lock().unwrap())?;
                if !outcome.0.lock().unwrap().owner_step_trace_complete {
                    return Err("owner evidence did not cover every actual integration".into());
                }
            }
            #[cfg(feature = "g1_constraint_diagnostic")]
            if let Some(evidence) = &mut runtime.assist_evidence {
                evidence.drain(&mut outcome.0.lock().unwrap())?;
                if !outcome.0.lock().unwrap().owner_step_trace_complete {
                    return Err(
                        "assisted owner evidence did not cover every actual integration".into(),
                    );
                }
            }
            outcome.0.lock().unwrap().capture_succeeded = true;
            exit.write(AppExit::Success);
        }
        Ok(())
    })();
    let error = result.err().or_else(|| {
        (!runtime.interactive && runtime.started.elapsed() > runtime.options.timeout)
            .then(|| format!("native camera diagnostic timed out: {}", port.progress()))
    });
    if let Some(error) = error {
        runtime.worker.pause();
        outcome.0.lock().unwrap().failure_reason = Some(error);
        exit.write(AppExit::error());
    }
}
