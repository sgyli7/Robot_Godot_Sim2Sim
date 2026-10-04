//! Bounded native G1 camera evidence in the runner's actual simple floor world.
//!
//! The default scene retains the same 40 x 0.5 x 40 metre native floor. Explicit
//! station configuration binds T1 display and static collisions to one checked
//! station read and removes that floor before the owner starts. Neither
//! initialization nor these captures qualify standing,
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

mod local_model_startup;
#[cfg(feature = "g1_constraint_diagnostic")]
mod mobile_continuous;
mod mobile_qwen_dispatch;
#[cfg(feature = "g1_constraint_diagnostic")]
mod static_grip_check;
#[cfg(feature = "g1_constraint_diagnostic")]
mod static_observed_grasp;
#[cfg(feature = "g1_constraint_diagnostic")]
mod static_observed_place;
#[cfg(feature = "g1_constraint_diagnostic")]
mod static_visual_transfer;
#[cfg(feature = "g1_constraint_diagnostic")]
mod station_carry;
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
    g1_task_fiducial::{
        G1TaskFiducialGate, G1TaskFiducialModel, G1TaskFiducialPlugin, G1TaskFiducialReceipt,
    },
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
use super::g1_marker_vision::{StaticMarkerVisionConfiguration, StaticMarkerWorker};
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
    /// Separate zero-Tick preparation; the owning launcher starts models only
    /// after the renderer-ready receipt. Original task deadlines stay in force.
    #[serde(default)]
    local_model_startup: Option<local_model_startup::Configuration>,
    visual_path: PathBuf,
    visual_sha256: String,
    #[serde(default)]
    task_visual_path: Option<PathBuf>,
    #[serde(default)]
    task_visual_sha256: Option<String>,
    #[serde(default)]
    policy: Option<LivePolicyConfiguration>,
    /// Explicit 60 real startup ticks in the same static task world, before
    /// the first fresh image. Never enabled implicitly or on model failure.
    #[serde(default)]
    static_startup: bool,
    /// Separate no-policy static RGB calibration entry. Never inferred from
    /// task identity or mixed into the original unmarked VLA profile.
    #[serde(default)]
    static_marker_assets: Option<BackgroundVisualConfiguration>,
    /// Separate static-only CPU localization of the actual final native image.
    #[serde(default)]
    static_marker_vision: Option<StaticMarkerVisionConfiguration>,
    #[serde(default)]
    static_grasp_template: Option<BackgroundVisualConfiguration>,
    /// A single extra actual RGB view at the unchanged completed grasp Tick.
    /// Uses the two existing fixed mounts; never advances the physical world.
    #[serde(default)]
    static_grasp_fixed_camera_pair: bool,
    /// Pinned pure CPU paired-RGB program for the separate placement entry.
    #[serde(default)]
    static_pair_program: Option<BackgroundVisualConfiguration>,
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
    /// Explicit post-grasp source-task test; never enables the interactive UI.
    #[serde(default)]
    diagnostic_qwen_dispatch: Option<mobile_qwen_dispatch::Configuration>,
    #[serde(default)]
    station: Option<super::g1_station_environment::G1StationConfiguration>,
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

#[cfg(feature = "g1_constraint_diagnostic")]
struct PendingMobileBoxPair {
    stage: String,
    observation: ObservationStamp,
    rotation: [f32; 4],
    primary_mount: G1CameraMountProfile,
    secondary_mount: G1CameraMountProfile,
    started: Instant,
}

struct MobileAssistCaptureRuntime {
    scan_only: bool,
    view_with_lowering: bool,
    visual_approach: bool,
    raising_view: bool,
    memory_view: bool,
    restored_view: bool,
    auxiliary_view: bool,
    auxiliary_approach: bool,
    auxiliary_release: bool,
    station_motion: bool,
    station_hold_submitted: bool,
    station_placement_view_active: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    pending_box_pair: Option<PendingMobileBoxPair>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    box_pair_restore_mount: Option<G1CameraMountProfile>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    box_pair_camera_activation_frame: Option<u32>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    submitted_carry_observation: Option<ObservationStamp>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    fine_alignment_confirmed: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    fine_goals_submitted: u32,
    #[cfg(feature = "g1_constraint_diagnostic")]
    release_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    hold_submitted: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    thumb_preparation_submitted: bool,
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
    fn camera_mount(&self) -> G1CameraMountProfile {
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(pair) = &self.pending_box_pair {
            return pair.secondary_mount;
        }
        if self.station_placement_view_active {
            G1CameraMountProfile::AuxiliaryBinPlacement
        } else if self.auxiliary_view {
            G1CameraMountProfile::AuxiliaryGripOverview
        } else {
            G1CameraMountProfile::ArenaEgo
        }
    }

    #[cfg(feature = "g1_constraint_diagnostic")]
    fn auxiliary_camera_name(&self) -> &'static str {
        if self.station_placement_view_active {
            "auxiliary_bin_placement"
        } else {
            "auxiliary_grip_overview"
        }
    }

    fn new(
        configuration: MobileAssistStage,
        raising_view: bool,
        memory_view: bool,
        restored_view: bool,
        auxiliary_view: bool,
        auxiliary_approach: bool,
        auxiliary_release: bool,
        station_motion: bool,
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
            auxiliary_approach,
            auxiliary_release,
            station_motion,
            station_hold_submitted: false,
            station_placement_view_active: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            pending_box_pair: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            box_pair_restore_mount: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            box_pair_camera_activation_frame: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            submitted_carry_observation: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            fine_alignment_confirmed: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            fine_goals_submitted: 0,
            #[cfg(feature = "g1_constraint_diagnostic")]
            release_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            hold_submitted: false,
            #[cfg(feature = "g1_constraint_diagnostic")]
            thumb_preparation_submitted: false,
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
    #[serde(default)]
    boundary_images_with_wait: bool,
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
    wait_after_chunks: bool,
    boundary_images_with_wait: bool,
    boundary_image_window: Option<[u64; 2]>,
    wait_image_min_tick: Option<u64>,
    completed: bool,
}

/// Fixed development windows preserve each original decoder's full horizon.
/// Observation time remains the actual image time; execution has its own slot.
fn profile_prefetch_allowed(
    profile: TaskProfile,
    trigger_tick: u32,
    calls: u32,
    total_ticks: u32,
) -> bool {
    match profile {
        // Existing static validation/whole-chunk startup guards stay intact.
        TaskProfile::StaticApple => trigger_tick == 10 && calls >= 2,
        TaskProfile::MobileBox => trigger_tick == 25 && calls == 4 && total_ticks == 200,
    }
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
        static_startup: bool,
    ) -> Result<CaptureWorker, String> {
        if static_startup {
            let Self::Task(config) = self else {
                return Err("static startup requires matched T1 task owner".into());
            };
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                return simulation_minigame::g1::worker::StaticStartupWorker::spawn_static_startup(
                    config,
                )
                .map(CaptureWorker::StaticStartup)
                .map_err(|e| e.to_string());
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            {
                let _ = config;
                return Err("static startup requires explicit development feature".into());
            }
        }
        if assisted_carry {
            let Self::Task(config) = self else {
                return Err("assisted carry requires a matched mobile task owner".into());
            };
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                let station_fixture = matches!(&config.body, ArenaTaskBodyConfig::MobileHomieV2(body) if body.task_objects.as_ref().and_then(|scene| scene.source_t2_background.as_ref()).is_some_and(|background| background.selection == robot_minigame::g1::task_fixtures::T2BackgroundSelection::StationTaskFixtures));
                let worker = if station_fixture {
                    simulation_minigame::g1::worker::MobileAssistWorker::spawn_station_mobile_assist(
                        config,
                    )
                } else {
                    simulation_minigame::g1::worker::MobileAssistWorker::spawn_mobile_assist(config)
                };
                return worker
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
                let station_fixtures = matches!(&config.body, ArenaTaskBodyConfig::MobileHomieV2(body) if body.task_objects.as_ref().and_then(|scene| scene.source_t2_background.as_ref()).is_some_and(|background| background.selection == robot_minigame::g1::task_fixtures::T2BackgroundSelection::StationTaskFixtures));
                let worker = if station_fixtures {
                    ArenaTaskWorker::spawn_mobile_station_fixture_diagnostic(config)
                } else {
                    ArenaTaskWorker::spawn_mobile_constraint_diagnostic(config)
                };
                return worker.map(CaptureWorker::Task).map_err(|e| e.to_string());
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
    StaticStartup(simulation_minigame::g1::worker::StaticStartupWorker),
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
    startup_step: Option<Arc<simulation_minigame::g1::static_startup::StaticStartupStep>>,
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
            startup_step: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            assist_step: None,
        }
    }
}
impl CaptureWorker {
    fn take_latest(&self) -> Option<Arc<CaptureSnapshot>> {
        match self {
            #[cfg(feature = "g1_constraint_diagnostic")]
            Self::StaticStartup(w) => w.take_latest().map(|s| {
                let mut snapshot=CaptureSnapshot::from_worker(&s);
                snapshot.startup_step=s.step.clone();
                if let Some(step)=&s.step {
                    if let simulation_minigame::g1::static_startup::StaticStartupExecution::OriginalVla(execution) | simulation_minigame::g1::static_startup::StaticStartupExecution::OriginalRegrasp(execution)=&step.execution {
                        snapshot.task_step=Some(Arc::new(ArenaTaskStep{execution:execution.clone(),body:step.body.clone()}));
                    }
                }
                Arc::new(snapshot)
            }),
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
            Self::StaticStartup(w) => w.pause(),
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
            Self::StaticStartup(_) => {
                return Err(
                    "startup owner requires explicit typed initialization or original VLA".into(),
                );
            }
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
            Self::StaticStartup(worker) => worker
                .submit(simulation_minigame::g1::worker::TimedCommand {
                    episode_id: timed.episode_id,
                    valid_until_sim_ns: timed.valid_until_sim_ns,
                    valid_until_wall: timed.valid_until_wall,
                    command:
                        simulation_minigame::g1::static_startup::StaticStartupCommand::OriginalVla(
                            timed.command,
                        ),
                })
                .map_err(|e| e.to_string()),
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

/// Two unchanged original T1 chunks, then disclose markers and observe the
/// current physical grasp boundary. No geometric correction is executed here.
pub fn run_static_visual_grasp_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticVisualGrasp)
}

/// Current RGB-mediated classical transfer after exactly two original T1 chunks.
pub fn run_static_visual_transfer_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticVisualTransfer)
}

/// Same transfer and body, then one explicitly declared auxiliary sensor view.
pub fn run_static_visual_transfer_auxiliary_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticVisualTransferAuxiliary)
}

/// The declared near-table T1 sensor after the identical held transfer.
pub fn run_static_visual_transfer_placement_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticVisualTransferPlacement)
}

/// Fresh original grasp/RGB transfer followed by explicitly disclosed memory placement.
pub fn run_static_memory_place_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticMemoryPlace)
}

/// Same live placement plus a finite physical withdrawal to recover the final view.
pub fn run_static_memory_place_observe_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticMemoryPlaceObserve)
}

/// Current100Tick visual admission before the original second chunk closes fingers.
pub fn run_static_pregrasp_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticPregrasp)
}

/// Current preclosure RGB admits a separately labelled classical grasp/lift/hold.
pub fn run_static_observed_grasp_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticObservedGrasp)
}

/// Same observed grasp plus current fixed-pair RGB and bounded classical placement.
pub fn run_static_observed_place_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticObservedPlace)
}

/// One third original chunk from fresh190Tick RGB after an unverified lift.
pub fn run_static_unheld_regrasp_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StaticUnheldRegrasp)
}

/// Four fresh original chunks separated by explicit native standing waits.
/// This source-scene timing preflight has no Qwen/task qualification.
pub fn run_mobile_wait_grasp_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileWaitGrasp)
}

pub fn run_mobile_continuous_release_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileContinuousRelease)
}

/// Two fixed1000/1500Tick original mobile carry-stage budgets. This does
/// not widen the ordinary camera/static-task budgets or qualify task execution.
pub fn run_mobile_carry_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileCarry)
}

/// Fresh station RGB/VLA, disclosed self-state hold, then fixed public-aisle2m
/// locomotion/stop. This development entry cannot grant a target-bin task.
pub fn run_station_mobile_carry_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StationMobileCarry)
}

/// Fresh station grasp/hold, followed by actual marked RGB bin localization,
/// bounded traditional approach and release. Completion is not placement audit.
pub fn run_station_mobile_release_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::StationMobileRelease)
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

/// Current auxiliary RGB binds one coarse carry waypoint. Capture a new
/// stationary two-marker view afterward; retain the original transport grip.
pub fn run_mobile_auxiliary_approach_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileAuxiliaryApproach)
}

/// Current RGB fine carry, a new full-object placement image, then gated real
/// palm opening and physical settling. This is a finite development entry.
pub fn run_mobile_auxiliary_release_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
    run_from_file(path, options, CaptureMode::MobileAuxiliaryRelease)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureMode {
    Camera,
    StaticVisualGrasp,
    StaticVisualTransfer,
    StaticVisualTransferAuxiliary,
    StaticVisualTransferPlacement,
    StaticMemoryPlace,
    StaticMemoryPlaceObserve,
    StaticUnheldRegrasp,
    StaticPregrasp,
    StaticObservedGrasp,
    StaticObservedPlace,
    MobileWaitGrasp,
    MobileContinuousRelease,
    TaskLab,
    MobileCarry,
    MobileAssist,
    StationMobileCarry,
    StationMobileRelease,
    MobileScan,
    MobileTargetView,
    MobileTargetApproach,
    MobileTargetRaiseView,
    MobileTargetMemoryView,
    MobileTargetRestoredView,
    MobileAuxiliaryView,
    MobileAuxiliaryApproach,
    MobileAuxiliaryRelease,
}

impl CaptureMode {
    fn capture_tick_limit(self, task_owner: bool) -> u32 {
        self.assisted_tick_limit().unwrap_or_else(|| match self {
            Self::StaticObservedPlace => 1100,
            Self::StaticMemoryPlaceObserve => 840,
            Self::StaticMemoryPlace => 715,
            Self::MobileCarry => 1500,
            _ if task_owner => 400,
            _ => 150,
        })
    }
    fn assisted_tick_limit(self) -> Option<u32> {
        match self {
            Self::MobileWaitGrasp => Some(1000),
            Self::MobileAssist => Some(2050),
            Self::StationMobileCarry => Some(2300),
            Self::StationMobileRelease => Some(3300),
            Self::MobileScan | Self::MobileAuxiliaryView => Some(1050),
            Self::MobileTargetView => Some(1300),
            Self::MobileTargetApproach
            | Self::MobileTargetRaiseView
            | Self::MobileTargetMemoryView
            | Self::MobileTargetRestoredView
            | Self::MobileAuxiliaryApproach
            | Self::MobileAuxiliaryRelease
            | Self::MobileContinuousRelease => Some(3150),
            Self::Camera
            | Self::StaticVisualGrasp
            | Self::StaticVisualTransfer
            | Self::StaticVisualTransferAuxiliary
            | Self::StaticVisualTransferPlacement
            | Self::StaticMemoryPlace
            | Self::StaticMemoryPlaceObserve
            | Self::StaticUnheldRegrasp
            | Self::StaticPregrasp
            | Self::StaticObservedGrasp
            | Self::StaticObservedPlace
            | Self::TaskLab
            | Self::MobileCarry => None,
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
    let station_fixture_initialization = mode == CaptureMode::Camera
        && options.ticks == 0
        && config.station.is_some()
        && matches!(&config.runner, CaptureRunnerConfig::Task(task) if matches!(&task.body, ArenaTaskBodyConfig::MobileHomieV2(body) if body.task_objects.as_ref().and_then(|scene| scene.source_t2_background.as_ref()).is_some_and(|background| background.selection == robot_minigame::g1::task_fixtures::T2BackgroundSelection::StationTaskFixtures)));
    if matches!(config.runner, CaptureRunnerConfig::Task(_)) != config.policy.is_some()
        && !(config.static_marker_assets.is_some() && config.policy.is_none())
        && !station_fixture_initialization
    {
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
        config.diagnostic_qwen_dispatch,
        config.station,
        config.static_startup,
        config.static_marker_assets,
        config.static_marker_vision,
        config.static_grasp_template,
        config.static_grasp_fixed_camera_pair,
        config.static_pair_program,
        config.local_model_startup,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_station_preparation: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_station_illumination: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_startup: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_model_startup: Option<serde_json::Value>,
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
    pub explicit_standing_wait_maximum_ticks_per_chunk: Option<u32>,
    pub prefetch_discarded_image_stamps: Vec<serde_json::Value>,
    pub prefetch_image_events: Vec<serde_json::Value>,
    pub final_camera_progress: Option<rendering_minigame::g1_camera::G1CaptureProgress>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_marker_localization: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_grip_localization: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_pair_localization: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_place_handoff: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_visual_grasp_handoff: Option<serde_json::Value>,
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
            native_station_preparation: None,
            native_station_illumination: None,
            static_startup: None,
            local_model_startup: None,
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
            explicit_standing_wait_maximum_ticks_per_chunk: None,
            prefetch_discarded_image_stamps: Vec::new(),
            prefetch_image_events: Vec::new(),
            final_camera_progress: None,
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
            static_marker_localization: None,
            static_grip_localization: None,
            static_pair_localization: None,
            static_place_handoff: None,
            static_visual_grasp_handoff: None,
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
        self.owner_step_trace_complete = self.owner_step_trace_dropped == 0
            && self.owner_step_records > 0
            && self.owner_step_records == self.actual_integrations;
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
    station_ambient_override: Option<f32>,
    station_directional_override: Option<f32>,
    station_shadow_override: Option<bool>,
}

#[derive(Resource)]
struct NativeStationIllumination(serde_json::Value);

/// Runs after the native station has created its actual lights. No extra
/// lights or renderer-only foreground are introduced for this comparison.
fn configure_native_station_illumination(
    station: Option<Res<super::g1_station_environment::G1StationSceneActive>>,
    setting: Res<DiagnosticIllumination>,
    mut ambient: Option<ResMut<GlobalAmbientLight>>,
    mut directional: Query<(&mut DirectionalLight, &Transform)>,
    mut commands: Commands,
) {
    if station.is_none() {
        return;
    }
    let Some(ambient) = ambient.as_mut() else {
        commands.insert_resource(NativeStationIllumination(
            serde_json::json!({"error":"station has no effective ambient light"}),
        ));
        return;
    };
    if let Some(brightness) = setting.station_ambient_override {
        ambient.brightness = brightness;
        ambient.color = Color::WHITE;
    }
    let mut lights = Vec::new();
    for (mut light, transform) in &mut directional {
        if let Some(illuminance) = setting.station_directional_override {
            light.illuminance = illuminance;
        }
        if let Some(shadows) = setting.station_shadow_override {
            light.shadow_maps_enabled = shadows;
        }
        lights.push(serde_json::json!({"illuminance":light.illuminance,"shadow_maps_enabled":light.shadow_maps_enabled,
            "color_srgba":light.color.to_srgba().to_f32_array(),"rotation_xyzw":transform.rotation.to_array()}));
    }
    let mut report = serde_json::json!({"schema":"g1_effective_native_station_illumination_v1",
        "read_actual_ecs_lights":true,"explicit_comparison":setting.station_ambient_override.is_some()
            || setting.station_directional_override.is_some() || setting.station_shadow_override.is_some(),
        "ambient_brightness":ambient.brightness,"ambient_color_srgba":ambient.color.to_srgba().to_f32_array(),
        "directional_lights":lights,"added_directional_lights":0});
    if lights.len() != 1 {
        report["error"] = serde_json::json!("station requires one actual directional light");
    }
    commands.insert_resource(NativeStationIllumination(report));
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
    local_model_startup: Option<local_model_startup::Gate>,
    render_frames: u32,
    latest: Option<Arc<CaptureSnapshot>>,
    command_submitted: bool,
    startup_submitted: bool,
    startup_ticks: u64,
    requested: bool,
    ego_saved: bool,
    main_saved: Arc<Mutex<Result<bool, String>>>,
    live_policy: Option<LivePolicyRuntime>,
    owner_evidence: Option<OwnerEvidence>,
    mobile_stand_evidence: Option<OwnerEvidence<simulation_minigame::g1::runner::G1Step>>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    startup_evidence:
        Option<OwnerEvidence<simulation_minigame::g1::static_startup::StaticStartupStep>>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    assist_evidence:
        Option<OwnerEvidence<simulation_minigame::g1::mobile_assist::MobileAssistStep>>,
    mobile_assist: Option<MobileAssistCaptureRuntime>,
    interactive: bool,
    #[cfg(feature = "g1_constraint_diagnostic")]
    continuous_route: Option<mobile_continuous::ContinuousMobileRoute>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    qwen_dispatch: Option<Mutex<mobile_qwen_dispatch::Dispatch>>,
    static_marker_worker: Option<StaticMarkerWorker>,
    static_visual_grasp: bool,
    static_visual_transfer: bool,
    static_memory_place: bool,
    static_memory_observe: bool,
    static_unheld_regrasp: bool,
    static_pregrasp: bool,
    static_observed_grasp: bool,
    static_grasp_fixed_camera_pair: bool,
    static_grasp_pair_secondary: bool,
    static_observed_place: bool,
    static_observed_place_target_tick: Option<u64>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    static_observed_place_route: Option<static_observed_place::StaticObservedPlaceRoute>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    static_observed_grasp_route: Option<static_observed_grasp::StaticObservedGraspRoute>,
    static_regrasp_started: bool,
    static_transfer_auxiliary: bool,
    static_transfer_placement: bool,
    static_auxiliary_activation_frame: Option<u32>,
    #[cfg(feature = "g1_constraint_diagnostic")]
    static_transfer_route: Option<static_visual_transfer::StaticTransferRoute>,
    static_marker_activation_render_frame: Option<u32>,
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
            CaptureWorker::StaticStartup(worker) => worker.reset(),
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
            live.next_boundary_tick = self.startup_ticks;
        }
        self.episode_id = episode_id;
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(assist) = &mut self.mobile_assist {
            if let Some(pair) = assist.pending_box_pair.take() {
                assist.box_pair_restore_mount = Some(pair.primary_mount);
            }
            assist.box_pair_camera_activation_frame = None;
            if matches!(&assist.configuration, MobileAssistStage::Scan(c)
                if c.vision.as_ref().is_some_and(|v| v.same_tick_box_pair))
            {
                assist.vision_job.take();
            }
        }
        self.startup_submitted = false;
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
    type Body: Serialize;
    fn execution(&self) -> &Self::Execution;
    fn body(&self) -> &Self::Body;
    fn static_self_measurement(&self) -> Option<&simulation_minigame::g1::runner::G1Measurement> {
        None
    }
    fn image_admission(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        Ok(None)
    }
}
impl CapturedTaskStep for ArenaTaskStep {
    type Execution = simulation_minigame::g1::task_runner::ArenaTaskExecution;
    type Body = simulation_minigame::g1::task_runner::ArenaBodyStep;
    fn execution(&self) -> &Self::Execution {
        &self.execution
    }
    fn body(&self) -> &simulation_minigame::g1::task_runner::ArenaBodyStep {
        &self.body
    }
}
#[cfg(feature = "g1_constraint_diagnostic")]
impl CapturedTaskStep for simulation_minigame::g1::static_startup::StaticStartupStep {
    type Execution = simulation_minigame::g1::static_startup::StaticStartupExecution;
    type Body = simulation_minigame::g1::task_runner::ArenaBodyStep;
    fn execution(&self) -> &Self::Execution {
        &self.execution
    }
    fn body(&self) -> &simulation_minigame::g1::task_runner::ArenaBodyStep {
        &self.body
    }
    fn static_self_measurement(&self) -> Option<&simulation_minigame::g1::runner::G1Measurement> {
        match &self.body {
            simulation_minigame::g1::task_runner::ArenaBodyStep::StaticAgile(s) => {
                Some(&s.measurement)
            }
            _ => None,
        }
    }
}
#[cfg(feature = "g1_constraint_diagnostic")]
impl CapturedTaskStep for simulation_minigame::g1::mobile_assist::MobileAssistStep {
    type Execution = simulation_minigame::g1::mobile_assist::MobileAssistExecution;
    type Body = simulation_minigame::g1::task_runner::ArenaBodyStep;
    fn execution(&self) -> &Self::Execution {
        &self.execution
    }
    fn body(&self) -> &simulation_minigame::g1::task_runner::ArenaBodyStep {
        &self.body
    }
    fn image_admission(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        self.image_admission
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
    }
}

impl CapturedTaskStep for simulation_minigame::g1::runner::G1Step {
    type Execution = robot_minigame::g1::policy::HomieResult;
    type Body = Self;
    fn execution(&self) -> &Self::Execution {
        &self.inference
    }
    fn body(&self) -> &Self::Body {
        self
    }
}

struct OwnerEvidence<S = ArenaTaskStep> {
    trace: OwnerStepTrace<S>,
    output: BufWriter<fs::File>,
    records: u64,
    first_boundary_wall_ms: Option<f64>,
    static_self_samples: Vec<simulation_minigame::g1::runner::G1Measurement>,
}

impl<S: CapturedTaskStep> OwnerEvidence<S> {
    fn drain(&mut self, receipt: &mut G1CaptureReceipt) -> Result<(), String> {
        for record in self.trace.drain() {
            // Separate bounded self-sensor copy; task/body truth never enters
            // the memory estimator, despite sharing this audit subscription.
            if let Some(state) = record.step.static_self_measurement() {
                if (140..=390).contains(&state.source_tick) {
                    if self.static_self_samples.len() >= 251 {
                        return Err("static self history overflow".into());
                    }
                    self.static_self_samples.push(state.clone());
                }
            }

            let mut value = serde_json::json!({
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
            if let Some(admission) = record.step.image_admission().map_err(|e| e.to_string())? {
                value["image_admission"] = admission;
            }
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
        None,
        None,
        false,
        None,
        None,
        None,
        false,
        None,
        None,
        CaptureMode::Camera,
    )
}

fn run_capture_owner(
    mut config: CaptureRunnerConfig,
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
    diagnostic_qwen_dispatch: Option<mobile_qwen_dispatch::Configuration>,
    station: Option<super::g1_station_environment::G1StationConfiguration>,
    static_startup: bool,
    static_marker_assets: Option<BackgroundVisualConfiguration>,
    static_marker_vision: Option<StaticMarkerVisionConfiguration>,
    static_grasp_template: Option<BackgroundVisualConfiguration>,
    static_grasp_fixed_camera_pair: bool,
    static_pair_program: Option<BackgroundVisualConfiguration>,
    local_model_startup: Option<local_model_startup::Configuration>,
    mode: CaptureMode,
) -> Result<G1CaptureReceipt, String> {
    let interactive = mode == CaptureMode::TaskLab;
    if local_model_startup.is_some() && mode != CaptureMode::StationMobileRelease {
        return Err(
            "renderer-first local model preparation is an explicit station-release entry".into(),
        );
    }
    let local_model_startup = local_model_startup
        .map(local_model_startup::Gate::new)
        .transpose()?;
    let mobile_carry = mode == CaptureMode::MobileCarry;
    let station_carry = mode == CaptureMode::StationMobileCarry;
    let station_release = mode == CaptureMode::StationMobileRelease;
    let station_motion = station_carry || station_release;
    let station_t2_fixtures = config
        .task_objects()
        .and_then(|scene| scene.source_t2_background.as_ref())
        .is_some_and(|background| {
            background.selection
                == robot_minigame::g1::task_fixtures::T2BackgroundSelection::StationTaskFixtures
        });
    if station_motion && !station_t2_fixtures {
        return Err(
            "station mobile motion requires explicit frozen station fixture coverage".into(),
        );
    }
    if station_t2_fixtures
        && (!cfg!(feature = "g1_constraint_diagnostic")
            || !matches!(
                mode,
                CaptureMode::Camera
                    | CaptureMode::StationMobileCarry
                    | CaptureMode::StationMobileRelease
            )
            || station.is_none()
            || !matches!(&config, CaptureRunnerConfig::Task(task) if task.body.profile() == TaskProfile::MobileBox)
            || if station_motion {
                options.ticks != mode.assisted_tick_limit().unwrap_or(0)
            } else {
                !matches!(options.ticks, 0 | 200)
            }
            || if options.ticks == 0 {
                policy.is_some()
            } else {
                policy.as_ref().is_none_or(|policy| {
                    policy.max_calls != 4
                        || policy.prefetch_after_ticks.is_some()
                        || policy.boundary_images_with_wait
                })
            }
            || diagnostic_constraint_sweeps != Some(4)
            || predictive_limit_diagnostic
            || background_visual.is_none()
            || task_visual.is_none()
            || render_only_environment_translation.is_some()
            || diagnostic_source_rect_lighting.is_some()
            || task_lab.is_some()
            || if station_carry {
                mobile_assist.as_ref().is_none_or(|c| {
                    c.relative_distance_m != 2.
                        || c.heading_yaw_source_rad != std::f32::consts::FRAC_PI_2
                })
            } else {
                mobile_assist.is_some()
            }
            || if station_release {
                mobile_scan.as_ref().is_none_or(|c| {
                    c.heading_yaw_source_rad != -std::f32::consts::FRAC_PI_2
                        || c.lowering.is_some()
                        || c.vision.is_none()
                })
            } else {
                mobile_scan.is_some()
            }
            || diagnostic_qwen_dispatch
                .as_ref()
                .is_some_and(|q| !station_release || !q.is_station_initial()))
    {
        return Err("station T2 fixture entry permits zero-Tick camera initialization,200Tick/four-fresh-chunk grasp,2300Tick public-aisle carry/stop, or3300Tick actual marked RGB bin approach/release, with100Tick self-state hold; only the explicit initial station Qwen scope may dispatch the3300Tick fixture; source warehouse and prefetch remain inadmissible".into());
    }
    if let Some(qwen) = &diagnostic_qwen_dispatch {
        if !cfg!(feature = "g1_constraint_diagnostic") {
            return Err(
                "Qwen transport requires the explicit constraint diagnostic feature".into(),
            );
        }
        qwen.validate_entry(mode, &config)?;
    }
    let qwen_dispatch_enabled = diagnostic_qwen_dispatch.is_some();
    let static_memory_observe = mode == CaptureMode::StaticMemoryPlaceObserve;
    let static_unheld_regrasp = mode == CaptureMode::StaticUnheldRegrasp;
    let static_pregrasp = mode == CaptureMode::StaticPregrasp;
    let static_observed_place = mode == CaptureMode::StaticObservedPlace;
    #[cfg(not(feature = "g1_constraint_diagnostic"))]
    if static_observed_place {
        return Err(
            "observed placement requires the explicit constraint diagnostic feature".into(),
        );
    }
    let static_observed_grasp = matches!(
        mode,
        CaptureMode::StaticObservedGrasp | CaptureMode::StaticObservedPlace
    );
    if static_pair_program.is_some() != static_observed_place
        || (static_observed_place && !static_grasp_fixed_camera_pair)
    {
        return Err("observed placement requires its explicit paired-RGB program and entry".into());
    }
    if static_grasp_fixed_camera_pair && !static_observed_grasp {
        return Err("fixed camera pair requires the explicit observed-grasp entry".into());
    }
    if static_grasp_template.is_some() != static_observed_grasp {
        return Err("observed grasp template requires its explicit entry".into());
    }
    let static_memory_place = matches!(
        mode,
        CaptureMode::StaticMemoryPlace | CaptureMode::StaticMemoryPlaceObserve
    );
    let static_transfer_placement = matches!(
        mode,
        CaptureMode::StaticVisualTransferPlacement
            | CaptureMode::StaticMemoryPlace
            | CaptureMode::StaticMemoryPlaceObserve
            | CaptureMode::StaticUnheldRegrasp
    );
    let static_transfer_auxiliary = matches!(
        mode,
        CaptureMode::StaticVisualTransferAuxiliary
            | CaptureMode::StaticVisualTransferPlacement
            | CaptureMode::StaticMemoryPlace
            | CaptureMode::StaticMemoryPlaceObserve
            | CaptureMode::StaticUnheldRegrasp
    );
    let static_visual_transfer = matches!(
        mode,
        CaptureMode::StaticVisualTransfer
            | CaptureMode::StaticVisualTransferAuxiliary
            | CaptureMode::StaticVisualTransferPlacement
            | CaptureMode::StaticMemoryPlace
            | CaptureMode::StaticMemoryPlaceObserve
            | CaptureMode::StaticUnheldRegrasp
    );
    let static_visual_grasp = matches!(
        mode,
        CaptureMode::StaticVisualGrasp
            | CaptureMode::StaticVisualTransfer
            | CaptureMode::StaticVisualTransferAuxiliary
            | CaptureMode::StaticVisualTransferPlacement
            | CaptureMode::StaticMemoryPlace
            | CaptureMode::StaticMemoryPlaceObserve
            | CaptureMode::StaticUnheldRegrasp
            | CaptureMode::StaticPregrasp
            | CaptureMode::StaticObservedGrasp
            | CaptureMode::StaticObservedPlace
    );
    let static_visual_budget = if static_observed_place {
        1100
    } else if static_observed_grasp {
        400
    } else if static_pregrasp {
        100
    } else if static_unheld_regrasp {
        230
    } else if static_memory_observe {
        840
    } else if static_memory_place {
        715
    } else if static_visual_transfer {
        390
    } else {
        140
    };
    let memory_view = mode == CaptureMode::MobileTargetMemoryView;
    let restored_view = mode == CaptureMode::MobileTargetRestoredView;
    let continuous = mode == CaptureMode::MobileContinuousRelease;
    let auxiliary_release =
        mode == CaptureMode::MobileAuxiliaryRelease || continuous || station_release;
    let auxiliary_approach = mode == CaptureMode::MobileAuxiliaryApproach || auxiliary_release;
    let auxiliary_view = mode == CaptureMode::MobileAuxiliaryView || auxiliary_approach;
    let raising_view = mode == CaptureMode::MobileTargetRaiseView || memory_view || restored_view;
    let visual_approach =
        mode == CaptureMode::MobileTargetApproach || raising_view || auxiliary_view;
    let target_view = mode == CaptureMode::MobileTargetView || (visual_approach && !auxiliary_view);
    let scan_only = mode == CaptureMode::MobileScan || target_view || auxiliary_view;
    let waited_grasp = mode == CaptureMode::MobileWaitGrasp || continuous;
    let assisted_carry =
        mode == CaptureMode::MobileAssist || station_carry || scan_only || waited_grasp;
    if static_startup
        && (!matches!(
            mode,
            CaptureMode::Camera
                | CaptureMode::StaticVisualGrasp
                | CaptureMode::StaticVisualTransfer
                | CaptureMode::StaticVisualTransferAuxiliary
                | CaptureMode::StaticVisualTransferPlacement
                | CaptureMode::StaticMemoryPlace
                | CaptureMode::StaticMemoryPlaceObserve
                | CaptureMode::StaticUnheldRegrasp
                | CaptureMode::StaticPregrasp
                | CaptureMode::StaticObservedGrasp
                | CaptureMode::StaticObservedPlace
        ) || assisted_carry
            || !predictive_limit_diagnostic
            || diagnostic_constraint_sweeps != Some(16)
            || !matches!(&config,CaptureRunnerConfig::Task(c) if c.body.profile()==TaskProfile::StaticApple)
            || if static_marker_assets.is_some() {
                policy.is_some() != static_visual_grasp
                    || options.ticks
                        != if static_visual_grasp {
                            static_visual_budget
                        } else {
                            60
                        }
            } else {
                policy.as_ref().is_none_or(|p| p.boundary_images_with_wait)
            })
    {
        return Err("explicit static startup requires bounded T1 camera/live-policy mode and predictive 16-sweep owner without mobile boundary waits".into());
    }
    let startup_ticks = if static_startup { 60 } else { 0 };
    if policy.as_ref().is_some_and(|p| p.boundary_images_with_wait)
        && (!waited_grasp
            || policy
                .as_ref()
                .is_some_and(|p| p.prefetch_after_ticks.is_some()))
    {
        return Err(
            "boundary original images require the explicit finite waited mobile scene".into(),
        );
    }
    let mobile_prefetch = matches!(&config, CaptureRunnerConfig::Task(c) if c.body.profile() == TaskProfile::MobileBox)
        && policy
            .as_ref()
            .is_some_and(|p| p.prefetch_after_ticks.is_some());
    if mobile_prefetch
        && (mode != CaptureMode::Camera
            || !cfg!(feature = "g1_constraint_diagnostic")
            || options.ticks != 200
            || policy
                .as_ref()
                .is_none_or(|p| p.max_calls != 4 || p.prefetch_after_ticks != Some(25))
            || config
                .task_objects()
                .is_none_or(|c| c.source_t2_background.is_none())
            || diagnostic_constraint_sweeps != Some(4)
            || predictive_limit_diagnostic
            || diagnostic_source_rect_lighting.is_none()
            || !diagnostic_aces_fitted
            || task_visual.is_none()
            || task_lab.is_some()
            || mobile_assist.is_some()
            || mobile_scan.is_some())
    {
        return Err("mobile prefetch is only the explicit200Tick/four-original-chunk matched source-camera/light/4PGS grasp diagnostic".into());
    }
    if matches!(
        mode,
        CaptureMode::MobileAssist | CaptureMode::StationMobileCarry
    ) != mobile_assist.is_some()
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
            || if station_motion {
                diagnostic_source_rect_lighting.is_some()
            } else {
                diagnostic_source_rect_lighting.is_none()
            }
            || !diagnostic_aces_fitted
            || task_lab.is_some()
            || ((scan_only || waited_grasp) && task_visual.is_none())
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
                    || c.vision.as_ref().is_some_and(|v| {
                        v.task_geometry.is_some() != (raising_view || auxiliary_release)
                    })
                    || c.lowering
                        .as_ref()
                        .is_some_and(|l| l.distance_m != 0.18 || l.duration_ticks != 150)
            }))
    {
        return Err("mobile development entry requires its exact finite Tick budget, four fresh original grasp chunks and matched scene/light/4PGS profile; manual carry is exactly2m and station bin release is separately bounded3300Ticks".into());
    }
    if let Some(vision) = mobile_scan.as_ref().and_then(|c| c.vision.as_ref()) {
        vision.validate()?;
    }
    if static_marker_assets.is_some()
        && (!matches!(
            mode,
            CaptureMode::Camera
                | CaptureMode::StaticVisualGrasp
                | CaptureMode::StaticVisualTransfer
                | CaptureMode::StaticVisualTransferAuxiliary
                | CaptureMode::StaticVisualTransferPlacement
                | CaptureMode::StaticMemoryPlace
                | CaptureMode::StaticMemoryPlaceObserve
                | CaptureMode::StaticUnheldRegrasp
                | CaptureMode::StaticPregrasp
                | CaptureMode::StaticObservedGrasp
                | CaptureMode::StaticObservedPlace
        ) || !matches!(&config, CaptureRunnerConfig::Task(c) if c.body.profile()==TaskProfile::StaticApple)
            || policy.is_some() != static_visual_grasp
            || if static_visual_grasp {
                !static_startup || options.ticks != static_visual_budget
            } else {
                static_startup != (options.ticks == 60) || !matches!(options.ticks, 0 | 60)
            }
            || mobile_scan.is_some()
            || task_visual.is_none()
            || station.is_none()
            || !predictive_limit_diagnostic
            || diagnostic_constraint_sweeps != Some(16))
    {
        return Err("static labels require explicit0/60Tick calibration,140Tick grasp view or390Tick visual transfer".into());
    }
    if static_visual_grasp
        && (static_marker_assets.is_none()
            || static_marker_vision.is_none()
            || policy.as_ref().is_none_or(|p| {
                p.max_calls
                    != if static_pregrasp || static_observed_grasp {
                        1
                    } else if static_unheld_regrasp {
                        3
                    } else {
                        2
                    }
                    || p.prefetch_after_ticks.is_some()
                    || p.boundary_images_with_wait
            })
            || !static_startup
            || options.ticks != static_visual_budget)
    {
        return Err("static visual mode requires60startupTicks, exactly two original40frame chunks, its bound CPUworker and exact140/390Tick scene budget".into());
    }
    let static_labels = static_marker_assets.is_some();
    if let Some(vision) = &static_marker_vision {
        if !static_labels || interactive || (policy.is_some() && !static_visual_grasp) {
            return Err("static CPU localization requires calibration or the separate original two-chunk visual handoff".into());
        }
        if vision.calibration_version == 3
            && !matches!(
                mode,
                CaptureMode::StaticObservedGrasp
                    | CaptureMode::StaticObservedPlace
                    | CaptureMode::StaticPregrasp
                    | CaptureMode::Camera
            )
        {
            return Err(
                "multi-face static labels require their separate observed-grasp/calibration entry"
                    .into(),
            );
        }
        vision.validate()?;
        let labels = static_marker_assets.as_ref().unwrap();
        if labels.path != vision.fiducial_path || labels.sha256 != vision.fiducial_sha256 {
            return Err(
                "static CPU calibration differs from the actually printed scene labels".into(),
            );
        }
    }
    let fiducial_model = mobile_scan
        .as_ref()
        .map(|c| &c.fiducial_assets)
        .or(static_marker_assets.as_ref())
        .map(|c| G1TaskFiducialModel::load(&c.path, &c.sha256))
        .transpose()?;
    if let Some(model) = &fiducial_model {
        let auxiliary_labels = matches!(model.receipt.layout_profile,
            rendering_minigame::g1_task_fiducial::G1FiducialLayoutProfile::AuxiliaryGripTargets
            | rendering_minigame::g1_task_fiducial::G1FiducialLayoutProfile::AuxiliaryBinBoardTargets);
        let static_profile = model.receipt.layout_profile.is_static();
        if auxiliary_labels != auxiliary_view || static_profile != static_labels {
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
    #[cfg(feature = "g1_constraint_diagnostic")]
    let continuous_route = if continuous {
        Some(mobile_continuous::ContinuousMobileRoute::new(
            mobile_scan
                .as_ref()
                .ok_or("continuous marker configuration absent")?,
        )?)
    } else {
        None
    };
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
    let station_shadow_override = directional_shadow_maps;
    let directional_shadow_maps = directional_shadow_maps.unwrap_or(true);
    let ambient_brightness = diagnostic_ambient_brightness.unwrap_or(450.);
    let directional_illuminance = diagnostic_directional_illuminance.unwrap_or(15_000.);
    if !ambient_brightness.is_finite()
        || !(0.0..=20_000.).contains(&ambient_brightness)
        || !directional_illuminance.is_finite()
        || !(0.0..=50_000.).contains(&directional_illuminance)
        || ((diagnostic_ambient_brightness.is_some()
            || diagnostic_directional_illuminance.is_some())
            && background_visual.is_none()
            && station.is_none())
    {
        return Err("diffuse-light comparison requires an explicit source or native station environment and finite bounded ambient/directional energies".into());
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
                let load = if station_t2_fixtures {
                    G1BackgroundVisualModel::load_mobile_station_fixtures
                } else {
                    G1BackgroundVisualModel::load_mobile
                };
                load(
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
    let tick_limit = mode.capture_tick_limit(matches!(config, CaptureRunnerConfig::Task(_)));
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
                || (!interactive
                    && !assisted_carry
                    && !static_visual_transfer
                    && !static_observed_grasp
                    && u64::from(options.ticks)
                        != u64::from(policy.max_calls * horizon) + startup_ticks)
                || policy.timeout_ms == 0
                || policy.timeout_ms > 20_000
                || config.max_observation_wall_age_ms == 0
                || config.max_observation_wall_age_ms > 20_000
                || policy.prefetch_after_ticks.is_some_and(|tick| {
                    !profile_prefetch_allowed(profile, tick, policy.max_calls, options.ticks)
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
                next_boundary_tick: startup_ticks,
                prefetch_after_ticks: policy.prefetch_after_ticks,
                wait_after_chunks: waited_grasp,
                boundary_images_with_wait: policy.boundary_images_with_wait,
                boundary_image_window: None,
                wait_image_min_tick: None,
                completed: false,
            })
        }
        (_, None) => None,
        _ => return Err("live policy requires the matched task owner".into()),
    };
    let assets = validate_render_asset_root(&default_asset_root())?;
    let mut scene = StationScene::load(&assets)?;
    let mut station_preparation = None;
    if let Some(station) = &station {
        if (background_model.is_some() && !station_t2_fixtures)
            || render_only_environment_translation.is_some()
            || (assisted_carry && !station_motion)
            || source_rect_lighting.is_some()
        {
            return Err("station preparation rejects source background overlays, render-only poses or unvalidated assisted task factories".into());
        }
        let (floor_contact_friction, startup_environment) = match &mut config {
            CaptureRunnerConfig::Static(body) =>
                (body.floor_contact_friction, &mut body.startup_environment),
            CaptureRunnerConfig::Mobile(body) if body.task_objects.is_none() =>
                (body.floor_contact_friction, &mut body.startup_environment),
            CaptureRunnerConfig::Task(task) => {
                match &mut task.body {
                    ArenaTaskBodyConfig::StaticAgile(body) => (body.floor_contact_friction, &mut body.startup_environment),
                    ArenaTaskBodyConfig::MobileHomieV2(body) if station_t2_fixtures => (body.floor_contact_friction, &mut body.startup_environment),
                    _ => return Err("scientific-station mobile task scenes require the separately bounded frozen support fixture diagnostic".into()),
                }
            }
            _ => return Err(
                "scientific-station preparation requires static AGILE or mobile standing without task objects"
                    .into(),
            ),
        };
        let (static_scene, environment, receipt) =
            super::g1_station_environment::prepare(&scene, station, floor_contact_friction)?;
        *startup_environment = Some(environment);
        scene = static_scene;
        station_preparation = Some(receipt);
    }
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
        receipt.explicit_standing_wait_maximum_ticks_per_chunk = waited_grasp.then_some(200);
        receipt.pauses_for_camera_and_policy =
            receipt.diagnostic_prefetch_after_ticks.is_none() && !waited_grasp;
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
    if static_observed_place {
        outcome.0.lock().unwrap().scope =
            "native_actual_rgb_observed_grasp_fixed_pair_classical_place_diagnostic_not_qualified";
    } else if static_observed_grasp {
        outcome.0.lock().unwrap().scope =
            "native_actual_preclosure_rgb_and_disclosed_classical_template_grasp_diagnostic";
    }
    if waited_grasp {
        outcome.0.lock().unwrap().scope = "native_four_fresh_original_grasp_chunks_with_explicit_physical_standing_waits_not_qualified";
    }
    if continuous {
        outcome.0.lock().unwrap().scope = "native_bounded_continuous_actual_rgb_grasp_classical_carry_release_source_scene_not_qualified";
    }
    if qwen_dispatch_enabled {
        outcome.0.lock().unwrap().scope = if station_release {
            "native_station_initial_local_qwen_fixed_profile_and_rgb_transport_feedback_diagnostic_not_qualified"
        } else {
            "native_source_mobile_postgrasp_local_qwen_transport_and_feedback_diagnostic_not_qualified"
        };
    }
    if let Some(preparation) = station_preparation {
        let mut receipt = outcome.0.lock().unwrap();
        receipt.environment = if station_t2_fixtures {
            receipt.scope = if station_release {
                "native_scientific_station_t2_fresh_grasp_hold_actual_marked_rgb_bin_approach_release_not_task_qualified"
            } else if station_carry {
                "native_scientific_station_t2_fresh_grasp_self_state_hold_public_aisle2m_carry_stop_not_target_bin_task_qualified"
            } else {
                "native_scientific_station_t2_frozen_support_fixtures_initialization_or_fresh_grasp_diagnostic_not_qualified"
            };
            "scientific_station_native_static_world_t2_support_fixture_development"
        } else if matches!(&config, CaptureRunnerConfig::Mobile(_)) {
            receipt.scope = "native_scientific_station_mobile_standing_camera_diagnostic";
            "scientific_station_native_static_world_mobile_stand_development"
        } else {
            "scientific_station_native_static_world_t1_development"
        };
        receipt.native_station_preparation = Some(preparation);
        receipt.floor_center_engine = [0.; 3];
        receipt.floor_full_extents_m = [0.; 3];
    }
    #[cfg(feature = "g1_constraint_diagnostic")]
    let static_transfer_route = if static_visual_transfer {
        let CaptureRunnerConfig::Task(c) = &config else {
            return Err("static transfer task owner absent".into());
        };
        let ArenaTaskBodyConfig::StaticAgile(b) = &c.body else {
            return Err("static transfer AGILE body absent".into());
        };
        Some(static_visual_transfer::StaticTransferRoute::new(
            &b.definition,
            &b.definition_sha256,
            if static_memory_place {
                b.task_objects.as_ref()
            } else {
                None
            },
        )?)
    } else {
        None
    };
    #[cfg(feature = "g1_constraint_diagnostic")]
    let static_observed_grasp_route = if let Some(template) = static_grasp_template.as_ref() {
        let CaptureRunnerConfig::Task(c) = &config else {
            return Err("observed grasp task absent".into());
        };
        let ArenaTaskBodyConfig::StaticAgile(b) = &c.body else {
            return Err("observed grasp AGILE absent".into());
        };
        Some(static_observed_grasp::StaticObservedGraspRoute::new(
            &b.definition,
            &b.definition_sha256,
            template,
        )?)
    } else {
        None
    };
    #[cfg(feature = "g1_constraint_diagnostic")]
    let static_observed_place_route = if let Some(program) = static_pair_program {
        let CaptureRunnerConfig::Task(c) = &config else {
            return Err("observed placement task absent".into());
        };
        let ArenaTaskBodyConfig::StaticAgile(b) = &c.body else {
            return Err("observed placement AGILE absent".into());
        };
        Some(static_observed_place::StaticObservedPlaceRoute::new(
            static_marker_vision
                .as_ref()
                .ok_or("observed placement CPU configuration absent")?,
            program,
            b.task_objects
                .as_ref()
                .ok_or("observed placement public geometry absent")?,
        )?)
    } else {
        None
    };
    #[cfg(feature = "g1_constraint_diagnostic")]
    let qwen_dispatch = diagnostic_qwen_dispatch
        .map(|c| mobile_qwen_dispatch::Dispatch::new(c, episode_id).map(Mutex::new))
        .transpose()?;
    let worker = config.spawn(
        predictive_limit_diagnostic,
        diagnostic_constraint_sweeps,
        assisted_carry,
        static_startup,
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
            static_self_samples: Vec::new(),
        }),
        CaptureWorker::Mobile(_) | CaptureWorker::Static { .. } => None,
        #[cfg(feature = "g1_constraint_diagnostic")]
        CaptureWorker::AssistedMobile(_) => None,
        #[cfg(feature = "g1_constraint_diagnostic")]
        CaptureWorker::StaticStartup(_) => None,
    };
    let mobile_stand_evidence = match &worker {
        CaptureWorker::Mobile(worker) if station.is_some() => Some(OwnerEvidence {
            trace: worker.subscribe_steps(256).map_err(|e| e.to_string())?,
            output: BufWriter::new(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output.join("owner_steps.jsonl"))
                    .map_err(|e| e.to_string())?,
            ),
            records: 0,
            first_boundary_wall_ms: None,
            static_self_samples: Vec::new(),
        }),
        _ => None,
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
            static_self_samples: Vec::new(),
        }),
        _ => None,
    };
    #[cfg(feature = "g1_constraint_diagnostic")]
    let startup_evidence = match &worker {
        CaptureWorker::StaticStartup(worker) => Some(OwnerEvidence {
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
            static_self_samples: Vec::new(),
        }),
        _ => None,
    };
    if static_startup {
        outcome.0.lock().unwrap().static_startup = Some(serde_json::json!({
            "requested_real_ticks":60,"completed":false,"ready":false,
            "same_owner_world":true,"self_velocity_and_imu_only":true,
            "original_vla_output":false,"task_qualified":false,
        }));
    }
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
    let static_marker_worker = static_marker_vision
        .map(|c| StaticMarkerWorker::spawn(c, &options.output, episode_id))
        .transpose()?;
    let mut app = App::new();
    // Default mode uses the scene's enamel configuration only. Explicit station
    // mode below spawns the matching static meshes with the robot's main camera.
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
            station_ambient_override: diagnostic_ambient_brightness,
            station_directional_override: diagnostic_directional_illuminance,
            station_shadow_override,
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
            local_model_startup,
            render_frames: 0,
            latest: None,
            command_submitted: false,
            startup_submitted: false,
            startup_ticks,
            requested: false,
            ego_saved: false,
            main_saved: Arc::new(Mutex::new(Ok(false))),
            live_policy,
            owner_evidence,
            mobile_stand_evidence,
            #[cfg(feature = "g1_constraint_diagnostic")]
            startup_evidence,
            #[cfg(feature = "g1_constraint_diagnostic")]
            assist_evidence,
            mobile_assist: if continuous {
                None
            } else {
                mobile_stage.map(|c| {
                    MobileAssistCaptureRuntime::new(
                        c,
                        raising_view,
                        memory_view,
                        restored_view,
                        auxiliary_view,
                        auxiliary_approach,
                        auxiliary_release,
                        station_motion,
                    )
                })
            },
            interactive,
            #[cfg(feature = "g1_constraint_diagnostic")]
            continuous_route,
            #[cfg(feature = "g1_constraint_diagnostic")]
            qwen_dispatch,
            static_marker_worker,
            static_visual_grasp,
            static_visual_transfer,
            static_memory_place,
            static_memory_observe,
            static_unheld_regrasp,
            static_pregrasp,
            static_observed_grasp,
            static_grasp_fixed_camera_pair,
            static_grasp_pair_secondary: false,
            static_observed_place,
            static_observed_place_target_tick: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            static_observed_place_route,
            #[cfg(feature = "g1_constraint_diagnostic")]
            static_observed_grasp_route,
            static_regrasp_started: false,
            static_transfer_auxiliary,
            static_transfer_placement,
            static_auxiliary_activation_frame: None,
            #[cfg(feature = "g1_constraint_diagnostic")]
            static_transfer_route,
            static_marker_activation_render_frame: None,
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
                        title: if station.is_some() {
                            "G1 科学站任务验证".into()
                        } else {
                            "G1 native camera diagnostic — unqualified floor candidate".into()
                        },
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
    if std::env::var("G1_URI_PRESENTATION").as_deref() != Ok("0") {
        app.add_plugins(rendering_minigame::g1_uri_presentation::G1UriPresentationPlugin);
    }
    if station.is_some() {
        app.insert_resource(super::g1_station_environment::G1StationSceneActive)
            .insert_resource(rendering_minigame::StationExternalCamera)
            .add_plugins(rendering_minigame::StationVisualPlugin)
            .add_systems(PostStartup, configure_native_station_illumination);
    }
    if let Some(model) = task_model {
        app.insert_resource(model).add_plugins(G1TaskVisualPlugin);
    }
    if let Some(model) = fiducial_model {
        if continuous || static_visual_grasp {
            app.insert_resource(G1TaskFiducialGate { enabled: false });
        }
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
    use super::{CaptureMode, profile_prefetch_allowed};
    use task_minigame::types::TaskProfile;

    #[test]
    fn observed_place_budget_is_registered_in_the_actual_capture_guard_only() {
        assert_eq!(
            CaptureMode::StaticObservedPlace.capture_tick_limit(true),
            1100
        );
        for mode in [
            CaptureMode::StaticObservedGrasp,
            CaptureMode::StaticPregrasp,
            CaptureMode::StaticVisualGrasp,
            CaptureMode::Camera,
            CaptureMode::TaskLab,
        ] {
            assert_eq!(mode.capture_tick_limit(true), 400);
            assert_eq!(mode.capture_tick_limit(false), 150);
        }
        assert_eq!(CaptureMode::StaticMemoryPlace.capture_tick_limit(true), 715);
        assert_eq!(
            CaptureMode::StaticMemoryPlaceObserve.capture_tick_limit(true),
            840
        );
        assert_eq!(CaptureMode::MobileCarry.capture_tick_limit(true), 1500);
        assert_eq!(
            CaptureMode::MobileContinuousRelease.capture_tick_limit(true),
            3150
        );
    }
    #[test]
    fn prefetch_windows_do_not_mix_profiles_or_change_original_horizons() {
        assert!(profile_prefetch_allowed(
            TaskProfile::StaticApple,
            10,
            2,
            80
        ));
        assert!(profile_prefetch_allowed(
            TaskProfile::StaticApple,
            10,
            8,
            320
        ));
        assert!(profile_prefetch_allowed(TaskProfile::MobileBox, 25, 4, 200));
        for (profile, trigger, calls, ticks) in [
            (TaskProfile::StaticApple, 25, 4, 160),
            (TaskProfile::MobileBox, 10, 4, 200),
            (TaskProfile::MobileBox, 25, 3, 150),
            (TaskProfile::MobileBox, 25, 4, 199),
            (TaskProfile::MobileBox, 25, 8, 400),
            (TaskProfile::StaticApple, 10, 1, 40),
        ] {
            assert!(!profile_prefetch_allowed(profile, trigger, calls, ticks));
        }
    }
    #[test]
    fn both_startup_guards_share_the_explicit_stage_budget() {
        for (mode, budget) in [
            (CaptureMode::MobileWaitGrasp, 1000),
            (CaptureMode::MobileContinuousRelease, 3150),
            (CaptureMode::MobileAssist, 2050),
            (CaptureMode::StationMobileCarry, 2300),
            (CaptureMode::StationMobileRelease, 3300),
            (CaptureMode::MobileScan, 1050),
            (CaptureMode::MobileTargetView, 1300),
            (CaptureMode::MobileTargetApproach, 3150),
            (CaptureMode::MobileTargetRaiseView, 3150),
            (CaptureMode::MobileTargetMemoryView, 3150),
            (CaptureMode::MobileTargetRestoredView, 3150),
            (CaptureMode::MobileAuxiliaryView, 1050),
            (CaptureMode::MobileAuxiliaryApproach, 3150),
            (CaptureMode::MobileAuxiliaryRelease, 3150),
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

#[cfg(test)]
mod station_lighting_tests {
    use super::*;

    #[test]
    fn actual_station_lights_apply_explicit_settings_and_preserve_defaults() {
        for explicit in [false, true] {
            let mut app = App::new();
            app.insert_resource(super::super::g1_station_environment::G1StationSceneActive)
                .insert_resource(GlobalAmbientLight {
                    color: Color::srgb(0.78, 0.76, 0.85),
                    brightness: 450.,
                    ..default()
                })
                .insert_resource(DiagnosticIllumination {
                    ambient_brightness: 2328.,
                    directional_illuminance: 981.,
                    source_rect_lighting: None,
                    station_ambient_override: explicit.then_some(2328.),
                    station_directional_override: explicit.then_some(981.),
                    station_shadow_override: explicit.then_some(false),
                })
                .add_systems(PostStartup, configure_native_station_illumination);
            let light = app
                .world_mut()
                .spawn((
                    DirectionalLight {
                        illuminance: 10000.,
                        shadow_maps_enabled: true,
                        ..default()
                    },
                    Transform::default(),
                ))
                .id();
            app.update();
            let actual = app.world().get::<DirectionalLight>(light).unwrap();
            let ambient = app.world().resource::<GlobalAmbientLight>();
            let report = &app.world().resource::<NativeStationIllumination>().0;
            assert_eq!(actual.illuminance, if explicit { 981. } else { 10000. });
            assert_eq!(actual.shadow_maps_enabled, !explicit);
            assert_eq!(ambient.brightness, if explicit { 2328. } else { 450. });
            assert_eq!(
                report["directional_lights"][0]["illuminance"]
                    .as_f64()
                    .unwrap(),
                f64::from(actual.illuminance)
            );
            assert_eq!(
                report["ambient_brightness"].as_f64().unwrap(),
                f64::from(ambient.brightness)
            );
            assert_eq!(report["added_directional_lights"], 0);
            assert_eq!(report["explicit_comparison"], explicit);
            assert!(report.get("error").is_none());
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
    station: Option<Res<super::g1_station_environment::G1StationSceneActive>>,
) {
    if station.is_none() {
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
    }
    commands.spawn((
        Camera3d::default(),
        rendering_minigame::g1_uri_presentation::G1PresentationCamera,
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
    if let Some(target) = runtime.static_observed_place_target_tick {
        let latest = runtime
            .latest
            .as_ref()
            .ok_or("observed placement owner snapshot absent")?;
        if latest.phase == G1WorkerPhase::Failed {
            return Err(latest
                .reason
                .clone()
                .unwrap_or("observed placement owner failed".into()));
        }
        return Ok(
            latest.phase == G1WorkerPhase::Paused && latest.timing.episode_integrations == target
        );
    }
    #[cfg(feature = "g1_constraint_diagnostic")]
    if runtime.static_observed_grasp_route.is_some() {
        return static_observed_grasp::drive(runtime, outcome, port);
    }
    #[cfg(feature = "g1_constraint_diagnostic")]
    if runtime.static_transfer_route.is_some() {
        return static_visual_transfer::drive(runtime, outcome, port);
    }
    #[cfg(feature = "g1_constraint_diagnostic")]
    if runtime.continuous_route.is_some() {
        return mobile_continuous::drive(runtime, outcome, port);
    }
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
    if assist.station_motion && assist.station_hold_submitted && !assist.submitted {
        return station_carry::continue_after_hold(runtime, outcome, port);
    }
    if assist.submitted {
        let blocked_stop = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref()
            .is_some_and(|step| matches!(&step.execution, MobileAssistExecution::ClassicalCarry { navigation, .. } if navigation.blocked_stop_completed));
        if blocked_stop && assist.station_motion && !assist.auxiliary_release {
            return Err("station carry safely stopped after blocked progress; requested endpoint was not reached".into());
        }
        let scan_complete = latest.phase == G1WorkerPhase::Paused && latest.assist_step.as_ref()
            .is_some_and(|step| matches!(&step.execution, MobileAssistExecution::ClassicalScan { navigation, .. } if navigation.completed));
        if assist.auxiliary_approach && scan_complete && !assist.visual_goal_submitted {
            return drive_visual_approach(runtime, outcome, port);
        }
        if assist.auxiliary_view
            && !assist.auxiliary_approach
            && scan_complete
            && !assist.auxiliary_view_completed
        {
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
        let carry_complete = latest.phase == G1WorkerPhase::Paused
            && latest.assist_step.as_ref().is_some_and(|step| {
                station_carry::carry_boundary_completed(
                    &step.execution,
                    assist.submitted_carry_observation,
                )
            });
        if assist.auxiliary_release && carry_complete && !assist.fine_alignment_confirmed {
            return drive_auxiliary_fine_approach(runtime, outcome, port);
        }
        if assist.auxiliary_release
            && carry_complete
            && assist.fine_alignment_confirmed
            && !assist.hold_submitted
        {
            return drive_auxiliary_hold(runtime, outcome, port);
        }
        let release_boundary = latest.phase == G1WorkerPhase::Paused
            && latest.assist_step.as_ref().is_some_and(|s| {
                release_observation_boundary(assist.thumb_preparation_submitted, &s.execution)
            });
        if assist.auxiliary_release
            && release_boundary
            && assist.hold_submitted
            && !assist.release_submitted
        {
            return drive_auxiliary_release(runtime, outcome, port);
        }
        if assist.auxiliary_approach
            && !assist.auxiliary_release
            && carry_complete
            && !assist.auxiliary_view_completed
        {
            return drive_auxiliary_marker_view(runtime, outcome, port);
        }
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
            if assist.auxiliary_release {
                assist.release_submitted && matches!(&step.execution, MobileAssistExecution::ClassicalRelease { opening, .. } if opening.completed)
            } else if assist.auxiliary_view {
                assist.auxiliary_view_completed
                    && if assist.auxiliary_approach {
                        matches!(&step.execution, MobileAssistExecution::ClassicalCarry { navigation, .. } if navigation.completed)
                    } else {
                        matches!(&step.execution, MobileAssistExecution::ClassicalScan { navigation, .. } if navigation.completed)
                    }
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
        || frame.stamp.mount_profile != assist.camera_mount()
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
        MobileAssistStage::Carry(_) | MobileAssistStage::Scan(_) if assist.station_motion => {
            let goal = simulation_minigame::g1::mobile_hold::MobileHoldGoal { observation };
            (
                MobileAssistCommand::ClassicalGripSettle(goal.clone()),
                serde_json::to_value(goal).map_err(|e| e.to_string())?,
                "native_station_original_self_state_grip_calibration_and_minimum100Tick_hold",
                9_000_000_000,
                8,
            )
        }
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
    merge_mobile_handoff(
        &mut outcome.0.lock().unwrap().mobile_assist_handoff,
        serde_json::json!({
            "goal":goal,"goal_origin":origin,
            "original_vla_grasp_chunks":4,"actual_rgb_handoff":true,"traditional_grip_navigation":true,
            "object_truth_in_command":false,"task_qualified":false,"autonomous_goal_selection":false,
        }),
    )?;
    let assist = runtime.mobile_assist.as_mut().unwrap();
    if assist.station_motion {
        assist.station_hold_submitted = true;
    } else {
        assist.submitted = true;
    }
    runtime.requested = false;
    Ok(false)
}

/// Initial decisions can precede the physical grasp handoff. Updating grasp
/// fields must retain those original decision events and acquisition identities.
fn merge_mobile_handoff(
    handoff: &mut Option<serde_json::Value>,
    fields: serde_json::Value,
) -> Result<(), String> {
    let fields = fields
        .as_object()
        .ok_or("mobile handoff update must be an object")?;
    let current = handoff.get_or_insert_with(|| serde_json::json!({}));
    current
        .as_object_mut()
        .ok_or("mobile handoff receipt must be an object")?
        .extend(fields.clone());
    Ok(())
}

#[cfg(all(test, feature = "g1_constraint_diagnostic"))]
mod initial_handoff_tests {
    use super::*;
    #[test]
    fn zero_tick_decisions_survive_the_later_physical_grasp_handoff() {
        let mut handoff = None;
        merge_mobile_handoff(&mut handoff, serde_json::json!({})).unwrap();
        handoff.as_mut().unwrap()["qwen_postgrasp_transport"] = serde_json::json!({
            "events":[{"event":"initial_profile_decision_admitted","observation":{"episode_id":20645,"sim_time_ns":0,"frame_id":38}}]
        });
        let original = handoff.as_ref().unwrap()["qwen_postgrasp_transport"].clone();
        merge_mobile_handoff(&mut handoff,serde_json::json!({"goal":{"sim_time_ns":4_000_000_000_u64},"original_vla_grasp_chunks":4})).unwrap();
        assert_eq!(
            handoff.as_ref().unwrap()["qwen_postgrasp_transport"],
            original
        );
        assert_eq!(handoff.as_ref().unwrap()["original_vla_grasp_chunks"], 4);
    }
    #[test]
    fn malformed_handoff_receipt_is_rejected_before_mutation() {
        let mut handoff = Some(serde_json::json!([]));
        assert!(merge_mobile_handoff(&mut handoff, serde_json::json!({})).is_err());
        assert_eq!(handoff, Some(serde_json::json!([])));
    }
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_auxiliary_fine_approach(
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
        .ok_or("fine approach lacks owner")?
        .clone();
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("fine approach stage absent")?;
    if latest.phase != G1WorkerPhase::Paused || assist.fine_alignment_confirmed {
        return Err("fine approach requires a new completed carry boundary".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        let observation = job.observation;
        if observation.episode_id != runtime.episode_id
            || observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
            || reply["camera_mount_profile"] != assist.auxiliary_camera_name()
            || reply["clearance_proposal"]["required_raise_m"]
                .as_f64()
                .is_none_or(|r| r > 0.)
        {
            return Err(
                "fine approach lacks current auxiliary RGB/rim clearance; owner remains paused"
                    .into(),
            );
        }
        let state = reply["fine_approach_proposal"]["state"]
            .as_str()
            .ok_or("current fine containment state unavailable")?;
        let aligned = state == "aligned";
        let blocked_stop = latest.assist_step.as_ref().is_some_and(|step|
            matches!(&step.execution, simulation_minigame::g1::mobile_assist::MobileAssistExecution::ClassicalCarry { navigation, .. }
                if navigation.blocked_stop_completed));
        if blocked_stop && !aligned {
            return Err("fresh RGB after protected blocked stop requires replan; further forward motion is not admitted".into());
        }
        let mut goal = None;
        if !aligned {
            if state != "advance" || assist.fine_goals_submitted >= 5 {
                return Err("current geometry cannot admit another bounded fine segment; owner remains paused".into());
            }
            let parsed: MobileCarryGoal =
                serde_json::from_value(reply["fine_approach_proposal"]["goal"].clone())
                    .map_err(|e| format!("current short fine step unavailable: {e}"))?;
            parsed.validate().map_err(|e| e.to_string())?;
            let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
                return Err("fine approach lost sole native owner".into());
            };
            owner
                .submit(TimedCommand {
                    episode_id: runtime.episode_id,
                    valid_until_sim_ns: observation.sim_time_ns + 8_000_000_000,
                    valid_until_wall: Instant::now() + Duration::from_secs(10),
                    command: MobileAssistCommand::ClassicalCarry(parsed.clone()),
                })
                .map_err(|e| e.to_string())?;
            goal = Some(parsed);
        }
        let record = serde_json::json!({
                "actual_localization":reply,"executed_goal":goal,"whole_object_containment_interval":true,
                "alignment_confirmed":aligned,"new_image_before_each_segment":true,
                "protected_blocked_stop_reobserved":blocked_stop,
            "same_owner_boundary_tick":latest.timing.episode_integrations,
            "localization_wall_ms":job.started.elapsed().as_secs_f64()*1000.,
            "object_truth_in_command":false,"task_qualified":false,
        });
        {
            let mut receipt = outcome.0.lock().unwrap();
            let handoff = receipt
                .mobile_assist_handoff
                .as_mut()
                .ok_or("fine provenance absent")?;
            if assist.fine_goals_submitted == 0 {
                handoff["visual_fine_approach"] = record.clone();
            }
            let object = handoff
                .as_object_mut()
                .ok_or("fine provenance not an object")?;
            object
                .entry("visual_fine_steps")
                .or_insert_with(|| serde_json::json!([]))
                .as_array_mut()
                .ok_or("fine step provenance not an array")?
                .push(record);
        }
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.fine_alignment_confirmed = aligned;
        if !aligned {
            assist.fine_goals_submitted += 1;
            assist.submitted_carry_observation = Some(observation);
        }
        runtime.requested = false;
        return Ok(false);
    }
    let directory = if assist.fine_goals_submitted == 0 {
        "visual_fine_approach".to_string()
    } else {
        format!(
            "visual_fine_approach_{:02}",
            assist.fine_goals_submitted + 1
        )
    };
    start_marker_job(runtime, port, &directory)?;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_auxiliary_hold(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::MobileAssistCommand, mobile_hold::MobileHoldGoal, worker::TimedCommand,
    };
    let Some((observation, _)) = capture_current_marker_frame(runtime, port, "stationary_hold")?
    else {
        return Ok(false);
    };
    let goal = MobileHoldGoal { observation };
    goal.validate().map_err(|e| e.to_string())?;
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("standing hold lost sole owner".into());
    };
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: observation.sim_time_ns + 6_000_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(8),
            command: MobileAssistCommand::ClassicalHold(goal.clone()),
        })
        .map_err(|e| e.to_string())?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .ok_or("hold provenance absent")?["stationary_hold"] = serde_json::json!({
        "executed_goal":goal,"same_owner_boundary_tick":observation.sim_time_ns/20_000_000,
        "maximum_ticks":250,"minimum_ticks":100,"consecutive_self_speed_samples":20,
        "maximum_stable_self_speed_m_s":0.03,"object_truth_in_command":false,"task_qualified":false,
    });
    runtime.mobile_assist.as_mut().unwrap().hold_submitted = true;
    runtime.requested = false;
    Ok(false)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn release_observation_boundary(
    thumb_submitted: bool,
    execution: &simulation_minigame::g1::mobile_assist::MobileAssistExecution,
) -> bool {
    use simulation_minigame::g1::mobile_assist::MobileAssistExecution;
    if thumb_submitted {
        matches!(execution, MobileAssistExecution::ClassicalThumbClearance { preparing, .. } if preparing.completed)
    } else {
        matches!(execution, MobileAssistExecution::ClassicalHold { holding, .. } if holding.completed)
    }
}

#[cfg(all(test, feature = "g1_constraint_diagnostic"))]
mod release_boundary_tests {
    use super::release_observation_boundary;
    use robot_minigame::g1::contract::G1Command;
    use simulation_minigame::g1::{
        mobile_assist::MobileAssistExecution,
        mobile_hold::{MobileHoldGoal, MobileHoldStep},
        mobile_thumb::{MobileThumbGoal, MobileThumbStep},
    };
    use task_minigame::types::ObservationStamp;
    #[test]
    fn pending_thumb_command_cannot_recapture_the_completed_old_hold() {
        let observation = ObservationStamp {
            episode_id: 20642,
            frame_id: 14,
            sim_time_ns: 2182 * 20_000_000,
            captured_at_unix_ms: 1,
        };
        let old = MobileAssistExecution::ClassicalHold {
            goal: MobileHoldGoal { observation },
            holding: MobileHoldStep {
                command: G1Command::default(),
                holding_ticks: 100,
                stable_velocity_ticks: 20,
                self_speed_m_s: 0.,
                completed: true,
            },
        };
        assert!(release_observation_boundary(false, &old));
        // Native0442 copied2182 immediately after submitting the next100Tick
        // command; the owner mailbox had not published its new phase yet.
        assert!(!release_observation_boundary(true, &old));
        let mut next = MobileAssistExecution::ClassicalThumbClearance {
            goal: MobileThumbGoal {
                observation,
                bounded_thumb_targets_rad: None,
            },
            preparing: MobileThumbStep {
                command: G1Command::default(),
                preparation_ticks: 100,
                maximum_measured_tracking_error_rad: 0.,
                completed: true,
                fresh_release_observation_required: true,
            },
        };
        assert!(release_observation_boundary(true, &next));
        assert!(!release_observation_boundary(false, &next));
        if let MobileAssistExecution::ClassicalThumbClearance { preparing, .. } = &mut next {
            preparing.completed = false;
        }
        assert!(!release_observation_boundary(true, &next));
    }
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_auxiliary_release(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::MobileAssistCommand, mobile_release::MobileReleaseGoal,
        mobile_thumb::MobileThumbGoal, worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("release lacks owner")?
        .clone();
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("release stage absent")?;
    if latest.phase != G1WorkerPhase::Paused
        || assist.release_submitted
        || !assist.fine_alignment_confirmed
    {
        return Err("release requires its completed fine approach boundary".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        let observation = job.observation;
        // Store both admitted and rejected alignment images before returning.
        outcome
            .0
            .lock()
            .unwrap()
            .mobile_assist_handoff
            .as_mut()
            .ok_or("release provenance absent")?["visual_release_alignment"] = serde_json::json!({
            "actual_localization":reply,"same_owner_boundary_tick":latest.timing.episode_integrations,
            "localization_wall_ms":job.started.elapsed().as_secs_f64()*1000.,
            "object_truth_in_command":false,"release_executed":false,"task_qualified":false,
        });
        if observation.episode_id != runtime.episode_id
            || observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
            || reply["camera_mount_profile"] != assist.auxiliary_camera_name()
        {
            return Err("release image detached from the current owner/camera".into());
        }
        if reply["release_proposal"]["release_admitted"] != true {
            if assist.station_motion
                && assist.station_placement_view_active
                && !assist.thumb_preparation_submitted
                && reply["thumb_preparation_proposal"]["preparation_admitted"] == true
            {
                let goal: MobileThumbGoal = serde_json::from_value(
                    reply["thumb_preparation_proposal"]["preparation_goal"].clone(),
                )
                .map_err(|e| e.to_string())?;
                goal.validate().map_err(|e| e.to_string())?;
                let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
                    return Err("thumb preparation lost sole native owner".into());
                };
                owner
                    .submit(TimedCommand {
                        episode_id: runtime.episode_id,
                        valid_until_sim_ns: observation.sim_time_ns + 4_000_000_000,
                        valid_until_wall: Instant::now() + Duration::from_secs(6),
                        command: MobileAssistCommand::ClassicalThumbClearance(goal.clone()),
                    })
                    .map_err(|e| e.to_string())?;
                outcome
                    .0
                    .lock()
                    .unwrap()
                    .mobile_assist_handoff
                    .as_mut()
                    .unwrap()["visual_thumb_preparation"] = serde_json::json!({
                    "actual_localization":reply,"executed_goal":goal,
                    "same_owner_boundary_tick":latest.timing.episode_integrations,
                    "object_truth_in_command":false,"release_executed":false,
                    "fresh_release_observation_required":true,"task_qualified":false,
                });
                let assist = runtime.mobile_assist.as_mut().unwrap();
                assist.vision_job.take();
                assist.thumb_preparation_submitted = true;
                runtime.requested = false;
                return Ok(false);
            }
            return Err(
                "current visual box footprint/release envelope rejected; owner remains paused"
                    .into(),
            );
        }
        let goal: MobileReleaseGoal =
            serde_json::from_value(reply["release_proposal"]["release_goal"].clone())
                .map_err(|e| e.to_string())?;
        goal.validate().map_err(|e| e.to_string())?;
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("release lost sole native owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: observation.sim_time_ns + 10_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(12),
                command: MobileAssistCommand::ClassicalRelease(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        let alignment =
            &mut receipt.mobile_assist_handoff.as_mut().unwrap()["visual_release_alignment"];
        alignment["release_executed"] = true.into();
        alignment["executed_goal"] = serde_json::to_value(goal).map_err(|e| e.to_string())?;
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job.take();
        assist.release_submitted = true;
        runtime.requested = false;
        return Ok(false);
    }
    let directory = if assist.thumb_preparation_submitted {
        "visual_release_alignment_after_thumb"
    } else {
        "visual_release_alignment"
    };
    start_marker_job(runtime, port, directory)?;
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
        return Err("auxiliary view requires a completed stationary native boundary".into());
    }
    if let Some(job) = &assist.vision_job {
        let Some(reply) = job.try_take() else {
            return Ok(false);
        };
        let reply = reply?;
        if job.observation.episode_id != runtime.episode_id
            || job.observation.sim_time_ns != latest.timing.episode_integrations * 20_000_000
            || reply["camera_mount_profile"] != "auxiliary_grip_overview"
            || !matches!(
                reply["marker_layout_profile"].as_str(),
                Some("auxiliary_grip_targets" | "auxiliary_bin_board_targets")
            )
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
            "coarse_approach_executed":assist.auxiliary_approach,
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
    if let Some(dispatch) = runtime.qwen_dispatch.take() {
        let result = {
            let mut guard = dispatch.lock().map_err(|_| "Qwen dispatch poisoned")?;
            let result = guard.transport(runtime, outcome, port);
            if let Err(reason) = &result {
                guard.record_failure(outcome, reason)?;
            }
            result
        };
        runtime.qwen_dispatch = Some(dispatch);
        return result;
    }
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
        return Err(
            "visual approach requires its completed stationary observation boundary".into(),
        );
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
        assist.submitted_carry_observation = Some(observation);
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
        port.request_physics_frame(runtime.episode_id, latest.timing.episode_integrations)?;
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
            != runtime
                .mobile_assist
                .as_ref()
                .ok_or("mobile camera stage absent")?
                .camera_mount()
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
fn capture_current_marker_pair(
    runtime: &mut CaptureRuntime,
    port: &G1CameraPort,
    stage: &str,
) -> Result<Option<(ObservationStamp, [f32; 4])>, String> {
    if !(stage == "visual_approach"
        || stage.starts_with("visual_fine_approach")
        || stage == "visual_release_alignment"
        || stage == "visual_release_alignment_after_thumb")
    {
        return Err("box pair is limited to current auxiliary approach/release boundaries".into());
    }
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("box pair stage absent")?;
    if let Some(pair) = &assist.pending_box_pair {
        if pair.stage != stage || pair.started.elapsed() > Duration::from_secs(2) {
            return Err("box pair capture stage changed or timed out; owner remains paused".into());
        }
        let Some(_) = capture_current_marker_frame(runtime, port, &format!("{stage}/secondary"))?
        else {
            return Ok(None);
        };
        let directory = runtime.options.output.join(stage);
        let first = serde_json::from_slice(
            &fs::read(directory.join("observation.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let second = serde_json::from_slice(
            &fs::read(directory.join("secondary/observation.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        super::g1_marker_vision::validate_mobile_pair_inputs(&first, &second)?;
        let assist = runtime.mobile_assist.as_mut().unwrap();
        let pair = assist.pending_box_pair.take().unwrap();
        assist.box_pair_restore_mount = Some(pair.primary_mount);
        runtime.requested = false;
        return Ok(Some((pair.observation, pair.rotation)));
    }
    let primary_mount = assist.camera_mount();
    let secondary_mount = match primary_mount {
        G1CameraMountProfile::AuxiliaryGripOverview => G1CameraMountProfile::AuxiliaryBinPlacement,
        G1CameraMountProfile::AuxiliaryBinPlacement => G1CameraMountProfile::AuxiliaryGripOverview,
        G1CameraMountProfile::ArenaEgo | G1CameraMountProfile::StaticPlacementOverview => {
            return Err("box pair cannot change the learned camera or static task profile".into());
        }
    };
    let Some((observation, rotation)) = capture_current_marker_frame(runtime, port, stage)? else {
        return Ok(None);
    };
    runtime.mobile_assist.as_mut().unwrap().pending_box_pair = Some(PendingMobileBoxPair {
        stage: stage.to_owned(),
        observation,
        rotation,
        primary_mount,
        secondary_mount,
        started: Instant::now(),
    });
    runtime.requested = false;
    Ok(None)
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
    let captured = if config.same_tick_box_pair {
        capture_current_marker_pair(runtime, port, stage_directory)?
    } else {
        capture_current_marker_frame(runtime, port, stage_directory)?
    };
    let Some((observation, rotation)) = captured else {
        return Ok(());
    };
    let directory = runtime.options.output.join(stage_directory);
    let assist = runtime
        .mobile_assist
        .as_ref()
        .ok_or("marker job stage absent")?;
    if matches!(
        stage_directory,
        "visual_release_alignment" | "visual_release_alignment_after_thumb"
    ) {
        runtime.mobile_assist.as_mut().unwrap().vision_job = Some(
            MarkerVisionJob::start_placement_view(config, directory, observation)?,
        );
        return Ok(());
    }
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
    match stamp.mount_profile {
        G1CameraMountProfile::AuxiliaryGripOverview => {
            observation["camera_mount_profile"] = "auxiliary_grip_overview".into();
        }
        G1CameraMountProfile::AuxiliaryBinPlacement => {
            observation["camera_mount_profile"] = "auxiliary_bin_placement".into();
        }
        G1CameraMountProfile::ArenaEgo | G1CameraMountProfile::StaticPlacementOverview => {}
    }
    Ok(observation)
}

/// Shared explicit initialization; no task-model call or hidden physics step.
fn drive_static_startup(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
) -> Result<bool, String> {
    #[cfg(feature = "g1_constraint_diagnostic")]
    if runtime.startup_ticks != 0 {
        use simulation_minigame::g1::static_startup::{
            StaticStartupCommand, StaticStartupExecution,
        };
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("explicit startup lost its unique owner".into());
        };
        let latest = runtime
            .latest
            .as_ref()
            .ok_or("missing startup owner snapshot")?;
        if latest.phase == G1WorkerPhase::Failed {
            return Err(latest
                .reason
                .clone()
                .unwrap_or("startup physics failed".into()));
        }
        if !runtime.startup_submitted {
            if latest.phase != G1WorkerPhase::Paused || latest.timing.episode_integrations != 0 {
                return Ok(false);
            }
            worker
                .submit(simulation_minigame::g1::worker::TimedCommand {
                    episode_id: runtime.episode_id,
                    valid_until_sim_ns: runtime.startup_ticks * ARENA_ACTION_PERIOD_NS,
                    valid_until_wall: Instant::now() + Duration::from_secs(8),
                    command: StaticStartupCommand::Startup {
                        episode_id: runtime.episode_id,
                    },
                })
                .map_err(|e| e.to_string())?;
            runtime.startup_submitted = true;
            return Ok(false);
        }
        if latest.timing.episode_integrations < runtime.startup_ticks {
            return Ok(false);
        }
        if latest.timing.episode_integrations == runtime.startup_ticks {
            let step = latest
                .startup_step
                .as_ref()
                .ok_or("missing completed startup evidence")?;
            let StaticStartupExecution::Startup {
                ticks,
                stable_self_velocity_ticks,
                completed,
                ready,
                ..
            } = &step.execution
            else {
                return Err("startup boundary contains unexpected task execution".into());
            };
            if *ticks != runtime.startup_ticks || !completed || !ready {
                return Err(
                    "finite static startup did not establish its self-state gate; reset required"
                        .into(),
                );
            }
            outcome.0.lock().unwrap().static_startup = Some(serde_json::json!({
                "requested_real_ticks":runtime.startup_ticks,"actual_real_ticks":ticks,
                "stable_self_velocity_ticks":stable_self_velocity_ticks,"completed":completed,"ready":ready,
                "same_owner_world":true,"self_velocity_and_imu_only":true,
                "first_task_observation_earliest_sim_time_ns":runtime.startup_ticks*ARENA_ACTION_PERIOD_NS,
                "original_vla_output":false,"task_qualified":false,
            }));
        }
    }
    Ok(true)
}

fn drive_live_policy(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    if !drive_static_startup(runtime, outcome)? {
        return Ok(false);
    }
    if runtime
        .live_policy
        .as_ref()
        .is_some_and(|live| live.wait_after_chunks)
    {
        return drive_waited_mobile_policy(runtime, outcome, port);
    }
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
        let timed = TimedArenaTaskCommand {
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
        };
        if runtime.static_regrasp_started {
            #[cfg(feature = "g1_constraint_diagnostic")]
            {
                let CaptureWorker::StaticStartup(owner) = &runtime.worker else {
                    return Err("finite regrasp selected a foreign physical owner".into());
                };
                owner.submit(simulation_minigame::g1::worker::TimedCommand {
                    episode_id:timed.episode_id,
                    valid_until_sim_ns:timed.valid_until_sim_ns,
                    valid_until_wall:timed.valid_until_wall,
                    command:simulation_minigame::g1::static_startup::StaticStartupCommand::OriginalRegrasp(timed.command),
                }).map_err(|e|e.to_string())?;
            }
            #[cfg(not(feature = "g1_constraint_diagnostic"))]
            return Err("finite regrasp requires the separate diagnostic feature".into());
        } else {
            runtime
                .worker
                .submit_live_chunk(timed)
                .map_err(|e| e.to_string())?;
        }
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

#[cfg(not(feature = "g1_constraint_diagnostic"))]
fn drive_waited_mobile_policy(
    _: &mut CaptureRuntime,
    _: &CaptureOutcome,
    _: &G1CameraPort,
) -> Result<bool, String> {
    Err("waited mobile grasp requires its explicit diagnostic feature".into())
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn drive_waited_mobile_policy(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    use simulation_minigame::g1::{
        mobile_assist::{MobileAssistCommand, MobileAssistExecution},
        worker::TimedCommand,
    };
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("waited grasp owner snapshot absent")?;
    let live = runtime
        .live_policy
        .as_mut()
        .ok_or("waited grasp model worker absent")?;
    let tick = latest.timing.episode_integrations;
    if tick > u64::from(runtime.options.ticks) || latest.phase == G1WorkerPhase::Failed {
        return Err(latest
            .reason
            .clone()
            .unwrap_or("waited grasp owner failed or exceeded its total budget".into()));
    }
    let waiting = latest
        .assist_step
        .as_ref()
        .and_then(|step| match &step.execution {
            MobileAssistExecution::ClassicalModelWait { goal, waiting } => Some((goal, waiting)),
            _ => None,
        });
    if tick > 0 && latest.phase == G1WorkerPhase::Paused {
        if live.submitted_chunks == 4
            && waiting.is_some_and(|(g, w)| g.request_id == 4 && w.completed)
        {
            live.completed = true;
            return Ok(true);
        }
        return Err(
            "finite standing wait expired before replacement; owner explicitly paused".into(),
        );
    }
    let reply_boundary = live.submitted_chunks == 0
        || waiting.is_some_and(|(g, w)| {
            g.request_id == u64::from(live.submitted_chunks)
                && !w.completed
                && (live.boundary_images_with_wait || w.observation_ready)
        });
    let reply = if reply_boundary {
        live.worker.try_take_reply()
    } else {
        None
    };
    if let Some(reply) = reply {
        let initial = live.submitted_chunks == 0;
        if live.pending.take() != Some(reply.observation)
            || reply.profile != TaskProfile::MobileBox
            || reply.sequence_id + 1 != live.next_sequence
            || (initial && (tick != 0 || latest.phase != G1WorkerPhase::Paused))
            || (!initial
                && !waiting.is_some_and(|(g, w)| {
                    g.request_id == u64::from(live.submitted_chunks)
                        && (live.boundary_images_with_wait || w.observation_ready)
                        && !w.completed
                }))
        {
            return Err(
                "waited original reply is foreign or no longer in a stationary waiting interval"
                    .into(),
            );
        }
        let chunk = reply
            .result
            .map_err(|e| format!("waited original policy: {e:?}"))?;
        if chunk.frames.len() != 50 {
            return Err("waited original decoder changed its50-frame horizon".into());
        }
        fs::write(
            runtime
                .options
                .output
                .join(format!("live_reply_{:04}.json", reply.sequence_id)),
            serde_json::to_vec(&chunk).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("waited grasp lost its unique native owner".into());
        };
        let original = ArenaTaskCommand {
            chunk: Arc::new(chunk),
            scheduled_start_sim_ns: None,
        };
        let command = if live.boundary_images_with_wait {
            MobileAssistCommand::OriginalVlaBoundaryImageThenWait(original)
        } else {
            MobileAssistCommand::OriginalVlaThenWait(original)
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: u64::from(runtime.options.ticks) * ARENA_ACTION_PERIOD_NS,
                valid_until_wall: Instant::now() + Duration::from_secs(6),
                command,
            })
            .map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        receipt.live_policy_successes += 1;
        receipt.live_action_chunks.push(serde_json::json!({
            "sequence_id":reply.sequence_id,"observation":reply.observation,"inference_ms":reply.elapsed.as_millis(),
            "arrival_display_tick":tick,"execution_start":"recorded_by_actual_owner_step",
            "frame_count":50,"original_frame_interval_ns":ARENA_ACTION_PERIOD_NS,
            "following_classical_wait_maximum_ticks":200,"observation_restamped":false,
            "future_slot_rebased":false,"same_owner_world":true,"task_qualified":false,
            "completed_predecessor_boundary_image":live.boundary_images_with_wait && !initial,
        }));
        live.submitted_chunks += 1;
        return Ok(false);
    }
    if live.pending.is_some() || live.submitted_chunks == 4 {
        return Ok(false);
    }
    let initial = live.submitted_chunks == 0;
    if !initial
        && !live.boundary_images_with_wait
        && !waiting.is_some_and(|(g, w)| {
            g.request_id == u64::from(live.submitted_chunks) && w.observation_ready && !w.completed
        })
    {
        return Ok(false);
    }
    if !runtime.requested {
        let image_min_tick = if !initial && live.boundary_images_with_wait {
            let Some(MobileAssistExecution::OriginalVla(execution)) =
                latest.assist_step.as_ref().map(|s| &s.execution)
            else {
                return Ok(false);
            };
            if execution.sequence_id != u64::from(live.submitted_chunks) {
                return Ok(false);
            }
            let start = execution.execution_start_sim_ns / ARENA_ACTION_PERIOD_NS;
            live.boundary_image_window = Some([start + 50, start + 75]);
            start + 50
        } else {
            tick
        };
        let sequence = port.request_physics_frame(runtime.episode_id, image_min_tick)?;
        live.wait_image_min_tick = Some(image_min_tick);
        runtime.requested = true;
        let mut receipt = outcome.0.lock().unwrap();
        if receipt.prefetch_image_events.len() >= 16 {
            return Err("waited image event budget exhausted".into());
        }
        receipt.prefetch_image_events.push(serde_json::json!({"event":"standing_wait_request","capture_sequence":sequence,"minimum_physics_tick":image_min_tick,"completed_predecessor_image_window":live.boundary_image_window,"submitted_chunks":live.submitted_chunks,"runtime_elapsed_ms":runtime.started.elapsed().as_millis()}));
        return Ok(false);
    }
    if let Some(frame) = port.take() {
        let frame = frame?;
        let frame_tick = frame.stamp.source_ticks[0];
        if frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.source_ticks != [frame_tick; 2]
            || frame.stamp.sim_time_ns != frame_tick * ARENA_ACTION_PERIOD_NS
            || frame_tick
                < live
                    .wait_image_min_tick
                    .ok_or("standing image minimum boundary absent")?
            || frame_tick > tick
            || (initial && frame_tick != 0)
            || (!initial
                && live.boundary_images_with_wait
                && live
                    .boundary_image_window
                    .is_none_or(|[start, end]| frame_tick < start || frame_tick >= end))
        {
            return Err(
                "standing RGB/self stamp is foreign, pipelined old or future; no restamping".into(),
            );
        }
        {
            let mut receipt = outcome.0.lock().unwrap();
            if receipt.prefetch_image_events.len() >= 16 {
                return Err("waited image event budget exhausted".into());
            }
            receipt.prefetch_image_events.push(serde_json::json!({"event":"standing_wait_readback_consumed","capture_sequence":frame.stamp.capture_sequence,"actual_image_tick":frame_tick,"display_tick":tick,"captured_at_unix_ms":frame.stamp.captured_at_unix_ms,"copy_encoded_at_unix_ms":frame.stamp.copy_encoded_at_unix_ms,"readback_completed_at_unix_ms":frame.stamp.readback_completed_at_unix_ms}));
        }
        submit_live_observation(live, frame, &runtime.options.output)?;
        runtime.requested = false;
        live.wait_image_min_tick = None;
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
    let initial_tick = runtime.startup_ticks;
    let horizon = profile_contract(live.profile).action_horizon as u64;
    if latest.phase == G1WorkerPhase::Failed {
        return Err(latest
            .reason
            .clone()
            .unwrap_or("prefetch physical owner failed".into()));
    }
    if tick > initial_tick && latest.phase == G1WorkerPhase::Paused {
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
            || (initial && (tick != initial_tick || latest.phase != G1WorkerPhase::Paused))
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
        initial_tick
    } else {
        live.next_boundary_tick - horizon + u64::from(live.prefetch_after_ticks.unwrap())
    };
    if !initial && tick >= live.next_boundary_tick {
        return Err("prefetch image missed replacement window; explicit pause required".into());
    }
    if !runtime.requested {
        let sequence = port.request_physics_frame(runtime.episode_id, trigger_tick)?;
        let mut receipt = outcome.0.lock().unwrap();
        if receipt.prefetch_image_events.len() >= 40 {
            return Err("bounded prefetch image event budget exhausted".into());
        }
        receipt.prefetch_image_events.push(serde_json::json!({
            "event": "request", "capture_sequence": sequence,
            "display_tick": tick, "minimum_physics_tick": trigger_tick,
            "fixed_action_start_tick": live.next_boundary_tick,
            "runtime_elapsed_ms": runtime.started.elapsed().as_millis(),
        }));
        runtime.requested = true;
        return Ok(false);
    }
    if tick < trigger_tick {
        return Ok(false);
    }
    if let Some(frame) = port.take() {
        let frame = frame?;
        let frame_tick = frame.stamp.source_ticks[0];
        let mut receipt = outcome.0.lock().unwrap();
        if receipt.prefetch_image_events.len() >= 40 {
            return Err("bounded prefetch image event budget exhausted".into());
        }
        receipt.prefetch_image_events.push(serde_json::json!({
            "event": "readback_consumed", "capture_sequence": frame.stamp.capture_sequence,
            "display_tick": tick, "actual_image_tick": frame_tick,
            "minimum_physics_tick": trigger_tick,
            "fixed_action_start_tick": live.next_boundary_tick,
            "captured_at_unix_ms": frame.stamp.captured_at_unix_ms,
            "copy_encoded_at_unix_ms": frame.stamp.copy_encoded_at_unix_ms,
            "readback_completed_at_unix_ms": frame.stamp.readback_completed_at_unix_ms,
            "runtime_elapsed_ms": runtime.started.elapsed().as_millis(),
        }));
        drop(receipt);
        if frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.source_ticks != [frame_tick; 2]
            || frame.stamp.sim_time_ns != frame_tick * ARENA_ACTION_PERIOD_NS
            || (initial && frame_tick != initial_tick)
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
    (outcome, station_illumination): (Res<CaptureOutcome>, Option<Res<NativeStationIllumination>>),
    health: Res<StationRenderHealth>,
    visual: Res<G1VisualStatus>,
    task_model: Option<Res<G1TaskVisualModel>>,
    task_status: Option<Res<G1TaskVisualStatus>>,
    mut task_input: ResMut<G1TaskVisualInput>,
    mut input: ResMut<G1BodyObservationInput>,
    (mut camera_mount, mut marker_gate): (
        ResMut<G1ActiveCameraMount>,
        Option<ResMut<G1TaskFiducialGate>>,
    ),
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
        if let Some(worker) = &mut runtime.static_marker_worker {
            if let Some(localization) = worker.poll()? {
                let tick = localization["localization"]["observation"]["sim_time_ns"]
                    .as_u64()
                    .ok_or("static localization time absent")?
                    / 20_000_000;
                if (runtime.static_observed_grasp && tick != 100)
                    || ((runtime.static_memory_observe || runtime.static_unheld_regrasp)
                        && tick == 190)
                {
                    outcome.0.lock().unwrap().static_grip_localization = Some(localization);
                } else {
                    outcome.0.lock().unwrap().static_marker_localization = Some(localization);
                }
            }
        }
        if let Some(lighting) = &station_illumination {
            if let Some(error) = lighting.0.get("error").and_then(|e| e.as_str()) {
                return Err(error.into());
            }
            let mut receipt = outcome.0.lock().unwrap();
            receipt.native_station_illumination = Some(lighting.0.clone());
            receipt.camera_ambient_brightness = lighting.0["ambient_brightness"]
                .as_f64()
                .ok_or("missing effective ambient light")?
                as f32;
            receipt.camera_directional_illuminance =
                lighting.0["directional_lights"][0]["illuminance"]
                    .as_f64()
                    .ok_or("missing effective directional light")? as f32;
            receipt.diagnostic_directional_shadow_maps =
                lighting.0["directional_lights"][0]["shadow_maps_enabled"]
                    .as_bool()
                    .ok_or("missing effective shadow setting")?;
        }
        if let Some(evidence) = &mut runtime.owner_evidence {
            evidence.drain(&mut outcome.0.lock().unwrap())?;
        }
        if let Some(evidence) = &mut runtime.mobile_stand_evidence {
            evidence.drain(&mut outcome.0.lock().unwrap())?;
        }
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(evidence) = &mut runtime.assist_evidence {
            evidence.drain(&mut outcome.0.lock().unwrap())?;
        }
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(evidence) = &mut runtime.startup_evidence {
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
        #[cfg(feature = "g1_constraint_diagnostic")]
        let box_pair_render_frame = runtime.render_frames;
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(assist) = &mut runtime.mobile_assist {
            let desired = assist
                .pending_box_pair
                .as_ref()
                .map(|pair| pair.secondary_mount)
                .or(assist.box_pair_restore_mount);
            if let Some(desired) = desired {
                if latest.phase != G1WorkerPhase::Paused {
                    return Err("box pair camera switch requires the same paused owner Tick".into());
                }
                if camera_mount.0 != desired {
                    camera_mount.0 = desired;
                    assist.box_pair_camera_activation_frame = Some(box_pair_render_frame);
                    return Ok(());
                }
                if assist
                    .box_pair_camera_activation_frame
                    .is_some_and(|n| box_pair_render_frame < n + 2)
                {
                    return Ok(());
                }
                assist.box_pair_camera_activation_frame = None;
                assist.box_pair_restore_mount = None;
            }
        }
        #[cfg(feature = "g1_constraint_diagnostic")]
        if runtime
            .continuous_route
            .as_ref()
            .is_some_and(|r| r.needs_auxiliary_view())
            && camera_mount.0 == G1CameraMountProfile::ArenaEgo
        {
            camera_mount.0 = G1CameraMountProfile::AuxiliaryGripOverview;
            marker_gate
                .as_mut()
                .ok_or("continuous logical marker gate absent")?
                .enabled = true;
            if let Some(handoff) = &mut outcome.0.lock().unwrap().mobile_assist_handoff {
                handoff["markers_activated_at_actual_display_tick"] =
                    latest.timing.episode_integrations.into();
            }
            return Ok(());
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
        #[cfg(feature = "g1_constraint_diagnostic")]
        if runtime.mobile_assist.as_ref().is_some_and(|assist|
            assist.station_motion && assist.auxiliary_release && !assist.station_placement_view_active)
            && latest.phase == G1WorkerPhase::Paused
            && latest.assist_step.as_ref().is_some_and(|step|
                matches!(&step.execution, simulation_minigame::g1::mobile_assist::MobileAssistExecution::ClassicalCarry { navigation, .. }
                    if navigation.blocked_stop_completed))
        {
            if runtime.requested || runtime.mobile_assist.as_ref().is_some_and(|a| a.vision_job.is_some()) {
                return Err("placement camera switch requires an idle completed RGB boundary".into());
            }
            camera_mount.0 = G1CameraMountProfile::AuxiliaryBinPlacement;
            runtime.mobile_assist.as_mut().unwrap().station_placement_view_active = true;
            outcome.0.lock().unwrap().mobile_assist_handoff.as_mut()
                .ok_or("station placement camera provenance absent")?["placement_camera_activation"] = serde_json::json!({
                    "source_tick":latest.timing.episode_integrations,"camera_mount_profile":"auxiliary_bin_placement",
                    "head_height_from_arena_m":0.23,"upward_optical_pitch_from_arena_degrees":7,
                    "original_learned_camera_unchanged":true,"world_or_contact_truth_input":false,
                    "physics_paused":true,"task_qualified":false,
                });
            return Ok(());
        }
        if runtime.static_observed_grasp
            && latest.timing.episode_integrations == 100
            && latest.phase == G1WorkerPhase::Paused
            && runtime.static_marker_activation_render_frame.is_none()
        {
            marker_gate
                .as_mut()
                .ok_or("observed grasp marker gate absent")?
                .enabled = true;
            runtime.static_marker_activation_render_frame = Some(runtime.render_frames);
            return Ok(());
        }
        if runtime.static_visual_transfer
            && latest.timing.episode_integrations == 140
            && latest.phase == G1WorkerPhase::Paused
            && runtime.static_marker_activation_render_frame.is_none()
        {
            marker_gate
                .as_mut()
                .ok_or("static transfer marker gate absent")?
                .enabled = true;
            runtime.static_marker_activation_render_frame = Some(runtime.render_frames);
            return Ok(());
        }
        if runtime.static_transfer_auxiliary
            && latest.timing.episode_integrations == 390
            && latest.phase == G1WorkerPhase::Paused
            && camera_mount.0 == G1CameraMountProfile::ArenaEgo
        {
            camera_mount.0 = if runtime.static_transfer_placement {
                G1CameraMountProfile::StaticPlacementOverview
            } else {
                G1CameraMountProfile::AuxiliaryGripOverview
            };
            runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
            return Ok(());
        }
        if (runtime.static_memory_observe || runtime.static_unheld_regrasp)
            && !runtime.static_regrasp_started
            && latest.timing.episode_integrations == 190
            && latest.phase == G1WorkerPhase::Paused
            && camera_mount.0 == G1CameraMountProfile::ArenaEgo
        {
            camera_mount.0 = G1CameraMountProfile::StaticPlacementOverview;
            runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
            return Ok(());
        }
        if runtime.static_unheld_regrasp
            && runtime.static_regrasp_started
            && latest.phase == G1WorkerPhase::Paused
        {
            let gate = marker_gate.as_mut().ok_or("regrasp marker gate absent")?;
            if latest.timing.episode_integrations == 190 && gate.enabled {
                gate.enabled = false;
                camera_mount.0 = G1CameraMountProfile::ArenaEgo;
                runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
                return Ok(());
            }
            if latest.timing.episode_integrations == 230 && !gate.enabled {
                gate.enabled = true;
                camera_mount.0 = G1CameraMountProfile::StaticPlacementOverview;
                runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
                return Ok(());
            }
        }
        if (runtime.static_transfer_auxiliary || runtime.static_observed_grasp)
            && runtime
                .static_auxiliary_activation_frame
                .is_some_and(|n| runtime.render_frames < n + 2)
        {
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
            (live.prefetch_after_ticks.is_some() || live.wait_after_chunks)
                && latest.timing.episode_integrations > 0
        });
        if !render_ready && !prefetch_running {
            return Ok(());
        }
        if runtime.local_model_startup.is_some() {
            if latest.phase != G1WorkerPhase::Paused || latest.timing.episode_integrations != 0 {
                return Err("local model preparation requires a paused zero-Tick owner".into());
            }
            let episode_id = runtime.episode_id;
            let output = runtime.options.output.clone();
            let gate = runtime.local_model_startup.as_mut().unwrap();
            if !gate.renderer_announced {
                fs::write(
                    output.join("local_models_waiting.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "schema":"g1_renderer_ready_before_local_models_v1","episode_id":episode_id,
                        "native_tick":0,"renderer_ready":true,"first_task_image_not_requested":true,
                        "task_qualified":false,
                    }))
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                gate.renderer_announced = true;
            }
            let Some(admission) = gate.poll(episode_id, latest.timing.episode_integrations)? else {
                return Ok(());
            };
            outcome.0.lock().unwrap().local_model_startup = Some(admission);
            runtime.local_model_startup.take();
            // Preparation has its own explicit bound and receipt. The existing
            // physical-task timeout and every image TTL begin with fresh input.
            runtime.started = Instant::now();
        }
        #[cfg(feature = "g1_constraint_diagnostic")]
        if let Some(dispatch) = runtime.qwen_dispatch.take() {
            let result = {
                let mut guard = dispatch
                    .lock()
                    .map_err(|_| "Qwen initial dispatch poisoned")?;
                let result = guard.initial(&mut runtime, &outcome, &port);
                if let Err(reason) = &result {
                    guard.record_failure(&outcome, reason)?;
                }
                result
            };
            runtime.qwen_dispatch = Some(dispatch);
            if !result? {
                return Ok(());
            }
        }
        if runtime.live_policy.is_some() && !drive_bounded_task(&mut runtime, &outcome, &port)? {
            return Ok(());
        }
        if runtime.static_observed_grasp
            && !runtime.static_grasp_pair_secondary
            && latest.phase == G1WorkerPhase::Paused
            && camera_mount.0 == G1CameraMountProfile::ArenaEgo
        {
            camera_mount.0 = G1CameraMountProfile::StaticPlacementOverview;
            runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
            return Ok(());
        }
        if runtime.live_policy.is_none() && !runtime.command_submitted && runtime.options.ticks > 0
        {
            if runtime.startup_ticks != 0 {
                if !drive_static_startup(&mut runtime, &outcome)? {
                    return Ok(());
                }
            } else {
                runtime
                    .worker
                    .submit_stand(runtime.episode_id, runtime.options.ticks)?;
            }
            runtime.command_submitted = true;
            return Ok(());
        }
        #[cfg(feature = "g1_constraint_diagnostic")]
        let continuous_complete = runtime
            .continuous_route
            .as_ref()
            .is_some_and(|r| r.completed());
        #[cfg(not(feature = "g1_constraint_diagnostic"))]
        let continuous_complete = false;
        #[cfg(feature = "g1_constraint_diagnostic")]
        let observed_grasp_complete = runtime
            .static_observed_grasp_route
            .as_ref()
            .is_some_and(|r| r.completed());
        #[cfg(not(feature = "g1_constraint_diagnostic"))]
        let observed_grasp_complete = false;
        let observed_place_complete =
            runtime
                .static_observed_place_target_tick
                .is_some_and(|target| {
                    latest.phase == G1WorkerPhase::Paused
                        && latest.timing.episode_integrations == target
                });
        let capture_boundary = if observed_place_complete {
            true
        } else if observed_grasp_complete {
            latest.phase == G1WorkerPhase::Paused
        } else if continuous_complete {
            latest.phase == G1WorkerPhase::Paused
        } else if runtime
            .live_policy
            .as_ref()
            .is_some_and(|live| live.wait_after_chunks && live.completed)
        {
            latest.phase == G1WorkerPhase::Paused
        } else if runtime
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
            if runtime.static_visual_grasp
                && !runtime.static_visual_transfer
                && !runtime.static_observed_grasp
            {
                if latest.phase != G1WorkerPhase::Paused
                    || latest.timing.total_integrations
                        != if runtime.static_pregrasp { 100 } else { 140 }
                {
                    return Err(
                        "static visual handoff did not reach its fixed completed boundary".into(),
                    );
                }
                if runtime.static_marker_activation_render_frame.is_none() {
                    marker_gate
                        .as_mut()
                        .ok_or("static visual marker gate absent")?
                        .enabled = true;
                    runtime.static_marker_activation_render_frame = Some(runtime.render_frames);
                    outcome.0.lock().unwrap().static_visual_grasp_handoff = Some(
                        serde_json::json!({
                            "phase":if runtime.static_pregrasp {"one_original_chunk_then_preclosure_marker_rgb"} else {"two_original_unmarked_chunks_then_current_marker_rgb"},
                            "markers_activated_at_completed_tick":if runtime.static_pregrasp {100} else {140},"unmarked_original_vla_calls":if runtime.static_pregrasp {1} else {2},
                            "first_original_action_tick":60,"current_observation_required_tick":if runtime.static_pregrasp {100} else {140},
                            "geometric_correction_executed":false,"task_qualified":false,
                        }),
                    );
                    return Ok(());
                }
                if runtime.render_frames
                    < runtime.static_marker_activation_render_frame.unwrap() + 2
                {
                    return Ok(());
                }
            }
            if runtime
                .static_marker_worker
                .as_ref()
                .is_some_and(|w| !w.ready())
            {
                return Ok(());
            }
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
                receipt.physics_outcome = if observed_place_complete {
                    "actual_fixed_pair_rgb_classical_transfer_lower_release_settle_withdraw_complete_not_task_qualified".into()
                } else if observed_grasp_complete {
                    "bounded_actual100RGB_classical_template_grasp_lift_hold_complete_not_task_qualified".into()
                } else if continuous_complete {
                    "continuous_actual_rgb_vla_grasp_classical_carry_hold_release_and_explicit_waits_complete_not_task_qualified".into()
                } else if runtime.live_policy.as_ref().is_some_and(|live| live.wait_after_chunks && live.completed) {
                    "four_fresh_original_chunks_and_explicit_standing_waits_complete_not_task_qualified".into()
                } else if runtime
                    .mobile_assist
                    .as_ref()
                    .is_some_and(|assist| assist.completed)
                {
                    if runtime.mobile_assist.as_ref().is_some_and(|a| a.station_motion && a.auxiliary_release) {
                        "bounded_native_station_actual_auxiliary_rgb_bin_approach_open_settle_complete_not_task_qualified"
                    } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.station_motion) {
                        "bounded_native_station_public_aisle_2m_carry_stop_complete_not_target_bin_task_qualified"
                    } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.scan_only) {
                        if runtime.mobile_assist.as_ref().is_some_and(|a| a.auxiliary_release) { "bounded_actual_auxiliary_rgb_fine_approach_open_settle_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.auxiliary_approach) { "bounded_actual_auxiliary_rgb_coarse_approach_and_new_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.auxiliary_view) { "bounded_actual_auxiliary_rgb_two_marker_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.restored_view) { "bounded_actual_restored_grip_standing_turn_rgb_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.memory_view) { "bounded_actual_raised_rgb_static_target_memory_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.raising_view) { "bounded_actual_near_rgb_public_geometry_raise_view_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.visual_approach) { "bounded_actual_rgb_visual_coarse_approach_complete_not_task_qualified" } else if runtime.mobile_assist.as_ref().is_some_and(|a| a.view_with_lowering) { "bounded_classical_scan_and_lower_visibility_pose_complete_not_task_qualified" } else { "bounded_classical_scan_complete_not_task_qualified" }
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
            let use_auxiliary = runtime.static_transfer_auxiliary
                || runtime
                    .mobile_assist
                    .as_ref()
                    .is_some_and(|a| a.auxiliary_view);
            #[cfg(feature = "g1_constraint_diagnostic")]
            let use_auxiliary = use_auxiliary
                || runtime
                    .continuous_route
                    .as_ref()
                    .is_some_and(|r| r.needs_auxiliary_view());
            if frame.stamp.source != CameraPoseSource::PhysicsBody
                || frame.stamp.mount_profile
                    != if runtime.static_grasp_pair_secondary {
                        G1CameraMountProfile::ArenaEgo
                    } else if runtime.static_transfer_placement || runtime.static_observed_grasp {
                        G1CameraMountProfile::StaticPlacementOverview
                    } else if use_auxiliary {
                        runtime
                            .mobile_assist
                            .as_ref()
                            .filter(|assist| assist.auxiliary_view)
                            .map_or(
                                G1CameraMountProfile::AuxiliaryGripOverview,
                                MobileAssistCaptureRuntime::camera_mount,
                            )
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
            if !runtime.static_visual_transfer {
                let observed_grasp = runtime.static_observed_grasp;
                let pair_secondary = runtime.static_grasp_pair_secondary;
                let placement_started = runtime.static_observed_place_target_tick.is_some();
                let passive_pair = runtime.static_grasp_fixed_camera_pair;
                if let Some(worker) = &mut runtime.static_marker_worker {
                    if observed_grasp {
                        if !pair_secondary && !placement_started {
                            // A passive published second view must remain available
                            // when the first view misses the apple. This receipt
                            // cannot actuate; placement still requires both roles
                            // from the separate strict current paired-RGB result.
                            if passive_pair {
                                worker.submit_unverified_grip_capture(&frame)?;
                            } else {
                                worker.submit_grip_capture(&frame)?;
                            }
                        }
                    } else {
                        worker.submit_capture(&frame)?;
                    }
                }
            }
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
            let image_directory = if runtime.static_grasp_pair_secondary {
                let directory = runtime
                    .options
                    .output
                    .join("static_grip_secondary_vision_input");
                fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
                let observation = super::g1_marker_vision::static_observation(&frame.stamp)?;
                let first: serde_json::Value = serde_json::from_slice(
                    &fs::read(
                        runtime
                            .options
                            .output
                            .join("static_grip_vision_input/observation.json"),
                    )
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                super::g1_marker_vision::validate_fixed_pair(&first, &observation)?;
                fs::write(
                    directory.join("observation.json"),
                    serde_json::to_vec_pretty(&observation).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                Some(directory)
            } else {
                None
            };
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
            .save(image_directory.as_ref().map_or_else(
                || runtime.options.output.join("ego_640x480.png"),
                |p| p.join("ego.png"),
            ))
            .map_err(|error| error.to_string())?;
            fs::write(
                image_directory.as_ref().map_or_else(
                    || runtime.options.output.join("ego_stamp.json"),
                    |p| p.join("stamp.json"),
                ),
                serde_json::to_vec_pretty(&frame.stamp).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            if !runtime.static_grasp_pair_secondary {
                outcome.0.lock().unwrap().ego_stamp = Some(frame.stamp);
            }
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
            if runtime.static_observed_grasp
                && outcome.0.lock().unwrap().static_grip_localization.is_none()
            {
                return Ok(());
            }
            if runtime.static_marker_worker.is_some()
                && outcome
                    .0
                    .lock()
                    .unwrap()
                    .static_marker_localization
                    .is_none()
            {
                return Ok(());
            }
            if runtime.static_grasp_fixed_camera_pair
                && runtime.static_observed_place_target_tick.is_none()
                && !runtime.static_grasp_pair_secondary
            {
                if runtime.static_observed_place {
                    for (original, preserved) in [
                        ("ego_640x480.png", "static_grip_primary_640x480.png"),
                        ("ego_stamp.json", "static_grip_primary_stamp.json"),
                        ("main_1920x1080.png", "static_grip_main_1920x1080.png"),
                    ] {
                        fs::copy(
                            runtime.options.output.join(original),
                            runtime.options.output.join(preserved),
                        )
                        .map_err(|e| e.to_string())?;
                    }
                }
                runtime.static_grasp_pair_secondary = true;
                runtime.requested = false;
                runtime.ego_saved = false;
                camera_mount.0 = G1CameraMountProfile::ArenaEgo;
                runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
                if let Some(handoff) = &mut outcome.0.lock().unwrap().static_visual_grasp_handoff {
                    handoff["fixed_camera_pair_same_native_tick"] = true.into();
                    handoff["secondary_camera_profile"] = "arena_ego".into();
                    handoff["secondary_capture_proposes_actuation"] = false.into();
                }
                return Ok(());
            }
            #[cfg(feature = "g1_constraint_diagnostic")]
            if runtime.static_observed_place && runtime.static_observed_place_target_tick.is_none()
            {
                if static_observed_place::prepare(&mut runtime, &outcome)? {
                    runtime.requested = false;
                    runtime.ego_saved = false;
                    runtime.static_grasp_pair_secondary = false;
                    *runtime.main_saved.lock().unwrap() = Ok(false);
                    camera_mount.0 = G1CameraMountProfile::StaticPlacementOverview;
                    runtime.static_auxiliary_activation_frame = Some(runtime.render_frames);
                }
                return Ok(());
            }
            if let Some(evidence) = &mut runtime.owner_evidence {
                evidence.drain(&mut outcome.0.lock().unwrap())?;
                if !outcome.0.lock().unwrap().owner_step_trace_complete {
                    return Err("owner evidence did not cover every actual integration".into());
                }
            }
            if let Some(evidence) = &mut runtime.mobile_stand_evidence {
                evidence.drain(&mut outcome.0.lock().unwrap())?;
                if !outcome.0.lock().unwrap().owner_step_trace_complete {
                    return Err("mobile standing evidence did not cover every integration".into());
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
            #[cfg(feature = "g1_constraint_diagnostic")]
            if let Some(evidence) = &mut runtime.startup_evidence {
                evidence.drain(&mut outcome.0.lock().unwrap())?;
                if !outcome.0.lock().unwrap().owner_step_trace_complete {
                    return Err(
                        "startup/task owner evidence did not cover every integration".into(),
                    );
                }
            }
            #[cfg(feature = "g1_constraint_diagnostic")]
            if let Some(dispatch) = runtime.qwen_dispatch.take() {
                let result = {
                    let mut guard = dispatch.lock().map_err(|_| "Qwen dispatch poisoned")?;
                    let result = guard.final_feedback(&runtime, &outcome);
                    if let Err(reason) = &result {
                        guard.record_failure(&outcome, reason)?;
                    }
                    result
                };
                runtime.qwen_dispatch = Some(dispatch);
                if !result? {
                    return Ok(());
                }
            }
            outcome.0.lock().unwrap().capture_succeeded = true;
            exit.write(AppExit::Success);
        }
        Ok(())
    })();
    let error = result.err().or_else(|| {
        if runtime
            .local_model_startup
            .as_ref()
            .is_some_and(local_model_startup::Gate::expired)
        {
            return Some("local model preparation exceeded its explicit360s bound".into());
        }
        (!runtime.interactive
            && runtime.started.elapsed() > runtime.options.timeout
            && !runtime
                .local_model_startup
                .as_ref()
                .is_some_and(|gate| gate.renderer_announced))
        .then(|| format!("native camera diagnostic timed out: {}", port.progress()))
    });
    if let Some(error) = error {
        runtime.worker.pause();
        let mut receipt = outcome.0.lock().unwrap();
        receipt.failure_reason = Some(error);
        receipt.final_camera_progress = Some(port.capture_progress());
        exit.write(AppExit::error());
    }
}
