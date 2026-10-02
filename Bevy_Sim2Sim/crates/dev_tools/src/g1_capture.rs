//! Bounded native G1 camera evidence in the runner's actual simple floor world.
//!
//! This development scene does not load scientific-station geometry. Its sole
//! environment mesh is the same 40 x 0.5 x 40 metre floor as G1Runner. Real RGB
//! and measured self state can therefore be tested before station task physics
//! is integrated. Neither initialization nor these captures qualify standing,
//! manipulation, locomotion or a visual decision loop.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    winit::WinitPlugin,
};
use rendering_minigame::{
    StationRenderHealth, StationScene, default_asset_root,
    g1_camera::{
        CameraPoseSource, G1BodyObservationInput, G1CameraJointState, G1CameraNativeState,
        G1CameraPlugin, G1CameraPort, G1CaptureStamp, G1ObservationPlugin,
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
        AgileWorker, ArenaTaskWorker, G1Worker, G1WorkerPhase, G1WorkerTiming, TimedAgileCommand,
        TimedArenaTaskCommand, TimedG1Command, WorkerSnapshot,
    },
};
use task_minigame::{
    policy::{
        ARENA_ACTION_PERIOD_NS, MobilePolicyClient, PolicyInferenceRequest, PolicyObservation,
        PolicyWorker, StaticPolicyClient, map_measured_joints, profile_contract,
    },
    types::{ObservationStamp, TaskProfile},
};

/// Explicit evidence settings. Zero ticks captures the native initialized world
/// without inference or integration. A nonzero budget performs at most 3 seconds
/// of the unqualified real standing candidate and also records any failure.
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
    predictive_limit_diagnostic: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LivePolicyConfiguration {
    endpoint: String,
    max_calls: u32,
    timeout_ms: u64,
}

struct LivePolicyRuntime {
    worker: PolicyWorker,
    profile: TaskProfile,
    next_sequence: u64,
    pending: Option<ObservationStamp>,
    submitted_chunks: u32,
    max_calls: u32,
    next_boundary_tick: u64,
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
    fn spawn(self, predictive_limit_diagnostic: bool) -> Result<CaptureWorker, String> {
        if predictive_limit_diagnostic {
            let Self::Task(config) = self else {
                return Err("predictive limit comparison requires the static task owner".into());
            };
            #[cfg(feature = "g1_constraint_diagnostic")]
            return ArenaTaskWorker::spawn_static_predictive_limit_diagnostic(config)
                .map(CaptureWorker::Task)
                .map_err(|e| e.to_string());
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
        }
    }
}
impl CaptureWorker {
    fn take_latest(&self) -> Option<Arc<CaptureSnapshot>> {
        match self {
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
            Self::Task(w) => w.pause(),
            Self::Mobile(w) => w.pause(),
            Self::Static { worker, .. } => worker.pause(),
        }
    }
    fn submit_stand(&self, episode_id: u64, ticks: u32) -> Result<(), String> {
        let valid_until_sim_ns = u64::from(ticks) * 20_000_000;
        let valid_until_wall = Instant::now() + Duration::from_secs(8);
        match self {
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

/// Read the caller's frozen configuration and run this diagnostic explicitly.
pub fn run_capture_from_file(
    path: &Path,
    options: G1CaptureOptions,
) -> Result<G1CaptureReceipt, String> {
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
        config.predictive_limit_diagnostic,
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
    pub actuator_backend: G1ActuatorBackend,
    pub physics_hz: u32,
    pub integrations_per_tick: u32,
    pub requested_ticks: u32,
    pub actual_integrations: u64,
    pub actual_torque_updates: u64,
    pub actual_model_attempts: u64,
    pub actual_model_successes: u64,
    pub active_wall_seconds: f64,
    pub active_sim_seconds: f64,
    pub control_deadlines_missed: u64,
    pub pending_ticks: u64,
    pub physics_outcome: String,
    pub failure_reason: Option<String>,
    pub ego_stamp: Option<G1CaptureStamp>,
    pub main_resolution: [u32; 2],
    pub live_policy_inference_calls: u32,
    pub live_policy_successes: u32,
    pub live_action_chunks: Vec<serde_json::Value>,
    pub pauses_for_camera_and_policy: bool,
    pub camera_exposure_ev100: f32,
    pub factory_verified_predictive_limit_joints: usize,
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
            actuator_backend: backend,
            physics_hz: 50,
            integrations_per_tick: 1,
            requested_ticks: ticks,
            actual_integrations: 0,
            actual_torque_updates: 0,
            actual_model_attempts: 0,
            actual_model_successes: 0,
            active_wall_seconds: 0.,
            active_sim_seconds: 0.,
            control_deadlines_missed: 0,
            pending_ticks: 0,
            physics_outcome: "loading".into(),
            failure_reason: None,
            ego_stamp: None,
            main_resolution: [1920, 1080],
            live_policy_inference_calls: 0,
            live_policy_successes: 0,
            live_action_chunks: Vec::new(),
            pauses_for_camera_and_policy: false,
            camera_exposure_ev100: Exposure::default().ev100,
            factory_verified_predictive_limit_joints: 0,
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
struct DiagnosticExposure(f32);

fn configure_diagnostic_exposure(
    mut commands: Commands,
    setting: Res<DiagnosticExposure>,
    mut cameras: Query<(Entity, Option<&mut Exposure>), With<Camera3d>>,
) {
    for (entity, camera) in &mut cameras {
        if let Some(mut camera) = camera {
            camera.ev100 = setting.0;
        } else {
            commands
                .entity(entity)
                .insert(Exposure { ev100: setting.0 });
        }
    }
}

#[derive(Resource)]
struct CaptureRuntime {
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
        false,
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
    predictive_limit_diagnostic: bool,
) -> Result<G1CaptureReceipt, String> {
    let exposure_ev100 = exposure_ev100.unwrap_or(Exposure::default().ev100);
    if !exposure_ev100.is_finite() || !(0.0..=20.0).contains(&exposure_ev100) {
        return Err("diagnostic exposure must be finite EV100 in 0..=20".into());
    }
    let tick_limit = if matches!(config, CaptureRunnerConfig::Task(_)) {
        200
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
                || policy.max_calls > 4
                || options.ticks != policy.max_calls * horizon
                || policy.timeout_ms == 0
                || policy.timeout_ms > 20_000
                || config.max_observation_wall_age_ms == 0
                || config.max_observation_wall_age_ms > 20_000
            {
                return Err("live diagnostic requires 1..=4 whole chunks, exact tick budget and bounded image/inference age <=20s".into());
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
            })
        }
        (_, None) => None,
        _ => return Err("live policy requires the matched task owner".into()),
    };
    let assets = validate_render_asset_root(&default_asset_root())?;
    let scene = StationScene::load(&assets)?;
    let model = G1VisualModel::load(visual_path, visual_sha256)?;
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
    outcome.0.lock().unwrap().camera_exposure_ev100 = exposure_ev100;
    if live_policy.is_some() {
        let mut receipt = outcome.0.lock().unwrap();
        receipt.scope = "native_live_rgb_to_matched_policy_to_same_owner_diagnostic";
        receipt.pauses_for_camera_and_policy = true;
    }
    let worker = config.spawn(predictive_limit_diagnostic)?;
    if predictive_limit_diagnostic {
        outcome
            .0
            .lock()
            .unwrap()
            .factory_verified_predictive_limit_joints = 43;
    }
    let mut app = App::new();
    // Scene supplies the existing enamel configuration only. Its meshes, props,
    // fixtures and camera shots are not spawned in this floor diagnostic.
    app.insert_resource(scene)
        .insert_resource(DiagnosticExposure(exposure_ev100))
        .insert_resource(model)
        .insert_resource(outcome.clone())
        .init_resource::<G1TaskVisualInput>()
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
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
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
        .add_systems(Startup, setup_floor_scene)
        .add_systems(
            Update,
            (configure_diagnostic_exposure, drive_capture).chain(),
        );
    if let Some(model) = task_model {
        app.insert_resource(model).add_plugins(G1TaskVisualPlugin);
    }
    if episode_id > 0 {
        app.world()
            .resource::<G1CameraPort>()
            .reset(episode_id)
            .map_err(|error| format!("camera episode: {error}"))?;
    }
    install_station_render_health(&mut app)?;
    let exit = app.run();
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
    if !receipt.capture_succeeded {
        return Err(receipt
            .failure_reason
            .unwrap_or_else(|| "native camera capture did not finish".into()));
    }
    Ok(receipt)
}

fn setup_floor_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 450.,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(3., 5., 2.).looking_at(Vec3::ZERO, Vec3::Y),
    ));
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

/// Diagnostic pauses are explicit: rendering/inference never advances physics,
/// and each reply resumes the existing owner without a scene reset or restamp.
fn drive_live_policy(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
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
        let CaptureWorker::Task(owner) = &runtime.worker else {
            return Err("live task lost its unique physical owner".into());
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
                    .ok_or("live time overflow")?,
                valid_until_wall: Instant::now()
                    + Duration::from_nanos(horizon * ARENA_ACTION_PERIOD_NS)
                    + Duration::from_secs(1),
                command: ArenaTaskCommand {
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
            runtime
                .options
                .output
                .join(format!("live_ego_{:04}.png", live.next_sequence)),
            camera.png(),
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            runtime
                .options
                .output
                .join(format!("live_stamp_{:04}.json", live.next_sequence)),
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
    port: Res<G1CameraPort>,
    mut exit: MessageWriter<AppExit>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shelf_visual: Query<&mut Transform, With<SourceShelfVisual>>,
) {
    runtime.render_frames += 1;
    let result = (|| -> Result<(), String> {
        if let Some(error) = health.snapshot().error {
            return Err(error);
        }
        if let Some(error) = &visual.error {
            return Err(error.clone());
        }
        if let Some(error) = task_status.as_ref().and_then(|s| s.error.as_ref()) {
            return Err(error.clone());
        }
        if let Some(snapshot) = runtime.worker.take_latest() {
            input.0 = observation(&snapshot)?;
            if let Some(shelf) = snapshot
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
        if latest.phase == G1WorkerPhase::Stopped {
            return Err("worker stopped before capture".into());
        }
        if latest.phase == G1WorkerPhase::Failed && input.0.is_none() {
            return Err(latest
                .reason
                .clone()
                .unwrap_or_else(|| "worker failed before publishing a native snapshot".into()));
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
        if !render_ready {
            return Ok(());
        }
        if runtime.live_policy.is_some() && !drive_live_policy(&mut runtime, &outcome, &port)? {
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
        let capture_boundary = if runtime.options.ticks == 0 {
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
            if latest.phase != G1WorkerPhase::Failed {
                receipt.physics_outcome = if runtime.options.ticks == 0 {
                    "native_initialization_zero_integrations"
                } else {
                    "bounded_candidate_ticks_complete"
                }
                .into();
            }
        }
        if let Some(frame) = port.take() {
            let frame = frame?;
            if frame.stamp.source != CameraPoseSource::PhysicsBody
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
            outcome.0.lock().unwrap().capture_succeeded = true;
            exit.write(AppExit::Success);
        }
        Ok(())
    })();
    let error = result.err().or_else(|| {
        (runtime.started.elapsed() > runtime.options.timeout)
            .then(|| format!("native camera diagnostic timed out: {}", port.progress()))
    });
    if let Some(error) = error {
        runtime.worker.pause();
        outcome.0.lock().unwrap().failure_reason = Some(error);
        exit.write(AppExit::error());
    }
}
