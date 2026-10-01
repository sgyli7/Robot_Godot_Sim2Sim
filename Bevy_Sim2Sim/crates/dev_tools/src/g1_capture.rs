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
    g1_visual::{G1VisualModel, G1VisualPlugin, G1VisualStatus},
    install_station_render_health, validate_render_asset_root,
};
use robot_minigame::g1::contract::G1Command;
use serde::{Deserialize, Serialize};
use simulation_minigame::g1::{
    runner::{G1ActuatorBackend, G1RunnerConfig},
    worker::{G1Worker, G1WorkerPhase, G1WorkerSnapshot, TimedG1Command},
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
    runner: G1RunnerConfig,
    visual_path: PathBuf,
    visual_sha256: String,
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
    run_capture(
        config.runner,
        &config.visual_path,
        &config.visual_sha256,
        options,
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
    pub stand_sha256: &'static str,
    pub walk_sha256: &'static str,
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
    pub unqualified: Vec<&'static str>,
}

impl G1CaptureReceipt {
    fn initial(ticks: u32, config: &G1RunnerConfig, visual_sha256: &str) -> Self {
        Self {
            schema_version: 1,
            scope: "native_g1_camera_pairing_diagnostic",
            capture_succeeded: false,
            task_qualified: false,
            environment: "source_near_simple_floor_not_science_station",
            floor_center_engine: [0., -0.25, 0.],
            floor_full_extents_m: [40., 0.5, 40.],
            floor_contact_friction: config.floor_contact_friction,
            robot_contact_friction: config.robot_contact_friction,
            definition_sha256: config.definition_sha256.clone(),
            ort_sha256: config.ort_sha256.clone(),
            stand_sha256: robot_minigame::g1::policy::STAND_SHA256,
            walk_sha256: robot_minigame::g1::policy::WALK_SHA256,
            visual_sha256: visual_sha256.into(),
            actuator_backend: config.actuator_backend,
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

    fn observe(&mut self, snapshot: &G1WorkerSnapshot) {
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

#[derive(Resource)]
struct CaptureRuntime {
    worker: G1Worker,
    options: G1CaptureOptions,
    episode_id: u64,
    started: Instant,
    render_frames: u32,
    latest: Option<Arc<G1WorkerSnapshot>>,
    command_submitted: bool,
    requested: bool,
    ego_saved: bool,
    main_saved: Arc<Mutex<Result<bool, String>>>,
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
    if options.ticks > 150
        || options.timeout.is_zero()
        || options.timeout > Duration::from_secs(180)
    {
        return Err("capture requires 0..=150 ticks and a finite timeout <=180 seconds".into());
    }
    let assets = validate_render_asset_root(&default_asset_root())?;
    let scene = StationScene::load(&assets)?;
    let model = G1VisualModel::load(visual_path, visual_sha256)?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::create_dir(&options.output)
        .map_err(|error| format!("capture output must be new: {error}"))?;
    let output = options.output.clone();
    let episode_id = config.episode_id;
    let outcome = CaptureOutcome(Arc::new(Mutex::new(G1CaptureReceipt::initial(
        options.ticks,
        &config,
        visual_sha256,
    ))));
    let worker = G1Worker::spawn(config).map_err(|error| error.to_string())?;
    let mut app = App::new();
    // Scene supplies the existing enamel configuration only. Its meshes, props,
    // fixtures and camera shots are not spawned in this floor diagnostic.
    app.insert_resource(scene)
        .insert_resource(model)
        .insert_resource(outcome.clone())
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
        .add_systems(Update, drive_capture);
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

fn observation(snapshot: &G1WorkerSnapshot) -> Result<Option<Arc<G1CameraNativeState>>, String> {
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

fn drive_capture(
    mut commands: Commands,
    mut runtime: ResMut<CaptureRuntime>,
    outcome: Res<CaptureOutcome>,
    health: Res<StationRenderHealth>,
    visual: Res<G1VisualStatus>,
    mut input: ResMut<G1BodyObservationInput>,
    port: Res<G1CameraPort>,
    mut exit: MessageWriter<AppExit>,
) {
    runtime.render_frames += 1;
    let result = (|| -> Result<(), String> {
        if let Some(error) = health.snapshot().error {
            return Err(error);
        }
        if let Some(error) = &visual.error {
            return Err(error.clone());
        }
        if let Some(snapshot) = runtime.worker.take_latest() {
            input.0 = observation(&snapshot)?;
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
        if !render_ready {
            return Ok(());
        }
        if !runtime.command_submitted && runtime.options.ticks > 0 {
            runtime
                .worker
                .submit(TimedG1Command {
                    episode_id: runtime.episode_id,
                    valid_until_sim_ns: u64::from(runtime.options.ticks) * 20_000_000,
                    valid_until_wall: Instant::now() + Duration::from_secs(8),
                    command: G1Command::default(),
                })
                .map_err(|error| error.to_string())?;
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
