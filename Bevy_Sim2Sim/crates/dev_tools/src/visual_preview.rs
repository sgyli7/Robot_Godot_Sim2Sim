//! Actual GPU preview, pipeline status and screenshot capture lifecycle.
mod sequence;

use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
};
use rendering_minigame::{
    RobotVisualInput, RobotVisualModel, RobotVisualPhase, RobotVisualPlugin, RobotVisualStatus,
    RobotVisualSystems, StationCameraControl, StationLabelBakeStatus, StationRenderHealth,
    StationScene, StationVisualPlugin, install_station_render_health, validate_render_asset_root,
};
pub use sequence::{
    RobotPoseCaptureFrame, RobotPoseSequenceCaptureOptions, RobotPoseSequenceCaptureReceipt,
    RobotPoseSequenceCaptureResources, run_robot_pose_sequence_capture,
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
/// Reproducible development preview and capture settings.
#[derive(Clone, Debug, Default)]
pub struct PreviewOptions {
    pub capture_path: Option<PathBuf>,
    pub frames: Option<u32>,
    pub view: rendering_minigame::StationView,
}

#[derive(Resource, Clone)]
struct PipelineStatus(Arc<Mutex<Status>>);
#[derive(Default)]
struct Status {
    ready: bool,
    error: Option<String>,
    compiled: usize,
    pending: usize,
    captured: bool,
    captured_count: usize,
    initialization_ready: bool,
}
#[derive(Resource)]
struct PreviewRuntime {
    options: PreviewOptions,
    started: Instant,
    frames: u32,
    ready_frames: u32,
    capture_requested: bool,
    announced: bool,
    first_error_frame: Option<u32>,
    kind: PreviewKind,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum PreviewKind {
    Station,
    RobotInitialization,
}
/// Caller-owned verified data for a real robot initialization geometry preview.
/// The caller produces input from its sole physics world. This entry point never
/// creates that world, computes FK, integrates dynamics or runs a controller.
pub struct RobotInitializationPreviewResources {
    pub asset_root: PathBuf,
    pub scene: StationScene,
    pub model: RobotVisualModel,
    pub input: RobotVisualInput,
    /// Preserved exactly; PreviewOptions.view applies to the old station entry.
    pub camera: StationCameraControl,
}

/// Open a real window using an externally produced, completed initialization
/// snapshot. Readiness/capture/error handling is shared with the station preview.
/// This is a geometry check, not qualified gameplay or controller validation.
pub fn run_robot_initialization_preview(
    resources: RobotInitializationPreviewResources,
    options: PreviewOptions,
) -> Result<(), String> {
    validate_preview_options(&options)?;
    resources
        .input
        .validate_for_model(resources.model.definition())
        .map_err(|e| format!("Robot initialization preview: {e}"))?;
    let asset_root = validate_render_asset_root(&resources.asset_root)?;
    run_preview_app(
        options,
        asset_root,
        resources.scene,
        resources.camera,
        Some((resources.model, resources.input)),
    )
}

fn validate_preview_options(options: &PreviewOptions) -> Result<(), String> {
    if options.frames == Some(0) {
        Err("Preview frame count must be greater than zero".into())
    } else {
        Ok(())
    }
}

/// Launch the visual preview, with no robot or physics simulation.
pub fn run_preview() -> Result<(), String> {
    run_preview_with_options(PreviewOptions::default())
}
/// Render real frames and optionally capture them. Failed assets and shaders are errors.
pub fn run_preview_with_options(options: PreviewOptions) -> Result<(), String> {
    validate_preview_options(&options)?;
    let asset_root = rendering_minigame::default_asset_root();
    let asset_root = validate_render_asset_root(&asset_root)?;
    let scene = StationScene::load(&asset_root)?;
    let camera = StationCameraControl {
        view: options.view,
        ..default()
    };
    run_preview_app(options, asset_root, scene, camera, None)
}

fn run_preview_app(
    options: PreviewOptions,
    asset_root: PathBuf,
    scene: StationScene,
    camera: StationCameraControl,
    robot: Option<(RobotVisualModel, RobotVisualInput)>,
) -> Result<(), String> {
    let capture_required = options.capture_path.is_some();
    if let Some(p) = &options.capture_path {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("Capture directory: {e}"))?;
        }
    }
    let status = PipelineStatus(Arc::new(Mutex::new(Status::default())));
    let kind = if robot.is_some() {
        PreviewKind::RobotInitialization
    } else {
        PreviewKind::Station
    };
    let mut app = App::new();
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .insert_resource(scene)
        .insert_resource(camera)
        .insert_resource(status.clone())
        .insert_resource(PreviewRuntime {
            options,
            started: Instant::now(),
            frames: 0,
            ready_frames: 0,
            capture_requested: false,
            announced: false,
            first_error_frame: None,
            kind,
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: match kind {
                            PreviewKind::Station => {
                                "Bevy Science Station — visual preview (no robot)"
                            }
                            PreviewKind::RobotInitialization => {
                                "Bevy MicroDuck — initialization geometry preview"
                            }
                        }
                        .into(),
                        resolution: (1920, 1080).into(),
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(StationVisualPlugin);
    if let Some((model, input)) = robot {
        app.insert_resource(model)
            .insert_resource(input)
            .add_plugins(RobotVisualPlugin)
            // Observe the current frame's validation result before requesting a
            // screenshot; a stale Ready status must never admit a bad new frame.
            .add_systems(
                PostUpdate,
                preview_lifecycle.after(RobotVisualSystems::ApplyPoses),
            );
    } else {
        app.add_systems(Update, preview_lifecycle);
    }
    install_station_render_health(&mut app)?;
    let render_app = app
        .get_sub_app_mut(RenderApp)
        .ok_or("GPU render application unavailable")?;
    render_app.world_mut().insert_resource(status.clone());
    render_app.add_systems(
        Render,
        report_pipeline_status.in_set(RenderSystems::PostCleanup),
    );
    let exit = app.run();
    // App::run hands ownership to its runner and replaces app with App::empty.
    // Completion evidence must survive outside the consumed application.
    let state = status.0.lock().map_err(|_| "Render status poisoned")?;
    validate_preview_exit(kind, &state, exit, capture_required)?;
    if kind == PreviewKind::RobotInitialization {
        info!(
            "ROBOT_INITIALIZATION_PREVIEW_COMPLETE (initial geometry only; no dynamics or controller qualification)"
        );
    }
    Ok(())
}

fn validate_preview_exit(
    kind: PreviewKind,
    state: &Status,
    exit: AppExit,
    capture_required: bool,
) -> Result<(), String> {
    if let Some(error) = &state.error {
        return Err(error.clone());
    }
    if !state.ready {
        return Err("Renderer exited before required station pipelines were ready".into());
    }
    if exit.is_error() {
        return Err("Renderer returned an error exit".into());
    }
    if kind == PreviewKind::RobotInitialization {
        if !state.initialization_ready {
            return Err(
                "Robot initialization preview exited before its frame/font/pipelines were ready"
                    .into(),
            );
        }
        if capture_required && !state.captured {
            return Err(
                "Robot initialization preview exited before its requested screenshot was saved"
                    .into(),
            );
        }
    }
    Ok(())
}
fn report_pipeline_status(health: Res<StationRenderHealth>, status: Res<PipelineStatus>) {
    let health = health.snapshot();
    let mut state = status.0.lock().unwrap();
    state.compiled = health.compiled;
    state.pending = health.pending;
    state.ready = health.ready && state.error.is_none();
    if let Some(error) = health.error {
        state.error.get_or_insert(error);
    }
}
fn preview_lifecycle(
    mut commands: Commands,
    status: Res<PipelineStatus>,
    mut runtime: ResMut<PreviewRuntime>,
    mut exit: MessageWriter<AppExit>,
    labels: Res<StationLabelBakeStatus>,
    robot_status: Option<Res<RobotVisualStatus>>,
) {
    runtime.frames += 1;
    let robot_ready = if runtime.kind == PreviewKind::RobotInitialization {
        match robot_initialization_ready(robot_status.as_ref().map(|s| s.phase())) {
            Ok(ready) => ready,
            Err(error) => {
                status.0.lock().unwrap().error = Some(error);
                false
            }
        }
    } else {
        true
    };
    let mut state = status.0.lock().unwrap();
    if runtime.kind == PreviewKind::RobotInitialization {
        state.initialization_ready = false;
    }
    if let Some(error) = &state.error {
        let frame = runtime.frames;
        if runtime.first_error_frame.is_none() {
            error!("{error}");
        }
        let first = *runtime.first_error_frame.get_or_insert(frame);
        // Let independent shader jobs and submitted render work settle before
        // dropping the application. A failure never becomes a successful capture.
        if (frame - first >= 8 && state.pending == 0) || frame - first >= 180 {
            exit.write(AppExit::error());
        }
        return;
    }
    let ready = state.ready && robot_ready && labels.is_ready();
    if runtime.kind == PreviewKind::RobotInitialization {
        state.initialization_ready = ready && runtime.announced;
    }
    let captured = state.captured;
    drop(state);
    if ready {
        runtime.ready_frames += 1;
    } else {
        runtime.ready_frames = 0;
    }
    if runtime.ready_frames >= 12 && !runtime.announced {
        info!(
            frames = runtime.frames,
            "STATION_PIPELINES_READY (enamel + ink, actual GPU compilation)"
        );
        runtime.announced = true;
        if runtime.kind == PreviewKind::RobotInitialization {
            status.0.lock().unwrap().initialization_ready = true;
            info!(phase=?robot_status.as_ref().map(|s| s.phase()),
                "ROBOT_INITIALIZATION_FRAME_READY (verified geometry/pose display; no dynamic qualification)");
        }
    }
    if !ready && runtime.started.elapsed() > Duration::from_secs(60) {
        status.0.lock().unwrap().error = Some(
            "Preview GPU pipelines/font/required frame did not become ready within 60 seconds"
                .into(),
        );
        exit.write(AppExit::error());
        return;
    }
    let target = runtime.options.frames.unwrap_or(120).max(12);
    if runtime.ready_frames >= target && !runtime.capture_requested {
        if let Some(path) = &runtime.options.capture_path {
            let p = path.clone();
            commands.spawn(Screenshot::primary_window()).observe(
                move |capture: On<ScreenshotCaptured>, status: Res<PipelineStatus>| {
                    let saved = capture
                        .image
                        .clone()
                        .try_into_dynamic()
                        .map_err(|e| format!("Screenshot conversion: {e}"))
                        .and_then(|img| {
                            img.to_rgb8()
                                .save(&p)
                                .map_err(|e| format!("Screenshot {}: {e}", p.display()))
                        });
                    let mut state = status.0.lock().unwrap();
                    match saved {
                        Ok(()) => state.captured = true,
                        Err(e) => state.error = Some(e),
                    }
                },
            );
            runtime.capture_requested = true;
        } else if runtime.options.frames.is_some() {
            exit.write(AppExit::Success);
        }
    }
    if captured {
        let path = runtime.options.capture_path.as_ref().unwrap();
        match std::fs::metadata(path) {
            Ok(m) if m.len() > 0 => {
                info!(path=%path.display(),ready_frames=runtime.ready_frames,"STATION_CAPTURE_SAVED");
                exit.write(AppExit::Success);
            }
            _ => {
                status.0.lock().unwrap().error =
                    Some(format!("Screenshot was not saved: {}", path.display()));
                exit.write(AppExit::error());
            }
        }
    }
    if runtime.capture_requested && runtime.started.elapsed() > Duration::from_secs(90) {
        status.0.lock().unwrap().error = Some("Screenshot capture timed out".into());
        exit.write(AppExit::error());
    }
}

fn robot_initialization_ready(phase: Option<&RobotVisualPhase>) -> Result<bool, String> {
    match phase {
        Some(RobotVisualPhase::Ready { .. }) => Ok(true),
        Some(RobotVisualPhase::AwaitingModel | RobotVisualPhase::AwaitingFrame) => Ok(false),
        Some(RobotVisualPhase::Failed { reason }) => {
            Err(format!("Robot initialization preview failed: {reason}"))
        }
        None => Err("Robot initialization preview is missing its required visual status".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_frame_preview_fails_before_asset_resolution_or_gpu_creation() {
        let options = PreviewOptions {
            frames: Some(0),
            ..default()
        };
        assert_eq!(
            run_preview_with_options(options).unwrap_err(),
            "Preview frame count must be greater than zero"
        );
    }
    #[test]
    fn robot_capture_admission_requires_the_current_validated_pose_status() {
        assert!(robot_initialization_ready(None).is_err());
        assert!(!robot_initialization_ready(Some(&RobotVisualPhase::AwaitingFrame)).unwrap());
        assert!(!robot_initialization_ready(Some(&RobotVisualPhase::AwaitingModel)).unwrap());
        assert!(
            robot_initialization_ready(Some(&RobotVisualPhase::Ready {
                episode_id: 1,
                global_step: 0,
                episode_step: 0,
            }))
            .unwrap()
        );
        assert!(
            robot_initialization_ready(Some(&RobotVisualPhase::Failed {
                reason: "stale assembly generation".into(),
            }))
            .unwrap_err()
            .contains("stale assembly generation")
        );
    }
    #[test]
    fn initialization_exit_requires_readiness_and_actual_capture_completion() {
        let mut state = Status {
            ready: true,
            ..default()
        };
        assert!(
            validate_preview_exit(
                PreviewKind::RobotInitialization,
                &state,
                AppExit::Success,
                false
            )
            .is_err()
        );
        state.initialization_ready = true;
        assert!(
            validate_preview_exit(
                PreviewKind::RobotInitialization,
                &state,
                AppExit::Success,
                true
            )
            .is_err()
        );
        state.captured = true;
        validate_preview_exit(
            PreviewKind::RobotInitialization,
            &state,
            AppExit::Success,
            true,
        )
        .unwrap();
        state.error = Some("shader failure".into());
        assert!(
            validate_preview_exit(
                PreviewKind::RobotInitialization,
                &state,
                AppExit::Success,
                true
            )
            .is_err()
        );
    }
    #[test]
    fn readiness_record_survives_the_app_runner_consuming_the_application() {
        let status = PipelineStatus(Arc::new(Mutex::new(Status::default())));
        let mut app = App::new();
        app.insert_resource(status.clone()).set_runner(|app| {
            let mut state = app.world().resource::<PipelineStatus>().0.lock().unwrap();
            state.ready = true;
            state.initialization_ready = true;
            state.captured = true;
            AppExit::Success
        });
        let exit = app.run();
        assert!(!app.world().contains_resource::<PipelineStatus>());
        validate_preview_exit(
            PreviewKind::RobotInitialization,
            &status.0.lock().unwrap(),
            exit,
            true,
        )
        .unwrap();
    }
}
