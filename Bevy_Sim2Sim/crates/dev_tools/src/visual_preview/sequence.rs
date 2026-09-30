//! GPU capture of caller-produced, completed 60 Hz robot poses.
//!
//! The renderer samples a contiguous episode and never derives a pose, steps
//! physics, runs a policy, or duplicates one pose to fill a video timeline.

use super::{PipelineStatus, Status, report_pipeline_status};
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
    RobotVisualSystems, StationCameraControl, StationLabelBakeStatus, StationScene,
    StationVisualPlugin, install_station_render_health, validate_render_asset_root,
};
use robot_minigame::body_pose::RobotPoseFrame;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// Verified display inputs from a caller-owned, single physics world.
///
/// Supply one completed pose per consecutive 60 Hz simulation tick in a single
/// episode. The renderer checks identity, handles and counters but cannot prove
/// the caller's physics provenance or actual wall-clock simulation frequency.
pub struct RobotPoseSequenceCaptureResources {
    pub asset_root: PathBuf,
    pub scene: StationScene,
    pub model: RobotVisualModel,
    pub camera: StationCameraControl,
    pub poses_60hz: Vec<Arc<RobotPoseFrame>>,
}

/// Capture at any integer frame rate from 1 through 60 fps. The exact source
/// tick chosen for output frame n is floor(n * 60 / output_fps).
pub struct RobotPoseSequenceCaptureOptions {
    pub output_dir: PathBuf,
    pub output_fps: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RobotPoseCaptureFrame {
    pub output_index: usize,
    pub source_index: usize,
    pub episode_id: u64,
    pub global_step: u64,
    pub episode_step: u64,
    pub pose_sha256: String,
    pub png: PathBuf,
}

/// A display/capture record. This does not certify control, contact or policy.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RobotPoseSequenceCaptureReceipt {
    pub model_file_sha256: String,
    pub declared_source_hz: u32,
    pub output_fps: u32,
    pub frames: Vec<RobotPoseCaptureFrame>,
}

#[derive(Resource)]
struct SequenceRuntime {
    started: Instant,
    frames: u32,
    ready_frames: u32,
    announced: bool,
    first_error_frame: Option<u32>,
    capture_started: Option<Instant>,
    capture_requested: bool,
    index: usize,
    poses: Vec<Arc<RobotPoseFrame>>,
    paths: Vec<PathBuf>,
}

pub fn run_robot_pose_sequence_capture(
    resources: RobotPoseSequenceCaptureResources,
    options: RobotPoseSequenceCaptureOptions,
) -> Result<RobotPoseSequenceCaptureReceipt, String> {
    let RobotPoseSequenceCaptureResources {
        asset_root,
        scene,
        model,
        camera,
        poses_60hz,
    } = resources;
    let selected = select_source_indices(&model, &poses_60hz, options.output_fps)?;
    let asset_root = validate_render_asset_root(&asset_root)?;
    let mut input = RobotVisualInput::new(model.definition(), poses_60hz[0].clone())
        .map_err(|e| format!("Robot sequence initial binding: {e}"))?;
    for pose in &poses_60hz {
        input.publish(pose.clone());
        input
            .validate_for_model(model.definition())
            .map_err(|e| format!("Robot sequence pose validation: {e}"))?;
    }
    input.publish(poses_60hz[0].clone());

    std::fs::create_dir_all(&options.output_dir)
        .map_err(|e| format!("Capture directory {}: {e}", options.output_dir.display()))?;
    let output_dir = options
        .output_dir
        .canonicalize()
        .map_err(|e| format!("Capture directory {}: {e}", options.output_dir.display()))?;
    let paths: Vec<_> = (0..selected.len())
        .map(|index| output_dir.join(format!("frame_{index:06}.png")))
        .collect();
    let manifest_path = output_dir.join("capture_manifest.json");
    for path in paths.iter().chain(std::iter::once(&manifest_path)) {
        if path.exists() {
            return Err(format!("Capture output already exists: {}", path.display()));
        }
    }
    let frames = selected
        .iter()
        .enumerate()
        .map(|(output_index, &source_index)| {
            let pose = &poses_60hz[source_index];
            let json = serde_json::to_vec(pose.as_ref())
                .map_err(|e| format!("Serialize source pose for receipt: {e}"))?;
            Ok(RobotPoseCaptureFrame {
                output_index,
                source_index,
                episode_id: pose.episode_id,
                global_step: pose.global_step,
                episode_step: pose.episode_step,
                pose_sha256: format!("{:x}", Sha256::digest(json)),
                png: paths[output_index].clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let receipt = RobotPoseSequenceCaptureReceipt {
        model_file_sha256: model.definition().file_sha256().into(),
        declared_source_hz: 60,
        output_fps: options.output_fps,
        frames,
    };
    let poses = selected
        .iter()
        .map(|&source_index| poses_60hz[source_index].clone())
        .collect();

    let status = PipelineStatus(Arc::new(Mutex::new(Status::default())));
    let mut app = App::new();
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .insert_resource(scene)
        .insert_resource(camera)
        .insert_resource(model)
        .insert_resource(input)
        .insert_resource(status.clone())
        .insert_resource(SequenceRuntime {
            started: Instant::now(),
            frames: 0,
            ready_frames: 0,
            announced: false,
            first_error_frame: None,
            capture_started: None,
            capture_requested: false,
            index: 0,
            poses,
            paths,
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Bevy MicroDuck — completed-pose sequence capture".into(),
                        resolution: (1920, 1080).into(),
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(StationVisualPlugin)
        .add_plugins(RobotVisualPlugin)
        .add_systems(
            PostUpdate,
            sequence_lifecycle.after(RobotVisualSystems::ApplyPoses),
        );

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
    let state = status.0.lock().map_err(|_| "Render status poisoned")?;
    if let Some(error) = &state.error {
        return Err(error.clone());
    }
    if exit.is_error()
        || !state.ready
        || !state.initialization_ready
        || state.captured_count != receipt.frames.len()
    {
        return Err(format!(
            "Robot sequence capture incomplete: saved {}/{} frames, GPU ready={}, pose ready={}",
            state.captured_count,
            receipt.frames.len(),
            state.ready,
            state.initialization_ready
        ));
    }
    drop(state);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&receipt)
            .map_err(|e| format!("Serialize sequence capture receipt: {e}"))?,
    )
    .map_err(|e| format!("Write capture manifest {}: {e}", manifest_path.display()))?;
    Ok(receipt)
}

fn select_source_indices(
    model: &RobotVisualModel,
    poses: &[Arc<RobotPoseFrame>],
    output_fps: u32,
) -> Result<Vec<usize>, String> {
    if !(1..=60).contains(&output_fps) {
        return Err("Sequence output_fps must be in 1..=60".into());
    }
    let first = poses
        .first()
        .ok_or("Robot sequence requires at least two completed pose frames")?;
    first
        .validate(
            model.definition().file_sha256(),
            model.definition().model().counts.nbody,
        )
        .map_err(|e| format!("Robot sequence first pose: {e}"))?;
    validate_contiguous_steps(poses)?;
    for (index, pose) in poses.iter().enumerate() {
        pose.validate(
            model.definition().file_sha256(),
            model.definition().model().counts.nbody,
        )
        .map_err(|e| format!("Robot sequence pose {index}: {e}"))?;
    }
    let selected = sampled_indices(poses.len(), output_fps);
    if selected.len() < 2 {
        return Err(
            "Robot sequence must contain at least two distinct source ticks at output_fps".into(),
        );
    }
    require_sampled_motion(poses, &selected)?;
    Ok(selected)
}

fn validate_contiguous_steps(poses: &[Arc<RobotPoseFrame>]) -> Result<(), String> {
    let first = poses
        .first()
        .ok_or("Robot sequence requires at least two completed pose frames")?;
    for (index, pose) in poses.iter().enumerate() {
        let expected_global = first
            .global_step
            .checked_add(index as u64)
            .ok_or("Robot sequence global step overflow")?;
        let expected_episode = first
            .episode_step
            .checked_add(index as u64)
            .ok_or("Robot sequence episode step overflow")?;
        if pose.episode_id != first.episode_id
            || pose.global_step != expected_global
            || pose.episode_step != expected_episode
        {
            return Err(format!(
                "Robot sequence pose {index} must be the next completed 60 Hz tick in one episode"
            ));
        }
    }
    Ok(())
}

fn require_sampled_motion(poses: &[Arc<RobotPoseFrame>], selected: &[usize]) -> Result<(), String> {
    // A stationary physical state is allowed, but bit-identical poses across
    // the whole capture cannot substantiate a dynamic video.
    let first_sample = &poses[selected[0]];
    if selected.iter().skip(1).all(|&source_index| {
        let sample = &poses[source_index];
        first_sample.poses.iter().all(|body| {
            sample.poses.iter().any(|other| {
                other.source_body_id == body.source_body_id
                    && other.translation == body.translation
                    && other.rotation_xyzw == body.rotation_xyzw
            })
        })
    }) {
        return Err("Robot sequence contains no visible pose change across sampled ticks".into());
    }
    Ok(())
}

fn sampled_indices(source_count: usize, output_fps: u32) -> Vec<usize> {
    (0..source_count)
        .map(|output_index| (output_index as u128 * 60 / output_fps as u128) as usize)
        .take_while(|&source_index| source_index < source_count)
        .collect()
}

fn sequence_lifecycle(
    mut commands: Commands,
    status: Res<PipelineStatus>,
    mut runtime: ResMut<SequenceRuntime>,
    mut input: ResMut<RobotVisualInput>,
    mut camera_control: ResMut<StationCameraControl>,
    mut exit: MessageWriter<AppExit>,
    labels: Res<StationLabelBakeStatus>,
    robot_status: Option<Res<RobotVisualStatus>>,
) {
    runtime.frames += 1;
    if runtime.index == runtime.poses.len() {
        exit.write(AppExit::Success);
        return;
    }
    let expected = (
        runtime.poses[runtime.index].episode_id,
        runtime.poses[runtime.index].global_step,
        runtime.poses[runtime.index].episode_step,
    );
    let robot_ready = match robot_status.as_ref().map(|s| s.phase()) {
        Some(RobotVisualPhase::Ready {
            episode_id,
            global_step,
            episode_step,
        }) if (*episode_id, *global_step, *episode_step) == expected => true,
        Some(RobotVisualPhase::AwaitingModel | RobotVisualPhase::AwaitingFrame) => false,
        Some(RobotVisualPhase::Failed { reason }) => {
            status.0.lock().unwrap().error = Some(format!("Robot sequence display: {reason}"));
            false
        }
        Some(RobotVisualPhase::Ready { .. }) => {
            status.0.lock().unwrap().error =
                Some("Robot display accepted a different source tick than requested".into());
            false
        }
        None => {
            status.0.lock().unwrap().error = Some("Robot visual status is missing".into());
            false
        }
    };
    let state = status.0.lock().unwrap();
    if let Some(error) = &state.error {
        let frame = runtime.frames;
        if runtime.first_error_frame.is_none() {
            error!("{error}");
        }
        let first = *runtime.first_error_frame.get_or_insert(frame);
        if (frame - first >= 8 && state.pending == 0) || frame - first >= 180 {
            exit.write(AppExit::error());
        }
        return;
    }
    let ready = state.ready && robot_ready && labels.is_ready();
    let captured = state.captured;
    drop(state);
    if ready {
        runtime.ready_frames += 1;
    } else {
        runtime.ready_frames = 0;
    }
    if runtime.ready_frames >= 12 && !runtime.announced {
        runtime.announced = true;
        status.0.lock().unwrap().initialization_ready = true;
        info!("ROBOT_SEQUENCE_GPU_READY (completed pose display only)");
    }
    if !ready && runtime.started.elapsed() > Duration::from_secs(60) {
        status.0.lock().unwrap().error =
            Some("Sequence GPU pipelines/font/robot frame did not become ready".into());
        exit.write(AppExit::error());
        return;
    }
    if runtime.capture_requested && captured {
        let path = &runtime.paths[runtime.index];
        match std::fs::metadata(path) {
            Ok(meta) if meta.len() > 0 => {
                let mut state = status.0.lock().unwrap();
                state.captured = false;
                state.captured_count += 1;
                info!(
                    path = %path.display(),
                    global_step = expected.1,
                    "ROBOT_SEQUENCE_FRAME_SAVED"
                );
            }
            _ => {
                status.0.lock().unwrap().error = Some(format!(
                    "Sequence screenshot was not saved: {}",
                    path.display()
                ));
                exit.write(AppExit::error());
                return;
            }
        }
        runtime.index += 1;
        runtime.capture_requested = false;
        runtime.capture_started = None;
        runtime.ready_frames = 0;
        if runtime.index == runtime.poses.len() {
            exit.write(AppExit::Success);
        } else {
            runtime.started = Instant::now();
            let next = runtime.poses[runtime.index].clone();
            if let Some(root) = next.poses.iter().find(|pose| pose.source_body_id == 1) {
                camera_control.target = Vec3::from_array(root.translation) + Vec3::Y * 0.02;
            } else {
                status.0.lock().unwrap().error =
                    Some("Sequence next pose is missing the source root body".into());
                exit.write(AppExit::error());
                return;
            }
            input.publish(next);
        }
        return;
    }
    if runtime.announced && runtime.ready_frames >= 2 && !runtime.capture_requested {
        let path = runtime.paths[runtime.index].clone();
        commands.spawn(Screenshot::primary_window()).observe(
            move |capture: On<ScreenshotCaptured>, status: Res<PipelineStatus>| {
                let saved = capture
                    .image
                    .clone()
                    .try_into_dynamic()
                    .map_err(|e| format!("Screenshot conversion: {e}"))
                    .and_then(|image| {
                        image
                            .to_rgb8()
                            .save(&path)
                            .map_err(|e| format!("Screenshot {}: {e}", path.display()))
                    });
                let mut state = status.0.lock().unwrap();
                match saved {
                    Ok(()) => state.captured = true,
                    Err(error) => state.error = Some(error),
                }
            },
        );
        runtime.capture_requested = true;
        runtime.capture_started = Some(Instant::now());
    }
    if runtime
        .capture_started
        .is_some_and(|started| started.elapsed() > Duration::from_secs(90))
    {
        status.0.lock().unwrap().error = Some("Sequence screenshot capture timed out".into());
        exit.write(AppExit::error());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::body_pose::RobotBodyPose;

    #[test]
    fn sixty_hz_source_samples_at_thirty_and_twenty_four_fps_without_repeats() {
        assert_eq!(sampled_indices(10, 30), vec![0, 2, 4, 6, 8]);
        assert_eq!(sampled_indices(10, 24), vec![0, 2, 5, 7]);
    }

    fn pose(tick: u64, position: f32) -> Arc<RobotPoseFrame> {
        Arc::new(RobotPoseFrame {
            model_file_sha256: "synthetic_timeline_only".into(),
            episode_id: 3,
            global_step: 100 + tick,
            episode_step: tick,
            poses: vec![RobotBodyPose {
                source_body_id: 1,
                backend_handle: [1, 0],
                translation: [position, 0., 0.],
                rotation_xyzw: [0., 0., 0., 1.],
            }],
        })
    }

    #[test]
    fn capture_rejects_gaps_resets_and_repeated_static_images() {
        let mut sequence = vec![pose(0, 0.), pose(1, 0.), pose(2, 0.), pose(3, 0.)];
        validate_contiguous_steps(&sequence).unwrap();
        assert!(require_sampled_motion(&sequence, &sampled_indices(4, 30)).is_err());
        sequence[2] = pose(2, 0.1);
        require_sampled_motion(&sequence, &sampled_indices(4, 30)).unwrap();
        sequence[3] = pose(5, 0.2);
        assert!(validate_contiguous_steps(&sequence).is_err());
        sequence[3] = pose(3, 0.2);
        Arc::make_mut(&mut sequence[3]).episode_id += 1;
        assert!(validate_contiguous_steps(&sequence).is_err());
    }
}
