//! Development-only live window over the actual 60 Hz diagnostic world.
//!
//! The robot is driven by the existing P-only diagnostic controller. Neither
//! this window nor its report qualifies contact, BAM, props, or a skill.

use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::{CachedPipelineState, PipelineCache, PollType},
        renderer::RenderDevice,
    },
    shader::{Shader, ShaderCacheError},
};
use dev_tools_minigame::station_robot_diagnostic::{DiagnosticSession, initial_report, prepare};
use rendering_minigame::{
    RobotVisualInput, RobotVisualModel, RobotVisualPhase, RobotVisualPlugin, RobotVisualStatus,
    RobotVisualSystems, StationCameraControl, StationScene, StationView, StationVisualPlugin,
    robot_mesh::VerifiedRobotAppearance,
};
use robot_minigame::body_pose::RobotPoseFrame;
use serde_json::{Value, json};
use simulation_minigame::fixed_step_runtime::FrameOutcome;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
struct GpuReadiness {
    ready: bool,
    error: Option<String>,
    completed_render_passes: u64,
}

#[derive(Resource, Clone)]
struct SharedGpuReadiness(Arc<Mutex<GpuReadiness>>);

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
) {
    let mut status = readiness.0.lock().unwrap();
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
    status.completed_render_passes += 1;
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
    session: DiagnosticSession,
    output: PathBuf,
    started: Instant,
    last_running_frame: Option<Instant>,
    display_frames: u64,
    completed_publishes: u64,
    last_displayed_step: u64,
    pending_ticks: u128,
    awaiting_final_visual: bool,
    final_pose_render_passes: Option<u64>,
    observed_render_passes: u64,
    report_written: bool,
}

impl LiveRun {
    fn report(
        &mut self,
        passed: bool,
        error: Option<&str>,
        visual: &RobotVisualPhase,
    ) -> Result<(), String> {
        let actual_steps = self.session.global_step();
        let minimum_ticks = self.session.ticks();
        let counts_passed = self.session.finish_expected_steps(actual_steps);
        let report = self.session.report_mut();
        report["schema"] = json!("station_robot_live_preview_v1");
        report["scope"] =
            json!("dev_only_live_bevy_single_rapier_world_native_ort_p_only_diagnostic");
        report["minimum_ticks"] = json!(minimum_ticks);
        report["actual_completed_ticks"] = json!(actual_steps);
        report["overshoot_ticks"] = json!(actual_steps.saturating_sub(minimum_ticks as u64));
        report["display_frames"] = json!(self.display_frames);
        report["completed_publishes"] = json!(self.completed_publishes);
        report["last_displayed_step"] = json!(self.last_displayed_step);
        report["pending_ticks"] = json!(self.pending_ticks);
        report["robot_visual_status"] = json!(visual);
        report["props_physical"] = json!(false);
        report["live_visual_qualified"] = json!(false);
        report["performance_qualified"] = json!(false);
        report["completed_render_passes"] = json!(self.observed_render_passes);
        report["render_passes_after_final_publish"] = json!(
            self.observed_render_passes
                .saturating_sub(self.final_pose_render_passes.unwrap_or(0))
        );
        let accepted = passed && counts_passed && actual_steps >= minimum_ticks as u64;
        report["passed"] = json!(accepted);
        report["error"] = match error {
            Some(reason) => json!(reason),
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
    if !gpu_state.ready || !matches!(visual.phase(), RobotVisualPhase::Ready { .. }) {
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
    drop(gpu_state);
    let now = Instant::now();
    let elapsed = run
        .last_running_frame
        .replace(now)
        .map_or(Duration::ZERO, |last| now.duration_since(last));
    let outcome = match run.session.advance_frame(elapsed) {
        Ok(outcome) => outcome,
        Err(error) => {
            fail(&mut run, &mut input, &mut exit, error);
            return;
        }
    };
    let poses = run.session.take_published_frames();
    let pose = match display_pose_for_frame(
        &poses,
        outcome,
        run.last_displayed_step,
        run.session.global_step(),
    ) {
        Ok(pose) => pose,
        Err(error) => {
            fail(&mut run, &mut input, &mut exit, error);
            return;
        }
    };
    if let Some(last) = pose {
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
    run.completed_publishes += poses.len() as u64;
    run.pending_ticks = outcome.pending_ticks;
    run.display_frames += 1;
    if run.session.global_step() >= run.session.ticks() as u64 {
        run.awaiting_final_visual = true;
        run.final_pose_render_passes = Some(gpu.0.lock().unwrap().completed_render_passes);
    }
}

fn verify_final_visual(
    mut run: NonSendMut<LiveRun>,
    visual: Res<RobotVisualStatus>,
    gpu: Res<SharedGpuReadiness>,
    mut exit: MessageWriter<AppExit>,
    mut input: ResMut<RobotVisualInput>,
) {
    if !run.awaiting_final_visual || run.report_written {
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
    run.observed_render_passes = gpu.completed_render_passes;
    if run
        .final_pose_render_passes
        .is_none_or(|before| gpu.completed_render_passes <= before)
    {
        if run.started.elapsed() > Duration::from_secs(120) {
            drop(gpu);
            fail(
                &mut run,
                &mut input,
                &mut exit,
                "final pose was never followed by a GPU render pass".into(),
            );
        }
        return;
    }
    drop(gpu);
    let expected = run.session.global_step();
    match visual.phase() {
        RobotVisualPhase::Ready { global_step, .. } if *global_step == expected => {
            let phase = visual.phase().clone();
            match run.report(true, None, &phase) {
                Ok(()) if run.session.report()["passed"] == true => {
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
    Ok(())
}

fn run(args: &[String], output: PathBuf) -> Result<(), String> {
    let mut report = initial_report();
    report["schema"] = json!("station_robot_live_preview_v1");
    report["scope"] = json!("dev_only_live_bevy_single_rapier_world_native_ort_p_only_diagnostic");
    let session = match prepare(&args[..9], &mut report) {
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
        ..default()
    };
    let completion_report = output.clone();
    let live = LiveRun {
        session,
        output,
        started: Instant::now(),
        last_running_frame: None,
        display_frames: 0,
        completed_publishes: 0,
        last_displayed_step: 0,
        pending_ticks: 0,
        awaiting_final_visual: false,
        final_pose_render_passes: None,
        observed_render_passes: 0,
        report_written: false,
    };
    let readiness = SharedGpuReadiness(Arc::new(Mutex::new(GpuReadiness::default())));
    let mut app = App::new();
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .insert_resource(scene)
        .insert_resource(camera)
        .insert_resource(model)
        .insert_resource(input)
        .insert_resource(readiness.clone())
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
                        resolution: (1920, 1080).into(),
                        present_mode: bevy::window::PresentMode::AutoVsync,
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
    let result = app.run();
    let completed_report: Value = serde_json::from_slice(
        &fs::read(&completion_report)
            .map_err(|error| format!("read live completion report: {error}"))?,
    )
    .map_err(|error| format!("parse live completion report: {error}"))?;
    validate_live_exit(result, &completed_report)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 12 {
        return Err("usage: station_robot_live_preview MODEL MODEL_SHA QPOS QPOS_SHA ASSETS ONNX POLICY_CONTRACT|--legacy-original NATIVE_ORT_LIB MIN_TICKS APPEARANCE APPEARANCE_SHA NEW_REPORT.json".into());
    }
    let output = PathBuf::from(&args[11]);
    if output.exists() {
        return Err("live report output must be new".into());
    }
    let result = run(&args, output.clone());
    if let Err(error) = &result {
        if !output.exists() {
            let mut report = initial_report();
            report["schema"] = json!("station_robot_live_preview_v1");
            report["scope"] =
                json!("dev_only_live_bevy_single_rapier_world_native_ort_p_only_diagnostic");
            report["error"] = json!(error);
            fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        }
    }
    result.map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(validate_live_exit(AppExit::error(), &json!({"passed": true})).is_err());
    }
}
