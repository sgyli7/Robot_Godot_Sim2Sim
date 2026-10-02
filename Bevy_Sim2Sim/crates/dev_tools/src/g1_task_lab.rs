//! Interactive native-camera development window. All physical execution
//! capabilities remain disabled until independent qualification is delivered.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bevy::{
    camera::RenderTarget,
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
};
use rendering_minigame::g1_camera::{G1BodyObservationInput, G1CameraPort};
use rendering_minigame::g1_task_visual::{G1TaskVisualInput, G1TaskVisualStatus};
use rendering_minigame::{StationRenderHealth, g1_visual::G1VisualStatus};
use serde::{Deserialize, Serialize};
use task_minigame::{
    decision::{CameraRgb, ExecutionFeedback, FeedbackStatus},
    interactive::{
        InteractiveDecisionConfig, TaskCameraObservation, TaskDecisionPlugin, TaskDecisionSystems,
        TaskDecisionTelemetry, TaskExecutionFeedback, TaskOwnerClock, TaskRuntimeIntent,
    },
    types::{SkillAvailability, SkillRequest, TaskProfile},
    ui::{TaskUiAction, TaskUiPlugin, TaskUiStatus},
};

use super::g1_capture::{CaptureRuntime, CaptureSystems};

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LabSystems {
    Clock,
    Consume,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1TaskLabConfiguration {
    pub qwen: InteractiveDecisionConfig,
    /// Finite development injection through the same intent path as UI buttons.
    #[serde(default)]
    pub smoke_reset_during_request: bool,
}

#[derive(Clone, Resource)]
pub(super) struct LabOutcome(pub Arc<Mutex<LabReceipt>>);

#[derive(Serialize)]
pub(super) struct LabReceipt {
    schema: &'static str,
    scope: &'static str,
    task_qualified: bool,
    available_skills: SkillAvailability,
    pub window_closed: bool,
    pub smoke_required: bool,
    pub smoke_succeeded: bool,
    pub failure: Option<String>,
    owner_counts: serde_json::Value,
    http_attempts: u64,
    http_results: u64,
    discarded_old_results: u64,
    accepted_decisions: u64,
    reset_requests: u64,
    pause_requests: u64,
    observed_frames: u64,
    last_stage: String,
    last_decision: String,
    last_failure: Option<String>,
    events: Vec<serde_json::Value>,
}

#[derive(Clone, Copy)]
enum SmokePhase {
    Disabled,
    Owner,
    FirstRequest,
    Reset,
    FreshDecision,
    Ui { first_render_frame: u64 },
    Screenshot,
}

#[derive(Resource)]
struct LabRuntime {
    output: PathBuf,
    requested: Option<(u64, u64)>,
    started: Instant,
    initial_episode: u64,
    smoke: SmokePhase,
    screenshot: Arc<Mutex<Option<Result<(), String>>>>,
}

pub(super) fn install(
    app: &mut App,
    configuration: G1TaskLabConfiguration,
    episode_id: u64,
    profile: TaskProfile,
    output: &Path,
) -> Result<LabOutcome, String> {
    let outcome = LabOutcome(Arc::new(Mutex::new(LabReceipt {
        schema: "g1_interactive_local_decision_lab_v1",
        scope: "native_rgb_self_state_ui_qwen_reset_observe_stop_only",
        task_qualified: false,
        available_skills: SkillAvailability::default(),
        window_closed: false,
        smoke_required: configuration.smoke_reset_during_request,
        smoke_succeeded: false,
        failure: None,
        owner_counts: serde_json::Value::Null,
        http_attempts: 0,
        http_results: 0,
        discarded_old_results: 0,
        accepted_decisions: 0,
        reset_requests: 0,
        pause_requests: 0,
        observed_frames: 0,
        last_stage: "初始化".into(),
        last_decision: "无".into(),
        last_failure: None,
        events: Vec::new(),
    })));
    app.add_plugins(TaskUiPlugin);
    TaskDecisionPlugin {
        configuration: configuration.qwen,
        episode_id,
        profile,
        availability: SkillAvailability::default(),
    }
    .install(app)?;
    app.world_mut().resource_mut::<TaskOwnerClock>().episode_id = episode_id;
    app.insert_resource(outcome.clone())
        .insert_resource(LabRuntime {
            output: output.to_owned(),
            requested: None,
            started: Instant::now(),
            initial_episode: episode_id,
            smoke: if configuration.smoke_reset_during_request {
                SmokePhase::Owner
            } else {
                SmokePhase::Disabled
            },
            screenshot: Arc::new(Mutex::new(None)),
        })
        .configure_sets(
            Update,
            (
                CaptureSystems::OwnerSnapshot,
                LabSystems::Clock,
                TaskDecisionSystems::Decision,
                LabSystems::Consume,
            )
                .chain(),
        )
        .add_systems(Update, publish_owner_clock.in_set(LabSystems::Clock))
        .add_systems(
            Update,
            (apply_intents, receive_camera, smoke_and_receipt)
                .chain()
                .in_set(LabSystems::Consume),
        );
    Ok(outcome)
}

fn publish_owner_clock(
    runtime: Res<CaptureRuntime>,
    mut clock: ResMut<TaskOwnerClock>,
    health: Res<StationRenderHealth>,
    visual: Res<G1VisualStatus>,
    task: Res<G1TaskVisualStatus>,
    robot: Res<G1BodyObservationInput>,
) {
    if let Some(mut actual) = runtime.lab_clock() {
        let tick = actual.sim_time_ns / 20_000_000;
        actual.ready &= health.snapshot().ready
            && robot.0.is_some()
            && visual.source_tick == Some(tick)
            && task.source_tick == Some(tick)
            && task.episode_id == Some(actual.episode_id);
        *clock = actual;
    } else {
        clock.ready = false;
    }
}

fn event(outcome: &LabOutcome, value: serde_json::Value) -> Result<(), String> {
    let mut receipt = outcome.0.lock().map_err(|_| "lab receipt poisoned")?;
    if receipt.events.len() >= 1024 {
        return Err("bounded interactive evidence event budget exhausted".into());
    }
    receipt.events.push(value);
    Ok(())
}

fn apply_intents(
    mut runtime: ResMut<CaptureRuntime>,
    mut lab: ResMut<LabRuntime>,
    outcome: Res<LabOutcome>,
    clock: Res<TaskOwnerClock>,
    mut status: ResMut<TaskUiStatus>,
    mut intents: MessageReader<TaskRuntimeIntent>,
    mut feedback: MessageWriter<TaskExecutionFeedback>,
    port: Res<G1CameraPort>,
    mut robot_input: ResMut<G1BodyObservationInput>,
    mut task_input: ResMut<G1TaskVisualInput>,
    mut cameras: Query<(&RenderTarget, &mut Transform), With<Camera3d>>,
    mut exit: MessageWriter<AppExit>,
) {
    for intent in intents.read() {
        let result = (|| -> Result<(), String> {
            match intent {
                TaskRuntimeIntent::Pause { reason } => {
                    runtime.lab_pause();
                    lab.requested = None;
                    outcome.0.lock().unwrap().pause_requests += 1;
                    event(
                        &outcome,
                        serde_json::json!({"kind":"pause","episode_id":clock.episode_id,"reason":reason,"sim_time_ns":clock.sim_time_ns}),
                    )?;
                }
                TaskRuntimeIntent::Reset => {
                    let episode_id = runtime.lab_reset()?;
                    port.reset(episode_id)?;
                    lab.requested = None;
                    robot_input.0 = None;
                    task_input.0 = None;
                    outcome.0.lock().unwrap().reset_requests += 1;
                    event(
                        &outcome,
                        serde_json::json!({"kind":"reset","requested_episode_id":episode_id,"old_episode_id":clock.episode_id}),
                    )?;
                }
                TaskRuntimeIntent::Capture { episode_id } => {
                    if *episode_id != clock.episode_id || !clock.ready {
                        return Err(
                            "interactive capture requires current ready owner generation".into(),
                        );
                    }
                    if lab.requested.is_some() {
                        return Err("interactive capture already pending".into());
                    }
                    // Old GPU maps can still finish after reset. The port owns
                    // one buffer and rejects their old-generation results.
                    lab.requested = Some((*episode_id, 0));
                }
                TaskRuntimeIntent::Overview | TaskRuntimeIntent::GraspDetail => {
                    for (target, mut transform) in &mut cameras {
                        if matches!(target, RenderTarget::Window(_)) {
                            *transform = if matches!(intent, TaskRuntimeIntent::Overview) {
                                Transform::from_xyz(2.1, 1.45, 2.2)
                                    .looking_at(Vec3::new(0., 0.7, 0.), Vec3::Y)
                            } else {
                                Transform::from_xyz(1.15, 1.02, 0.65)
                                    .looking_at(Vec3::new(0.58, 0.82, -0.12), Vec3::Y)
                            };
                        }
                    }
                }
                TaskRuntimeIntent::Admitted(decision) => {
                    let (feedback_status, summary) = match &decision.request {
                        SkillRequest::Observe => (
                            FeedbackStatus::Succeeded,
                            "当前视觉观察完成；未执行物理任务".to_owned(),
                        ),
                        SkillRequest::Stop { reason } => {
                            runtime.lab_pause();
                            (
                                FeedbackStatus::SafelyStopped,
                                format!("明确暂停仿真：{reason}"),
                            )
                        }
                        SkillRequest::ExecuteTask { .. }
                        | SkillRequest::NavigateAdjustment { .. } => {
                            runtime.lab_pause();
                            (FeedbackStatus::Rejected, "本窗口未开放物理执行资格".into())
                        }
                    };
                    status.stage = summary.clone();
                    event(
                        &outcome,
                        serde_json::json!({"kind":"decision_admitted","decision_id":decision.decision_id,"stamp":decision.stamp,"request":decision.request,"reason":decision.reason,"service_elapsed_ms":decision.service_elapsed_ms,"image_to_decision_ms":decision.image_to_decision_ms,"feedback":summary,"task_qualified":false}),
                    )?;
                    feedback.write(TaskExecutionFeedback(ExecutionFeedback {
                        episode_id: clock.episode_id,
                        decision_id: decision.decision_id,
                        status: feedback_status,
                        summary,
                        sim_time_ns: clock.sim_time_ns,
                    }));
                }
            }
            Ok(())
        })();
        if let Err(reason) = result {
            runtime.lab_pause();
            status.failure = Some(reason.clone());
            outcome.0.lock().unwrap().failure = Some(reason);
            exit.write(AppExit::error());
        }
    }
}

fn receive_camera(
    mut lab: ResMut<LabRuntime>,
    outcome: Res<LabOutcome>,
    clock: Res<TaskOwnerClock>,
    port: Res<G1CameraPort>,
    mut frames: MessageWriter<TaskCameraObservation>,
    mut status: ResMut<TaskUiStatus>,
    runtime: Res<CaptureRuntime>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some((episode_id, 0)) = lab.requested
        && episode_id == clock.episode_id
        && clock.ready
        && port.progress() == "idle"
    {
        match port.request() {
            Ok(sequence) => {
                lab.requested = Some((episode_id, sequence));
                if let Err(reason) = event(
                    &outcome,
                    serde_json::json!({"kind":"capture_requested","episode_id":episode_id,"capture_sequence":sequence}),
                ) {
                    outcome.0.lock().unwrap().failure = Some(reason);
                    exit.write(AppExit::error());
                }
            }
            Err(reason) => {
                outcome.0.lock().unwrap().failure = Some(reason);
                exit.write(AppExit::error());
            }
        }
    }
    let Some(frame) = port.take() else {
        return;
    };
    let Some((episode_id, sequence)) = lab.requested.take() else {
        return;
    };
    if sequence == 0 {
        lab.requested = Some((episode_id, 0));
        return;
    }
    let result = (|| -> Result<(), String> {
        let frame = frame?;
        if frame.stamp.episode_id != episode_id
            || clock.episode_id != episode_id
            || frame.stamp.capture_sequence != sequence
            || frame.stamp.sim_time_ns != clock.sim_time_ns
        {
            return Err("interactive RGB identity differs from paused native owner".into());
        }
        let png = CameraRgb::from_rgb("g1_head_rgb", frame.width, frame.height, frame.rgb)
            .map_err(|error| error.to_string())?
            .png()
            .to_vec();
        let observation =
            super::g1_decision_diagnostic::snapshot_from_capture(&frame.stamp, png.clone())?;
        let stem = format!("camera_e{episode_id}_s{sequence}");
        fs::write(lab.output.join(format!("{stem}.png")), png)
            .map_err(|error| error.to_string())?;
        fs::write(
            lab.output.join(format!("{stem}.json")),
            serde_json::to_vec_pretty(&frame.stamp).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        outcome.0.lock().unwrap().observed_frames += 1;
        event(
            &outcome,
            serde_json::json!({"kind":"native_camera_observation","stamp":observation.stamp,"saved_stem":stem,"world_truth_in_decision":false}),
        )?;
        frames.write(TaskCameraObservation(observation));
        Ok(())
    })();
    if let Err(reason) = result {
        runtime.lab_pause();
        status.failure = Some(reason.clone());
        outcome.0.lock().unwrap().failure = Some(reason);
        exit.write(AppExit::error());
    }
}

fn smoke_and_receipt(
    mut commands: Commands,
    mut lab: ResMut<LabRuntime>,
    outcome: Res<LabOutcome>,
    runtime: Res<CaptureRuntime>,
    clock: Res<TaskOwnerClock>,
    telemetry: Res<TaskDecisionTelemetry>,
    status: Res<TaskUiStatus>,
    mut actions: MessageWriter<TaskUiAction>,
    mut exit: MessageWriter<AppExit>,
) {
    {
        let mut receipt = outcome.0.lock().unwrap();
        receipt.owner_counts = runtime.lab_counts();
        receipt.http_attempts = telemetry.http_attempts;
        receipt.http_results = telemetry.http_results;
        receipt.discarded_old_results = telemetry.discarded_old_results;
        receipt.accepted_decisions = telemetry.accepted_decisions;
        receipt.last_stage = status.stage.clone();
        receipt.last_decision = status.last_decision.clone();
        receipt.last_failure = status.failure.clone();
    }
    match lab.smoke {
        SmokePhase::Disabled => return,
        SmokePhase::Owner if clock.ready => {
            actions.write(TaskUiAction::Start {
                instruction: "观察当前场景中的苹果和盘子，报告下一步。".into(),
            });
            lab.smoke = SmokePhase::FirstRequest;
        }
        SmokePhase::FirstRequest if telemetry.http_attempts >= 1 && telemetry.waiting_for_model => {
            actions.write(TaskUiAction::Reset);
            lab.smoke = SmokePhase::Reset;
        }
        SmokePhase::Reset if clock.ready && clock.episode_id == lab.initial_episode + 2 => {
            actions.write(TaskUiAction::Start {
                instruction: "观察当前场景中的苹果和盘子，报告下一步。".into(),
            });
            lab.smoke = SmokePhase::FreshDecision;
        }
        SmokePhase::FreshDecision if status.failure.is_some() => {
            runtime.lab_pause();
            outcome.0.lock().unwrap().failure = status.failure.clone();
            exit.write(AppExit::error());
        }
        SmokePhase::FreshDecision
            if telemetry.accepted_decisions == 1
                && telemetry.discarded_old_results >= 1
                && clock.episode_id == lab.initial_episode + 3 =>
        {
            lab.smoke = SmokePhase::Ui {
                first_render_frame: runtime.lab_render_frames(),
            };
        }
        // UI layout and render extraction can trail the owner-side resource.
        // Wait actual display updates before asking for the native screenshot.
        SmokePhase::Ui { first_render_frame }
            if runtime.lab_render_frames() >= first_render_frame + 4 =>
        {
            let path = lab.output.join("task_lab_window.png");
            let completed = lab.screenshot.clone();
            commands.spawn(Screenshot::primary_window()).observe(
                move |capture: On<ScreenshotCaptured>| {
                    let result = capture
                        .image
                        .clone()
                        .try_into_dynamic()
                        .map_err(|error| error.to_string())
                        .and_then(|image| image.save(&path).map_err(|error| error.to_string()));
                    *completed.lock().unwrap() = Some(result);
                },
            );
            lab.smoke = SmokePhase::Screenshot;
        }
        SmokePhase::Screenshot => {
            let result = lab.screenshot.lock().unwrap().clone();
            if let Some(result) = result {
                match result {
                    Ok(()) => {
                        outcome.0.lock().unwrap().smoke_succeeded = true;
                        runtime.lab_pause();
                        exit.write(AppExit::Success);
                    }
                    Err(reason) => {
                        outcome.0.lock().unwrap().failure = Some(reason);
                        exit.write(AppExit::error());
                    }
                }
            }
        }
        SmokePhase::Owner
        | SmokePhase::FirstRequest
        | SmokePhase::Reset
        | SmokePhase::FreshDecision
        | SmokePhase::Ui { .. } => {}
    }
    if lab.started.elapsed() > Duration::from_secs(90) {
        runtime.lab_pause();
        outcome.0.lock().unwrap().failure =
            Some("finite interactive reset/decision smoke exceeded 90 seconds".into());
        exit.write(AppExit::error());
    }
}
