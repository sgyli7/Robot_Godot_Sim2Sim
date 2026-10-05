//! Explicit fixed-task controls. The original automated captures stay unchanged.
//! Stop invalidates owner commands; only reset can admit a subsequent run.
use super::*;
use bevy::camera::RenderTarget;
use task_minigame::ui::{TaskUiAction, TaskUiPlugin, TaskUiPresentation, TaskUiStatus};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    /// Finite number of fresh worlds in this development window.
    pub maximum_episodes: u32,
    /// Exercise the same button message path with a finite native run.
    #[serde(default)]
    pub smoke_stop_reset: bool,
}
impl Configuration {
    pub(super) fn validate(&self, mode: CaptureMode, task_lab: bool) -> Result<(), String> {
        if !(2..=10).contains(&self.maximum_episodes)
            || mode != CaptureMode::StaticObservedPlace
            || task_lab
        {
            return Err("station controls require the separate static observed-place experiment and 2..=10 fresh episodes".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Serialize, PartialEq, Eq)]
enum Phase {
    Waiting,
    Running,
    Stopped,
    Complete,
    Failed,
}
#[derive(Clone, Copy)]
enum Smoke {
    Disabled,
    FirstStart,
    FirstStop,
    Paused { frame: u32, integrations: u64 },
    FreshStart,
    Finish { frame: u32 },
    Done,
}

pub(super) struct State {
    configuration: Configuration,
    root: PathBuf,
    initial_receipt: G1CaptureReceipt,
    initial_episode: u64,
    phase: Phase,
    episodes: u32,
    instruction: String,
    failure: Option<String>,
    events: Vec<serde_json::Value>,
    smoke: Smoke,
    stop_integrations: Option<u64>,
    stop_stable_frames: u32,
    smoke_passed: bool,
}
impl State {
    pub(super) fn running(&self) -> bool {
        self.phase == Phase::Running
    }
    pub(super) fn complete(&mut self) {
        self.phase = Phase::Complete;
    }
    pub(super) fn fail(&mut self, reason: String) {
        self.phase = Phase::Failed;
        self.failure = Some(reason);
    }
    fn event(&mut self, value: serde_json::Value) -> Result<(), String> {
        if self.events.len() >= 64 {
            return Err("station UI event budget exhausted".into());
        }
        self.events.push(value);
        Ok(())
    }
    fn save(&self, runtime: &CaptureRuntime, window_closed: bool) -> Result<(), String> {
        fs::write(self.root.join("station_controls_receipt.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"g1_native_fixed_static_task_controls_v1",
            "scope":"chinese_buttons_bound_experiment_stop_reset_fresh_world",
            "task_qualified":false,"Qwen_decision_admission":false,
            "original_VLA_then_disclosed_classical_grasp_place":true,
            "physics_hz":50,"integrations_per_tick":1,
            "initial_episode":self.initial_episode,"current_episode":runtime.episode_id,
            "phase":self.phase,"instruction":self.instruction,"failure":self.failure,
            "maximum_episodes":self.configuration.maximum_episodes,"episodes":self.episodes,
            "smoke_required":self.configuration.smoke_stop_reset,"smoke_passed":self.smoke_passed,
            "stop_stable_render_frames":self.stop_stable_frames,
            "window_closed":window_closed,"session_owner_trace":self.root.join("owner_steps.jsonl"),
            "current_artifact_directory":runtime.output,"owner_counts":runtime.lab_counts(),"events":self.events,
        })).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    }
}

pub(super) fn install(
    app: &mut App,
    configuration: Configuration,
    root: &Path,
    outcome: &CaptureOutcome,
) -> Result<(), String> {
    let baseline = outcome
        .0
        .lock()
        .map_err(|_| "controls receipt poisoned")?
        .clone();
    let runtime = &mut *app.world_mut().resource_mut::<CaptureRuntime>();
    runtime.controls = Some(State {
        smoke: if configuration.smoke_stop_reset {
            Smoke::FirstStart
        } else {
            Smoke::Disabled
        },
        configuration,
        root: root.into(),
        initial_receipt: baseline,
        initial_episode: runtime.episode_id,
        phase: Phase::Waiting,
        episodes: 1,
        instruction: String::new(),
        failure: None,
        events: Vec::new(),
        stop_integrations: None,
        stop_stable_frames: 0,
        smoke_passed: false,
    });
    runtime.controls.as_ref().unwrap().save(runtime, false)?;
    app.insert_resource(TaskUiPresentation {
        title: "G1 · 静态取放实验".into(),
        help: "仅支持下方固定任务。Ctrl+Enter 启动，Esc 暂停；再次启动前请重置。".into(),
        initial_instruction: "把苹果放到盘子里".into(),
    })
    .add_plugins(TaskUiPlugin)
    .add_systems(
        Update,
        (smoke_actions, apply_actions)
            .chain()
            .before(CaptureSystems::OwnerSnapshot),
    );
    Ok(())
}

fn bound_instruction(instruction: &str) -> bool {
    let text = instruction.trim();
    !text.is_empty()
        && text.len() <= 4096
        && text.chars().count() <= 1024
        && matches!(
            text,
            "把苹果放到盘子里" | "把苹果放到盘中" | "将苹果放入盘子"
        )
}
fn ready(runtime: &CaptureRuntime) -> bool {
    runtime.latest.as_ref().is_some_and(|snapshot| {
        snapshot.episode_id == runtime.episode_id
            && snapshot.phase == G1WorkerPhase::Paused
            && snapshot.timing.episode_integrations == 0
            && snapshot.measurement.is_some()
    }) && runtime.render_frames > 30
}

fn smoke_actions(
    mut runtime: ResMut<CaptureRuntime>,
    mut actions: MessageWriter<TaskUiAction>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut state) = runtime.controls.take() else {
        return;
    };
    let tick = runtime
        .latest
        .as_ref()
        .map_or(0, |s| s.timing.episode_integrations);
    let total = runtime
        .latest
        .as_ref()
        .map_or(0, |s| s.timing.total_integrations);
    match state.smoke {
        Smoke::Disabled | Smoke::Done => {}
        Smoke::FirstStart if ready(&runtime) => {
            actions.write(TaskUiAction::Start {
                instruction: "把苹果放到盘子里".into(),
            });
            state.smoke = Smoke::FirstStop;
        }
        Smoke::FirstStop if tick >= 60 => {
            actions.write(TaskUiAction::Stop);
            state.smoke = Smoke::Paused {
                frame: runtime.render_frames,
                integrations: total,
            };
        }
        Smoke::Paused {
            frame,
            integrations,
        } if runtime.render_frames >= frame + 30 => {
            // A single already-running boundary may complete after button delivery.
            // Beyond that, stopped physics must stay fixed for thirty actual renders.
            if state.stop_stable_frames < 20 || total > integrations + 1 {
                state.fail(
                    "physical Stop continued integrating during paused render interval".into(),
                );
                exit.write(AppExit::error());
                state.smoke = Smoke::Done;
            } else {
                actions.write(TaskUiAction::Reset);
                state.smoke = Smoke::FreshStart;
            }
        }
        Smoke::FreshStart if ready(&runtime) && runtime.episode_id > state.initial_episode => {
            actions.write(TaskUiAction::Start {
                instruction: "把苹果放到盘子里".into(),
            });
            state.smoke = Smoke::Finish {
                frame: runtime.render_frames,
            };
        }
        Smoke::Finish { .. } if state.phase == Phase::Failed => {
            let _ = state.save(&runtime, false);
            state.smoke = Smoke::Done;
            exit.write(AppExit::error());
        }
        Smoke::Finish { frame }
            if state.phase == Phase::Complete && runtime.render_frames > frame + 30 =>
        {
            state.smoke_passed = true;
            state.smoke = Smoke::Done;
            if let Err(reason) = state.save(&runtime, false) {
                state.fail(reason);
                exit.write(AppExit::error());
            } else {
                exit.write(AppExit::Success);
            }
        }
        _ => {}
    }
    runtime.controls = Some(state);
}

fn apply_actions(
    mut runtime: ResMut<CaptureRuntime>,
    outcome: Res<CaptureOutcome>,
    mut actions: MessageReader<TaskUiAction>,
    mut status: ResMut<TaskUiStatus>,
    port: Res<G1CameraPort>,
    mut mount: ResMut<G1ActiveCameraMount>,
    mut gate: Option<ResMut<G1TaskFiducialGate>>,
    mut robot: ResMut<G1BodyObservationInput>,
    mut task: ResMut<G1TaskVisualInput>,
    mut cameras: Query<(&RenderTarget, &mut Transform), With<Camera3d>>,
) {
    let Some(mut state) = runtime.controls.take() else {
        return;
    };
    let actions: Vec<_> = actions.read().cloned().collect();
    // Stop/reset outrank every start delivered in the same UI frame.
    let had_actions = !actions.is_empty();
    let start_allowed = !actions
        .iter()
        .any(|a| matches!(a, TaskUiAction::Stop | TaskUiAction::Reset));
    let result = (|| -> Result<(), String> {
        for action in actions {
            match action {
                TaskUiAction::Start { instruction } if start_allowed => {
                    if state.phase != Phase::Waiting || !ready(&runtime) {
                        status.failure = Some("请等待初始化；停止或完成后先重置".into());
                        continue;
                    }
                    if !bound_instruction(&instruction) {
                        status.failure = Some("当前固定任务入口仅支持：把苹果放到盘子里".into());
                        continue;
                    }
                    state.instruction = instruction;
                    state.phase = Phase::Running;
                    state.failure = None;
                    runtime.started = Instant::now();
                    state.event(serde_json::json!({"action":"start","episode":runtime.episode_id,"tick":0,"instruction":state.instruction}))?;
                }
                TaskUiAction::Start { .. } => {}
                TaskUiAction::Stop => {
                    runtime.lab_pause();
                    state.phase = Phase::Stopped;
                    state.stop_integrations = None;
                    state.stop_stable_frames = 0;
                    state.event(serde_json::json!({"action":"stop","episode":runtime.episode_id,"counts":runtime.lab_counts(),"resume_requires_reset":true}))?;
                }
                TaskUiAction::Reset => {
                    if state.episodes >= state.configuration.maximum_episodes {
                        return Err("window episode budget exhausted; close and relaunch".into());
                    }
                    runtime.lab_pause();
                    fs::write(
                        runtime.output.join("control_episode_receipt.json"),
                        serde_json::to_vec_pretty(
                            &*outcome.0.lock().map_err(|_| "receipt poisoned")?,
                        )
                        .map_err(|e| e.to_string())?,
                    )
                    .map_err(|e| e.to_string())?;
                    let next = runtime
                        .episode_id
                        .checked_add(1)
                        .ok_or("episode exhausted")?;
                    let directory = state.root.join(format!("episode_{next}"));
                    fs::create_dir(&directory).map_err(|e| e.to_string())?;
                    let calibration = runtime
                        .static_marker_worker
                        .as_ref()
                        .map(StaticMarkerWorker::reset_configuration);
                    let actual = runtime.lab_reset()?;
                    if actual != next {
                        return Err("reset generation mismatch".into());
                    }
                    runtime.output = directory;
                    runtime.static_marker_worker.take();
                    runtime.static_marker_worker = calibration
                        .map(|c| StaticMarkerWorker::spawn(c, &runtime.output, actual))
                        .transpose()?;
                    port.reset(actual)?;
                    mount.0 = G1CameraMountProfile::ArenaEgo;
                    if let Some(gate) = &mut gate {
                        gate.enabled = false;
                    }
                    robot.0 = None;
                    task.0 = None;
                    *outcome.0.lock().map_err(|_| "receipt poisoned")? =
                        state.initial_receipt.clone();
                    state.episodes += 1;
                    state.phase = Phase::Waiting;
                    state.failure = None;
                    state.stop_integrations = None;
                    state.event(serde_json::json!({"action":"reset","new_episode":actual,"old_requests_actions_and_history_cleared":true,"fresh_artifact_directory":runtime.output}))?;
                }
                TaskUiAction::Overview | TaskUiAction::GraspDetail => {
                    for (target, mut transform) in &mut cameras {
                        if matches!(target, RenderTarget::Window(_)) {
                            *transform = if action == TaskUiAction::Overview {
                                Transform::from_xyz(2.1, 1.45, 2.2)
                                    .looking_at(Vec3::new(0., 0.7, 0.), Vec3::Y)
                            } else {
                                Transform::from_xyz(1.15, 1.02, 0.65)
                                    .looking_at(Vec3::new(0.58, 0.82, -0.12), Vec3::Y)
                            };
                        }
                    }
                }
            }
        }
        if state.phase == Phase::Stopped {
            let total = runtime.latest.as_ref().map(|s| s.timing.total_integrations);
            if total == state.stop_integrations {
                state.stop_stable_frames += 1;
            } else {
                state.stop_integrations = total;
                state.stop_stable_frames = 0;
            }
        }
        let mut next_status = status.clone();
        next_status.task = if state.instruction.is_empty() {
            "把苹果放到盘子里".into()
        } else {
            state.instruction.clone()
        };
        next_status.profile = "固定静态取放实验（3/10，未达标）".into();
        next_status.stage = match state.phase {
            Phase::Waiting => "等待启动",
            Phase::Running => "执行中",
            Phase::Stopped => "已暂停；重置后才能再次启动",
            Phase::Complete => "执行结束；成功以独立验收结果为准",
            Phase::Failed => "执行失败；物理已暂停",
        }
        .into();
        next_status.model_service = "本地 N1.7；随后使用已披露的几何控制".into();
        next_status.last_decision = format!(
            "episode {}，实际 {} Tick",
            runtime.episode_id,
            runtime
                .latest
                .as_ref()
                .map_or(0, |s| s.timing.episode_integrations)
        );
        next_status.failure = state.failure.clone();
        status.set_if_neq(next_status);
        if !state.running() {
            runtime.lab_pause();
        }
        // Write on state changes/events only; never add disk work to physics.
        if had_actions || runtime.render_frames % 30 == 0 {
            state.save(&runtime, false)?;
        }
        Ok(())
    })();
    if let Err(reason) = result {
        runtime.lab_pause();
        state.fail(reason.clone());
        status.failure = Some(reason);
    }
    runtime.controls = Some(state);
}

pub(super) fn finish(_app: &mut App, root: &Path) -> Result<(), String> {
    // App::run consumes its runner and world. Use the persisted bounded receipt,
    // exactly as the native screenshot/owner evidence survives window shutdown.
    let path = root.join("station_controls_receipt.json");
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 128 * 1024 {
        return Err("station controls receipt is not an owned bounded file".into());
    }
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if value["schema"] != "g1_native_fixed_static_task_controls_v1" {
        return Err("foreign controls receipt".into());
    }
    value["window_closed"] = true.into();
    fs::write(
        path,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if value["smoke_required"] == true && value["smoke_passed"] != true {
        return Err("native UI stop/reset smoke did not finish".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_do_not_enable_unqualified_arbitrary_task_or_original_cli_mode() {
        let config = Configuration {
            maximum_episodes: 2,
            smoke_stop_reset: false,
        };
        assert!(
            config
                .validate(CaptureMode::StaticObservedPlace, false)
                .is_ok()
        );
        assert!(config.validate(CaptureMode::Camera, false).is_err());
        assert!(
            config
                .validate(CaptureMode::StationMobileRelease, false)
                .is_err()
        );
        assert!(
            config
                .validate(CaptureMode::StaticObservedPlace, true)
                .is_err()
        );
        assert!(bound_instruction("把苹果放到盘子里"));
        assert!(!bound_instruction("把苹果放到地板上"));
        assert!(!bound_instruction(""));
    }
}
