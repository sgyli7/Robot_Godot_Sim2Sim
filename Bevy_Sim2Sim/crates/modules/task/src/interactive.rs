//! UI-to-local-decision orchestration. The application supplies a completed
//! camera observation and owner clock, and consumes bounded execution intents.
//! This module cannot read physics truth or modify a robot/world.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use serde::Deserialize;

use crate::{
    decision::{
        DecisionLimits, DecisionSession, DecisionWorker, ExecutionFeedback, FeedbackStatus,
        LocalQwenClient, LocalQwenConfig, ObservationSnapshot, TaskGoal, ValidatedDecision,
    },
    types::{SkillAvailability, SkillRequest, TaskProfile},
    ui::{TaskUiAction, TaskUiStatus},
};

/// Connection settings only; capabilities are supplied by the application.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractiveDecisionConfig {
    pub endpoint: String,
    pub model: String,
    pub timeout_ms: u64,
    pub max_output_tokens: u32,
}

/// The application publishes actual owner identity before polling decisions.
#[derive(Resource, Default)]
pub struct TaskOwnerClock {
    pub episode_id: u64,
    pub sim_time_ns: u64,
    pub ready: bool,
    pub failure: Option<String>,
}

#[derive(Message)]
pub struct TaskCameraObservation(pub ObservationSnapshot);

#[derive(Message)]
pub struct TaskExecutionFeedback(pub ExecutionFeedback);

/// Transport counts do not imply that the server ran or succeeded. Admission
/// and actual physics counters remain separate in the application receipt.
#[derive(Resource, Default, serde::Serialize)]
pub struct TaskDecisionTelemetry {
    pub http_attempts: u64,
    pub http_results: u64,
    pub discarded_old_results: u64,
    pub accepted_decisions: u64,
    pub waiting_for_model: bool,
}

#[derive(Message)]
pub enum TaskRuntimeIntent {
    Pause { reason: String },
    Reset,
    Capture { episode_id: u64 },
    Overview,
    GraspDetail,
    Admitted(ValidatedDecision),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Resetting { expected_episode: u64 },
    Camera,
    Model,
}

struct InteractiveRuntime {
    worker: DecisionWorker,
    episode_id: u64,
    profile: TaskProfile,
    availability: SkillAvailability,
    session: Option<DecisionSession>,
    pending_goal: Option<String>,
    phase: Phase,
}

pub struct TaskDecisionPlugin {
    pub configuration: InteractiveDecisionConfig,
    pub episode_id: u64,
    pub profile: TaskProfile,
    pub availability: SkillAvailability,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskDecisionSystems {
    Decision,
}

impl TaskDecisionPlugin {
    /// Validate transport before app.run; no model call or service launch occurs.
    pub fn install(self, app: &mut App) -> Result<(), String> {
        let config = self.configuration;
        let client = LocalQwenClient::new(LocalQwenConfig {
            base_url: config.endpoint.clone(),
            model: config.model,
            timeout: Duration::from_millis(config.timeout_ms),
            max_output_tokens: config.max_output_tokens,
        })
        .map_err(|error| error.to_string())?;
        let worker =
            DecisionWorker::spawn(client, self.episode_id).map_err(|error| error.to_string())?;
        if !app.world().contains_resource::<Messages<TaskUiAction>>() {
            app.add_message::<TaskUiAction>();
        }
        app.init_resource::<TaskUiStatus>()
            .init_resource::<TaskOwnerClock>()
            .init_resource::<TaskDecisionTelemetry>()
            .add_message::<TaskCameraObservation>()
            .add_message::<TaskExecutionFeedback>()
            .add_message::<TaskRuntimeIntent>()
            .insert_non_send(InteractiveRuntime {
                worker,
                episode_id: self.episode_id,
                profile: self.profile,
                availability: self.availability,
                session: None,
                pending_goal: None,
                phase: Phase::Idle,
            })
            .add_systems(
                Update,
                drive_decisions.in_set(TaskDecisionSystems::Decision),
            );
        let mut status = app.world_mut().resource_mut::<TaskUiStatus>();
        status.model_service = format!("等待本地调用 · {}", config.endpoint);
        status.profile = match self.profile {
            TaskProfile::StaticApple => "静态苹果／盘 · 执行资格待验证",
            TaskProfile::MobileBox => "移动箱子／容器 · 执行资格待验证",
        }
        .into();
        Ok(())
    }
}

fn unix_ms() -> Result<u64, String> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis()
        .try_into()
        .map_err(|_| "wall clock exceeds u64")?)
}

fn pause(
    runtime: &mut InteractiveRuntime,
    status: &mut TaskUiStatus,
    intents: &mut MessageWriter<TaskRuntimeIntent>,
    reason: String,
) {
    runtime.pending_goal = None;
    runtime.phase = Phase::Idle;
    if let Some(session) = &mut runtime.session {
        session.cancel_pending(reason.clone());
        session.take_safe_stop();
    }
    status.stage = "仿真暂停".into();
    intents.write(TaskRuntimeIntent::Pause { reason });
}

fn drive_decisions(
    mut runtime: NonSendMut<InteractiveRuntime>,
    clock: Res<TaskOwnerClock>,
    mut status: ResMut<TaskUiStatus>,
    mut actions: MessageReader<TaskUiAction>,
    mut observations: MessageReader<TaskCameraObservation>,
    mut feedback: MessageReader<TaskExecutionFeedback>,
    mut intents: MessageWriter<TaskRuntimeIntent>,
    mut telemetry: ResMut<TaskDecisionTelemetry>,
) {
    let (attempts, results, discarded) = runtime.worker.transport_counts();
    telemetry.http_attempts = attempts;
    telemetry.http_results = results;
    telemetry.discarded_old_results = discarded;
    telemetry.waiting_for_model = runtime.phase == Phase::Model;
    let now = match unix_ms() {
        Ok(now) => now,
        Err(reason) => {
            pause(&mut runtime, &mut status, &mut intents, reason.clone());
            status.failure = Some(reason);
            return;
        }
    };
    // Reset/stop wins over every same-frame start, before receiving HTTP results.
    let actions: Vec<_> = actions.read().cloned().collect();
    let reset = actions
        .iter()
        .any(|action| matches!(action, TaskUiAction::Reset));
    let stop = actions
        .iter()
        .any(|action| matches!(action, TaskUiAction::Stop));
    if reset || stop {
        pause(
            &mut runtime,
            &mut status,
            &mut intents,
            if reset {
                "用户重置"
            } else {
                "用户停止；未验证持物等待，明确暂停仿真"
            }
            .into(),
        );
        if reset {
            intents.write(TaskRuntimeIntent::Reset);
            status.stage = "重建场景并清空历史".into();
        }
        observations.clear();
    } else if let Some(instruction) = actions.iter().rev().find_map(|action| {
        if let TaskUiAction::Start { instruction } = action {
            Some(instruction.clone())
        } else {
            None
        }
    }) {
        let goal = TaskGoal {
            instruction: instruction.clone(),
            profile: runtime.profile,
            public_scene_description: String::new(),
        };
        if let Err(error) = goal.validate() {
            status.failure = Some(error.to_string());
        } else if let Some(expected_episode) = clock.episode_id.checked_add(1) {
            pause(
                &mut runtime,
                &mut status,
                &mut intents,
                "新任务从重置场景开始".into(),
            );
            runtime.pending_goal = Some(instruction.clone());
            runtime.phase = Phase::Resetting { expected_episode };
            intents.write(TaskRuntimeIntent::Reset);
            status.task = instruction;
            status.stage = "初始化新任务／等待场景重置".into();
            status.failure = None;
            status.last_decision = "等待新图像".into();
            observations.clear();
        } else {
            status.failure = Some("episode identity exhausted".into());
        }
    }
    for action in &actions {
        match action {
            TaskUiAction::Overview => {
                intents.write(TaskRuntimeIntent::Overview);
            }
            TaskUiAction::GraspDetail => {
                intents.write(TaskRuntimeIntent::GraspDetail);
            }
            TaskUiAction::Start { .. } | TaskUiAction::Stop | TaskUiAction::Reset => {}
        }
    }
    if clock.episode_id > runtime.episode_id {
        if let Err(error) = runtime.worker.reset_episode(clock.episode_id) {
            status.failure = Some(error.to_string());
            pause(&mut runtime, &mut status, &mut intents, error.to_string());
            return;
        }
        runtime.episode_id = clock.episode_id;
        runtime.session = None;
    }
    if let Some(reason) = &clock.failure {
        status.failure = Some(reason.clone());
        pause(&mut runtime, &mut status, &mut intents, reason.clone());
        observations.clear();
        return;
    }
    for feedback in feedback.read() {
        if feedback.0.episode_id != clock.episode_id {
            continue;
        }
        if let Some(session) = &mut runtime.session {
            if let Err(error) = session.record_feedback(feedback.0.clone()) {
                status.failure = Some(error.to_string());
            } else {
                status.stage = format!("执行反馈：{}", feedback.0.summary);
                if matches!(
                    feedback.0.status,
                    FeedbackStatus::Failed | FeedbackStatus::Rejected
                ) {
                    status.failure = Some(feedback.0.summary.clone());
                    pause(
                        &mut runtime,
                        &mut status,
                        &mut intents,
                        feedback.0.summary.clone(),
                    );
                }
            }
        }
    }
    if let Some(session) = &mut runtime.session {
        if session.expire_pending(now, clock.sim_time_ns) {
            let reason = "决策图像已过期；暂停仿真".to_string();
            status.failure = Some(reason.clone());
            pause(&mut runtime, &mut status, &mut intents, reason);
        }
    }
    match runtime.worker.try_recv() {
        Ok(Some(reply)) => {
            if runtime.phase == Phase::Model {
                let result = runtime
                    .session
                    .as_mut()
                    .ok_or("missing decision session".into())
                    .and_then(|session| {
                        session
                            .accept(reply, now, clock.sim_time_ns)
                            .map_err(|e| e.to_string())
                    });
                match result {
                    Ok(decision) => {
                        telemetry.accepted_decisions += 1;
                        let skill = match &decision.request {
                            SkillRequest::Observe => "观察",
                            SkillRequest::Stop { .. } => "停止",
                            SkillRequest::ExecuteTask {
                                task_profile: TaskProfile::StaticApple,
                                ..
                            } => "静态苹果取放",
                            SkillRequest::ExecuteTask {
                                task_profile: TaskProfile::MobileBox,
                                ..
                            } => "移动搬箱",
                            SkillRequest::NavigateAdjustment { .. } => "导航调整",
                        };
                        status.last_decision = format!("{skill} · {}", decision.reason);
                        status.model_service = format!(
                            "本地响应 {} ms · 图像年龄 {} ms",
                            decision.service_elapsed_ms, decision.image_to_decision_ms
                        );
                        status.stage = "决策已验证／等待执行反馈".into();
                        runtime.phase = Phase::Idle;
                        intents.write(TaskRuntimeIntent::Admitted(decision));
                    }
                    Err(reason) => {
                        status.failure = Some(reason.clone());
                        pause(&mut runtime, &mut status, &mut intents, reason);
                    }
                }
            }
            // Cancelled/reset results cannot activate a new session.
        }
        Ok(None) => {}
        Err(error) => {
            status.failure = Some(error.to_string());
            pause(&mut runtime, &mut status, &mut intents, error.to_string());
        }
    }
    if let Phase::Resetting { expected_episode } = runtime.phase {
        if clock.ready && clock.episode_id == expected_episode && !runtime.worker.is_busy() {
            let goal = TaskGoal {
                instruction: runtime
                    .pending_goal
                    .take()
                    .expect("goal exists during reset"),
                profile: runtime.profile,
                public_scene_description: String::new(),
            };
            match DecisionSession::new(
                clock.episode_id,
                goal,
                runtime.availability,
                DecisionLimits::default(),
            ) {
                Ok(session) => {
                    runtime.session = Some(session);
                    runtime.phase = Phase::Camera;
                    status.stage = "获取当前本体相机图像".into();
                    intents.write(TaskRuntimeIntent::Capture {
                        episode_id: clock.episode_id,
                    });
                }
                Err(error) => {
                    status.failure = Some(error.to_string());
                    pause(&mut runtime, &mut status, &mut intents, error.to_string());
                }
            }
        }
    }
    for observation in observations.read() {
        if runtime.phase != Phase::Camera {
            continue;
        }
        let result = runtime
            .session
            .as_mut()
            .ok_or("missing camera decision session".into())
            .and_then(|session| {
                session
                    .prepare(observation.0.clone(), now, clock.sim_time_ns)
                    .map_err(|e| e.to_string())
            });
        match result {
            Ok(input) => match runtime.worker.try_submit(input) {
                Ok(()) => {
                    runtime.phase = Phase::Model;
                    status.stage = "本地视觉决策中／仿真暂停".into();
                    status.model_service = "Qwen 本地推理中".into();
                }
                Err(error) => {
                    status.failure = Some(error.to_string());
                    pause(&mut runtime, &mut status, &mut intents, error.to_string());
                }
            },
            Err(reason) => {
                status.failure = Some(reason.clone());
                pause(&mut runtime, &mut status, &mut intents, reason);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        TaskDecisionPlugin {
            configuration: InteractiveDecisionConfig {
                endpoint: "http://127.0.0.1:1/v1".into(),
                model: "test".into(),
                timeout_ms: 100,
                max_output_tokens: 512,
            },
            episode_id: 1,
            profile: TaskProfile::StaticApple,
            availability: SkillAvailability::default(),
        }
        .install(&mut app)
        .unwrap();
        *app.world_mut().resource_mut::<TaskOwnerClock>() = TaskOwnerClock {
            episode_id: 1,
            ready: true,
            ..default()
        };
        app
    }

    #[test]
    fn same_frame_stop_prevents_start_and_model_submission() {
        let mut app = app();
        app.world_mut().write_message(TaskUiAction::Start {
            instruction: "观察苹果".into(),
        });
        app.world_mut().write_message(TaskUiAction::Stop);
        app.update();
        let runtime = app.world().non_send::<InteractiveRuntime>();
        assert!(runtime.pending_goal.is_none());
        assert!(runtime.phase == Phase::Idle && !runtime.worker.is_busy());
        let mut cursor = app
            .world()
            .resource::<Messages<TaskRuntimeIntent>>()
            .get_cursor();
        let intents: Vec<_> = cursor
            .read(app.world().resource::<Messages<TaskRuntimeIntent>>())
            .collect();
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0], TaskRuntimeIntent::Pause { .. }));
    }

    #[test]
    fn start_waits_for_actual_new_generation_before_requesting_camera() {
        let mut app = app();
        app.world_mut().write_message(TaskUiAction::Start {
            instruction: "把苹果放盘里".into(),
        });
        app.update();
        let runtime = app.world().non_send::<InteractiveRuntime>();
        assert!(
            runtime.phase
                == Phase::Resetting {
                    expected_episode: 2
                }
        );
        assert!(!runtime.worker.is_busy());
        *app.world_mut().resource_mut::<TaskOwnerClock>() = TaskOwnerClock {
            episode_id: 2,
            ready: true,
            ..default()
        };
        app.update();
        let runtime = app.world().non_send::<InteractiveRuntime>();
        assert!(runtime.phase == Phase::Camera);
        assert_eq!(runtime.episode_id, 2);
        assert!(!runtime.worker.is_busy());
        let mut cursor = app
            .world()
            .resource::<Messages<TaskRuntimeIntent>>()
            .get_cursor();
        assert!(
            cursor
                .read(app.world().resource::<Messages<TaskRuntimeIntent>>())
                .any(|intent| matches!(intent, TaskRuntimeIntent::Capture { episode_id: 2 }))
        );
    }
}
