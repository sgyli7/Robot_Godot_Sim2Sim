//! Small in-scene task controls. The integrator owns execution and all status;
//! this module only edits text, emits user intent and displays reported state.

use bevy::{
    camera::RenderTarget,
    input_focus::{
        InputFocus,
        tab_navigation::{TabIndex, TabNavigationPlugin},
    },
    prelude::*,
    text::{EditableText, TextCursorStyle},
};

#[derive(Message, Debug, Clone, PartialEq, Eq)]
pub enum TaskUiAction {
    Start { instruction: String },
    Stop,
    Reset,
    Overview,
    GraspDetail,
}

/// The application replaces these display strings after actual state changes.
#[derive(Resource, Clone)]
pub struct TaskUiStatus {
    pub task: String,
    pub profile: String,
    pub stage: String,
    pub model_service: String,
    pub last_decision: String,
    pub failure: Option<String>,
}

impl Default for TaskUiStatus {
    fn default() -> Self {
        Self {
            task: "未启动".into(),
            profile: "尚未通过执行资格".into(),
            stage: "等待任务".into(),
            model_service: "未连接".into(),
            last_decision: "无".into(),
            failure: None,
        }
    }
}

/// Disable station orbit hotkeys while the task editor has keyboard focus.
#[derive(Resource, Default)]
pub struct TaskUiFocus(pub bool);

#[derive(Component)]
struct TaskUiRoot;
#[derive(Component)]
struct TaskEditor;
#[derive(Component)]
struct TaskStatusText;
#[derive(Component, Clone, Copy)]
enum TaskButton {
    Start,
    Stop,
    Reset,
    Overview,
    GraspDetail,
}

pub struct TaskUiPlugin;

impl Plugin for TaskUiPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TabNavigationPlugin>() {
            app.add_plugins(TabNavigationPlugin);
        }
        app.init_resource::<TaskUiStatus>()
            .init_resource::<TaskUiFocus>()
            .add_message::<TaskUiAction>()
            .add_systems(Startup, setup_ui)
            .add_systems(Update, (user_actions, refresh_status))
            .add_systems(PostUpdate, attach_to_window_camera);
    }
}

fn setup_ui(mut commands: Commands, assets: Res<AssetServer>) {
    let font: Handle<Font> = assets.load("third_party/fonts/noto_sans_cjk_regular.otf");
    let normal = TextFont {
        font: font.clone().into(),
        font_size: FontSize::Px(18.0),
        ..default()
    };
    let root = commands
        .spawn((
            TaskUiRoot,
            Node {
                position_type: PositionType::Absolute,
                top: px(18),
                right: px(18),
                width: px(430),
                padding: px(16).all(),
                row_gap: px(10),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.035, 0.055, 0.075, 0.94)),
        ))
        .id();
    let title = commands
        .spawn((
            Text::new("G1 · 本地视觉任务"),
            normal.clone(),
            TextColor(Color::WHITE),
        ))
        .id();
    let help = commands
        .spawn((
            Text::new("点击输入框可粘贴中文；Ctrl+Enter 启动，Esc 停止。"),
            TextFont {
                font: font.clone().into(),
                font_size: FontSize::Px(15.0),
                ..default()
            },
            TextColor(Color::srgb(0.72, 0.8, 0.85)),
        ))
        .id();
    let mut editor = EditableText::new("观察当前场景");
    editor.max_characters = Some(1024);
    editor.allow_newlines = true;
    editor.visible_lines = Some(3.0);
    let editor = commands
        .spawn((
            TaskEditor,
            editor,
            TabIndex(0),
            normal.clone(),
            TextColor(Color::WHITE),
            TextCursorStyle::default(),
            Node {
                width: percent(100),
                min_height: px(90),
                padding: px(8).all(),
                border: px(1).all(),
                ..default()
            },
            BackgroundColor(Color::srgb(0.08, 0.11, 0.14)),
            BorderColor::from(Color::srgb(0.3, 0.5, 0.58)),
        ))
        .id();
    let buttons = commands
        .spawn(Node {
            column_gap: px(8),
            flex_wrap: FlexWrap::Wrap,
            row_gap: px(8),
            ..default()
        })
        .id();
    for (button, label) in [
        (TaskButton::Start, "启动"),
        (TaskButton::Stop, "停止"),
        (TaskButton::Reset, "重置"),
        (TaskButton::Overview, "全景"),
        (TaskButton::GraspDetail, "抓取细节"),
    ] {
        let entity = commands
            .spawn((
                Button,
                button,
                Node {
                    padding: UiRect::axes(px(12), px(8)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.15, 0.25, 0.3)),
            ))
            .id();
        let label = commands
            .spawn((Text::new(label), normal.clone(), TextColor(Color::WHITE)))
            .id();
        commands.entity(entity).add_child(label);
        commands.entity(buttons).add_child(entity);
    }
    let status = commands
        .spawn((
            TaskStatusText,
            Text::new(""),
            normal,
            TextColor(Color::srgb(0.85, 0.9, 0.92)),
            TextLayout {
                linebreak: LineBreak::WordOrCharacter,
                ..default()
            },
        ))
        .id();
    commands
        .entity(root)
        .add_children(&[title, help, editor, buttons, status]);
}

fn user_actions(
    buttons: Query<(&Interaction, &TaskButton), Changed<Interaction>>,
    editor: Query<(Entity, &EditableText), With<TaskEditor>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    focus: Res<InputFocus>,
    mut ui_focus: ResMut<TaskUiFocus>,
    mut actions: MessageWriter<TaskUiAction>,
) {
    let Ok((entity, text)) = editor.single() else {
        return;
    };
    ui_focus.0 = focus.get() == Some(entity);
    let mut start = ui_focus.0
        && keyboard.just_pressed(KeyCode::Enter)
        && (keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight));
    let mut stop = keyboard.just_pressed(KeyCode::Escape);
    let mut reset = false;
    for (interaction, button) in &buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            TaskButton::Start => start = true,
            TaskButton::Stop => stop = true,
            TaskButton::Reset => reset = true,
            TaskButton::Overview => {
                actions.write(TaskUiAction::Overview);
            }
            TaskButton::GraspDetail => {
                actions.write(TaskUiAction::GraspDetail);
            }
        }
    }
    // A same-frame reset/stop wins over start; the UI never reports execution
    // merely because it emitted one of these messages.
    if reset {
        actions.write(TaskUiAction::Reset);
    } else if stop {
        actions.write(TaskUiAction::Stop);
    } else if start {
        let instruction = text.value().to_string();
        if let Ok(instruction) = validate_task_text(&instruction) {
            actions.write(TaskUiAction::Start { instruction });
        }
    }
}

fn refresh_status(status: Res<TaskUiStatus>, mut labels: Query<&mut Text, With<TaskStatusText>>) {
    if !status.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = format!(
            "任务：{}\n能力：{}\n阶段：{}\n模型：{}\n最近决策：{}\n失败：{}",
            shortened(&status.task, 200),
            shortened(&status.profile, 120),
            shortened(&status.stage, 120),
            shortened(&status.model_service, 120),
            shortened(&status.last_decision, 400),
            shortened(status.failure.as_deref().unwrap_or("无"), 400)
        );
    }
}

fn attach_to_window_camera(
    mut commands: Commands,
    roots: Query<Entity, (With<TaskUiRoot>, Without<UiTargetCamera>)>,
    cameras: Query<(Entity, &RenderTarget, &Camera)>,
) {
    let main = cameras
        .iter()
        .filter(|(_, target, camera)| camera.is_active && matches!(target, RenderTarget::Window(_)))
        .max_by_key(|(_, _, camera)| camera.order)
        .map(|(entity, _, _)| entity);
    if let Some(camera) = main {
        for root in &roots {
            commands.entity(root).insert(UiTargetCamera(camera));
        }
    }
}

fn validate_task_text(value: &str) -> Result<String, &'static str> {
    let value = value.trim();
    if value.is_empty() || value.len() > 4096 || value.chars().count() > 1024 {
        Err("task text must contain 1..=1024 characters and at most 4096 UTF-8 bytes")
    } else {
        Ok(value.into())
    }
}

fn shortened(value: &str, max_characters: usize) -> String {
    value.chars().take(max_characters).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chinese_task_paste_preserves_unicode_and_rejects_empty_input() {
        assert_eq!(
            validate_task_text("  把 A 区的箱子搬到 B 区\n").unwrap(),
            "把 A 区的箱子搬到 B 区"
        );
        assert!(validate_task_text(" \n ").is_err());
        assert!(validate_task_text(&"苹".repeat(1025)).is_err());
        assert_eq!(shortened("观察苹果", 2), "观察");
    }
}
