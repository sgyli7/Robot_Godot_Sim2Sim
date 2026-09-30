//! Original station labels, rendered into transparent glyph textures and placed in 3D.
use crate::StationScene;
use bevy::render::{
    Render, RenderApp, RenderSystems,
    render_resource::{CachedPipelineState, PipelineCache},
};
use bevy::{
    camera::{RenderTarget, visibility::RenderLayers},
    prelude::*,
    render::render_resource::TextureFormat,
    text::{FontSize, LineBreak},
};
use std::sync::{Arc, Mutex};

/// Readiness of the immutable station labels, including their rendered textures.
#[derive(Resource, Clone, Default)]
pub struct StationLabelBakeStatus(Arc<Mutex<LabelBakeState>>);

#[derive(Default)]
struct LabelBakeState {
    pipelines_ready: bool,
    ready_frames: u32,
    baked_camera_count: Option<usize>,
}

impl StationLabelBakeStatus {
    /// The glyph textures have been submitted and their cameras are frozen.
    pub fn is_ready(&self) -> bool {
        self.0.lock().unwrap().baked_camera_count.is_some()
    }

    /// Number of static label cameras retired after baking, if ready.
    pub fn baked_camera_count(&self) -> Option<usize> {
        self.0.lock().unwrap().baked_camera_count
    }
}

#[derive(Component)]
struct StationLabelGlyph;

pub(crate) fn install_bake_lifecycle(app: &mut App) {
    app.init_resource::<StationLabelBakeStatus>()
        .add_systems(Update, bake_static_labels);
    let status = app.world().resource::<StationLabelBakeStatus>().clone();
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(status).add_systems(
            Render,
            observe_label_pipelines.in_set(RenderSystems::Cleanup),
        );
    }
}

fn observe_label_pipelines(cache: Res<PipelineCache>, status: Res<StationLabelBakeStatus>) {
    let mut state = status.0.lock().unwrap();
    if state.baked_camera_count.is_some() {
        return;
    }
    // Cleanup follows render submission. Queue order preserves the baked textures
    // when the following main-world frame deactivates these offscreen cameras.
    state.pipelines_ready = cache.pipelines().next().is_some()
        && cache
            .pipelines()
            .all(|p| matches!(p.state, CachedPipelineState::Ok(_)));
}

fn bake_static_labels(
    status: Res<StationLabelBakeStatus>,
    font: Option<Res<StationLabelFont>>,
    fonts: Res<Assets<Font>>,
    labels: Query<&bevy::text::TextLayoutInfo, With<StationLabelGlyph>>,
    scene: Res<StationScene>,
    mut cameras: Query<&mut Camera, With<StationLabelCamera>>,
) {
    let mut state = status.0.lock().unwrap();
    if state.baked_camera_count.is_some() {
        return;
    }
    let ready = state.pipelines_ready
        && font.as_ref().is_some_and(|f| fonts.contains(f.0.id()))
        && labels.iter().count() == scene.0.labels.len()
        && labels.iter().all(|text| !text.glyphs.is_empty());
    state.ready_frames = if ready { state.ready_frames + 1 } else { 0 };
    if state.ready_frames >= 10 {
        let mut count = 0;
        for mut camera in &mut cameras {
            camera.is_active = false;
            count += 1;
        }
        state.baked_camera_count = Some(count);
        info!(
            count,
            "STATION_LABEL_TEXTURES_READY (static glyph textures retained; camera passes frozen)"
        );
    }
}
#[derive(Component)]
pub(crate) struct StationLabelCamera;
#[derive(Resource)]
pub(crate) struct StationLabelFont(pub Handle<Font>);
pub(crate) fn setup_signage(
    mut commands: Commands,
    scene: Res<StationScene>,
    server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let font: Handle<Font> = server.load("third_party/fonts/noto_sans_cjk_regular.otf");
    commands.insert_resource(StationLabelFont(font.clone()));
    for (i, label) in scene.0.labels.iter().enumerate() {
        let width = label
            .text
            .lines()
            .map(|line| {
                line.chars()
                    .map(|c| if c.is_ascii() { 0.65 } else { 1.05 })
                    .sum::<f32>()
            })
            .fold(1., f32::max);
        let w = (width * 128. + 32.).ceil().clamp(128., 2048.) as u32;
        let h = (label.text.lines().count().max(1) as f32 * 160. + 16.).ceil() as u32;
        let image = images.add(Image::new_target_texture(
            w,
            h,
            TextureFormat::Rgba8Unorm,
            Some(TextureFormat::Rgba8UnormSrgb),
        ));
        let layer = RenderLayers::layer(i + 1);
        let rgba = scene
            .0
            .palette
            .get(&label.role)
            .copied()
            .unwrap_or([0.1, 0.1, 0.1, 1.]);
        commands.spawn((
            Text2d::new(label.text.clone()),
            StationLabelGlyph,
            TextFont {
                font: font.clone().into(),
                font_size: FontSize::Px(128.),
                ..default()
            },
            TextColor(Color::srgba(rgba[0], rgba[1], rgba[2], rgba[3])),
            TextLayout::new(Justify::Center, LineBreak::NoWrap),
            layer.clone(),
        ));
        commands.spawn((
            Camera2d,
            Camera {
                order: -1,
                clear_color: Color::NONE.into(),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            Msaa::Off,
            StationLabelCamera,
            layer,
        ));
        let material = materials.add(StandardMaterial {
            base_color_texture: Some(image),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            depth_bias: 0.02,
            ..default()
        });
        let [rx, ry, rz] = label.rotation.map(f32::to_radians);
        commands.spawn((
            Name::new(format!("Station label: {}", label.text)),
            Mesh3d(meshes.add(Rectangle::new(
                w as f32 * label.em / 128.,
                h as f32 * label.em / 128.,
            ))),
            MeshMaterial3d(material),
            Transform {
                translation: Vec3::from_array(label.position),
                rotation: Quat::from_euler(EulerRot::YXZ, ry, rx, rz),
                ..default()
            },
            bevy::light::NotShadowCaster,
            bevy::light::NotShadowReceiver,
        ));
    }
    info!(
        labels = scene.0.labels.len(),
        "STATION_LABELS_CREATED (original text and source font)"
    );
}
