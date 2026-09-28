//! Original station labels, rendered into transparent glyph textures and placed in 3D.
use crate::StationScene;
use bevy::{
    camera::{RenderTarget, visibility::RenderLayers},
    prelude::*,
    render::render_resource::TextureFormat,
    text::{FontSize, LineBreak},
};
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
