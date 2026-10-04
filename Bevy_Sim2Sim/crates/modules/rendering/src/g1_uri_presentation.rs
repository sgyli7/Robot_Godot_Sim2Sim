//! URI spectator rendering. Source geometry, materials and sensor cameras stay untouched.

use std::collections::{HashMap, HashSet};

use bevy::{
    camera::{
        RenderTarget,
        visibility::{RenderLayers, VisibilitySystems},
    },
    core_pipeline::tonemapping::Tonemapping,
    light::{NotShadowCaster, NotShadowReceiver},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
    transform::TransformSystems,
};

use crate::{
    g1_background_visual::BackgroundMaterial,
    material::{InkMaterial, StationMaterial},
};

/// Only the explicitly tagged spectator camera receives the URI layer.
#[derive(Component)]
pub struct G1PresentationCamera;

#[derive(Component)]
struct PresentationCopy;

#[derive(Component)]
struct SourcePipelineWarmup;

#[derive(Component)]
struct SourcePipelineWarmupMesh;

type UriMaterial = ExtendedMaterial<StandardMaterial, UriEnamel>;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct UriEnamel {
    #[uniform(100)]
    parameters: Vec4,
    #[uniform(100)]
    plaza: Vec4,
}

impl MaterialExtension for UriEnamel {
    fn fragment_shader() -> ShaderRef {
        "game/shaders/g1_uri_enamel.wgsl".into()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SourceMaterial {
    Standard(AssetId<StandardMaterial>),
    Station(AssetId<StationMaterial>),
    Background(AssetId<BackgroundMaterial>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SurfaceRole {
    Original,
    Head,
    Silver,
    RobotDark,
    Blue,
    White,
    Orange,
    Dark,
    Glass,
    Ground,
    Sky,
    Table,
}

struct Copies {
    surface: Entity,
    outline: Option<Entity>,
}

#[derive(Resource, Default)]
struct PresentationState {
    copies: HashMap<Entity, Copies>,
    materials: HashMap<(SourceMaterial, SurfaceRole), Handle<UriMaterial>>,
    meshes: HashMap<AssetId<Mesh>, Handle<Mesh>>,
    ink: Handle<InkMaterial>,
}

/// Read-only mesh copies on layer 1, with their own pigments and no shadow interaction.
pub struct G1UriPresentationPlugin;

impl Plugin for G1UriPresentationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<UriMaterial>::default());
        if !app.is_plugin_added::<MaterialPlugin<InkMaterial>>() {
            app.add_plugins(MaterialPlugin::<InkMaterial>::default());
        }
        app.init_resource::<PresentationState>()
            .add_systems(PostStartup, warm_source_pipelines)
            .add_systems(Update, retire_warmup)
            .add_systems(
                PostUpdate,
                sync_copies
                    .after(TransformSystems::Propagate)
                    .after(VisibilitySystems::VisibilityPropagate)
                    .before(VisibilitySystems::CheckVisibility),
            );
    }
}

fn warm_source_pipelines(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<(&Transform, &Projection), With<G1PresentationCamera>>,
    source_materials: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (transform, projection) in &cameras {
        let image = images.add(Image::new_target_texture(
            64,
            64,
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            Some(bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb),
        ));
        commands.spawn((
            Name::new("g1_source_pipeline_warmup"),
            SourcePipelineWarmup,
            Camera3d::default(),
            Camera {
                order: -99,
                ..default()
            },
            RenderTarget::Image(image.into()),
            Msaa::Sample8,
            Tonemapping::None,
            *transform,
            projection.clone(),
            RenderLayers::from_layers(&[0, 2]),
        ));
        // Hidden printed markers would otherwise specialize their source PBR
        // pipeline only during a one-shot sensor capture. Queue every startup
        // standard-material variant on an isolated, unread offscreen layer.
        for (mesh, material) in &source_materials {
            commands.spawn((
                SourcePipelineWarmupMesh,
                mesh.clone(),
                material.clone(),
                Transform {
                    translation: transform.translation + transform.forward().as_vec3(),
                    rotation: transform.rotation
                        * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    ..default()
                },
                Visibility::Visible,
                RenderLayers::layer(2),
                NotShadowCaster,
                NotShadowReceiver,
            ));
        }
    }
}

fn retire_warmup(
    mut commands: Commands,
    health: Option<Res<crate::StationRenderHealth>>,
    meshes: Query<Entity, With<SourcePipelineWarmupMesh>>,
) {
    if health.is_some_and(|health| health.snapshot().ready) {
        for entity in &meshes {
            commands.entity(entity).despawn();
        }
    }
    // Retain the unread source view for the whole run. Source materials that
    // become visible later must keep specializing ahead of one-shot sensors,
    // just as they did when the original window was the source spectator.
}

fn role(name: &str, source: SourceMaterial, base: &StandardMaterial) -> SurfaceRole {
    if name.starts_with("/g1_29dof_with_hand_rev_1_0/") {
        let body = name.rsplit('/').nth(1).unwrap_or_default();
        if body == "head_link" {
            SurfaceRole::Head
        } else if body == "pelvis"
            || [
                "waist_",
                "hip_yaw",
                "ankle_pitch",
                "logo_link",
                "elbow_link",
                "wrist_roll",
                "hand_",
                "ankle_roll",
            ]
            .iter()
            .any(|part| body.contains(part))
        {
            SurfaceRole::RobotDark
        } else {
            SurfaceRole::Silver
        }
    } else if name == "/Lab/Structure/Floor" {
        SurfaceRole::Ground
    } else if name.starts_with("/Lab/TaskAssets/shelf/") {
        SurfaceRole::Table
    } else if name.starts_with("/Lab/BackgroundAssets/rack/")
        || name.starts_with("/Lab/BackgroundAssets/rack_")
    {
        if name.contains("/shelf") {
            SurfaceRole::White
        } else {
            SurfaceRole::Blue
        }
    } else if name == "actual_source_procedural_support_geometry" {
        SurfaceRole::Table
    } else if name == "g1_runner_actual_floor" {
        SurfaceRole::Ground
    } else if matches!(source, SourceMaterial::Station(_)) {
        if name.is_empty() && base.unlit {
            return SurfaceRole::Sky;
        }
        match name.rsplit('/').next().unwrap_or_default() {
            "sand" => SurfaceRole::Ground,
            "blue" | "pale_blue" | "cyan" => SurfaceRole::Blue,
            "paper" | "porcelain" | "cream" | "white" => SurfaceRole::White,
            "ochre" | "orange" | "signal" | "yellow" => SurfaceRole::Orange,
            "ink" | "rubber" => SurfaceRole::Dark,
            "glass" | "screen" => SurfaceRole::Glass,
            _ => SurfaceRole::Original,
        }
    } else {
        // Task objects, fiducials and source background textures retain their colors.
        SurfaceRole::Original
    }
}

fn pigment(base: &StandardMaterial, role: SurfaceRole, plaza: Vec4) -> UriMaterial {
    let mut base = base.clone();
    base.base_color = match role {
        SurfaceRole::Head => Color::srgb_u8(57, 60, 65),
        SurfaceRole::Silver => Color::srgb_u8(176, 181, 186),
        SurfaceRole::RobotDark => Color::srgb_u8(32, 35, 39),
        SurfaceRole::Blue => Color::srgb_u8(34, 143, 208),
        SurfaceRole::White => Color::srgb_u8(243, 242, 234),
        SurfaceRole::Orange => Color::srgb_u8(255, 181, 46),
        SurfaceRole::Dark => Color::srgb_u8(37, 51, 67),
        SurfaceRole::Glass => Color::srgb_u8(25, 55, 81),
        SurfaceRole::Ground => Color::srgb_u8(224, 228, 229),
        SurfaceRole::Sky => Color::srgb_u8(101, 175, 218),
        SurfaceRole::Table => Color::srgb_u8(74, 81, 89),
        SurfaceRole::Original => base.base_color,
    };
    base.unlit = true;
    let kind = match role {
        SurfaceRole::Ground => 1.,
        SurfaceRole::Sky => 2.,
        SurfaceRole::Original if base.base_color_texture.is_some() => 3.,
        SurfaceRole::Glass => 4.,
        SurfaceRole::Head => 5.,
        SurfaceRole::Silver => 6.,
        SurfaceRole::RobotDark => 7.,
        _ => 0.,
    };
    UriMaterial {
        base,
        extension: UriEnamel {
            parameters: Vec4::new(
                kind,
                if role == SurfaceRole::White { 1. } else { 0. },
                0.,
                0.,
            ),
            plaza,
        },
    }
}

struct SourceSnapshot {
    entity: Entity,
    name: String,
    mesh: Handle<Mesh>,
    material: SourceMaterial,
    transform: GlobalTransform,
    inherited: InheritedVisibility,
}

// Repair shading attributes on a private presentation mesh. Positions and
// triangle indices remain identical; source/sensor mesh assets are never edited.
fn repair_normals(mesh: &mut Mesh) {
    mesh.compute_custom_smooth_normals(|indices, positions, normals| {
        let [a, b, c] = indices.map(|i| Vec3::from_array(positions[i]).as_dvec3());
        let face = (b - a).cross(c - a).normalize_or_zero();
        if face == bevy::math::DVec3::ZERO {
            return;
        }
        for (index, (u, v)) in
            indices
                .into_iter()
                .zip([(b - a, c - a), (a - b, c - b), (a - c, b - c)])
        {
            let denominator = (u.length_squared() * v.length_squared()).sqrt();
            if denominator > 0. {
                let angle = (u.dot(v) / denominator).clamp(-1., 1.).acos();
                normals[index] += face.as_vec3() * angle as f32;
            }
        }
    });
}

fn sync_pose(world: &mut World, entity: Entity, source: &SourceSnapshot) {
    if world.get::<GlobalTransform>(entity) != Some(&source.transform) {
        world
            .entity_mut(entity)
            .insert((source.transform.compute_transform(), source.transform));
    }
    let visibility = if source.inherited.get() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    if world.get::<Visibility>(entity) != Some(&visibility) {
        world.entity_mut(entity).insert(visibility);
    }
    if world.get::<InheritedVisibility>(entity) != Some(&source.inherited) {
        world.entity_mut(entity).insert(source.inherited);
    }
    if world.get::<Mesh3d>(entity).map(|m| &m.0) != Some(&source.mesh) {
        world.entity_mut(entity).insert(Mesh3d(source.mesh.clone()));
    }
}

fn sync_copies(world: &mut World) {
    let plaza = world
        .get_resource::<crate::StationScene>()
        .map_or(Vec4::new(0., 0., 20., 20.), |scene| {
            Vec4::from_array(scene.0.layout.plaza)
        });
    let mut cameras = world.query_filtered::<Entity, With<G1PresentationCamera>>();
    let cameras: Vec<_> = cameras.iter(world).collect();
    for camera in cameras {
        if world.get::<RenderLayers>(camera) != Some(&RenderLayers::layer(1)) {
            world
                .entity_mut(camera)
                .insert((RenderLayers::layer(1), Tonemapping::None));
            if let Some(mut view) = world.get_mut::<Camera>(camera) {
                view.clear_color = ClearColorConfig::Custom(Color::srgb_u8(163, 187, 204));
            }
        }
    }
    let mut query = world.query_filtered::<(
        Entity,
        Option<&Name>,
        &Mesh3d,
        &GlobalTransform,
        &InheritedVisibility,
        Option<&MeshMaterial3d<StandardMaterial>>,
        Option<&MeshMaterial3d<StationMaterial>>,
        Option<&MeshMaterial3d<BackgroundMaterial>>,
    ), (Without<PresentationCopy>, Without<SourcePipelineWarmupMesh>)>();
    let sources: Vec<_> = query
        .iter(world)
        .filter_map(
            |(entity, name, mesh, transform, inherited, standard, station, background)| {
                let material = standard
                    .map(|m| SourceMaterial::Standard(m.0.id()))
                    .or_else(|| station.map(|m| SourceMaterial::Station(m.0.id())))
                    .or_else(|| background.map(|m| SourceMaterial::Background(m.0.id())))?;
                Some(SourceSnapshot {
                    entity,
                    name: name.map_or_else(String::new, |n| n.as_str().to_owned()),
                    mesh: mesh.0.clone(),
                    material,
                    transform: *transform,
                    inherited: *inherited,
                })
            },
        )
        .collect();
    let alive: HashSet<_> = sources.iter().map(|s| s.entity).collect();
    world.resource_scope(|world, mut state: Mut<PresentationState>| {
        state.copies.retain(|source, copies| {
            if alive.contains(source) {
                true
            } else {
                world.despawn(copies.surface);
                if let Some(outline) = copies.outline {
                    world.despawn(outline);
                }
                false
            }
        });
        if state.ink == Handle::default() {
            state.ink = world
                .resource_mut::<Assets<InkMaterial>>()
                .add(InkMaterial::default().with_pixels(0.9));
        }
        for mut source in sources {
            let base = match source.material {
                SourceMaterial::Standard(id) => world
                    .resource::<Assets<StandardMaterial>>()
                    .get(id)
                    .cloned(),
                SourceMaterial::Station(id) => world
                    .resource::<Assets<StationMaterial>>()
                    .get(id)
                    .map(|m| m.base.clone()),
                SourceMaterial::Background(id) => world
                    .get_resource::<Assets<BackgroundMaterial>>()
                    .and_then(|materials| materials.get(id))
                    .map(|m| m.base.clone()),
            };
            let Some(base) = base else { continue };
            let role = role(&source.name, source.material, &base);
            if matches!(
                role,
                SurfaceRole::Head | SurfaceRole::Silver | SurfaceRole::RobotDark
            ) {
                if !state.meshes.contains_key(&source.mesh.id()) {
                    let private = world
                        .get_resource::<Assets<Mesh>>()
                        .and_then(|meshes| meshes.get(&source.mesh))
                        .cloned();
                    if let Some(mut private) = private {
                        repair_normals(&mut private);
                        let handle = world.resource_mut::<Assets<Mesh>>().add(private);
                        state.meshes.insert(source.mesh.id(), handle);
                    }
                }
                if let Some(mesh) = state.meshes.get(&source.mesh.id()) {
                    source.mesh = mesh.clone();
                }
            }
            let material = state
                .materials
                .entry((source.material, role))
                .or_insert_with(|| {
                    world
                        .resource_mut::<Assets<UriMaterial>>()
                        .add(pigment(&base, role, plaza))
                })
                .clone();
            let ink = state.ink.clone();
            let copies = state.copies.entry(source.entity).or_insert_with(|| {
                let surface = world
                    .spawn((
                        Name::new(format!("uri/{}", source.name)),
                        PresentationCopy,
                        Mesh3d(source.mesh.clone()),
                        MeshMaterial3d(material.clone()),
                        RenderLayers::layer(1),
                        NotShadowCaster,
                        NotShadowReceiver,
                    ))
                    .id();
                let outline = (!matches!(
                    role,
                    SurfaceRole::Head
                        | SurfaceRole::Silver
                        | SurfaceRole::RobotDark
                        | SurfaceRole::Ground
                        | SurfaceRole::Sky
                        | SurfaceRole::Glass
                        | SurfaceRole::Original
                ))
                .then(|| {
                    world
                        .spawn((
                            PresentationCopy,
                            Mesh3d(source.mesh.clone()),
                            MeshMaterial3d(ink.clone()),
                            RenderLayers::layer(1),
                            NotShadowCaster,
                            NotShadowReceiver,
                        ))
                        .id()
                });
                Copies { surface, outline }
            });
            sync_pose(world, copies.surface, &source);
            if world
                .get::<MeshMaterial3d<UriMaterial>>(copies.surface)
                .map(|m| &m.0)
                != Some(&material)
            {
                world
                    .entity_mut(copies.surface)
                    .insert(MeshMaterial3d(material));
            }
            if let Some(outline) = copies.outline {
                sync_pose(world, outline, &source);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_source_faces_get_finite_presentation_normals_without_geometry_changes() {
        use bevy::asset::RenderAssetUsages;
        use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
        let mut source = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        let positions = vec![[0., 0., 0.], [0.001, 0., 0.], [0., 0.001, 0.]];
        source.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
        source.insert_indices(Indices::U32(vec![0, 1, 2]));
        source.compute_smooth_normals();
        let mut copy = source.clone();
        repair_normals(&mut copy);
        assert_eq!(
            copy.attribute(Mesh::ATTRIBUTE_POSITION),
            source.attribute(Mesh::ATTRIBUTE_POSITION)
        );
        assert_eq!(
            copy.indices().unwrap().iter().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let Some(VertexAttributeValues::Float32x3(normals)) =
            copy.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("missing normals")
        };
        for normal in normals {
            assert!(Vec3::from_array(*normal).abs_diff_eq(Vec3::Z, 1e-6));
        }
        // The unchanged source still reproduces the meter-scale normal degeneracy.
        let Some(VertexAttributeValues::Float32x3(normals)) =
            source.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("missing source normals")
        };
        assert!(normals.iter().all(|normal| *normal == [0.; 3]));
    }

    #[test]
    fn copies_follow_same_frame_visibility_and_removal_without_mutating_sources() {
        let mut world = World::new();
        world.init_resource::<Assets<StandardMaterial>>();
        world.init_resource::<Assets<StationMaterial>>();
        world.init_resource::<Assets<UriMaterial>>();
        world.init_resource::<Assets<InkMaterial>>();
        world.init_resource::<PresentationState>();
        let original = StandardMaterial {
            base_color: Color::srgb(0.4, 0.5, 0.6),
            ..default()
        };
        let material = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(original.clone());
        let source = world
            .spawn((
                Name::new("/g1_29dof_with_hand_rev_1_0/head_link/visuals"),
                Mesh3d::default(),
                MeshMaterial3d(material.clone()),
                GlobalTransform::from_translation(Vec3::new(1., 2., 3.)),
                InheritedVisibility::VISIBLE,
            ))
            .id();
        let torso = world
            .spawn((
                Name::new("/g1_29dof_with_hand_rev_1_0/torso_link/visuals"),
                Mesh3d::default(),
                MeshMaterial3d(material.clone()),
                GlobalTransform::IDENTITY,
                InheritedVisibility::VISIBLE,
            ))
            .id();
        let observer = world
            .spawn((G1PresentationCamera, Tonemapping::AcesFitted))
            .id();
        let sensor = world
            .spawn((Camera3d::default(), Tonemapping::AcesFitted))
            .id();
        sync_copies(&mut world);
        let copy = world.resource::<PresentationState>().copies[&source].surface;
        let torso_copy = world.resource::<PresentationState>().copies[&torso].surface;
        let torso_material = &world
            .get::<MeshMaterial3d<UriMaterial>>(torso_copy)
            .unwrap()
            .0;
        // Official neutral silver shells stay neutral; the shared prefix contains "hand".
        assert_eq!(
            world
                .resource::<Assets<UriMaterial>>()
                .get(torso_material)
                .unwrap()
                .base
                .base_color,
            Color::srgb_u8(176, 181, 186)
        );
        assert_ne!(
            torso_material,
            &world.get::<MeshMaterial3d<UriMaterial>>(copy).unwrap().0
        );
        assert_eq!(
            world.get::<GlobalTransform>(source),
            world.get::<GlobalTransform>(copy)
        );
        assert!(world.get::<NotShadowCaster>(copy).is_some());
        assert!(world.get::<NotShadowReceiver>(copy).is_some());
        let layer = world.get::<RenderLayers>(copy).unwrap();
        assert!(layer.intersects(world.get::<RenderLayers>(observer).unwrap()));
        assert!(!layer.intersects(&RenderLayers::default()));
        assert!(world.get::<RenderLayers>(sensor).is_none());
        assert_eq!(world.get::<Tonemapping>(observer), Some(&Tonemapping::None));
        assert_eq!(
            world.get::<Tonemapping>(sensor),
            Some(&Tonemapping::AcesFitted)
        );
        assert!(world.get::<RenderLayers>(source).is_none());
        assert_eq!(
            world
                .resource::<Assets<StandardMaterial>>()
                .get(&material)
                .unwrap()
                .base_color,
            original.base_color
        );
        world.entity_mut(source).insert((
            GlobalTransform::from_translation(Vec3::new(4., 5., 6.)),
            InheritedVisibility::HIDDEN,
        ));
        sync_copies(&mut world);
        assert_eq!(
            world.get::<GlobalTransform>(source),
            world.get::<GlobalTransform>(copy)
        );
        assert_eq!(world.get::<Visibility>(copy), Some(&Visibility::Hidden));
        world.despawn(source);
        world.despawn(torso);
        sync_copies(&mut world);
        assert!(world.get_entity(copy).is_err());
        assert!(world.resource::<PresentationState>().copies.is_empty());
    }
}
