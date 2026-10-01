//! Scientific station rendering with source-derived shared geometry.
//!
//! Runtime visual plugins consume immutable scene and robot display inputs.
//! All physical body transforms come from the unique simulation backend.
//! Development window orchestration and capture live in dev_tools.
pub mod geometry;
pub mod g1_camera;
pub mod g1_visual;
mod material;
mod render_health;
pub use render_health::{
    StationRenderHealth, StationRenderSnapshot, install_station_render_health,
    validate_render_asset_root,
};
pub mod robot_mesh;
pub mod robot_visual;
mod signage;
pub use signage::StationLabelBakeStatus;
mod site_graphics;

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
};
use geometry::{StationGeometry, load_station_geometry};
use material::{InkMaterial, StationEnamel, StationMaterial};
pub use robot_visual::{
    RobotRenderStyle, RobotVisualInput, RobotVisualModel, RobotVisualPhase, RobotVisualPlugin,
    RobotVisualStatus, RobotVisualSystems,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Fixed station cameras retained from the source scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StationView {
    #[default]
    Arrival,
    Overview,
    Towers,
    Samples,
    Berth,
    Hills,
    Follow,
}
impl StationView {
    /// Parse the public view name without silently substituting another camera.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "arrival" => Ok(Self::Arrival),
            "overview" => Ok(Self::Overview),
            "towers" => Ok(Self::Towers),
            "samples" => Ok(Self::Samples),
            "berth" => Ok(Self::Berth),
            "hills" => Ok(Self::Hills),
            "follow" => Ok(Self::Follow),
            _ => Err(format!("Unknown station view: {name}")),
        }
    }
    fn shot(self, layout: &geometry::StationLayout) -> ([f32; 3], [f32; 3], f32) {
        let name = match self {
            Self::Arrival => "arrival",
            Self::Overview => "overview",
            Self::Towers => "towers",
            Self::Samples => "samples",
            Self::Berth => "berth",
            Self::Hills => "hills",
            Self::Follow => return ([0.65, 0.45, 7.55], [0., 0.15, 6.8], 48.),
        };
        let c = layout
            .cameras
            .iter()
            .find(|c| c.name == name)
            .expect("Station camera checked by loader");
        (c.eye, c.target, c.vertical_fov)
    }
}
/// Resolve deployed assets, with the local source tree as a development fallback.
pub fn default_asset_root() -> PathBuf {
    if let Some(path) = std::env::var_os("BEVY_SIM2SIM_ASSETS") {
        return PathBuf::from(path);
    }
    let next_to_binary = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join("assets")));
    if let Some(path) = next_to_binary.filter(|p| p.is_dir()) {
        path
    } else if PathBuf::from("assets").is_dir() {
        PathBuf::from("assets")
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../assets")
    }
}

/// Shared source scene inserted by the application before adding the visual plugin.
#[derive(Resource, Clone)]
pub struct StationScene(pub Arc<StationGeometry>);
impl StationScene {
    /// Load the immutable scene once for both display and simulation.
    pub fn load(asset_root: &Path) -> Result<Self, String> {
        Ok(Self(Arc::new(load_station_geometry(asset_root)?)))
    }
}
/// Movable visual root. The simulation writes this root's pose by exact prop name.
#[derive(Component, Debug)]
pub struct StationPropVisual {
    pub name: String,
}
/// Camera target written from the simulation's robot base after state synchronization.
#[derive(Resource)]
pub struct StationCameraControl {
    pub view: StationView,
    pub target: Vec3,
    /// Freeze user orbit/shot input while a diagnostic screenshot is pending.
    pub input_enabled: bool,
    /// Optional distance supplied by the simulation's Rapier scene query.
    pub unobstructed_distance: Option<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}
impl Default for StationCameraControl {
    fn default() -> Self {
        Self {
            view: StationView::Arrival,
            target: Vec3::new(0., 0.15, 6.8),
            input_enabled: true,
            unobstructed_distance: None,
            yaw: 0.7,
            pitch: 0.35,
            distance: 1.05,
        }
    }
}
#[derive(Component)]
struct StationCamera;
/// Station materials, meshes and camera, independent of the physics backend.
pub struct StationVisualPlugin;
impl Plugin for StationVisualPlugin {
    fn build(&self, app: &mut App) {
        signage::install_bake_lifecycle(app);
        app.init_resource::<StationCameraControl>()
            .add_plugins((
                MaterialPlugin::<StationMaterial>::default(),
                MaterialPlugin::<InkMaterial>::default(),
            ))
            .add_systems(
                Startup,
                (
                    setup_station,
                    signage::setup_signage,
                    site_graphics::setup_surface_graphics,
                )
                    .chain(),
            )
            .add_systems(Update, update_camera);
    }
}
fn mesh_from_surface(s: &geometry::StationSurface) -> Mesh {
    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, s.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, s.normals.clone());
    mesh.insert_indices(bevy::mesh::Indices::U32(s.indices.clone()));
    mesh
}
fn setup_station(
    mut commands: Commands,
    scene: Res<StationScene>,
    control: Res<StationCameraControl>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StationMaterial>>,
    mut inks: ResMut<Assets<InkMaterial>>,
) {
    let ink = inks.add(InkMaterial::default());
    let mut material_handles = std::collections::BTreeMap::new();
    for (role, color) in &scene.0.palette {
        let kind = match role.as_str() {
            "sand" => 1,
            "glass" | "screen" => 2,
            "distant" | "sandstone" | "sandstone_silt" | "sandstone_chalk" | "rose_stone" => 4,
            _ => 0,
        };
        let c = Color::srgba(color[0], color[1], color[2], color[3]);
        let handle = materials.add(StationMaterial {
            base: StandardMaterial {
                base_color: c,
                perceptual_roughness: if kind == 2 { 0.19 } else { 0.95 },
                metallic: if kind == 2 { 0.35 } else { 0. },
                reflectance: if kind == 2 { 0.75 } else { 0. },
                cull_mode: None,
                ..default()
            },
            extension: StationEnamel::new(kind, &scene.0.layout),
        });
        material_handles.insert(role.clone(), handle);
    }
    let mut prop_roots = std::collections::BTreeMap::new();
    for prop in &scene.0.props {
        let entity = commands
            .spawn((
                Name::new(prop.name.clone()),
                StationPropVisual {
                    name: prop.name.clone(),
                },
                Transform {
                    translation: Vec3::from_array(prop.position),
                    rotation: Quat::from_array(prop.rotation),
                    ..default()
                },
                Visibility::default(),
            ))
            .id();
        prop_roots.insert(prop.name.clone(), entity);
    }
    for s in &scene.0.surfaces {
        let handle = meshes.add(mesh_from_surface(s));
        let material = material_handles
            .get(&s.role)
            .expect("Station palette checked by loader")
            .clone();
        let main = commands
            .spawn((
                Name::new(format!("{}/{}", s.owner, s.role)),
                Mesh3d(handle.clone()),
                MeshMaterial3d(material),
                Transform::default(),
            ))
            .id();
        if let Some(parent) = prop_roots.get(&s.owner) {
            commands.entity(*parent).add_child(main);
        }
        if !matches!(
            s.role.as_str(),
            "sand" | "distant" | "glass" | "screen" | "strata" | "ink"
        ) {
            let outline = commands
                .spawn((
                    Mesh3d(handle),
                    MeshMaterial3d(ink.clone()),
                    Transform::default(),
                    bevy::light::NotShadowCaster,
                    bevy::light::NotShadowReceiver,
                ))
                .id();
            if let Some(parent) = prop_roots.get(&s.owner) {
                commands.entity(*parent).add_child(outline);
            }
        }
    }
    // The far sky is original shader color, never a third-party background image.
    let sky_material = materials.add(StationMaterial {
        base: StandardMaterial {
            unlit: true,
            cull_mode: Some(bevy::render::render_resource::Face::Front),
            ..default()
        },
        extension: StationEnamel::new(3, &scene.0.layout),
    });
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(440.).mesh().ico(5).unwrap())),
        MeshMaterial3d(sky_material),
        Transform::default(),
        bevy::light::NotShadowCaster,
        bevy::light::NotShadowReceiver,
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.78, 0.76, 0.85),
        brightness: 450.,
        affects_lightmapped_meshes: true,
    });
    commands.insert_resource(bevy::light::DirectionalLightShadowMap { size: 4096 });
    commands.spawn((
        bevy::light::CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 0.015,
            maximum_distance: 65.,
            first_cascade_far_bound: 5.,
            overlap_proportion: 0.2,
        }
        .build(),
        DirectionalLight {
            illuminance: 10000.,
            shadow_maps_enabled: true,
            shadow_depth_bias: 0.02,
            shadow_normal_bias: 1.,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::YXZ,
            -36_f32.to_radians(),
            -48_f32.to_radians(),
            0.,
        )),
    ));
    let (eye, target, fov) = control.view.shot(&scene.0.layout);
    commands.spawn((
        Camera3d::default(),
        StationCamera,
        Msaa::Sample8,
        Tonemapping::None,
        Projection::Perspective(PerspectiveProjection {
            fov: fov.to_radians(),
            near: 0.015,
            far: 500.,
            ..default()
        }),
        Transform::from_translation(Vec3::from_array(eye))
            .looking_at(Vec3::from_array(target), Vec3::Y),
    ));
    info!(surfaces=scene.0.surfaces.len(),colliders=scene.0.colliders.len(),props=scene.0.props.len(),labels=scene.0.labels.len(),model_sha256=%scene.0.model_sha256,manifest_sha256=%scene.0.manifest_sha256,layout_sha256=%scene.0.layout_sha256,layout=%scene.0.layout.identity,"STATION_GEOMETRY_READY");
}
fn update_camera(
    scene: Res<StationScene>,
    mut camera: Query<(&mut Transform, &mut Projection), With<StationCamera>>,
    mut control: ResMut<StationCameraControl>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
) {
    if control.input_enabled && keys.just_pressed(KeyCode::Tab) {
        control.view = match control.view {
            StationView::Arrival => StationView::Overview,
            StationView::Overview => StationView::Towers,
            StationView::Towers => StationView::Samples,
            StationView::Samples => StationView::Berth,
            StationView::Berth => StationView::Hills,
            StationView::Hills => StationView::Follow,
            StationView::Follow => StationView::Arrival,
        };
    }
    let (eye, target, fov) = if control.view == StationView::Follow {
        if control.input_enabled && buttons.pressed(MouseButton::Right) {
            control.yaw -= motion.delta.x * 0.004;
            control.pitch = (control.pitch + motion.delta.y * 0.004).clamp(0.08, 1.25);
        }
        if control.input_enabled {
            control.distance = (control.distance - scroll.delta.y * 0.08).clamp(0.25, 4.);
        }
        let d = control
            .unobstructed_distance
            .unwrap_or(control.distance)
            .min(control.distance)
            .max(0.08);
        let offset = Vec3::new(
            control.yaw.sin() * control.pitch.cos(),
            control.pitch.sin(),
            control.yaw.cos() * control.pitch.cos(),
        ) * d;
        (
            (control.target + offset).to_array(),
            control.target.to_array(),
            48.,
        )
    } else {
        control.view.shot(&scene.0.layout)
    };
    for (mut transform, mut projection) in &mut camera {
        // Reapplying a fixed shot must not invalidate transform and projection
        // consumers when the actual camera state is unchanged.
        transform.set_if_neq(
            Transform::from_translation(Vec3::from_array(eye))
                .looking_at(Vec3::from_array(target), Vec3::Y),
        );
        let next_fov = fov.to_radians();
        if matches!(&*projection, Projection::Perspective(p) if p.fov != next_fov)
            && let Projection::Perspective(p) = &mut *projection
        {
            p.fov = next_fov;
        }
    }
}

#[cfg(test)]
mod camera_tests {
    use super::*;
    use bevy::ecs::system::SystemId;
    use geometry::{StationLayout, StationReviewCamera};

    fn camera_world() -> (World, Entity, SystemId) {
        let layout = StationLayout {
            identity: "camera_test".into(),
            revision: 1,
            plaza: [0.; 4],
            loop_center_radii: [0.; 4],
            loop_width: 0.,
            berth: [0.; 4],
            skills: [0.; 4],
            safe_points: Vec::new(),
            paths: Vec::new(),
            cameras: vec![
                StationReviewCamera {
                    name: "overview".into(),
                    eye: [3., 2., 4.],
                    target: [0., 0., 0.],
                    vertical_fov: 44.,
                },
                StationReviewCamera {
                    name: "towers".into(),
                    eye: [6., 5., 7.],
                    target: [1., 0., 0.],
                    vertical_fov: 52.,
                },
            ],
            zones: Vec::new(),
            ridge_waypoints: Vec::new(),
            ridge_width: 0.,
            ridge_grade_plane: [0.; 3],
            fixtures: Vec::new(),
            graphics: Vec::new(),
        };
        let scene = StationScene(Arc::new(StationGeometry {
            model_sha256: String::new(),
            manifest_sha256: String::new(),
            source: "synthetic camera change detection".into(),
            surfaces: Vec::new(),
            colliders: Vec::new(),
            props: Vec::new(),
            labels: Vec::new(),
            palette: Default::default(),
            safe_points: Vec::new(),
            layout,
            layout_sha256: String::new(),
        }));
        let mut world = World::new();
        world.insert_resource(scene);
        world.insert_resource(StationCameraControl {
            view: StationView::Overview,
            ..default()
        });
        world.insert_resource(ButtonInput::<KeyCode>::default());
        world.insert_resource(ButtonInput::<MouseButton>::default());
        world.insert_resource(AccumulatedMouseMotion::default());
        world.insert_resource(AccumulatedMouseScroll::default());
        let camera = world
            .spawn((StationCamera, Transform::default(), Projection::default()))
            .id();
        let system = world.register_system(update_camera);
        (world, camera, system)
    }

    #[test]
    fn repeated_station_shot_does_not_dirty_camera_components() {
        let (mut world, camera, system) = camera_world();
        world.run_system(system).unwrap();
        let original = *world.get::<Transform>(camera).unwrap();
        world.clear_trackers();
        world.run_system(system).unwrap();
        let entity = world.entity(camera);
        assert_eq!(*entity.get::<Transform>().unwrap(), original);
        assert!(!entity.get_ref::<Transform>().unwrap().is_changed());
        assert!(!entity.get_ref::<Projection>().unwrap().is_changed());
    }

    #[test]
    fn switching_and_following_still_update_actual_camera_state() {
        let (mut world, camera, system) = camera_world();
        world.run_system(system).unwrap();
        world.clear_trackers();
        world
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Tab);
        world.run_system(system).unwrap();
        assert_eq!(
            world.resource::<StationCameraControl>().view,
            StationView::Towers
        );
        assert_eq!(
            world.get::<Transform>(camera).unwrap().translation,
            Vec3::new(6., 5., 7.)
        );
        assert!(
            world
                .entity(camera)
                .get_ref::<Projection>()
                .unwrap()
                .is_changed()
        );
        world.resource_mut::<ButtonInput<KeyCode>>().clear();
        world.resource_mut::<StationCameraControl>().view = StationView::Follow;
        world.run_system(system).unwrap();
        let previous = *world.get::<Transform>(camera).unwrap();
        world.clear_trackers();
        world.resource_mut::<StationCameraControl>().target += Vec3::X;
        world.run_system(system).unwrap();
        assert_eq!(
            world.get::<Transform>(camera).unwrap().translation,
            previous.translation + Vec3::X
        );
        assert!(
            world
                .entity(camera)
                .get_ref::<Transform>()
                .unwrap()
                .is_changed()
        );
        assert!(
            !world
                .entity(camera)
                .get_ref::<Projection>()
                .unwrap()
                .is_changed()
        );
    }
}
