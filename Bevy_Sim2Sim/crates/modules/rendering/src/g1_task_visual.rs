//! Original task meshes/textures following immutable owner-world poses.
//!
//! Visual input is for rendering only. It is not a model observation, actuator
//! command or set-pose API. The caller binds the physics definition at startup.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

use bevy::{
    asset::RenderAssetUsages,
    image::{
        CompressedImageFormats, ImageAddressMode, ImageSampler, ImageSamplerDescriptor, ImageType,
    },
    math::Affine2,
    prelude::*,
    transform::TransformSystems,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    units: String,
    source_query_sha256: String,
    exporter_sha256: String,
    source_renderer_parity_proven: bool,
    objects: Vec<Object>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Object {
    kind: String,
    usd_sha256: String,
    source_scale_override: [f64; 3],
    meshes: Vec<SourceMesh>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMesh {
    path: String,
    points: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
    uvs: Vec<[f64; 2]>,
    triangles: Vec<[u32; 3]>,
    source_normal_interpolation: String,
    source_uv_interpolation: String,
    source_normal_fallback_vertices: usize,
    material: Material,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Texture {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Material {
    source_material: String,
    base_color: [f32; 3],
    roughness: f32,
    metallic: f32,
    albedo: Texture,
    normal: Option<Texture>,
    orm: Option<Texture>,
    uv_scale: [f32; 2],
    normal_flip_tangent_v: bool,
    analytic_cube_uv: bool,
}

const ASSETS: [(&str, &str, [f64; 3]); 4] = [
    (
        "t1_apple",
        "2e0e0462c4345b340e1c6040c11abe1818437ced1db228a53ec0944001bb46d8",
        [0.009; 3],
    ),
    (
        "t1_plate",
        "286238c8f957e3267fa21a0003b320868130f903a4b1d991a4ffeb49faaea5a8",
        [0.5; 3],
    ),
    (
        "t2_box",
        "50dc139612086b9483770a1abc17dc600445aa4f85323d74fa97069f7c2eb4ed",
        [1.; 3],
    ),
    (
        "t2_bin",
        "b9ffec2e70fd009863a3fa8bd699aca808403522eafb259d5638135e63506999",
        [4., 2., 1.],
    ),
];

#[derive(Resource)]
pub struct G1TaskVisualModel {
    document: Arc<Document>,
    textures: HashMap<(String, bool), Vec<u8>>,
    physics_sha256: String,
    pub file_sha256: String,
}

impl G1TaskVisualModel {
    pub fn load(path: &Path, visual_sha256: &str, physics_sha256: &str) -> Result<Self, String> {
        let bytes = bounded_bytes(path, visual_sha256, 128 * 1024 * 1024)?;
        let document: Document = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if document.schema != "native_g1_task_visual_v2"
            || document.units != "metres_z_up"
            || !sha_text(&document.source_query_sha256)
            || !sha_text(&document.exporter_sha256)
            || document.source_renderer_parity_proven
            || !sha_text(physics_sha256)
            || document.objects.len() != 4
        {
            return Err("task visual schema/source identity mismatch".into());
        }
        let mut kinds = HashSet::new();
        let mut textures = HashMap::new();
        for object in &document.objects {
            let expected = ASSETS
                .iter()
                .find(|a| a.0 == object.kind)
                .ok_or("unknown task visual kind")?;
            if !kinds.insert(object.kind.clone())
                || object.usd_sha256 != expected.1
                || object.source_scale_override != expected.2
                || object.meshes.is_empty()
                || object.meshes.len() > 16
            {
                return Err("task visual source asset coverage mismatch".into());
            }
            for mesh in &object.meshes {
                if mesh.path.is_empty()
                    || mesh.points.len() < 3
                    || mesh.points.len() > 1_000_000
                    || mesh.normals.len() != mesh.points.len()
                    || mesh.uvs.len() != mesh.points.len()
                    || mesh.triangles.is_empty()
                    || mesh.triangles.len() > 1_000_000
                    || mesh
                        .points
                        .iter()
                        .flatten()
                        .chain(mesh.normals.iter().flatten())
                        .chain(mesh.uvs.iter().flatten())
                        .any(|v| !v.is_finite() || !(*v as f32).is_finite())
                    || mesh
                        .normals
                        .iter()
                        .any(|n| (n.iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 1e-4)
                    || mesh
                        .triangles
                        .iter()
                        .flatten()
                        .any(|i| *i as usize >= mesh.points.len())
                    || !matches!(
                        mesh.source_normal_interpolation.as_str(),
                        "uniform" | "vertex" | "faceVarying"
                    )
                    || !matches!(
                        mesh.source_uv_interpolation.as_str(),
                        "vertex" | "faceVarying"
                    )
                    || mesh.source_normal_fallback_vertices
                        != if object.kind == "t1_plate" { 36 } else { 0 }
                {
                    return Err("invalid authored task visual geometry/attributes".into());
                }
                let material = &mesh.material;
                if material.source_material.is_empty()
                    || material
                        .base_color
                        .iter()
                        .chain([&material.roughness, &material.metallic])
                        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                    || material.uv_scale.iter().any(|v| !v.is_finite() || *v <= 0.)
                    || material.analytic_cube_uv != (object.kind == "t2_box")
                {
                    return Err("invalid original task surface inputs".into());
                }
                for (texture, srgb) in [
                    Some((&material.albedo, true)),
                    material.normal.as_ref().map(|t| (t, false)),
                    material.orm.as_ref().map(|t| (t, false)),
                ]
                .into_iter()
                .flatten()
                {
                    let key = (texture.path.clone(), srgb);
                    if let Some(previous) = textures.get(&key) {
                        if format!("{:x}", Sha256::digest(previous)) != texture.sha256 {
                            return Err("conflicting task texture identity".into());
                        }
                    } else {
                        textures.insert(
                            key,
                            bounded_bytes(
                                Path::new(&texture.path),
                                &texture.sha256,
                                32 * 1024 * 1024,
                            )?,
                        );
                    }
                }
            }
        }
        Ok(Self {
            document: Arc::new(document),
            textures,
            physics_sha256: physics_sha256.into(),
            file_sha256: visual_sha256.into(),
        })
    }

    pub fn object_index(&self, kind: &str) -> Result<usize, String> {
        self.document
            .objects
            .iter()
            .position(|o| o.kind == kind)
            .ok_or_else(|| "task pose has no original visual object".into())
    }
}

#[derive(Clone, Debug)]
pub struct G1TaskVisualPose {
    pub object_index: usize,
    pub translation_engine: [f32; 3],
    pub rotation_engine_xyzw: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct G1TaskVisualFrame {
    pub physics_definition_sha256: String,
    pub episode_id: u64,
    pub source_tick: u64,
    pub sim_time: f64,
    pub poses: Vec<G1TaskVisualPose>,
}

#[derive(Resource, Default)]
pub struct G1TaskVisualInput(pub Option<G1TaskVisualFrame>);

#[derive(Resource, Default, Debug, Clone, Serialize)]
pub struct G1TaskVisualStatus {
    pub mesh_count: usize,
    pub texture_count: usize,
    pub episode_id: Option<u64>,
    pub source_tick: Option<u64>,
    pub visible_objects: usize,
    pub error: Option<String>,
}

#[derive(Component)]
pub(crate) struct ObjectRoot(pub(crate) usize);

pub struct G1TaskVisualPlugin;
impl Plugin for G1TaskVisualPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<G1TaskVisualInput>()
            .init_resource::<G1TaskVisualStatus>()
            .add_systems(Startup, spawn)
            .add_systems(PostUpdate, sync.before(TransformSystems::Propagate));
    }
}

fn spawn(
    mut commands: Commands,
    model: Res<G1TaskVisualModel>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut status: ResMut<G1TaskVisualStatus>,
) {
    let result = (|| -> Result<(), String> {
        let mut textures = HashMap::new();
        for ((path, srgb), bytes) in &model.textures {
            let extension = Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .ok_or("texture lacks format")?;
            let image = Image::from_buffer(
                bytes,
                ImageType::Extension(extension),
                CompressedImageFormats::NONE,
                *srgb,
                ImageSampler::Descriptor(ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    ..default()
                }),
                RenderAssetUsages::default(),
            )
            .map_err(|e| e.to_string())?;
            if u64::from(image.width()) * u64::from(image.height()) > 128 * 1024 * 1024 {
                return Err("decoded task texture exceeds pixel budget".into());
            }
            textures.insert((path.clone(), *srgb), images.add(image));
        }
        for (index, object) in model.document.objects.iter().enumerate() {
            let root = commands
                .spawn((
                    Name::new(format!("g1_task_{}", object.kind)),
                    ObjectRoot(index),
                    Transform::IDENTITY,
                    Visibility::Hidden,
                ))
                .id();
            for source in &object.meshes {
                let mut mesh = Mesh::new(
                    bevy::mesh::PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                );
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    source
                        .points
                        .iter()
                        .map(|p| [p[0] as f32, p[2] as f32, -p[1] as f32])
                        .collect::<Vec<_>>(),
                );
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_NORMAL,
                    source
                        .normals
                        .iter()
                        .map(|p| [p[0] as f32, p[2] as f32, -p[1] as f32])
                        .collect::<Vec<_>>(),
                );
                // USD st has bottom-left image origin. Bevy image buffers use top-left.
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_UV_0,
                    source
                        .uvs
                        .iter()
                        .map(|uv| [uv[0] as f32, 1. - uv[1] as f32])
                        .collect::<Vec<_>>(),
                );
                mesh.insert_indices(bevy::mesh::Indices::U32(
                    source.triangles.iter().flatten().copied().collect(),
                ));
                if source.material.normal.is_some() {
                    mesh.generate_tangents().map_err(|e| e.to_string())?;
                }
                let source_material = &source.material;
                let texture = |t: &Texture, srgb| {
                    textures
                        .get(&(t.path.clone(), srgb))
                        .cloned()
                        .ok_or_else(|| "task surface texture was not decoded".to_string())
                };
                let orm = source_material
                    .orm
                    .as_ref()
                    .map(|t| texture(t, false))
                    .transpose()?;
                let material = materials.add(StandardMaterial {
                    base_color: Color::srgb_from_array(source_material.base_color),
                    base_color_texture: Some(texture(&source_material.albedo, true)?),
                    normal_map_texture: source_material
                        .normal
                        .as_ref()
                        .map(|t| texture(t, false))
                        .transpose()?,
                    flip_normal_map_y: source_material.normal_flip_tangent_v,
                    metallic: if orm.is_some() {
                        1.
                    } else {
                        source_material.metallic
                    },
                    perceptual_roughness: if orm.is_some() {
                        1.
                    } else {
                        source_material.roughness
                    },
                    occlusion_texture: orm.clone(),
                    metallic_roughness_texture: orm,
                    uv_transform: Affine2::from_scale(Vec2::from_array(source_material.uv_scale)),
                    ..default()
                });
                let surface = commands
                    .spawn((
                        Name::new(source.path.clone()),
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material),
                        Transform::IDENTITY,
                    ))
                    .id();
                commands.entity(root).add_child(surface);
                status.mesh_count += 1;
            }
        }
        status.texture_count = textures.len();
        Ok(())
    })();
    if let Err(error) = result {
        status.error = Some(error);
    }
}

fn sync(
    input: Res<G1TaskVisualInput>,
    model: Res<G1TaskVisualModel>,
    mut status: ResMut<G1TaskVisualStatus>,
    mut roots: Query<(&ObjectRoot, &mut Transform, &mut Visibility)>,
) {
    let Some(frame) = &input.0 else {
        for (_, _, mut visibility) in &mut roots {
            *visibility = Visibility::Hidden;
        }
        status.visible_objects = 0;
        return;
    };
    let result = validate_frame(frame, &model, &status);
    if let Err(error) = result {
        status.error = Some(error);
        status.visible_objects = 0;
        for (_, _, mut visibility) in &mut roots {
            *visibility = Visibility::Hidden;
        }
        return;
    }
    if status.error.is_some() {
        return;
    }
    for (root, mut transform, mut visibility) in &mut roots {
        if let Some(pose) = frame.poses.iter().find(|p| p.object_index == root.0) {
            let next = Transform {
                translation: Vec3::from_array(pose.translation_engine),
                rotation: Quat::from_array(pose.rotation_engine_xyzw),
                scale: Vec3::ONE,
            };
            if *transform != next {
                *transform = next;
            }
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
    status.episode_id = Some(frame.episode_id);
    status.source_tick = Some(frame.source_tick);
    status.visible_objects = frame.poses.len();
}

fn validate_frame(
    frame: &G1TaskVisualFrame,
    model: &G1TaskVisualModel,
    status: &G1TaskVisualStatus,
) -> Result<(), String> {
    if frame.physics_definition_sha256 != model.physics_sha256
        || frame.poses.is_empty()
        || frame.poses.len() > 4
        || !frame.sim_time.is_finite()
        || (frame.sim_time - frame.source_tick as f64 * 0.02).abs() > 1e-8
        || status.episode_id.is_some_and(|episode| {
            frame.episode_id < episode
                || (frame.episode_id == episode
                    && status
                        .source_tick
                        .is_some_and(|tick| frame.source_tick < tick))
        })
    {
        return Err("stale or mismatched task visual boundary".into());
    }
    let mut indices = HashSet::new();
    for pose in &frame.poses {
        if pose.object_index >= model.document.objects.len()
            || !indices.insert(pose.object_index)
            || pose
                .translation_engine
                .iter()
                .chain(pose.rotation_engine_xyzw.iter())
                .any(|v| !v.is_finite())
            || (pose.rotation_engine_xyzw.iter().map(|v| v * v).sum::<f32>() - 1.).abs() > 1e-5
        {
            return Err("invalid task object visual pose".into());
        }
    }
    Ok(())
}

fn sha_text(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn bounded_bytes(path: &Path, sha: &str, limit: u64) -> Result<Vec<u8>, String> {
    if !sha_text(sha) || std::fs::metadata(path).map_err(|e| e.to_string())?.len() > limit {
        return Err("invalid task visual identity or byte budget".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit || format!("{:x}", Sha256::digest(&bytes)) != sha {
        return Err("task visual or texture SHA256 mismatch".into());
    }
    Ok(bytes)
}
