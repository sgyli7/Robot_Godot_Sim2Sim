//! Bounded original task background visuals, separate from physics ownership.
//! Authored basic PBR inputs are mapped; the receipt discloses omitted surfaces
//! and the absence of MDL/default/lighting parity. This is a render diagnostic.

use std::{collections::HashMap, path::Path};

use bevy::{
    asset::RenderAssetUsages,
    image::{
        CompressedImageFormats, ImageAddressMode, ImageSampler, ImageSamplerDescriptor, ImageType,
    },
    math::Affine2,
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SOURCE_USD_SHA: &str = "7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    #[serde(default)]
    source_t2_background_sha256: Option<String>,
    units: String,
    source_usd_sha256: String,
    exporter_sha256: String,
    source_arena_commit: String,
    source_deactivated_prims: Vec<String>,
    source_background_translation: [f64; 3],
    native_environment_translation: [f64; 3],
    selection_region_min: [f64; 3],
    selection_region_max: [f64; 3],
    mesh_selection: String,
    material_scope: String,
    source_renderer_parity_proven: bool,
    physics_integrations: u64,
    background_physics_registered: bool,
    skipped_outside_region: Vec<String>,
    omitted_unmapped_geometry: Vec<serde_json::Value>,
    meshes: Vec<SourceMesh>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMesh {
    #[serde(default)]
    owner_path: Option<String>,
    path: String,
    material: Material,
    source_face_subset: Option<Vec<u32>>,
    points: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
    uvs: Vec<[f64; 2]>,
    triangles: Vec<[u32; 3]>,
    source_normal_interpolation: String,
    source_uv_interpolation: String,
    source_normal_fallback_vertices: usize,
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
    base_color_space: String,
    #[serde(default)]
    albedo_add: f32,
    roughness: f32,
    metallic: f32,
    albedo: Option<Texture>,
    normal: Option<Texture>,
    orm: Option<Texture>,
    uv_scale: [f32; 2],
    normal_flip_tangent_v: bool,
    authored_shader_inputs: HashMap<String, String>,
    #[serde(default)]
    warehouse_mdl: Option<WarehouseMdl>,
}

/// Only the inspected source graph's color and green-channel roughness map.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WarehouseMdl {
    source_mdl_sha256: String,
    albedo_desaturation: f32,
    roughness_min_max: [f32; 2],
}

type BackgroundMaterial = ExtendedMaterial<StandardMaterial, BackgroundAlbedo>;

/// Original OmniPBR adds the scalar to the linear texture lookup before tint
/// and lighting. StandardMaterial alone can only multiply its texture lookup.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct BackgroundAlbedo {
    #[uniform(100)]
    tint_and_add: Vec4,
    #[uniform(101)]
    warehouse_color_roughness: Vec4,
}

impl MaterialExtension for BackgroundAlbedo {
    fn fragment_shader() -> ShaderRef {
        "game/shaders/g1_background_albedo.wgsl".into()
    }
}

#[derive(Resource)]
pub struct G1BackgroundVisualModel {
    document: Document,
    textures: HashMap<(String, bool), Vec<u8>>,
    pub file_sha256: String,
}

impl G1BackgroundVisualModel {
    /// Translation is supplied by the caller's frozen physics scene profile.
    pub fn load(path: &Path, sha256: &str, native_translation: [f64; 3]) -> Result<Self, String> {
        Self::load_profile(path, sha256, native_translation, None)
    }

    pub fn load_mobile(
        path: &Path,
        sha256: &str,
        native_translation: [f64; 3],
        background_physics_sha256: &str,
    ) -> Result<Self, String> {
        Self::load_profile(
            path,
            sha256,
            native_translation,
            Some(background_physics_sha256),
        )
    }

    fn load_profile(
        path: &Path,
        sha256: &str,
        native_translation: [f64; 3],
        background_physics_sha256: Option<&str>,
    ) -> Result<Self, String> {
        let mobile = background_physics_sha256.is_some();
        let bytes = bounded_bytes(path, sha256, 128 * 1024 * 1024)?;
        let document: Document = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if document.schema
            != if mobile {
                "native_g1_mobile_background_visual_v1"
            } else {
                "native_g1_static_background_visual_v1"
            }
            || document.source_t2_background_sha256.as_deref() != background_physics_sha256
            || document.units != "metres_z_up"
            || document.source_usd_sha256 != SOURCE_USD_SHA
            || !sha_text(&document.exporter_sha256)
            || document.source_arena_commit != "8b4a3a47fc53de23e8205089d71109a2e2348acd"
            || document.source_background_translation != [4.420, 1.408, -0.795]
            || document.native_environment_translation != native_translation
            || native_translation != [0., 0., 0.795]
            || document.selection_region_min
                != if mobile {
                    [-1.5, -3.5, 0.]
                } else {
                    [-0.25, -1.2, 0.]
                }
            || document.selection_region_max
                != if mobile {
                    [2.5, 1.5, 2.5]
                } else {
                    [1.6, 1.2, 2.5]
                }
            || document.mesh_selection
                != if mobile {
                    "whole source meshes intersecting disclosed mobile-task region"
                } else {
                    "whole source meshes intersecting disclosed static-task region"
                }
            || document.material_scope
                != "authored basic PBR inputs; MDL defaults/effects and lighting parity unproven"
            || document.source_renderer_parity_proven
            || document.physics_integrations != 0
            || document.background_physics_registered
            || document.meshes.is_empty()
            || document.meshes.len() > if mobile { 256 } else { 128 }
            || document.skipped_outside_region.len() > 4096
            || document.omitted_unmapped_geometry.len() > 4096
        {
            return Err("background source/schema/coverage contract mismatch".into());
        }
        let expected = ["jetson_orin_06", "jetson_orin_03", "hesai_box_06"];
        if (mobile && !document.source_deactivated_prims.is_empty())
            || (!mobile
                && (document.source_deactivated_prims.len() != expected.len()
                    || !document
                        .source_deactivated_prims
                        .iter()
                        .zip(expected)
                        .all(|(p, name)| p == &format!("/Lab/BackgroundAssets/boxes/{name}"))))
            || document
                .meshes
                .iter()
                .filter(|m| m.path.contains("/TaskAssets/shelf/"))
                .count()
                != 3
        {
            return Err("background startup deactivation/shelf coverage changed".into());
        }
        let dynamic_owners = [
            "/World/envs/env_0/galileo_locomanip/TaskAssets/table/Geometry/sm_tabletop_a01_01",
            "/World/envs/env_0/galileo_locomanip/TaskAssets/power_drill_physics",
        ];
        if document.meshes.iter().any(|m| {
            m.owner_path
                .as_ref()
                .is_some_and(|p| !mobile || !dynamic_owners.contains(&p.as_str()))
        }) || (mobile
            && dynamic_owners.iter().any(|p| {
                !document
                    .meshes
                    .iter()
                    .any(|m| m.owner_path.as_deref() == Some(p))
            }))
        {
            return Err("mobile background dynamic visual coverage/identity changed".into());
        }
        let mut textures = HashMap::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;
        for mesh in &document.meshes {
            total_vertices += mesh.points.len();
            total_triangles += mesh.triangles.len();
            if mesh.path.is_empty()
                || mesh.points.len() < 3
                || total_vertices > 1_000_000
                || total_triangles > 1_000_000
                || mesh.normals.len() != mesh.points.len()
                || mesh.uvs.len() != mesh.points.len()
                || mesh.triangles.is_empty()
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
                || mesh.source_normal_fallback_vertices > mesh.points.len()
                || mesh
                    .source_face_subset
                    .as_ref()
                    .is_some_and(|s| s.is_empty() || s.len() > 1_000_000)
            {
                return Err("invalid original background geometry/attributes".into());
            }
            let material = &mesh.material;
            if let Some(warehouse) = &material.warehouse_mdl {
                if warehouse.source_mdl_sha256
                    != "2e2aef644544b619f5df253eacbaed941be89bc1018fb58cbb47d0e8ac1d4192"
                    || !warehouse.albedo_desaturation.is_finite()
                    || !(0.0..=1.0).contains(&warehouse.albedo_desaturation)
                    || warehouse
                        .roughness_min_max
                        .iter()
                        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                    || warehouse.roughness_min_max[0] > warehouse.roughness_min_max[1]
                    || material.albedo.is_none()
                    || material.normal.is_none()
                    || material.orm.is_none()
                    || material.albedo_add != 0.
                    || material.roughness != 1.
                    || material.metallic != 1.
                {
                    return Err("invalid inspected warehouse MDL mapping".into());
                }
            }
            if material.source_material.is_empty()
                || material.base_color_space != "linear_rec709"
                || !material.albedo_add.is_finite()
                || !(-1.0..=1.0).contains(&material.albedo_add)
                || (material.albedo.is_none() && material.albedo_add != 0.)
                || material
                    .base_color
                    .iter()
                    .chain([&material.roughness, &material.metallic])
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || material.uv_scale.iter().any(|v| !v.is_finite() || *v <= 0.)
                || material.authored_shader_inputs.len() > 256
                || material
                    .authored_shader_inputs
                    .iter()
                    .any(|(k, v)| k.len() > 256 || v.len() > 4096)
            {
                return Err("invalid disclosed background PBR material".into());
            }
            for (texture, srgb) in [
                material.albedo.as_ref().map(|t| (t, true)),
                material.normal.as_ref().map(|t| (t, false)),
                material.orm.as_ref().map(|t| (t, false)),
            ]
            .into_iter()
            .flatten()
            {
                let key = (texture.path.clone(), srgb);
                if let Some(previous) = textures.get(&key) {
                    if format!("{:x}", Sha256::digest(previous)) != texture.sha256 {
                        return Err("conflicting background texture identity".into());
                    }
                } else {
                    textures.insert(
                        key,
                        bounded_bytes(
                            Path::new(&texture.path),
                            &texture.sha256,
                            128 * 1024 * 1024,
                        )?,
                    );
                }
            }
        }
        if textures.len() > 32 || textures.values().map(Vec::len).sum::<usize>() > 512 * 1024 * 1024
        {
            return Err("background textures exceed aggregate budget".into());
        }
        Ok(Self {
            document,
            textures,
            file_sha256: sha256.into(),
        })
    }
}

#[derive(Resource, Default, Clone, Debug, Serialize)]
pub struct G1BackgroundVisualStatus {
    pub visual_sha256: Option<String>,
    pub mesh_count: usize,
    pub texture_count: usize,
    pub omitted_unmapped_surfaces: usize,
    pub source_renderer_parity_proven: bool,
    pub background_physics_registered: bool,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct BackgroundBodyPose {
    pub path: String,
    pub translation_engine: [f32; 3],
    pub rotation_engine_xyzw: [f32; 4],
}
pub struct BackgroundOwnerFrame {
    pub physics_definition_sha256: String,
    pub bodies: Vec<BackgroundBodyPose>,
}
#[derive(Resource, Default)]
pub struct G1BackgroundVisualInput(pub Option<BackgroundOwnerFrame>);
#[derive(Component)]
struct DynamicBackgroundOwner(String);

pub struct G1BackgroundVisualPlugin;
impl Plugin for G1BackgroundVisualPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<BackgroundMaterial>::default())
            .init_resource::<G1BackgroundVisualStatus>()
            .init_resource::<G1BackgroundVisualInput>()
            .add_systems(Startup, spawn)
            .add_systems(
                PostUpdate,
                update_dynamic_owners.before(bevy::transform::TransformSystems::Propagate),
            );
    }
}

fn spawn(
    mut commands: Commands,
    model: Res<G1BackgroundVisualModel>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut status: ResMut<G1BackgroundVisualStatus>,
) {
    status.visual_sha256 = Some(model.file_sha256.clone());
    status.omitted_unmapped_surfaces = model.document.omitted_unmapped_geometry.len();
    let result = (|| -> Result<(), String> {
        let mut textures = HashMap::new();
        for ((path, srgb), bytes) in &model.textures {
            let extension = Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .ok_or("background texture has no format")?;
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
                return Err("background decoded texture exceeds pixel budget".into());
            }
            textures.insert((path.clone(), *srgb), images.add(image));
        }
        for source in &model.document.meshes {
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
            let m = &source.material;
            let texture = |t: &Texture, srgb| {
                textures
                    .get(&(t.path.clone(), srgb))
                    .cloned()
                    .ok_or_else(|| "background texture was not decoded".to_string())
            };
            let orm = m.orm.as_ref().map(|t| texture(t, false)).transpose()?;
            let material = materials.add(BackgroundMaterial {
                base: StandardMaterial {
                    // UE4 desaturates the untinted linear lookup first. Its
                    // tint is applied by the inspected fragment extension.
                    base_color: if m.warehouse_mdl.is_some() {
                        Color::WHITE
                    } else {
                        Color::linear_rgb(m.base_color[0], m.base_color[1], m.base_color[2])
                    },
                    // An unbound Gprim authors diffuse displayColor only. Do
                    // not add a specular material to these restored surfaces.
                    reflectance: if m.source_material.starts_with("usd_displayColor:") {
                        0.
                    } else {
                        0.5
                    },
                    base_color_texture: m.albedo.as_ref().map(|t| texture(t, true)).transpose()?,
                    normal_map_texture: m.normal.as_ref().map(|t| texture(t, false)).transpose()?,
                    flip_normal_map_y: m.normal_flip_tangent_v,
                    metallic: if orm.is_some() { 1. } else { m.metallic },
                    perceptual_roughness: if orm.is_some() { 1. } else { m.roughness },
                    // The warehouse graph uses only green roughness and blue
                    // metallic; its red channel is not ambient occlusion.
                    occlusion_texture: if m.warehouse_mdl.is_some() {
                        None
                    } else {
                        orm.clone()
                    },
                    metallic_roughness_texture: orm,
                    uv_transform: Affine2::from_scale(Vec2::from_array(m.uv_scale)),
                    ..default()
                },
                extension: BackgroundAlbedo {
                    tint_and_add: Vec4::new(
                        m.base_color[0],
                        m.base_color[1],
                        m.base_color[2],
                        m.albedo_add,
                    ),
                    warehouse_color_roughness: m.warehouse_mdl.as_ref().map_or(Vec4::ZERO, |w| {
                        Vec4::new(
                            w.albedo_desaturation,
                            w.roughness_min_max[0],
                            w.roughness_min_max[1],
                            1.,
                        )
                    }),
                },
            });
            let mut entity = commands.spawn((
                Name::new(source.path.clone()),
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material),
                Transform::IDENTITY,
            ));
            if let Some(path) = &source.owner_path {
                entity.insert((DynamicBackgroundOwner(path.clone()), Visibility::Hidden));
            }
            status.mesh_count += 1;
        }
        status.texture_count = textures.len();
        Ok(())
    })();
    if let Err(error) = result {
        status.error = Some(error);
    }
}

fn update_dynamic_owners(
    model: Res<G1BackgroundVisualModel>,
    input: Res<G1BackgroundVisualInput>,
    mut entities: Query<(&DynamicBackgroundOwner, &mut Transform, &mut Visibility)>,
    mut status: ResMut<G1BackgroundVisualStatus>,
) {
    let Some(frame) = &input.0 else {
        return;
    };
    if model.document.source_t2_background_sha256.as_deref()
        != Some(&frame.physics_definition_sha256)
    {
        status.error = Some("mobile background physical/render definition identity differs".into());
        return;
    }
    status.background_physics_registered = true;
    for (owner, mut transform, mut visibility) in &mut entities {
        let Some(pose) = frame.bodies.iter().find(|p| p.path == owner.0) else {
            status.error =
                Some("actual mobile background owner missing from completed frame".into());
            continue;
        };
        let q = Quat::from_array(pose.rotation_engine_xyzw);
        let p = Vec3::from_array(pose.translation_engine);
        if !p.is_finite() || !q.is_finite() || (q.length_squared() - 1.).abs() > 1e-4 {
            status.error = Some("invalid actual mobile background render pose".into());
            continue;
        }
        transform.translation = p;
        transform.rotation = q;
        *visibility = Visibility::Inherited;
    }
}

fn sha_text(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn bounded_bytes(path: &Path, sha: &str, max: u64) -> Result<Vec<u8>, String> {
    if !sha_text(sha) || std::fs::metadata(path).map_err(|e| e.to_string())?.len() > max {
        return Err("background file identity/byte bound invalid".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != sha {
        return Err("background file SHA-256 mismatch".into());
    }
    Ok(bytes)
}
