//! Disclosed render-only printed markers on the original task objects.
//! They inherit renderer object transforms and never modify physics. Visibility
//! has a legacy Tick200minimum. Continuous routes add a logical owner gate
//! after the four original unmarked grasp observations actually complete.

use std::{fs, path::Path};

use bevy::{
    asset::RenderAssetUsages,
    image::{CompressedImageFormats, ImageType},
    prelude::*,
    transform::TransformSystems,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::g1_task_visual::{G1TaskVisualInput, G1TaskVisualModel, ObjectRoot};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    dictionary: String,
    png_paths: [String; 2],
    png_sha256: [String; 2],
    #[serde(default)]
    layout_profile: G1FiducialLayoutProfile,
    #[serde(default)]
    calibration_version: Option<u32>,
    #[serde(default)]
    marker_mounts_source_m: Option<[[f32; 3]; 2]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum G1FiducialLayoutProfile {
    #[default]
    OriginalArena,
    /// Fixed public layout validated in geometry control0236. These labels
    /// and the auxiliary sensor never enter original learned grasp images.
    AuxiliaryGripTargets,
    /// Public small labels for a static RGB localization diagnostic. This
    /// profile never silently enters original unmarked model observations.
    StaticApplePlate,
}

/// Asset identities only. Marker dimensions/mounts are a public fixed contract.
#[derive(Resource)]
pub struct G1TaskFiducialModel {
    pngs: [Vec<u8>; 2],
    pub receipt: G1TaskFiducialReceipt,
}

/// Optional logical admission gate in addition to the original minimum Tick.
/// Continuous routes keep markers hidden until original grasp actually ends.
#[derive(Resource)]
pub struct G1TaskFiducialGate {
    pub enabled: bool,
}
impl Default for G1TaskFiducialGate {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct G1TaskFiducialReceipt {
    pub layout_profile: G1FiducialLayoutProfile,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_version: Option<u32>,
    pub schema: &'static str,
    pub dictionary: &'static str,
    pub marker_ids: [u32; 2],
    pub printed_black_square_size_m: [f32; 2],
    pub white_margin_overall_size_m: [f32; 2],
    pub object_kinds: [&'static str; 2],
    pub object_local_center_source_m: [[f32; 3]; 2],
    pub object_local_rotation_wxyz: [[f32; 4]; 2],
    pub activation_tick: u64,
    pub asset_sha256: String,
    pub physics_modified: bool,
    pub original_grasp_images_marked: bool,
    #[serde(skip_serializing_if = "unmarked")]
    pub static_localization_images_marked: bool,
}

fn unmarked(value: &bool) -> bool {
    !*value
}

impl G1TaskFiducialModel {
    pub fn load(path: &Path, sha256: &str) -> Result<Self, String> {
        let bytes = checked_bytes(path, sha256, 16 * 1024)?;
        let document: Document = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if document.schema != "g1_task_fiducials_v1" || document.dictionary != "DICT_4X4_50" {
            return Err("unsupported original-task printed-marker contract".into());
        }
        if document.layout_profile == G1FiducialLayoutProfile::StaticApplePlate {
            if document.calibration_version != Some(2)
                || document.marker_mounts_source_m != Some([[0.002, 0., 0.046], [0., 0., 0.0255]])
            {
                return Err("static public marker calibration version/mount changed".into());
            }
        } else if document.calibration_version.is_some()
            || document.marker_mounts_source_m.is_some()
        {
            return Err("static marker calibration cannot enter a mobile layout".into());
        }
        let png_path = |name: &str| {
            let png = Path::new(name);
            if png.is_absolute() {
                png.to_path_buf()
            } else {
                path.parent().unwrap_or(Path::new(".")).join(png)
            }
        };
        let pngs = [
            checked_bytes(
                &png_path(&document.png_paths[0]),
                &document.png_sha256[0],
                1024 * 1024,
            )?,
            checked_bytes(
                &png_path(&document.png_paths[1]),
                &document.png_sha256[1],
                1024 * 1024,
            )?,
        ];
        for png in &pngs {
            let image = Image::from_buffer(
                png,
                ImageType::Extension("png"),
                CompressedImageFormats::NONE,
                true,
                bevy::image::ImageSampler::nearest(),
                RenderAssetUsages::default(),
            )
            .map_err(|e| e.to_string())?;
            if (image.width(), image.height()) != (480, 480) {
                return Err(
                    "printed-marker PNG requires480square pixels with384square black tag".into(),
                );
            }
        }
        Ok(Self {
            pngs,
            receipt: G1TaskFiducialReceipt {
                layout_profile: document.layout_profile,
                calibration_version: document.calibration_version,
                schema: if document.layout_profile == G1FiducialLayoutProfile::StaticApplePlate {
                    "g1_static_disclosed_printed_markers_v1"
                } else {
                    "g1_mobile_disclosed_printed_markers_v1"
                },
                dictionary: "DICT_4X4_50",
                marker_ids: if document.layout_profile == G1FiducialLayoutProfile::StaticApplePlate
                {
                    [31, 32]
                } else {
                    [21, 22]
                },
                printed_black_square_size_m: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => [0.16, 0.10],
                    G1FiducialLayoutProfile::AuxiliaryGripTargets => [0.16, 0.06],
                    G1FiducialLayoutProfile::StaticApplePlate => [0.02, 0.06],
                },
                white_margin_overall_size_m: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => [0.20, 0.125],
                    G1FiducialLayoutProfile::AuxiliaryGripTargets => [0.20, 0.075],
                    G1FiducialLayoutProfile::StaticApplePlate => [0.025, 0.075],
                },
                object_kinds: if document.layout_profile
                    == G1FiducialLayoutProfile::StaticApplePlate
                {
                    ["t1_apple", "t1_plate"]
                } else {
                    ["t2_bin", "t2_box"]
                },
                object_local_center_source_m: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => {
                        [[0.008351, 0.0113635, 0.0045], [0.1005, 0., 0.]]
                    }
                    G1FiducialLayoutProfile::AuxiliaryGripTargets => {
                        [[0., 0.18, 0.60], [0.1005, 0., -0.04]]
                    }
                    G1FiducialLayoutProfile::StaticApplePlate => {
                        document.marker_mounts_source_m.unwrap()
                    }
                },
                object_local_rotation_wxyz: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => {
                        [[1., 0., 0., 0.], [0.70710677, 0., 0.70710677, 0.]]
                    }
                    G1FiducialLayoutProfile::AuxiliaryGripTargets => [
                        [0.70710677, 0.70710677, 0., 0.],
                        [0.70710677, 0., 0.70710677, 0.],
                    ],
                    G1FiducialLayoutProfile::StaticApplePlate => [[1., 0., 0., 0.]; 2],
                },
                activation_tick: if document.layout_profile
                    == G1FiducialLayoutProfile::StaticApplePlate
                {
                    0
                } else {
                    200
                },
                asset_sha256: sha256.into(),
                physics_modified: false,
                original_grasp_images_marked: false,
                static_localization_images_marked: document.layout_profile
                    == G1FiducialLayoutProfile::StaticApplePlate,
            },
        })
    }
}

#[derive(Component)]
struct PrintedMarker;

/// Add only with the explicit marker model and original task visual plugin.
pub struct G1TaskFiducialPlugin;
impl Plugin for G1TaskFiducialPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<G1TaskFiducialGate>();
        app.add_systems(PostStartup, spawn)
            .add_systems(PostUpdate, reveal.before(TransformSystems::Propagate));
    }
}

fn spawn(
    mut commands: Commands,
    model: Res<G1TaskFiducialModel>,
    objects: Res<G1TaskVisualModel>,
    roots: Query<(Entity, &ObjectRoot)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    for marker in 0..2 {
        let index = objects
            .object_index(model.receipt.object_kinds[marker])
            .expect("validated original task object kind");
        let (parent, _) = roots
            .iter()
            .find(|(_, root)| root.0 == index)
            .expect("original visual root is spawned before PostStartup");
        let mut image = Image::from_buffer(
            &model.pngs[marker],
            ImageType::Extension("png"),
            CompressedImageFormats::NONE,
            true,
            bevy::image::ImageSampler::nearest(),
            RenderAssetUsages::default(),
        )
        .expect("checked printed-marker PNG decodes");
        // The generator's384px black square has48px white margins on each side.
        assert_eq!((image.width(), image.height()), (480, 480));
        image.texture_descriptor.usage |=
            bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;
        let half = model.receipt.white_margin_overall_size_m[marker] * 0.5;
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [-half, 0., -half],
                [half, 0., -half],
                [half, 0., half],
                [-half, 0., half],
            ],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0., 1., 0.]; 4]);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        );
        mesh.insert_indices(bevy::mesh::Indices::U32(vec![0, 2, 1, 0, 3, 2]));
        let [x, y, z] = model.receipt.object_local_center_source_m[marker];
        let material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(images.add(image)),
            perceptual_roughness: 1.,
            metallic: 0.,
            reflectance: 0.,
            cull_mode: None,
            ..default()
        });
        let child = commands
            .spawn((
                Name::new(format!(
                    "g1_disclosed_marker_{}",
                    model.receipt.marker_ids[marker]
                )),
                PrintedMarker,
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material),
                Transform {
                    translation: Vec3::new(x, z, -y),
                    rotation: if model.receipt.layout_profile
                        == G1FiducialLayoutProfile::StaticApplePlate
                    {
                        Quat::IDENTITY
                    } else if marker == 1 {
                        Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
                    } else if model.receipt.layout_profile
                        == G1FiducialLayoutProfile::AuxiliaryGripTargets
                    {
                        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
                    } else {
                        Quat::IDENTITY
                    },
                    scale: Vec3::ONE,
                },
                Visibility::Hidden,
                bevy::light::NotShadowCaster,
            ))
            .id();
        commands.entity(parent).add_child(child);
    }
}

fn reveal(
    model: Res<G1TaskFiducialModel>,
    input: Res<G1TaskVisualInput>,
    gate: Res<G1TaskFiducialGate>,
    mut markers: Query<&mut Visibility, With<PrintedMarker>>,
) {
    let visible = gate.enabled
        && input
            .0
            .as_ref()
            .is_some_and(|f| f.source_tick >= model.receipt.activation_tick);
    for mut visibility in &mut markers {
        let next = if visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != next {
            *visibility = next;
        }
    }
}

fn checked_bytes(path: &Path, hash: &str, limit: u64) -> Result<Vec<u8>, String> {
    if hash.len() != 64 || fs::metadata(path).map_err(|e| e.to_string())?.len() > limit {
        return Err("invalid printed-marker asset identity/budget".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit || format!("{:x}", Sha256::digest(&bytes)) != hash {
        return Err("printed-marker asset SHA256 mismatch".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod static_profile_tests {
    use super::*;

    #[test]
    fn repository_static_labels_bind_relative_pixels_and_publish_exact_mounts() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../assets/g1_fiducials/static_apple_plate.json");
        let hash = format!("{:x}", Sha256::digest(fs::read(&path).unwrap()));
        let model = G1TaskFiducialModel::load(&path, &hash).unwrap();
        assert_eq!(model.receipt.marker_ids, [31, 32]);
        assert_eq!(model.receipt.object_kinds, ["t1_apple", "t1_plate"]);
        assert_eq!(model.receipt.printed_black_square_size_m, [0.02, 0.06]);
        assert_eq!(model.receipt.white_margin_overall_size_m, [0.025, 0.075]);
        assert_eq!(
            model.receipt.object_local_center_source_m,
            [[0.002, 0., 0.046], [0., 0., 0.0255]]
        );
        assert_eq!(model.receipt.activation_tick, 0);
        assert!(!model.receipt.physics_modified);
        assert!(!model.receipt.original_grasp_images_marked);
        assert!(model.receipt.static_localization_images_marked);
        assert!(G1TaskFiducialModel::load(&path, &"0".repeat(64)).is_err());
    }
}
