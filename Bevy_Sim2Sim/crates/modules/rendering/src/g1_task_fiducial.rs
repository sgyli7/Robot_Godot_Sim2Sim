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
    #[serde(default)]
    static_apple_side_markers: Vec<StaticAppleSideMarker>,
    #[serde(default)]
    mobile_bin_board_markers: Vec<MobileBinBoardMarker>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum G1FiducialLayoutProfile {
    #[default]
    OriginalArena,
    /// Fixed public layout validated in geometry control0236. These labels
    /// and the auxiliary sensor never enter original learned grasp images.
    AuxiliaryGripTargets,
    /// Separate public rigid three-label T2 board. These extra observations
    /// remain hidden throughout the original learned grasp images.
    AuxiliaryBinBoardTargets,
    /// Public small labels for a static RGB localization diagnostic. This
    /// profile never silently enters original unmarked model observations.
    StaticApplePlate,
    /// Separate fixed public six-face layout for grasp-induced object rotation.
    StaticApplePlateMultiFace,
}

impl G1FiducialLayoutProfile {
    pub fn is_static(self) -> bool {
        matches!(
            self,
            Self::StaticApplePlate | Self::StaticApplePlateMultiFace
        )
    }
}
/// Fixed public marker-to-apple transform; bytes and poses are calibration only.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StaticAppleSideMarker {
    pub marker_id: u32,
    pub png_path: String,
    pub png_sha256: String,
    pub center_source_m: [f32; 3],
    pub rotation_wxyz: [f32; 4],
}
/// Same printed asset format, attached only to the explicitly selected T2 bin.
pub type MobileBinBoardMarker = StaticAppleSideMarker;

fn validate_bin_board_layout(
    profile: G1FiducialLayoutProfile,
    markers: &[MobileBinBoardMarker],
) -> Result<(), String> {
    if profile != G1FiducialLayoutProfile::AuxiliaryBinBoardTargets {
        return if markers.is_empty() {
            Ok(())
        } else {
            Err("bin board labels require their separate public profile".into())
        };
    }
    let q = [
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
        0.,
        0.,
    ];
    if markers.len() != 2
        || markers
            .iter()
            .zip([(23, -0.25), (24, 0.25)])
            .any(|(m, (id, x))| {
                m.marker_id != id
                    || m.center_source_m != [x, 0.18, 0.45]
                    || m.rotation_wxyz
                        .iter()
                        .zip(q)
                        .any(|(a, b)| (a - b).abs() > 1e-7)
            })
    {
        return Err("public rigid bin board identity/mount changed".into());
    }
    Ok(())
}
fn validate_side_layout(
    profile: G1FiducialLayoutProfile,
    sides: &[StaticAppleSideMarker],
) -> Result<(), String> {
    if profile != G1FiducialLayoutProfile::StaticApplePlateMultiFace {
        return if sides.is_empty() {
            Ok(())
        } else {
            Err("side labels require their separate static multi-face profile".into())
        };
    }
    let expected = [
        (
            33,
            [-0.030, 0., 0.014],
            [
                std::f32::consts::FRAC_1_SQRT_2,
                0.,
                -std::f32::consts::FRAC_1_SQRT_2,
                0.,
            ],
        ),
        (
            34,
            [0.002, -0.031, 0.014],
            [
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.,
                0.,
            ],
        ),
        (
            35,
            [0.033, 0., 0.014],
            [
                std::f32::consts::FRAC_1_SQRT_2,
                0.,
                std::f32::consts::FRAC_1_SQRT_2,
                0.,
            ],
        ),
        (
            36,
            [0.002, 0.031, 0.014],
            [
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2,
                0.,
                0.,
            ],
        ),
        (37, [0.002, 0., -0.017], [0., 1., 0., 0.]),
    ];
    if sides.len() != expected.len()
        || sides.iter().zip(expected).any(|(a, (id, p, q))| {
            a.marker_id != id
                || a.center_source_m != p
                || a.rotation_wxyz
                    .iter()
                    .zip(q)
                    .any(|(a, b)| (a - b).abs() > 1e-7)
        })
    {
        return Err("public multi-face marker identity/mount changed".into());
    }
    Ok(())
}
/// Asset identities only. Marker dimensions/mounts are a public fixed contract.
#[derive(Resource)]
pub struct G1TaskFiducialModel {
    pngs: [Vec<u8>; 2],
    side_pngs: Vec<Vec<u8>>,
    bin_board_pngs: Vec<Vec<u8>>,
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub static_apple_side_markers: Vec<StaticAppleSideMarker>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mobile_bin_board_markers: Vec<MobileBinBoardMarker>,
}

fn unmarked(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bin_board_cannot_change_its_rigid_mount_or_enter_an_old_profile() {
        let markers: Vec<_> = [(23, -0.25), (24, 0.25)]
            .into_iter()
            .map(|(marker_id, x)| MobileBinBoardMarker {
                marker_id,
                png_path: "bound_print.png".into(),
                png_sha256: "0".repeat(64),
                center_source_m: [x, 0.18, 0.45],
                rotation_wxyz: [
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                    0.,
                    0.,
                ],
            })
            .collect();
        let profile = G1FiducialLayoutProfile::AuxiliaryBinBoardTargets;
        assert!(validate_bin_board_layout(profile, &markers).is_ok());
        assert!(
            validate_bin_board_layout(G1FiducialLayoutProfile::AuxiliaryGripTargets, &markers)
                .is_err()
        );
        assert!(validate_bin_board_layout(profile, &markers[..1]).is_err());
        let mut changed = markers.clone();
        changed[0].center_source_m[2] += 0.001;
        assert!(validate_bin_board_layout(profile, &changed).is_err());
        let mut changed = markers.clone();
        changed[1].marker_id = 23;
        assert!(validate_bin_board_layout(profile, &changed).is_err());
    }
    #[test]
    #[ignore = "explicit hash-bound public marker profile;0physics/camera/model calls"]
    fn load_public_six_face_calibration_and_check_source_axes() -> Result<(), String> {
        let path = std::env::var("G1_MULTI_FACE_FIDUCIAL_PATH").map_err(|e| e.to_string())?;
        let hash = std::env::var("G1_MULTI_FACE_FIDUCIAL_SHA256").map_err(|e| e.to_string())?;
        let model = G1TaskFiducialModel::load(Path::new(&path), &hash)?;
        assert_eq!(model.receipt.calibration_version, Some(3));
        assert_eq!(model.side_pngs.len(), 5);
        assert!(!model.receipt.physics_modified);
        let normals = [Vec3::NEG_X, Vec3::Z, Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y];
        for (side, normal) in model.receipt.static_apple_side_markers.iter().zip(normals) {
            let [w, x, y, z] = side.rotation_wxyz;
            let rotation = Quat::from_xyzw(x, z, -y, w).normalize();
            assert!((rotation * Vec3::Y - normal).length() < 1e-6);
        }
        let mut sides = model.receipt.static_apple_side_markers;
        assert!(validate_side_layout(G1FiducialLayoutProfile::StaticApplePlate, &sides).is_err());
        sides[0].center_source_m[0] += 0.001;
        assert!(
            validate_side_layout(G1FiducialLayoutProfile::StaticApplePlateMultiFace, &sides)
                .is_err()
        );
        Ok(())
    }
}

impl G1TaskFiducialModel {
    pub fn load(path: &Path, sha256: &str) -> Result<Self, String> {
        let bytes = checked_bytes(path, sha256, 16 * 1024)?;
        let document: Document = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if document.schema != "g1_task_fiducials_v1" || document.dictionary != "DICT_4X4_50" {
            return Err("unsupported original-task printed-marker contract".into());
        }
        if document.layout_profile.is_static() {
            if document.calibration_version
                != Some(
                    if document.layout_profile == G1FiducialLayoutProfile::StaticApplePlateMultiFace
                    {
                        3
                    } else {
                        2
                    },
                )
                || document.marker_mounts_source_m != Some([[0.002, 0., 0.046], [0., 0., 0.0255]])
            {
                return Err("static public marker calibration version/mount changed".into());
            }
        } else if document.calibration_version.is_some()
            || document.marker_mounts_source_m.is_some()
        {
            return Err("static marker calibration cannot enter a mobile layout".into());
        }
        validate_side_layout(document.layout_profile, &document.static_apple_side_markers)?;
        validate_bin_board_layout(document.layout_profile, &document.mobile_bin_board_markers)?;
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
        let side_pngs: Vec<_> = document
            .static_apple_side_markers
            .iter()
            .map(|side| checked_bytes(&png_path(&side.png_path), &side.png_sha256, 1024 * 1024))
            .collect::<Result<_, _>>()?;
        let bin_board_pngs: Vec<_> = document
            .mobile_bin_board_markers
            .iter()
            .map(|side| checked_bytes(&png_path(&side.png_path), &side.png_sha256, 1024 * 1024))
            .collect::<Result<_, _>>()?;
        for png in pngs
            .iter()
            .chain(side_pngs.iter())
            .chain(bin_board_pngs.iter())
        {
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
            side_pngs,
            bin_board_pngs,
            receipt: G1TaskFiducialReceipt {
                layout_profile: document.layout_profile,
                calibration_version: document.calibration_version,
                schema: if document.layout_profile.is_static() {
                    "g1_static_disclosed_printed_markers_v1"
                } else {
                    "g1_mobile_disclosed_printed_markers_v1"
                },
                dictionary: "DICT_4X4_50",
                marker_ids: if document.layout_profile.is_static() {
                    [31, 32]
                } else {
                    [21, 22]
                },
                printed_black_square_size_m: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => [0.16, 0.10],
                    G1FiducialLayoutProfile::AuxiliaryGripTargets
                    | G1FiducialLayoutProfile::AuxiliaryBinBoardTargets => [0.16, 0.06],
                    G1FiducialLayoutProfile::StaticApplePlate
                    | G1FiducialLayoutProfile::StaticApplePlateMultiFace => [0.02, 0.06],
                },
                white_margin_overall_size_m: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => [0.20, 0.125],
                    G1FiducialLayoutProfile::AuxiliaryGripTargets
                    | G1FiducialLayoutProfile::AuxiliaryBinBoardTargets => [0.20, 0.075],
                    G1FiducialLayoutProfile::StaticApplePlate
                    | G1FiducialLayoutProfile::StaticApplePlateMultiFace => [0.025, 0.075],
                },
                object_kinds: if document.layout_profile.is_static() {
                    ["t1_apple", "t1_plate"]
                } else {
                    ["t2_bin", "t2_box"]
                },
                object_local_center_source_m: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => {
                        [[0.008351, 0.0113635, 0.0045], [0.1005, 0., 0.]]
                    }
                    G1FiducialLayoutProfile::AuxiliaryGripTargets
                    | G1FiducialLayoutProfile::AuxiliaryBinBoardTargets => {
                        [[0., 0.18, 0.60], [0.1005, 0., -0.04]]
                    }
                    G1FiducialLayoutProfile::StaticApplePlate
                    | G1FiducialLayoutProfile::StaticApplePlateMultiFace => {
                        document.marker_mounts_source_m.unwrap()
                    }
                },
                object_local_rotation_wxyz: match document.layout_profile {
                    G1FiducialLayoutProfile::OriginalArena => {
                        [[1., 0., 0., 0.], [0.70710677, 0., 0.70710677, 0.]]
                    }
                    G1FiducialLayoutProfile::AuxiliaryGripTargets
                    | G1FiducialLayoutProfile::AuxiliaryBinBoardTargets => [
                        [0.70710677, 0.70710677, 0., 0.],
                        [0.70710677, 0., 0.70710677, 0.],
                    ],
                    G1FiducialLayoutProfile::StaticApplePlate
                    | G1FiducialLayoutProfile::StaticApplePlateMultiFace => [[1., 0., 0., 0.]; 2],
                },
                activation_tick: if document.layout_profile.is_static() {
                    0
                } else {
                    200
                },
                asset_sha256: sha256.into(),
                physics_modified: false,
                original_grasp_images_marked: false,
                static_localization_images_marked: document.layout_profile.is_static(),
                static_apple_side_markers: document.static_apple_side_markers,
                mobile_bin_board_markers: document.mobile_bin_board_markers,
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
                    rotation: if model.receipt.layout_profile.is_static() {
                        Quat::IDENTITY
                    } else if marker == 1 {
                        Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
                    } else if matches!(
                        model.receipt.layout_profile,
                        G1FiducialLayoutProfile::AuxiliaryGripTargets
                            | G1FiducialLayoutProfile::AuxiliaryBinBoardTargets
                    ) {
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
    // All additional quads remain renderer children; no physics component,
    // collision shape, mass, material or actuator parameter is added.
    for (side, png, kind, half) in model
        .receipt
        .static_apple_side_markers
        .iter()
        .zip(&model.side_pngs)
        .map(|(side, png)| (side, png, "t1_apple", 0.0125))
        .chain(
            model
                .receipt
                .mobile_bin_board_markers
                .iter()
                .zip(&model.bin_board_pngs)
                .map(|(side, png)| (side, png, "t2_bin", 0.075)),
        )
    {
        let index = objects
            .object_index(kind)
            .expect("validated labelled task object");
        let (parent, _) = roots
            .iter()
            .find(|(_, root)| root.0 == index)
            .expect("labelled task root spawned");
        let mut image = Image::from_buffer(
            png,
            ImageType::Extension("png"),
            CompressedImageFormats::NONE,
            true,
            bevy::image::ImageSampler::nearest(),
            RenderAssetUsages::default(),
        )
        .expect("checked side marker");
        image.texture_descriptor.usage |=
            bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;
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
        let material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(images.add(image)),
            perceptual_roughness: 1.,
            metallic: 0.,
            reflectance: 0.,
            cull_mode: Some(bevy::render::render_resource::Face::Back),
            ..default()
        });
        let [x, y, z] = side.center_source_m;
        let [w, qx, qy, qz] = side.rotation_wxyz;
        let child = commands
            .spawn((
                Name::new(format!("g1_disclosed_marker_{}", side.marker_id)),
                PrintedMarker,
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material),
                Transform {
                    translation: Vec3::new(x, z, -y),
                    rotation: Quat::from_xyzw(qx, qz, -qy, w).normalize(),
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
            .join("../../../crates/dev_tools/python/fixtures/g1_fiducials/static_apple_plate.json");
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
