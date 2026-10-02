//! Exact frozen USD triangle geometry following read-only native body poses.

use std::{path::Path, sync::Arc};

use bevy::{asset::RenderAssetUsages, prelude::*, transform::TransformSystems};
use robot_minigame::g1::definition::{G1BodyFrame, USD_SHA256};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{
    StationScene,
    material::{InkMaterial, StationEnamel, StationMaterial},
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VisualDocument {
    usd_sha256: String,
    body_names: Vec<String>,
    visuals: Vec<SourceVisual>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceVisual {
    body: usize,
    path: String,
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    display_color: [f64; 3],
    #[serde(default)]
    bound_material: Option<SourceVisualMaterial>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceVisualMaterial {
    path: String,
    shader_id: String,
    source_input: String,
    diffuse_color_linear: [f32; 3],
}

/// Frozen model data; loading binds both the triangle file and source USD hash.
#[derive(Resource, Clone)]
pub struct G1VisualModel {
    document: Arc<VisualDocument>,
    pub file_sha256: String,
}

impl G1VisualModel {
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
        if file_sha256 != expected_sha256 {
            return Err("G1 visual file SHA256 mismatch".into());
        }
        let document: VisualDocument =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if document.usd_sha256 != USD_SHA256
            || document.body_names.len() != 53
            || document.body_names[19] != "head_link"
            || document.visuals.len() != 49
        {
            return Err("G1 visual USD identity or body topology mismatch".into());
        }
        for visual in &document.visuals {
            if visual.body >= 53
                || visual.path.is_empty()
                || visual.points.len() < 3
                || visual.triangles.is_empty()
                || visual
                    .points
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite() || !(*value as f32).is_finite())
                || visual
                    .triangles
                    .iter()
                    .flatten()
                    .any(|index| *index as usize >= visual.points.len())
                || visual
                    .display_color
                    .iter()
                    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                || visual.bound_material.as_ref().is_some_and(|material| {
                    material.path.is_empty()
                        || material.shader_id != "UsdPreviewSurface"
                        || material.source_input != "diffuse_color_constant"
                        || material
                            .diffuse_color_linear
                            .iter()
                            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                })
            {
                return Err("invalid frozen G1 visual mesh".into());
            }
        }
        Ok(Self {
            document: Arc::new(document),
            file_sha256,
        })
    }

    pub fn mesh_count(&self) -> usize {
        self.document.visuals.len()
    }
    pub fn source_usd_sha256(&self) -> &str {
        &self.document.usd_sha256
    }
    pub fn source_material_mesh_count(&self) -> usize {
        self.document
            .visuals
            .iter()
            .filter(|visual| visual.bound_material.is_some())
            .count()
    }
}

/// Supplied by the same native completed frame that drives the ego camera.
#[derive(Resource, Default)]
pub struct G1VisualInput(pub Option<G1BodyFrame>);

#[derive(Resource, Default, Debug)]
pub struct G1VisualStatus {
    pub mesh_count: usize,
    pub episode_id: Option<u64>,
    pub source_tick: Option<u64>,
    pub error: Option<String>,
}

#[derive(Component)]
struct G1VisualBody(usize);

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum G1VisualSystems {
    SyncPose,
}

pub struct G1VisualPlugin;

impl Plugin for G1VisualPlugin {
    fn build(&self, app: &mut App) {
        // A source-task diagnostic may render the exact G1 with a matching
        // simple physical floor, without loading unrelated station geometry.
        if !app.is_plugin_added::<MaterialPlugin<StationMaterial>>() {
            app.add_plugins(MaterialPlugin::<StationMaterial>::default());
        }
        if !app.is_plugin_added::<MaterialPlugin<InkMaterial>>() {
            app.add_plugins(MaterialPlugin::<InkMaterial>::default());
        }
        app.init_resource::<G1VisualInput>()
            .init_resource::<G1VisualStatus>()
            .add_systems(Startup, spawn_visuals)
            .add_systems(
                PostUpdate,
                sync_visuals
                    .in_set(G1VisualSystems::SyncPose)
                    .before(TransformSystems::Propagate),
            );
    }
}

fn spawn_visuals(
    mut commands: Commands,
    model: Res<G1VisualModel>,
    station: Res<StationScene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StationMaterial>>,
    mut source_materials: ResMut<Assets<StandardMaterial>>,
    mut inks: ResMut<Assets<InkMaterial>>,
    mut status: ResMut<G1VisualStatus>,
) {
    let roots: Vec<_> = model
        .document
        .body_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            commands
                .spawn((
                    Name::new(format!("g1_{name}")),
                    G1VisualBody(index),
                    Transform::IDENTITY,
                    Visibility::Hidden,
                ))
                .id()
        })
        .collect();
    let ink = inks.add(InkMaterial::default());
    for source in &model.document.visuals {
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        let points: Vec<[f32; 3]> = source
            .points
            .iter()
            .map(|point| [point[0] as f32, point[2] as f32, -point[1] as f32])
            .collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, points);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; source.points.len()]);
        mesh.insert_indices(bevy::mesh::Indices::U32(
            source.triangles.iter().flatten().copied().collect(),
        ));
        if source.bound_material.is_some() {
            compute_source_normals(&mut mesh);
        } else {
            mesh.compute_smooth_normals();
        }
        let mesh = meshes.add(mesh);
        let color = source.bound_material.as_ref().map_or_else(
            || {
                let color = source.display_color.map(|value| value as f32);
                Color::srgb(color[0], color[1], color[2])
            },
            |material| {
                let [r, g, b] = material.diffuse_color_linear;
                Color::linear_rgba(r, g, b, 1.)
            },
        );
        let base = StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.6,
            metallic: 0.0,
            ..default()
        };
        if source.bound_material.is_some() {
            // The station shader imposes cool bands and procedural hatching,
            // absent from the source robot. Bound USD colors use the normal
            // PBR path, with no ink shell, under the same scene illumination.
            let surface = commands
                .spawn((
                    Name::new(source.path.clone()),
                    Mesh3d(mesh),
                    MeshMaterial3d(source_materials.add(base)),
                    Transform::IDENTITY,
                ))
                .id();
            commands.entity(roots[source.body]).add_child(surface);
        } else {
            let material = materials.add(StationMaterial {
                base,
                extension: StationEnamel::new(0, &station.0.layout),
            });
            let surface = commands
                .spawn((
                    Name::new(source.path.clone()),
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material),
                    Transform::IDENTITY,
                ))
                .id();
            let outline = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(ink.clone()),
                    Transform::IDENTITY,
                ))
                .id();
            commands
                .entity(roots[source.body])
                .add_children(&[surface, outline]);
        }
    }
    status.mesh_count = model.document.visuals.len();
}

fn sync_visuals(
    input: Res<G1VisualInput>,
    mut status: ResMut<G1VisualStatus>,
    mut roots: Query<(&G1VisualBody, &mut Transform, &mut Visibility)>,
) {
    let Some(frame) = &input.0 else {
        for (_, _, mut visibility) in &mut roots {
            *visibility = Visibility::Hidden;
        }
        return;
    };
    let order_valid = status.episode_id.is_none_or(|episode| {
        frame.episode_id > episode
            || (frame.episode_id == episode
                && status
                    .source_tick
                    .is_none_or(|tick| frame.source_tick >= tick))
    });
    if let Err(error) = frame
        .validate()
        .map_err(|error| error.to_string())
        .and_then(|()| {
            if order_valid {
                Ok(())
            } else {
                Err("stale G1 body frame".into())
            }
        })
    {
        status.error = Some(error);
        for (_, _, mut visibility) in &mut roots {
            *visibility = Visibility::Hidden;
        }
        return;
    }
    for (body, mut transform, mut visibility) in &mut roots {
        let pose = frame
            .bodies
            .iter()
            .find(|pose| pose.body == body.0)
            .expect("validated complete body frame");
        transform.translation = Vec3::from_array(pose.translation);
        transform.rotation = Quat::from_array(pose.rotation_xyzw);
        *visibility = Visibility::Inherited;
    }
    status.episode_id = Some(frame.episode_id);
    status.source_tick = Some(frame.source_tick);
    status.error = None;
}

/// Compute source-render normals without modifying any mesh positions/indices.
fn compute_source_normals(mesh: &mut Mesh) {
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
            // Bevy's default compares squared edge-length products against
            // f32::EPSILON, discarding millimeter-scale G1 faces in meters.
            // Keep angle weighting, evaluating only truly degenerate edges
            // as zero; no mesh scaling or replacement normals are introduced.
            let denominator = (u.length_squared() * v.length_squared()).sqrt();
            if denominator > 0. {
                let angle = (u.dot(v) / denominator).clamp(-1., 1.).acos();
                normals[index] += face.as_vec3() * angle as f32;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_triangle_in_meters_keeps_unit_normals_without_changing_geometry() {
        let positions = vec![[0., 0., 0.], [0.001, 0., 0.], [0., 0.002, 0.]];
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
        mesh.insert_indices(bevy::mesh::Indices::U32(vec![0, 1, 2]));
        compute_source_normals(&mut mesh);
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("missing source normals");
        };
        for normal in normals {
            assert!(Vec3::from_array(*normal).abs_diff_eq(Vec3::Z, 1e-6));
        }
        assert_eq!(
            mesh.indices().unwrap().iter().collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap(),
            &bevy::mesh::VertexAttributeValues::Float32x3(positions)
        );
    }

    #[test]
    #[ignore = "requires frozen actual six source normal meshes; no GPU or physical steps"]
    fn actual_source_arm_mesh_normals_diagnostic() {
        use sha2::{Digest, Sha256};
        use std::{fs, path::PathBuf};
        #[derive(Deserialize)]
        struct Source {
            usd_sha256: String,
            qualified: bool,
            physical_steps: u64,
            rows: Vec<Row>,
        }
        #[derive(Deserialize)]
        struct Row {
            path: String,
            points_body_source: Vec<[f64; 3]>,
            triangles: Vec<[u32; 3]>,
            authored_face_corner_normals_body_source: Vec<[f64; 3]>,
            orientation: String,
        }
        let input = PathBuf::from(std::env::var("G1_SOURCE_NORMALS_FIXTURE").unwrap());
        let expected = std::env::var("G1_SOURCE_NORMALS_FIXTURE_SHA256").unwrap();
        assert!(fs::metadata(&input).unwrap().len() < 64 * 1024 * 1024);
        let bytes = fs::read(input).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
        let source: Source = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            source.usd_sha256,
            "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"
        );
        assert!(!source.qualified);
        assert_eq!(source.physical_steps, 0);
        assert_eq!(source.rows.len(), 6);
        let mut results = Vec::new();
        for row in source.rows {
            assert_eq!(row.orientation, "rightHanded");
            assert_eq!(
                row.authored_face_corner_normals_body_source.len(),
                3 * row.triangles.len()
            );
            let positions = row
                .points_body_source
                .iter()
                .map(|p| p.map(|v| v as f32))
                .collect::<Vec<_>>();
            let mut mesh = Mesh::new(
                bevy::mesh::PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_indices(bevy::mesh::Indices::U32(
                row.triangles.iter().flatten().copied().collect(),
            ));
            compute_source_normals(&mut mesh);
            let Some(bevy::mesh::VertexAttributeValues::Float32x3(normals)) =
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                panic!("missing actual source normals");
            };
            let mut zero = 0;
            let mut cosine_sum = 0.0_f64;
            let mut minimum_cosine = 1.0_f64;
            for (index, author) in row
                .triangles
                .iter()
                .flatten()
                .zip(&row.authored_face_corner_normals_body_source)
            {
                let computed = Vec3::from_array(normals[*index as usize]).as_dvec3();
                zero += usize::from(computed.length_squared() == 0.);
                let authored = bevy::math::DVec3::from_array(*author).normalize();
                let cosine = computed.dot(authored);
                cosine_sum += cosine;
                minimum_cosine = minimum_cosine.min(cosine);
            }
            results.push(serde_json::json!({
                "path": row.path, "corner_count": row.authored_face_corner_normals_body_source.len(),
                "zero_corners": zero, "minimum_cosine": minimum_cosine,
                "mean_cosine": cosine_sum / row.authored_face_corner_normals_body_source.len() as f64,
            }));
        }
        let output = PathBuf::from(std::env::var("G1_SOURCE_NORMALS_OUTPUT").unwrap());
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)
            .unwrap();
        serde_json::to_writer_pretty(
            &mut file,
            &serde_json::json!({
                "qualified": false, "physical_steps": 0, "gpu_frames": 0,
                "source_fixture_sha256": expected, "actual_rust_source_render_normal_rule": true,
                "results": results,
            }),
        )
        .unwrap();
        assert!(
            results
                .iter()
                .all(|r| r["zero_corners"] == 0 && r["mean_cosine"].as_f64().unwrap() > 0.99)
        );
    }
}
