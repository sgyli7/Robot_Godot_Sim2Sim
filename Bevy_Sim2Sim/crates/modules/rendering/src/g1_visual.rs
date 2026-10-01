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
        mesh.compute_smooth_normals();
        let mesh = meshes.add(mesh);
        let color = source.display_color.map(|value| value as f32);
        let material = materials.add(StationMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(color[0], color[1], color[2]),
                perceptual_roughness: 0.6,
                metallic: 0.0,
                ..default()
            },
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
