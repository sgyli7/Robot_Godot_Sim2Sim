//! Bind public scientific-station rendering and native static collision geometry.

use rendering_minigame::{StationScene, geometry::StationCollisionShape};
use serde::{Deserialize, Serialize};
use simulation_minigame::g1::static_environment::{
    PreparedStaticEnvironment, StaticEnvironmentIdentity, StaticEnvironmentShape,
};
use std::sync::Arc;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct G1StationConfiguration {
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub layout_sha256: String,
}

#[derive(bevy::prelude::Resource)]
pub(super) struct G1StationSceneActive;

pub(super) fn prepare(
    scene: &StationScene,
    configuration: &G1StationConfiguration,
    friction: f32,
) -> Result<
    (
        StationScene,
        Arc<PreparedStaticEnvironment>,
        serde_json::Value,
    ),
    String,
> {
    if scene.0.model_sha256 != configuration.model_sha256
        || scene.0.manifest_sha256 != configuration.manifest_sha256
        || scene.0.layout_sha256 != configuration.layout_sha256
    {
        return Err(
            "scientific-station asset identity differs from the frozen configuration".into(),
        );
    }
    let shapes = scene
        .0
        .colliders
        .iter()
        .filter(|c| c.owner == "station")
        .map(|c| match &c.shape {
            StationCollisionShape::Convex { vertices } => StaticEnvironmentShape::Convex {
                vertices: vertices.clone(),
            },
            StationCollisionShape::Triangles { vertices, indices } => {
                StaticEnvironmentShape::Triangles {
                    vertices: vertices.clone(),
                    indices: indices.clone(),
                }
            }
        })
        .collect::<Vec<_>>();
    let collider_count = shapes.len();
    let environment = PreparedStaticEnvironment::prepare(
        StaticEnvironmentIdentity {
            source: scene.0.source.clone(),
            model_sha256: scene.0.model_sha256.clone(),
            manifest_sha256: scene.0.manifest_sha256.clone(),
            layout_sha256: scene.0.layout_sha256.clone(),
        },
        shapes,
        friction,
    )
    .map_err(|e| e.to_string())?;
    // Unrelated movable station props are omitted from both channels. The
    // matched task's apple/plate are rendered from their real owner snapshots.
    let mut static_scene = scene.0.as_ref().clone();
    static_scene.surfaces.retain(|s| s.owner == "station");
    static_scene.colliders.retain(|c| c.owner == "station");
    static_scene.props.clear();
    let receipt = serde_json::json!({
        "schema":"g1_native_scientific_station_preparation_v1",
        "configuration":configuration, "coordinate_frame":"right_handed_y_up_meters",
        "native_static_colliders":collider_count, "static_visual_surfaces":static_scene.surfaces.len(),
        "unrelated_movable_props_omitted_in_both_physics_and_display":scene.0.props.len(),
        "render_physics_prepared_from_same_source_read":true,
        "original_broad_floor_removed_before_first_tick":true,
        "world_or_contact_truth_input_to_decision":false, "task_qualified":false,
    });
    Ok((
        StationScene(Arc::new(static_scene)),
        Arc::new(environment),
        receipt,
    ))
}
