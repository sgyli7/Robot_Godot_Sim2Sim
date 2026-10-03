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

#[cfg(all(test, feature = "g1_constraint_diagnostic"))]
mod tests {
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use simulation_minigame::g1::task_runner::{
        ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskRunner, ArenaTaskRunnerConfig,
    };
    use std::{
        fs,
        io::Write,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    use task_minigame::policy::PolicyActionChunk;

    /// Exactly the same saved actions in two fresh native worlds. Wall time is
    /// explicitly refreshed for this offline mechanical fixture only; source
    /// episode, frame, simulation time, action values and intervals stay fixed.
    #[test]
    #[ignore = "requires pinned T1 source actions; 320 actual single-50-Hz steps per world; no images or VLA calls"]
    fn saved_t1_station_contact_comparison() -> Result<(), String> {
        let read = |name: &str| -> Result<Vec<u8>, String> {
            let path = PathBuf::from(std::env::var(name).map_err(|e| e.to_string())?);
            let sha = std::env::var(format!("{name}_SHA256")).map_err(|e| e.to_string())?;
            let bytes = bound_bytes(&path, &sha).map_err(|e| e.to_string())?;
            if bytes.len() > 1_048_576 {
                return Err("offline fixture exceeds bound".into());
            }
            Ok(bytes)
        };
        let value: serde_json::Value = serde_json::from_slice(&read("G1_STATION_REPLAY_CONFIG")?)
            .map_err(|e| e.to_string())?;
        let mut config: ArenaTaskRunnerConfig =
            serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
        let mut chunks: Vec<PolicyActionChunk> =
            serde_json::from_slice(&read("G1_STATION_REPLAY_ACTIONS")?)
                .map_err(|e| e.to_string())?;
        if chunks.len() != 8
            || chunks.iter().enumerate().any(|(i, c)| {
                c.profile != task_minigame::types::TaskProfile::StaticApple
                    || c.frames.len() != 40
                    || c.observation.episode_id != config.body.episode_id()
                    || c.observation.sim_time_ns != i as u64 * 800_000_000
            })
        {
            return Err("offline fixture changed matched episode/action timeline".into());
        }
        let use_station =
            std::env::var("G1_STATION_REPLAY_ENVIRONMENT").map_err(|e| e.to_string())?;
        if use_station == "station" {
            let scene = StationScene::load(
                &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"),
            )?;
            let identity: G1StationConfiguration =
                serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
            let ArenaTaskBodyConfig::StaticAgile(body) = &mut config.body else {
                return Err("T1 AGILE required".into());
            };
            let (_, prepared, _) = prepare(&scene, &identity, body.floor_contact_friction)?;
            body.startup_environment = Some(prepared);
        } else if use_station != "original_floor" {
            return Err("unknown comparison world".into());
        }
        let output =
            PathBuf::from(std::env::var("G1_STATION_REPLAY_OUTPUT").map_err(|e| e.to_string())?);
        let mut trace = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&output)
            .map_err(|e| e.to_string())?;
        let mut owner = ArenaTaskRunner::load_static_predictive_constraint_diagnostic(&config)
            .map_err(|e| e.to_string())?;
        for (index, chunk) in chunks.iter_mut().enumerate() {
            chunk.observation.captured_at_unix_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_millis() as u64;
            let command = ArenaTaskCommand {
                chunk: Arc::new(chunk.clone()),
                scheduled_start_sim_ns: None,
            };
            for frame in 0..40 {
                let step = owner
                    .step_with_guard(&command, &mut || Ok(()))
                    .map_err(|e| e.to_string())?;
                let tick = index as u64 * 40 + frame + 1;
                if owner.progress_counts().integration_count != tick {
                    return Err("integration count changed".into());
                }
                writeln!(trace, "{}", serde_json::json!({"phase":"offline_saved_T1_action", "body":step.body,"execution":step.execution,"environment":use_station,"offline_wall_time_refreshed":true,"fresh_images":0,"fresh_vla_calls":0,"qualified":false})).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
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
