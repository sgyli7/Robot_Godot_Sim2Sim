//! Display a hash-bound robot's actual native initialization in the station.
//!
//! This development tool creates exactly one physics world, takes its initial
//! snapshot, and never integrates or admits policy control. Input qpos is an
//! explicitly hash-bound source-coordinate array, not a visual animation.

use dev_tools_minigame::visual_preview::{
    PreviewOptions, RobotInitializationPreviewResources, run_robot_initialization_preview,
};
use rapier3d::{
    math::{Pose, Rotation},
    prelude::{
        ColliderBuilder, ColliderHandle, RigidBodyBuilder, RigidBodyHandle, RigidBodyType, Vector,
    },
};
use rendering_minigame::{
    RobotRenderStyle, RobotVisualInput, RobotVisualModel, StationCameraControl, StationScene,
    StationView,
    geometry::{StationCollider, StationCollisionShape},
    robot_mesh::{RobotNormalPolicy, VerifiedRobotAppearance},
};
use robot_minigame::definition::RobotDefinition;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};
use std::{collections::HashSet, error::Error, fs, io::Write, path::Path, sync::Arc};

/// Development-only mapping: these source station shapes were never given an
/// environment contact profile. The preview must remain at zero integrations.
#[derive(Debug, Serialize)]
struct StationColliderRow {
    source_index: usize,
    tag: String,
    shape: &'static str,
    source_vertices: usize,
    source_triangles: usize,
    zero_area_source_triangles: usize,
    first_zero_area_source_triangle: Option<usize>,
    zero_area_source_triangle_indices: Vec<usize>,
    convex_hull_point_count: Option<usize>,
    convex_hull_point_bits_sha256: Option<String>,
    collider_handle: [u32; 2],
}

#[derive(Debug, Serialize)]
struct StationImportAudit {
    fixed_body_handle: [u32; 2],
    source_station_collider_count: usize,
    imported_station_collider_count: usize,
    deferred_prop_collider_count: usize,
    zero_area_source_triangle_count: usize,
    colliders: Vec<StationColliderRow>,
}

fn convex_hull_fingerprint(collider: &rapier3d::prelude::Collider) -> Option<(usize, String)> {
    let hull = collider.shape().as_convex_polyhedron()?;
    let mut digest = Sha256::new();
    digest.update((hull.points().len() as u64).to_le_bytes());
    for point in hull.points() {
        for component in point.to_array() {
            digest.update(component.to_bits().to_le_bytes());
        }
    }
    Some((hull.points().len(), format!("{:x}", digest.finalize())))
}

fn prepared_shape(
    source: &StationCollider,
    index: usize,
) -> Result<
    (
        ColliderBuilder,
        &'static str,
        usize,
        usize,
        usize,
        Option<usize>,
        Vec<usize>,
    ),
    String,
> {
    let (vertices, triangles) = match &source.shape {
        StationCollisionShape::Convex { vertices } => (vertices, None),
        StationCollisionShape::Triangles { vertices, indices } => (vertices, Some(indices)),
    };
    if vertices.iter().flatten().any(|v| !v.is_finite()) {
        return Err(format!(
            "station collider {index} ({}) has non-finite vertices",
            source.tag
        ));
    }
    let points: Vec<Vector> = vertices.iter().copied().map(Vector::from_array).collect();
    match triangles {
        None => {
            if points.len() < 4 {
                return Err(format!(
                    "station collider {index} ({}) has fewer than four convex vertices",
                    source.tag
                ));
            }
            let builder = ColliderBuilder::convex_hull(&points).ok_or_else(|| {
                format!(
                    "station collider {index} ({}) has no valid Rapier convex hull",
                    source.tag
                )
            })?;
            Ok((builder, "convex", points.len(), 0, 0, None, Vec::new()))
        }
        Some(indices) => {
            if points.len() < 3 || indices.is_empty() {
                return Err(format!(
                    "station collider {index} ({}) has an empty triangle mesh",
                    source.tag
                ));
            }
            let mut zero_area_triangles = 0usize;
            let mut first_zero_area = None;
            let mut zero_area_indices = Vec::new();
            for (triangle_index, triangle) in indices.iter().enumerate() {
                let [a, b, c] = triangle.map(|value| value as usize);
                if a >= points.len() || b >= points.len() || c >= points.len() {
                    return Err(format!(
                        "station collider {index} ({}) has invalid triangle {triangle_index}",
                        source.tag
                    ));
                }
                if (points[b] - points[a])
                    .cross(points[c] - points[a])
                    .length_squared()
                    == 0.0
                {
                    zero_area_triangles += 1;
                    first_zero_area.get_or_insert(triangle_index);
                    zero_area_indices.push(triangle_index);
                }
            }
            let builder = ColliderBuilder::trimesh(points, indices.clone()).map_err(|error| {
                format!(
                    "station collider {index} ({}) Rapier trimesh: {error:?}",
                    source.tag
                )
            })?;
            let built = builder.shape.as_trimesh().ok_or_else(|| {
                format!(
                    "station collider {index} ({}) changed shape type",
                    source.tag
                )
            })?;
            if built.indices() != indices.as_slice()
                || built.vertices().len() != vertices.len()
                || built
                    .vertices()
                    .iter()
                    .zip(vertices)
                    .any(|(actual, source)| {
                        actual.to_array().map(f32::to_bits) != (*source).map(f32::to_bits)
                    })
            {
                return Err(format!(
                    "station collider {index} ({}) lost or rewrote source triangle indices/vertex bits",
                    source.tag
                ));
            }
            Ok((
                builder,
                "triangles",
                vertices.len(),
                indices.len(),
                zero_area_triangles,
                first_zero_area,
                zero_area_indices,
            ))
        }
    }
}

/// Prebuild every station shape before touching the sole development world.
/// Movable props remain visual-only and are reported separately, never silently
/// counted as imported physical objects.
fn import_static_station(
    simulation: &mut SimulationWorld,
    sources: &[StationCollider],
) -> Result<StationImportAudit, String> {
    if simulation.counts().bodies != 0 || simulation.counts().colliders != 0 {
        return Err("station import requires the empty sole development world".into());
    }
    let mut prepared = Vec::new();
    let mut deferred_props = 0usize;
    for (index, source) in sources.iter().enumerate() {
        if source.owner != "station" {
            deferred_props += 1;
            continue;
        }
        let (
            builder,
            shape,
            vertex_count,
            triangle_count,
            zero_area_count,
            first_zero_area,
            zero_area_indices,
        ) = prepared_shape(source, index)?;
        prepared.push((
            index,
            source.tag.clone(),
            shape,
            vertex_count,
            triangle_count,
            zero_area_count,
            first_zero_area,
            zero_area_indices,
            builder,
        ));
    }
    if prepared.is_empty() {
        return Err("station scene contains no static station collider".into());
    }
    let expected = prepared.len();
    // The station's authored vertices are already in right-handed Y-up world
    // coordinates. One identity fixed body preserves that frame exactly.
    let fixed = simulation.world.bodies.insert(RigidBodyBuilder::fixed());
    let mut rows = Vec::with_capacity(expected);
    for (
        source_index,
        tag,
        shape,
        source_vertices,
        source_triangles,
        zero_area_source_triangles,
        first_zero_area_source_triangle,
        zero_area_source_triangle_indices,
        builder,
    ) in prepared
    {
        let handle = simulation.world.colliders.insert_with_parent(
            builder,
            fixed,
            &mut simulation.world.bodies,
        );
        let convex_fingerprint = if shape == "convex" {
            Some(
                convex_hull_fingerprint(&simulation.world.colliders[handle]).ok_or_else(|| {
                    format!(
                        "station collider {source_index} ({tag}) lost its convex hull on insertion"
                    )
                })?,
            )
        } else {
            None
        };
        let (index, generation) = handle.into_raw_parts();
        rows.push(StationColliderRow {
            source_index,
            tag,
            shape,
            source_vertices,
            source_triangles,
            zero_area_source_triangles,
            first_zero_area_source_triangle,
            zero_area_source_triangle_indices,
            convex_hull_point_count: convex_fingerprint.as_ref().map(|value| value.0),
            convex_hull_point_bits_sha256: convex_fingerprint.map(|value| value.1),
            collider_handle: [index, generation],
        });
    }
    let attached: HashSet<ColliderHandle> = simulation.world.bodies[fixed]
        .colliders()
        .iter()
        .copied()
        .collect();
    let imported: HashSet<ColliderHandle> = rows
        .iter()
        .map(|row| ColliderHandle::from_raw_parts(row.collider_handle[0], row.collider_handle[1]))
        .collect();
    if rows.len() != expected
        || attached != imported
        || imported.len() != expected
        || simulation.world.bodies.len() != 1
        || simulation.world.colliders.len() != expected
        || rows.iter().any(|row| {
            let handle =
                ColliderHandle::from_raw_parts(row.collider_handle[0], row.collider_handle[1]);
            simulation
                .world
                .colliders
                .get(handle)
                .is_none_or(|collider| collider.parent() != Some(fixed))
        })
        || simulation.snapshot().integration_count != 0
    {
        return Err("station collider inventory or zero-step boundary is incomplete".into());
    }
    let (fixed_index, fixed_generation) = fixed.into_raw_parts();
    Ok(StationImportAudit {
        fixed_body_handle: [fixed_index, fixed_generation],
        source_station_collider_count: expected,
        imported_station_collider_count: rows.len(),
        deferred_prop_collider_count: deferred_props,
        zero_area_source_triangle_count: rows
            .iter()
            .map(|row| row.zero_area_source_triangles)
            .sum(),
        colliders: rows,
    })
}

/// Recheck all source IDs and live generations after robot construction. No
/// contact query or physics pipeline entry is involved.
fn validate_station_after_robot(
    simulation: &SimulationWorld,
    audit: &StationImportAudit,
    sources: &[StationCollider],
) -> Result<(), String> {
    let fixed =
        RigidBodyHandle::from_raw_parts(audit.fixed_body_handle[0], audit.fixed_body_handle[1]);
    let body = simulation
        .world
        .bodies
        .get(fixed)
        .ok_or("static station body generation is no longer live")?;
    if body.body_type() != RigidBodyType::Fixed
        || body.translation().to_array().map(f32::to_bits)
            != Vector::ZERO.to_array().map(f32::to_bits)
        || body.rotation().to_array().map(f32::to_bits)
            != Rotation::IDENTITY.to_array().map(f32::to_bits)
        || body.colliders().len() != audit.colliders.len()
        || audit.colliders.len() != audit.source_station_collider_count
    {
        return Err("static station body inventory changed during robot construction".into());
    }
    let mut seen = HashSet::new();
    for row in &audit.colliders {
        let source = sources
            .get(row.source_index)
            .ok_or("station source index disappeared")?;
        if source.owner != "station" || source.tag != row.tag {
            return Err("station collider source identity changed".into());
        }
        let handle = ColliderHandle::from_raw_parts(row.collider_handle[0], row.collider_handle[1]);
        let collider = simulation
            .world
            .colliders
            .get(handle)
            .ok_or("station collider generation is no longer live")?;
        if !seen.insert(handle)
            || collider.parent() != Some(fixed)
            || collider.position_wrt_parent() != Some(&Pose::IDENTITY)
            || !body.colliders().contains(&handle)
        {
            return Err("station collider parent or generation changed".into());
        }
        match &source.shape {
            StationCollisionShape::Convex { .. } => {
                let actual =
                    convex_hull_fingerprint(collider).ok_or("station convex shape changed type")?;
                if row.convex_hull_point_count != Some(actual.0)
                    || row.convex_hull_point_bits_sha256.as_deref() != Some(actual.1.as_str())
                {
                    return Err(format!(
                        "station convex hull points changed after robot construction: {} ({})",
                        row.source_index, row.tag
                    ));
                }
            }
            StationCollisionShape::Triangles { vertices, indices } => {
                let mesh = collider
                    .shape()
                    .as_trimesh()
                    .ok_or("station triangle shape changed type")?;
                if mesh.indices() != indices.as_slice()
                    || mesh.vertices().len() != vertices.len()
                    || mesh
                        .vertices()
                        .iter()
                        .zip(vertices)
                        .any(|(actual, source)| {
                            actual.to_array().map(f32::to_bits) != (*source).map(f32::to_bits)
                        })
                {
                    return Err(format!(
                        "station triangle indices or vertex bits changed after robot construction: {} ({})",
                        row.source_index, row.tag
                    ));
                }
            }
        }
    }
    if seen.len() != audit.source_station_collider_count
        || simulation.snapshot().integration_count != 0
    {
        return Err("static station coverage or zero-step clock changed".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(9..=10).contains(&args.len()) {
        return Err("expected MODEL MODEL_SHA APPEARANCE APPEARANCE_SHA QPOS QPOS_SHA ASSETS CAPTURE REPORT [--zero-step-only|baseline|ink_off|hatch_off|both_off|thin|ink_shadow_off|robot_receive_shadow_off|robot_shadow_off|raw_normals]".into());
    }
    let zero_step_only = args.get(9).is_some_and(|value| value == "--zero-step-only");
    let mut style = RobotRenderStyle::default();
    match args.get(9).map(String::as_str).unwrap_or("baseline") {
        "baseline" | "--zero-step-only" => {}
        "raw_normals" => style.normal_policy = RobotNormalPolicy::RawCompiledDiagnostic,
        "ink_off" => style.ink_enabled = false,
        "hatch_off" => style.hatch_strength = 0.0,
        "both_off" => {
            style.ink_enabled = false;
            style.hatch_strength = 0.0;
        }
        "thin" => style.ink_pixels = 0.36,
        "ink_shadow_off" => style.ink_casts_shadows = false,
        "robot_receive_shadow_off" => {
            style.ink_casts_shadows = false;
            style.base_receives_shadows = false;
        }
        "robot_shadow_off" => {
            style.ink_casts_shadows = false;
            style.base_casts_shadows = false;
            style.base_receives_shadows = false;
        }
        _ => return Err("unknown robot-only style variant".into()),
    }
    let capture = Path::new(&args[7]);
    let output = Path::new(&args[8]);
    if capture.exists() || output.exists() {
        return Err("capture and report must be new paths".into());
    }
    let mut report = json!({
        "schema": "dev_station_robot_initialization_same_world_v1",
        "scope": "dev_only_zero_step_static_station_and_robot_geometry",
        "stage": "load_definition",
        "passed": false,
        "preflight_passed": false,
        "gpu_attempted": false,
        "zero_step_only": zero_step_only,
        "physics_world_count": 0,
        "physics_integrations": 0,
        "policy_inferences": 0,
        "station_environment_contact_qualified": false,
        "contact_filter_qualified": false,
        "full_plant_qualified": false,
        "controller_qualified": false,
        "props_physical": false,
        "trimesh_vertex_f32_bits_preserved": false,
        "convex_hull_backend_fingerprint_preserved_after_robot": false,
        "station_fixed_body_pose_bits_identity": false,
        "error": Value::Null,
    });
    let result = (|| -> Result<(), String> {
        let definition_path = Path::new(&args[0]);
        let definition = Arc::new(
            match definition_path.extension().and_then(|v| v.to_str()) {
                Some("ron") => RobotDefinition::load_ron(definition_path, &args[1]),
                Some("json") => RobotDefinition::load_json(definition_path, &args[1]),
                _ => return Err("compiled model must be .ron or .json".into()),
            }
            .map_err(|error| error.to_string())?,
        );
        report["family"] = json!(definition.model().family);
        report["model_file_sha256"] = json!(definition.file_sha256());
        report["stage"] = json!("load_appearance");
        let appearance_path = Path::new(&args[2]);
        let appearance = Arc::new(
            match appearance_path.extension().and_then(|v| v.to_str()) {
                Some("ron") => {
                    VerifiedRobotAppearance::load_ron(appearance_path, &args[3], &definition)
                }
                Some("json") => {
                    VerifiedRobotAppearance::load_json(appearance_path, &args[3], &definition)
                }
                _ => return Err("appearance must be .ron or .json".into()),
            }
            .map_err(|error| error.to_string())?,
        );
        report["appearance_file_sha256"] = json!(appearance.file_sha256());
        report["stage"] = json!("load_qpos");
        let qpos_bytes = fs::read(&args[4]).map_err(|error| error.to_string())?;
        let qpos_sha256 = digest(&qpos_bytes);
        report["qpos_file_sha256"] = json!(qpos_sha256);
        if qpos_sha256 != args[5] {
            return Err("source qpos identity mismatch".into());
        }
        let qpos: Vec<f64> =
            serde_json::from_slice(&qpos_bytes).map_err(|error| error.to_string())?;
        report["qpos_source"] = json!(qpos);
        report["stage"] = json!("load_station_scene");
        let asset_root = fs::canonicalize(&args[6]).map_err(|error| error.to_string())?;
        let scene = StationScene::load(&asset_root)?;
        report["station_identity"] = json!({
            "model_sha256": scene.0.model_sha256,
            "manifest_sha256": scene.0.manifest_sha256,
            "layout_sha256": scene.0.layout_sha256,
        });
        report["source_station_collider_count"] = json!(
            scene
                .0
                .colliders
                .iter()
                .filter(|c| c.owner == "station")
                .count()
        );
        report["deferred_prop_collider_count"] = json!(
            scene
                .0
                .colliders
                .iter()
                .filter(|c| c.owner != "station")
                .count()
        );
        let mut simulation = SimulationWorld::new();
        report["physics_world_count"] = json!(1);
        report["stage"] = json!("import_static_station_colliders");
        let station_import = import_static_station(&mut simulation, &scene.0.colliders)?;
        report["zero_area_source_triangle_count"] =
            json!(station_import.zero_area_source_triangle_count);
        report["zero_area_contact_risk"] =
            json!(if station_import.zero_area_source_triangle_count > 0 {
                "source triangles preserved exactly; contact and traversal behavior unverified"
            } else {
                "none detected; contact and traversal behavior still unverified"
            });
        report["convex_hull_identity_scope"] = json!(
            "source vertices generate Rapier convex hull; byte-level vertex equivalence is not claimed"
        );
        report["station_import"] = json!(&station_import);
        report["world_counts_after_station_import"] = json!(simulation.counts());
        report["stage"] = json!("build_robot_in_same_world");
        let assembly = build_structure(&mut simulation.world, &definition, &qpos)
            .map_err(|error| error.to_string())?;
        validate_station_after_robot(&simulation, &station_import, &scene.0.colliders)?;
        report["trimesh_vertex_f32_bits_preserved"] = json!(true);
        report["convex_hull_backend_fingerprint_preserved_after_robot"] = json!(true);
        report["station_fixed_body_pose_bits_identity"] = json!(true);
        let counts = simulation.counts();
        let snapshot = simulation.snapshot();
        report["world_counts"] = json!(counts);
        report["initial_native_snapshot"] = json!(snapshot);
        if counts.bodies != definition.model().counts.nbody
            || counts.colliders
                != station_import.imported_station_collider_count
                    + assembly.collision_handles().len()
            || snapshot.bodies.iter().filter(|body| !body.dynamic).count() != 1
            || snapshot.integration_count != 0
            || snapshot.torque_update_count != 0
            || snapshot.global_step != 0
            || snapshot.episode_step != 0
            || assembly.control_ready()
        {
            return Err(
                "same-world station/robot inventory or zero-step boundary is incomplete".into(),
            );
        }
        report["stage"] = json!("build_complete_robot_pose_frame");
        let frame = Arc::new(
            assembly
                .pose_frame(&snapshot)
                .map_err(|error| error.to_string())?,
        );
        report["body_pose_frame"] = json!(frame.as_ref());
        let root_pose = frame
            .poses
            .iter()
            .find(|pose| pose.source_body_id == 1)
            .ok_or("missing actual root pose")?;
        let mut target = root_pose.translation;
        target[1] += 0.02;
        let camera = StationCameraControl {
            view: StationView::Follow,
            target: target.into(),
            yaw: 0.7,
            pitch: 0.2,
            distance: 0.8,
            ..Default::default()
        };
        report["camera"] = json!({"target": target, "yaw": 0.7, "pitch": 0.2, "distance": 0.8});
        let model = RobotVisualModel::new(definition.clone(), appearance.clone())
            .map_err(|error| error.to_string())?
            .with_render_style(style)
            .map_err(|error| error.to_string())?;
        report["material_mapping"] = json!(model.material_mapping_report());
        report["robot_render_style"] = json!(style);
        let input = RobotVisualInput::new(&definition, frame).map_err(|error| error.to_string())?;
        report["preflight_passed"] = json!(true);
        if zero_step_only {
            report["stage"] = json!("zero_step_preflight_complete");
            report["passed"] = json!(true);
            return Ok(());
        }
        report["stage"] = json!("gpu_preview");
        report["gpu_attempted"] = json!(true);
        run_robot_initialization_preview(
            RobotInitializationPreviewResources {
                asset_root,
                scene,
                model,
                input,
                camera,
            },
            PreviewOptions {
                capture_path: Some(capture.into()),
                frames: Some(120),
                view: StationView::Follow,
            },
        )?;
        let capture_sha256 = digest(&fs::read(capture).map_err(|error| error.to_string())?);
        report["capture"] = json!(args[7]);
        report["capture_sha256"] = json!(capture_sha256);
        report["stage"] = json!("gpu_preview_complete");
        report["passed"] = json!(true);
        Ok(())
    })();
    if let Err(error) = &result {
        report["error"] = json!(error);
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?
        .write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"output": args[8], "passed": result.is_ok(), "gpu_attempted": !zero_step_only && report["gpu_attempted"] == true})
        )?
    );
    result.map_err(|error| -> Box<dyn Error> { error.into() })
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra(owner: &str, tag: &str) -> StationCollider {
        StationCollider {
            owner: owner.into(),
            tag: tag.into(),
            shape: StationCollisionShape::Convex {
                vertices: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
    }

    #[test]
    fn zero_step_import_records_every_static_generation_and_defers_props() {
        let sources = vec![
            tetra("station", "pillar"),
            StationCollider {
                owner: "station".into(),
                tag: "terrain".into(),
                shape: StationCollisionShape::Triangles {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    indices: vec![[0, 1, 2]],
                },
            },
            tetra("sample_crate", "prop"),
        ];
        let mut simulation = SimulationWorld::new();
        let audit = import_static_station(&mut simulation, &sources).unwrap();
        assert_eq!(audit.source_station_collider_count, 2);
        assert_eq!(audit.imported_station_collider_count, 2);
        assert_eq!(audit.deferred_prop_collider_count, 1);
        assert_eq!(
            audit
                .colliders
                .iter()
                .map(|row| row.source_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(simulation.counts().bodies, 1);
        assert_eq!(simulation.counts().colliders, 2);
        assert_eq!(simulation.snapshot().integration_count, 0);
        assert_eq!(simulation.snapshot().torque_update_count, 0);
        validate_station_after_robot(&simulation, &audit, &sources).unwrap();
        let fixed =
            RigidBodyHandle::from_raw_parts(audit.fixed_body_handle[0], audit.fixed_body_handle[1]);
        simulation.remove_body(fixed).unwrap();
        assert!(validate_station_after_robot(&simulation, &audit, &sources).is_err());
    }

    #[test]
    fn invalid_later_triangle_cannot_leave_a_partial_world() {
        let sources = vec![
            tetra("station", "valid_before_failure"),
            StationCollider {
                owner: "station".into(),
                tag: "bad_triangle".into(),
                shape: StationCollisionShape::Triangles {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    indices: vec![[0, 1, 3]],
                },
            },
        ];
        let mut simulation = SimulationWorld::new();
        let error = import_static_station(&mut simulation, &sources).unwrap_err();
        assert!(error.contains("bad_triangle"));
        assert_eq!(simulation.counts().bodies, 0);
        assert_eq!(simulation.counts().colliders, 0);
        assert_eq!(simulation.snapshot().integration_count, 0);
    }

    #[test]
    fn zero_area_source_triangle_is_preserved_and_flagged_without_contact_claim() {
        let sources = vec![StationCollider {
            owner: "station".into(),
            tag: "source_zero_area".into(),
            shape: StationCollisionShape::Triangles {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                indices: vec![[0, 1, 2]],
            },
        }];
        let mut simulation = SimulationWorld::new();
        let audit = import_static_station(&mut simulation, &sources).unwrap();
        assert_eq!(audit.zero_area_source_triangle_count, 1);
        assert_eq!(audit.colliders[0].first_zero_area_source_triangle, Some(0));
        assert_eq!(audit.colliders[0].zero_area_source_triangle_indices, [0]);
        let handle = ColliderHandle::from_raw_parts(
            audit.colliders[0].collider_handle[0],
            audit.colliders[0].collider_handle[1],
        );
        assert_eq!(
            simulation.world.colliders[handle]
                .shape()
                .as_trimesh()
                .unwrap()
                .indices(),
            &[[0, 1, 2]]
        );
        assert_eq!(simulation.snapshot().integration_count, 0);
    }
}
