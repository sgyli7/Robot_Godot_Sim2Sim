//! Inspect unchanged Goose collider leaves at zero native integrations.
//! Component/FK inputs never construct or qualify an articulated robot.

use std::{collections::HashMap, error::Error, fs, io::Write, path::Path, process::ExitCode};

use rapier3d::math::Vector;
use robot_minigame::{
    basis::source_to_engine_vector,
    goose::plant::{GooseCollider, GoosePlant},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::goose::builder::{native_collider_leaf, native_source_pose};

#[derive(Deserialize)]
struct BodyPose {
    name: String,
    translation_world_m: [f64; 3],
    rotation_world_wxyz: [f64; 4],
}

#[derive(Deserialize)]
struct GeometryInput {
    schema: String,
    bodies: Vec<BodyPose>,
    colliders: Vec<GooseCollider>,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn inspect(path: &Path) -> Result<Value, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let input: GeometryInput = serde_json::from_slice(&bytes)?;
    match input.schema.as_str() {
        "goose_collision_component_import_v1" => {}
        "goose_plant50_v1" | "goose_plant50_experimental_v2" => {
            // Complete candidates still need the existing identity/tree/tensor validation.
            // Only collider leaves are inspected here; assembly guards remain untouched.
            serde_json::from_slice::<GoosePlant>(&bytes)?.validate()?;
        }
        _ => return Err("unsupported Goose geometry inspection schema".into()),
    }
    let mut poses = HashMap::new();
    for body in &input.bodies {
        if poses
            .insert(
                body.name.clone(),
                native_source_pose(body.translation_world_m, body.rotation_world_wxyz)?,
            )
            .is_some()
        {
            return Err("duplicate source body pose".into());
        }
    }
    let mut rows = Vec::new();
    let mut rejected = 0usize;
    let mut invalid_native_faces = 0usize;
    for collider in &input.colliders {
        let body = poses.get(&collider.body).ok_or("collider owner absent")?;
        match native_collider_leaf(collider) {
            Err(error) => {
                rejected += 1;
                rows.push(json!({"name":collider.name,"body":collider.body,
                    "accepted":false,"reason":error.to_string()}));
            }
            Ok((local, shape)) => {
                let world = *body * local;
                let bounds = shape.compute_aabb(&world);
                // Observe the same public pre-merge hull stage. This does not
                // supply an alternative runtime shape or modify any vertices.
                let unmerged_hull = if let Some(source) = &collider.vertices_local_m {
                    let input: Vec<_> = source
                        .iter()
                        .map(|point| {
                            Vector::from_array(source_to_engine_vector(
                                point.map(|value| value as f32),
                            ))
                        })
                        .collect();
                    let (points, triangles) =
                        rapier3d::parry::transformation::try_convex_hull(&input)?;
                    Some(
                        json!({"points_engine_local_m":points.iter().map(|point|point.to_array()).collect::<Vec<_>>(),
                        "triangles":triangles}),
                    )
                } else {
                    None
                };
                let mut native_faces = Vec::new();
                let (points, triangles, plane_error) =
                    if let Some(convex) = shape.as_convex_polyhedron() {
                        let (points, triangles) = convex.to_trimesh();
                        let mut error = 0.0_f32;
                        for face in convex.faces() {
                            let start = face.first_vertex_or_edge as usize;
                            let end = start + face.num_vertices_or_edges as usize;
                            native_faces.push(json!({
                                "indices":&convex.vertices_adj_to_face()[start..end],
                                "normal_engine":face.normal.to_array()
                            }));
                            let first =
                                convex.vertices_adj_to_face()[face.first_vertex_or_edge as usize];
                            let anchor = convex.points()[first as usize];
                            for point in convex.points() {
                                error = error.max((*point - anchor).dot(face.normal));
                            }
                        }
                        (points, triangles, Some(error))
                    } else if let Some(cuboid) = shape.as_cuboid() {
                        let extent = cuboid.half_extents;
                        let points = [-1.0, 1.0]
                            .into_iter()
                            .flat_map(|x| {
                                [-1.0, 1.0].into_iter().flat_map(move |y| {
                                    [-1.0, 1.0].into_iter().map(move |z| {
                                        Vector::new(x * extent.x, y * extent.y, z * extent.z)
                                    })
                                })
                            })
                            .collect();
                        (points, Vec::new(), Some(0.0))
                    } else {
                        return Err("unexpected native Goose leaf type".into());
                    };
                let world_points: Vec<_> = points
                    .iter()
                    .map(|point| (world * *point).to_array())
                    .collect();
                // Exactly the native ConvexPolyhedron::check_geometry criterion,
                // reported without panicking or changing the source point set.
                let native_face_check_passed =
                    plane_error.is_some_and(|error| error <= f32::EPSILON);
                invalid_native_faces += usize::from(!native_face_check_passed);
                rows.push(json!({"name":collider.name,"body":collider.body,"accepted":true,
                    "engine_local_position_m":local.translation.to_array(),
                    "engine_local_rotation_xyzw":local.rotation.to_array(),
                    "engine_world_position_m":world.translation.to_array(),
                    "engine_world_rotation_xyzw":world.rotation.to_array(),
                    "native_vertices_engine_local_m":points.iter().map(|point|point.to_array()).collect::<Vec<_>>(),
                    "native_vertices_engine_world_m":world_points,
                    "native_triangles":triangles,"native_shape_volume_m3":shape.mass_properties(1.0).mass(),
                    "native_convex_plane_violation_m":plane_error,
                    "native_faces":native_faces,"native_face_check_passed":native_face_check_passed,
                    "unmerged_hull":unmerged_hull,
                    "native_world_aabb_min_m":bounds.mins.to_array(),
                    "native_world_aabb_max_m":bounds.maxs.to_array()}));
            }
        }
    }
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let code_paths = [
        "crates/dev_tools/src/bin/goose_collision_import_probe.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/robot/src/basis.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "Cargo.lock",
    ];
    let mut code = serde_json::Map::new();
    for path in code_paths {
        code.insert(path.into(), json!(hash(&fs::read(project.join(path))?)));
    }
    Ok(
        json!({"schema":"goose_native_collision_import_report_v1","input_sha256":hash(&bytes),
        "executable_sha256":hash(&fs::read(std::env::current_exe()?)?),"runtime_code_sha256":code,
        "input_schema":input.schema,"input_body_pose_count":input.bodies.len(),
        "collider_count":rows.len(),"rejected_count":rejected,"rows":rows,
        "all_leaves_constructed":rejected==0,"invalid_native_face_count":invalid_native_faces,
        "native_face_validation_passed":rejected==0 && invalid_native_faces==0,
        "integrations":0,"torque_updates":0,
        "inferences":0,"optimizer_updates":0,"initial_lift_m":0.0,
        "scope":"shape construction and zero-step native coordinates only; no body mass or collision masks changed",
        "robot_constructed":false,"source_qualified":false,"target_qualified":false,
        "contact_or_ccd_qualified":false,"condensed_contact_guard_bypassed":false}),
    )
}

fn main() -> Result<ExitCode, Box<dyn Error>> {
    let raw: Vec<_> = std::env::args().skip(1).collect();
    if raw.len() != 4 || raw[0] != "--input" || raw[2] != "--output" {
        return Err("expected --input JSON --output fresh_JSON".into());
    }
    let result = inspect(Path::new(&raw[1]))?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&raw[3])?
        .write_all(&serde_json::to_vec_pretty(&result)?)?;
    let accepted = result["native_face_validation_passed"]
        .as_bool()
        .unwrap_or(false);
    println!(
        "{}",
        json!({"output":raw[3],"all_leaves_constructed":result["all_leaves_constructed"],"native_face_validation_passed":accepted,"integrations":0})
    );
    Ok(if accepted {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
