//! Scratch-only Rapier contact diagnostic for the D and E3 geology candidates.
//!
//! The sphere is a small proxy, not a MicroDuck foot. No controller, policy,
//! source collision world, or qualification ledger is involved.

use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, PhysicsWorld, Ray, RigidBodyBuilder, RigidBodyHandle, Vector,
};
use rendering_minigame::geometry::{StationCollisionShape, StationGeometry, load_station_geometry};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    path::Path,
};

const SOURCE_MODEL_SHA: &str = "f54b5870aa03dbb777abe1d577ce1d7e45cf2f06d7210ba0190a1d1411e445f8";
const SOURCE_MANIFEST_SHA: &str =
    "9967a5a867159267a43b3a59c34a03ad7d060a54bf0c64101fec9d02371e6163";
const D_MODEL_SHA: &str = "957f7286bcf5e8026ca4010cc988b94d8ecc6bf85ca051ad2dc27eb50d2df24d";
const D_MANIFEST_SHA: &str = "afa1ceaef0b2d35c76afe9b63a58e16baa47024aaf61cd98ec2bf032932d3193";
const E3_MODEL_SHA: &str = "91ced303df8c29387a73f678d2a37477452f6184595d0a510bb0e6a6ea1488aa";
const E3_MANIFEST_SHA: &str = "50b30290f3f8d4c507519d258b0aa317ffb1c3db78e99e695bc7b8db049bd4ce";
const STEPS: usize = 420;
const RADIUS: f32 = 0.25;

struct Imported {
    world: PhysicsWorld,
    tags: HashMap<ColliderHandle, String>,
    terrain: ColliderHandle,
    rocks: Vec<ColliderHandle>,
    station_count: usize,
}

fn file_sha(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn input_hashes(root: &Path, model: &str, manifest: &str) -> Result<Value, Box<dyn Error>> {
    let model_path = root.join("game/arts/environment/models/science_station.glb");
    let manifest_path = root.join("game/dynamic_assets/game_data/science_station.ron");
    let layout_path = root.join("game/dynamic_assets/game_data/science_station_layout.ron");
    let model_actual = file_sha(&model_path)?;
    let manifest_actual = file_sha(&manifest_path)?;
    let layout_actual = file_sha(&layout_path)?;
    if model_actual != model || manifest_actual != manifest {
        return Err(format!(
            "unexpected asset identity: model={model_actual}, manifest={manifest_actual}"
        )
        .into());
    }
    Ok(json!({
        "model_path": model_path, "model_sha256": model_actual,
        "manifest_path": manifest_path, "manifest_sha256": manifest_actual,
        "layout_path": layout_path, "layout_sha256": layout_actual,
    }))
}

fn import_station(scene: &StationGeometry) -> Result<Imported, Box<dyn Error>> {
    let mut world = PhysicsWorld::new();
    world.gravity = -Vector::Y * 9.81;
    world.integration_parameters.dt = 1.0 / 60.0;
    world.integration_parameters.num_solver_iterations = 1;
    world.integration_parameters.max_ccd_substeps = 1;
    let fixed = world.bodies.insert(RigidBodyBuilder::fixed());
    let mut tags = HashMap::new();
    let mut terrain = None;
    let mut rocks = Vec::new();
    let mut station_count = 0;
    for (i, source) in scene.colliders.iter().enumerate() {
        if source.owner != "station" {
            continue;
        }
        let builder = match &source.shape {
            StationCollisionShape::Convex { vertices } => {
                let points: Vec<_> = vertices.iter().copied().map(Vector::from_array).collect();
                ColliderBuilder::convex_hull(&points)
                    .ok_or_else(|| format!("invalid convex station shape {i}"))?
            }
            StationCollisionShape::Triangles { vertices, indices } => {
                let points: Vec<_> = vertices.iter().copied().map(Vector::from_array).collect();
                ColliderBuilder::trimesh(points, indices.clone())
                    .map_err(|error| format!("invalid station trimesh {i}: {error:?}"))?
            }
        };
        let handle = world
            .colliders
            .insert_with_parent(builder, fixed, &mut world.bodies);
        if let StationCollisionShape::Triangles { indices, .. } = &source.shape {
            if indices.len() == 25_689 {
                if terrain.replace(handle).is_some() {
                    return Err("more than one 25,689-triangle source terrain collider".into());
                }
            }
        }
        if source.tag.starts_with("v8_") {
            rocks.push(handle);
        }
        tags.insert(handle, format!("{}#{i}", source.tag));
        station_count += 1;
    }
    let terrain = terrain.ok_or("25,689-triangle source terrain collider missing")?;
    Ok(Imported {
        world,
        tags,
        terrain,
        rocks,
        station_count,
    })
}

fn down_hit(imported: &Imported, handles: &[ColliderHandle], x: f32, z: f32) -> Option<Value> {
    let ray = Ray::new(Vector::new(x, 80.0, z), -Vector::Y);
    let mut best: Option<(ColliderHandle, f32, [f32; 3])> = None;
    for &handle in handles {
        let collider = &imported.world.colliders[handle];
        if let Some(hit) =
            collider
                .shape()
                .cast_ray_and_get_normal(collider.position(), &ray, 160.0, false)
        {
            let y = ray.origin.y - hit.time_of_impact;
            if best.is_none_or(|(_, best_y, _)| y > best_y) {
                best = Some((handle, y, hit.normal.to_array()));
            }
        }
    }
    best.map(
        |(handle, y, normal)| json!({"tag": imported.tags.get(&handle), "y": y, "normal": normal}),
    )
}

struct Probe {
    name: &'static str,
    kind: &'static str,
    body: RigidBodyHandle,
    collider: ColliderHandle,
    initial: [f32; 3],
    initial_velocity: [f32; 3],
    start_terrain_y: f32,
    samples: Vec<Value>,
    active_contact_frames_by_tag: BTreeMap<String, usize>,
    normal_y_range_by_tag: BTreeMap<String, [f32; 2]>,
    max_penetration_m: f32,
    max_penetration_step: Option<usize>,
    max_penetration_tag: Option<String>,
    simultaneous_terrain_rock_contact_frames: usize,
    first_terrain_contact_step: Option<usize>,
    first_rock_contact_step: Option<usize>,
}

fn y_of(value: &Value) -> Result<f32, Box<dyn Error>> {
    Ok(value["y"].as_f64().ok_or("ray has no finite y")? as f32)
}

fn add_probe(
    imported: &mut Imported,
    name: &'static str,
    kind: &'static str,
    x: f32,
    z: f32,
    start_y: f32,
    initial_velocity: Vector,
    terrain_y: f32,
) -> Probe {
    let body = imported.world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(x, start_y, z))
            .linvel(initial_velocity)
            .can_sleep(false),
    );
    let collider = imported.world.colliders.insert_with_parent(
        ColliderBuilder::ball(RADIUS)
            .density(1.0)
            .friction(0.6)
            .restitution(0.0),
        body,
        &mut imported.world.bodies,
    );
    Probe {
        name,
        kind,
        body,
        collider,
        initial: [x, start_y, z],
        initial_velocity: initial_velocity.to_array(),
        start_terrain_y: terrain_y,
        samples: Vec::new(),
        active_contact_frames_by_tag: BTreeMap::new(),
        normal_y_range_by_tag: BTreeMap::new(),
        max_penetration_m: 0.0,
        max_penetration_step: None,
        max_penetration_tag: None,
        simultaneous_terrain_rock_contact_frames: 0,
        first_terrain_contact_step: None,
        first_rock_contact_step: None,
    }
}

fn record_step(imported: &Imported, probe: &mut Probe, step: usize) -> Result<(), Box<dyn Error>> {
    let body = &imported.world.bodies[probe.body];
    let pos = body.translation().to_array();
    let vel = body.linvel().to_array();
    if pos.iter().chain(vel.iter()).any(|v| !v.is_finite()) {
        return Err(format!("{} developed nonfinite state at step {step}", probe.name).into());
    }
    let mut active = Vec::new();
    for pair in imported.world.narrow_phase.contact_pairs() {
        let other = if pair.collider1 == probe.collider {
            pair.collider2
        } else if pair.collider2 == probe.collider {
            pair.collider1
        } else {
            continue;
        };
        if !pair.has_any_active_contact() {
            continue;
        }
        let Some(tag) = imported.tags.get(&other) else {
            return Err(format!("{} contacted an untagged collider", probe.name).into());
        };
        active.push(tag.clone());
        *probe
            .active_contact_frames_by_tag
            .entry(tag.clone())
            .or_default() += 1;
        if other == imported.terrain {
            probe.first_terrain_contact_step.get_or_insert(step);
        }
        if imported.rocks.contains(&other) {
            probe.first_rock_contact_step.get_or_insert(step);
        }
        for manifold in pair.solver_manifolds() {
            let toward_sphere = if pair.collider1 == probe.collider {
                -manifold.data.normal
            } else {
                manifold.data.normal
            };
            let range = probe
                .normal_y_range_by_tag
                .entry(tag.clone())
                .or_insert([f32::INFINITY, f32::NEG_INFINITY]);
            range[0] = range[0].min(toward_sphere.y);
            range[1] = range[1].max(toward_sphere.y);
            for point in &manifold.points {
                let penetration = (-point.dist).max(0.0);
                if penetration > probe.max_penetration_m {
                    probe.max_penetration_m = penetration;
                    probe.max_penetration_step = Some(step);
                    probe.max_penetration_tag = Some(tag.clone());
                }
            }
        }
    }
    if active
        .iter()
        .any(|tag| tag == imported.tags.get(&imported.terrain).unwrap())
        && active.iter().any(|tag| tag.starts_with("v8_"))
    {
        probe.simultaneous_terrain_rock_contact_frames += 1;
    }
    if step == 0 || step == 1 || step % 30 == 0 || step == STEPS {
        probe.samples.push(json!({
            "step": step, "position": pos, "velocity": vel, "active_contact_tags": active
        }));
    }
    Ok(())
}

fn probe_result(probe: Probe, imported: &Imported) -> Value {
    let body = &imported.world.bodies[probe.body];
    json!({
        "name": probe.name, "kind": probe.kind,
        "initial_position": probe.initial, "initial_velocity": probe.initial_velocity,
        "start_terrain_y": probe.start_terrain_y, "sphere_radius_m": RADIUS,
        "final_position": body.translation().to_array(),
        "final_velocity": body.linvel().to_array(),
        "samples": probe.samples,
        "active_contact_frames_by_tag": probe.active_contact_frames_by_tag,
        "normal_y_range_by_tag": probe.normal_y_range_by_tag,
        "maximum_contact_penetration_m": probe.max_penetration_m,
        "maximum_contact_penetration_step": probe.max_penetration_step,
        "maximum_contact_penetration_tag": probe.max_penetration_tag,
        "simultaneous_terrain_rock_contact_frames": probe.simultaneous_terrain_rock_contact_frames,
        "first_terrain_contact_step": probe.first_terrain_contact_step,
        "first_rock_contact_step": probe.first_rock_contact_step,
    })
}

fn run(
    scene: &StationGeometry,
    variant: &'static str,
    top_start_heights: [f32; 2],
    west_only: bool,
    expected_rock_colliders: usize,
) -> Result<Value, Box<dyn Error>> {
    let mut imported = import_station(scene)?;
    if (variant == "candidate" && imported.rocks.len() != expected_rock_colliders)
        || (variant == "source" && !imported.rocks.is_empty())
    {
        return Err(format!("{variant} geology collider inventory mismatch").into());
    }
    let definition = [
        ("west", -88.0_f32, -58.0_f32, 13.0_f32),
        ("north", 15.0_f32, -78.0_f32, 15.0_f32),
    ];
    let mut surface_rays = Vec::new();
    let mut probes = Vec::new();
    let definition = if west_only {
        &definition[..1]
    } else {
        &definition[..]
    };
    for (i, &(name, x, z, radius)) in definition.iter().enumerate() {
        let terrain_top = down_hit(&imported, &[imported.terrain], x, z)
            .ok_or_else(|| format!("{name} center missed source terrain"))?;
        let rock_top = down_hit(&imported, &imported.rocks, x, z);
        let terrain_y = y_of(&terrain_top)?;
        surface_rays.push(json!({
            "name": format!("{name}_center"), "xz": [x,z],
            "source_terrain": terrain_top, "candidate_rock": rock_top
        }));
        let top_name = if i == 0 {
            "west_top_drop"
        } else {
            "north_top_drop"
        };
        probes.push(add_probe(
            &mut imported,
            top_name,
            "vertical_drop_from_candidate_top_plus_2m",
            x,
            z,
            top_start_heights[i],
            Vector::ZERO,
            terrain_y,
        ));
        let seam_x = if i == 0 { -76.5 } else { 28.5 };
        let seam_terrain = down_hit(&imported, &[imported.terrain], seam_x, z)
            .ok_or_else(|| format!("{name} seam start missed source terrain"))?;
        let seam_rock = down_hit(&imported, &imported.rocks, seam_x, z);
        let seam_y = y_of(&seam_terrain)?;
        if seam_rock.as_ref().is_some_and(|hit| {
            hit["y"]
                .as_f64()
                .is_some_and(|rock_y| rock_y > seam_y as f64)
        }) {
            return Err(format!("{name} seam start has exposed new rock above terrain").into());
        }
        surface_rays.push(json!({
            "name": format!("{name}_seam_start"), "xz": [seam_x,z],
            "source_terrain": seam_terrain, "candidate_rock": seam_rock
        }));
        for step in 0..=32 {
            let profile_x = x + radius + 2.0 - step as f32 * 0.5;
            surface_rays.push(json!({
                "name": format!("{name}_east_profile_{step}"), "xz": [profile_x,z],
                "source_terrain": down_hit(&imported, &[imported.terrain], profile_x, z),
                "candidate_rock": down_hit(&imported, &imported.rocks, profile_x, z)
            }));
        }
        let seam_name = if i == 0 {
            "west_seam_approach"
        } else {
            "north_seam_approach"
        };
        probes.push(add_probe(
            &mut imported,
            seam_name,
            "free_rolling_initial_velocity_2m_per_s_from_near_boundary",
            seam_x,
            z,
            seam_y + RADIUS + 0.35,
            Vector::new(-2.0, 0.0, 0.0),
            seam_y,
        ));
    }
    let body_count = imported.world.bodies.len();
    let collider_count = imported.world.colliders.len();
    for probe in &mut probes {
        record_step(&imported, probe, 0)?;
    }
    for step in 1..=STEPS {
        imported.world.step();
        if !imported.world.quarantine().is_empty() {
            return Err(format!("{variant} quarantined a body at step {step}").into());
        }
        for probe in &mut probes {
            record_step(&imported, probe, step)?;
        }
    }
    let results: Vec<_> = probes
        .into_iter()
        .map(|probe| probe_result(probe, &imported))
        .collect();
    Ok(json!({
        "variant": variant, "station_static_collider_count": imported.station_count,
        "new_geology_triangle_collider_count": imported.rocks.len(),
        "world_body_count": body_count, "world_collider_count": collider_count,
        "source_terrain_tag": imported.tags.get(&imported.terrain),
        "surface_rays": surface_rays, "probes": results,
        "integration_steps": STEPS, "dt_s": 1.0/60.0,
        "source_collision_world_used": false,
        "robot_or_policy_used": false,
    }))
}

fn run_low_drop(scene: &StationGeometry, rock_top_y: f32) -> Result<Value, Box<dyn Error>> {
    let mut imported = import_station(scene)?;
    if imported.rocks.len() != 2 {
        return Err("low-drop control requires exactly the two E3 west colliders".into());
    }
    let terrain = down_hit(&imported, &[imported.terrain], -88.0, -58.0)
        .ok_or("low-drop center missed source terrain")?;
    let rock = down_hit(&imported, &imported.rocks, -88.0, -58.0)
        .ok_or("low-drop center missed E3 west rock")?;
    if (y_of(&rock)? - rock_top_y).abs() > 1.0e-6 {
        return Err("low-drop rock height differs from high-drop rock height".into());
    }
    // The sphere *bottom* begins 0.1 m above the same crown point. The high
    // probe uses center=surface+2 m, equivalent to 1.75 m bottom clearance.
    let mut probe = add_probe(
        &mut imported,
        "west_top_low_drop_control",
        "sphere_bottom_clearance_0p1m_above_same_west_crown",
        -88.0,
        -58.0,
        rock_top_y + RADIUS + 0.1,
        Vector::ZERO,
        y_of(&terrain)?,
    );
    let body_count = imported.world.bodies.len();
    let collider_count = imported.world.colliders.len();
    record_step(&imported, &mut probe, 0)?;
    for step in 1..=STEPS {
        imported.world.step();
        if !imported.world.quarantine().is_empty() {
            return Err(format!("low-drop control quarantined a body at step {step}").into());
        }
        record_step(&imported, &mut probe, step)?;
    }
    Ok(json!({
        "variant":"E3_candidate_low_drop_control",
        "station_static_collider_count": imported.station_count,
        "new_geology_triangle_collider_count": imported.rocks.len(),
        "world_body_count": body_count, "world_collider_count": collider_count,
        "terrain_center_ray": terrain, "rock_center_ray": rock,
        "probe": probe_result(probe, &imported),
        "integration_steps": STEPS, "dt_s": 1.0/60.0,
        "source_collision_world_used": false,
        "robot_or_policy_used": false,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let e3 = args.len() == 8 && args[6] == "--scenario" && args[7] == "e3";
    if !(args.len() == 6 || e3)
        || args[0] != "--candidate-root"
        || args[2] != "--source-root"
        || args[4] != "--output"
    {
        return Err(
            "expected --candidate-root DIR --source-root DIR --output FILE [--scenario e3]".into(),
        );
    }
    let (candidate_model_sha, candidate_manifest_sha, west_only, expected_rock_colliders) = if e3 {
        (E3_MODEL_SHA, E3_MANIFEST_SHA, true, 2)
    } else {
        (D_MODEL_SHA, D_MANIFEST_SHA, false, 4)
    };
    let candidate_root = Path::new(&args[1]);
    let source_root = Path::new(&args[3]);
    let output = Path::new(&args[5]);
    let candidate_hashes =
        input_hashes(candidate_root, candidate_model_sha, candidate_manifest_sha)?;
    let source_hashes = input_hashes(source_root, SOURCE_MODEL_SHA, SOURCE_MANIFEST_SHA)?;
    let candidate_scene = load_station_geometry(candidate_root)?;
    let source_scene = load_station_geometry(source_root)?;
    if candidate_scene.layout_sha256 != source_scene.layout_sha256 {
        return Err("source and candidate layouts differ".into());
    }
    let candidate_centers = [(-88.0_f32, -58.0_f32), (15.0_f32, -78.0_f32)];
    let candidate_static = import_station(&candidate_scene)?;
    let mut top_start_heights = [0.0; 2];
    let candidate_centers = if west_only {
        &candidate_centers[..1]
    } else {
        &candidate_centers[..]
    };
    for (i, &(x, z)) in candidate_centers.iter().enumerate() {
        let hit = down_hit(&candidate_static, &candidate_static.rocks, x, z)
            .ok_or("center ray missed candidate rock")?;
        top_start_heights[i] = y_of(&hit)? + 2.0;
    }
    let candidate = run(
        &candidate_scene,
        "candidate",
        top_start_heights,
        west_only,
        expected_rock_colliders,
    )?;
    let source = run(
        &source_scene,
        "source",
        top_start_heights,
        west_only,
        expected_rock_colliders,
    )?;
    let low_drop = if e3 {
        let rock_top_y = top_start_heights[0] - 2.0;
        Some(run_low_drop(&candidate_scene, rock_top_y)?)
    } else {
        None
    };
    let candidate_hashes_after =
        input_hashes(candidate_root, candidate_model_sha, candidate_manifest_sha)?;
    let source_hashes_after = input_hashes(source_root, SOURCE_MODEL_SHA, SOURCE_MANIFEST_SHA)?;
    if candidate_hashes != candidate_hashes_after || source_hashes != source_hashes_after {
        return Err("inputs changed during contact probe".into());
    }
    let report = json!({
        "scope": if e3 {"scratch E3 west candidate, source static comparison, sphere only"} else {"scratch D candidate, source static comparison, sphere only"},
        "candidate_inputs": candidate_hashes, "source_inputs": source_hashes,
        "candidate": candidate, "source": source,
        "candidate_low_drop_control": low_drop,
        "limits": [
            "The proxy is a free sphere, not a MicroDuck foot or leg contact path.",
            "It tests one or two crown points and near-edge approach lines, not every rock edge or spawn path.",
            "No source collision registration, action/observation contract, policy, PPO or robot navigation qualification."
        ],
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!("STATUS: success; report={}", output.display());
    Ok(())
}
