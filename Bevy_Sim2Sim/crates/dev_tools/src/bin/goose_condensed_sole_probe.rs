//! Isolated condensed-foot normal-contact experiment, never whole-body admission.

use std::{collections::HashMap, env, fs, path::Path, process::ExitCode, time::Instant};

use rapier3d::{
    geometry::ExperimentalNormalSpring, pipeline::ContactModificationContext, prelude::*,
};
use robot_minigame::goose::plant::{GooseCollider, GoosePlant};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{
    PhysicsClockProfile, SimulationWorld, goose::builder::native_mass_properties,
};

const DT: f64 = 0.02;
const TICKS: usize = 100;
const PROGRAMS: [&str; 8] = [
    "load_5",
    "load_10",
    "load_20",
    "load_30",
    "load_40",
    "nominal_half_weight",
    "release_half_weight",
    "impact_half_weight",
];

fn vector([x, y, z]: [f64; 3]) -> Vector {
    Vector::new(x as f32, z as f32, -y as f32)
}

#[derive(Clone, Copy)]
struct PhysicalMaterial {
    stiffness: f32,
    damping: f32,
}

struct Materials {
    ground: ColliderHandle,
    soft: HashMap<ColliderHandle, PhysicalMaterial>,
}

impl PhysicsHooks for Materials {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        assert!(
            context.collider1 == self.ground || context.collider2 == self.ground,
            "guided fixture admits only the original patches/backings versus ground"
        );
        assert!(
            context.normal.x.abs() < 1e-5
                && context.normal.z.abs() < 1e-5
                && (context.normal.y.abs() - 1.0).abs() < 1e-5,
            "tilted normal needs an independent force/moment mapping"
        );
        let patch = if context.collider1 == self.ground {
            context.collider2
        } else {
            context.collider1
        };
        if let Some(material) = self.soft.get(&patch) {
            if !context.solver_contacts.is_empty() {
                let count = context.solver_contacts.len() as f32;
                *context.experimental_normal_spring = Some(ExperimentalNormalSpring {
                    stiffness_n_m: material.stiffness / count,
                    damping_n_s_m: material.damping / count,
                });
            }
        }
    }
}

fn load(program: &str, tick: usize, robot_mass: f64) -> f64 {
    match program {
        "load_5" => 30.0,
        "load_10" => 60.0,
        "load_20" => 120.0,
        "load_30" => 180.0,
        "load_40" => 240.0,
        "release_half_weight" if tick >= 50 => 0.0,
        _ => robot_mass * 9.81 / 2.0,
    }
}

fn parameter(raw: &Value, name: &str) -> Result<f64, Box<dyn std::error::Error>> {
    raw["contact_parameters"][name]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("missing finite original contact parameter {name}").into())
}

fn case(
    plant: &GoosePlant,
    raw: &Value,
    program: &str,
    normal_mode: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let foot = plant
        .bodies
        .iter()
        .find(|b| b.name == "right_ankle_roll")
        .ok_or("foot absent")?;
    let patches: Vec<&GooseCollider> = plant
        .colliders
        .iter()
        .filter(|c| {
            c.body == foot.name
                && c.contact_patch
                    .as_deref()
                    .is_some_and(|n| n.starts_with("right_sole_pad_"))
        })
        .collect();
    if patches.len() != 6 || foot.rotation_world_wxyz != [1.0, 0.0, 0.0, 0.0] {
        return Err("only the frozen six-patch condensed foot is supported".into());
    }
    let initial_gap = if program == "impact_half_weight" {
        0.0105
    } else {
        0.0005
    };
    let bottom = patches[0].local_position_m[2] - patches[0].half_extents_m.ok_or("box absent")?[2];
    let position = Vector::new(0.0, (-bottom + initial_gap) as f32, 0.0);
    let search_band = 240.0 * DT * DT / foot.mass_kg + 0.5 * DT + 0.0105 + 0.0015;
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    let world = &mut simulation.world;
    world.gravity = Vector::ZERO;
    world.integration_parameters.num_internal_pgs_iterations = 32;
    world.integration_parameters.max_ccd_substeps = 0;
    world.integration_parameters.contact_recycling = false;
    world
        .integration_parameters
        .experimental_guided_normal_block = normal_mode == "guided_block";
    world.integration_parameters.normalized_prediction_distance = search_band as f32;
    world.integration_parameters.normalized_allowed_linear_error = 0.0;
    world
        .integration_parameters
        .normalized_max_corrective_velocity = 100.0;
    world.integration_parameters.static_contact_softness = SpringCoefficients::new(1e6, 1.0);
    world.integration_parameters.contact_softness = SpringCoefficients::new(1e6, 1.0);
    let anchor = world.insert_body(RigidBodyBuilder::fixed());
    let body = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(position)
            .can_sleep(false)
            .additional_mass_properties(native_mass_properties(foot)?),
    );
    let rotation = Rotation::from_rotation_arc(Vector::X, Vector::Y);
    let guide = world
        .insert_multibody_joint(
            anchor,
            body,
            GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES)
                .local_frame1(Pose::from_parts(position, rotation))
                .local_frame2(Pose::from_rotation(rotation))
                .contacts_enabled(false),
        )
        .ok_or("guide absent")?;
    let ground = world.colliders.insert(
        ColliderBuilder::halfspace(rapier3d::na::Unit::new_unchecked(Vector::Y))
            .friction(0.0)
            .restitution(0.0)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                Group::GROUP_2,
                InteractionTestMode::And,
            )),
    );
    let mut materials = Materials {
        ground,
        soft: HashMap::new(),
    };
    let mut bindings = Vec::new();
    let mut ledger = Vec::new();
    for patch in &patches {
        let [hx, hy, hz] = patch
            .half_extents_m
            .ok_or("only original patch boxes are supported")?;
        if patch.kind != "box"
            || patch.local_rotation_wxyz != [1.0, 0.0, 0.0, 0.0]
            || (patch.local_position_m[2] - hz - bottom).abs() > 1e-12
        {
            return Err("flat original patch geometry changed".into());
        }
        let original = raw["colliders"]
            .as_array()
            .ok_or("collider ledger absent")?
            .iter()
            .find(|p| p["name"].as_str() == Some(&patch.name))
            .ok_or("source patch absent")?;
        let k = parameter(original, "stiffness_n_m")?;
        let c = parameter(original, "damping_n_s_m")?;
        let travel = parameter(original, "travel_m")?;
        if k <= 0.0 || c < 0.0 || (travel - 0.0015).abs() > 1e-12 {
            return Err("invalid original physical K/C/travel".into());
        }
        ledger.push(original.clone());
        for stop in [false, true] {
            let local = vector(patch.local_position_m)
                + if stop {
                    Vector::Y * travel as f32
                } else {
                    Vector::ZERO
                };
            let collider = world.colliders.insert_with_parent(
                ColliderBuilder::cuboid(hx as f32, hz as f32, hy as f32)
                    .translation(local)
                    .density(0.0)
                    .friction(0.0)
                    .restitution(0.0)
                    .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS)
                    .collision_groups(InteractionGroups::new(
                        Group::GROUP_2,
                        Group::GROUP_1,
                        InteractionTestMode::And,
                    )),
                body,
                &mut world.bodies,
            );
            if !stop {
                materials.soft.insert(
                    collider,
                    PhysicalMaterial {
                        stiffness: k as f32,
                        damping: c as f32,
                    },
                );
            }
            bindings.push((collider, patch.name.clone(), stop));
        }
    }
    world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
    let (mb, id) = world.multibody_joints.get_mut(guide).unwrap();
    mb.forward_kinematics(&world.bodies, true);
    mb.damping_mut().fill(0.0);
    mb.armature_mut().fill(0.0);
    mb.frictions_mut().fill(0.0);
    if program == "impact_half_weight" {
        let slot = mb.link(id).unwrap().assembly_id();
        mb.generalized_velocity_mut()[slot] = -0.5;
    }
    mb.update_rigid_bodies(&mut world.bodies, true);
    let mut frames = Vec::new();
    let mut elapsed = Vec::new();
    for tick in 0..TICKS {
        let force = load(program, tick, plant.robot_mass_kg);
        world.bodies[body].reset_forces(true);
        world.bodies[body].add_force(-Vector::Y * force as f32, true);
        let start = Instant::now();
        world.step_with_events(&materials, &());
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        let (mb, id) = world.multibody_joints.get(guide).unwrap();
        let observation = mb
            .sim2sim_observation()
            .ok_or("invalid native single-step observation")?;
        if observation.full_step_dt().to_bits() != (DT as f32).to_bits()
            || world.integration_parameters.num_solver_iterations != 1
            || !world.quarantine().is_empty()
        {
            return Err("native timing/quarantine failure; do not reset".into());
        }
        let velocity = mb.generalized_velocity()[mb.link(id).unwrap().assembly_id()] as f64;
        let compression = -(world.bodies[body].translation().y as f64 + bottom);
        let mut soft_force = serde_json::Map::new();
        let mut stop_force = serde_json::Map::new();
        let mut contacts = Vec::new();
        let mut total = 0.0;
        for (collider, patch, stop) in &bindings {
            let mut impulse = 0.0;
            if let Some(pair) = world.narrow_phase.contact_pair(*collider, ground) {
                for manifold in pair.solver_manifolds() {
                    for point in &manifold.data.solver_contacts {
                        let index = point.contact_indices()[0] as usize;
                        impulse += manifold.points[index].data.impulse as f64;
                    }
                    contacts.push(json!({"patch":patch,"travel_stop":stop,
                        "pre_integration_signed_distances_m":manifold.points.iter().map(|p| p.dist).collect::<Vec<_>>(),
                        "solver_points":manifold.data.solver_contacts.len(),
                        "point_stiffness_n_m":manifold.data.experimental_normal_spring.map(|s| s.stiffness_n_m),
                        "point_damping_n_s_m":manifold.data.experimental_normal_spring.map(|s| s.damping_n_s_m)}));
                }
            }
            let normal_force = impulse / DT;
            total += normal_force;
            if *stop {
                stop_force.insert(patch.clone(), json!(normal_force));
            } else {
                soft_force.insert(patch.clone(), json!(normal_force));
            }
        }
        if ![velocity, compression, total].iter().all(|v| v.is_finite()) {
            return Err("nonfinite fixture".into());
        }
        frames.push(
            json!({"tick":tick,"time_s":(tick+1) as f64*DT,"integration_count":tick+1,
            "native_epoch":observation.epoch,"native_full_step_dt_s":observation.full_step_dt(),
            "applied_load_n":force,"compression_m":compression,"velocity_m_s":velocity,
            "native_total_normal_force_n":total,"soft_force_per_patch_n":soft_force,
            "backing_force_per_patch_n":stop_force,"contacts":contacts}),
        );
    }
    elapsed.sort_by(f64::total_cmp);
    Ok(
        json!({"program":program,"integrations":TICKS,"physical_mass_kg":foot.mass_kg,
        "native_mass_kg":world.bodies[body].mass(),"foot_physical_ledger":foot,"patches":ledger,
        "collision_search_band_m":search_band,"physics_p95_ms":elapsed[94],"frames":frames}),
    )
}

fn hash(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: goose_condensed_sole_probe FROZEN_CONDENSED_PLANT OUTPUT_DIRECTORY [native|guided_block]".into(),
        );
    }
    let normal_mode = args
        .get(2)
        .map(|s| s.to_str().ok_or("invalid normal mode"))
        .transpose()?
        .unwrap_or("native");
    if !["native", "guided_block"].contains(&normal_mode) {
        return Err("unknown explicit normal mode".into());
    }
    let plant_path = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    if output.exists() {
        return Err("preserve prior evidence; output exists".into());
    }
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    if output
        .parent()
        .ok_or("output parent missing")?
        .canonicalize()?
        .starts_with(&project)
    {
        return Err("experimental evidence must stay outside the project".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&plant_path)?;
    if plant.candidate_id != "goose_460_condensed50_v1" {
        return Err("frozen condensed physical ledger required".into());
    }
    let raw: Value = serde_json::from_slice(&fs::read(&plant_path)?)?;
    let started = Instant::now();
    let cases: Vec<Value> = PROGRAMS
        .into_iter()
        .map(|p| case(&plant, &raw, p, normal_mode))
        .collect::<Result<_, _>>()?;
    let mut hashes = serde_json::Map::new();
    for relative in [
        "crates/dev_tools/src/bin/goose_condensed_sole_probe.rs",
        "crates/dev_tools/Cargo.toml",
        "crates/modules/simulation/src/lib.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "third_party/rapier3d/Cargo.toml",
        "third_party/rapier3d/src/geometry/contact_pair.rs",
        "third_party/rapier3d/src/geometry/narrow_phase/pair_update.rs",
        "third_party/rapier3d/src/geometry/mod.rs",
        "third_party/rapier3d/src/pipeline/physics_hooks.rs",
        "third_party/rapier3d/src/dynamics/solver/contact_constraint/generic_contact_constraint.rs",
        "third_party/rapier3d/src/dynamics/solver/contact_constraint/generic_contact_constraint_element.rs",
        "third_party/rapier3d/src/dynamics/integration_parameters.rs",
        "third_party/rapier3d/src/dynamics/solver/staged_island_solver/solve.rs",
        "Cargo.lock",
    ] {
        hashes.insert(relative.to_owned(), json!(hash(&project.join(relative))?));
    }
    let executable = env::current_exe()?.canonicalize()?;
    let receipt = json!({"schema":"goose_condensed_guided_target_physical_contact_v1",
        "revision":"goose_guided_condensed_physical_contact_v1","plant_path":plant_path,"plant_sha256":plant_hash,
        "physics_dt_s":DT,"integrations_per_tick":1,"substeps":0,"inference_count":0,"optimizer_updates":0,
        "qualified":false,"m0_passed":false,"coordinate_writes_after_initialization":0,
        "normal_model":"physical backward Euler K/C per point; restoring force retained in algebraic relaxation",
        "normal_mode":normal_mode,
        "backing_model":if normal_mode == "guided_block" { "actual gap/h; inverse-kg regularization max(1e-12,1e-7*J*WJ), same source guard" } else { "predictive native contact, static frequency 1e6 Hz damping ratio 1; differs from source finite R guard" },
        "pgs_iterations":32,"ccd_passes":0,"fixture_scope":"one native vertical multibody guide; frozen condensed tensor and six boxes; zero friction",
        "backing_scope":"six shifted diagnostic boxes, no whole-body CAD/collision admission",
        "impact_scope":"real foot inertia plus declared half-weight force, not full robot impact inertia",
        "runtime_code_sha256":hashes,"executable":executable,"executable_sha256":hash(&executable)?,
        "elapsed_wall_s":started.elapsed().as_secs_f64(),"cases":cases});
    fs::create_dir(output)?;
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!(
        "{}",
        json!({"status":"DIAGNOSTIC_ONLY","receipt":output.join("receipt.json"),"integrations":800})
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
