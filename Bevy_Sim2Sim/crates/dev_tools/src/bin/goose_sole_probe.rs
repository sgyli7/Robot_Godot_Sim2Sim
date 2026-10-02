//! Guided physical foot and six real spring pads in contact with a fixed ground.

use std::{env, fs, io::Write, path::Path, process::ExitCode, time::Instant};

use rapier3d::prelude::*;
use robot_minigame::goose::plant::GoosePlant;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{
    PhysicsClockProfile, SimulationWorld, goose::builder::native_mass_properties,
};

const PROGRAMS: [&str; 7] = [
    "load_5n_per_patch",
    "load_10n_per_patch",
    "load_20n_per_patch",
    "load_30n_per_patch",
    "nominal_half_weight",
    "release_half_weight",
    "impact_half_weight",
];
const TICKS: usize = 100;

fn vector([x, y, z]: [f64; 3]) -> Vector {
    Vector::new(x as f32, z as f32, -y as f32)
}

fn load(program: &str, tick: usize, total_mass: f64) -> f64 {
    match program {
        "load_5n_per_patch" => 30.0,
        "load_10n_per_patch" => 60.0,
        "load_20n_per_patch" => 120.0,
        "load_30n_per_patch" => 180.0,
        "release_half_weight" if tick >= 50 => 0.0,
        _ => total_mass * 9.81 / 2.0,
    }
}

fn case(
    plant: &GoosePlant,
    program: &str,
    pgs: usize,
    ccd: usize,
    normal_mode: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let foot = plant
        .bodies
        .iter()
        .find(|b| b.name == "right_ankle_roll")
        .ok_or("foot absent")?;
    let pads: Vec<_> = plant
        .joints
        .iter()
        .filter(|j| j.parent == foot.name && j.kind == "slide")
        .collect();
    if pads.len() != 6 || foot.rotation_world_wxyz != [1.0, 0.0, 0.0, 0.0] {
        return Err("fixture requires six original pads and an identity foot frame".into());
    }
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    let world = &mut simulation.world;
    world.gravity = Vector::ZERO;
    world.integration_parameters.num_internal_pgs_iterations = pgs;
    world.integration_parameters.normalized_prediction_distance = 0.002;
    world.integration_parameters.max_ccd_substeps = ccd;
    if normal_mode == "predictive_rigid" {
        world.integration_parameters.normalized_allowed_linear_error = 0.0;
        world
            .integration_parameters
            .normalized_max_corrective_velocity = 100.0;
        world.integration_parameters.contact_softness = SpringCoefficients::new(1.0e6, 1.0);
        world.integration_parameters.static_contact_softness = SpringCoefficients::new(1.0e6, 1.0);
        world.integration_parameters.contact_recycling = false;
    }
    let initial_lift = if program == "impact_half_weight" {
        0.01
    } else {
        0.0
    };
    let lift = Vector::Y * initial_lift;
    let axis_rotation = Rotation::from_rotation_arc(Vector::X, Vector::Y);
    let anchor = world.bodies.insert(RigidBodyBuilder::fixed());
    let foot_pose = Pose::from_translation(vector(foot.translation_world_m) + lift);
    let foot_handle = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .pose(foot_pose)
            .can_sleep(false)
            .additional_solver_iterations(0)
            .additional_mass_properties(native_mass_properties(foot)?),
    );
    let foot_frame = Pose::from_parts(foot_pose.translation, axis_rotation);
    let guide = world
        .multibody_joints
        .insert(
            anchor,
            foot_handle,
            GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES)
                .local_frame1(foot_frame)
                .local_frame2(foot_pose.inverse() * foot_frame)
                .contacts_enabled(false),
            true,
        )
        .ok_or("vertical guide absent")?;
    let ground = world.colliders.insert(
        ColliderBuilder::cuboid(1.0, 0.1, 1.0)
            .translation(Vector::new(0.0, -0.1, 0.0))
            .friction(0.0)
            .restitution(0.0)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                Group::GROUP_2,
                InteractionTestMode::And,
            )),
    );
    let mut handles = Vec::new();
    let mut physical_mass = foot.mass_kg;
    for joint in &pads {
        let body = plant
            .bodies
            .iter()
            .find(|b| b.name == joint.child)
            .ok_or("pad absent")?;
        let geometry = plant
            .colliders
            .iter()
            .find(|c| c.body == body.name)
            .ok_or("pad shape absent")?;
        if body.rotation_world_wxyz != [1.0, 0.0, 0.0, 0.0]
            || geometry.local_position_m != [0.0; 3]
            || geometry.local_rotation_wxyz != [1.0, 0.0, 0.0, 0.0]
            || joint.axis_world != [0.0, 0.0, 1.0]
        {
            return Err("fixture does not support changed pad geometry or axis".into());
        }
        physical_mass += body.mass_kg;
        let pose = Pose::from_translation(vector(body.translation_world_m) + lift);
        let child = world.bodies.insert(
            RigidBodyBuilder::dynamic()
                .pose(pose)
                .can_sleep(false)
                .additional_solver_iterations(0)
                .additional_mass_properties(native_mass_properties(body)?),
        );
        let [hx, hy, hz] = geometry
            .half_extents_m
            .ok_or("pad must remain an original box")?;
        let collider = world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(hx as f32, hz as f32, hy as f32)
                .density(0.0)
                .friction(0.0)
                .restitution(0.0)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_2,
                    Group::GROUP_1,
                    InteractionTestMode::And,
                )),
            child,
            &mut world.bodies,
        );
        let frame = Pose::from_parts(vector(joint.origin_world_m) + lift, axis_rotation);
        let handle = world
            .multibody_joints
            .insert(
                foot_handle,
                child,
                GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES)
                    .local_frame1(foot_pose.inverse() * frame)
                    .local_frame2(pose.inverse() * frame)
                    .limits(JointAxis::LinX, joint.range.map(|x| x as f32))
                    .contacts_enabled(false),
                true,
            )
            .ok_or("pad link absent")?;
        handles.push((handle, child, collider, body, *joint, hz));
    }
    for body in world.bodies.iter_mut().map(|(_, body)| body) {
        body.recompute_mass_properties_from_colliders(&world.colliders);
    }
    let (multibody, _) = world.multibody_joints.get_mut(guide).unwrap();
    multibody.forward_kinematics(&world.bodies, true);
    multibody.damping_mut().fill(0.0);
    multibody.armature_mut().fill(0.0);
    multibody.frictions_mut().fill(0.0);
    for (handle, _, _, _, joint, _) in &handles {
        let (multibody, index) = world.multibody_joints.get_mut(*handle).unwrap();
        let slot = multibody.link(index).unwrap().assembly_id();
        multibody.damping_mut()[slot] = joint.damping as f32;
        multibody.armature_mut()[slot] = joint.armature as f32;
        multibody.frictions_mut()[slot] = joint.frictionloss as f32;
        let native = &mut multibody.link_mut(index).unwrap().joint;
        native.set_spring(0, joint.stiffness_n_m as f32, 0.0);
        if !native.set_predictive_limits_enabled(true) {
            return Err("predictive stop rejected".into());
        }
    }
    let (multibody, index) = world.multibody_joints.get_mut(guide).unwrap();
    if program == "impact_half_weight" {
        let slot = multibody.link(index).unwrap().assembly_id();
        multibody.generalized_velocity_mut()[slot] = -0.5;
    }
    multibody.forward_kinematics(&world.bodies, true);
    multibody.update_rigid_bodies(&mut world.bodies, true);
    let mut rows = Vec::new();
    let mut walls = Vec::new();
    for tick in 0..TICKS {
        let downward = load(program, tick, plant.robot_mass_kg);
        simulation.world.bodies[foot_handle].reset_forces(true);
        simulation.world.bodies[foot_handle].add_force(-Vector::Y * downward as f32, true);
        let start = Instant::now();
        let snapshot = simulation.step_with_torques(&[])?;
        walls.push(start.elapsed().as_secs_f64() * 1000.0);
        if snapshot.integration_count != tick as u64 + 1 {
            return Err("integration count mismatch".into());
        }
        let world = &simulation.world;
        let (multibody, index) = world.multibody_joints.get(guide).unwrap();
        let foot_v =
            multibody.generalized_velocity()[multibody.link(index).unwrap().assembly_id()] as f64;
        let mut pad_rows = Vec::new();
        for (handle, child, collider, body, joint, hz) in &handles {
            let (multibody, index) = world.multibody_joints.get(*handle).unwrap();
            let q = multibody.link(index).unwrap().joint.coords()[0] as f64;
            let v = multibody.generalized_velocity()[multibody.link(index).unwrap().assembly_id()]
                as f64;
            let bottom = world.bodies[*child].translation().y as f64 - hz;
            let fk_error = (world.bodies[*child].translation()
                - multibody.link(index).unwrap().local_to_world().translation)
                .length() as f64;
            let expected_bottom = world.bodies[foot_handle].translation().y as f64
                + body.translation_world_m[2]
                - foot.translation_world_m[2]
                + q
                - hz;
            let pair = world.narrow_phase.contact_pair(*collider, ground);
            let distance = pair.and_then(|pair| {
                pair.solver_manifolds()
                    .iter()
                    .flat_map(|m| &m.points)
                    .map(|p| p.dist as f64)
                    .reduce(f64::min)
            });
            let impulse = pair
                .map(|pair| pair.total_impulse_magnitude() as f64)
                .unwrap_or(0.0);
            if ![q, v, bottom, impulse, foot_v]
                .iter()
                .all(|x| x.is_finite())
            {
                return Err("nonfinite foot state".into());
            }
            pad_rows.push(
                json!({"patch":body.name,"compression_m":q,"relative_velocity_m_s":v,
                "limit_violation_m":0.0_f64.max(-q).max(q-joint.range[1]),
                "post_integration_bottom_m":bottom,"pre_integration_solver_distance_m":distance,
                "body_to_native_link_translation_error_m":fk_error,
                "body_to_joint_geometry_error_m":(bottom-expected_bottom).abs(),
                "normal_impulse_n_s":impulse,"physical_spring_force_n":joint.stiffness_n_m*q}),
            );
        }
        rows.push(json!({"tick":tick,"integration_count":snapshot.integration_count,"time_s":(tick+1) as f64*0.02,
            "downward_load_n":downward,"foot_vertical_velocity_m_s":foot_v,
            "foot_origin_height_m":world.bodies[foot_handle].translation().y,"pads":pad_rows}));
    }
    walls.sort_by(f64::total_cmp);
    let max_penetration = rows
        .iter()
        .flat_map(|r| r["pads"].as_array().unwrap())
        .map(|r| -r["post_integration_bottom_m"].as_f64().unwrap())
        .fold(0.0_f64, f64::max);
    let max_violation = rows
        .iter()
        .flat_map(|r| r["pads"].as_array().unwrap())
        .map(|r| r["limit_violation_m"].as_f64().unwrap())
        .fold(0.0_f64, f64::max);
    let static_rows = if program.starts_with("load_") || program == "nominal_half_weight" {
        &rows[75..]
    } else {
        &rows[0..0]
    };
    let max_fk_error = rows
        .iter()
        .flat_map(|r| r["pads"].as_array().unwrap())
        .map(|r| {
            r["body_to_native_link_translation_error_m"]
                .as_f64()
                .unwrap()
        })
        .fold(0.0_f64, f64::max);
    let curve_error = static_rows
        .iter()
        .map(|r| {
            let expected = r["downward_load_n"].as_f64().unwrap();
            let spring_sum: f64 = r["pads"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["physical_spring_force_n"].as_f64().unwrap())
                .sum();
            (spring_sum - expected).abs() / expected
        })
        .fold(0.0_f64, f64::max);
    Ok(
        json!({"program":program,"integrations":TICKS,"physical_mass_kg":physical_mass,
        "max_post_integration_penetration_m":max_penetration,"max_limit_violation_m":max_violation,
        "max_body_to_native_link_translation_error_m":max_fk_error,
        "static_total_spring_load_relative_error":curve_error,"physics_p95_ms":walls[94],"rows":rows}),
    )
}

fn hash(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: goose_sole_probe PHYSICAL_PLANT OUTPUT PGS(4|32) CCD(0|1) [native|predictive_rigid]".into());
    }
    let plant_path = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    if output.exists() {
        return Err("preserve previous evidence; output already exists".into());
    }
    let pgs: usize = args[2].to_str().ok_or("PGS invalid")?.parse()?;
    if ![4, 32].contains(&pgs) {
        return Err("only declared bounded PGS4/32 controls are supported".into());
    }
    let ccd: usize = args[3].to_str().ok_or("CCD invalid")?.parse()?;
    if ccd > 1 {
        return Err("only zero or one CCD pass, never time subdivision".into());
    }
    let normal_mode = match args.get(4) {
        Some(value) => value
            .to_str()
            .ok_or("normal-contact experiment must be valid UTF-8")?,
        None => "native",
    };
    if !["native", "predictive_rigid"].contains(&normal_mode) {
        return Err("unknown explicit normal-contact experiment".into());
    }
    if normal_mode == "predictive_rigid" && ccd != 0 {
        return Err("normal-contact experiment requires coherent CCD-off integration".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&plant_path)?;
    if plant.candidate_id != "goose_460_full50_v1" {
        return Err("use unchanged original physical ledger".into());
    }
    let start = Instant::now();
    let mut cases = Vec::new();
    for program in PROGRAMS {
        cases.push(case(&plant, program, pgs, ccd, normal_mode)?);
    }
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let mut hashes = serde_json::Map::new();
    for relative in [
        "crates/dev_tools/src/bin/goose_sole_probe.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "Cargo.lock",
    ] {
        hashes.insert(relative.into(), json!(hash(&project.join(relative))?));
    }
    let executable = env::current_exe()?;
    let report = json!({"schema":"goose_guided_sole_target_diagnostic_v1","engine":"rapier3d_0.35.3_local",
        "plant_sha256":plant_hash,"source_model_sha256":plant.model_sha256,"physics_dt_s":0.02,
        "integrations_per_tick":1,"substeps":0,"coordinate_writes_after_initialization":0,
        "qualified":false,"m0_passed":false,"optimizer_updates":0,"inference_count":0,
        "programs":PROGRAMS,"ticks_per_program":TICKS,"internal_pgs_iterations":pgs,
        "max_ccd_substeps":ccd,"post_integration_motion_clamping":ccd == 1,
        "normal_contact_experiment":normal_mode,
        "prediction_distance_m":0.002,"contact_recycling":normal_mode != "predictive_rigid",
        "normal_softness_hz":if normal_mode=="predictive_rigid" {[1.0e6,1.0e6]} else {[30.0,60.0]},
        "normal_softness_damping_ratio":if normal_mode=="predictive_rigid" {1.0} else {10.0},
        "allowed_linear_error_m":if normal_mode=="predictive_rigid" {0.0} else {0.005},
        "max_corrective_velocity_m_s":if normal_mode=="predictive_rigid" {100.0} else {3.0},
        "contact_law":"native normal-contact solve with declared coefficients; restitution=0; friction=0",
        "fixture_scope":"original physical right foot + six pads; vertical guide locks horizontal/rotational motion; no other foot geometry",
        "external_load":"applied downward on physical foot; zero gravity; no artificial mass augmentation",
        "initial_impact_gap_added_m":0.01,"initial_impact_velocity_m_s":-0.5,
        "runtime_code_sha256":hashes,"executable":executable,"executable_sha256":hash(&executable)?,
        "cases":cases,"elapsed_wall_s":start.elapsed().as_secs_f64(),
        "limitations":["no whole-body qualification","source/target native normal contact laws differ and must be reported","guided fixture does not test foot rotation, friction, gait or policy"]});
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    println!(
        "STATUS DIAGNOSTIC_ONLY cases=7 ticks=700 {}",
        output.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("STATUS FAILED {error}");
            ExitCode::FAILURE
        }
    }
}
