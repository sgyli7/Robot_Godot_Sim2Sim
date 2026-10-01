//! Fixed-foot pad evidence using physical tensors and the game's single 20 ms step.

use std::{env, fs, io::Write, path::Path, process::ExitCode, time::Instant};

use rapier3d::prelude::*;
use robot_minigame::goose::plant::{GooseBody, GooseJoint, GoosePlant};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{
    PhysicsClockProfile, SimulationWorld, goose::builder::native_mass_properties,
};

const PROGRAMS: [&str; 8] = [
    "step_5n",
    "step_10n",
    "step_20n",
    "step_30n",
    "release_30n",
    "upper_stop_40n",
    "lower_stop_40n",
    "unforced_velocity",
];
const TICKS: usize = 100;

fn case(
    body: &GooseBody,
    joint: &GooseJoint,
    program: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    let world = &mut simulation.world;
    world.gravity = Vector::ZERO;
    world.integration_parameters.num_internal_pgs_iterations = 4;
    let parent = world.bodies.insert(RigidBodyBuilder::fixed());
    let child = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .additional_solver_iterations(0)
            .additional_mass_properties(native_mass_properties(body)?),
    );
    world.bodies[child].recompute_mass_properties_from_colliders(&world.colliders);
    let handle = world
        .multibody_joints
        .insert(
            parent,
            child,
            GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES)
                .limits(JointAxis::LinX, joint.range.map(|v| v as f32))
                .contacts_enabled(false),
            true,
        )
        .ok_or("pad native tree absent")?;
    let (multibody, link) = world
        .multibody_joints
        .get_mut(handle)
        .ok_or("pad native link absent")?;
    multibody.forward_kinematics(&world.bodies, true);
    let slot = multibody.link(link).unwrap().assembly_id();
    multibody.damping_mut().fill(0.0);
    multibody.damping_mut()[slot] = joint.damping as f32;
    multibody.armature_mut()[slot] = joint.armature as f32;
    multibody.frictions_mut()[slot] = joint.frictionloss as f32;
    let initial = if program == "lower_stop_40n" {
        joint.range[1]
    } else if program == "unforced_velocity" {
        joint.range[1] / 2.0
    } else {
        0.0
    };
    let initial_velocity = if program == "unforced_velocity" {
        0.1
    } else {
        0.0
    };
    multibody.generalized_velocity_mut()[slot] = initial_velocity;
    let native_joint = &mut multibody.link_mut(link).unwrap().joint;
    native_joint.set_spring(0, joint.stiffness_n_m as f32, 0.0);
    if !native_joint.set_predictive_limits_enabled(true) {
        return Err("native predictive pad stop rejected".into());
    }
    native_joint.apply_displacement(&[initial as f32]);
    multibody.forward_kinematics(&world.bodies, true);
    multibody.update_rigid_bodies(&mut world.bodies, true);
    let (mass, k, c, dt, travel) = (
        body.mass_kg,
        joint.stiffness_n_m,
        joint.damping,
        0.02,
        joint.range[1],
    );
    let mut previous_energy =
        0.5 * mass * (initial_velocity as f64).powi(2) + 0.5 * k * initial.powi(2);
    let initial_energy = previous_energy;
    let mut maximum_free_energy_gain = 0.0_f64;
    let mut maximum_be_error = 0.0_f64;
    let mut rows = Vec::new();
    for tick in 0..TICKS {
        let force = match program {
            "step_5n" => 5.0,
            "step_10n" => 10.0,
            "step_20n" => 20.0,
            "step_30n" => 30.0,
            "release_30n" => {
                if tick < 50 {
                    30.0
                } else {
                    0.0
                }
            }
            "upper_stop_40n" => 40.0,
            "lower_stop_40n" => -40.0,
            "unforced_velocity" => 0.0,
            _ => return Err("unknown frozen pad program".into()),
        };
        let (multibody, link) = simulation.world.multibody_joints.get(handle).unwrap();
        let before_q = multibody.link(link).unwrap().joint.coords()[0] as f64;
        let before_v = multibody.generalized_velocity()[slot] as f64;
        simulation.world.bodies[child].reset_forces(true);
        simulation.world.bodies[child].add_force(Vector::X * force as f32, true);
        let snapshot = simulation.step_with_torques(&[])?;
        let (multibody, link) = simulation.world.multibody_joints.get(handle).unwrap();
        let q = multibody.link(link).unwrap().joint.coords()[0] as f64;
        let v = multibody.generalized_velocity()[slot] as f64;
        if !q.is_finite() || !v.is_finite() {
            return Err("nonfinite pad state".into());
        }
        let energy = 0.5 * mass * v * v + 0.5 * k * q * q;
        if force == 0.0 {
            maximum_free_energy_gain = maximum_free_energy_gain.max(energy - previous_energy);
        }
        previous_energy = energy;
        let be_q = before_q
            + dt * (mass * before_v + dt * (force - k * before_q)) / (mass + dt * c + dt * dt * k);
        if 1e-8 < be_q && be_q < travel - 1e-8 {
            maximum_be_error = maximum_be_error.max((q - be_q).abs());
        }
        rows.push(json!({"tick":tick,"integration_count":snapshot.integration_count,
            "load_n":force,"compression_m":q,"velocity_m_s":v,"physical_energy_j":energy,
            "unconstrained_backward_euler_compression_m":be_q,"limit_violation_m":0.0_f64.max(-q).max(q-travel)}));
    }
    let maximum_violation = rows
        .iter()
        .map(|r| r["limit_violation_m"].as_f64().unwrap())
        .fold(0.0_f64, f64::max);
    let static_error = if program.starts_with("step_") {
        rows.iter()
            .skip(75)
            .map(|r| {
                (r["compression_m"].as_f64().unwrap() - r["load_n"].as_f64().unwrap() / k).abs()
            })
            .fold(0.0_f64, f64::max)
    } else {
        0.0
    };
    let curve_error = if program.starts_with("step_") {
        rows.iter()
            .skip(75)
            .map(|r| {
                (k * r["compression_m"].as_f64().unwrap() - r["load_n"].as_f64().unwrap()).abs()
                    / r["load_n"].as_f64().unwrap()
            })
            .fold(0.0_f64, f64::max)
    } else {
        0.0
    };
    Ok(
        json!({"patch":body.name,"program":program,"integrations":TICKS,"rows":rows,
        "physical_body":body,"physical_joint":joint,"numerical_mass_augmentation":false,
        "max_limit_violation_m":maximum_violation,"max_unconstrained_be_error_m":maximum_be_error,
        "static_compression_error_m":static_error,"static_load_curve_relative_error":curve_error,
        "max_unforced_energy_increment_j":maximum_free_energy_gain,
        "local_checks_passed":maximum_violation<=1e-8 && maximum_be_error<=1e-8 && static_error<=0.00005 && curve_error<=0.1
            && maximum_free_energy_gain<=1e-10_f64.max(initial_energy*1e-6)}),
    )
}

fn hash(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: goose_pad_probe PLANT OUTPUT".into());
    }
    let plant_path = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    if output.exists() {
        return Err("preserve previous evidence; output already exists".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&plant_path)?;
    if plant.candidate_id != "goose_460_full50_v1" {
        return Err("fixed-foot diagnostic requires original 33-body physical ledger".into());
    }
    let started = Instant::now();
    let mut cases = Vec::new();
    for joint in plant
        .joints
        .iter()
        .filter(|j| j.kind == "slide" && j.stiffness_n_m > 0.0)
    {
        let body = plant
            .bodies
            .iter()
            .find(|b| b.name == joint.child)
            .ok_or("pad physical body absent")?;
        for program in PROGRAMS {
            cases.push(case(body, joint, program)?);
        }
    }
    if cases.len() != 96 {
        return Err("12 physical pads must each execute all eight programs".into());
    }
    let passed = cases
        .iter()
        .filter(|r| r["local_checks_passed"] == true)
        .count();
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let source_paths = [
        "crates/dev_tools/src/bin/goose_pad_probe.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/solver/joint_constraint/generic_joint_constraint_builder.rs",
    ];
    let mut hashes = serde_json::Map::new();
    for relative in source_paths {
        let path = project.join(relative);
        hashes.insert(relative.into(), json!(hash(&path)?));
    }
    let report = json!({"schema":"goose_pad_implicit_target_diagnostic_v1","engine":"rapier","engine_version":"0.35.3+local",
        "source_model_sha256":plant.model_sha256,"source_contract_sha256":plant.derived_contract_sha256,
        "plant_sha256":plant_hash,"physics_dt_s":0.02,"integrations_per_tick":1,"substeps":0,
        "qpos_writes_after_initialization":0,"solver_iterations":1,"internal_pgs_iterations":4,"max_ccd_substeps":1,
        "programs":PROGRAMS,"ticks_per_program":TICKS,"cases":cases,"passed_cases":passed,"case_count":96,
        "qualified":false,"optimizer_updates":0,"inference_count":0,"elapsed_wall_s":started.elapsed().as_secs_f64(),
        "code_sha256":hashes,"binary_sha256":hash(&env::current_exe()?)?,
        "limits":["fixed parent, no ground or friction contacts","not whole-body or impact-contact qualification"]});
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
        "STATUS DIAGNOSTIC_ONLY passed={passed}/96 {}",
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
