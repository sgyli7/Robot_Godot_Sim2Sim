//! Bounded fixed-head four-bar evidence, using the game's guarded 50 Hz step.

use std::{
    collections::HashMap,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use rapier3d::{
    math::{Matrix, Pose, Rotation, Vector},
    prelude::*,
};
use robot_minigame::{
    basis::{engine_to_source_vector, source_to_engine_rotation, source_to_engine_vector},
    goose::plant::GoosePlant,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{
    BodyTorque, PhysicsClockProfile, SimulationWorld, goose::builder::native_mass_properties,
};

const JOINT_NAMES: [&str; 3] = ["beak_hinge", "beak_input_rotor", "beak_coupler_link"];
const ANGLE_TOLERANCE_RAD: f32 = 0.005;
const PIN_TOLERANCE_M: f32 = 0.0001;
const LIMIT_TOLERANCE_RAD: f32 = 0.005;

struct JawFixture {
    simulation: SimulationWorld,
    bodies: HashMap<String, RigidBodyHandle>,
    joints: [MultibodyJointHandle; 3],
    pin: ImpulseJointHandle,
    pin_local_jaw: Vector,
    pin_local_coupler: Vector,
    grip_local_jaw: Vector,
    jaw_anchor_local: Vector,
    axis: Vector,
    ranges: [[f32; 2]; 3],
    physical_ledger: Value,
}

impl JawFixture {
    fn build(
        plant: &GoosePlant,
        grip: [f64; 3],
        initial: f32,
        pgs: usize,
        pin_basis: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
        let world = &mut simulation.world;
        world.gravity = Vector::ZERO;
        world.integration_parameters.num_internal_pgs_iterations = pgs;
        let mut bodies = HashMap::new();
        let mut poses = HashMap::new();
        let mut ledger = Vec::new();
        for name in std::iter::once("head_roll").chain(JOINT_NAMES) {
            let body = plant
                .bodies
                .iter()
                .find(|body| body.name == name)
                .ok_or("missing jaw body")?;
            let pose = source_pose(body.translation_world_m, body.rotation_world_wxyz)?;
            let builder = if name == "head_roll" {
                RigidBodyBuilder::fixed()
            } else {
                RigidBodyBuilder::dynamic()
            };
            let handle = world.bodies.insert(
                builder
                    .pose(pose)
                    .additional_mass_properties(native_mass_properties(body)?)
                    .can_sleep(false)
                    .additional_solver_iterations(0),
            );
            world.bodies[handle].recompute_mass_properties_from_colliders(&world.colliders);
            let native = &world.bodies[handle].mass_properties().local_mprops;
            let observed = native.reconstruct_inertia_matrix();
            // Independently undo B=[x,z,-y] for the measured complete tensor.
            let index = [0, 2, 1];
            let sign = [1.0, -1.0, 1.0];
            let tensor: [[f64; 3]; 3] = std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    sign[row] * sign[column] * observed.col(index[column])[index[row]] as f64
                })
            });
            let norm = body
                .inertia_at_com_body_kg_m2
                .iter()
                .flatten()
                .map(|v| v * v)
                .sum::<f64>();
            let error = (tensor
                .iter()
                .flatten()
                .zip(body.inertia_at_com_body_kg_m2.iter().flatten())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / norm)
                .sqrt();
            if error > 2e-5 {
                return Err("native jaw full inertia differs from source".into());
            }
            ledger.push(json!({"source_body":body,"native_mass_kg":native.mass(),
                "native_com_source_local_m":to_source(native.local_com),
                "native_full_inertia_source_kg_m2":tensor,"relative_tensor_error":error,
                "fixed":name=="head_roll"}));
            bodies.insert(name.to_owned(), handle);
            poses.insert(name.to_owned(), pose);
        }
        let mut joints = Vec::new();
        let mut physical_joints = Vec::new();
        for name in JOINT_NAMES {
            let joint = plant
                .joints
                .iter()
                .find(|joint| joint.name == name)
                .ok_or("missing jaw joint")?;
            let axis = source_vector(joint.axis_world);
            let frame = Pose::from_parts(
                source_vector(joint.origin_world_m),
                Rotation::from_rotation_arc(Vector::X, axis),
            );
            let handle = world
                .multibody_joints
                .insert(
                    bodies[&joint.parent],
                    bodies[&joint.child],
                    GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
                        .local_frame1(poses[&joint.parent].inverse() * frame)
                        .local_frame2(poses[&joint.child].inverse() * frame)
                        .limits(JointAxis::AngX, joint.range.map(|v| v as f32))
                        .contacts_enabled(false),
                    true,
                )
                .ok_or("jaw articulation is not a native tree")?;
            joints.push(handle);
            physical_joints.push(joint);
        }
        let joints: [MultibodyJointHandle; 3] =
            joints.try_into().map_err(|_| "wrong jaw joint count")?;
        let (multibody, _) = world
            .multibody_joints
            .get_mut(joints[0])
            .ok_or("jaw multibody absent")?;
        multibody.forward_kinematics(&world.bodies, true);
        if multibody.ndofs() != 3 {
            return Err("fixed-head fixture must have exactly three DoFs".into());
        }
        multibody.damping_mut().fill(0.0);
        multibody.armature_mut().fill(0.0);
        multibody.frictions_mut().fill(0.0);
        for (joint, displacement) in physical_joints.iter().zip([initial, initial, -initial]) {
            let link_id = multibody
                .links()
                .find(|link| link.rigid_body_handle() == bodies[&joint.child])
                .ok_or("missing native jaw link")?
                .link_id();
            let slot = multibody.link(link_id).unwrap().assembly_id();
            multibody.damping_mut()[slot] = joint.damping as f32;
            // These are the physical source armatures, never M+h*C augmentation.
            multibody.armature_mut()[slot] = joint.armature as f32;
            multibody.frictions_mut()[slot] = joint.frictionloss as f32;
            multibody
                .link_mut(link_id)
                .unwrap()
                .joint
                .apply_displacement(&[displacement]);
        }
        // Initial conditions only. All subsequent poses come from native integration.
        multibody.forward_kinematics(&world.bodies, true);
        multibody.update_rigid_bodies(&mut world.bodies, true);
        let axis = source_vector(plant.jaw_loop.rotation_axis_world);
        let pin_rotation = if pin_basis == "coupler_axis" {
            // Same planar coincidence constraint, with its first row along the
            // coupler. That row has no coupler-hinge angular lever, avoiding
            // two near-parallel rows dominated by the light coupler's response.
            let x = (source_vector(plant.jaw_loop.output_pin_world_m)
                - source_vector(physical_joints[2].origin_world_m))
            .normalize();
            Rotation::from_mat3(&Matrix::from_cols(x, axis.cross(x), axis)).normalize()
        } else {
            Rotation::from_rotation_arc(Vector::Z, axis)
        };
        let frame = Pose::from_parts(
            source_vector(plant.jaw_loop.output_pin_world_m),
            pin_rotation,
        );
        let frame1 = poses["beak_coupler_link"].inverse() * frame;
        let frame2 = poses["beak_hinge"].inverse() * frame;
        let pin = world.impulse_joints.insert(
            bodies["beak_coupler_link"],
            bodies["beak_hinge"],
            GenericJointBuilder::new(JointAxesMask::LIN_X | JointAxesMask::LIN_Y)
                .local_frame1(frame1)
                .local_frame2(frame2)
                .contacts_enabled(false),
            true,
        );
        let jaw_joint = physical_joints[0];
        let ranges = std::array::from_fn(|i| physical_joints[i].range.map(|v| v as f32));
        let physical_ledger = json!({"bodies":ledger,"joints":physical_joints,
            "native_dof_armature":multibody.armature().as_slice(),
            "native_dof_damping":multibody.damping().as_slice(),
            "numerical_mass_augmentation":false});
        Ok(Self {
            simulation,
            bodies,
            joints,
            pin,
            axis,
            ranges,
            physical_ledger,
            pin_local_jaw: frame2.translation,
            pin_local_coupler: frame1.translation,
            grip_local_jaw: poses["beak_hinge"].inverse() * source_vector(grip),
            jaw_anchor_local: poses["beak_hinge"].inverse()
                * source_vector(jaw_joint.origin_world_m),
        })
    }

    fn state(&self) -> Value {
        let world = &self.simulation.world;
        let mut angles = serde_json::Map::new();
        let mut velocities = serde_json::Map::new();
        let mut q = [0.0_f32; 3];
        for (i, handle) in self.joints.iter().enumerate() {
            let (multibody, link_id) = world.multibody_joints.get(*handle).unwrap();
            let link = multibody.link(link_id).unwrap();
            q[i] = link.joint.coords()[3];
            angles.insert(JOINT_NAMES[i].into(), json!(q[i]));
            velocities.insert(
                JOINT_NAMES[i].into(),
                json!(multibody.joint_velocity(link)[0]),
            );
        }
        let jaw_pin = world.bodies[self.bodies["beak_hinge"]].position() * self.pin_local_jaw;
        let coupler_pin =
            world.bodies[self.bodies["beak_coupler_link"]].position() * self.pin_local_coupler;
        let violation = q
            .iter()
            .zip(self.ranges)
            .map(|(q, [min, max])| (min - q).max(q - max).max(0.0))
            .fold(0.0_f32, f32::max);
        let snapshot = self.simulation.snapshot();
        json!({"time_s":snapshot.global_seconds,"integration_count":snapshot.integration_count,
            "torque_update_count":snapshot.torque_update_count,"angles_rad":angles,
            "velocities_rad_s":velocities,"rotor_jaw_error_rad":q[1]-q[0],
            "coupler_jaw_error_rad":q[2]+q[0],"output_pin_distance_m":(jaw_pin-coupler_pin).length(),
            "jaw_output_pin_world_m":to_source(jaw_pin),"coupler_output_pin_world_m":to_source(coupler_pin),
            "limit_violation_rad":violation})
    }

    fn case(
        &mut self,
        torque_nm: f32,
        load_n: f32,
        load_site: &str,
        ticks: usize,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let initial_state = self.state();
        let mut rows = Vec::new();
        let mut failure = None;
        for tick in 0..ticks {
            let world = &mut self.simulation.world;
            for (_, body) in world.bodies.iter_mut() {
                body.reset_forces(true);
            }
            let jaw_handle = self.bodies["beak_hinge"];
            let jaw = &world.bodies[jaw_handle];
            let point = jaw.position()
                * if load_site == "jaw_output_pin" {
                    self.pin_local_jaw
                } else {
                    self.grip_local_jaw
                };
            let anchor = jaw.position() * self.jaw_anchor_local;
            let tangent = self.axis.cross(point - anchor).normalize();
            let force = -tangent * load_n;
            let moment = (point - jaw.center_of_mass()).cross(force);
            let generalized_load = self.axis.dot((point - anchor).cross(force));
            let jaw_pin_pre = jaw.position() * self.pin_local_jaw;
            let pin_frame = world.bodies[self.bodies["beak_coupler_link"]].position()
                * world
                    .impulse_joints
                    .get(self.pin)
                    .unwrap()
                    .data
                    .local_frame1;
            world.bodies[jaw_handle].add_force(force, true);
            let mut torques = BodyTorque::joint_pair(
                self.bodies["head_roll"],
                self.bodies["beak_input_rotor"],
                (self.axis * torque_nm).to_array(),
            )
            .to_vec();
            // The public step clears torque queues; retain the point-force lever here.
            torques.push(BodyTorque {
                body: jaw_handle,
                world_torque: moment.to_array(),
            });
            let started = Instant::now();
            let before = self.simulation.snapshot();
            let result = self.simulation.step_with_torques(&torques);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if let Err(error) = result {
                failure = Some(error.to_string());
                break;
            }
            let after = self.simulation.snapshot();
            if after.integration_count != before.integration_count + 1
                || after.torque_update_count != before.torque_update_count + 1
                || (after.global_seconds - before.global_seconds - 0.02).abs() > 1e-12
            {
                failure = Some("not_one_20ms_integration".into());
                break;
            }
            let impulse = &self
                .simulation
                .world
                .impulse_joints
                .get(self.pin)
                .unwrap()
                .impulses;
            let local_impulse = Vector::new(impulse[0], impulse[1], impulse[2]);
            // Native solver sign: +J1^T*p on coupler, -J2^T*p on jaw.
            let force_on_jaw =
                -(pin_frame.rotation * local_impulse) / self.simulation.configuration().dt;
            let mut row = self.state();
            if row["angles_rad"]
                .as_object()
                .unwrap()
                .values()
                .chain(row["velocities_rad_s"].as_object().unwrap().values())
                .any(|value| value.as_f64().is_none())
                || !force_on_jaw.is_finite()
            {
                failure = Some("nonfinite_native_joint_or_pin_impulse".into());
                rows.push(row);
                break;
            }
            row["tick"] = json!(tick);
            row["physics_wall_ms"] = json!(elapsed_ms);
            row["input_torque_nm"] = json!(torque_nm);
            row["load_force_world_n"] = json!(to_source(force));
            row["external_jaw_generalized_torque_nm"] = json!(generalized_load);
            row["load_point_preintegration_world_m"] = json!(to_source(point));
            row["native_pin_impulses_n_s"] = json!(impulse.as_slice());
            row["final_pin_impulse_divided_by_dt_on_jaw_world_n"] = json!(to_source(force_on_jaw));
            row["final_pin_impulse_divided_by_dt_jaw_generalized_torque_nm"] =
                json!(self.axis.dot((jaw_pin_pre - anchor).cross(force_on_jaw)));
            row["actual_external_loads"] = json!(
                self.bodies
                    .iter()
                    .map(|(name, handle)| {
                        let body = &self.simulation.world.bodies[*handle];
                        (
                            name.clone(),
                            json!({"force_source_world_n":to_source(body.user_force()),
                    "torque_source_world_nm":to_source(body.user_torque())}),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>()
            );
            rows.push(row);
        }
        let maximum = |name: &str| {
            rows.iter()
                .filter_map(|r| r[name].as_f64().map(f64::abs))
                .fold(0.0_f64, f64::max)
        };
        let angle_error = maximum("rotor_jaw_error_rad").max(maximum("coupler_jaw_error_rad"));
        let pin_error = maximum("output_pin_distance_m");
        let limit_error = maximum("limit_violation_rad");
        let pass = failure.is_none()
            && rows.len() == ticks
            && angle_error <= ANGLE_TOLERANCE_RAD as f64
            && pin_error <= PIN_TOLERANCE_M as f64
            && limit_error <= LIMIT_TOLERANCE_RAD as f64;
        Ok(
            json!({"initial_state":initial_state,"input_torque_nm":torque_nm,"load_n":load_n,"load_site":load_site,
            "integrations":rows.len(),"failure_reason":failure,"max_angle_relation_error_rad":angle_error,
            "max_output_pin_distance_m":pin_error,"max_limit_violation_rad":limit_error,
            "local_checks_passed":pass,"rows":rows}),
        )
    }
}

fn source_vector(value: [f64; 3]) -> Vector {
    Vector::from_array(source_to_engine_vector(value.map(|v| v as f32)))
}
fn to_source(value: Vector) -> [f32; 3] {
    engine_to_source_vector(value.to_array())
}
fn source_pose(position: [f64; 3], rotation: [f64; 4]) -> Result<Pose, Box<dyn std::error::Error>> {
    let q = source_to_engine_rotation(rotation.map(|v| v as f32))?;
    Ok(Pose::from_parts(
        source_vector(position),
        Rotation::from_xyzw(q[0], q[1], q[2], q[3]),
    ))
}
fn hash_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    let option = |name: &str| {
        arguments
            .iter()
            .position(|v| v == name)
            .and_then(|i| arguments.get(i + 1))
    };
    let plant_path =
        PathBuf::from(option("--plant").ok_or("--plant is required")?).canonicalize()?;
    let contract_path =
        PathBuf::from(option("--contract").ok_or("--contract is required")?).canonicalize()?;
    let output = PathBuf::from(option("--output").ok_or("--output is required")?);
    if output.exists() {
        return Err("receipt exists; use a fresh experiment path".into());
    }
    let pgs: usize = option("--pgs").map(String::as_str).unwrap_or("4").parse()?;
    let ticks: usize = option("--ticks")
        .map(String::as_str)
        .unwrap_or("100")
        .parse()?;
    let pin_basis = option("--pin-basis")
        .map(String::as_str)
        .unwrap_or("source_axes");
    if !["source_axes", "coupler_axis"].contains(&pin_basis) {
        return Err("pin basis must be source_axes or coupler_axis".into());
    }
    if ![4, 8, 16, 32].contains(&pgs) || !(1..=100).contains(&ticks) {
        return Err("bounded probe requires PGS4/8/16/32, ticks1..100".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&plant_path)?;
    if hash_file(&contract_path)? != plant.derived_contract_sha256 {
        return Err("source contract identity differs from plant".into());
    }
    let contract: Value = serde_json::from_slice(&fs::read(&contract_path)?)?;
    let grip: [f64; 3] =
        serde_json::from_value(contract["sites"]["grip"]["world_at_zero_m"].clone())?;
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let mut code = serde_json::Map::new();
    for name in [
        "crates/dev_tools/src/bin/goose_jaw_probe.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/modules/robot/src/basis.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "Cargo.toml",
        "Cargo.lock",
        "third_party/rapier3d/Cargo.toml",
        "third_party/rapier3d/src/dynamics/integration_parameters.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/unit_multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/solver/joint_constraint/generic_joint_constraint.rs",
        "third_party/rapier3d/src/dynamics/solver/joint_constraint/generic_joint_constraint_builder.rs",
    ] {
        code.insert(name.into(), json!(hash_file(&project.join(name))?));
    }
    let mut cases = Vec::new();
    let mut ledger = Value::Null;
    let mut configuration = Value::Null;
    let mut schedule = vec![
        (0.0, 0.24, 0.0, "jaw_grip_load_point"),
        (0.275, 0.24, 0.0, "jaw_grip_load_point"),
    ];
    for site in ["jaw_output_pin", "jaw_grip_load_point"] {
        for load in [5.0, 10.0, 20.0] {
            schedule.push((0.275, 0.24, load, site));
        }
    }
    schedule.extend([
        (0.275, 4.4, 0.0, "jaw_grip_load_point"),
        (0.275, -4.4, 0.0, "jaw_grip_load_point"),
    ]);
    for (initial, torque, load, site) in schedule {
        let mut fixture = JawFixture::build(&plant, grip, initial, pgs, pin_basis)?;
        ledger = fixture.physical_ledger.clone();
        configuration = serde_json::to_value(fixture.simulation.configuration())?;
        let row = fixture.case(torque, load, site, ticks)?;
        println!(
            "EVENT initial={initial} torque={torque} load={load} site={site} local_pass={}",
            row["local_checks_passed"]
        );
        cases.push(row);
    }
    let report = json!({"schema":"goose_jaw_native_rapier_comparison_v1","engine":"rapier","engine_version":"0.35.3",
        "candidate":plant.candidate_id,"source_model_sha256":plant.model_sha256,"source_contract_sha256":plant.derived_contract_sha256,
        "plant_sha256":plant_hash,"runtime_code_sha256":code,"executable_sha256":hash_file(&env::current_exe()?)?,
        "debug_assertions":cfg!(debug_assertions),"physical_ledger":ledger,"configuration":configuration,
        "physics_dt_s":0.02,"integrations_per_tick":1,"substeps":0,"fixed_head":true,"gravity_m_s2":[0,0,0],
        "contacts_enabled":false,"qpos_writes_after_initialization":0,"inference_count":0,"optimizer_updates":0,
        "fresh_world_per_case":true,"native_tree_hinges":3,"native_planar_pin_rows":2,"native_colliders":0,
        "pin_basis":pin_basis,
        "reaction_time_phase":"final native impulse after stabilization divided by live dt; world basis and lever from preintegration poses; reported pose postintegration; position-advance impulse not separately captured",
        "load_definition":"resistance opposite positive opening at actual output pin or contract grip mapped into moving jaw frame",
        "frozen_local_thresholds":{"angle_relation_rad":ANGLE_TOLERANCE_RAD,"output_pin_distance_m":PIN_TOLERANCE_M,"limit_violation_rad":LIMIT_TOLERANCE_RAD},
        "local_checks_passed":cases.iter().all(|c| c["local_checks_passed"]==true),"cases":cases,
        "qualified":false,"limits":["fixed head; no body, ground or grasp contacts","no whole-Goose M0 or GPU qualification","native internal hinge-limit impulses lack public writeback","final pin impulse is not a complete audit of the integration-stage reaction"]});
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!(
        "STATUS JAW_DIAGNOSTIC_ONLY {} local_pass={}",
        output.display(),
        report["local_checks_passed"]
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Goose jaw probe: {error}");
            ExitCode::FAILURE
        }
    }
}
