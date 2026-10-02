//! Six-DoF condensed-foot force/moment diagnostic; no whole-body admission.

use std::{
    collections::HashMap, env, fs, path::Path, process::ExitCode, sync::Mutex, time::Instant,
};

use rapier3d::{
    geometry::ExperimentalNormalSpring, pipeline::ContactModificationContext, prelude::*,
};
use robot_minigame::{
    basis::engine_to_source_vector,
    goose::plant::{GooseCollider, GoosePlant},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{
    PhysicsClockProfile, SimulationWorld, goose::builder::native_mass_properties,
};

const DT: f64 = 0.02;
const PROGRAMS: [&str; 5] = [
    "flat",
    "pitch_plus",
    "pitch_minus",
    "roll_plus",
    "roll_minus",
];

fn vector([x, y, z]: [f64; 3]) -> Vector {
    Vector::new(x as f32, z as f32, -y as f32)
}
fn source(v: Vector) -> [f32; 3] {
    engine_to_source_vector(v.to_array())
}

struct Materials {
    ground: ColliderHandle,
    foot: RigidBodyHandle,
    soft: HashMap<ColliderHandle, (f32, f32)>,
    points: Mutex<HashMap<(ColliderHandle, usize), Vector>>,
}

impl PhysicsHooks for Materials {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        assert!(
            context.collider1 == self.ground || context.collider2 == self.ground,
            "isolated foot only admits ground contact"
        );
        assert!(
            context.normal.x.abs() < 1e-5 && context.normal.z.abs() < 1e-5,
            "only frictionless horizontal ground supported"
        );
        let patch = if context.collider1 == self.ground {
            context.collider2
        } else {
            context.collider1
        };
        let cosine = (context.bodies[self.foot].rotation() * Vector::Y)
            .dot(*context.normal)
            .abs();
        assert!(cosine >= 0.5, "foot tilt outside declared prototype");
        if let Some((k, c)) = self.soft.get(&patch) {
            if !context.solver_contacts.is_empty() {
                let count = context.solver_contacts.len() as f32;
                *context.experimental_normal_spring = Some(ExperimentalNormalSpring {
                    stiffness_n_m: k / (count * cosine * cosine),
                    damping_n_s_m: c / (count * cosine * cosine),
                });
            }
        }
        let mut points = self.points.lock().unwrap();
        for p in context.solver_contacts.iter() {
            // Hook anchors are fresh world points, before native local conversion.
            assert!(
                points
                    .insert(
                        (patch, p.contact_indices()[0] as usize),
                        (p.anchor1 + p.anchor2) * 0.5
                    )
                    .is_none(),
                "duplicate original point identity"
            );
        }
    }
}

fn corners(patch: &GooseCollider) -> Result<Vec<Vector>, Box<dyn std::error::Error>> {
    let [hx, hy, hz] = patch.half_extents_m.ok_or("patch box absent")?;
    let mut points = Vec::new();
    for x in [-hx, hx] {
        for y in [-hy, hy] {
            for z in [-hz, hz] {
                points.push(vector(patch.local_position_m) + vector([x, y, z]));
            }
        }
    }
    Ok(points)
}

fn parameter(patch: &Value, name: &str) -> Result<f64, Box<dyn std::error::Error>> {
    patch["contact_parameters"][name]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing physical parameter {name}").into())
}

fn case(
    plant: &GoosePlant,
    raw: &Value,
    program: &str,
    mode: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let foot = plant
        .bodies
        .iter()
        .find(|b| b.name == "right_ankle_roll")
        .ok_or("foot absent")?;
    let patches: Vec<_> = plant
        .colliders
        .iter()
        .filter(|p| {
            p.body == foot.name
                && p.contact_patch
                    .as_deref()
                    .is_some_and(|n| n.starts_with("right_sole_pad_"))
        })
        .collect();
    if patches.len() != 6 || foot.rotation_world_wxyz != [1.0, 0.0, 0.0, 0.0] {
        return Err("frozen right condensed foot required".into());
    }
    let angle = (3.0_f32).to_radians();
    let scaled_axis = match program {
        "pitch_plus" => -Vector::Z * angle,
        "pitch_minus" => Vector::Z * angle,
        "roll_plus" => Vector::X * angle,
        "roll_minus" => -Vector::X * angle,
        "flat" => Vector::ZERO,
        _ => return Err("undeclared program".into()),
    };
    let rotation = Rotation::from_scaled_axis(scaled_axis);
    let all_corners: Vec<_> = patches
        .iter()
        .map(|p| corners(p))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    let height = 0.0005
        - all_corners
            .iter()
            .map(|p| (rotation * *p).y)
            .fold(f32::INFINITY, f32::min);
    let pose = Pose::from_parts(Vector::Y * height, rotation);
    let load = plant.robot_mass_kg * 9.81 / 2.0;
    let band = load * DT * DT / foot.mass_kg + 0.02 + 0.0015;
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    let world = &mut simulation.world;
    world.gravity = Vector::ZERO;
    world.integration_parameters.num_internal_pgs_iterations = 32;
    world.integration_parameters.max_ccd_substeps = 0;
    world.integration_parameters.contact_recycling = false;
    world.integration_parameters.experimental_free_normal_block =
        ["free_block", "free_block_plain"].contains(&mode);
    world.integration_parameters.normalized_prediction_distance = band as f32;
    world.integration_parameters.normalized_allowed_linear_error = 0.0;
    world
        .integration_parameters
        .normalized_max_corrective_velocity = 100.0;
    world.integration_parameters.contact_softness = SpringCoefficients::new(1e6, 1.0);
    world.integration_parameters.static_contact_softness = SpringCoefficients::new(1e6, 1.0);
    let anchor = world.insert_body(RigidBodyBuilder::fixed());
    let body = world.insert_body(
        RigidBodyBuilder::dynamic()
            .pose(pose)
            .can_sleep(false)
            .additional_mass_properties(native_mass_properties(foot)?),
    );
    let joint = world
        .insert_multibody_joint(
            anchor,
            body,
            GenericJointBuilder::new(JointAxesMask::empty())
                .local_frame1(pose)
                .contacts_enabled(false),
        )
        .ok_or("free multibody joint absent")?;
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
    let materials = Materials {
        ground,
        foot: body,
        soft: HashMap::new(),
        points: Mutex::new(HashMap::new()),
    };
    let mut materials = materials;
    let mut bindings = Vec::new();
    let mut ledger = Vec::new();
    for patch in &patches {
        if patch.kind != "box" || patch.local_rotation_wxyz != [1.0, 0.0, 0.0, 0.0] {
            return Err("original box geometry changed".into());
        }
        let original = raw["colliders"]
            .as_array()
            .ok_or("colliders absent")?
            .iter()
            .find(|p| p["name"].as_str() == Some(&patch.name))
            .ok_or("original patch absent")?;
        let k = parameter(original, "stiffness_n_m")?;
        let c = parameter(original, "damping_n_s_m")?;
        let travel = parameter(original, "travel_m")?;
        if k <= 0.0 || c < 0.0 || (travel - 0.0015).abs() > 1e-12 {
            return Err("physical K/C/travel changed".into());
        }
        let [hx, hy, hz] = patch.half_extents_m.ok_or("box extents absent")?;
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
                materials.soft.insert(collider, (k as f32, c as f32));
            }
            bindings.push((collider, patch.name.clone(), stop));
        }
    }
    world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
    let (mb, _) = world.multibody_joints.get_mut(joint).unwrap();
    mb.forward_kinematics(&world.bodies, true);
    assert_eq!(mb.ndofs(), 6);
    mb.damping_mut().fill(0.0);
    mb.armature_mut().fill(0.0);
    mb.frictions_mut().fill(0.0);
    #[cfg(feature = "sim2sim_plain_mass_probe")]
    mb.sim2sim_set_plain_mass_probe(mode == "free_block_plain");
    #[cfg(not(feature = "sim2sim_plain_mass_probe"))]
    if mode == "free_block_plain" {
        return Err("plain mass mode requires explicit sim2sim_plain_mass_probe feature".into());
    }
    mb.update_rigid_bodies(&mut world.bodies, true);
    let mut frames = Vec::new();
    let mut elapsed = Vec::new();
    for tick in 0..100 {
        let com = world.bodies[body].center_of_mass();
        let origin = world.bodies[body].translation();
        let applied = -Vector::Y * load as f32;
        let torque = (origin - com).cross(applied);
        world.bodies[body].reset_forces(true);
        world.bodies[body].reset_torques(true);
        world.bodies[body].add_force_at_point(applied, origin, true);
        materials.points.lock().unwrap().clear();
        let start = Instant::now();
        world.step_with_events(&materials, &());
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        let (mb, _) = world.multibody_joints.get(joint).unwrap();
        let observation = mb
            .sim2sim_observation()
            .ok_or("invalid completed native observation")?;
        if observation.full_step_dt().to_bits() != (DT as f32).to_bits()
            || world.integration_parameters.num_solver_iterations != 1
            || !world.quarantine().is_empty()
        {
            return Err("timing or native quarantine failure; no reset".into());
        }
        let points = materials.points.lock().unwrap();
        let mut force = Vector::ZERO;
        let mut moment = Vector::ZERO;
        let mut patch_force = serde_json::Map::new();
        let mut rows = Vec::new();
        for (collider, patch, stop) in &bindings {
            let mut total = 0.0;
            if let Some(pair) = world.narrow_phase.contact_pair(*collider, ground) {
                for manifold in pair.solver_manifolds() {
                    let normal = if pair.collider1 == *collider {
                        -manifold.data.normal
                    } else {
                        manifold.data.normal
                    };
                    for p in &manifold.data.solver_contacts {
                        let id = p.contact_indices()[0] as usize;
                        let magnitude = manifold.points[id].data.impulse as f64 / DT;
                        let point = *points
                            .get(&(*collider, id))
                            .ok_or("original hook point identity absent")?;
                        let f = normal * magnitude as f32;
                        force += f;
                        moment += (point - com).cross(f);
                        total += magnitude;
                        rows.push(json!({"patch":patch,"travel_stop":stop,"normal_force_n":magnitude,
                            "point_world_m":source(point),"normal_world":source(normal),"gap_m":manifold.points[id].dist,
                            "point_stiffness_n_m":manifold.data.experimental_normal_spring.map(|s| s.stiffness_n_m),
                            "point_damping_n_s_m":manifold.data.experimental_normal_spring.map(|s| s.damping_n_s_m)}));
                    }
                }
            }
            let entry = patch_force.entry(patch.clone()).or_insert(json!(0.0));
            *entry = json!(entry.as_f64().unwrap() + total);
        }
        let pose = world.bodies[body].position();
        let mut qpos = source(pose.translation).to_vec();
        // Raw diagnostic transform preserves any native norm drift as evidence.
        // No Actor interface accepts it and no state is normalized here.
        let [x, y, z, w] = pose.rotation.to_array();
        let quaternion_norm_error = (x * x + y * y + z * z + w * w - 1.0).abs();
        qpos.extend([w, x, -z, y]);
        let mut gaps = serde_json::Map::new();
        for patch in &patches {
            let gap = corners(patch)?
                .iter()
                .map(|p| (*pose * *p).y)
                .fold(f32::INFINITY, f32::min);
            gaps.insert(patch.name.clone(), json!(gap));
        }
        if !mb.generalized_velocity().iter().all(|v| v.is_finite())
            || !force.is_finite()
            || !moment.is_finite()
        {
            return Err("nonfinite free foot".into());
        }
        frames.push(json!({"tick":tick,"time_s":(tick+1) as f64*DT,"integration_count":tick+1,
            "native_epoch":observation.epoch,"native_full_step_dt_s":observation.full_step_dt(),
            "qpos_wxyz":qpos,"generalized_velocity_native":mb.generalized_velocity().as_slice(),
            "quaternion_norm_error":quaternion_norm_error,
            "com_before_m":source(com),"applied_force_world_n":source(applied),"applied_moment_world_nm":source(torque),
            "contact_force_world_n":source(force),"contact_moment_world_nm":source(moment),
            "patch_force_n":patch_force,"patch_min_gap_m":gaps,"rows":rows}));
    }
    elapsed.sort_by(f64::total_cmp);
    Ok(
        json!({"program":program,"integrations":100,"foot_physical_ledger":foot,"patches":ledger,
        "native_mass_kg":world.bodies[body].mass(),"search_band_m":band,"physics_p95_ms":elapsed[94],"frames":frames}),
    )
}

fn hash(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

/// Contact-free two-Tick comparison starts from an independently frozen source
/// state. A centered COM is an explicit diagnostic control, never a plant edit.
fn inertia_case(
    plant: &GoosePlant,
    input: &Value,
    mode: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let original = plant
        .bodies
        .iter()
        .find(|b| b.name == "right_ankle_roll")
        .ok_or("foot absent")?;
    let mut foot = original.clone();
    let centered = !input["com_local_override_m"].is_null();
    if centered {
        let override_com: [f64; 3] = serde_json::from_value(input["com_local_override_m"].clone())?;
        if override_com != [0.0; 3] {
            return Err("only declared centered-COM control supported".into());
        }
        foot.com_local_m = override_com;
    }
    let q: [f64; 7] = serde_json::from_value(input["initial_qpos_wxyz"].clone())?;
    let v: [f64; 6] = serde_json::from_value(input["initial_qvel_source"].clone())?;
    let rotation = Rotation::from_xyzw(q[4] as f32, q[6] as f32, -q[5] as f32, q[3] as f32);
    let pose = Pose::from_parts(vector([q[0], q[1], q[2]]), rotation);
    let linear = rotation.inverse() * vector([v[0], v[1], v[2]]);
    // Source angular free-joint velocity is body-local; this joint's initial
    // frame equals that body frame. Both local vectors use the proper B basis.
    let angular = vector([v[3], v[4], v[5]]);
    let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
    let world = &mut simulation.world;
    world.gravity = Vector::ZERO;
    world.integration_parameters.max_ccd_substeps = 0;
    let anchor = world.insert_body(RigidBodyBuilder::fixed());
    let body = world.insert_body(
        RigidBodyBuilder::dynamic()
            .pose(pose)
            .can_sleep(false)
            .additional_mass_properties(native_mass_properties(&foot)?),
    );
    let joint = world
        .insert_multibody_joint(
            anchor,
            body,
            GenericJointBuilder::new(JointAxesMask::empty())
                .local_frame1(pose)
                .contacts_enabled(false),
        )
        .ok_or("free multibody absent")?;
    let (mb, _) = world.multibody_joints.get_mut(joint).unwrap();
    mb.forward_kinematics(&world.bodies, true);
    mb.damping_mut().fill(0.0);
    mb.armature_mut().fill(0.0);
    mb.frictions_mut().fill(0.0);
    for (index, value) in linear
        .to_array()
        .into_iter()
        .chain(angular.to_array())
        .enumerate()
    {
        mb.generalized_velocity_mut()[index] = value;
    }
    #[cfg(feature = "sim2sim_plain_mass_probe")]
    mb.sim2sim_set_plain_mass_probe(mode == "inertia_plain");
    #[cfg(not(feature = "sim2sim_plain_mass_probe"))]
    if mode == "inertia_plain" {
        return Err("plain mass control requires explicit sim2sim_plain_mass_probe feature".into());
    }
    mb.update_rigid_bodies(&mut world.bodies, true);
    let mut frames = Vec::new();
    for tick in 0..2 {
        let before_pose = *world.bodies[body].position();
        let before_velocity = world
            .multibody_joints
            .get(joint)
            .unwrap()
            .0
            .generalized_velocity()
            .as_slice()
            .to_vec();
        world.step_with_events(&(), &());
        if world.narrow_phase.contact_pairs().next().is_some() || !world.quarantine().is_empty() {
            return Err("contact-free diagnostic had contact or quarantine".into());
        }
        let (mb, id) = world.multibody_joints.get(joint).unwrap();
        let observation = mb
            .sim2sim_observation()
            .ok_or("completed native observation absent")?;
        if observation.full_step_dt().to_bits() != (DT as f32).to_bits() {
            return Err("native integration timing mismatch".into());
        }
        let twist = mb.body_jacobian(id) * mb.generalized_velocity();
        let velocity = Vector::new(twist[0], twist[1], twist[2]);
        let omega = Vector::new(twist[3], twist[4], twist[5]);
        let pose = world.bodies[body].position();
        let [x, y, z, w] = pose.rotation.to_array();
        let mut qpos = source(pose.translation).to_vec();
        qpos.extend([w, x, -z, y]);
        #[cfg(feature = "sim2sim_limit_row_trace")]
        let mass = {
            let m = mb.sim2sim_constraint_mass_matrix();
            (0..m.nrows())
                .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        #[cfg(not(feature = "sim2sim_limit_row_trace"))]
        let mass: Vec<Vec<f32>> = Vec::new();
        #[cfg(feature = "sim2sim_limit_row_trace")]
        let acceleration_mass = {
            let m = mb.sim2sim_acceleration_mass_matrix();
            (0..m.nrows())
                .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        #[cfg(not(feature = "sim2sim_limit_row_trace"))]
        let acceleration_mass: Vec<Vec<f32>> = Vec::new();
        frames.push(json!({"tick":tick,"time_s":(tick+1) as f64*DT,"integration_count":tick+1,
            "native_epoch":observation.epoch,"native_full_step_dt_s":observation.full_step_dt(),"contacts":0,
            "generalized_velocity_before":before_velocity,"generalized_velocity_after":mb.generalized_velocity().as_slice(),
            "before_pose_engine":{ "translation":before_pose.translation.to_array(),"rotation_xyzw":before_pose.rotation.to_array() },
            "constraint_mass_matrix_before":mass,"acceleration_mass_matrix_before":acceleration_mass,
            "inertial_projection":observation.inertial_projection,
            "energy_guard_evaluated":observation.energy_guard_evaluated,"energy_guard_fallback":observation.energy_guard_fallback,
            "energy_guard_acceleration_cleared":observation.energy_guard_acceleration_cleared,
            "after":{"qpos_wxyz":qpos,"com_world_m":source(world.bodies[body].center_of_mass()),
                "velocity_com_world_m_s":source(velocity),"angular_velocity_world_rad_s":source(omega),
                "linear_momentum_world_ns":source(velocity*world.bodies[body].mass()),
                "quaternion_norm_error":(x*x+y*y+z*z+w*w-1.0).abs()},
            "pre_integration_cached_body_velocity":source(world.bodies[body].linvel())}));
    }
    Ok(
        json!({"program":input["program"],"foot_physical_ledger":original,
        "com_local_override_m":input["com_local_override_m"],"native_mass_kg":world.bodies[body].mass(),
        "initial_qpos_wxyz":q,"initial_qvel_source":v,"integrations":2,"frames":frames}),
    )
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err(
            "usage: goose_free_sole_probe FROZEN_CONDENSED_PLANT NEW_OUTPUT_DIR native|free_block|free_block_plain|inertia_native|inertia_plain [FROZEN_SOURCE_INERTIA_RECEIPT]"
                .into(),
        );
    }
    let mode = args[2].to_str().ok_or("invalid mode")?;
    if ![
        "native",
        "free_block",
        "free_block_plain",
        "inertia_native",
        "inertia_plain",
    ]
    .contains(&mode)
    {
        return Err("undeclared normal mode".into());
    }
    let plant_path = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    if output.exists()
        || output
            .parent()
            .ok_or("parent missing")?
            .canonicalize()?
            .starts_with(&project)
    {
        return Err("preserve evidence outside project in a new directory".into());
    }
    let (plant, plant_hash) = GoosePlant::read(&plant_path)?;
    if plant.candidate_id != "goose_460_condensed50_v1" {
        return Err("frozen condensed physical ledger required".into());
    }
    let raw: Value = serde_json::from_slice(&fs::read(&plant_path)?)?;
    let start = Instant::now();
    if mode.starts_with("inertia_") {
        let source_path =
            Path::new(args.get(3).ok_or("source inertia receipt required")?).canonicalize()?;
        let source: Value = serde_json::from_slice(&fs::read(&source_path)?)?;
        if source["schema"] != "goose_free_inertia_source_v1"
            || source["plant_sha256"] != plant_hash
        {
            return Err("source schema or frozen physical plant mismatch".into());
        }
        let inputs = source["cases"]
            .as_array()
            .ok_or("source inertia cases absent")?;
        if inputs.len() != 5 {
            return Err("exact frozen five-case program required".into());
        }
        let cases = inputs
            .iter()
            .map(|c| inertia_case(&plant, c, mode))
            .collect::<Result<Vec<_>, _>>()?;
        let mut hashes = serde_json::Map::new();
        for path in [
            "crates/dev_tools/src/bin/goose_free_sole_probe.rs",
            "crates/modules/simulation/src/goose/builder.rs",
            "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody.rs",
            "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
            "third_party/rapier3d/src/dynamics/solver/staged_island_solver/helpers.rs",
            "Cargo.lock",
        ] {
            hashes.insert(path.to_owned(), json!(hash(&project.join(path))?));
        }
        let executable = env::current_exe()?.canonicalize()?;
        fs::create_dir(output)?;
        fs::write(
            output.join("receipt.json"),
            serde_json::to_vec_pretty(&json!({
            "schema":"goose_free_inertia_target_v1","plant_path":plant_path,"plant_sha256":plant_hash,
            "physics_dt_s":DT,"integrations_per_tick":1,"substeps":0,"coordinate_writes_after_initialization":0,
            "qualified":false,"m0_passed":false,"optimizer_updates":0,"inference_count":0,"mode":mode,"cases":cases,
            "source_receipt":source_path,"source_receipt_sha256":hash(&source_path)?,"runtime_code_sha256":hashes,
            "executable":executable,"executable_sha256":hash(&executable)?,"elapsed_wall_s":start.elapsed().as_secs_f64(),
            "control_scope":"centered cases change only local COM for causal isolation; not a candidate physics modification"}))?,
        )?;
        println!(
            "{}",
            json!({"receipt":output.join("receipt.json"),"integrations":10,"status":"DIAGNOSTIC_ONLY"})
        );
        return Ok(());
    }
    if args.len() != 3 {
        return Err("unexpected source input for contact mode".into());
    }
    let cases = PROGRAMS
        .into_iter()
        .map(|p| case(&plant, &raw, p, mode))
        .collect::<Result<Vec<_>, _>>()?;
    let mut hashes = serde_json::Map::new();
    for path in [
        "crates/dev_tools/src/bin/goose_free_sole_probe.rs",
        "crates/dev_tools/Cargo.toml",
        "crates/modules/robot/src/basis.rs",
        "crates/modules/robot/src/goose/plant.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/simulation/src/lib.rs",
        "third_party/rapier3d/src/dynamics/integration_parameters.rs",
        "third_party/rapier3d/src/dynamics/solver/contact_constraint/generic_contact_constraint.rs",
        "third_party/rapier3d/src/dynamics/solver/staged_island_solver/worker.rs",
        "Cargo.lock",
    ] {
        hashes.insert(path.to_owned(), json!(hash(&project.join(path))?));
    }
    let executable = env::current_exe()?.canonicalize()?;
    fs::create_dir(output)?;
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&json!({"schema":"goose_free_condensed_target_v2",
        "revision":if mode == "free_block_plain" { "goose_free_condensed_be_contact_v3" } else { "goose_free_condensed_contact_v2" },
        "plant_path":plant_path,"plant_sha256":plant_hash,
        "physics_dt_s":DT,"integrations_per_tick":1,"substeps":0,"coordinate_writes_after_initialization":0,
        "qualified":false,"m0_passed":false,"optimizer_updates":0,"inference_count":0,
        "normal_mode":mode,"cases":cases,"runtime_code_sha256":hashes,
        "executable":executable,"executable_sha256":hash(&executable)?,"elapsed_wall_s":start.elapsed().as_secs_f64(),
        "limitations":["isolated frictionless six-DoF right foot; not whole body",
            "per-point foundation approximates shared sliding patches; not exact pad force/moment equivalence",
            "shifted backing boxes not CAD qualified; no high-frequency equivalence"]}))?,
    )?;
    println!(
        "{}",
        json!({"status":"DIAGNOSTIC_ONLY","receipt":output.join("receipt.json"),"integrations":500})
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
