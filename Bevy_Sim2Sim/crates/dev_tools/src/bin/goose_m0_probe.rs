//! Bounded Goose Rapier evidence. No policy inference or M0 promotion is implied.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use rapier3d::prelude::*;
use robot_minigame::goose::plant::GoosePlant;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{PhysicsClockProfile, SimulationWorld, goose::builder::GooseAssembly};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Goose M0 probe: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    let option = |name: &str| {
        arguments
            .iter()
            .position(|value| value == name)
            .and_then(|index| arguments.get(index + 1))
    };
    let plant_path = PathBuf::from(option("--plant").ok_or("--plant is required")?);
    let output = PathBuf::from(option("--output").ok_or("--output is required")?);
    if output.exists() {
        return Err("output receipt already exists; use a fresh experiment path".into());
    }
    let ticks: usize = option("--ticks")
        .map(String::as_str)
        .unwrap_or("100")
        .parse()?;
    let resets: usize = option("--cold-resets")
        .map(String::as_str)
        .unwrap_or("20")
        .parse()?;
    let pgs: usize = option("--pgs").map(String::as_str).unwrap_or("4").parse()?;
    let scene = option("--scene").map(String::as_str).unwrap_or("floor");
    if !["floor", "free_fall", "zero_gravity"].contains(&scene) {
        return Err("diagnostic scene must be floor, free_fall or zero_gravity".into());
    }
    let stop_scope = option("--stop-scope").map(String::as_str).unwrap_or("all");
    if !["all", "jaw_only"].contains(&stop_scope) {
        return Err("diagnostic stop scope must be all or jaw_only".into());
    }
    let recycling = option("--contact-recycling")
        .map(String::as_str)
        .unwrap_or("default");
    if !["default", "off"].contains(&recycling) {
        return Err("diagnostic contact recycling must be default or off".into());
    }
    let prediction = option("--prediction-mode")
        .map(String::as_str)
        .unwrap_or("default");
    if !["default", "two_mm"].contains(&prediction) {
        return Err("diagnostic prediction mode must be default or two_mm".into());
    }
    if ticks == 0 || ticks > 100 || resets == 0 || resets > 20 || ![4, 8, 16, 32].contains(&pgs) {
        return Err("bounded M0 requires ticks1..100, cold-resets1..20, PGS4/8/16/32".into());
    }
    let (plant, plant_sha256) = GoosePlant::read(&plant_path)?;
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let code_paths = [
        "crates/modules/robot/src/goose/plant.rs",
        "crates/modules/robot/src/goose/contract.rs",
        "crates/modules/simulation/src/goose/builder.rs",
        "crates/modules/simulation/src/lib.rs",
        "crates/modules/simulation/src/fixed_step_runtime.rs",
        "crates/dev_tools/src/bin/goose_m0_probe.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/unit_multibody_joint.rs",
        "third_party/rapier3d/src/dynamics/joint/multibody_joint/sim2sim_observation.rs",
        "third_party/rapier3d/src/dynamics/solver/joint_constraint/generic_joint_constraint.rs",
        "third_party/rapier3d/src/dynamics/solver/joint_constraint/generic_joint_constraint_builder.rs",
        "third_party/rapier3d/src/dynamics/solver/staged_island_solver/solve.rs",
        "third_party/rapier3d/src/dynamics/solver/staged_island_solver/worker.rs",
        "Cargo.lock",
    ];
    let code_hashes: serde_json::Map<String, Value> = code_paths
        .into_iter()
        .map(|path| {
            fs::read(project.join(path)).map(|bytes| {
                (
                    path.to_owned(),
                    json!(format!("{:x}", Sha256::digest(bytes))),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let executable = env::current_exe()?;
    let executable_sha256 = format!("{:x}", Sha256::digest(fs::read(&executable)?));
    let mut receipt = json!({"schema":"goose_m0_target_receipt_v1","engine":"rapier","engine_version":"0.35.3",
    "candidate":plant.candidate_id,"contract_sha256":plant.derived_contract_sha256,"model_sha256":plant.model_sha256,
    "neutral_contract_sha256":plant.neutral_contract_sha256,"plant_sha256":plant_sha256,
    "timing":{"physics_dt_s":0.02,"torque_dt_s":0.02,"policy_dt_s":0.02,"substeps":1},
    "runtime_code_sha256":code_hashes,"qualified":false,"target_complete":false,"optimizer_updates":0,"inference_count":0,
    "executable":executable,"executable_sha256":executable_sha256,
    "debug_assertions":cfg!(debug_assertions),"target_arch":std::env::consts::ARCH,
    "actuation_mode":"native prescribed effort: beak input rotor0.24Nm, other motors0; no learned actor",
    "diagnostic_scene":scene,"floor_enabled":scene=="floor","gravity_enabled":scene!="zero_gravity",
    "diagnostic_predictive_stop_scope":stop_scope,"numerical_contract_override":stop_scope!="all",
    "cold_reset_initial_root_lift_m":0.002,
    "checks":{
        "mass_com_full_inertia":{"status":"not_checked"},"cold_resets_single_integrations":{"status":"not_checked"},
        "named_axes":{"status":"not_checked"},"jaw_constraints":{"status":"not_checked"},
        "pad_load_curve":{"status":"not_checked"},"energy_growth":{"status":"not_checked"},
        "penetration_escape":{"status":"not_checked"},"foot_release_dissipation":{"status":"not_checked"},
        "foot_impact":{"status":"not_checked"},"collision_proxy":{"status":"not_checked"},
        "ground_reach_payload":{"status":"not_checked"},"source_target_pad_comparison":{"status":"not_checked"},
        "controller_gravity_feedforward":{"status":"not_checked"},
        "same_world_reset_cleanup":{"status":"not_checked"}
    }});
    receipt["contact_reporting"] = json!(
        "ContactPair::solver_manifolds() matches native solver clustering; raw source manifolds reported separately"
    );
    receipt["diagnostic_contact_recycling"] = json!(recycling);
    receipt["diagnostic_prediction_mode"] = json!(prediction);
    receipt["numerical_contract_override"] =
        json!(stop_scope != "all" || recycling != "default" || prediction != "default");
    let mut trace = Vec::new();
    let mut times = Vec::new();
    let mut failures = Vec::new();
    for reset in 0..resets {
        let mut simulation = SimulationWorld::new_with_profile(PhysicsClockProfile::Goose50);
        simulation
            .world
            .integration_parameters
            .num_internal_pgs_iterations = pgs;
        if scene == "zero_gravity" {
            simulation.world.gravity = Vector::ZERO;
        }
        if recycling == "off" {
            simulation.world.integration_parameters.contact_recycling = false;
        }
        if prediction == "two_mm" {
            simulation
                .world
                .integration_parameters
                .normalized_prediction_distance = 0.002;
        }
        if reset == 0 {
            receipt["native_contact_parameters"] = json!({
                "prediction_distance_m":simulation.world.integration_parameters.prediction_distance(),
                "contact_recycling":simulation.world.integration_parameters.contact_recycling});
        }
        let assembly = match GooseAssembly::build(&mut simulation, &plant) {
            Ok(assembly) => assembly,
            Err(error) => {
                failures.push(error.to_string());
                break;
            }
        };
        if stop_scope == "jaw_only" {
            let root = simulation
                .world
                .multibody_joints
                .rigid_body_link(assembly.body_handles["torso"])
                .ok_or("Goose diagnostic root absent")?;
            let multibody = simulation
                .world
                .multibody_joints
                .get_multibody_mut(root.multibody)
                .ok_or("Goose diagnostic multibody absent")?;
            let links: Vec<_> = multibody
                .links()
                .filter(|link| {
                    !["beak_hinge", "beak_input_rotor", "beak_coupler_link"]
                        .iter()
                        .any(|name| assembly.body_handles[*name] == link.rigid_body_handle())
                })
                .map(|link| link.link_id())
                .collect();
            for link in links {
                if !multibody
                    .link_mut(link)
                    .unwrap()
                    .joint
                    .set_predictive_limits_enabled(false)
                {
                    return Err("Goose original nonjaw diagnostic limit rejected".into());
                }
            }
        }
        if reset == 0 {
            receipt["configuration"] = serde_json::to_value(simulation.configuration())?;
            receipt["body_measurements"] = serde_json::to_value(&assembly.body_measurements)?;
            receipt["native_robot_body_count"] = json!(assembly.body_handles.len());
            receipt["native_robot_collider_count"] = json!(assembly.collider_count);
            receipt["source_geometry_count"] = json!(assembly.source_geometry_count);
            receipt["native_collider_source_groups"] = json!(assembly.collider_source_groups);
            receipt["geometry_representation"] = json!(
                "unchanged convex hulls packed by rigid body, mask and friction; no convex filling of hollow structures"
            );
            receipt["native_passive_spring_count"] = json!(assembly.passive_spring_count);
            receipt["checks"]["mass_com_full_inertia"] = json!({"status":"passed",
                "scope":"native per-body mass, COM, reconstructed full tensor at initialization; no dynamic equivalence"});
        }
        if scene == "floor" {
            let floor = simulation.world.bodies.insert(
                RigidBodyBuilder::fixed()
                    .translation(Vector::new(0.0, -0.1, 0.0))
                    .additional_solver_iterations(0),
            );
            simulation.world.colliders.insert_with_parent(
                ColliderBuilder::cuboid(2.0, 0.1, 2.0)
                    .density(0.0)
                    .friction(0.65)
                    .collision_groups(InteractionGroups::new(
                        Group::GROUP_1,
                        Group::GROUP_2,
                        InteractionTestMode::Or,
                    )),
                floor,
                &mut simulation.world.bodies,
            );
        }
        let reset_started = Instant::now();
        let mut completed = 0;
        let mut reason = None;
        let mut maximum_pin_error = 0.0_f32;
        for tick in 0..ticks {
            let started = Instant::now();
            let collider_poses_before: std::collections::HashMap<_, _> = simulation
                .world
                .colliders
                .iter()
                .map(|(handle, collider)| (handle, *collider.position()))
                .collect();
            let mut torque = [0.0; 18];
            torque[5] = if tick < 25 { 0.24 } else { 0.0 };
            let snapshot = match assembly.step(&mut simulation, torque) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    reason = Some(error.to_string());
                    break;
                }
            };
            let physics_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            times.push(physics_wall_ms);
            completed += 1;
            let state = assembly.state(&simulation)?;
            let pin_error = assembly.jaw_pin_error_m(&simulation);
            maximum_pin_error = maximum_pin_error.max(pin_error);
            // Native manifolds were detected before this Tick's integration.
            // Name the actual owners and signed distances without rebuilding
            // collisions from the post-integration published poses.
            let contact_pairs: Vec<_> = simulation.live_contact_pairs()
                .filter(|pair| pair.has_any_active_contact())
                .map(|pair| {
                    let owner = |collider: ColliderHandle| {
                        let body = simulation.world.colliders[collider].parent();
                        assembly.body_handles.iter().find_map(|(name, handle)| {
                            (Some(*handle) == body).then_some(name.clone())
                        }).unwrap_or_else(|| "scene".into())
                    };
                    let solver_manifolds = pair.solver_manifolds();
                    let minimum_distance = solver_manifolds.iter().flat_map(|manifold| manifold.points.iter())
                        .map(|point| point.dist).reduce(f32::min);
                    let source_minimum_distance = pair.manifolds.iter().flat_map(|manifold| manifold.points.iter())
                        .map(|point| point.dist).reduce(f32::min);
                    let source_geometry = |collider: ColliderHandle, subshape: u32| {
                        assembly.collider_source_groups.get(collider.into_raw_parts().0 as usize)
                            .and_then(|group|group.get(subshape as usize))
                    };
                    let closest_source = pair.manifolds.iter().flat_map(|manifold| {
                        manifold.points.iter().map(move |point|(manifold,point))
                    }).min_by(|a,b|a.1.dist.total_cmp(&b.1.dist)).map(|(manifold,point)| {
                        let leaf_query = simulation.world.colliders[pair.collider1].shape().as_compound()
                            .zip(simulation.world.colliders[pair.collider2].shape().as_compound())
                            .and_then(|(a,b)|a.shapes().get(manifold.subshape1 as usize)
                                .zip(b.shapes().get(manifold.subshape2 as usize)))
                            .map(|((local1,shape1),(local2,shape2))| {
                                let pose1 = collider_poses_before[&pair.collider1] * *local1;
                                let pose2 = collider_poses_before[&pair.collider2] * *local2;
                                let distance = rapier3d::parry::query::distance(&pose1,&**shape1,&pose2,&**shape2);
                                let contact = rapier3d::parry::query::contact(&pose1,&**shape1,&pose2,&**shape2,0.02);
                                let intersection = rapier3d::parry::query::intersection_test(&pose1,&**shape1,&pose2,&**shape2);
                                let aabb1 = shape1.compute_aabb(&pose1);
                                let aabb2 = shape2.compute_aabb(&pose2);
                                let fresh_manifolds: Vec<_> = [0.02,0.002].into_iter().map(|margin| {
                                    let mut manifolds: Vec<rapier3d::geometry::ContactManifold> = Vec::new();
                                    let mut workspace = None;
                                    let supported = rapier3d::parry::query::PersistentQueryDispatcher::contact_manifolds(
                                        &rapier3d::parry::query::DefaultQueryDispatcher,
                                        &pose1.inv_mul(&pose2),&**shape1,&**shape2,margin,
                                        &mut manifolds,&mut workspace).is_ok();
                                    json!({"prediction_m":margin,"supported":supported,
                                        "point_count":manifolds.iter().map(|m|m.points.len()).sum::<usize>(),
                                        "minimum_distance_m":manifolds.iter().flat_map(|m|m.points.iter()).map(|p|p.dist).reduce(f32::min)})
                                }).collect();
                                json!({"distance_m":distance.ok(),"contact_distance_m":contact.ok().flatten().map(|c|c.dist),
                                    "fresh_manifold_queries":fresh_manifolds,
                                    "intersection":intersection.ok(),"aabb1_min":aabb1.mins.to_array(),
                                    "aabb1_max":aabb1.maxs.to_array(),"aabb2_min":aabb2.mins.to_array(),
                                    "aabb2_max":aabb2.maxs.to_array(),"pose_time":"before this Tick integration"})
                            });
                        json!({
                        "geom1":source_geometry(pair.collider1,manifold.subshape1),
                        "geom2":source_geometry(pair.collider2,manifold.subshape2),
                        "subshape1":manifold.subshape1,"subshape2":manifold.subshape2,
                        "distance_m":point.dist,"standalone_native_leaf_query":leaf_query})
                    });
                    json!({"body1":owner(pair.collider1),"body2":owner(pair.collider2),
                        "minimum_distance_m":minimum_distance,"source_minimum_distance_m":source_minimum_distance,
                        "closest_source_geometry":closest_source,
                        "source_manifold_count":pair.manifolds.len(),"solver_manifold_count":solver_manifolds.len(),
                        "solver_contact_count":solver_manifolds.iter().map(|m| m.data.solver_contacts.len()).sum::<usize>()})
                }).collect();
            trace.push(json!({"reset":reset,"tick":tick,"snapshot":snapshot,"q_rad":state.joint_position_rad,
                "qd_rad_s":state.joint_velocity_rad_s,"jaw_pin_error_m":pin_error,
                "native_contact_pairs_pre_integration":contact_pairs,
                "physics_wall_ms":physics_wall_ms}));
            #[cfg(feature = "sim2sim_limit_row_trace")]
            {
                let root = simulation
                    .world
                    .multibody_joints
                    .rigid_body_link(assembly.body_handles["torso"])
                    .ok_or("native observation root absent")?;
                let multibody = simulation
                    .world
                    .multibody_joints
                    .get_multibody(root.multibody)
                    .ok_or("native observation owner absent")?;
                if let Some(observation) = multibody.sim2sim_observation() {
                    let matrix = multibody.sim2sim_constraint_mass_matrix();
                    let rows: Vec<_> = observation.joint_row_timing.iter().map(|row| json!({
                        "phase":format!("{:?}",row.phase),"substep_id":row.substep_id,"joint_local_dof":row.joint_local_dof,
                        "impulse":row.impulse,"jacobian":row.jacobian_side2,"weighted_jacobian":row.weighted_jacobian_side2,
                        "solver_velocity":row.solver_velocity_side2,"rhs":row.rhs,"cfm_gain":row.cfm_gain,
                        "inverse_row_inertia":row.inverse_row_inertia})).collect();
                    let mapping: Vec<_> = multibody.links().map(|link| json!({"body":assembly.body_handles.iter()
                        .find_map(|(name,handle)|(*handle==link.rigid_body_handle()).then_some(name)),
                        "backend_dof_start":link.assembly_id(),"ndofs":link.joint.ndofs()})).collect();
                    let limit_rows: Vec<_> = observation
                        .limit_row_timing
                        .iter()
                        .map(|row| {
                            json!({
                        "phase":format!("{:?}",row.phase),"row_index":row.row_index,
                        "backend_dof":row.backend_dof,"jacobian_sign":row.jacobian_sign,
                        "coordinate":row.coordinate,
                        "solver_velocity":row.generalized_velocity,"rhs":row.rhs,
                        "rhs_without_bias":row.rhs_without_bias,"impulse":row.impulse,
                        "impulse_bounds":row.impulse_bounds})
                        })
                        .collect();
                    let updates: Vec<_> = observation.generic_joint_updates.iter().map(|row| json!({
                        "row_index":row.row_index,"without_bias":row.without_bias,
                        "writeback_kind":row.writeback_kind,"joint_index":row.joint_index,
                        "impulse_before":row.impulse_before,"impulse_after":row.impulse_after,
                        "impulse_bounds":row.impulse_bounds,"jacobian":row.jacobian,
                        "weighted_jacobian":row.weighted_jacobian,"velocity_before":row.velocity_before,
                        "velocity_after":row.velocity_after,"rhs":row.rhs,"cfm_gain":row.cfm_gain,
                        "inverse_row_inertia":row.inverse_row_inertia})).collect();
                    trace.last_mut().unwrap()["native_coupled_solve"] = json!({
                        "valid":true,"epoch":observation.epoch,"dt_s":observation.full_step_dt(),
                        "energy_guard_fallback":observation.energy_guard_fallback,
                        "energy_guard_acceleration_cleared":observation.energy_guard_acceleration_cleared,
                        "mass_matrix_phase":"native constraint matrix last computed before integration",
                        "mass_matrix":(0..matrix.nrows()).map(|i|matrix.row(i).iter().copied().collect::<Vec<_>>()).collect::<Vec<_>>(),
                        "backend_dof_mapping":mapping,"pin_rows":rows,"limit_rows":limit_rows,
                        "generic_joint_updates":updates});
                } else {
                    trace.last_mut().unwrap()["native_coupled_solve"] = json!({"valid":false});
                }
            }
            if !pin_error.is_finite()
                || state
                    .joint_position_rad
                    .iter()
                    .chain(&state.joint_velocity_rad_s)
                    .any(|value| !value.is_finite())
            {
                reason = Some("nonfinite_native_joint_state".into());
                break;
            }
            if reset_started.elapsed().as_secs_f64() >= 5.0 {
                reason = Some("bounded_probe_wall_budget_5s_per_reset".into());
                break;
            }
        }
        if let Some(reason) = &reason {
            failures.push(reason.clone());
        }
        receipt["resets"].as_array_mut().map(|rows| {
            rows.push(json!({"reset":reset,"ticks_completed":completed,
            "failure_reason":reason,"max_jaw_pin_error_m":maximum_pin_error}))
        });
        if !receipt["resets"].is_array() {
            receipt["resets"] = json!([{"reset":reset,"ticks_completed":completed,
            "failure_reason":reason,"max_jaw_pin_error_m":maximum_pin_error}]);
        }
    }
    times.sort_by(f64::total_cmp);
    let actual_resets = receipt["resets"].as_array().map_or(0, Vec::len);
    let finished = failures.is_empty() && actual_resets == resets && trace.len() == resets * ticks;
    receipt["checks"]["cold_resets_single_integrations"] = json!({"status":if finished && resets==20 && ticks==100 && scene=="floor" && stop_scope=="all" && recycling=="default" && prediction=="default" {"passed"} else if finished {"partial"} else {"failed"},
        "measurement_passed":finished,"scope":"fresh_world_reinitialization; same-world scene retention/handle cleanup not tested",
        "reset_count":actual_resets,"requested_resets":resets,"requested_ticks_per_reset":ticks,
        "completed_integrations":trace.len(),"failures":failures,
        "qualification_limits":"prescribed effort diagnostic; energy, penetration, actual controller, full M0 not qualified"});
    receipt["physics_only_p95_ms"] = if times.is_empty() {
        Value::Null
    } else {
        json!(times[(times.len() * 95 / 100).min(times.len() - 1)])
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let trace_path = output.with_file_name(format!(
        "{}_trace.json",
        output
            .file_stem()
            .ok_or("invalid output name")?
            .to_string_lossy()
    ));
    let trace_bytes = serde_json::to_vec(&json!({"schema":"goose_target_trace_v1","rows":trace}))?;
    receipt["trace_path"] = json!(trace_path);
    receipt["trace_sha256"] = json!(format!("{:x}", Sha256::digest(&trace_bytes)));
    fs::write(&trace_path, trace_bytes)?;
    fs::write(&output, serde_json::to_vec_pretty(&receipt)?)?;
    println!("STATUS M0_UNQUALIFIED {}", output.display());
    Ok(())
}
