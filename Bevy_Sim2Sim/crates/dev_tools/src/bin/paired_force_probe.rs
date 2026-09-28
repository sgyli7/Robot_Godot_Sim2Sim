//! One-step, robot-only Rapier force evidence for a separate MuJoCo comparison.
//!
//! This is deliberately outside the guarded source-collision world. It cannot
//! qualify BAM external load or source contacts; it exposes the raw terms and
//! source DOF identity needed to decide whether that bridge is even plausible.

use std::{env, fs, path::Path};

use rapier3d::{na::DVector, prelude::Multibody};
use robot_minigame::{basis::source_to_engine_vector, definition::RobotDefinition};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_checked(path: &str, expected_sha256: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if sha256(&bytes) != expected_sha256 {
        return Err(format!("SHA256 mismatch: {path}"));
    }
    Ok(bytes)
}

fn run(
    args: &[String],
    plain_mass_diagnostic: bool,
    head_roll_free_acceleration_diagnostic: bool,
    source_limit_diagnostic: bool,
) -> Result<Value, String> {
    if !Multibody::sim2sim_observation_backend_supported() {
        return Err("native Rapier observation backend is unavailable".into());
    }
    #[cfg(not(feature = "sim2sim_plain_mass_probe"))]
    if plain_mass_diagnostic {
        return Err("plain-mass diagnostic requires sim2sim_plain_mass_probe feature".into());
    }
    #[cfg(not(feature = "sim2sim_source_limit_probe"))]
    if source_limit_diagnostic {
        return Err("source-limit diagnostic requires sim2sim_source_limit_probe feature".into());
    }
    let definition_bytes = read_checked(&args[0], &args[1])?;
    let definition = match Path::new(&args[0]).extension().and_then(|v| v.to_str()) {
        Some("json") => RobotDefinition::load_json(Path::new(&args[0]), &args[1]),
        Some("ron") => RobotDefinition::load_ron(Path::new(&args[0]), &args[1]),
        _ => return Err("definition must be JSON or RON".into()),
    }
    .map_err(|error| error.to_string())?;
    let qpos_bytes = read_checked(&args[2], &args[3])?;
    let qpos: Vec<f64> = serde_json::from_slice(&qpos_bytes).map_err(|error| error.to_string())?;
    let head_roll_joint = definition
        .model()
        .names
        .jnt
        .iter()
        .position(|name| name.as_deref() == Some("head_roll"))
        .ok_or("source head_roll joint is missing")?;
    let head_roll_qpos = *definition
        .model()
        .fields
        .jnt_qposadr
        .get(head_roll_joint)
        .ok_or("source head_roll position address is missing")?;
    let head_roll_source_dof = *definition
        .model()
        .fields
        .jnt_dofadr
        .get(head_roll_joint)
        .ok_or("source head_roll velocity address is missing")?;
    let head_roll_input_position = *qpos
        .get(head_roll_qpos)
        .ok_or("source head_roll input position is missing")?;
    let mut simulation = SimulationWorld::new();
    if source_limit_diagnostic {
        let params = &simulation.world.integration_parameters;
        if (params.dt - 1.0 / 60.0).abs() > 1.0e-8
            || params.num_solver_iterations != 1
            || params.max_ccd_substeps != 1
        {
            return Err("source-limit diagnostic requires one native 1/60 s solver step".into());
        }
    }
    let assembly = build_structure(&mut simulation.world, &definition, &qpos)
        .map_err(|error| error.to_string())?;
    #[cfg(feature = "sim2sim_source_limit_probe")]
    if source_limit_diagnostic {
        let expected_sha = match definition.model().family.as_str() {
            "leg_allcollisions" => {
                "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb"
            }
            "roller_allcollisions" => {
                "6f88aa286e487c031250b427595a640b9f597604c9a20dcbca6fd69c49109c40"
            }
            _ => {
                return Err(
                    "source-limit experiment only accepts the frozen leg/roller family".into(),
                );
            }
        };
        if sha256(&definition_bytes) != expected_sha {
            return Err(
                "source-limit experiment requires the SHA-bound enriched scratch definition".into(),
            );
        }
        let fields = &definition.model().fields;
        let solref = fields
            .jnt_solref
            .as_ref()
            .ok_or("source jnt_solref is absent")?[head_roll_joint];
        let solimp = fields
            .jnt_solimp
            .as_ref()
            .ok_or("source jnt_solimp is absent")?[head_roll_joint];
        let margin = fields
            .jnt_margin
            .as_ref()
            .ok_or("source jnt_margin is absent")?[head_roll_joint];
        let dof_invweight0 = fields
            .dof_invweight0
            .as_ref()
            .ok_or("source dof_invweight0 is absent")?[head_roll_source_dof];
        let mapping = assembly
            .joint_mapping()
            .iter()
            .find(|mapping| mapping.source_joint == head_roll_joint)
            .ok_or("head_roll mapping is absent")?;
        let (multibody, link_id) = simulation
            .world
            .multibody_joints
            .get_mut(mapping.handle)
            .ok_or("head_roll articulation is absent")?;
        let joint = &mut multibody
            .link_mut(link_id)
            .ok_or("head_roll link is absent")?
            .joint;
        if !joint.sim2sim_set_source_limit_probe(
            solref.map(|v| v as f32),
            solimp.map(|v| v as f32),
            margin as f32,
            dof_invweight0 as f32,
        ) {
            return Err(
                "source-limit parameters fail the frozen single-axis probe contract".into(),
            );
        }
    }
    let (qvel_sha256, head_roll_input_velocity) = if args.len() >= 7 {
        let bytes = read_checked(&args[4], &args[5])?;
        let qvel: Vec<f64> = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if qvel.len() != definition.model().counts.nv || qvel.len() < 6 {
            return Err("qvel must have source nv entries".into());
        }
        let head_roll_input_velocity = qvel[head_roll_source_dof];
        let qvel: Vec<f32> = qvel.into_iter().map(|value| value as f32).collect();
        if qvel.iter().any(|value| !value.is_finite()) {
            return Err("source qvel exceeds the float32 backend range".into());
        }
        let world = &mut simulation.world;
        let first = assembly
            .joint_mapping()
            .first()
            .ok_or("source articulation has no mapped joints")?;
        let (multibody, _) = world
            .multibody_joints
            .get_mut(first.handle)
            .ok_or("source articulation disappeared")?;
        let linear = source_to_engine_vector([qvel[0], qvel[1], qvel[2]]);
        let angular = source_to_engine_vector([qvel[3], qvel[4], qvel[5]]);
        for (index, velocity) in [
            linear[0], linear[1], linear[2], angular[0], angular[1], angular[2],
        ]
        .into_iter()
        .enumerate()
        {
            multibody.generalized_velocity_mut()[index] = velocity;
        }
        for mapping in assembly.joint_mapping() {
            multibody.generalized_velocity_mut()[mapping.backend_dof] = qvel[mapping.source_dof];
        }
        multibody.update_rigid_bodies(&mut world.bodies, true);
        world
            .bodies
            .propagate_modified_body_positions_to_colliders(&mut world.colliders);
        (Some(sha256(&bytes)), head_roll_input_velocity)
    } else {
        (None, 0.0_f64)
    };
    let root_mapping = assembly
        .joint_mapping()
        .first()
        .ok_or("source articulation has no mapped joints")?;
    let (root_multibody, _) = simulation
        .world
        .multibody_joints
        .get(root_mapping.handle)
        .ok_or("source articulation disappeared")?;
    let initial_native_root_velocity: [f32; 6] =
        std::array::from_fn(|index| root_multibody.generalized_velocity()[index]);
    #[cfg(feature = "sim2sim_plain_mass_probe")]
    if plain_mass_diagnostic {
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get_mut(root_mapping.handle)
            .ok_or("source articulation disappeared before diagnostic selection")?;
        multibody.sim2sim_set_plain_mass_probe(true);
    }
    let before = assembly
        .actuator_joint_feedback(&simulation.world, &definition)
        .map_err(|error| error.to_string())?;
    let (actuator_torque_file_sha256, source_applied_actuator_nm, contributions) = if args.len()
        == 9
    {
        let bytes = read_checked(&args[6], &args[7])?;
        let values: Vec<f64> = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        let torques: [f32; robot_minigame::ACTION_DIMENSION] = values
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "actuator torque file must have 14 source-ordered values")?;
        if torques.iter().any(|torque| !torque.is_finite()) {
            return Err("actuator torque exceeds the float32 backend range".into());
        }
        if torques.iter().enumerate().any(|(index, torque)| {
            let [lower, upper] = definition.model().fields.actuator_forcerange[index];
            f64::from(*torque) < lower || f64::from(*torque) > upper
        }) {
            return Err("actuator torque exceeds the frozen source force range".into());
        }
        let contributions = assembly
            .actuator_body_torques(&simulation.world, &definition, &torques)
            .map_err(|error| error.to_string())?;
        (Some(sha256(&bytes)), Some(torques), contributions)
    } else {
        (None, None, Vec::new())
    };
    let snapshot = simulation
        .step_with_torques(&contributions)
        .map_err(|error| error.to_string())?;
    let after = assembly
        .actuator_joint_feedback(&simulation.world, &definition)
        .map_err(|error| error.to_string())?;
    let head_roll_feedback = before
        .iter()
        .zip(&after)
        .find(|(initial, _)| initial.source_joint == head_roll_joint)
        .ok_or("head_roll actuator feedback is missing")?;
    if head_roll_free_acceleration_diagnostic {
        let (initial, _) = head_roll_feedback;
        if initial.source_dof != head_roll_source_dof
            || (f64::from(initial.position) - head_roll_input_position).abs() > 1.0e-6
            || (f64::from(initial.velocity) - head_roll_input_velocity).abs() > 1.0e-6
        {
            return Err("head_roll initial q/v does not match the SHA-checked source input".into());
        }
        if snapshot.integration_count != 1
            || simulation.world.integration_parameters.max_ccd_substeps != 1
            || (simulation.world.integration_parameters.dt - 1.0 / 60.0).abs() > 1.0e-8
        {
            return Err("head_roll diagnostic requires exactly one 1/60 s integration".into());
        }
    }
    let mut rows = Vec::with_capacity(before.len());
    let mut common_epoch = None;
    let mut common_topology = None;
    #[cfg(feature = "sim2sim_plain_mass_probe")]
    let mut common_mass_diagnostic = None;
    for (initial, final_state) in before.iter().zip(&after) {
        if initial.source_joint != final_state.source_joint
            || initial.source_dof != final_state.source_dof
            || initial.backend_dof != final_state.backend_dof
            || initial.joint_handle != final_state.joint_handle
            || initial.body_handle != final_state.body_handle
        {
            return Err("driven joint identity changed over one step".into());
        }
        let mapping = assembly
            .joint_mapping()
            .iter()
            .find(|mapping| mapping.source_joint == initial.source_joint)
            .ok_or("source joint mapping disappeared")?;
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get(mapping.handle)
            .ok_or("native articulation disappeared")?;
        let (epoch, valid, coverage) = multibody.sim2sim_observation_status();
        if !valid || !coverage || epoch != snapshot.integration_count {
            return Err(format!(
                "native observation incomplete: source joint {} epoch {epoch} valid {valid} contact coverage {coverage}",
                initial.source_joint
            ));
        }
        let observation = multibody
            .sim2sim_contact_complete_observation()
            .ok_or("native observation getter disagrees with status")?;
        if observation.full_step_dt().to_bits()
            != simulation.world.integration_parameters.dt.to_bits()
            || common_epoch.is_some_and(|first| first != epoch)
            || common_topology.is_some_and(|first| first != observation.topology_epoch)
        {
            return Err(
                "native observation tick, dt or topology differs across driven DOFs".into(),
            );
        }
        common_epoch = Some(epoch);
        common_topology = Some(observation.topology_epoch);
        #[cfg(feature = "sim2sim_plain_mass_probe")]
        {
            let mass_diagnostic = (
                observation.plain_mass_probe_selected,
                observation.energy_guard_evaluated,
                observation.energy_guard_fallback,
                observation.energy_guard_acceleration_cleared,
            );
            if common_mass_diagnostic.is_some_and(|first| first != mass_diagnostic) {
                return Err("free-acceleration matrix status differs across driven DOFs".into());
            }
            common_mass_diagnostic = Some(mass_diagnostic);
        }
        let dof = initial.backend_dof;
        let inertial = *observation
            .inertial_projection
            .get(dof)
            .ok_or("inertial DOF missing")?;
        let gravity = *observation
            .gravity_projection
            .get(dof)
            .ok_or("gravity DOF missing")?;
        let generic = *observation
            .generic_joint_impulse
            .get(dof)
            .ok_or("generic joint DOF missing")?;
        let own_friction = *observation
            .own_dry_friction_impulse
            .get(dof)
            .ok_or("own dry-friction DOF missing")?;
        let contact_normal = *observation
            .contact_normal_impulse
            .get(dof)
            .ok_or("contact normal DOF missing")?;
        let contact_tangent = *observation
            .contact_tangent_impulse
            .get(dof)
            .ok_or("contact tangent DOF missing")?;
        let dt = observation.full_step_dt();
        let residual =
            gravity - inertial + (generic - own_friction + contact_normal + contact_tangent) / dt;
        if [
            inertial,
            gravity,
            generic,
            own_friction,
            contact_normal,
            contact_tangent,
            residual,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("native force term is non-finite".into());
        }
        #[allow(unused_mut)]
        let mut row = json!({
            "source_joint":initial.source_joint,
            "source_dof":initial.source_dof,
            "backend_dof":dof,
            "initial_position":initial.position,
            "initial_velocity":initial.velocity,
            "post_position":final_state.position,
            "post_velocity":final_state.velocity,
            "inertial_projection_nm":inertial,
            "gravity_projection_nm":gravity,
            "generic_joint_impulse_nms":generic,
            "own_dry_friction_impulse_nms":own_friction,
            "contact_normal_impulse_nms":contact_normal,
            "contact_tangent_impulse_nms":contact_tangent,
            "candidate_residual_nm":residual,
        });
        if let Some(torques) = source_applied_actuator_nm {
            let projected = *observation
                .user_force_projection
                .get(dof)
                .ok_or("native applied-force projection DOF missing")?;
            if !projected.is_finite() {
                return Err("native applied-force projection is non-finite".into());
            }
            let input_index = rows.len();
            row["native_user_force_projection_nm"] = json!(projected);
            row["requested_source_actuator_nm"] = json!(torques[input_index]);
        }
        #[cfg(feature = "sim2sim_limit_row_trace")]
        {
            let limit_samples: Vec<_> = observation
                .limit_row_timing
                .iter()
                .filter(|sample| sample.backend_dof == dof)
                .collect();
            if limit_samples.iter().any(|sample| {
                [
                    sample.coordinate,
                    sample.generalized_velocity,
                    sample.rhs,
                    sample.rhs_without_bias,
                    sample.impulse,
                    sample.impulse_bounds[0],
                    sample.impulse_bounds[1],
                ]
                .iter()
                .any(|value| !value.is_finite())
            }) {
                return Err("native limit-row trace contains a non-finite operand".into());
            }
            row["limit_row_timing"] = json!(
                limit_samples
                    .into_iter()
                    .map(|sample| json!({
                        "phase":sample.phase.as_str(),
                        "substep_id":sample.substep_id,
                        "row_index":sample.row_index,
                        "joint_local_dof":sample.joint_local_dof,
                        "backend_dof":sample.backend_dof,
                        "coordinate":sample.coordinate,
                        "generalized_velocity":sample.generalized_velocity,
                        "rhs":sample.rhs,
                        "rhs_without_bias":sample.rhs_without_bias,
                        "impulse_nms":sample.impulse,
                        "signed_generalized_impulse_nms":-sample.impulse,
                        "impulse_bounds":sample.impulse_bounds,
                    }))
                    .collect::<Vec<_>>()
            );
        }
        rows.push(row);
    }
    let active_contact_pairs: Vec<Value> = simulation
        .world
        .narrow_phase
        .contact_pairs()
        .filter(|pair| pair.has_any_active_contact())
        .map(|pair| {
            let collider1 = &simulation.world.colliders[pair.collider1];
            let collider2 = &simulation.world.colliders[pair.collider2];
            let body1 = collider1.parent().map(|body| simulation.world.bodies[body].user_data);
            let body2 = collider2.parent().map(|body| simulation.world.bodies[body].user_data);
            json!({
                "source_geom_ids":[collider1.user_data,collider2.user_data],
                "source_body_ids":[body1,body2],
                "solver_contact_count":pair.manifolds.iter().map(|m| m.data.solver_contacts.len()).sum::<usize>(),
                "normal_impulse_sum":pair.manifolds.iter().flat_map(|m| &m.points).map(|p| p.data.impulse).sum::<f32>(),
            })
        })
        .collect();
    let head_roll_free_acceleration = if head_roll_free_acceleration_diagnostic {
        if rows.iter().any(|row| {
            row["contact_normal_impulse_nms"].as_f64() != Some(0.0)
                || row["contact_tangent_impulse_nms"].as_f64() != Some(0.0)
        }) || active_contact_pairs
            .iter()
            .any(|pair| pair["normal_impulse_sum"].as_f64() != Some(0.0))
        {
            return Err("head_roll diagnostic requires zero active contact impulses".into());
        }
        let (initial, final_state) = head_roll_feedback;
        let head_roll_mapping = assembly
            .joint_mapping()
            .iter()
            .find(|mapping| mapping.source_joint == head_roll_joint)
            .ok_or("head_roll native joint mapping disappeared")?;
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get(head_roll_mapping.handle)
            .ok_or("head_roll native articulation disappeared")?;
        let observation = multibody
            .sim2sim_contact_complete_observation()
            .ok_or("head_roll native observation disappeared")?;
        if observation
            .contact_normal_impulse
            .iter()
            .chain(&observation.contact_tangent_impulse)
            .any(|impulse| *impulse != 0.0)
        {
            return Err("head_roll diagnostic requires zero articulation contact impulse".into());
        }
        let dof = initial.backend_dof;
        if dof >= multibody.ndofs()
            || multibody.inv_augmented_mass().l().nrows() != multibody.ndofs()
        {
            return Err("head_roll native mass matrix has an unexpected reduced DOF layout".into());
        }
        let mut unit = DVector::zeros(multibody.ndofs());
        unit[dof] = 1.0;
        let solved = multibody
            .inv_augmented_mass()
            .solve(&unit)
            .ok_or("head_roll plain mass matrix is singular")?;
        let inverse_inertia = solved[dof];
        let acceleration = *multibody
            .generalized_acceleration()
            .get(dof)
            .ok_or("head_roll native free acceleration is missing")?;
        if !inverse_inertia.is_finite() || inverse_inertia <= 0.0 || !acceleration.is_finite() {
            return Err("head_roll free dynamics has non-finite or non-positive inertia".into());
        }
        Some(json!({
            "source_joint":head_roll_joint,
            "source_dof":head_roll_source_dof,
            "backend_dof":dof,
            "initial_position_rad":initial.position,
            "initial_velocity_rad_s":initial.velocity,
            "post_position_rad":final_state.position,
            "post_velocity_rad_s":final_state.velocity,
            "upper_limit_row_jacobian_sign":-1,
            "pre_constraint_free_generalized_acceleration_rad_s2":acceleration,
            "pre_constraint_free_row_acceleration_rad_s2":-acceleration,
            "plain_mass_effective_inverse_inertia_rad_s2_per_nm":inverse_inertia,
            "read_after_one_completed_step":true,
        }))
    } else {
        None
    };
    let mut report = json!({
        "scope":"robot_only_rapier_one_step_raw_force_diagnostic",
        "model_file_sha256":sha256(&definition_bytes),
        "qpos_file_sha256":sha256(&qpos_bytes),
        "qvel_file_sha256":qvel_sha256,
        "initial_native_root_velocity":initial_native_root_velocity,
        "family":definition.model().family,
        "physics_integrations":snapshot.integration_count,
        "policy_inferences":0,
        "actual_step_dt_seconds":simulation.world.integration_parameters.dt,
        "native_observation_epoch":common_epoch,
        "native_topology_epoch":common_topology,
        "active_contact_pair_count":snapshot.active_contact_pair_count,
        "active_contact_pairs":active_contact_pairs,
        "rows":rows,
        "bam_external_load_qualified":false,
        "source_target_equivalent":false,
    });
    if let (Some(sha), Some(torques)) = (actuator_torque_file_sha256, source_applied_actuator_nm) {
        report["actuator_torque_file_sha256"] = json!(sha);
        report["source_applied_actuator_nm"] = json!(torques);
    }
    if let Some(head_roll_free_acceleration) = head_roll_free_acceleration {
        report["head_roll_free_acceleration_diagnostic"] = head_roll_free_acceleration;
    }
    if source_limit_diagnostic {
        report["source_limit_probe_selected"] = json!(true);
        report["source_limit_probe_joint"] = json!("head_roll");
    }
    #[cfg(feature = "sim2sim_plain_mass_probe")]
    {
        let (diagnostic, evaluated, fallback, cleared) =
            common_mass_diagnostic.ok_or("free-acceleration matrix status was not recorded")?;
        if diagnostic != plain_mass_diagnostic {
            return Err("diagnostic mass selection was not applied to this articulation".into());
        }
        report["selected_free_acceleration_matrix"] = json!(if diagnostic {
            "plain_mass_diagnostic"
        } else if cleared {
            "acceleration_cleared_after_plain_mass_fallback"
        } else if fallback {
            "plain_mass_energy_guard_fallback"
        } else {
            "implicit_gyro_coriolis_mass"
        });
        report["energy_guard_evaluated"] = json!(evaluated);
        report["energy_guard_fallback"] = json!(fallback);
        report["energy_guard_acceleration_cleared"] = json!(cleared);
    }
    Ok(report)
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut plain_mass_diagnostic = false;
    let mut head_roll_free_acceleration_diagnostic = false;
    let mut source_limit_diagnostic = false;
    while let Some(option) = args.first() {
        match option.as_str() {
            "--plain-mass-diagnostic" if !plain_mass_diagnostic => {
                plain_mass_diagnostic = true;
            }
            "--head-roll-free-acceleration-diagnostic"
                if !head_roll_free_acceleration_diagnostic =>
            {
                head_roll_free_acceleration_diagnostic = true;
            }
            "--head-roll-source-limit-diagnostic" if !source_limit_diagnostic => {
                source_limit_diagnostic = true;
            }
            _ => break,
        }
        args.remove(0);
    }
    if args.len() != 5 && args.len() != 7 && args.len() != 9 {
        eprintln!(
            "usage: paired_force_probe [--plain-mass-diagnostic] [--head-roll-free-acceleration-diagnostic] [--head-roll-source-limit-diagnostic] MODEL MODEL_SHA QPOS QPOS_SHA [QVEL QVEL_SHA [TORQUES TORQUES_SHA]] OUTPUT.json"
        );
        std::process::exit(2);
    }
    let report = match run(
        &args,
        plain_mass_diagnostic,
        head_roll_free_acceleration_diagnostic,
        source_limit_diagnostic,
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("paired force probe: {error}");
            std::process::exit(1);
        }
    };
    let content = serde_json::to_vec_pretty(&report).expect("force report serializes");
    if let Err(error) = fs::write(args.last().expect("output path argument exists"), content) {
        eprintln!("paired force probe: {error}");
        std::process::exit(1);
    }
}
