//! Same-world, robot-only, eight-step source-limit diagnostic.
//! The feature and explicit CLI switch are both required. Nothing here selects
//! the diagnostic in a production assembly or qualifies contact/BAM behavior.

use std::{env, fs, path::Path};

use rapier3d::prelude::Multibody;
use robot_minigame::definition::RobotDefinition;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};

const STEPS: usize = 8;

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

fn run(args: &[String]) -> Result<Value, String> {
    if !Multibody::sim2sim_observation_backend_supported() {
        return Err("native Rapier observation backend is unavailable".into());
    }
    let model_bytes = read_checked(&args[0], &args[1])?;
    let model_sha = sha256(&model_bytes);
    let definition = RobotDefinition::load_json(Path::new(&args[0]), &args[1])
        .map_err(|error| error.to_string())?;
    let (expected_model_sha, expected_qpos_sha) = match definition.model().family.as_str() {
        "leg_allcollisions" => (
            "e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb",
            "e16f8e7ddf1d1cf2061b4c760faa1cbf78536d1cd8e6b992ca415eef10ae21bf",
        ),
        "roller_allcollisions" => (
            "6f88aa286e487c031250b427595a640b9f597604c9a20dcbca6fd69c49109c40",
            "df50ef2c31243ef01c6419f18238008c9a2d8d3e444073b82c61cb27b3fb7773",
        ),
        _ => return Err("continuous probe only accepts the frozen leg/roller family".into()),
    };
    if model_sha != expected_model_sha {
        return Err("continuous probe requires the SHA-bound enriched scratch definition".into());
    }
    let qpos_bytes = read_checked(&args[2], &args[3])?;
    if sha256(&qpos_bytes) != expected_qpos_sha {
        return Err("continuous probe requires the frozen upper+0.02 rad qpos bytes".into());
    }
    let qpos: Vec<f64> = serde_json::from_slice(&qpos_bytes).map_err(|error| error.to_string())?;
    let qvel_bytes = read_checked(&args[4], &args[5])?;
    let qvel: Vec<f64> = serde_json::from_slice(&qvel_bytes).map_err(|error| error.to_string())?;
    if qvel.len() != definition.model().counts.nv || qvel.iter().any(|value| *value != 0.0) {
        return Err("continuous probe requires a source-sized all-zero qvel".into());
    }
    let fields = &definition.model().fields;
    let source_joint = definition
        .model()
        .names
        .jnt
        .iter()
        .position(|name| name.as_deref() == Some("head_roll"))
        .ok_or("source head_roll joint is missing")?;
    let source_qpos = fields.jnt_qposadr[source_joint];
    let source_dof = fields.jnt_dofadr[source_joint];
    let upper = fields.jnt_range[source_joint][1];
    if qpos.len() != definition.model().counts.nq
        || (qpos[source_qpos] - (upper + 0.02)).abs() > 1.0e-10
    {
        return Err("continuous probe requires the frozen upper+0.02 rad head_roll qpos".into());
    }
    let solref = fields
        .jnt_solref
        .as_ref()
        .ok_or("source jnt_solref is absent")?[source_joint];
    let solimp = fields
        .jnt_solimp
        .as_ref()
        .ok_or("source jnt_solimp is absent")?[source_joint];
    let margin = fields
        .jnt_margin
        .as_ref()
        .ok_or("source jnt_margin is absent")?[source_joint];
    let dof_invweight0 = fields
        .dof_invweight0
        .as_ref()
        .ok_or("source dof_invweight0 is absent")?[source_dof];

    let mut simulation = SimulationWorld::new();
    let params = &simulation.world.integration_parameters;
    if (params.dt - 1.0 / 60.0).abs() > 1.0e-8
        || params.num_solver_iterations != 1
        || params.max_ccd_substeps != 1
    {
        return Err("continuous probe requires one native 1/60 s solver step".into());
    }
    let assembly = build_structure(&mut simulation.world, &definition, &qpos)
        .map_err(|error| error.to_string())?;
    let mapping = assembly
        .joint_mapping()
        .iter()
        .find(|mapping| mapping.source_joint == source_joint)
        .ok_or("head_roll source-to-native mapping is absent")?;
    let (multibody, link_id) = simulation
        .world
        .multibody_joints
        .get_mut(mapping.handle)
        .ok_or("head_roll articulation is absent")?;
    if multibody
        .generalized_velocity()
        .iter()
        .any(|value| *value != 0.0)
    {
        return Err("native articulation did not initialize at zero velocity".into());
    }
    let joint = &mut multibody
        .link_mut(link_id)
        .ok_or("head_roll link is absent")?
        .joint;
    if !joint.sim2sim_set_source_limit_probe(
        solref.map(|value| value as f32),
        solimp.map(|value| value as f32),
        margin as f32,
        dof_invweight0 as f32,
    ) {
        return Err("frozen source limit parameters are unsupported".into());
    }

    let mut rows = Vec::with_capacity(STEPS);
    let mut topology_epoch = None;
    for step in 1..=STEPS {
        let before = assembly
            .actuator_joint_feedback(&simulation.world, &definition)
            .map_err(|error| error.to_string())?;
        let pre = before
            .iter()
            .find(|channel| channel.source_joint == source_joint)
            .ok_or("head_roll pre-step feedback is absent")?;
        let snapshot = simulation
            .step_with_torques(&[])
            .map_err(|error| error.to_string())?;
        let after = assembly
            .actuator_joint_feedback(&simulation.world, &definition)
            .map_err(|error| error.to_string())?;
        let post = after
            .iter()
            .find(|channel| channel.source_joint == source_joint)
            .ok_or("head_roll post-step feedback is absent")?;
        let (multibody, _) = simulation
            .world
            .multibody_joints
            .get(mapping.handle)
            .ok_or("head_roll articulation disappeared")?;
        let (epoch, valid, coverage) = multibody.sim2sim_observation_status();
        if !valid || !coverage || epoch != step as u64 || snapshot.integration_count != step as u64
        {
            return Err(format!(
                "step {step}: native observation epoch or coverage is invalid"
            ));
        }
        let observation = multibody
            .sim2sim_contact_complete_observation()
            .ok_or("head_roll native observation is absent")?;
        if topology_epoch.is_some_and(|first| first != observation.topology_epoch) {
            return Err("native topology changed during continuous probe".into());
        }
        topology_epoch = Some(observation.topology_epoch);
        let final_rows: Vec<_> = observation
            .limit_row_timing
            .iter()
            .filter(|sample| {
                sample.backend_dof == mapping.backend_dof
                    && sample.phase.as_str() == "after_unbiased_solve"
            })
            .collect();
        if final_rows.len() != 1 {
            return Err(format!(
                "step {step}: expected exactly one native head_roll limit row"
            ));
        }
        let row = final_rows[0];
        if (observation.generic_joint_impulse[mapping.backend_dof] + row.impulse).abs() > 1.0e-10 {
            return Err(format!(
                "step {step}: final limit impulse disagrees with native generic-joint observation"
            ));
        }
        let contact_normal = observation.contact_normal_impulse[mapping.backend_dof];
        let contact_tangent = observation.contact_tangent_impulse[mapping.backend_dof];
        if [
            pre.position,
            pre.velocity,
            post.position,
            post.velocity,
            row.impulse,
            contact_normal,
            contact_tangent,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(format!("step {step}: non-finite native result"));
        }
        let mut contact_pairs: Vec<_> = simulation
            .world
            .narrow_phase
            .contact_pairs()
            .filter(|pair| pair.has_any_active_contact())
            .map(|pair| {
                let first = &simulation.world.colliders[pair.collider1];
                let second = &simulation.world.colliders[pair.collider2];
                let first_body = first
                    .parent()
                    .map(|handle| simulation.world.bodies[handle].user_data);
                let second_body = second
                    .parent()
                    .map(|handle| simulation.world.bodies[handle].user_data);
                let (geom_ids, body_ids) = if first.user_data <= second.user_data {
                    (
                        [first.user_data, second.user_data],
                        [first_body, second_body],
                    )
                } else {
                    (
                        [second.user_data, first.user_data],
                        [second_body, first_body],
                    )
                };
                (
                    geom_ids,
                    body_ids,
                    pair.manifolds
                        .iter()
                        .map(|manifold| manifold.data.solver_contacts.len())
                        .sum::<usize>(),
                    pair.manifolds
                        .iter()
                        .flat_map(|manifold| &manifold.points)
                        .map(|point| point.data.impulse)
                        .sum::<f32>(),
                )
            })
            .collect();
        contact_pairs.sort_by_key(|pair| pair.0);
        if contact_pairs.len() != snapshot.active_contact_pair_count {
            return Err(format!(
                "step {step}: native contact-pair count changed during read"
            ));
        }
        let dt = f64::from(simulation.world.integration_parameters.dt);
        rows.push(json!({
            "step":step,
            "integration_count":snapshot.integration_count,
            "native_observation_epoch":epoch,
            "native_topology_epoch":observation.topology_epoch,
            "pre_position_rad":pre.position,
            "pre_velocity_rad_s":pre.velocity,
            "post_position_rad":post.position,
            "post_velocity_rad_s":post.velocity,
            "limit_row_active":row.impulse_bounds[1] > 0.0,
            "limit_row_impulse_nms":row.impulse,
            "limit_row_signed_generalized_impulse_nms":-row.impulse,
            "limit_row_signed_generalized_force_nm":-f64::from(row.impulse) / dt,
            "limit_row_rhs_rad_s":row.rhs,
            "limit_row_rhs_without_bias_rad_s":row.rhs_without_bias,
            "generic_joint_impulse_nms":observation.generic_joint_impulse[mapping.backend_dof],
            "contact_normal_impulse_nms":contact_normal,
            "contact_tangent_impulse_nms":contact_tangent,
            "all_dof_contact_normal_max_abs_nms":observation.contact_normal_impulse.iter().copied().map(f32::abs).fold(0.0_f32,f32::max),
            "all_dof_contact_tangent_max_abs_nms":observation.contact_tangent_impulse.iter().copied().map(f32::abs).fold(0.0_f32,f32::max),
            "active_contact_pair_count":snapshot.active_contact_pair_count,
            "active_contact_pairs":contact_pairs.iter().map(|(geom_ids,body_ids,solver_contacts,normal_impulse)| json!({
                "source_geom_ids":geom_ids,
                "source_body_ids":body_ids,
                "solver_contact_count":solver_contacts,
                "normal_impulse_sum_nms":normal_impulse,
            })).collect::<Vec<_>>(),
        }));
    }
    let integration_count = rows.len();
    Ok(json!({
        "scope":"same_world_eight_step_robot_only_source_limit_diagnostic",
        "family":definition.model().family,
        "model_file_sha256":model_sha,
        "qpos_file_sha256":sha256(&qpos_bytes),
        "qvel_file_sha256":sha256(&qvel_bytes),
        "source_joint":source_joint,
        "source_dof":source_dof,
        "backend_dof":mapping.backend_dof,
        "initial_position_rad":qpos[source_qpos],
        "initial_velocity_rad_s":0.0,
        "actual_step_dt_seconds":simulation.world.integration_parameters.dt,
        "assembly_builds":1,
        "physics_integrations":integration_count,
        "policy_inferences":0,
        "actuator_torque_nm":0.0,
        "rows":rows,
        "bam_external_load_qualified":false,
        "source_target_equivalent":false,
    }))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 8 || args[0] != "--source-limit-continuous-diagnostic" {
        eprintln!(
            "usage: source_limit_continuous_probe --source-limit-continuous-diagnostic MODEL.json MODEL_SHA QPOS.json QPOS_SHA QVEL.json QVEL_SHA OUTPUT.json"
        );
        std::process::exit(2);
    }
    let report = match run(&args[1..]) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("source limit continuous probe: {error}");
            std::process::exit(1);
        }
    };
    let content = serde_json::to_vec_pretty(&report).expect("continuous report serializes");
    if let Err(error) = fs::write(&args[7], content) {
        eprintln!("source limit continuous probe: {error}");
        std::process::exit(1);
    }
}
