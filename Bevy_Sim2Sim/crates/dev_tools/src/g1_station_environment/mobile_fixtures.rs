//! One bounded mechanics preflight before station T2 fresh-image inference.

use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::{
    contract::G1Command, policy::bound_bytes, task_fixtures::T2BackgroundSelection,
};
use simulation_minigame::g1::{
    runner::G1Runner,
    task_runner::{ArenaTaskBodyConfig, ArenaTaskRunner, ArenaTaskRunnerConfig},
};
use std::{fs, io::Write, path::PathBuf};

#[test]
#[ignore = "frozen public station plus original shelf/table;100 actual50Hz steps,0VLA/Qwen; independent support preflight"]
fn real_station_t2_support_fixture_preflight() -> Result<(), String> {
    let path = PathBuf::from(std::env::var("G1_STATION_T2_CONFIG").map_err(|e| e.to_string())?);
    let hash = std::env::var("G1_STATION_T2_CONFIG_SHA256").map_err(|e| e.to_string())?;
    let bytes = bound_bytes(&path, &hash).map_err(|e| e.to_string())?;
    if bytes.len() > 1_048_576 {
        return Err("station T2 configuration exceeds bound".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let mut task: ArenaTaskRunnerConfig =
        serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
    let identity: G1StationConfiguration =
        serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::MobileHomieV2(body) = &mut task.body else {
        return Err("station T2 requires original Homie_v2".into());
    };
    if body
        .task_objects
        .as_ref()
        .and_then(|scene| scene.source_t2_background.as_ref())
        .is_none_or(|background| background.selection != T2BackgroundSelection::StationTaskFixtures)
    {
        return Err("preflight requires explicit station support selection".into());
    }
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, environment, preparation) = prepare(&scene, &identity, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let body = body.clone();
    // This must remain rejected by the original complete-warehouse factory.
    let source_rejection = ArenaTaskRunner::load_mobile_constraint_diagnostic(&task)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    if !source_rejection.contains("complete source background coverage") {
        return Err("original source factory incorrectly admitted station fixture coverage".into());
    }
    let mut owner =
        G1Runner::load_mobile_station_fixture_diagnostic(&body).map_err(|e| e.to_string())?;
    let output = PathBuf::from(std::env::var("G1_STATION_T2_OUTPUT").map_err(|e| e.to_string())?);
    let mut trace = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let initial = owner
        .task_object_frame()
        .map_err(|e| e.to_string())?
        .ok_or("task objects absent")?;
    writeln!(trace, "{}", serde_json::json!({"phase":"initialization", "preparation":preparation,"task_objects":initial,"body":owner.initial_frame().map_err(|e| e.to_string())?,"fresh_vla_calls":0,"qwen_calls":0,"task_qualified":false})).map_err(|e| e.to_string())?;
    for tick in 1..=100 {
        let step = owner
            .step(&G1Command::default())
            .map_err(|e| e.to_string())?;
        let objects = owner
            .task_object_frame()
            .map_err(|e| e.to_string())?
            .ok_or("task objects absent")?;
        let standing = step.root_upright_cosine > 0.95 && step.root_position_source[2] >= 0.35;
        let background = objects
            .source_t2_background
            .as_ref()
            .ok_or("support fixtures absent")?;
        let valid_world = step.integration_count == tick
            && step.torque_update_count == tick
            && step.inference.inference_count == tick
            && step.step_configuration.physics_hz == 50
            && step.step_configuration.dt == 0.02
            && step.step_configuration.num_solver_iterations == 1
            && step.step_configuration.num_internal_pgs_iterations == 4
            && step.step_configuration.max_ccd_substeps == 1
            && step
                .native_static_environment
                .as_ref()
                .is_some_and(|receipt| {
                    receipt.collider_count == 2553 && receipt.original_broad_floor_removed
                })
            && background.bodies.len() == 2
            && background
                .bodies
                .iter()
                .map(|body| body.collider_count)
                .sum::<usize>()
                == 6
            && !background.existing_floor_reused
            && background.station_task_fixture_selection.is_some();
        writeln!(trace,"{}",serde_json::json!({"phase":"station_support_standing_mechanics","body":{"mobile_homie_v2":step},"task_objects":objects,"standing":standing,"valid_world":valid_world,"fresh_vla_calls":0,"qwen_calls":0,"real_time_qualified":false,"task_qualified":false})).map_err(|e| e.to_string())?;
        trace.flush().map_err(|e| e.to_string())?;
        if !valid_world || !standing {
            return Err(format!(
                "station fixture mechanics guard failed at Tick{tick}"
            ));
        }
    }
    Ok(())
}
