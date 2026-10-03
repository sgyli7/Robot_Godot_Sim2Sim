//! Fixed ten-second continuation of the original station standing command.
//! Actual native mechanics only; no image, task, Qwen or real-time claim.

use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::{contract::G1Command, policy::bound_bytes};
use simulation_minigame::g1::runner::{G1ActuatorBackend, G1Runner, G1RunnerConfig};
use std::{fs, io::Write, path::PathBuf};

#[test]
#[ignore = "pinned public station and standalone Homie;500 actual50Hz native steps;0fresh images/VLA"]
fn real_station_stand_ten_seconds() -> Result<(), String> {
    let config_path =
        PathBuf::from(std::env::var("G1_STATION_STAND_CONFIG").map_err(|e| e.to_string())?);
    let hash = std::env::var("G1_STATION_STAND_CONFIG_SHA256").map_err(|e| e.to_string())?;
    let bytes = bound_bytes(&config_path, &hash).map_err(|e| e.to_string())?;
    if bytes.len() > 1_048_576 {
        return Err("station stand configuration exceeds bound".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let mut body: G1RunnerConfig =
        serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
    if body.task_objects.is_some() || body.actuator_backend != G1ActuatorBackend::NativeForceBased {
        return Err("stand continuation requires original standalone native Homie".into());
    }
    let identity: G1StationConfiguration =
        serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, prepared, _) = prepare(&scene, &identity, body.floor_contact_friction)?;
    body.startup_environment = Some(prepared);
    let output =
        PathBuf::from(std::env::var("G1_STATION_STAND_OUTPUT").map_err(|e| e.to_string())?);
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let mut owner = G1Runner::load(&body).map_err(|e| e.to_string())?;
    for tick in 1..=500 {
        let step = owner
            .step(&G1Command::default())
            .map_err(|e| e.to_string())?;
        if step.integration_count != tick
            || step.torque_update_count != tick
            || step.inference.inference_count != tick
            || step.step_configuration.physics_hz != 50
            || step.step_configuration.dt != 0.02
            || step.step_configuration.num_solver_iterations != 1
            || step.step_configuration.num_internal_pgs_iterations != 1
            || step.step_configuration.max_ccd_substeps != 1
            || step
                .native_static_environment
                .as_ref()
                .is_none_or(|r| r.collider_count != 2553)
        {
            return Err("standing continuation changed physical timeline/world".into());
        }
        let standing = step.root_upright_cosine > 0.95 && step.root_position_source[2] >= 0.35;
        writeln!(
            trace,
            "{}",
            serde_json::json!({
                "body":{"mobile_homie_v2":step},"phase":"fixed_stand_ten_second_mechanics",
                "standing":standing,"fresh_images":0,"fresh_vla_calls":0,"qwen_calls":0,
                "real_time_qualified":false,"task_qualified":false,
            })
        )
        .map_err(|e| e.to_string())?;
        if !standing {
            return Err(format!("finite standing guard failed at Tick{tick}"));
        }
    }
    trace.flush().map_err(|e| e.to_string())?;
    Ok(())
}
