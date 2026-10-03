//! Test-only replay to replace ambiguous cached contact receipts with active ones.
//! Original actions stay fixed; refreshed wall expiry is explicitly offline.

use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::policy::bound_bytes;
use serde::Deserialize;
use simulation_minigame::g1::task_runner::{
    ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskRunner, ArenaTaskRunnerConfig,
};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use task_minigame::{policy::PolicyActionChunk, types::TaskProfile};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedCase {
    configuration: PathBuf,
    configuration_sha256: String,
    chunks: PathBuf,
    chunks_sha256: String,
    output: PathBuf,
}

#[test]
#[ignore = "hash-bound ten-case offline T1 revalidation;3800single50Hz integrations;0freshimages/VLA/Qwen"]
fn revalidate_frozen_static_active_contacts() -> Result<(), String> {
    let fixture =
        PathBuf::from(std::env::var("G1_STATIC_CONTACT_CASES").map_err(|e| e.to_string())?);
    let hash = std::env::var("G1_STATIC_CONTACT_CASES_SHA256").map_err(|e| e.to_string())?;
    let bytes = bound_bytes(&fixture, &hash).map_err(|e| e.to_string())?;
    if bytes.len() > 64 * 1024 {
        return Err("static revalidation case manifest exceeds bound".into());
    }
    let cases: Vec<SavedCase> = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if cases.len() != 10 {
        return Err("static revalidation requires exactly the original ten cases".into());
    }
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    for case in cases {
        let config_bytes = bound_bytes(&case.configuration, &case.configuration_sha256)
            .map_err(|e| e.to_string())?;
        let action_bytes =
            bound_bytes(&case.chunks, &case.chunks_sha256).map_err(|e| e.to_string())?;
        if config_bytes.len() > 128 * 1024 || action_bytes.len() > 1024 * 1024 {
            return Err("static revalidation fixture exceeds its byte budget".into());
        }
        let value: serde_json::Value =
            serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
        let mut config: ArenaTaskRunnerConfig =
            serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
        let chunks: Vec<PolicyActionChunk> =
            serde_json::from_slice(&action_bytes).map_err(|e| e.to_string())?;
        if value["static_startup"] != true
            || value["predictive_limit_diagnostic"] != true
            || value["diagnostic_constraint_sweeps"] != 16
            || chunks.len() != 8
            || chunks.iter().enumerate().any(|(i, c)| {
                c.profile != TaskProfile::StaticApple
                    || c.frames.len() != 40
                    || c.observation.episode_id != config.body.episode_id()
                    || c.sequence_id != i as u64 + 1
                    || c.action_period_ns != 20_000_000
                    || c.observation.sim_time_ns != 1_200_000_000 + i as u64 * 800_000_000
            })
        {
            return Err("static revalidation changed original profile/timeline/startup".into());
        }
        let identity: G1StationConfiguration =
            serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
        let ArenaTaskBodyConfig::StaticAgile(body) = &mut config.body else {
            return Err("static revalidation requires AGILE".into());
        };
        let (_, environment, _) = prepare(&scene, &identity, body.floor_contact_friction)?;
        body.startup_environment = Some(environment);
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&case.output)
            .map_err(|e| e.to_string())?;
        let mut owner = ArenaTaskRunner::load_static_predictive_constraint_diagnostic(&config)
            .map_err(|e| e.to_string())?;
        for _ in 0..60 {
            let step = owner
                .diagnostic_static_startup_step(&mut || Ok(()))
                .map_err(|e| e.to_string())?;
            writeln!(
                trace,
                "{}",
                serde_json::json!({"body":step,"phase":"offline_exact_original_startup",
                "fresh_images":0,"fresh_vla_calls":0,"qwen_calls":0,"qualified":false})
            )
            .map_err(|e| e.to_string())?;
        }
        for original in chunks {
            let original_wall_ms = original.observation.captured_at_unix_ms;
            let mut chunk = original;
            chunk.observation.captured_at_unix_ms = u64::try_from(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| e.to_string())?
                    .as_millis(),
            )
            .map_err(|e| e.to_string())?;
            let command = ArenaTaskCommand {
                chunk: Arc::new(chunk),
                scheduled_start_sim_ns: None,
            };
            for _ in 0..40 {
                let step = owner
                    .step_with_guard(&command, &mut || Ok(()))
                    .map_err(|e| e.to_string())?;
                writeln!(trace,"{}",serde_json::json!({"body":step.body,"execution":step.execution,
                    "phase":"offline_exact_saved_original_t1_action","offline_wall_time_refreshed":true,
                    "original_captured_at_unix_ms":original_wall_ms,"fresh_images":0,"fresh_vla_calls":0,
                    "qwen_calls":0,"qualified":false})).map_err(|e| e.to_string())?;
            }
        }
        assert_eq!(owner.progress_counts().integration_count, 380);
    }
    Ok(())
}
