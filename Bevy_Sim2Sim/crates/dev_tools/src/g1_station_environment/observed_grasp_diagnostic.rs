//! One explicitly offline native mechanical probe after joint preflight.
//! Saved original RGB/actions are disclosed; no fresh camera or task inference.

use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::policy::bound_bytes;
use serde::Deserialize;
use simulation_minigame::g1::{
    static_grasp::StaticObservedGraspGoal,
    static_startup::{StaticStartupCommand, StaticStartupStep},
    task_runner::{ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskRunnerConfig},
    worker::{G1WorkerPhase, OwnerStepTrace, StaticStartupWorker, TimedCommand, WorkerSnapshot},
};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use task_minigame::policy::PolicyActionChunk;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    configuration: PathBuf,
    configuration_sha256: String,
    original_chunk: PathBuf,
    original_chunk_sha256: String,
    goal: PathBuf,
    goal_sha256: String,
    output: PathBuf,
}
fn wait(
    worker: &StaticStartupWorker,
    trace: &OwnerStepTrace<StaticStartupStep>,
    file: &mut fs::File,
    target: u64,
) -> Result<Arc<WorkerSnapshot<StaticStartupStep>>, String> {
    let end = Instant::now() + Duration::from_secs(15);
    loop {
        for r in trace.drain() {
            writeln!(file,"{}",serde_json::json!({"body":r.step.body,"execution":r.step.execution,"timing":r.timing})).map_err(|e|e.to_string())?;
        }
        if let Some(s) = worker.take_latest() {
            if s.phase == G1WorkerPhase::Failed {
                writeln!(file,"{}",serde_json::json!({"failed_owner_snapshot":true,"target_tick":target,"phase":format!("{:?}",s.phase),"reason":s.reason,"timing":s.timing})).map_err(|e|e.to_string())?;
                file.flush().map_err(|e| e.to_string())?;
                return Err(s
                    .reason
                    .clone()
                    .unwrap_or("observed grasp owner failed".into()));
            }
            if s.phase == G1WorkerPhase::Paused && s.timing.episode_integrations == target {
                return Ok(s);
            }
        }
        if Instant::now() > end {
            return Err("bounded observed grasp owner wait expired".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn wall() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[test]
#[ignore = "one offline original100Tick prefix plus bounded RGB/self classical grasp,lift,hold;0freshcamera/VLA calls"]
fn saved_observed_grasp_in_original_world() -> Result<(), String> {
    let path = PathBuf::from(
        std::env::var("G1_NATIVE_OBSERVED_GRASP_FIXTURE").map_err(|e| e.to_string())?,
    );
    let hash =
        std::env::var("G1_NATIVE_OBSERVED_GRASP_FIXTURE_SHA256").map_err(|e| e.to_string())?;
    let bytes = bound_bytes(&path, &hash).map_err(|e| e.to_string())?;
    if bytes.len() > 16384 {
        return Err("native grasp fixture exceeds bound".into());
    }
    let f: Fixture = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_slice(
        &bound_bytes(&f.configuration, &f.configuration_sha256).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut config: ArenaTaskRunnerConfig =
        serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
    let mut chunk: PolicyActionChunk = serde_json::from_slice(
        &bound_bytes(&f.original_chunk, &f.original_chunk_sha256).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut goal: StaticObservedGraspGoal =
        serde_json::from_slice(&bound_bytes(&f.goal, &f.goal_sha256).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    goal.validate().map_err(|e| e.to_string())?;
    if value["static_startup"] != true
        || value["predictive_limit_diagnostic"] != true
        || value["diagnostic_constraint_sweeps"] != 16
        || chunk.profile != task_minigame::types::TaskProfile::StaticApple
        || chunk.sequence_id != 1
        || chunk.frames.len() != 40
        || chunk.observation.sim_time_ns != 60 * 20_000_000
        || chunk.observation.episode_id != config.body.episode_id()
        || goal.observation.episode_id != config.body.episode_id()
    {
        return Err("changed offline observed-grasp profile/action identity".into());
    }
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let identity: G1StationConfiguration =
        serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::StaticAgile(body) = &mut config.body else {
        return Err("observed grasp requires AGILE".into());
    };
    let (_, environment, _) = prepare(&scene, &identity, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let episode = config.body.episode_id();
    let worker = StaticStartupWorker::spawn_static_startup(config).map_err(|e| e.to_string())?;
    let trace = worker.subscribe_steps(512).map_err(|e| e.to_string())?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(f.output)
        .map_err(|e| e.to_string())?;
    wait(&worker, &trace, &mut file, 0)?;
    worker
        .submit(TimedCommand {
            episode_id: episode,
            valid_until_sim_ns: 60 * 20_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(4),
            command: StaticStartupCommand::Startup {
                episode_id: episode,
            },
        })
        .map_err(|e| e.to_string())?;
    wait(&worker, &trace, &mut file, 60)?;
    // Only this explicitly offline fixture refreshes wall time. Simulation
    // identity, source pixels/actions, joint values and original timing stay bound.
    chunk.observation.captured_at_unix_ms = wall();
    worker
        .submit(TimedCommand {
            episode_id: episode,
            valid_until_sim_ns: 100 * 20_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(3),
            command: StaticStartupCommand::OriginalVla(ArenaTaskCommand {
                chunk: Arc::new(chunk),
                scheduled_start_sim_ns: None,
            }),
        })
        .map_err(|e| e.to_string())?;
    wait(&worker, &trace, &mut file, 100)?;
    let target = 100 + goal.increments.len() as u64;
    goal.observation.captured_at_unix_ms = wall();
    worker
        .submit(TimedCommand {
            episode_id: episode,
            valid_until_sim_ns: target * 20_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(12),
            command: StaticStartupCommand::ObservedGrasp(goal),
        })
        .map_err(|e| e.to_string())?;
    let final_state = wait(&worker, &trace, &mut file, target)?;
    assert_eq!(trace.dropped_records(), 0);
    assert_eq!(final_state.timing.total_integrations, target);
    assert_eq!(final_state.timing.total_successful_inferences, target);
    let simulation_minigame::g1::task_runner::ArenaBodyStep::StaticAgile(body) = &final_state
        .step
        .as_ref()
        .ok_or("final observed grasp step absent")?
        .body
    else {
        return Err("foreign final body".into());
    };
    assert_eq!(body.step_configuration.dt, 0.02);
    assert_eq!(body.step_configuration.num_internal_pgs_iterations, 16);
    drop(worker);
    Ok(())
}
