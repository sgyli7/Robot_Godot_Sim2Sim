//! Single hash-bound offline mechanical test of the separate static visual skill.
//! Saved original actions/RGB are disclosed; wall expiry refreshed only here.
use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::policy::bound_bytes;
use serde::Deserialize;
use simulation_minigame::g1::{
    static_startup::{StaticStartupCommand, StaticStartupStep},
    static_transfer::StaticVisualTransferGoal,
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
    chunks: PathBuf,
    chunks_sha256: String,
    goal: PathBuf,
    goal_sha256: String,
    #[serde(default)]
    placement: Option<PathBuf>,
    #[serde(default)]
    placement_sha256: Option<String>,
    output: PathBuf,
}
fn wall() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
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
                // Retain actual counters even if a post-integration guard
                // rejects the last boundary before emitting a complete step.
                writeln!(file,"{}",serde_json::json!({"failed_owner_snapshot":true,"target_tick":target,"phase":format!("{:?}",s.phase),"reason":s.reason,"timing":s.timing})).map_err(|e|e.to_string())?;
                file.flush().map_err(|e| e.to_string())?;
                return Err(s.reason.clone().unwrap_or("static owner failed".into()));
            }
            if s.phase == G1WorkerPhase::Paused && s.timing.episode_integrations == target {
                return Ok(s);
            }
        }
        if Instant::now() > end {
            return Err("bounded static owner wait expired".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
#[test]
#[ignore = "one offline saved-original140Tick prefix plus250classical50HzTicks;0freshcamera/model calls"]
fn saved_visual_static_transfer_in_original_world() -> Result<(), String> {
    run(false, false)
}
#[test]
#[ignore = "one offline390Tick saved prefix plus325bounded memory-placement50HzTicks;0freshcamera/model calls"]
fn saved_memory_static_place_in_original_world() -> Result<(), String> {
    run(true, false)
}
#[test]
#[ignore = "one fixed-input native4PGS/predictive comparator against16diagnostic;715Ticklimit,0freshcamera/models;no parameter grid"]
fn saved_memory_static_place_with_native_four_passes() -> Result<(), String> {
    run(true, true)
}
fn run(place: bool, native_four_passes: bool) -> Result<(), String> {
    let p = PathBuf::from(std::env::var("G1_STATIC_TRANSFER_FIXTURE").map_err(|e| e.to_string())?);
    let h = std::env::var("G1_STATIC_TRANSFER_FIXTURE_SHA256").map_err(|e| e.to_string())?;
    let bytes = bound_bytes(&p, &h).map_err(|e| e.to_string())?;
    if bytes.len() > 16384 {
        return Err("fixture too large".into());
    }
    let f: Fixture = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_slice(
        &bound_bytes(&f.configuration, &f.configuration_sha256).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut config: ArenaTaskRunnerConfig =
        serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
    let mut chunks: Vec<PolicyActionChunk> = serde_json::from_slice(
        &bound_bytes(&f.chunks, &f.chunks_sha256).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut goal: StaticVisualTransferGoal =
        serde_json::from_slice(&bound_bytes(&f.goal, &f.goal_sha256).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    goal.validate().map_err(|e| e.to_string())?;
    let mut placement: Option<simulation_minigame::g1::static_place::StaticMemoryPlaceGoal> =
        if place {
            let path = f.placement.as_ref().ok_or("missing placement fixture")?;
            let hash = f
                .placement_sha256
                .as_ref()
                .ok_or("missing placement fixture hash")?;
            Some(
                serde_json::from_slice(&bound_bytes(path, hash).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
    if let Some(p) = &placement {
        p.validate().map_err(|e| e.to_string())?;
    }
    if chunks.len() != 2
        || value["static_startup"] != true
        || value["predictive_limit_diagnostic"] != true
        || value["diagnostic_constraint_sweeps"] != 16
    {
        return Err("changed original fixture profile".into());
    }
    for (i, c) in chunks.iter().enumerate() {
        if c.profile != task_minigame::types::TaskProfile::StaticApple
            || c.observation.episode_id != config.body.episode_id()
            || c.sequence_id != i as u64 + 1
            || c.frames.len() != 40
            || c.observation.sim_time_ns != (60 + i as u64 * 40) * 20_000_000
        {
            return Err("changed original action identity".into());
        }
    }
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let identity: G1StationConfiguration =
        serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::StaticAgile(body) = &mut config.body else {
        return Err("foreign body".into());
    };
    let (_, environment, _) = prepare(&scene, &identity, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let episode = config.body.episode_id();
    let worker = if native_four_passes {
        StaticStartupWorker::spawn_static_native_four_passes_comparison(config)
    } else {
        StaticStartupWorker::spawn_static_startup(config)
    }
    .map_err(|e| e.to_string())?;
    let trace = worker.subscribe_steps(512).map_err(|e| e.to_string())?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&f.output)
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
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.observation.captured_at_unix_ms = wall();
        let target = 100 + i as u64 * 40;
        worker
            .submit(TimedCommand {
                episode_id: episode,
                valid_until_sim_ns: target * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(3),
                command: StaticStartupCommand::OriginalVla(ArenaTaskCommand {
                    chunk: Arc::new(chunk.clone()),
                    scheduled_start_sim_ns: None,
                }),
            })
            .map_err(|e| e.to_string())?;
        wait(&worker, &trace, &mut file, target)?;
    }
    goal.observation.captured_at_unix_ms = wall();
    worker
        .submit(TimedCommand {
            episode_id: episode,
            valid_until_sim_ns: 390 * 20_000_000,
            valid_until_wall: Instant::now() + Duration::from_secs(7),
            command: StaticStartupCommand::VisualTransfer(goal),
        })
        .map_err(|e| e.to_string())?;
    let mut final_state = wait(&worker, &trace, &mut file, 390)?;
    if let Some(p) = placement.as_mut() {
        // Offline saved-image experiment only. Preserve the disclosed memory
        // age; this is not a newly captured image or an autonomous trial.
        let age =
            p.current_observation.captured_at_unix_ms - p.origin_observation.captured_at_unix_ms;
        p.current_observation.captured_at_unix_ms = wall();
        p.origin_observation.captured_at_unix_ms = p.current_observation.captured_at_unix_ms - age;
        worker
            .submit(TimedCommand {
                episode_id: episode,
                valid_until_sim_ns: 715 * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(9),
                command: StaticStartupCommand::MemoryPlace(p.clone()),
            })
            .map_err(|e| e.to_string())?;
        final_state = wait(&worker, &trace, &mut file, 715)?;
    }
    assert_eq!(trace.dropped_records(), 0);
    assert_eq!(
        final_state.timing.total_integrations,
        if place { 715 } else { 390 }
    );
    let last = final_state
        .step
        .as_ref()
        .ok_or("final static step absent")?;
    let simulation_minigame::g1::task_runner::ArenaBodyStep::StaticAgile(body) = &last.body else {
        return Err("final body is foreign".into());
    };
    assert_eq!(
        body.step_configuration.num_internal_pgs_iterations,
        if native_four_passes { 4 } else { 16 }
    );
    assert_eq!(body.step_configuration.dt, 0.02);
    Ok(())
}
