//! One initial-position counterfactual; cached actions are disclosed diagnostics.
//! The unchanged replay must already prove equality with the original failure.

use super::*;
use simulation_minigame::g1::task_objects::TaskObjectKind;

#[test]
#[ignore = "one frozen-commands CPU50Hz counterfactual: initial box30mm nearer;1870steps,0freshVLA/Qwen/RGB; never task qualification"]
fn original_commands_initial_box_grasp_depth_counterfactual() -> Result<(), String> {
    let path = PathBuf::from(std::env::var("G1_MOTOR_REPLAY_FIXTURE").map_err(|e| e.to_string())?);
    let hash = std::env::var("G1_MOTOR_REPLAY_FIXTURE_SHA256").map_err(|e| e.to_string())?;
    let fixture: Fixture =
        serde_json::from_slice(&read(&path, &hash, 1_048_576)?).map_err(|e| e.to_string())?;
    if fixture.schema != "g1_recorded_station_motor_effort_fixture_v1"
        || fixture.stop_tick != 1870
        || fixture.window_start_tick != 1840
        || fixture.chunks.len() != 4
    {
        return Err("grasp-depth counterfactual changed frozen replay scope".into());
    }
    let trace = read(&fixture.trace, &fixture.trace_sha256, 64 * 1024 * 1024)?;
    let rows: Vec<Value> = trace
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(serde_json::from_slice)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let config: ArenaTaskRunnerConfig =
        serde_json::from_value(fixture.config["runner"].clone()).map_err(|e| e.to_string())?;
    let station: G1StationConfiguration =
        serde_json::from_value(fixture.config["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::MobileHomieV2(mut body) = config.body else {
        return Err("counterfactual requires original mobile G1".into());
    };
    if rows.len() != 1971 {
        return Err("counterfactual requires the complete original1971Tick partial trace".into());
    }
    let mut commands = Vec::with_capacity(1870);
    for (index, chunk) in fixture.chunks.iter().enumerate() {
        if chunk.profile != TaskProfile::MobileBox
            || chunk.sequence_id != index as u64 + 1
            || chunk.frames.len() != 50
            || chunk.action_period_ns != 20_000_000
            || chunk.observation.episode_id != body.episode_id
            || chunk.observation.sim_time_ns != index as u64 * 1_000_000_000
        {
            return Err("counterfactual changed original chunk identity or50Hz time".into());
        }
        for frame in &chunk.frames {
            let ArenaControllerCommand::MobileHomieV2(command) =
                controller_command(TaskProfile::MobileBox, frame, &config.limits)
                    .map_err(|e| format!("{e:?}"))?
                    .controller
            else {
                return Err("counterfactual decoded a foreign controller".into());
            };
            commands.push(command);
        }
    }
    for (index, row) in rows.iter().take(1870).enumerate() {
        if row["episode_id"] != body.episode_id
            || row["owner_episode_integrations"] != index as u64 + 1
        {
            return Err("counterfactual source trace has reset/gap/foreign episode".into());
        }
        if index >= 200 {
            commands.push(command(row)?);
        }
    }
    let original_configuration = fixture.config["runner"]["body"]["mobile_homie_v2"].clone();
    let objects = body
        .task_objects
        .as_mut()
        .ok_or("original task scene absent")?;
    let placement = objects
        .placements
        .iter_mut()
        .find(|placement| placement.kind == TaskObjectKind::BrownBox)
        .ok_or("original box placement absent")?;
    let before = placement.root_pose.position;
    placement.root_pose.position[0] -= 0.030;
    let after = placement.root_pose.position;
    let mut changed_configuration = original_configuration.clone();
    let changed_placement = changed_configuration["task_objects"]["placements"]
        .as_array_mut()
        .ok_or("original JSON placements absent")?
        .iter_mut()
        .find(|placement| placement["kind"] == "t2_box")
        .ok_or("original JSON box absent")?;
    changed_placement["root_pose"]["position"] = json!(after);
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, environment, preparation) = prepare(&scene, &station, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let output = PathBuf::from(std::env::var("G1_MOTOR_REPLAY_OUTPUT").map_err(|e| e.to_string())?);
    let mut evidence = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|e| e.to_string())?;
    let mut owner =
        G1Runner::load_mobile_station_fixture_diagnostic(&body).map_err(|e| e.to_string())?;
    writeln!(evidence,"{}",json!({"event":"grasp_depth_counterfactual_start",
        "fixture_sha256":hash,"original_trace_sha256":fixture.trace_sha256,
        "initial_box_before_source_m":before,"initial_box_after_source_m":after,
        "original_body_configuration":original_configuration,"counterfactual_body_configuration":changed_configuration,
        "preparation":preparation,"before_counts":owner.progress_counts(),
        "fixed_commands_not_fresh_visual_closed_loop":true,"new_VLA_Qwen_RGB":0,
        "task_qualified":false,"real_time_qualified":false})).map_err(|e|e.to_string())?;
    for (index, command) in commands.iter().enumerate() {
        let step = owner.step(command).map_err(|e| e.to_string())?;
        writeln!(
            evidence,
            "{}",
            json!({"event":"counterfactual_physical_step",
            "tick":index+1,"body":journal_body_value(&step)?,"command":command,
            "task_qualified":false})
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        evidence,
        "{}",
        json!({"event":"grasp_depth_counterfactual_complete",
        "actual_counts":owner.progress_counts(),"new_VLA_Qwen_RGB":0,
        "task_success":false,"task_qualified":false,"real_time_qualified":false})
    )
    .map_err(|e| e.to_string())?;
    evidence.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}
