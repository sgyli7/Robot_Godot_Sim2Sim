//! One mechanical probe of bounded insertion; no fresh image or task claim.

use super::*;
use robot_minigame::g1::definition::G1Definition;
use simulation_minigame::g1::{
    mobile_center::{MobileCenterGoal, MobileGripCentering},
    mobile_grip::MobileGripCalibration,
};
use task_minigame::types::ObservationStamp;

#[test]
#[ignore = "one CPU50Hz probe <=350steps; archived RGB300 numeric offset is synthetic200Tick admission;0freshVLA/Qwen/RGB; never qualification"]
fn recorded_pregrasp_bounded_insertion_mechanics() -> Result<(), String> {
    recorded_probe(false, false)
}

#[test]
#[ignore = "one distinct CPU50Hz probe <=400steps; open original palms before fixed insertion;0freshVLA/Qwen/RGB; never qualification"]
fn recorded_open_then_insertion_mechanics() -> Result<(), String> {
    recorded_probe(true, false)
}

#[test]
#[ignore = "one distinct CPU50Hz probe <=450steps; same300Tick open-insertion prefix then50 bounded close/100hold;0freshVLA/Qwen/RGB; never qualification"]
fn recorded_open_insert_bounded_close_mechanics() -> Result<(), String> {
    recorded_probe(true, true)
}

fn recorded_probe(opening_first: bool, bounded_close: bool) -> Result<(), String> {
    let path = PathBuf::from(std::env::var("G1_MOTOR_REPLAY_FIXTURE").map_err(|e| e.to_string())?);
    let hash = std::env::var("G1_MOTOR_REPLAY_FIXTURE_SHA256").map_err(|e| e.to_string())?;
    let fixture: Fixture =
        serde_json::from_slice(&read(&path, &hash, 1_048_576)?).map_err(|e| e.to_string())?;
    if fixture.schema != "g1_recorded_station_motor_effort_fixture_v1"
        || fixture.chunks.len() != 4
        || fixture.stop_tick != 1870
    {
        return Err("insertion probe requires frozen original fixture".into());
    }
    let bytes = read(&fixture.trace, &fixture.trace_sha256, 64 * 1024 * 1024)?;
    let rows: Vec<Value> = bytes
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(serde_json::from_slice)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let config: ArenaTaskRunnerConfig =
        serde_json::from_value(fixture.config["runner"].clone()).map_err(|e| e.to_string())?;
    let station: G1StationConfiguration =
        serde_json::from_value(fixture.config["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::MobileHomieV2(mut body) = config.body else {
        return Err("insertion requires original G1 body".into());
    };
    let definition =
        G1Definition::load(&body.definition, &body.definition_sha256).map_err(|e| e.to_string())?;
    let calibration = MobileGripCalibration::new(&definition).map_err(|e| e.to_string())?;
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, environment, preparation) = prepare(&scene, &station, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let output = PathBuf::from(std::env::var("G1_MOTOR_REPLAY_OUTPUT").map_err(|e| e.to_string())?);
    let mut evidence = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let mut owner =
        G1Runner::load_mobile_station_fixture_diagnostic(&body).map_err(|e| e.to_string())?;
    writeln!(evidence,"{}",json!({"event":"insertion_probe_start","fixture_sha256":hash,
        "original_scene_unchanged":true,"preparation":preparation,"before_counts":owner.progress_counts(),
        "fresh_RGB_N16_Qwen":0,"synthetic_goal_admission":true,"task_qualified":false,
        "opening_before_insertion":opening_first,"reacquisition_motion_is_unadmitted_diagnostic_only":true,
        "actual_visual_closed_loop":false,"close_without_fresh_image_is_diagnostic_only":true}))
        .map_err(|e|e.to_string())?;
    let mut last = None;
    let mut index = 0;
    for (sequence, chunk) in fixture.chunks.iter().enumerate() {
        if chunk.profile != TaskProfile::MobileBox
            || chunk.sequence_id != sequence as u64 + 1
            || chunk.frames.len() != 50
            || chunk.action_period_ns != 20_000_000
            || chunk.observation.episode_id != body.episode_id
            || chunk.observation.sim_time_ns != sequence as u64 * 1_000_000_000
        {
            return Err("insertion changed source chunk identity or time".into());
        }
        for frame in &chunk.frames {
            let ArenaControllerCommand::MobileHomieV2(command) =
                controller_command(TaskProfile::MobileBox, frame, &config.limits)
                    .map_err(|e| format!("{e:?}"))?
                    .controller
            else {
                return Err("insertion decoded foreign controller".into());
            };
            let step = owner.step(&command).map_err(|e| e.to_string())?;
            let mut actual = journal_body_value(&step)?;
            let mut recorded = rows[index]["body"]["mobile_homie_v2"].clone();
            remove_point_diagnostics(&mut actual);
            remove_point_diagnostics(&mut recorded);
            if actual != recorded {
                return Err(format!("insertion physical prefix differs at{}", index + 1));
            }
            writeln!(
                evidence,
                "{}",
                json!({"event":"insertion_physical_step","phase":"exact_original",
                "tick":index+1,"body":journal_body_value(&step)?,"command":command})
            )
            .map_err(|e| e.to_string())?;
            last = Some(command);
            index += 1;
        }
    }
    let original = last.ok_or("no original command")?;
    if opening_first {
        return opening_sequence(
            &mut owner,
            &calibration,
            original,
            &mut evidence,
            bounded_close,
        );
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    // Numeric proposal from archived actual RGB300, intentionally NOT current
    // RGB200. This seam tests mechanics only; native visual admission is open.
    let goal = MobileCenterGoal {
        observation: ObservationStamp {
            episode_id: state.episode_id,
            frame_id: 1,
            sim_time_ns: state.sim_time_ns,
            captured_at_unix_ms: 1,
        },
        common_offset_root_source_m: [
            0.04147769055522918,
            0.0017943696767093705,
            0.00008364434917117667,
        ],
        duration_ticks: 50,
    };
    let mut centering =
        MobileGripCentering::new(goal, &state, original.clone()).map_err(|e| e.to_string())?;
    for tick in 201..=250 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let update = centering
            .update(&state, &calibration)
            .map_err(|e| e.to_string())?;
        // A repeated state or reset must neither issue another increment nor
        // change the command/counters of the stage or the physical owner.
        let before = serde_json::to_value(centering.command()).map_err(|e| e.to_string())?;
        let counts = owner.progress_counts();
        if centering.update(&state, &calibration).is_ok() {
            return Err("repeated state advanced insertion".into());
        }
        let mut reset = state.clone();
        reset.episode_id += 1;
        if centering.update(&reset, &calibration).is_ok() {
            return Err("reset state advanced insertion".into());
        }
        if before != serde_json::to_value(centering.command()).map_err(|e| e.to_string())?
            || counts != owner.progress_counts()
        {
            return Err("rejected update mutated command/owner".into());
        }
        for slot in (22..29).chain(36..43) {
            let upper = slot - robot_minigame::g1::contract::LOWER_COUNT;
            if update.command.upper_positions[upper] != original.upper_positions[upper] {
                return Err("insertion changed original finger target".into());
            }
        }
        if update.command.navigation != [0.; 3] {
            return Err("insertion added navigation".into());
        }
        let step = owner.step(&update.command).map_err(|e| e.to_string())?;
        writeln!(evidence,"{}",json!({"event":"insertion_physical_step","phase":"bounded_insertion",
            "tick":tick,"body":journal_body_value(&step)?,"command":update.command,"centering":update}))
            .map_err(|e|e.to_string())?;
    }
    if !centering.completed() {
        return Err("insertion did not complete50updates".into());
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    if centering.update(&state, &calibration).is_ok() {
        return Err("completed insertion allowed another update".into());
    }
    let closing = calibration
        .correct(&state, centering.command())
        .map_err(|e| e.to_string())?;
    for tick in 251..=350 {
        let step = owner.step(&closing.command).map_err(|e| e.to_string())?;
        writeln!(
            evidence,
            "{}",
            json!({"event":"insertion_physical_step","phase":"diagnostic_close",
            "tick":tick,"body":journal_body_value(&step)?,"command":closing.command,
            "native_visual_close_admitted":false})
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        evidence,
        "{}",
        json!({"event":"insertion_probe_complete","actual_counts":owner.progress_counts(),
        "fresh_RGB_N16_Qwen":0,"actual_visual_closed_loop":false,"task_qualified":false})
    )
    .map_err(|e| e.to_string())?;
    evidence.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}

fn opening_sequence(
    owner: &mut G1Runner,
    calibration: &MobileGripCalibration,
    mut command: G1Command,
    evidence: &mut fs::File,
    bounded_close: bool,
) -> Result<(), String> {
    let original = command.clone();
    command.navigation = [0.; 3];
    let state = owner.measurement().map_err(|e| e.to_string())?;
    // Existing horizontal release interface supports a0.25..0.35m target.
    // This one diagnostic uses0.30m once, without an aperture search.
    let gap_increment = calibration
        .horizontal_opening_increment(&state, &command, 0.30, 50)
        .map_err(|e| e.to_string())?;
    for tick in 201..=300 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let (phase, correction) = if tick <= 250 {
            (
                "diagnostic_open",
                calibration.spread_horizontal(&state, &command, gap_increment),
            )
        } else {
            (
                "diagnostic_open_insertion",
                calibration.translate(
                    &state,
                    &command,
                    [
                        0.04147769055522918 / 50.,
                        0.0017943696767093705 / 50.,
                        0.00008364434917117667 / 50.,
                    ],
                ),
            )
        };
        let correction = correction.map_err(|e| e.to_string())?;
        command = correction.command;
        for slot in (22..29).chain(36..43) {
            let upper = slot - robot_minigame::g1::contract::LOWER_COUNT;
            if command.upper_positions[upper] != original.upper_positions[upper] {
                return Err("opening probe changed original finger targets".into());
            }
        }
        let step = owner.step(&command).map_err(|e| e.to_string())?;
        writeln!(evidence,"{}",json!({"event":"insertion_physical_step","phase":phase,
            "tick":tick,"body":journal_body_value(&step)?,"command":command,"correction":correction.receipt,
            "fresh_post_motion_image_required_in_product":true,"native_visual_close_admitted":false}))
            .map_err(|e|e.to_string())?;
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    if bounded_close {
        for tick in 301..=350 {
            let state = owner.measurement().map_err(|e| e.to_string())?;
            let correction = calibration
                .approach_source_gap(&state, &command, 351 - tick)
                .map_err(|e| e.to_string())?;
            command = correction.command;
            let step = owner.step(&command).map_err(|e| e.to_string())?;
            writeln!(
                evidence,
                "{}",
                json!({"event":"insertion_physical_step","phase":"diagnostic_bounded_close",
                "tick":tick,"body":journal_body_value(&step)?,"command":command,
                "correction":correction.receipt,"native_visual_close_admitted":false})
            )
            .map_err(|e| e.to_string())?;
        }
        for tick in 351..=450 {
            let step = owner.step(&command).map_err(|e| e.to_string())?;
            writeln!(
                evidence,
                "{}",
                json!({"event":"insertion_physical_step","phase":"diagnostic_post_close_hold",
                "tick":tick,"body":journal_body_value(&step)?,"command":command,
                "native_visual_close_admitted":false})
            )
            .map_err(|e| e.to_string())?;
        }
        writeln!(
            evidence,
            "{}",
            json!({"event":"insertion_probe_complete","actual_counts":owner.progress_counts(),
            "fresh_RGB_N16_Qwen":0,"actual_visual_closed_loop":false,"task_qualified":false,
            "opening_before_insertion":true,"bounded_source_gap_closing":true})
        )
        .map_err(|e| e.to_string())?;
        evidence.sync_all().map_err(|e| e.to_string())?;
        return Ok(());
    }
    let closing = calibration
        .correct(&state, &command)
        .map_err(|e| e.to_string())?;
    for tick in 301..=400 {
        let step = owner.step(&closing.command).map_err(|e| e.to_string())?;
        writeln!(
            evidence,
            "{}",
            json!({"event":"insertion_physical_step","phase":"diagnostic_open_close",
            "tick":tick,"body":journal_body_value(&step)?,"command":closing.command,
            "native_visual_close_admitted":false})
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        evidence,
        "{}",
        json!({"event":"insertion_probe_complete","actual_counts":owner.progress_counts(),
        "fresh_RGB_N16_Qwen":0,"actual_visual_closed_loop":false,"task_qualified":false,
        "opening_before_insertion":true})
    )
    .map_err(|e| e.to_string())?;
    evidence.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}
