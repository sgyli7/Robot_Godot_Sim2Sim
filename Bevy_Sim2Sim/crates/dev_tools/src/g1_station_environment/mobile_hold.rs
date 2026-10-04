//! Saved real station grasp followed by the existing self-state grip/hold.
//! Offline mechanics only: no new RGB, policy service or task admission.

use super::{G1StationConfiguration, prepare};
use rendering_minigame::StationScene;
use robot_minigame::g1::{definition::G1Definition, policy::bound_bytes};
use simulation_minigame::g1::{
    mobile_grip::MobileGripCalibration,
    mobile_hold::{MobileGripHolding, MobileHoldGoal},
    mobile_navigation::{
        MobileCarryGoal, MobileCarryNavigator, MobileScanGoal, MobileScanNavigator,
    },
    mobile_release::{MobileGripRelease, MobileReleaseGoal},
    mobile_thumb::{MobileThumbGoal, MobileThumbPreparation},
    runner::{G1Runner, G1Step},
    task_objects::TaskObjectKind,
    task_policy::{ArenaControllerCommand, controller_command},
    task_runner::{ArenaTaskBodyConfig, ArenaTaskRunnerConfig},
};
use std::{collections::HashSet, fs, io::Write, path::PathBuf};
use task_minigame::{
    policy::{PolicyActionChunk, profile_contract},
    types::{ObservationStamp, TaskProfile},
};

fn read(name: &str) -> Result<Vec<u8>, String> {
    let path = PathBuf::from(std::env::var(name).map_err(|e| e.to_string())?);
    let hash = std::env::var(format!("{name}_SHA256")).map_err(|e| e.to_string())?;
    let bytes = bound_bytes(&path, &hash).map_err(|e| e.to_string())?;
    if bytes.len() > 1_048_576 {
        return Err("saved station hold fixture exceeds bound".into());
    }
    Ok(bytes)
}

fn hand_support(step: &G1Step, hands: &HashSet<usize>) -> Result<bool, String> {
    let object = step
        .task_objects
        .as_ref()
        .and_then(|objects| {
            objects
                .objects
                .iter()
                .find(|o| o.kind == TaskObjectKind::BrownBox)
        })
        .ok_or("box audit sample absent")?;
    let active: Vec<_> = object
        .last_solve_contacts
        .iter()
        .filter(|c| c.active_solver_normal_impulse_n_s.is_some_and(|v| v > 1e-7))
        .collect();
    Ok(!active.is_empty()
        && active
            .iter()
            .all(|c| c.other_robot_body_index.is_some_and(|i| hands.contains(&i))))
}

fn check_clock(step: &G1Step, tick: u64) -> Result<(), String> {
    let cfg = &step.step_configuration;
    if step.integration_count != tick
        || step.torque_update_count != tick
        || step.inference.inference_count != tick
        || cfg.physics_hz != 50
        || cfg.dt != 0.02
        || cfg.num_solver_iterations != 1
        || cfg.num_internal_pgs_iterations != 4
        || cfg.max_ccd_substeps != 1
        || !step
            .native_static_environment
            .as_ref()
            .is_some_and(|s| s.collider_count == 2553 && s.original_broad_floor_removed)
        || step.root_upright_cosine <= 0.95
        || step.root_position_source[2] < 0.35
    {
        return Err(format!(
            "station hold clock/stand guard failed at Tick{tick}"
        ));
    }
    Ok(())
}

#[test]
#[ignore = "saved actual station four50frame chunks, then one original grip calibration and finite hold; maximum450 single50Hz steps,0newRGB/VLA/Qwen"]
fn saved_station_grasp_self_state_hold() -> Result<(), String> {
    saved_station_hold_and_carry(false, false, false, false)
}

#[test]
#[ignore = "saved station grasp/hold then unchanged traditional2m public-clear-aisle carry and stop; maximum2300 actual50Hz steps,0freshRGB/VLA/Qwen; not target-bin qualification"]
fn saved_station_grasp_two_metre_carry_stop() -> Result<(), String> {
    saved_station_hold_and_carry(true, false, false, false)
}

#[test]
#[ignore = "saved actual station grasp/100Tick hold then source-positive RGB-derived scan/carry/hold/release goals;3300 single50Hz steps maximum,0freshRGB/VLA/Qwen;mechanics only"]
fn saved_station_grasp_source_positive_bin_route() -> Result<(), String> {
    saved_station_hold_and_carry(false, true, false, false)
}

#[test]
#[ignore = "saved actual1982Tick station blocked prefix then finite zero-navigation hold and disclosed manual release;maximum2457steps,0freshRGB/VLA/Qwen;mechanical diagnosis only"]
fn saved_station_blocked_bin_hold_release() -> Result<(), String> {
    saved_station_hold_and_carry(false, false, true, false)
}

#[test]
#[ignore = "saved actual2182Tick station prefix; one bounded left-thumb motor ramp/settle then manual release;2507steps,0freshRGB/VLA/Qwen;mechanical diagnosis only"]
fn saved_station_blocked_bin_thumb_clearance_release() -> Result<(), String> {
    saved_station_hold_and_carry(false, false, true, true)
}

fn saved_station_hold_and_carry(
    run_carry: bool,
    run_placement: bool,
    run_blocked: bool,
    run_thumb: bool,
) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_slice(&read("G1_STATION_HOLD_CONFIG")?).map_err(|e| e.to_string())?;
    let config: ArenaTaskRunnerConfig =
        serde_json::from_value(value["runner"].clone()).map_err(|e| e.to_string())?;
    let identity: G1StationConfiguration =
        serde_json::from_value(value["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::MobileHomieV2(mut body) = config.body else {
        return Err("saved station hold requires original Homie_v2".into());
    };
    let chunks: Vec<PolicyActionChunk> =
        serde_json::from_slice(&read("G1_STATION_HOLD_CHUNKS")?).map_err(|e| e.to_string())?;
    let contract = profile_contract(TaskProfile::MobileBox);
    if chunks.len() != 4
        || chunks.iter().enumerate().any(|(i, c)| {
            c.profile != TaskProfile::MobileBox
                || c.model_revision != contract.revision
                || c.frames.len() != 50
                || c.action_period_ns != 20_000_000
                || c.sequence_id != i as u64 + 1
                || c.observation.episode_id != body.episode_id
                || c.observation.sim_time_ns != i as u64 * 1_000_000_000
        })
    {
        return Err("saved station fixture changed original profile/actions/timeline".into());
    }
    // Validate all source action frames before loading an inference session.
    let mut commands = Vec::with_capacity(200);
    for chunk in &chunks {
        for frame in &chunk.frames {
            let ArenaControllerCommand::MobileHomieV2(command) =
                controller_command(TaskProfile::MobileBox, frame, &config.limits)
                    .map_err(|e| format!("{e:?}"))?
                    .controller
            else {
                return Err("saved mobile frame decoded to a static controller".into());
            };
            commands.push(command);
        }
    }
    let definition =
        G1Definition::load(&body.definition, &body.definition_sha256).map_err(|e| e.to_string())?;
    let hands: HashSet<_> = definition
        .model()
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.name.contains("_hand_"))
        .map(|(i, _)| i)
        .collect();
    let calibration = MobileGripCalibration::new(&definition).map_err(|e| e.to_string())?;
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, environment, preparation) = prepare(&scene, &identity, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let mut owner =
        G1Runner::load_mobile_station_fixture_diagnostic(&body).map_err(|e| e.to_string())?;
    let path = PathBuf::from(std::env::var("G1_STATION_HOLD_OUTPUT").map_err(|e| e.to_string())?);
    let mut trace = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    writeln!(
        trace,
        "{}",
        serde_json::json!({"phase":"initialization","preparation":preparation,
        "offline_saved_fixture":true,"fresh_images":0,"fresh_vla_calls":0,"qwen_calls":0,
        "real_time_qualified":false,"task_qualified":false})
    )
    .map_err(|e| e.to_string())?;
    for (i, command) in commands.iter().enumerate() {
        let step = owner.step(command).map_err(|e| e.to_string())?;
        writeln!(
            trace,
            "{}",
            serde_json::json!({"phase":"saved_actual_station_grasp",
            "command":command,"body":{"mobile_homie_v2":step}})
        )
        .map_err(|e| e.to_string())?;
        check_clock(&step, i as u64 + 1)?;
        if i == 199 && !hand_support(&step, &hands)? {
            return Err("saved station grasp did not retain active hand-only support at200".into());
        }
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let correction = calibration
        .correct(&state, commands.last().unwrap())
        .map_err(|e| e.to_string())?;
    let mut command = correction.command;
    command.navigation = [0.; 3];
    // This synthetic identity exists only inside the explicitly offline fixture;
    // no old camera result is re-stamped or submitted to a runtime task queue.
    let mut hold = MobileGripHolding::new(
        MobileHoldGoal {
            observation: ObservationStamp {
                episode_id: state.episode_id,
                frame_id: 5,
                sim_time_ns: state.sim_time_ns,
                captured_at_unix_ms: 1,
            },
        },
        &state,
        command,
    )
    .map_err(|e| e.to_string())?;
    for index in 0..250 {
        let holding = hold
            .update(&owner.measurement().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let step = owner.step(&holding.command).map_err(|e| e.to_string())?;
        let supported = hand_support(&step, &hands)?;
        writeln!(
            trace,
            "{}",
            serde_json::json!({"phase":"existing_self_state_calibrated_hold",
            "body":{"mobile_homie_v2":step},"holding":holding,"grip":correction.receipt,
            "independent_active_hand_only_support":supported,
            "synthetic_offline_skill_identity":true,"autonomous_execution":false})
        )
        .map_err(|e| e.to_string())?;
        trace.flush().map_err(|e| e.to_string())?;
        check_clock(&step, 201 + index)?;
        if !supported {
            return Err(format!(
                "station calibrated hold lost active hand-only support atTick{}",
                201 + index
            ));
        }
        if holding.completed {
            if !run_carry && !run_placement && !run_blocked {
                return Ok(());
            }
            break;
        }
    }
    if !hold.completed() {
        return Err("station calibrated hold exceeded250Tick bound".into());
    }
    if run_blocked {
        return saved_blocked_bin_stop(&mut owner, &calibration, &hands, &mut trace, run_thumb);
    }
    if run_placement {
        return saved_bin_route(
            &mut owner,
            hold.command().clone(),
            &calibration,
            &hands,
            &mut trace,
        );
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut transport = hold.command().clone();
    // Fixed public-court aisle, opposite the shelf's target table. This isolates
    // held locomotion and stop; it is not a guessed bin location or task success.
    let mut navigator = MobileCarryNavigator::new(
        MobileCarryGoal {
            observation: ObservationStamp {
                episode_id: state.episode_id,
                frame_id: 6,
                sim_time_ns: state.sim_time_ns,
                captured_at_unix_ms: 1,
            },
            heading_yaw_source_rad: std::f32::consts::FRAC_PI_2,
            relative_distance_m: 2.,
        },
        &state,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..1850 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let navigation = navigator.update(&state).map_err(|e| e.to_string())?;
        transport.navigation = navigation.navigation;
        let step = owner.step(&transport).map_err(|e| e.to_string())?;
        let supported = hand_support(&step, &hands)?;
        writeln!(trace, "{}", serde_json::json!({
            "phase":"existing_self_state_public_aisle_carry_stop",
            "body":{"mobile_homie_v2":step},"navigation":navigation,"command":transport,
            "independent_active_hand_only_support":supported,"synthetic_offline_skill_identity":true,
            "target_bin_task":false,"autonomous_execution":false,"fresh_RGB_VLA_Qwen":0,
        })).map_err(|e| e.to_string())?;
        trace.flush().map_err(|e| e.to_string())?;
        check_clock(&step, state.source_tick + 1)?;
        if !supported {
            return Err(format!(
                "station carry lost active hand-only support atTick{} during{:?}",
                state.source_tick + 1,
                navigation.phase
            ));
        }
        if navigation.completed {
            return Ok(());
        }
    }
    Err("station public-aisle carry exceeded1850Tick bound".into())
}

fn saved_blocked_bin_stop(
    owner: &mut G1Runner,
    calibration: &MobileGripCalibration,
    hands: &HashSet<usize>,
    trace: &mut fs::File,
    run_thumb: bool,
) -> Result<(), String> {
    use robot_minigame::g1::contract::G1Command;
    let commands: Vec<G1Command> =
        serde_json::from_slice(&read("G1_STATION_BLOCKED_COMMANDS")?).map_err(|e| e.to_string())?;
    let saved_count = if run_thumb { 1882 } else { 1682 };
    if commands.len() != saved_count
        || owner.measurement().map_err(|e| e.to_string())?.source_tick != 300
    {
        return Err("blocked station diagnostic has an unexpected saved prefix length".into());
    }
    let mut command = commands.last().unwrap().clone();
    for saved in commands {
        let step = owner.step(&saved).map_err(|e| e.to_string())?;
        write_route_step(
            trace,
            &step,
            "saved_actual_blocked_station_prefix",
            &saved,
            true,
            hands,
        )?;
    }
    command.navigation = [0.; 3];
    if run_thumb {
        // Single pose selected by offline original-hull/current-self-FK geometry.
        // These are physical joint targets, never written body poses. Existing
        // release preserves all fingers; the live path does not call this probe.
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let mut thumb = MobileThumbPreparation::new(
            MobileThumbGoal {
                observation: ObservationStamp {
                    episode_id: state.episode_id,
                    frame_id: 90,
                    sim_time_ns: state.sim_time_ns,
                    captured_at_unix_ms: 1,
                },
            },
            &state,
            command,
        )
        .map_err(|e| e.to_string())?;
        for _ in 0..100 {
            let state = owner.measurement().map_err(|e| e.to_string())?;
            let preparing = thumb.update(&state).map_err(|e| e.to_string())?;
            let step = owner.step(&preparing.command).map_err(|e| e.to_string())?;
            write_route_step(
                trace,
                &step,
                "offline_bounded_left_thumb_motor_ramp_settle",
                &preparing,
                true,
                hands,
            )?;
        }
        if !thumb.completed() {
            return Err("thumb preparation exceeded its finite100Tick bound".into());
        }
        return saved_manual_release(owner, calibration, hands, trace, thumb.command().clone());
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut hold = MobileGripHolding::new(
        MobileHoldGoal {
            observation: ObservationStamp {
                episode_id: state.episode_id,
                frame_id: 90,
                sim_time_ns: state.sim_time_ns,
                captured_at_unix_ms: 1,
            },
        },
        &state,
        command,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..250 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let holding = hold.update(&state).map_err(|e| e.to_string())?;
        let step = owner.step(&holding.command).map_err(|e| e.to_string())?;
        write_route_step(
            trace,
            &step,
            "offline_blocked_zero_navigation_hold",
            &holding,
            true,
            hands,
        )?;
        if holding.completed {
            break;
        }
    }
    if !hold.completed() {
        return Err("blocked zero-navigation hold exceeded250Ticks".into());
    }
    saved_manual_release(owner, calibration, hands, trace, hold.command().clone())
}

fn saved_manual_release(
    owner: &mut G1Runner,
    calibration: &MobileGripCalibration,
    hands: &HashSet<usize>,
    trace: &mut fs::File,
    command: robot_minigame::g1::contract::G1Command,
) -> Result<(), String> {
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut release = MobileGripRelease::new(
        MobileReleaseGoal {
            observation: ObservationStamp {
                episode_id: state.episode_id,
                frame_id: 91,
                sim_time_ns: state.sim_time_ns,
                captured_at_unix_ms: 1,
            },
            target_palm_gap_m: 0.35,
            duration_ticks: 100,
        },
        &state,
        command,
        calibration,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..225 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let opening = release
            .update(&state, calibration)
            .map_err(|e| e.to_string())?;
        let step = owner.step(&opening.command).map_err(|e| e.to_string())?;
        write_route_step(
            trace,
            &step,
            "offline_disclosed_manual_release_after_blocked_hold",
            &opening,
            false,
            hands,
        )?;
        if opening.completed {
            return Ok(());
        }
    }
    Err("blocked diagnostic manual release exceeded225Ticks".into())
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedBinRoute {
    source_receipt_sha256: String,
    source_profile: String,
    scan_heading: f32,
    carries: Vec<[f32; 2]>,
    release_gap: f64,
    release_ticks: u32,
}

fn saved_bin_route(
    owner: &mut G1Runner,
    mut command: robot_minigame::g1::contract::G1Command,
    calibration: &MobileGripCalibration,
    hands: &HashSet<usize>,
    trace: &mut fs::File,
) -> Result<(), String> {
    let program: SavedBinRoute =
        serde_json::from_slice(&read("G1_STATION_BIN_PROGRAM")?).map_err(|e| e.to_string())?;
    if program.source_profile != "original_source_positive20620_RGB_goals_not_station_live"
        || program.source_receipt_sha256.len() != 64
        || program.scan_heading != -std::f32::consts::FRAC_PI_2
        || program.carries.len() != 5
        || program.release_gap != 0.35
        || program.release_ticks != 100
    {
        return Err("saved bin program must preserve the frozen source-positive route".into());
    }
    let stamp =
        |state: &simulation_minigame::g1::runner::G1Measurement, frame_id| ObservationStamp {
            episode_id: state.episode_id,
            frame_id,
            sim_time_ns: state.sim_time_ns,
            captured_at_unix_ms: 1,
        };
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut scan = MobileScanNavigator::new(
        MobileScanGoal {
            observation: stamp(&state, 6),
            heading_yaw_source_rad: program.scan_heading,
        },
        &state,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..800 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let navigation = scan.update(&state).map_err(|e| e.to_string())?;
        command.navigation = navigation.navigation;
        let step = owner.step(&command).map_err(|e| e.to_string())?;
        write_route_step(
            trace,
            &step,
            "source_positive_scan",
            &navigation,
            true,
            hands,
        )?;
        if navigation.completed {
            break;
        }
    }
    if !scan.completed() {
        return Err("bounded saved station scan did not complete".into());
    }
    for (i, [heading, distance]) in program.carries.into_iter().enumerate() {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let mut carry = MobileCarryNavigator::new(
            MobileCarryGoal {
                observation: stamp(&state, 7 + i as u64),
                heading_yaw_source_rad: heading,
                relative_distance_m: distance,
            },
            &state,
        )
        .map_err(|e| e.to_string())?;
        for _ in 0..900 {
            let state = owner.measurement().map_err(|e| e.to_string())?;
            if state.source_tick >= 2850 {
                return Err("saved station bin route exhausted finite bound".into());
            }
            let navigation = carry.update(&state).map_err(|e| e.to_string())?;
            command.navigation = navigation.navigation;
            let step = owner.step(&command).map_err(|e| e.to_string())?;
            write_route_step(
                trace,
                &step,
                "source_positive_bin_carry",
                &navigation,
                true,
                hands,
            )?;
            if navigation.completed {
                break;
            }
        }
        if !carry.completed() {
            return Err("saved bounded bin carry did not complete".into());
        }
    }
    command.navigation = [0.; 3];
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut hold = MobileGripHolding::new(
        MobileHoldGoal {
            observation: stamp(&state, 12),
        },
        &state,
        command,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..250 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let holding = hold.update(&state).map_err(|e| e.to_string())?;
        let step = owner.step(&holding.command).map_err(|e| e.to_string())?;
        write_route_step(
            trace,
            &step,
            "source_positive_prerelease_hold",
            &holding,
            true,
            hands,
        )?;
        if holding.completed {
            break;
        }
    }
    if !hold.completed() {
        return Err("saved station prerelease hold did not complete".into());
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut release = MobileGripRelease::new(
        MobileReleaseGoal {
            observation: stamp(&state, 13),
            target_palm_gap_m: program.release_gap,
            duration_ticks: program.release_ticks,
        },
        &state,
        hold.command().clone(),
        calibration,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..225 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let opening = release
            .update(&state, calibration)
            .map_err(|e| e.to_string())?;
        let step = owner.step(&opening.command).map_err(|e| e.to_string())?;
        write_route_step(
            trace,
            &step,
            "source_positive_open_and_settle",
            &opening,
            false,
            hands,
        )?;
        if opening.completed {
            return Ok(());
        }
    }
    Err("saved station finite release did not complete".into())
}

fn write_route_step<T: serde::Serialize>(
    trace: &mut fs::File,
    step: &G1Step,
    phase: &str,
    receipt: &T,
    require_hand_support: bool,
    hands: &HashSet<usize>,
) -> Result<(), String> {
    let supported = hand_support(step, hands)?;
    writeln!(trace, "{}", serde_json::json!({"phase":phase,
        "body":{"mobile_homie_v2":step},"execution":receipt,
        "independent_active_hand_only_support":supported,"synthetic_offline_skill_identity":true,
        "saved_commands_or_source_goals_not_fresh_station_RGB":true,"fresh_RGB_VLA_Qwen":0,"task_qualified":false
    })).map_err(|e| e.to_string())?;
    trace.flush().map_err(|e| e.to_string())?;
    check_clock(step, step.integration_count)?;
    if require_hand_support && !supported {
        return Err(format!(
            "saved station bin route lost hand-only support atTick{}",
            step.integration_count
        ));
    }
    if step.integration_count > 3300 {
        return Err("saved station bin route exceeded3300Ticks".into());
    }
    Ok(())
}
