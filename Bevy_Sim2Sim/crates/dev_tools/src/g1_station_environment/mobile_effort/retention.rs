//! One fixed-command mechanics horizon after reacquisition, not a vision task.

use super::*;
use robot_minigame::g1::definition::G1Definition;
use simulation_minigame::g1::{
    mobile_grip::MobileGripCalibration,
    mobile_raise::{MobileGripRaising, MobileRaiseGoal},
};
use task_minigame::types::ObservationStamp;

// The extra outer journal JSON pass can move the decoded f64 representation by
// one ULP. Compare the original G1Step's f32 physical fields by exact bits; keep
// integers/identities exact and exclude only the added point diagnostic field.
fn same_physical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            let keys = |m: &serde_json::Map<String, Value>| {
                m.keys()
                    .filter(|k| k.as_str() != "diagnostic_solver_contacts")
                    .count()
            };
            keys(a) == keys(b)
                && a.iter()
                    .filter(|(k, _)| k.as_str() != "diagnostic_solver_contacts")
                    .all(|(k, v)| b.get(k).is_some_and(|other| same_physical(v, other)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_physical(a, b))
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            (a.as_f64().unwrap() as f32).to_bits() == (b.as_f64().unwrap() as f32).to_bits()
        }
        _ => a == b,
    }
}

#[test]
#[ignore = "one frozen450command prefix plus300raise/settle and1370 savednavigation,2120CPU50Hzsteps;0freshVLA/Qwen/RGB; never qualification"]
fn recorded_regrasp_lift_and_carry_retention() -> Result<(), String> {
    let env = |name| std::env::var(name).map_err(|e| e.to_string());
    let fixture_hash = env("G1_MOTOR_REPLAY_FIXTURE_SHA256")?;
    let fixture: Fixture = serde_json::from_slice(&read(
        &PathBuf::from(env("G1_MOTOR_REPLAY_FIXTURE")?),
        &fixture_hash,
        1_048_576,
    )?)
    .map_err(|e| e.to_string())?;
    let prefix_hash = env("G1_GRASP_REPLAY_PREFIX_SHA256")?;
    let prefix: Vec<Value> = read(
        &PathBuf::from(env("G1_GRASP_REPLAY_PREFIX")?),
        &prefix_hash,
        64 * 1024 * 1024,
    )?
    .split(|b| *b == b'\n')
    .filter(|b| !b.is_empty())
    .map(serde_json::from_slice)
    .collect::<Result<_, _>>()
    .map_err(|e| e.to_string())?;
    if fixture.schema != "g1_recorded_station_motor_effort_fixture_v1"
        || prefix.len() != 452
        || prefix[0]["event"] != "insertion_probe_start"
        || prefix[451]["event"] != "insertion_probe_complete"
        || prefix[451]["actual_counts"]["integration_count"] != 450
    {
        return Err("retention requires frozen complete450Tick prefix".into());
    }
    let original: Vec<Value> = read(&fixture.trace, &fixture.trace_sha256, 64 * 1024 * 1024)?
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(serde_json::from_slice)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    if original.len() != 1971 {
        return Err("retention lacks original complete trace".into());
    }
    let config: ArenaTaskRunnerConfig =
        serde_json::from_value(fixture.config["runner"].clone()).map_err(|e| e.to_string())?;
    let station: G1StationConfiguration =
        serde_json::from_value(fixture.config["station"].clone()).map_err(|e| e.to_string())?;
    let ArenaTaskBodyConfig::MobileHomieV2(mut body) = config.body else {
        return Err("retention requires original G1".into());
    };
    let definition =
        G1Definition::load(&body.definition, &body.definition_sha256).map_err(|e| e.to_string())?;
    let calibration = MobileGripCalibration::new(&definition).map_err(|e| e.to_string())?;
    let scene =
        StationScene::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))?;
    let (_, environment, preparation) = prepare(&scene, &station, body.floor_contact_friction)?;
    body.startup_environment = Some(environment);
    let mut owner =
        G1Runner::load_mobile_station_fixture_diagnostic(&body).map_err(|e| e.to_string())?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(env("G1_MOTOR_REPLAY_OUTPUT")?)
        .map_err(|e| e.to_string())?;
    writeln!(output,"{}",json!({"event":"regrasp_retention_start","fixture_sha256":fixture_hash,
        "prefix_sha256":prefix_hash,"preparation":preparation,"original_scene_unchanged":true,
        "cached_commands_not_visual_closed_loop":true,"fresh_N16_Qwen_RGB":0,"task_qualified":false}))
        .map_err(|e|e.to_string())?;
    let mut last = None;
    for (i, row) in prefix[1..451].iter().enumerate() {
        let command: G1Command =
            serde_json::from_value(row["command"].clone()).map_err(|e| e.to_string())?;
        let step = owner.step(&command).map_err(|e| e.to_string())?;
        let actual = journal_body_value(&step)?;
        if row["tick"] != i + 1 || !same_physical(&actual, &row["body"]) {
            return Err(format!("retention physical f32prefix differs at{}", i + 1));
        }
        writeln!(
            output,
            "{}",
            json!({"event":"retention_physical_step","tick":i+1,
            "phase":"frozen_reacquisition_prefix","body":actual,"command":command})
        )
        .map_err(|e| e.to_string())?;
        last = Some(command);
    }
    let state = owner.measurement().map_err(|e| e.to_string())?;
    let mut raising = MobileGripRaising::new(
        MobileRaiseGoal {
            observation: ObservationStamp {
                episode_id: state.episode_id,
                frame_id: 2,
                sim_time_ns: state.sim_time_ns,
                captured_at_unix_ms: 1,
            },
            distance_m: 0.1,
            duration_ticks: 200,
        },
        &state,
        last.ok_or("no prefix command")?,
    )
    .map_err(|e| e.to_string())?;
    for tick in 451..=750 {
        let state = owner.measurement().map_err(|e| e.to_string())?;
        let update = raising
            .update(&state, &calibration)
            .map_err(|e| e.to_string())?;
        let step = owner.step(&update.command).map_err(|e| e.to_string())?;
        writeln!(output,"{}",json!({"event":"retention_physical_step","tick":tick,
            "phase":"diagnostic_lift_settle","body":journal_body_value(&step)?,"command":update.command,
            "raising":update,"synthetic_visual_admission":true})).map_err(|e|e.to_string())?;
    }
    if !raising.completed() {
        return Err("retention lift failed finite300Tick clock".into());
    }
    let mut fixed_grip = raising.command().clone();
    for (i, row) in original[500..1870].iter().enumerate() {
        let navigation = command(row)?;
        fixed_grip.navigation = navigation.navigation;
        fixed_grip.pelvis_height = navigation.pelvis_height;
        fixed_grip.torso_rpy = navigation.torso_rpy;
        let step = owner.step(&fixed_grip).map_err(|e| e.to_string())?;
        writeln!(
            output,
            "{}",
            json!({"event":"retention_physical_step","tick":751+i,
            "phase":"cached_navigation_fixed_new_grip","original_navigation_tick":501+i,
            "body":journal_body_value(&step)?,"command":fixed_grip})
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        output,
        "{}",
        json!({"event":"regrasp_retention_complete","actual_counts":owner.progress_counts(),
        "fresh_N16_Qwen_RGB":0,"task_qualified":false,"actual_visual_closed_loop":false})
    )
    .map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}
