//! Compare the runtime native boundary to the actual upstream observation terms.

use robot_minigame::{
    ACTION_DIMENSION,
    contract::{ACTUATOR_ORDER, Command, DelaySeconds, NativeState, PolicyContract},
};
use serde::Deserialize;
use std::{error::Error, fs};

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    dtype: String,
    actuator_order: [String; ACTION_DIMENSION],
    action_scale: [f32; ACTION_DIMENSION],
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    state: State,
    commands: Commands,
    home: [f32; ACTION_DIMENSION],
    last_raw_action: [f32; ACTION_DIMENSION],
    raw_action: [f32; ACTION_DIMENSION],
    expected_observation: Vec<f32>,
    expected_position_target: [f32; ACTION_DIMENSION],
}

#[derive(Deserialize)]
struct State {
    gyro: [f32; 3],
    gravity: [f32; 3],
    q: [f32; ACTION_DIMENSION],
    qd: [f32; ACTION_DIMENSION],
}

#[derive(Deserialize)]
struct Commands {
    locomotion: [f32; 3],
    head: [f32; 4],
    body: [f32; 6],
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected upstream math fixture path")?;
    let fixture: Fixture = serde_json::from_slice(&fs::read(path)?)?;
    if fixture.schema != "source_observation_action_reference_v1"
        || fixture.dtype != "float32"
        || fixture
            .actuator_order
            .iter()
            .map(String::as_str)
            .ne(ACTUATOR_ORDER)
        || fixture.cases.len() < 16
    {
        return Err("unsupported source boundary fixture or coverage".into());
    }
    let mut observation_error = 0.0_f32;
    let mut target_error = 0.0_f32;
    for (case_index, case) in fixture.cases.iter().enumerate() {
        // Hash sentinels deliberately cannot identify an exported actor. This
        // executable checks math only; it never constructs a policy or plant.
        let contract = PolicyContract {
            schema_version: "microduck_pollen_bam60_v1".into(),
            model_sha256: "0".repeat(64),
            robot_definition_sha256: "0".repeat(64),
            actuator_order: fixture.actuator_order.clone(),
            home: case.home,
            action_scale: fixture.action_scale,
            physics_hz: 60,
            policy_hz: 60,
            normalizer_embedded: true,
            delay_seconds: DelaySeconds {
                motor_source_samples: [0.015, 0.020, 0.025, 0.030],
                joint_velocity: 0.020,
                imu_source_samples: [0.0, 0.020],
                imu_resample_period: 1.28,
            },
        };
        contract.validate()?;
        let state = NativeState {
            gyro: case.state.gyro,
            projected_gravity: case.state.gravity,
            joint_position: case.state.q,
            joint_velocity: case.state.qd,
        };
        let command = Command {
            locomotion: case.commands.locomotion,
            head: case.commands.head,
            body: case.commands.body,
        };
        let observation = contract.observation(&state, &command, &case.last_raw_action)?;
        let target = contract.targets(&case.raw_action)?;
        observation_error = observation_error.max(compare(
            &observation,
            &case.expected_observation,
            case_index,
            "observation",
        )?);
        target_error = target_error.max(compare(
            &target,
            &case.expected_position_target,
            case_index,
            "target",
        )?);
    }
    println!(
        "{}",
        serde_json::json!({ "scope": "native_boundary_math_only", "cases": fixture.cases.len(),
        "observation_max_abs_error": observation_error, "target_max_abs_error": target_error,
        "tolerance": 1e-7, "policy_inferences": 0, "physics_integrations": 0,
        "skill_qualified": false })
    );
    Ok(())
}

fn compare(
    actual: &[f32],
    expected: &[f32],
    case: usize,
    term: &str,
) -> Result<f32, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{term} dimension mismatch case={case}: runtime={} source={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut maximum = 0.0_f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        if !actual.is_finite() || !expected.is_finite() || error > 1e-7 {
            return Err(format!(
                "{term} mismatch case={case} index={index} runtime={actual} source={expected}"
            )
            .into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
