//! Compare the runtime kernel with fixtures evaluated by the pinned BAM source.

use robot_minigame::{
    ACTION_DIMENSION,
    actuator::{self, BamEnvironment, BamInput, BamOutput, BamParameters},
};
use serde::Deserialize;
use std::{error::Error, fs};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    bam_source_sha256: String,
    parameters: BamParameters,
    cases: Vec<Case>,
    schema: Option<String>,
    dtype: Option<String>,
    seed: Option<u64>,
    kt_squared_f64: Option<f64>,
    parameter_sha256: Option<String>,
    limits: Option<String>,
    sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    input: Input,
    environment: BamEnvironment,
    expected: BamOutput,
    external_load_decomposition: Option<ExternalLoad>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalLoad {
    qfrc_bias: [f32; ACTION_DIMENSION],
    qfrc_constraint: [f32; ACTION_DIMENSION],
    qfrc_friction: [f32; ACTION_DIMENSION],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    target: [f32; ACTION_DIMENSION],
    position: [f32; ACTION_DIMENSION],
    velocity: [f32; ACTION_DIMENSION],
    previous_motor_computed: [f32; ACTION_DIMENSION],
    previous_actuator_applied: [f32; ACTION_DIMENSION],
    external_load: [f32; ACTION_DIMENSION],
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected BAM reference fixture path")?;
    let fixture: Fixture = serde_json::from_slice(&fs::read(path)?)?;
    if fixture.cases.len() < 100 || fixture.bam_source_sha256.len() != 64 {
        return Err("reference fixture lacks source identity or coverage".into());
    }
    if let Some(schema) = fixture.schema.as_deref() {
        if schema != "bam_torch_reference_v1"
            || fixture.dtype.as_deref() != Some("float32")
            || fixture.seed.is_none()
            || fixture.limits.is_none()
            || fixture
                .parameter_sha256
                .as_ref()
                .is_none_or(|hash| hash.len() != 64)
            || fixture.sha256.as_ref().is_none_or(|hash| hash.len() != 64)
            || fixture.kt_squared_f64 != Some(fixture.parameters.kt * fixture.parameters.kt)
        {
            return Err("invalid generated BAM reference metadata".into());
        }
    }
    let mut maximum_error = 0.0_f32;
    for (case_index, case) in fixture.cases.iter().enumerate() {
        let input = &case.input;
        if let Some(load) = &case.external_load_decomposition {
            for index in 0..ACTION_DIMENSION {
                let external = -load.qfrc_bias[index] + load.qfrc_constraint[index]
                    - load.qfrc_friction[index];
                if !external.is_finite() || external != input.external_load[index] {
                    return Err(format!(
                        "incorrect source load decomposition case={case_index} joint={index}"
                    )
                    .into());
                }
            }
        }
        let actual = actuator::compute(
            &fixture.parameters,
            case.environment,
            BamInput {
                target: &input.target,
                position: &input.position,
                velocity: &input.velocity,
                previous_motor_computed: &input.previous_motor_computed,
                previous_actuator_applied: &input.previous_actuator_applied,
                external_load: &input.external_load,
            },
        )?;
        let actual_values = actual
            .motor_torque
            .iter()
            .chain(&actual.friction_budget)
            .chain([&actual.effective_voltage, &actual.viscous_damping]);
        let expected_values = case
            .expected
            .motor_torque
            .iter()
            .chain(&case.expected.friction_budget)
            .chain([
                &case.expected.effective_voltage,
                &case.expected.viscous_damping,
            ]);
        for (field_index, (value, expected)) in actual_values.zip(expected_values).enumerate() {
            let error = (value - expected).abs();
            maximum_error = maximum_error.max(error);
            if !expected.is_finite() || error > 5e-6 + expected.abs() * 1e-6 {
                return Err(format!("BAM mismatch case={case_index} field={field_index}: runtime={value} source={expected} error={error}").into());
            }
        }
    }
    println!(
        "PASS BAM source={} cases={} maximum_absolute_error={maximum_error:.9}",
        fixture.bam_source_sha256,
        fixture.cases.len()
    );
    Ok(())
}
