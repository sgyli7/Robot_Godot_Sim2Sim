//! Compare recorded Torch calls with the pinned native CPU actor, without simulation.

use robot_minigame::{
    ACTION_DIMENSION, OBSERVATION_DIMENSION, contract::PolicyContract, policy::CpuPolicy,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf, process::ExitCode};

/// Original v6 natural-round call count used when the caller omits the option.
const DEFAULT_EXPECTED_CALLS: usize = 75;
/// Inclusive tool limit, twenty seconds at the 60 Hz call rate.
const MAXIMUM_EXPECTED_CALLS: usize = 20 * 60;
const MAXIMUM_ABSOLUTE_ERROR: f64 = 1e-5;
const REFERENCE_METHOD: &str =
    "actual inference input cloned before Torch policy call and output cloned before env.step";

#[derive(Debug)]
struct Arguments {
    fixture: PathBuf,
    contract: PathBuf,
    model: PathBuf,
    runtime: PathBuf,
    output: PathBuf,
    expected_calls: usize,
}

fn parse_expected_calls(value: &str) -> Result<usize, Box<dyn Error>> {
    if value.is_empty() || value.starts_with("--") {
        return Err("missing value --expected-calls".into());
    }
    if value.starts_with('-') || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid expected-calls {value}").into());
    }
    if value.len() > 1 && value.starts_with('0') {
        return Err(format!("invalid expected-calls {value}").into());
    }
    let parsed = match value.parse::<u64>() {
        Ok(parsed) => parsed,
        Err(error) if error.kind() == &std::num::IntErrorKind::PosOverflow => {
            return Err(format!("expected-calls overflow {value}").into());
        }
        Err(_) => return Err(format!("invalid expected-calls {value}").into()),
    };
    if parsed == 0 || parsed > MAXIMUM_EXPECTED_CALLS as u64 {
        return Err(
            format!("expected-calls out of range 1..={MAXIMUM_EXPECTED_CALLS}: {value}").into(),
        );
    }
    usize::try_from(parsed).map_err(|_| "expected-calls overflow".into())
}

impl Arguments {
    fn parse(values: impl IntoIterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut values = values.into_iter();
        let mut fixture = None;
        let mut contract = None;
        let mut model = None;
        let mut runtime = None;
        let mut output = None;
        let mut expected_calls = None;
        while let Some(flag) = values.next() {
            if flag == "--expected-calls" {
                if expected_calls.is_some() {
                    return Err(format!("duplicate argument {flag}").into());
                }
                let value = values
                    .next()
                    .ok_or_else(|| format!("missing value {flag}"))?;
                expected_calls = Some(parse_expected_calls(&value)?);
                continue;
            }
            let destination = match flag.as_str() {
                "--fixture" => &mut fixture,
                "--contract" => &mut contract,
                "--model" => &mut model,
                "--runtime" => &mut runtime,
                "--output" => &mut output,
                _ => return Err(format!("unknown argument {flag}").into()),
            };
            if destination.is_some() {
                return Err(format!("duplicate argument {flag}").into());
            }
            let value = values
                .next()
                .ok_or_else(|| format!("missing value {flag}"))?;
            if value.starts_with("--") || value.is_empty() {
                return Err(format!("missing value {flag}").into());
            }
            *destination = Some(PathBuf::from(value));
        }
        Ok(Self {
            fixture: fixture.ok_or("missing --fixture")?,
            contract: contract.ok_or("missing --contract")?,
            model: model.ok_or("missing --model")?,
            runtime: runtime.ok_or("missing --runtime")?,
            output: output.ok_or("missing --output")?,
            expected_calls: expected_calls.unwrap_or(DEFAULT_EXPECTED_CALLS),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema: String,
    dtype: String,
    method: String,
    model_sha256: String,
    normalizer_embedded: bool,
    skill_qualified: bool,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    physics_tick: u64,
    time_seconds: f64,
    input: Vec<f32>,
    expected_output: Vec<f32>,
}

#[derive(Debug)]
struct ValidatedCase {
    physics_tick: u64,
    input: [f32; OBSERVATION_DIMENSION],
    expected_output: [f32; ACTION_DIMENSION],
}

impl Fixture {
    fn validate(
        &self,
        contract: &PolicyContract,
        expected_calls: usize,
    ) -> Result<Vec<ValidatedCase>, Box<dyn Error>> {
        contract.validate()?;
        if self.schema != "torch_policy_call_reference_v1"
            || self.dtype != "float32"
            || self.method != REFERENCE_METHOD
            || !self.normalizer_embedded
            || self.skill_qualified
            || self.model_sha256 != contract.model_sha256
        {
            return Err("reference schema, identity or normalizer mismatch".into());
        }
        if !(1..=MAXIMUM_EXPECTED_CALLS).contains(&expected_calls)
            || self.cases.len() != expected_calls
        {
            return Err(format!(
                "expected-call coverage mismatch: expected {expected_calls}, recorded {}",
                self.cases.len()
            )
            .into());
        }
        self.cases
            .iter()
            .enumerate()
            .map(|(index, case)| {
                if case.physics_tick != index as u64
                    || !case.time_seconds.is_finite()
                    || (case.time_seconds - case.physics_tick as f64 / 60.0).abs() > 1e-12
                {
                    return Err(format!("invalid recorded tick/time case={index}").into());
                }
                if case.input.len() != OBSERVATION_DIMENSION
                    || case.expected_output.len() != ACTION_DIMENSION
                {
                    return Err(format!("expected input [61] and output [14] case={index}").into());
                }
                if !case
                    .input
                    .iter()
                    .chain(&case.expected_output)
                    .all(|v| v.is_finite())
                {
                    return Err(format!("non-finite reference input/output case={index}").into());
                }
                let mut input = [0.0; OBSERVATION_DIMENSION];
                let mut expected_output = [0.0; ACTION_DIMENSION];
                input.copy_from_slice(&case.input);
                expected_output.copy_from_slice(&case.expected_output);
                Ok(ValidatedCase {
                    physics_tick: case.physics_tick,
                    input,
                    expected_output,
                })
            })
            .collect()
    }
}

#[derive(Serialize)]
struct FileIdentity {
    path: PathBuf,
    sha256: String,
}

impl FileIdentity {
    fn read(path: &std::path::Path) -> Result<(Self, Vec<u8>), Box<dyn Error>> {
        let bytes = fs::read(path)?;
        let identity = Self {
            path: fs::canonicalize(path)?,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        Ok((identity, bytes))
    }

    fn ensure_unchanged(&self, path: &std::path::Path) -> Result<(), Box<dyn Error>> {
        let (current, _) = Self::read(path)?;
        if current.path != self.path || current.sha256 != self.sha256 {
            return Err(format!(
                "file identity changed during verification: {}",
                path.display()
            )
            .into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct CaseResult {
    case_index: usize,
    physics_tick: u64,
    maximum_absolute_error: f64,
    native_output: [f32; ACTION_DIMENSION],
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    scope: &'static str,
    passed: bool,
    failure: Option<String>,
    unqualified: [&'static str; 5],
    input_dimension: usize,
    output_dimension: usize,
    absolute_error_limit: f64,
    maximum_absolute_error: f64,
    expected_case_count: usize,
    reference_case_count: usize,
    compared_case_count: usize,
    native_inference_attempt_count: u64,
    native_inference_success_count: u64,
    warmup_inference_count: u64,
    physics_step_count: u64,
    fixture: FileIdentity,
    contract: FileIdentity,
    model: FileIdentity,
    native_library: FileIdentity,
    robot_definition_sha256: String,
    runtime_identity: robot_minigame::policy::CpuRuntimeIdentity,
    cases: Vec<CaseResult>,
}

fn verify(arguments: &Arguments) -> Result<Report, Box<dyn Error>> {
    let (fixture_identity, fixture_bytes) = FileIdentity::read(&arguments.fixture)?;
    let (contract_identity, contract_bytes) = FileIdentity::read(&arguments.contract)?;
    let fixture: Fixture = serde_json::from_slice(&fixture_bytes)?;
    let contract: PolicyContract = serde_json::from_slice(&contract_bytes)?;
    let cases = fixture.validate(&contract, arguments.expected_calls)?;
    let (model_identity, _) = FileIdentity::read(&arguments.model)?;
    if model_identity.sha256 != contract.model_sha256 {
        return Err("ONNX SHA256 mismatch".into());
    }
    let (runtime_identity, _) = FileIdentity::read(&arguments.runtime)?;
    let mut policy = CpuPolicy::load(&arguments.runtime, &arguments.model, &contract)?;
    if policy.inference_count() != 0 || policy.successful_inference_count() != 0 {
        return Err("unexpected inference during policy initialization".into());
    }
    let native_identity = policy.runtime_identity().clone();
    if native_identity.library_sha256 != runtime_identity.sha256
        || native_identity.version != "1.30.0"
        || native_identity.api_version != 28
        || native_identity.execution_provider != "CPU"
    {
        return Err("native CPU runtime identity mismatch".into());
    }

    let mut maximum_absolute_error = 0.0_f64;
    let mut results = Vec::with_capacity(cases.len());
    let mut failure = None;
    for (case_index, case) in cases.iter().enumerate() {
        let actual = match policy.infer(&case.input) {
            Ok(actual) => actual,
            Err(error) => {
                failure = Some(format!("native inference case={case_index}: {error}"));
                break;
            }
        };
        let case_error = actual
            .iter()
            .zip(&case.expected_output)
            .map(|(value, expected)| (f64::from(*value) - f64::from(*expected)).abs())
            .fold(0.0_f64, f64::max);
        maximum_absolute_error = maximum_absolute_error.max(case_error);
        if case_error > MAXIMUM_ABSOLUTE_ERROR && failure.is_none() {
            failure = Some(format!(
                "Torch/native mismatch case={case_index} error={case_error}"
            ));
        }
        results.push(CaseResult {
            case_index,
            physics_tick: case.physics_tick,
            maximum_absolute_error: case_error,
            native_output: actual,
        });
    }
    for (identity, path) in [
        (&fixture_identity, &arguments.fixture),
        (&contract_identity, &arguments.contract),
        (&model_identity, &arguments.model),
        (&runtime_identity, &arguments.runtime),
    ] {
        if let Err(error) = identity.ensure_unchanged(path) {
            failure = Some(error.to_string());
        }
    }
    if arguments.expected_calls != cases.len()
        || policy.inference_count() != arguments.expected_calls as u64
        || policy.successful_inference_count() != arguments.expected_calls as u64
        || results.len() != arguments.expected_calls
    {
        failure.get_or_insert_with(|| "native inference coverage incomplete".into());
    }
    Ok(Report {
        schema_version: 1,
        scope: "numerical_export_check",
        passed: failure.is_none(),
        failure,
        unqualified: [
            "training_quality",
            "source_skill_behavior",
            "target_physics",
            "source_target_behavior",
            "robot_control",
        ],
        input_dimension: OBSERVATION_DIMENSION,
        output_dimension: ACTION_DIMENSION,
        absolute_error_limit: MAXIMUM_ABSOLUTE_ERROR,
        maximum_absolute_error,
        expected_case_count: arguments.expected_calls,
        reference_case_count: cases.len(),
        compared_case_count: results.len(),
        native_inference_attempt_count: policy.inference_count(),
        native_inference_success_count: policy.successful_inference_count(),
        warmup_inference_count: 0,
        physics_step_count: 0,
        fixture: fixture_identity,
        contract: contract_identity,
        model: model_identity,
        native_library: runtime_identity,
        robot_definition_sha256: contract.robot_definition_sha256,
        runtime_identity: native_identity,
        cases: results,
    })
}

fn run() -> Result<bool, Box<dyn Error>> {
    let arguments = Arguments::parse(std::env::args().skip(1))?;
    // An output alias must not overwrite an immutable inference input.
    let output = if arguments.output.exists() {
        fs::canonicalize(&arguments.output)?
    } else {
        arguments.output.clone()
    };
    for input in [
        &arguments.fixture,
        &arguments.contract,
        &arguments.model,
        &arguments.runtime,
    ] {
        if output == fs::canonicalize(input)? {
            return Err("output must not overwrite a verification input".into());
        }
    }
    let report = verify(&arguments)?;
    if let Some(parent) = arguments
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&arguments.output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "scope={} passed={} expected={} cases={} attempts={} successes={} maximum_absolute_error={:.9}",
        report.scope,
        report.passed,
        report.expected_case_count,
        report.compared_case_count,
        report.native_inference_attempt_count,
        report.native_inference_success_count,
        report.maximum_absolute_error
    );
    if let Some(failure) = report.failure {
        eprintln!("{failure}");
    }
    Ok(report.passed)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> PolicyContract {
        PolicyContract {
            schema_version: "microduck_pollen_bam60_v1".into(),
            model_sha256: "a".repeat(64),
            robot_definition_sha256: "b".repeat(64),
            actuator_order: robot_minigame::contract::ACTUATOR_ORDER.map(String::from),
            home: [0.0; ACTION_DIMENSION],
            action_scale: [1.0; ACTION_DIMENSION],
            physics_hz: 60,
            policy_hz: 60,
            normalizer_embedded: true,
            delay_seconds: robot_minigame::contract::DelaySeconds {
                motor_source_samples: [0.0; 4],
                joint_velocity: 0.0,
                imu_source_samples: [0.0; 2],
                imu_resample_period: 0.0,
            },
        }
    }

    fn fixture_with_count(count: usize) -> Fixture {
        Fixture {
            schema: "torch_policy_call_reference_v1".into(),
            dtype: "float32".into(),
            method: REFERENCE_METHOD.into(),
            model_sha256: "a".repeat(64),
            normalizer_embedded: true,
            skill_qualified: false,
            cases: (0..count)
                .map(|index| Case {
                    physics_tick: index as u64,
                    time_seconds: index as f64 / 60.0,
                    input: vec![0.0; OBSERVATION_DIMENSION],
                    expected_output: vec![0.0; ACTION_DIMENSION],
                })
                .collect(),
        }
    }

    fn fixture() -> Fixture {
        fixture_with_count(DEFAULT_EXPECTED_CALLS)
    }

    fn complete_cli(extra: &[&str]) -> Vec<String> {
        let mut values = vec![
            "--fixture",
            "fixture.json",
            "--contract",
            "contract.json",
            "--model",
            "model.onnx",
            "--runtime",
            "runtime.so",
            "--output",
            "report.json",
        ];
        values.extend(extra);
        values.into_iter().map(String::from).collect()
    }

    #[test]
    fn rejects_wrong_dimensions_and_nonfinite_reference_before_native_loading() {
        let mut value = fixture();
        value.cases[0].input.pop();
        assert!(
            value
                .validate(&contract(), DEFAULT_EXPECTED_CALLS)
                .unwrap_err()
                .to_string()
                .contains("[61]")
        );
        value = fixture();
        value.cases[0].expected_output.push(0.0);
        assert!(value.validate(&contract(), DEFAULT_EXPECTED_CALLS).is_err());
        value = fixture();
        value.cases[0].input[0] = f32::NAN;
        assert!(
            value
                .validate(&contract(), DEFAULT_EXPECTED_CALLS)
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        value = fixture();
        value.cases[0].expected_output[0] = f32::INFINITY;
        assert!(value.validate(&contract(), DEFAULT_EXPECTED_CALLS).is_err());
    }

    #[test]
    fn rejects_changed_reference_model_and_unmatched_tick_or_coverage() {
        let mut value = fixture();
        value.model_sha256 = "c".repeat(64);
        assert!(value.validate(&contract(), DEFAULT_EXPECTED_CALLS).is_err());
        value = fixture();
        value.cases[3].physics_tick = 4;
        assert!(value.validate(&contract(), DEFAULT_EXPECTED_CALLS).is_err());
        value = fixture();
        value.cases[3].time_seconds = f64::NAN;
        assert!(value.validate(&contract(), DEFAULT_EXPECTED_CALLS).is_err());
        value = fixture();
        value.cases.pop();
        assert!(value.validate(&contract(), DEFAULT_EXPECTED_CALLS).is_err());
    }

    #[test]
    fn rejects_unknown_contract_fields_and_invalid_cli() {
        let mut value = serde_json::to_value(contract()).unwrap();
        value["hidden_physics_hz"] = serde_json::json!(200);
        assert!(serde_json::from_value::<PolicyContract>(value).is_err());
        for values in [
            vec!["--fixture"],
            vec!["--fixture", "--contract"],
            vec!["--unknown", "path"],
            vec!["--fixture", "a", "--fixture", "b"],
        ] {
            assert!(Arguments::parse(values.into_iter().map(String::from)).is_err());
        }
    }

    #[test]
    fn keeps_default_seventy_five_and_accepts_explicit_thirty_seven() {
        let validated = fixture()
            .validate(&contract(), DEFAULT_EXPECTED_CALLS)
            .unwrap();
        assert_eq!(validated.len(), DEFAULT_EXPECTED_CALLS);
        assert_eq!(validated[74].physics_tick, 74);
        let validated = fixture_with_count(37).validate(&contract(), 37).unwrap();
        assert_eq!(validated.len(), 37);
        assert_eq!(validated[36].physics_tick, 36);
        assert!(fixture_with_count(1).validate(&contract(), 1).is_ok());
        assert!(
            fixture_with_count(MAXIMUM_EXPECTED_CALLS)
                .validate(&contract(), MAXIMUM_EXPECTED_CALLS)
                .is_ok()
        );

        let default_arguments = Arguments::parse(complete_cli(&[])).unwrap();
        assert_eq!(default_arguments.expected_calls, DEFAULT_EXPECTED_CALLS);
        let explicit = Arguments::parse(complete_cli(&["--expected-calls", "37"])).unwrap();
        assert_eq!(explicit.expected_calls, 37);
        assert_eq!(
            Arguments::parse(complete_cli(&["--expected-calls", "1"]))
                .unwrap()
                .expected_calls,
            1
        );
        assert_eq!(
            Arguments::parse(complete_cli(&["--expected-calls", "1200"]))
                .unwrap()
                .expected_calls,
            MAXIMUM_EXPECTED_CALLS
        );
    }

    #[test]
    fn rejects_mismatched_and_empty_expected_coverage() {
        assert!(fixture_with_count(36).validate(&contract(), 37).is_err());
        assert!(fixture_with_count(38).validate(&contract(), 37).is_err());
        assert!(fixture_with_count(0).validate(&contract(), 37).is_err());
        assert!(fixture_with_count(0).validate(&contract(), 0).is_err());
        assert!(
            fixture_with_count(MAXIMUM_EXPECTED_CALLS + 1)
                .validate(&contract(), MAXIMUM_EXPECTED_CALLS + 1)
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_expected_calls_before_native_loading() {
        for extra in [
            &["--expected-calls"][..],
            &["--expected-calls", ""],
            &["--expected-calls", "--output"],
            &["--expected-calls", "0"],
            &["--expected-calls", "-1"],
            &["--expected-calls", "-75"],
            &["--expected-calls", "1201"],
            &["--expected-calls", "18446744073709551616"],
            &["--expected-calls", "999999999999999999999"],
            &["--expected-calls", "abc"],
            &["--expected-calls", "37.0"],
            &["--expected-calls", "+37"],
            &["--expected-calls", "00"],
            &["--expected-calls", "37", "--expected-calls", "37"],
            &["--expected-calls", "37", "--expected-calls", "38"],
            &["--not-a-flag", "1"],
        ] {
            assert!(
                Arguments::parse(complete_cli(extra)).is_err(),
                "accepted invalid expected-calls {extra:?}"
            );
        }
    }
}
