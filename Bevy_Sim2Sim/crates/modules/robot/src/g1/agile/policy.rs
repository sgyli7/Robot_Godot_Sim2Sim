//! Single-environment CPU execution of the byte-bound AGILE recurrent student.

use super::{
    ACTION_SIZE, AgileCommand, AgileHistory, AgileState, CONTRACT_VERSION, JOINT_COUNT,
    MODEL_SHA256, OBSERVATION_SIZE, RECURRENT_SIZE, finite,
};
use crate::{
    RobotError,
    g1::policy::{bound_bytes, initialize_runtime},
};
use ort::{
    session::{Session, SessionOutputs},
    value::{TensorElementType, TensorRef, ValueType},
};
use std::path::Path;

/// Bounded receipt of the actual inputs and accepted outputs of one ORT call.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AgileResult {
    pub contract_version: &'static str,
    pub inference_count: u64,
    pub observation: Vec<f32>,
    pub h_in: Vec<f32>,
    pub c_in: Vec<f32>,
    pub raw_action: [f32; ACTION_SIZE],
    pub h_out: Vec<f32>,
    pub c_out: Vec<f32>,
    pub targets: Vec<f32>,
}

/// An independent 80-input/12-action recurrent policy; no Homie history is used.
pub struct AgilePolicy {
    session: Session,
    history: AgileHistory,
    h_state: [f32; RECURRENT_SIZE],
    c_state: [f32; RECURRENT_SIZE],
    inference_count: u64,
    inference_attempt_count: u64,
    halted: bool,
}

impl AgilePolicy {
    /// Bind native runtime and model bytes before accepting any observations.
    /// The defaults are the caller's frozen task articulation defaults in WBC order.
    pub fn load(
        library: &Path,
        library_sha256: &str,
        model: &Path,
        default_positions: [f32; JOINT_COUNT],
    ) -> Result<Self, RobotError> {
        let history = AgileHistory::new(default_positions)?;
        initialize_runtime(library, library_sha256)?;
        let session = load_session(&bound_bytes(model, MODEL_SHA256)?)?;
        Ok(Self {
            session,
            history,
            h_state: [0.; RECURRENT_SIZE],
            c_state: [0.; RECURRENT_SIZE],
            inference_count: 0,
            inference_attempt_count: 0,
            halted: false,
        })
    }

    /// A failed input, ORT call or output validation latches failure until rebuild.
    pub fn infer(
        &mut self,
        state: &AgileState,
        command: &AgileCommand,
    ) -> Result<AgileResult, RobotError> {
        if self.halted {
            return Err(error("inference is halted after a prior error"));
        }
        let result = self.infer_inner(state, command);
        if result.is_err() {
            self.halted = true;
        }
        result
    }

    fn infer_inner(
        &mut self,
        state: &AgileState,
        command: &AgileCommand,
    ) -> Result<AgileResult, RobotError> {
        let observation = *self.history.observe(state, command)?;
        finite(&self.h_state)?;
        finite(&self.c_state)?;
        let obs = TensorRef::from_array_view(([1_usize, OBSERVATION_SIZE], observation.as_slice()))
            .map_err(error)?;
        let h_in =
            TensorRef::from_array_view(([1_usize, 1, RECURRENT_SIZE], self.h_state.as_slice()))
                .map_err(error)?;
        let c_in =
            TensorRef::from_array_view(([1_usize, 1, RECURRENT_SIZE], self.c_state.as_slice()))
                .map_err(error)?;
        // Counts actual native inference, not attempted input validation or replay.
        self.inference_attempt_count += 1;
        let outputs = self
            .session
            .run(ort::inputs! {
                "obs" => obs, "h_in" => h_in, "c_in" => c_in
            })
            .map_err(error)?;
        let raw_action = output::<ACTION_SIZE>(&outputs, "actions", &[1, ACTION_SIZE as i64])?;
        let h_out = output::<RECURRENT_SIZE>(&outputs, "h_out", &[1, 1, RECURRENT_SIZE as i64])?;
        let c_out = output::<RECURRENT_SIZE>(&outputs, "c_out", &[1, 1, RECURRENT_SIZE as i64])?;
        // Validate every recurrent output before committing any action/history.
        let targets = self.history.accept_action(raw_action, command)?;
        self.inference_count += 1;
        let result = AgileResult {
            contract_version: CONTRACT_VERSION,
            inference_count: self.inference_count,
            observation: observation.to_vec(),
            h_in: self.h_state.to_vec(),
            c_in: self.c_state.to_vec(),
            raw_action,
            h_out: h_out.to_vec(),
            c_out: c_out.to_vec(),
            targets: targets.to_vec(),
        };
        self.h_state = h_out;
        self.c_state = c_out;
        Ok(result)
    }

    /// Reset the episode's raw-action feedback and h/c, retaining task defaults.
    /// Commands are explicit per call; counters remain lifetime actual-call counts.
    pub fn reset_history(&mut self) -> Result<(), RobotError> {
        if self.halted {
            return Err(error("an errored policy must be rebuilt"));
        }
        self.history.reset();
        self.h_state.fill(0.);
        self.c_state.fill(0.);
        Ok(())
    }

    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    pub fn inference_attempt_count(&self) -> u64 {
        self.inference_attempt_count
    }

    pub fn halted(&self) -> bool {
        self.halted
    }
}

fn load_session(bytes: &[u8]) -> Result<Session, RobotError> {
    let session = Session::builder()
        .map_err(error)?
        .with_intra_threads(1)
        .map_err(error)?
        .with_inter_threads(1)
        .map_err(error)?
        .with_parallel_execution(false)
        .map_err(error)?
        .with_execution_providers([ort::ep::CPU::default().build()])
        .map_err(error)?
        .commit_from_memory(bytes)
        .map_err(error)?;
    // Only the pinned static-batch single-environment export is accepted.
    let inputs: [(&str, &[i64]); 3] = [
        ("obs", &[1, OBSERVATION_SIZE as i64]),
        ("h_in", &[1, 1, RECURRENT_SIZE as i64]),
        ("c_in", &[1, 1, RECURRENT_SIZE as i64]),
    ];
    let outputs: [(&str, &[i64]); 3] = [
        ("actions", &[1, ACTION_SIZE as i64]),
        ("h_out", &[1, 1, RECURRENT_SIZE as i64]),
        ("c_out", &[1, 1, RECURRENT_SIZE as i64]),
    ];
    for (actual, expected) in [(session.inputs(), inputs), (session.outputs(), outputs)] {
        if actual.len() != expected.len() {
            return Err(error("unexpected input/output count"));
        }
        for (name, shape) in expected {
            let outlet = actual
                .iter()
                .find(|outlet| outlet.name() == name)
                .ok_or_else(|| error(format!("missing tensor {name}")))?;
            tensor(outlet.dtype(), shape)?;
        }
    }
    Ok(session)
}

fn tensor(value: &ValueType, expected: &[i64]) -> Result<(), RobotError> {
    match value {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } if shape.as_ref() == expected => Ok(()),
        _ => Err(error("unexpected tensor type/shape")),
    }
}

fn output<const N: usize>(
    outputs: &SessionOutputs<'_>,
    name: &str,
    expected: &[i64],
) -> Result<[f32; N], RobotError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| error(format!("missing output {name}")))?;
    let (shape, values) = value.try_extract_tensor::<f32>().map_err(error)?;
    checked_output(shape.as_ref(), values, expected)
}

fn checked_output<const N: usize>(
    shape: &[i64],
    values: &[f32],
    expected: &[i64],
) -> Result<[f32; N], RobotError> {
    if shape != expected || values.len() != N {
        return Err(error("unexpected output shape/width"));
    }
    finite(values)?;
    let mut result = [0.; N];
    result.copy_from_slice(values);
    Ok(result)
}

fn error(value: impl std::fmt::Display) -> RobotError {
    RobotError::Policy(format!("AGILE: {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::{Shape, SymbolicDimensions};

    #[test]
    fn strict_static_tensor_and_finite_recurrent_outputs() {
        let tensor_type = |shape: Vec<i64>, ty| ValueType::Tensor {
            ty,
            dimension_symbols: SymbolicDimensions::empty(shape.len()),
            shape: Shape::new(shape),
        };
        assert!(
            tensor(
                &tensor_type(vec![1, 1, 256], TensorElementType::Float32),
                &[1, 1, 256]
            )
            .is_ok()
        );
        assert!(
            tensor(
                &tensor_type(vec![1, 256], TensorElementType::Float32),
                &[1, 1, 256]
            )
            .is_err()
        );
        assert!(
            tensor(
                &tensor_type(vec![1, -1, 256], TensorElementType::Float32),
                &[1, 1, 256]
            )
            .is_err()
        );
        assert!(
            tensor(
                &tensor_type(vec![1, 1, 256], TensorElementType::Float64),
                &[1, 1, 256]
            )
            .is_err()
        );
        assert!(checked_output::<256>(&[1, 1, 256], &[0.; 256], &[1, 1, 256]).is_ok());
        assert!(checked_output::<256>(&[1, 256], &[0.; 256], &[1, 1, 256]).is_err());
        assert!(checked_output::<256>(&[1, 1, 256], &[0.; 255], &[1, 1, 256]).is_err());
        assert!(checked_output::<256>(&[1, 1, 256], &[f32::NAN; 256], &[1, 1, 256]).is_err());
        assert!(checked_output::<256>(&[1, 1, 256], &[f32::INFINITY; 256], &[1, 1, 256]).is_err());
    }

    /// Runs actual Rust ORT calls against the upstream Python CPU oracle, including
    /// recurrence and a reused-session reset. It does not simulate physics.
    #[test]
    #[ignore = "requires G1_AGILE_ORACLE, G1_AGILE_ORACLE_SHA256, G1_AGILE_MODEL, G1_ORT, G1_ORT_SHA256, G1_AGILE_RECEIPT"]
    fn real_onnx_upstream_parity_and_reset() {
        let variable = |name| std::env::var(name).expect(name);
        let oracle_sha256 = variable("G1_AGILE_ORACLE_SHA256");
        let oracle_bytes =
            bound_bytes(Path::new(&variable("G1_AGILE_ORACLE")), &oracle_sha256).unwrap();
        let oracle: serde_json::Value = serde_json::from_slice(&oracle_bytes).unwrap();
        assert_eq!(oracle["schema"], "g1_agile_upstream_numerical_oracle_v1");
        assert_eq!(
            oracle["arena_commit"],
            "8b4a3a47fc53de23e8205089d71109a2e2348acd"
        );
        assert_eq!(oracle["weight_sha256"], MODEL_SHA256);
        assert_eq!(
            oracle["source_hashes"]["isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/policy/g1_agile_policy.py"],
            "d5b54c65312a20289faf1726c00c189869bced1b3021bd9ef1a749ce510a7f4f"
        );
        assert_eq!(
            oracle["execution_providers"],
            serde_json::json!(["CPUExecutionProvider"])
        );
        assert_eq!(oracle["actual_onnx_calls"], 12);
        assert_eq!(oracle["physics_integrations"], 0);
        let frames = oracle["frames"].as_array().unwrap();
        assert_eq!(frames.len(), 12);
        let vector = |frame: &serde_json::Value, name: &str| {
            frame[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let defaults: [f32; JOINT_COUNT] =
            vector(&frames[0], "default_positions").try_into().unwrap();
        let library = variable("G1_ORT");
        let library_sha256 = variable("G1_ORT_SHA256");
        let model = variable("G1_AGILE_MODEL");
        let mut policy = AgilePolicy::load(
            Path::new(&library),
            &library_sha256,
            Path::new(&model),
            defaults,
        )
        .unwrap();
        let mut maxima = [0_f32; 7];
        let mut results = Vec::new();
        let mut resets = 0;
        let mut last_state = None;
        let mut last_command = None;
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame["episode"].as_u64().unwrap(), (index / 6) as u64);
            assert_eq!(frame["index"].as_u64().unwrap(), (index % 6) as u64);
            assert_eq!(vector(frame, "default_positions"), defaults);
            if index % 6 == 0 {
                policy.reset_history().unwrap();
                resets += 1;
            }
            let state = AgileState {
                positions: vector(frame, "positions").try_into().unwrap(),
                velocities: vector(frame, "velocities").try_into().unwrap(),
                root_angular_velocity_com: vector(frame, "root_angular_velocity_body")
                    .try_into()
                    .unwrap(),
                projected_gravity_body: vector(frame, "projected_gravity_body").try_into().unwrap(),
            };
            let command = AgileCommand {
                navigation: vector(frame, "navigation").try_into().unwrap(),
                pelvis_height: frame["pelvis_height"].as_f64().unwrap() as f32,
                upper_positions: vector(frame, "upper_positions").try_into().unwrap(),
            };
            let result = policy.infer(&state, &command).unwrap();
            let fields: [(&str, &[f32]); 7] = [
                ("observation", &result.observation),
                ("raw_action", &result.raw_action),
                ("h_in", &result.h_in),
                ("c_in", &result.c_in),
                ("h_out", &result.h_out),
                ("c_out", &result.c_out),
                ("targets", &result.targets),
            ];
            for (field, (name, actual)) in fields.iter().enumerate() {
                let expected = vector(frame, name);
                assert_eq!(actual.len(), expected.len());
                for (slot, (a, b)) in actual.iter().zip(expected).enumerate() {
                    let diff = (a - b).abs();
                    maxima[field] = maxima[field].max(diff);
                    assert!(
                        diff <= 1e-5,
                        "frame {index} {name}[{slot}]: {a} vs {b}, diff={diff}"
                    );
                }
            }
            assert_eq!(result.inference_count, (index + 1) as u64);
            assert_eq!(policy.inference_attempt_count(), (index + 1) as u64);
            last_state = Some(state);
            last_command = Some(command);
            results.push(result);
        }
        // Identical episodes must reproduce actual h/c/action/targets bit for bit.
        for (a, b) in results[..6].iter().zip(&results[6..]) {
            assert_eq!(a.observation, b.observation);
            assert_eq!(a.raw_action, b.raw_action);
            assert_eq!(a.h_in, b.h_in);
            assert_eq!(a.c_in, b.c_in);
            assert_eq!(a.h_out, b.h_out);
            assert_eq!(a.c_out, b.c_out);
            assert_eq!(a.targets, b.targets);
        }
        assert_eq!(policy.inference_count(), 12);
        assert_eq!(policy.inference_attempt_count(), 12);
        let valid_state = last_state.unwrap();
        let command = last_command.unwrap();
        let mut invalid = valid_state.clone();
        invalid.positions[0] = f32::NAN;
        assert!(policy.infer(&invalid, &command).is_err());
        assert!(policy.halted());
        assert!(policy.reset_history().is_err());
        assert!(policy.infer(&valid_state, &command).is_err());
        assert_eq!(policy.inference_count(), 12);
        assert_eq!(policy.inference_attempt_count(), 12);
        let receipt = serde_json::json!({
            "schema": "g1_agile_rust_ort_parity_v1",
            "scope": "synthetic_cpu_upstream_numerical_parity_not_physics",
            "contract_version": CONTRACT_VERSION, "weight_sha256": MODEL_SHA256,
            "arena_commit": oracle["arena_commit"], "oracle_sha256": oracle_sha256,
            "ort_library": library, "ort_sha256": library_sha256, "model": model,
            "python_ort_version": oracle["onnxruntime_version"],
            "execution_providers": ["CPUExecutionProvider"], "intra_threads": 1, "inter_threads": 1,
            "actual_onnx_calls": policy.inference_attempt_count(),
            "accepted_inferences": policy.inference_count(), "physics_integrations": 0,
            "episode_resets": resets, "reset_reproduces_cold_sequence_bitwise": true,
            "invalid_input_halts_without_ort_call": true,
            "max_absolute_error": {
                "observation": maxima[0], "raw_action": maxima[1], "h_in": maxima[2],
                "c_in": maxima[3], "h_out": maxima[4], "c_out": maxima[5], "targets": maxima[6]
            }, "frames": results,
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(variable("G1_AGILE_RECEIPT"))
            .unwrap();
        serde_json::to_writer_pretty(&mut file, &receipt).unwrap();
        println!(
            "AGILE_ORACLE actual_calls=12 frames=12 resets={resets} max_abs={maxima:?} reset_bitwise=true failure_halt=true physics_integrations=0"
        );
    }
}
