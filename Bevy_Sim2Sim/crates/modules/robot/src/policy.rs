//! Explicit CPU ONNX loading with hash, type, shape and output checks.

use crate::contract::{PolicyContract, finite};
use crate::{ACTION_DIMENSION, OBSERVATION_DIMENSION, RobotError};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{TensorElementType, TensorRef, ValueType},
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Mutex};

/// Official 1.30.0 Linux aarch64 CPU release, verified after archive extraction.
/// Archive SHA256 is recorded separately in the dependency delivery manifest.
const NATIVE_LIBRARY_SHA256: &str =
    "64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b";
static CPU_RUNTIME: Mutex<Option<CpuRuntimeIdentity>> = Mutex::new(None);

#[derive(Debug, Clone, serde::Serialize)]
pub struct CpuRuntimeIdentity {
    pub version: &'static str,
    pub library_sha256: String,
    pub build_info: String,
    pub api_version: u32,
    pub execution_provider: &'static str,
}

fn initialize_runtime(library_path: &Path) -> Result<CpuRuntimeIdentity, RobotError> {
    if !cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        return Err(RobotError::Policy(
            "native runtime delivery currently targets Linux aarch64".into(),
        ));
    }
    let bytes = fs::read(library_path).map_err(policy_error)?;
    let hash = format!("{:x}", Sha256::digest(bytes));
    if hash != NATIVE_LIBRARY_SHA256 {
        return Err(RobotError::Policy(
            "native ONNX Runtime SHA256 mismatch".into(),
        ));
    }
    // ort stores a process-global native library. A second path must never be
    // accepted on the assumption that init_from actually replaces that library.
    let mut initialized = CPU_RUNTIME.lock().map_err(policy_error)?;
    if let Some(identity) = initialized.as_ref() {
        return Ok(identity.clone());
    }
    let committed = ort::init_from(library_path)
        .map_err(policy_error)?
        .with_name("bevy_microduck60")
        .with_telemetry(false)
        .commit();
    if !committed {
        return Err(RobotError::Policy(
            "ONNX environment was initialized outside the pinned CPU loader".into(),
        ));
    }
    let identity = CpuRuntimeIdentity {
        version: "1.30.0",
        library_sha256: hash,
        build_info: ort::info().to_owned(),
        api_version: ort::MINOR_VERSION,
        execution_provider: "CPU",
    };
    *initialized = Some(identity.clone());
    Ok(identity)
}

pub struct CpuPolicy {
    session: Session,
    input_name: String,
    output_name: String,
    inference_count: u64,
    successful_inference_count: u64,
    runtime_identity: CpuRuntimeIdentity,
}

impl CpuPolicy {
    pub fn load(
        library_path: &Path,
        model_path: &Path,
        contract: &PolicyContract,
    ) -> Result<Self, RobotError> {
        contract.validate()?;
        let bytes = fs::read(model_path).map_err(|error| RobotError::Policy(error.to_string()))?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if hash != contract.model_sha256 {
            return Err(RobotError::Contract("ONNX SHA256 mismatch".into()));
        }
        let runtime_identity = initialize_runtime(library_path)?;
        let session = Session::builder()
            .map_err(policy_error)?
            .with_intra_threads(1)
            .map_err(policy_error)?
            .with_inter_threads(1)
            .map_err(policy_error)?
            .with_parallel_execution(false)
            .map_err(policy_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(policy_error)?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(policy_error)?
            .commit_from_memory(&bytes)
            .map_err(policy_error)?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err(RobotError::Contract(
                "expected exactly one actor input and output".into(),
            ));
        }
        validate_tensor(session.inputs()[0].dtype(), OBSERVATION_DIMENSION)?;
        validate_tensor(session.outputs()[0].dtype(), ACTION_DIMENSION)?;
        let input_name = session.inputs()[0].name().to_owned();
        let output_name = session.outputs()[0].name().to_owned();
        Ok(Self {
            session,
            input_name,
            output_name,
            inference_count: 0,
            successful_inference_count: 0,
            runtime_identity,
        })
    }

    pub fn infer(
        &mut self,
        observation: &[f32; OBSERVATION_DIMENSION],
    ) -> Result<[f32; ACTION_DIMENSION], RobotError> {
        finite(observation, "policy input")?;
        let input =
            TensorRef::from_array_view(([1_usize, OBSERVATION_DIMENSION], observation.as_slice()))
                .map_err(policy_error)?;
        self.inference_count += 1;
        let outputs = self
            .session
            .run(ort::inputs! { self.input_name.as_str() => input })
            .map_err(policy_error)?;
        let (shape, values) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(policy_error)?;
        let dimensions: &[i64] = shape.as_ref();
        if dimensions != [1_i64, ACTION_DIMENSION as i64] || values.len() != ACTION_DIMENSION {
            return Err(RobotError::Contract(
                "expected actor output [1,14] float32".into(),
            ));
        }
        let mut action = [0.0; ACTION_DIMENSION];
        action.copy_from_slice(values);
        finite(&action, "policy output")?;
        self.successful_inference_count += 1;
        Ok(action)
    }

    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }
    pub fn successful_inference_count(&self) -> u64 {
        self.successful_inference_count
    }
    pub fn runtime_identity(&self) -> &CpuRuntimeIdentity {
        &self.runtime_identity
    }
}

fn policy_error(error: impl std::fmt::Display) -> RobotError {
    RobotError::Policy(error.to_string())
}

fn validate_tensor(dtype: &ValueType, width: usize) -> Result<(), RobotError> {
    let ValueType::Tensor { ty, shape, .. } = dtype else {
        return Err(RobotError::Contract("expected actor tensor".into()));
    };
    let dimensions: &[i64] = shape.as_ref();
    if *ty != TensorElementType::Float32
        || dimensions.len() != 2
        || ![1, -1].contains(&dimensions[0])
        || dimensions[1] != width as i64
    {
        return Err(RobotError::Contract(format!(
            "expected float32 actor tensor [1|batch,{width}]"
        )));
    }
    Ok(())
}
