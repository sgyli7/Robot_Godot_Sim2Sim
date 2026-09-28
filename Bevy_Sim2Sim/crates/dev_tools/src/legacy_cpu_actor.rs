//! Development-only loader for the nine original 200 Hz / 50 Hz MicroDuck actors.
//!
//! These models predate the 60 Hz policy contract. This module only proves that
//! their original bytes can be run on the pinned CPU runtime; it does not
//! qualify their actions for the current plant or update schedule.

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{TensorElementType, TensorRef, ValueType},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path, sync::Mutex};

const NATIVE_LIBRARY_SHA256: &str =
    "64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b";

const ORIGINAL_MODELS: [(&str, &str); 9] = [
    (
        "alpha_walking",
        "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    ),
    (
        "alpha_stand",
        "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
    ),
    (
        "alpha_sitstand",
        "c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8",
    ),
    (
        "alpha_ground_pick",
        "ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872",
    ),
    (
        "ball_kick_left",
        "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
    ),
    (
        "ball_kick_right",
        "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
    ),
    (
        "roller",
        "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd",
    ),
    (
        "roller_crouch",
        "a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990",
    ),
    (
        "roulade",
        "3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b",
    ),
];

#[derive(Debug, Clone, Serialize)]
pub struct LegacyActorIdentity {
    pub scope: &'static str,
    pub model_name: &'static str,
    pub model_sha256: String,
    pub model_path: String,
    pub original_physics_hz: u32,
    pub original_policy_hz: u32,
    pub native_ort_version: &'static str,
    pub native_library_sha256: String,
    pub native_library_path: String,
    pub native_build_info: String,
    pub native_api_version: u32,
    pub execution_provider: &'static str,
    pub input_name: String,
    pub output_name: String,
}

#[derive(Debug, Clone)]
struct RuntimeIdentity {
    library_sha256: String,
    library_path: String,
    build_info: String,
    api_version: u32,
}

// ORT holds the native library and environment process-wide. Do not pretend
// that init_from can switch the library for a later actor load.
static CPU_RUNTIME: Mutex<Option<RuntimeIdentity>> = Mutex::new(None);

pub struct LegacyCpuActor {
    session: Session,
    identity: LegacyActorIdentity,
    successful_inference_count: u64,
}

impl LegacyCpuActor {
    pub fn load(library_path: &Path, model_path: &Path) -> Result<Self, Box<dyn Error>> {
        if !cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            return Err("pinned native ONNX Runtime release targets Linux aarch64 only".into());
        }
        let model_name = model_path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("original ONNX model must have a UTF-8 file stem")?;
        if model_path
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("onnx")
        {
            return Err("original model must have an .onnx extension".into());
        }
        let (model_name, expected_sha256) = ORIGINAL_MODELS
            .iter()
            .copied()
            .find(|(name, _)| *name == model_name)
            .ok_or("model is not one of the nine original MicroDuck actors")?;
        let model_bytes = fs::read(model_path)?;
        let model_sha256 = sha256(&model_bytes);
        if model_sha256 != expected_sha256 {
            return Err(format!("{model_name}: original ONNX SHA256 mismatch").into());
        }
        let runtime = initialize_runtime(library_path)?;
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_parallel_execution(false)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([ort::ep::CPU::default().build()])?
            .commit_from_memory(&model_bytes)?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err(
                format!("{model_name}: expected exactly one actor input and output").into(),
            );
        }
        validate_tensor(session.inputs()[0].dtype(), 61)?;
        validate_tensor(session.outputs()[0].dtype(), 14)?;
        let input_name = session.inputs()[0].name().to_owned();
        let output_name = session.outputs()[0].name().to_owned();
        if input_name != "obs" || output_name != "actions" {
            return Err(format!("{model_name}: expected obs/actions tensor names").into());
        }
        let identity = LegacyActorIdentity {
            scope: "original_200_50_cpu_actor_not_60hz_qualified",
            model_name,
            model_sha256,
            model_path: fs::canonicalize(model_path)?.display().to_string(),
            original_physics_hz: 200,
            original_policy_hz: 50,
            native_ort_version: "1.30.0",
            native_library_sha256: runtime.library_sha256,
            native_library_path: runtime.library_path,
            native_build_info: runtime.build_info,
            native_api_version: runtime.api_version,
            execution_provider: "CPU",
            input_name,
            output_name,
        };
        Ok(Self {
            session,
            identity,
            successful_inference_count: 0,
        })
    }

    pub fn infer(&mut self, observation: &[f32; 61]) -> Result<[f32; 14], Box<dyn Error>> {
        if observation.iter().any(|value| !value.is_finite()) {
            return Err("legacy actor observation contains a non-finite value".into());
        }
        let input = TensorRef::from_array_view(([1_usize, 61], observation.as_slice()))?;
        let outputs = self
            .session
            .run(ort::inputs! { self.identity.input_name.as_str() => input })?;
        let (shape, values) =
            outputs[self.identity.output_name.as_str()].try_extract_tensor::<f32>()?;
        let dimensions: &[i64] = shape.as_ref();
        if dimensions != [1_i64, 14] || values.len() != 14 {
            return Err("legacy actor output is not float32 [1,14]".into());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("legacy actor output contains a non-finite value".into());
        }
        let mut action = [0.0_f32; 14];
        action.copy_from_slice(values);
        self.successful_inference_count += 1;
        Ok(action)
    }

    pub fn identity(&self) -> &LegacyActorIdentity {
        &self.identity
    }

    pub fn successful_inference_count(&self) -> u64 {
        self.successful_inference_count
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_tensor(dtype: &ValueType, width: i64) -> Result<(), Box<dyn Error>> {
    let ValueType::Tensor { ty, shape, .. } = dtype else {
        return Err("legacy actor input or output is not a tensor".into());
    };
    let dimensions: &[i64] = shape.as_ref();
    if *ty != TensorElementType::Float32 || dimensions != [1, width] {
        return Err(format!("expected legacy float32 actor tensor [1,{width}]").into());
    }
    Ok(())
}

fn initialize_runtime(library_path: &Path) -> Result<RuntimeIdentity, Box<dyn Error>> {
    let library_path = fs::canonicalize(library_path)?;
    let library_sha256 = sha256(&fs::read(&library_path)?);
    if library_sha256 != NATIVE_LIBRARY_SHA256 {
        return Err("native ONNX Runtime 1.30.0 library SHA256 mismatch".into());
    }
    let mut initialized = CPU_RUNTIME
        .lock()
        .map_err(|_| "legacy CPU runtime initialization lock is poisoned")?;
    if let Some(identity) = initialized.as_ref() {
        if identity.library_path != library_path.display().to_string() {
            return Err("legacy CPU runtime was already initialized from another path".into());
        }
        return Ok(identity.clone());
    }
    if !ort::init_from(&library_path)?
        .with_name("bevy_legacy_cpu_actor")
        .with_telemetry(false)
        .commit()
    {
        return Err(
            "native ONNX environment was initialized outside the pinned legacy CPU loader".into(),
        );
    }
    let identity = RuntimeIdentity {
        library_sha256,
        library_path: library_path.display().to_string(),
        build_info: ort::info().to_owned(),
        api_version: ort::MINOR_VERSION,
    };
    *initialized = Some(identity.clone());
    Ok(identity)
}
