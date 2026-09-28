//! Development-only CPU inference check for the nine original 200/50 policies.
//!
//! This uses synthetic input vectors. It proves model identity and the native
//! inference boundary, never 60 Hz behavior or a robot sensor/control loop.

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{TensorElementType, TensorRef, ValueType},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

const NATIVE_LIBRARY_SHA256: &str =
    "64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b";
const MODELS: [(&str, &str); 9] = [
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

#[derive(Serialize)]
struct ModelResult {
    name: &'static str,
    model_sha256: String,
    input_name: String,
    output_name: String,
    inference_count: usize,
    output_f32_bits_sha256: String,
    output_samples: Vec<[f32; 14]>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    scope: &'static str,
    passed: bool,
    native_ort_version: &'static str,
    native_library_sha256: String,
    native_build_info: String,
    native_api_version: u32,
    provider: &'static str,
    original_physics_hz: u32,
    original_policy_hz: u32,
    runtime_physics_steps: u32,
    robot_sensor_reads: u32,
    models: Vec<ModelResult>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn tensor_width(value: &ValueType, width: i64) -> Result<(), String> {
    let ValueType::Tensor { ty, shape, .. } = value else {
        return Err("actor input or output is not a tensor".into());
    };
    let dimensions: &[i64] = shape.as_ref();
    if *ty != TensorElementType::Float32
        || dimensions.len() != 2
        || ![1, -1].contains(&dimensions[0])
        || dimensions[1] != width
    {
        return Err(format!("expected float32 [1|batch,{width}] actor tensor"));
    }
    Ok(())
}

fn synthetic_observations() -> [[f32; 61]; 2] {
    let mut rest = [0.0_f32; 61];
    rest[4] = -1.0;
    let mut command = rest;
    command[48] = 0.4;
    command[49] = -0.1;
    command[51] = 0.2;
    [rest, command]
}

fn probe_model(
    directory: &Path,
    name: &'static str,
    expected: &str,
) -> Result<ModelResult, Box<dyn Error>> {
    let path = directory.join(format!("{name}.onnx"));
    let bytes = fs::read(&path)?;
    let hash = sha256(&bytes);
    if hash != expected {
        return Err(format!("{name}: original ONNX SHA256 mismatch").into());
    }
    let mut session = Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([ort::ep::CPU::default().build()])?
        .commit_from_memory(&bytes)?;
    if session.inputs().len() != 1 || session.outputs().len() != 1 {
        return Err(format!("{name}: expected exactly one input and output").into());
    }
    tensor_width(session.inputs()[0].dtype(), 61)?;
    tensor_width(session.outputs()[0].dtype(), 14)?;
    let input_name = session.inputs()[0].name().to_owned();
    let output_name = session.outputs()[0].name().to_owned();
    if input_name != "obs" || output_name != "actions" {
        return Err(format!("{name}: original actor tensor names changed").into());
    }
    let mut output_samples = Vec::with_capacity(2);
    let mut output_digest = Sha256::new();
    for observation in synthetic_observations() {
        let input = TensorRef::from_array_view(([1_usize, 61], observation.as_slice()))?;
        let outputs = session.run(ort::inputs! { input_name.as_str() => input })?;
        let (shape, values) = outputs[output_name.as_str()].try_extract_tensor::<f32>()?;
        let dimensions: &[i64] = shape.as_ref();
        if dimensions != [1_i64, 14] || values.len() != 14 || values.iter().any(|x| !x.is_finite())
        {
            return Err(format!("{name}: invalid runtime output tensor").into());
        }
        let sample: [f32; 14] = values.try_into()?;
        for value in sample {
            output_digest.update(value.to_bits().to_le_bytes());
        }
        output_samples.push(sample);
    }
    Ok(ModelResult {
        name,
        model_sha256: hash,
        input_name,
        output_name,
        inference_count: output_samples.len(),
        output_f32_bits_sha256: format!("{:x}", output_digest.finalize()),
        output_samples,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 3 {
        return Err(
            "usage: legacy_nine_onnx_probe NATIVE_ORT_LIB ORIGINAL_MODEL_DIR NEW_REPORT.json"
                .into(),
        );
    }
    if !cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        return Err("pinned native ORT release is Linux aarch64 only".into());
    }
    let library = Path::new(&arguments[0]);
    let model_dir = Path::new(&arguments[1]);
    let output = Path::new(&arguments[2]);
    if output.exists() {
        return Err("report output must not already exist".into());
    }
    let library_sha256 = sha256(&fs::read(library)?);
    if library_sha256 != NATIVE_LIBRARY_SHA256 {
        return Err("native ORT 1.30.0 library SHA256 mismatch".into());
    }
    if !ort::init_from(library)?
        .with_name("bevy_legacy_nine_cpu_probe")
        .with_telemetry(false)
        .commit()
    {
        return Err("native ORT was initialized before this pinned CPU probe".into());
    }
    let mut models = Vec::with_capacity(MODELS.len());
    for (name, expected) in MODELS {
        models.push(probe_model(model_dir, name, expected)?);
    }
    let report = Report {
        schema: "legacy_nine_cpu_probe_v1",
        scope: "original_200_50_models_synthetic_cpu_inference_only",
        passed: models.len() == 9 && models.iter().all(|row| row.inference_count == 2),
        native_ort_version: "1.30.0",
        native_library_sha256: library_sha256,
        native_build_info: ort::info().to_owned(),
        native_api_version: ort::MINOR_VERSION,
        provider: "CPU",
        original_physics_hz: 200,
        original_policy_hz: 50,
        runtime_physics_steps: 0,
        robot_sensor_reads: 0,
        models,
    };
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "STATUS: success; models=9; inferences=18; report={}",
        output.display()
    );
    Ok(())
}
