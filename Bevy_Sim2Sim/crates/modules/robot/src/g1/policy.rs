//! CPU ONNX execution of the two byte-bound Arena homie_v2 policies.

use super::contract::*;
use crate::RobotError;
use ort::{
    session::Session,
    value::{TensorElementType, TensorRef, ValueType},
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Mutex};

pub const STAND_SHA256: &str = "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18";
pub const WALK_SHA256: &str = "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407";
static RUNTIME: Mutex<Option<String>> = Mutex::new(None);

/// Inference receipt separates the selected network from the resulting targets.
#[derive(Clone, Debug, serde::Serialize)]
pub struct HomieResult {
    pub policy: &'static str,
    pub inference_count: u64,
    pub action: [f32; LOWER_COUNT],
    pub targets: Vec<f32>,
}

pub struct HomiePolicy {
    stand: Session,
    walk: Session,
    history: HomieHistory,
    inference_count: u64,
    halted: bool,
}

impl HomiePolicy {
    /// The caller binds the native library bytes, independently of model hashes.
    pub fn load(
        library: &Path,
        library_sha256: &str,
        stand: &Path,
        walk: &Path,
    ) -> Result<Self, RobotError> {
        bound_bytes(library, library_sha256)?;
        let mut identity = RUNTIME.lock().map_err(error)?;
        if let Some(existing) = identity.as_ref() {
            if existing != library_sha256 {
                return Err(RobotError::Policy(
                    "G1 cannot replace the process ONNX Runtime".into(),
                ));
            }
        } else {
            let installed = ort::init_from(library)
                .map_err(error)?
                .with_name("sai_g1_homie_v2")
                .with_telemetry(false)
                .commit();
            if !installed {
                return Err(RobotError::Policy(
                    "ONNX Runtime was initialized outside the G1 byte-bound loader".into(),
                ));
            }
            *identity = Some(library_sha256.to_owned());
        }
        drop(identity);
        let stand = load_session(&bound_bytes(stand, STAND_SHA256)?)?;
        let walk = load_session(&bound_bytes(walk, WALK_SHA256)?)?;
        Ok(Self {
            stand,
            walk,
            history: HomieHistory::default(),
            inference_count: 0,
            halted: false,
        })
    }

    pub fn infer(
        &mut self,
        state: &G1State,
        command: &G1Command,
    ) -> Result<HomieResult, RobotError> {
        if self.halted {
            return Err(RobotError::Policy(
                "G1 inference is halted after a prior error".into(),
            ));
        }
        let result = self.infer_inner(state, command);
        if result.is_err() {
            self.halted = true;
        }
        result
    }

    fn infer_inner(
        &mut self,
        state: &G1State,
        command: &G1Command,
    ) -> Result<HomieResult, RobotError> {
        let observation = self.history.observe(state, command)?;
        let input =
            TensorRef::from_array_view(([1_usize, OBSERVATION_SIZE], observation.as_slice()))
                .map_err(error)?;
        let standing = command.standing();
        let session = if standing {
            &mut self.stand
        } else {
            &mut self.walk
        };
        let outputs = session
            .run(ort::inputs! { "input" => input })
            .map_err(error)?;
        let (shape, values) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(error)?;
        if shape.as_ref() != [1_i64, LOWER_COUNT as i64] || values.len() != LOWER_COUNT {
            return Err(RobotError::Policy(
                "homie output must be [1,15] float32".into(),
            ));
        }
        let mut action = [0.; LOWER_COUNT];
        action.copy_from_slice(values);
        let targets = self.history.accept_action(action, command)?;
        self.inference_count += 1;
        Ok(HomieResult {
            policy: if standing { "stand" } else { "walk" },
            inference_count: self.inference_count,
            action,
            targets: targets.to_vec(),
        })
    }

    /// Episode reset clears history without reusing an errored session.
    pub fn reset_history(&mut self) -> Result<(), RobotError> {
        if self.halted {
            return Err(RobotError::Policy(
                "an errored G1 policy must be rebuilt".into(),
            ));
        }
        self.history.reset();
        Ok(())
    }

    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }
}

pub fn bound_bytes(path: &Path, expected: &str) -> Result<Vec<u8>, RobotError> {
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(RobotError::Contract("expected lowercase SHA256".into()));
    }
    let bytes = fs::read(path).map_err(error)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(RobotError::Contract(format!(
            "G1 SHA256 mismatch: {}",
            path.display()
        )));
    }
    Ok(bytes)
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
    if session.inputs().len() != 1
        || session.outputs().len() != 1
        || session.inputs()[0].name() != "input"
        || session.outputs()[0].name() != "output"
    {
        return Err(RobotError::Policy(
            "unexpected homie input/output names".into(),
        ));
    }
    tensor(session.inputs()[0].dtype(), OBSERVATION_SIZE)?;
    tensor(session.outputs()[0].dtype(), LOWER_COUNT)?;
    Ok(session)
}

fn tensor(value: &ValueType, width: usize) -> Result<(), RobotError> {
    match value {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } if shape.len() == 2 && (shape[0] == -1 || shape[0] == 1) && shape[1] == width as i64 => {
            Ok(())
        }
        _ => Err(RobotError::Policy(
            "unexpected homie tensor type/shape".into(),
        )),
    }
}

fn error(value: impl std::fmt::Display) -> RobotError {
    RobotError::Policy(value.to_string())
}
