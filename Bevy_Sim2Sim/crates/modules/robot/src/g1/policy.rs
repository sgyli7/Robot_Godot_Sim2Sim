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
    inference_attempt_count: u64,
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
        initialize_runtime(library, library_sha256)?;
        let stand = load_session(&bound_bytes(stand, STAND_SHA256)?)?;
        let walk = load_session(&bound_bytes(walk, WALK_SHA256)?)?;
        Ok(Self {
            stand,
            walk,
            history: HomieHistory::default(),
            inference_count: 0,
            inference_attempt_count: 0,
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
        // Count actual ORT calls separately from validated accepted outputs.
        self.inference_attempt_count += 1;
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

    /// Successful outputs whose shape/values and history update were accepted.
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    /// Actual ORT session calls, including calls that returned an error.
    pub fn inference_attempt_count(&self) -> u64 {
        self.inference_attempt_count
    }
}

/// Shared native-runtime identity, preserving the original Homie initialization.
/// Every loader verifies the library bytes, including subsequent process users.
pub(super) fn initialize_runtime(library: &Path, library_sha256: &str) -> Result<(), RobotError> {
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
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Executes genuine pinned ONNX files against an independent upstream Python
    /// oracle. Explicit paths keep external model weights out of unit-test fixtures.
    #[test]
    #[ignore = "requires G1_ORACLE, G1_ORACLE_SHA256, G1_MODEL_DIR, G1_ORT, G1_ORT_SHA256"]
    fn upstream_observation_and_real_onnx_parity() {
        assert_oracle(false);
    }

    /// Independently recorded original release scene buffers, including both
    /// networks and all 43 targets actually delivered to the articulation.
    #[test]
    #[ignore = "requires released actual G1_ORACLE + SHA, frozen models/ORT, fresh G1_PARITY_OUTPUT"]
    fn released_scene_observation_and_real_onnx_parity() {
        assert_oracle(true);
    }

    fn assert_oracle(released: bool) {
        let variable = |name| std::env::var(name).expect(name);
        let source = bound_bytes(
            Path::new(&variable("G1_ORACLE")),
            &variable("G1_ORACLE_SHA256"),
        )
        .unwrap();
        let receipt: serde_json::Value = serde_json::from_slice(&source).unwrap();
        if released {
            assert_eq!(receipt["schema"], "g1_released_homie_actual_contract_v1");
            assert_eq!(receipt["source_profile"], "release_0_2_1");
            assert_eq!(
                receipt["arena_commit"],
                "8b4a3a47fc53de23e8205089d71109a2e2348acd"
            );
            assert_eq!(
                receipt["lab_commit"],
                "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
            );
            assert_eq!(
                receipt["source_tree_sha256"],
                "b3e9519d606af00a80f24293f6c3952df8a2fcc134ce6f53eb5d57e031a55cee"
            );
            assert_eq!(receipt["body_assets_sha256"]["stand.onnx"], STAND_SHA256);
            assert_eq!(receipt["body_assets_sha256"]["walk.onnx"], WALK_SHA256);
            assert_eq!(
                receipt["body_assets_sha256"]["g1_29dof_with_hand_rev_1_0.usd"],
                super::super::definition::USD_SHA256
            );
            assert_eq!(
                receipt["source_runtime"]["isaacsim_standalone_build"],
                "6.0.0-rc.22+release.33481.407f3ea1.gl"
            );
            assert_eq!(receipt["joint_names"], serde_json::json!(&JOINT_NAMES[..]));
            assert_eq!(receipt["body_input_truth_fields"], serde_json::json!([]));
            assert_eq!(receipt["extra_inferences"], 0);
            assert_eq!(receipt["reading_changed_physics"], false);
        } else {
            assert_eq!(
                receipt["arena_commit"],
                "7d75c95934c51a0318c957a8831e862ca43c53b5"
            );
        }
        let dir = std::path::PathBuf::from(variable("G1_MODEL_DIR"));
        let mut policy = HomiePolicy::load(
            Path::new(&variable("G1_ORT")),
            &variable("G1_ORT_SHA256"),
            &dir.join("stand.onnx"),
            &dir.join("walk.onnx"),
        )
        .unwrap();
        let mut history = HomieHistory::default();
        let mut max_obs = 0_f32;
        let mut max_action = 0_f32;
        let mut max_target = 0_f32;
        let mut count = 0;
        for frame in receipt["frames"].as_array().unwrap() {
            let vector = |name: &str| {
                frame[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        let value = x.as_f64().unwrap() as f32;
                        assert!(value.is_finite());
                        value
                    })
                    .collect::<Vec<_>>()
            };
            let state = G1State {
                positions: vector("positions").try_into().unwrap(),
                velocities: vector("velocities").try_into().unwrap(),
                root_rotation_wxyz: vector("root_rotation_wxyz").try_into().unwrap(),
                root_angular_velocity_body: vector("root_angular_velocity_body")
                    .try_into()
                    .unwrap(),
            };
            let command = G1Command {
                navigation: vector("navigation").try_into().unwrap(),
                pelvis_height: frame["pelvis_height"].as_f64().unwrap() as f32,
                torso_rpy: vector("torso_rpy").try_into().unwrap(),
                upper_positions: if released {
                    vector("upper_positions").try_into().unwrap()
                } else {
                    [0.; UPPER_COUNT]
                },
            };
            assert_eq!(vector("observation").len(), OBSERVATION_SIZE);
            assert_eq!(vector("action").len(), LOWER_COUNT);
            assert_eq!(vector("lower_targets").len(), LOWER_COUNT);
            if released {
                assert_eq!(frame["control_tick"], count);
                assert_eq!(frame["policy"], if count < 5 { "stand" } else { "walk" });
                assert_eq!(vector("targets").len(), JOINT_COUNT);
                assert_eq!(frame["targets"], frame["articulation_targets"]);
            }
            for (&actual, expected) in history
                .observe(&state, &command)
                .unwrap()
                .iter()
                .zip(vector("observation"))
            {
                max_obs = max_obs.max((actual - expected).abs());
            }
            let result = policy.infer(&state, &command).unwrap();
            assert_eq!(result.policy, frame["policy"].as_str().unwrap());
            for (&actual, expected) in result.action.iter().zip(vector("action")) {
                max_action = max_action.max((actual - expected).abs());
            }
            for (&actual, expected) in result
                .targets
                .iter()
                .take(LOWER_COUNT)
                .zip(vector("lower_targets"))
            {
                max_target = max_target.max((actual - expected).abs());
            }
            if released {
                for (&actual, expected) in result.targets.iter().zip(vector("targets")) {
                    max_target = max_target.max((actual - expected).abs());
                }
            }
            history.accept_action(result.action, &command).unwrap();
            count += 1;
        }
        assert_eq!(count, 10);
        assert_eq!(policy.inference_count(), count);
        assert_eq!(policy.inference_attempt_count(), count);
        println!(
            "HOMIE_ORACLE frames={count} max_obs={max_obs:e} max_action={max_action:e} max_target={max_target:e}"
        );
        assert!(max_obs < 1e-5 && max_action < 1e-5 && max_target < 1e-5);
        if released {
            use std::io::Write;
            let mut output = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(variable("G1_PARITY_OUTPUT"))
                .unwrap();
            let summary = serde_json::json!({"schema":"g1_released_homie_parity_v1",
                "oracle_sha256":variable("G1_ORACLE_SHA256"), "frames":count,
                "stand_inferences":5,"walk_inferences":5,"actual_inference_attempts":policy.inference_attempt_count(),
                "max_observation_abs_error":max_obs,"max_action_abs_error":max_action,
                "max_all_43_targets_abs_error":max_target,"tolerance":1e-5,
                "physics_integrations":0,"native_stability_qualified":false});
            output
                .write_all(&serde_json::to_vec_pretty(&summary).unwrap())
                .unwrap();
        }
    }
}
