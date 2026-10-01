//! One saved native camera observation through a matched local policy client.
//!
//! This is an offline transport diagnostic. Acquisition identity is immutable;
//! no action queue, world, physical limits, executor or live admission is created.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{
    asset::RenderAssetUsages,
    image::{CompressedImageFormats, Image, ImageSampler, ImageType},
};
use rendering_minigame::g1_camera::G1CaptureStamp;
use robot_minigame::g1::contract::JOINT_NAMES;
use serde::Serialize;
use sha2::{Digest, Sha256};
use task_minigame::{
    decision::ObservationSnapshot,
    policy::{
        ARENA_ACTION_PERIOD_NS, MobilePolicyClient, PolicyActionChunk, PolicyInferenceRequest,
        PolicyObservation, StaticPolicyClient, map_measured_joints, profile_contract,
    },
    types::{ObservationStamp, RobotSelfState, TaskProfile},
};

use crate::g1_decision_diagnostic::{FileIdentity, snapshot_from_capture};

const HTTP_TIMEOUT: Duration = Duration::from_secs(20);
const PNG_BYTE_LIMIT: u64 = 16 * 1024 * 1024;
const STAMP_BYTE_LIMIT: u64 = 2 * 1024 * 1024;
const STAMP_KEYS: [&str; 13] = [
    "episode_id",
    "capture_sequence",
    "render_frame",
    "source_ticks",
    "interpolation_alpha",
    "sim_time_ns",
    "source",
    "captured_at_unix_ms",
    "copy_encoded_at_unix_ms",
    "readback_completed_at_unix_ms",
    "camera_translation",
    "camera_rotation_xyzw",
    "native_state",
];

/// Raw flags are validated only after reserving the new receipt.
pub struct G1PolicyProbeOptions {
    pub png: PathBuf,
    pub stamp: PathBuf,
    pub output: PathBuf,
    pub flags: Vec<String>,
}

struct ParsedOptions {
    profile: TaskProfile,
    endpoint: String,
}

#[derive(Debug, Serialize)]
pub struct PolicySourceIdentity {
    pub repository: &'static str,
    pub revision: &'static str,
    pub arena_revision: &'static str,
    pub body_backend: String,
    pub action_horizon: usize,
    pub action_period_ns: u64,
    pub reference_instruction: &'static str,
    pub instruction_is_fixed: bool,
    /// The returned revision is checked; server weight bytes are not exposed.
    pub server_weight_bytes_independently_verified: bool,
}

/// A successful reply proves this offline seam, never physical qualification.
#[derive(Debug, Serialize)]
pub struct G1PolicyProbeReceipt {
    pub schema: &'static str,
    pub scope: &'static str,
    pub qualified: bool,
    pub live_admitted: bool,
    pub physics_integrations: u32,
    pub executor_actions: u32,
    pub physical_limits_checked: bool,
    pub probe_succeeded: bool,
    pub stage: String,
    pub failure_reason: Option<String>,
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
    pub elapsed_ms: u64,
    pub offline_diagnostic_required: bool,
    pub profile: Option<TaskProfile>,
    pub requested_endpoint: Option<String>,
    pub effective_endpoint: Option<String>,
    pub configured_http_timeout_ms: u64,
    pub expected_model_source: Option<PolicySourceIdentity>,
    pub wire_schema: Option<&'static str>,
    pub png_input: Option<FileIdentity>,
    pub stamp_input: Option<FileIdentity>,
    pub executable: Option<FileIdentity>,
    pub compiled_source_root: &'static str,
    pub compiled_source_inputs: Vec<FileIdentity>,
    pub original_camera_stamp: Option<G1CaptureStamp>,
    pub original_robot_self_state: Option<RobotSelfState>,
    pub request_sequence_id: Option<u64>,
    pub sent_observation: Option<PolicyObservation>,
    pub decoded_rgb_sha256: Option<String>,
    /// Canonical probe input digest, not a claim to expose the client's wire body.
    pub canonical_probe_request_sha256: Option<String>,
    pub submitted_at_unix_ms: Option<u64>,
    pub returned_at_unix_ms: Option<u64>,
    pub image_age_at_submission_ms: Option<u64>,
    pub image_age_at_return_ms: Option<u64>,
    pub client_infer_calls: u32,
    pub client_results_received: u32,
    pub successful_chunk_results: u32,
    pub failed_client_results: u32,
    pub raw_http_attempt_counter_exposed: bool,
    pub observed_client_result: Option<String>,
    pub http_elapsed_ms: Option<u64>,
    pub action_horizon_ns: Option<u64>,
    pub http_latency_consumes_entire_horizon: Option<bool>,
    pub original_image_age_exceeds_horizon_at_return: Option<bool>,
    pub complete_actual_chunk: Option<PolicyActionChunk>,
}

impl G1PolicyProbeReceipt {
    fn initial() -> Self {
        Self {
            schema: "unitree_g1_visual_policy_probe_v1",
            scope: "unchanged_saved_native_frame_offline_transport_only",
            qualified: false,
            live_admitted: false,
            physics_integrations: 0,
            executor_actions: 0,
            physical_limits_checked: false,
            probe_succeeded: false,
            stage: "reserved_receipt".into(),
            failure_reason: None,
            started_at_unix_ms: None,
            finished_at_unix_ms: None,
            elapsed_ms: 0,
            offline_diagnostic_required: true,
            profile: None,
            requested_endpoint: None,
            effective_endpoint: None,
            configured_http_timeout_ms: milliseconds(HTTP_TIMEOUT),
            expected_model_source: None,
            wire_schema: None,
            png_input: None,
            stamp_input: None,
            executable: None,
            compiled_source_root: env!("CARGO_MANIFEST_DIR"),
            compiled_source_inputs: compiled_inputs(),
            original_camera_stamp: None,
            original_robot_self_state: None,
            request_sequence_id: None,
            sent_observation: None,
            decoded_rgb_sha256: None,
            canonical_probe_request_sha256: None,
            submitted_at_unix_ms: None,
            returned_at_unix_ms: None,
            image_age_at_submission_ms: None,
            image_age_at_return_ms: None,
            client_infer_calls: 0,
            client_results_received: 0,
            successful_chunk_results: 0,
            failed_client_results: 0,
            raw_http_attempt_counter_exposed: false,
            observed_client_result: None,
            http_elapsed_ms: None,
            action_horizon_ns: None,
            http_latency_consumes_entire_horizon: None,
            original_image_age_exceeds_horizon_at_return: None,
            complete_actual_chunk: None,
        }
    }
}

/// Reserve output first, then save ordinary validation/HTTP errors and fail.
pub fn run_visual_policy_probe(
    options: &G1PolicyProbeOptions,
) -> Result<G1PolicyProbeReceipt, String> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&options.output)
        .map_err(|error| format!("receipt must be new and parent must exist: {error}"))?;
    let started = Instant::now();
    let mut receipt = G1PolicyProbeReceipt::initial();
    let result = run_inner(options, &mut receipt);
    receipt.finished_at_unix_ms = unix_ms().ok();
    receipt.elapsed_ms = milliseconds(started.elapsed());
    match &result {
        Ok(()) => receipt.probe_succeeded = true,
        Err(error) => receipt.failure_reason = Some(error.clone()),
    }
    serde_json::to_writer_pretty(&mut output, &receipt).map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    output.sync_all().map_err(|error| error.to_string())?;
    result.map(|()| receipt)
}

fn run_inner(
    options: &G1PolicyProbeOptions,
    receipt: &mut G1PolicyProbeReceipt,
) -> Result<(), String> {
    receipt.started_at_unix_ms = Some(unix_ms()?);
    receipt.stage = "validating_explicit_offline_flags".into();
    let parsed = parse_options(&options.flags)?;
    receipt.profile = Some(parsed.profile);
    receipt.requested_endpoint = Some(parsed.endpoint.clone());
    let endpoint = loopback_alias(&parsed.endpoint);
    receipt.effective_endpoint = Some(endpoint.clone());
    let contract = profile_contract(parsed.profile);
    receipt.expected_model_source = Some(PolicySourceIdentity {
        repository: contract.repository,
        revision: contract.revision,
        arena_revision: contract.arena_revision,
        body_backend: format!("{:?}", contract.body_backend),
        action_horizon: contract.action_horizon,
        action_period_ns: ARENA_ACTION_PERIOD_NS,
        reference_instruction: contract.reference_instruction,
        instruction_is_fixed: contract.instruction_is_fixed,
        server_weight_bytes_independently_verified: false,
    });
    receipt.wire_schema = Some(match parsed.profile {
        TaskProfile::StaticApple => "unitree_g1_static_observation_v1",
        TaskProfile::MobileBox => "mobile_observation_v1",
    });
    let horizon_ns = (contract.action_horizon as u64)
        .checked_mul(ARENA_ACTION_PERIOD_NS)
        .ok_or("action horizon overflow")?;
    receipt.action_horizon_ns = Some(horizon_ns);
    receipt.executable = Some(hash_file(
        &std::env::current_exe().map_err(|error| error.to_string())?,
    )?);
    receipt.stage = "reading_unchanged_inputs".into();
    let (png, identity) = read_bound_input(&options.png, PNG_BYTE_LIMIT)?;
    receipt.png_input = Some(identity);
    let (stamp_bytes, identity) = read_bound_input(&options.stamp, STAMP_BYTE_LIMIT)?;
    receipt.stamp_input = Some(identity);
    let stamp = parse_stamp(&stamp_bytes)?;
    receipt.original_camera_stamp = Some(stamp.clone());
    receipt.stage = "validating_native_rgb_self_state_pair".into();
    let snapshot = snapshot_from_capture(&stamp, png)?;
    receipt.original_robot_self_state = Some(snapshot.robot.clone());
    // Original capture sequence binds this single offline request. No new frame ID.
    let request = request_from_snapshot(parsed.profile, stamp.capture_sequence, &snapshot)?;
    receipt.request_sequence_id = Some(request.sequence_id);
    receipt.decoded_rgb_sha256 = Some(digest(&request.observation.camera_rgb));
    let canonical =
        serde_json::to_vec(&(request.profile, request.sequence_id, &request.observation))
            .map_err(|error| error.to_string())?;
    receipt.canonical_probe_request_sha256 = Some(digest(&canonical));
    receipt.sent_observation = Some(request.observation.clone());
    receipt.stage = "constructing_matched_local_client".into();
    // Construction rejects non-loopback endpoints and unsupported timeout values.
    // Never silently lower the requested deadline or substitute another transport.
    let client = match parsed.profile {
        TaskProfile::StaticApple => MatchedClient::Static(
            StaticPolicyClient::new(&endpoint, HTTP_TIMEOUT)
                .map_err(|error| format!("static client: {error:?}"))?,
        ),
        TaskProfile::MobileBox => MatchedClient::Mobile(
            MobilePolicyClient::new(&endpoint, HTTP_TIMEOUT)
                .map_err(|error| format!("mobile client: {error:?}"))?,
        ),
    };
    let submitted_at = unix_ms()?;
    receipt.submitted_at_unix_ms = Some(submitted_at);
    receipt.image_age_at_submission_ms = Some(image_age(request.observation.stamp, submitted_at)?);
    receipt.stage = "one_client_infer_call".into();
    let http_started = Instant::now();
    receipt.client_infer_calls = 1;
    let result = match client {
        MatchedClient::Static(client) => client.infer(&request),
        MatchedClient::Mobile(client) => client.infer(&request),
    };
    receipt.http_elapsed_ms = Some(milliseconds(http_started.elapsed()));
    receipt.client_results_received = 1;
    match result {
        Ok(chunk) => {
            // Existing clients validate original stamp/sequence/profile/revision,
            // exact horizon/20ms and finiteness of all35 values in every frame.
            receipt.successful_chunk_results = 1;
            receipt.observed_client_result = Some("validated_action_chunk".into());
            receipt.complete_actual_chunk = Some(chunk);
        }
        Err(error) => {
            receipt.failed_client_results = 1;
            receipt.observed_client_result = Some(format!("{error:?}"));
        }
    }
    let returned_at = unix_ms()?;
    receipt.returned_at_unix_ms = Some(returned_at);
    let age = image_age(request.observation.stamp, returned_at)?;
    receipt.image_age_at_return_ms = Some(age);
    receipt.http_latency_consumes_entire_horizon =
        Some(receipt.http_elapsed_ms.unwrap_or(0) as u128 * 1_000_000 >= horizon_ns as u128);
    receipt.original_image_age_exceeds_horizon_at_return =
        Some(age as u128 * 1_000_000 >= horizon_ns as u128);
    if receipt.complete_actual_chunk.is_none() {
        return Err(format!(
            "policy infer failed: {}",
            receipt
                .observed_client_result
                .as_deref()
                .unwrap_or("missing result")
        ));
    }
    receipt.stage = "offline_reply_saved_no_live_admission".into();
    Ok(())
}

enum MatchedClient {
    Static(StaticPolicyClient),
    Mobile(MobilePolicyClient),
}

fn parse_options(flags: &[String]) -> Result<ParsedOptions, String> {
    let mut profile = None;
    let mut endpoint = None;
    let mut offline = false;
    let mut index = 0;
    while index < flags.len() {
        match flags[index].as_str() {
            "--offline-diagnostic" if !offline => {
                offline = true;
                index += 1;
            }
            flag @ ("--profile" | "--endpoint") => {
                let value = flags
                    .get(index + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--profile" if profile.is_none() => {
                        profile = Some(match value.as_str() {
                            "static_apple" => TaskProfile::StaticApple,
                            "mobile_box" => TaskProfile::MobileBox,
                            _ => return Err("profile must be static_apple or mobile_box".into()),
                        });
                    }
                    "--endpoint" if endpoint.is_none() && !value.is_empty() => {
                        endpoint = Some(value.clone())
                    }
                    _ => return Err(format!("repeated or invalid flag: {flag}")),
                }
                index += 2;
            }
            flag => return Err(format!("unknown or repeated flag: {flag}")),
        }
    }
    if !offline {
        return Err("--offline-diagnostic is required; this probe has no live mode".into());
    }
    Ok(ParsedOptions {
        profile: profile.ok_or("--profile is required")?,
        endpoint: endpoint.ok_or("--endpoint is required")?,
    })
}

fn loopback_alias(endpoint: &str) -> String {
    match endpoint.strip_prefix("http://localhost") {
        Some(suffix) if suffix.starts_with(':') || suffix.starts_with('/') => {
            format!("http://127.0.0.1{suffix}")
        }
        _ => endpoint.into(),
    }
}

fn parse_stamp(bytes: &[u8]) -> Result<G1CaptureStamp, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("camera stamp JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or("expected original native camera stamp object")?;
    if object.len() != STAMP_KEYS.len() || STAMP_KEYS.iter().any(|key| !object.contains_key(*key)) {
        return Err("expected original native camera stamp fields; service schemas/extra fields are rejected".into());
    }
    // Deserialize the original bytes, retaining serde's duplicate-field checks.
    serde_json::from_slice(bytes).map_err(|error| format!("camera stamp schema: {error}"))
}

fn request_from_snapshot(
    profile: TaskProfile,
    sequence_id: u64,
    snapshot: &ObservationSnapshot,
) -> Result<PolicyInferenceRequest, String> {
    snapshot.validate().map_err(|error| error.to_string())?;
    let names: Vec<String> = JOINT_NAMES.iter().map(|name| (*name).into()).collect();
    let positions = map_measured_joints(&names, &snapshot.robot.joint_positions)
        .map_err(|error| format!("native named joint mapping: {error:?}"))?;
    // CameraRgb already decoded/validated the original PNG. Use Bevy's existing
    // CPU codec to expose those RGB pixels; no app/GPU or new codec is created.
    let image = Image::from_buffer(
        snapshot.camera.png(),
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::default(),
        RenderAssetUsages::default(),
    )
    .map_err(|error| format!("validated camera RGB decode: {error}"))?;
    let pixels = image
        .try_into_dynamic()
        .map_err(|error| format!("validated camera RGB layout: {error}"))?
        .to_rgb8()
        .into_raw();
    let observation = PolicyObservation {
        stamp: snapshot.stamp,
        camera_rgb: pixels,
        camera_width: snapshot.camera.width(),
        camera_height: snapshot.camera.height(),
        measured_joint_positions_rad: positions,
    };
    observation
        .validate()
        .map_err(|error| format!("policy observation: {error:?}"))?;
    Ok(PolicyInferenceRequest {
        profile,
        sequence_id,
        observation,
    })
}

fn image_age(stamp: ObservationStamp, now: u64) -> Result<u64, String> {
    now.checked_sub(stamp.captured_at_unix_ms)
        .ok_or_else(|| "original acquisition time is in the future; no restamping allowed".into())
}

fn read_bound_input(path: &Path, limit: u64) -> Result<(Vec<u8>, FileIdentity), String> {
    let path = fs::canonicalize(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| error.to_string())?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > limit {
        return Err(format!("input exceeds byte limit: {}", path.display()));
    }
    let identity = FileIdentity {
        path,
        sha256: digest(&bytes),
        bytes: bytes.len() as u64,
    };
    Ok((bytes, identity))
}

fn hash_file(path: &Path) -> Result<FileIdentity, String> {
    let path = fs::canonicalize(path).map_err(|error| error.to_string())?;
    let mut file = File::open(&path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0;
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        bytes += count as u64;
    }
    Ok(FileIdentity {
        path,
        sha256: format!("{:x}", hash.finalize()),
        bytes,
    })
}

fn compiled_inputs() -> Vec<FileIdentity> {
    let inputs: &[(&str, &[u8])] = &[
        (
            "src/g1_policy_diagnostic.rs",
            include_bytes!("g1_policy_diagnostic.rs"),
        ),
        (
            "src/bin/g1_visual_policy_probe.rs",
            include_bytes!("bin/g1_visual_policy_probe.rs"),
        ),
        (
            "src/g1_decision_diagnostic.rs",
            include_bytes!("g1_decision_diagnostic.rs"),
        ),
        ("src/lib.rs", include_bytes!("lib.rs")),
        ("Cargo.toml", include_bytes!("../Cargo.toml")),
        (
            "../modules/task/src/policy/client.rs",
            include_bytes!("../../modules/task/src/policy/client.rs"),
        ),
        (
            "../modules/task/src/policy/mobile_client.rs",
            include_bytes!("../../modules/task/src/policy/mobile_client.rs"),
        ),
        (
            "../modules/task/src/policy/observation.rs",
            include_bytes!("../../modules/task/src/policy/observation.rs"),
        ),
        (
            "../modules/task/src/policy/action.rs",
            include_bytes!("../../modules/task/src/policy/action.rs"),
        ),
        (
            "../modules/task/src/decision/observation.rs",
            include_bytes!("../../modules/task/src/decision/observation.rs"),
        ),
        (
            "../modules/task/src/types.rs",
            include_bytes!("../../modules/task/src/types.rs"),
        ),
        (
            "../modules/rendering/src/g1_camera.rs",
            include_bytes!("../../modules/rendering/src/g1_camera.rs"),
        ),
        (
            "../modules/robot/src/g1/contract.rs",
            include_bytes!("../../modules/robot/src/g1/contract.rs"),
        ),
        (
            "../modules/robot/src/g1/definition.rs",
            include_bytes!("../../modules/robot/src/g1/definition.rs"),
        ),
        (
            "../modules/robot/src/basis.rs",
            include_bytes!("../../modules/robot/src/basis.rs"),
        ),
        ("../../Cargo.toml", include_bytes!("../../../Cargo.toml")),
        ("../../Cargo.lock", include_bytes!("../../../Cargo.lock")),
    ];
    inputs
        .iter()
        .map(|(path, bytes)| FileIdentity {
            path: Path::new(env!("CARGO_MANIFEST_DIR")).join(path),
            sha256: digest(bytes),
            bytes: bytes.len() as u64,
        })
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn milliseconds(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}
fn unix_ms() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(milliseconds)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use task_minigame::decision::CameraRgb;

    #[test]
    fn service_schema_and_extra_stamp_fields_cannot_be_native_input() {
        assert!(parse_stamp(br#"{"schema":"mobile_observation_v1","observation":{}}"#).is_err());
        let mut wrong = serde_json::Map::new();
        for key in STAMP_KEYS {
            wrong.insert(key.into(), serde_json::Value::Null);
        }
        wrong.insert("schema".into(), "unitree_g1_static_observation_v1".into());
        assert!(parse_stamp(&serde_json::to_vec(&wrong).unwrap()).is_err());
    }

    #[test]
    fn native_43_to_policy_31_mapping_preserves_identity_and_named_values() {
        // Pure seam fixture, never used by the CLI or called a native capture.
        let stamp = ObservationStamp {
            episode_id: 7,
            frame_id: 901,
            sim_time_ns: 240_000_000,
            captured_at_unix_ms: 1234,
        };
        let snapshot = ObservationSnapshot {
            stamp,
            camera: CameraRgb::from_rgb("fixture", 640, 480, vec![17; 640 * 480 * 3]).unwrap(),
            robot: RobotSelfState {
                joint_positions: (0..43).map(|i| i as f32 / 100.).collect(),
                joint_velocities: vec![0.; 43],
                base_velocity_mps: [0.; 3],
                projected_gravity: [0., 0., -1.],
            },
        };
        let request = request_from_snapshot(TaskProfile::MobileBox, 19, &snapshot).unwrap();
        assert_eq!(request.observation.stamp, stamp);
        assert_eq!(request.sequence_id, 19);
        assert_eq!(image_age(stamp, 1500).unwrap(), 266);
        let q = request.observation.measured_joint_positions_rad;
        assert_eq!(q[0], 0.15); // left_shoulder_pitch, not source joint0.
        assert_eq!(q[7], 0.29); // right_shoulder_pitch follows native left fingers.
        assert_eq!(q[14], 0.22); // left index remains named order, no export permutation here.
        assert_eq!(q[21], 0.36); // right index.
        assert_eq!(&q[28..], &[0.12, 0.13, 0.14]);
        assert_eq!(request.observation.camera_rgb, vec![17; 640 * 480 * 3]);
        let mut missing = snapshot.clone();
        missing.robot.joint_positions.pop();
        missing.robot.joint_velocities.pop();
        assert!(request_from_snapshot(TaskProfile::MobileBox, 19, &missing).is_err());
        // Exact native pairing/pose/chronology cases reuse the shared
        // g1_decision_diagnostic::tests snapshot guards; no second fake body model.
    }

    #[test]
    fn offline_profile_and_endpoint_are_explicit_and_cannot_enable_live_mode() {
        let flags: Vec<String> = [
            "--profile",
            "mobile_box",
            "--endpoint",
            "http://localhost:5558/infer",
            "--offline-diagnostic",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let parsed = parse_options(&flags).unwrap();
        assert_eq!(parsed.profile, TaskProfile::MobileBox);
        assert_eq!(
            loopback_alias(&parsed.endpoint),
            "http://127.0.0.1:5558/infer"
        );
        assert!(parse_options(&flags[..4]).is_err());
        let mut live = flags.clone();
        live.push("--live".into());
        assert!(parse_options(&live).is_err());
        let mut duplicate = flags;
        duplicate.extend(["--profile".into(), "static_apple".into()]);
        assert!(parse_options(&duplicate).is_err());
    }
}
