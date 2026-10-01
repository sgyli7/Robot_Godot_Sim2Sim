//! One real local visual request from an immutable native camera capture.
//!
//! The live path preserves acquisition time and uses DecisionSession's original
//! freshness limits. Warmup calls the same client/schema without creating a
//! session, admitting a skill or claiming that an old frame is live.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::prelude::{Quat, Transform, Vec3};
use rendering_minigame::g1_camera::{
    CameraPoseSource, EGO_HEIGHT, EGO_WIDTH, G1CameraSourceFrame, G1CaptureStamp,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use task_minigame::{
    decision::{
        CameraRgb, DecisionInput, DecisionLimits, DecisionSession, DecisionWorker, LocalQwenClient,
        LocalQwenConfig, ModelDecision, ObservationSnapshot, TaskGoal,
    },
    types::{ObservationStamp, RobotSelfState, SkillAvailability, SkillRequest, TaskProfile},
};

const LIVE_TIMEOUT: Duration = Duration::from_secs(30);
const LIVE_WAIT_LIMIT: Duration = Duration::from_secs(32);
const WARMUP_TIMEOUT: Duration = Duration::from_secs(120);
const PNG_BYTE_LIMIT: u64 = 16 * 1024 * 1024;
const STAMP_BYTE_LIMIT: u64 = 2 * 1024 * 1024;

/// This probe has no option that enables an unqualified physical capability.
pub struct G1DecisionProbeOptions {
    pub png: PathBuf,
    pub stamp: PathBuf,
    pub output: PathBuf,
    pub instruction: String,
    pub profile: TaskProfile,
    pub base_url: String,
    pub model: String,
    pub warmup: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FileIdentity {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct AdmissionReceipt {
    pub decision_id: u64,
    pub stamp: ObservationStamp,
    pub request: SkillRequest,
    pub reason: String,
    pub image_to_decision_ms: u64,
    pub service_elapsed_ms: u64,
}

/// Failed transport, expiration and admission are evidence, never live passes.
#[derive(Debug, Serialize)]
pub struct G1DecisionProbeReceipt {
    pub schema: &'static str,
    pub scope: &'static str,
    pub qualified: bool,
    pub warmup_only: bool,
    pub probe_succeeded: bool,
    pub live_admitted: bool,
    pub stage: String,
    pub failure_reason: Option<String>,
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
    pub elapsed_ms: u64,
    pub png_input: Option<FileIdentity>,
    pub stamp_input: Option<FileIdentity>,
    pub executable: Option<FileIdentity>,
    pub compiled_source_root: &'static str,
    pub compiled_source_inputs: Vec<FileIdentity>,
    pub original_camera_stamp: Option<G1CaptureStamp>,
    pub observation_stamp: Option<ObservationStamp>,
    pub sent_robot_self_state: Option<RobotSelfState>,
    pub wire_png_sha256: Option<String>,
    pub wire_png_bytes: Option<usize>,
    pub request_body_sha256: Option<String>,
    pub endpoint_base_url: String,
    pub model: String,
    pub non_thinking: bool,
    pub strict_json_schema: bool,
    pub available_skills: SkillAvailability,
    pub frame_ttl_ms: Option<u64>,
    pub sim_ttl_ns: Option<u64>,
    pub paused_sim_time_ns: Option<u64>,
    pub worker_submission_attempts: u32,
    pub worker_submissions: u32,
    pub direct_client_calls: u32,
    /// Counts returned client Results, including transport errors without an HTTP response.
    pub client_results_received: u32,
    pub http_result: Option<String>,
    pub http_result_error: Option<String>,
    pub http_elapsed_ms: Option<u64>,
    /// Worker submission is observable; the worker API has no network-start counter.
    pub http_attempt_counter_exposed: bool,
    pub model_decision: Option<ModelDecision>,
    pub admitted_decision: Option<AdmissionReceipt>,
    pub admission_at_unix_ms: Option<u64>,
    pub image_age_at_admission_ms: Option<u64>,
    pub safe_stop: Option<SkillRequest>,
    pub executor_actions: u32,
    pub physics_integrations: u32,
}

impl G1DecisionProbeReceipt {
    fn initial(options: &G1DecisionProbeOptions) -> Self {
        Self {
            schema: "g1_native_visual_decision_probe_v1",
            scope: if options.warmup {
                "warmup_only_not_live_admission"
            } else {
                "native_capture_local_decision_admission_only"
            },
            qualified: false,
            warmup_only: options.warmup,
            probe_succeeded: false,
            live_admitted: false,
            stage: "initializing".into(),
            failure_reason: None,
            started_at_unix_ms: None,
            finished_at_unix_ms: None,
            elapsed_ms: 0,
            png_input: None,
            stamp_input: None,
            executable: None,
            compiled_source_root: env!("CARGO_MANIFEST_DIR"),
            compiled_source_inputs: compiled_inputs(),
            original_camera_stamp: None,
            observation_stamp: None,
            sent_robot_self_state: None,
            wire_png_sha256: None,
            wire_png_bytes: None,
            request_body_sha256: None,
            endpoint_base_url: options.base_url.clone(),
            model: options.model.clone(),
            non_thinking: true,
            strict_json_schema: true,
            available_skills: SkillAvailability::default(),
            frame_ttl_ms: (!options.warmup).then_some(DecisionLimits::default().max_frame_age_ms),
            sim_ttl_ns: (!options.warmup).then_some(DecisionLimits::default().max_sim_age_ns),
            paused_sim_time_ns: None,
            worker_submission_attempts: 0,
            worker_submissions: 0,
            direct_client_calls: 0,
            client_results_received: 0,
            http_result: None,
            http_result_error: None,
            http_elapsed_ms: None,
            http_attempt_counter_exposed: false,
            model_decision: None,
            admitted_decision: None,
            admission_at_unix_ms: None,
            image_age_at_admission_ms: None,
            safe_stop: None,
            executor_actions: 0,
            physics_integrations: 0,
        }
    }
}

/// Reserve the receipt before reading inputs; once reserved, any diagnostic
/// failure is saved and returned as Err so the CLI exits nonzero.
pub fn run_visual_decision_probe(
    options: &G1DecisionProbeOptions,
) -> Result<G1DecisionProbeReceipt, String> {
    let mut output = reserve_output(&options.output)?;
    let started = Instant::now();
    let mut receipt = G1DecisionProbeReceipt::initial(options);
    let result = run_inner(options, &mut receipt);
    receipt.elapsed_ms = milliseconds(started.elapsed());
    receipt.finished_at_unix_ms = unix_ms().ok();
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
    options: &G1DecisionProbeOptions,
    receipt: &mut G1DecisionProbeReceipt,
) -> Result<(), String> {
    receipt.started_at_unix_ms = Some(unix_ms()?);
    receipt.stage = "reading_inputs".into();
    let (png, png_identity) = read_bound_input(&options.png, PNG_BYTE_LIMIT)?;
    receipt.png_input = Some(png_identity);
    let (stamp_bytes, stamp_identity) = read_bound_input(&options.stamp, STAMP_BYTE_LIMIT)?;
    receipt.stamp_input = Some(stamp_identity);
    receipt.executable = Some(hash_file(
        &std::env::current_exe().map_err(|error| error.to_string())?,
    )?);
    let stamp: G1CaptureStamp = serde_json::from_slice(&stamp_bytes)
        .map_err(|error| format!("camera stamp JSON: {error}"))?;
    receipt.original_camera_stamp = Some(stamp.clone());
    receipt.stage = "validating_native_camera_identity".into();
    let observation = snapshot_from_capture(&stamp, png)?;
    receipt.observation_stamp = Some(observation.stamp);
    receipt.paused_sim_time_ns = Some(observation.stamp.sim_time_ns);
    receipt.sent_robot_self_state = Some(observation.robot.clone());
    receipt.wire_png_sha256 = Some(digest(observation.camera.png()));
    receipt.wire_png_bytes = Some(observation.camera.png().len());
    receipt.stage = "validating_goal_and_local_client".into();
    let goal = TaskGoal {
        instruction: options.instruction.clone(),
        profile: options.profile,
        public_scene_description: String::new(),
    };
    goal.validate().map_err(|error| error.to_string())?;
    let client = LocalQwenClient::new(LocalQwenConfig {
        base_url: options.base_url.clone(),
        model: options.model.clone(),
        timeout: if options.warmup {
            WARMUP_TIMEOUT
        } else {
            LIVE_TIMEOUT
        },
        max_output_tokens: 512,
    })
    .map_err(|error| error.to_string())?;
    if options.warmup {
        let input = DecisionInput {
            request_id: 1,
            observation,
            goal,
            available_skills: SkillAvailability::default(),
            remembered_targets: Vec::new(),
            feedback: Vec::new(),
            recent_decisions: Vec::new(),
        };
        receipt.request_body_sha256 = Some(request_digest(&client, &input)?);
        receipt.stage = "warmup_direct_http".into();
        let started = Instant::now();
        receipt.direct_client_calls = 1;
        let result = client.decide(&input);
        receipt.http_elapsed_ms = Some(milliseconds(started.elapsed()));
        receipt.client_results_received = 1;
        match result {
            Ok(decision) => {
                receipt.model_decision = Some(decision);
                receipt.http_result = Some("client_success_http_200_and_typed_json".into());
                receipt.stage = "warmup_only_completed_no_admission".into();
                Ok(())
            }
            Err(error) => {
                receipt.http_result = Some("client_error".into());
                receipt.http_result_error = Some(error.to_string());
                Err(error.to_string())
            }
        }
    } else {
        let mut session = DecisionSession::new(
            stamp.episode_id,
            goal,
            SkillAvailability::default(),
            DecisionLimits::default(),
        )
        .map_err(|error| error.to_string())?;
        let result = run_live(&mut session, client, observation, receipt);
        if let Err(error) = &result {
            receipt.safe_stop = session.take_safe_stop();
            if receipt.safe_stop.is_none() {
                session.cancel_pending(error.clone());
                receipt.safe_stop = session.take_safe_stop();
            }
        }
        result
    }
}

fn run_live(
    session: &mut DecisionSession,
    client: LocalQwenClient,
    observation: ObservationSnapshot,
    receipt: &mut G1DecisionProbeReceipt,
) -> Result<(), String> {
    // The scene is paused at the captured boundary; wall time keeps advancing.
    let sim_time_ns = observation.stamp.sim_time_ns;
    receipt.stage = "session_prepare_with_original_acquisition_time".into();
    let input = session
        .prepare(observation, unix_ms()?, sim_time_ns)
        .map_err(|error| error.to_string())?;
    receipt.request_body_sha256 = Some(request_digest(&client, &input)?);
    let worker =
        DecisionWorker::spawn(client, session.episode_id()).map_err(|error| error.to_string())?;
    receipt.worker_submission_attempts = 1;
    worker
        .try_submit(input)
        .map_err(|error| error.to_string())?;
    receipt.worker_submissions = 1;
    receipt.stage = "waiting_for_one_worker_result".into();
    let started = Instant::now();
    let iterations = LIVE_WAIT_LIMIT.as_millis() / 10 + 2;
    for _ in 0..iterations {
        if let Some(reply) = worker.try_recv().map_err(|error| error.to_string())? {
            receipt.client_results_received = 1;
            receipt.http_elapsed_ms = Some(reply.elapsed_ms);
            match &reply.result {
                Ok(decision) => {
                    receipt.http_result = Some("client_success_http_200_and_typed_json".into());
                    receipt.model_decision = Some(decision.clone());
                }
                Err(error) => {
                    receipt.http_result = Some("client_error".into());
                    receipt.http_result_error = Some(error.to_string());
                }
            }
            let now = unix_ms()?;
            receipt.admission_at_unix_ms = Some(now);
            receipt.image_age_at_admission_ms = now.checked_sub(reply.stamp.captured_at_unix_ms);
            receipt.stage = "session_admission".into();
            let accepted = session
                .accept(reply, now, sim_time_ns)
                .map_err(|error| error.to_string())?;
            // Session is authoritative; the probe never sends even admitted skills
            // to any executor. The false capability contract admits only these two.
            if !matches!(
                &accepted.request,
                SkillRequest::Observe | SkillRequest::Stop { .. }
            ) {
                return Err(
                    "false-capability session unexpectedly admitted a physical skill".into(),
                );
            }
            receipt.admitted_decision = Some(AdmissionReceipt {
                decision_id: accepted.decision_id,
                stamp: accepted.stamp,
                request: accepted.request,
                reason: accepted.reason,
                image_to_decision_ms: accepted.image_to_decision_ms,
                service_elapsed_ms: accepted.service_elapsed_ms,
            });
            receipt.live_admitted = true;
            receipt.stage = "observe_or_stop_admitted_without_execution".into();
            return Ok(());
        }
        if session.expire_pending(unix_ms()?, sim_time_ns) {
            receipt.stage = "session_expired_waiting_for_http".into();
            return Err(
                "original camera observation expired under the unchanged 30-second TTL".into(),
            );
        }
        if started.elapsed() >= LIVE_WAIT_LIMIT {
            receipt.stage = "bounded_worker_wait_timeout".into();
            return Err("worker result exceeded the 32-second top-level wait bound".into());
        }
        thread::sleep(Duration::from_millis(10));
    }
    receipt.stage = "bounded_worker_wait_exhausted".into();
    Err("bounded worker wait iteration budget exhausted".into())
}

/// Strict pairing from the original 53-body/43-joint native snapshot. Neither
/// original timestamps nor poses are normalized, interpolated or replaced.
pub fn snapshot_from_capture(
    stamp: &G1CaptureStamp,
    png: Vec<u8>,
) -> Result<ObservationSnapshot, String> {
    let native = stamp
        .native_state
        .as_ref()
        .ok_or("camera stamp has no native self state")?;
    if stamp.source != CameraPoseSource::PhysicsBody
        || stamp.interpolation_alpha != 0.
        || stamp.source_ticks != [native.body_frame.source_tick; 2]
        || stamp.episode_id != native.body_frame.episode_id
        || stamp.capture_sequence == 0
        || native.body_frame.source_tick.checked_mul(20_000_000) != Some(stamp.sim_time_ns)
        || stamp.captured_at_unix_ms == 0
        || stamp.captured_at_unix_ms > stamp.copy_encoded_at_unix_ms
        || stamp.copy_encoded_at_unix_ms > stamp.readback_completed_at_unix_ms
    {
        return Err(
            "camera identity, 50-Hz native boundary or acquisition chronology differs".into(),
        );
    }
    G1CameraSourceFrame {
        episode_id: stamp.episode_id,
        source_ticks: stamp.source_ticks,
        interpolation_alpha: stamp.interpolation_alpha,
        sim_time_ns: stamp.sim_time_ns,
        source: stamp.source,
        world_from_camera: Transform {
            translation: Vec3::from_array(stamp.camera_translation),
            rotation: Quat::from_array(stamp.camera_rotation_xyzw),
            scale: Vec3::ONE,
        },
        native_state: Some(Arc::new(native.clone())),
    }
    .validate()?;
    let camera = CameraRgb::from_png("g1_head_rgb", png).map_err(|error| error.to_string())?;
    if camera.width() != EGO_WIDTH || camera.height() != EGO_HEIGHT {
        return Err("expected the original 640x480 native ego PNG".into());
    }
    let measured = &native.measured_joints;
    let [w, x, y, z] = measured.root_rotation_wxyz.map(f64::from);
    let observation = ObservationSnapshot {
        stamp: ObservationStamp {
            episode_id: stamp.episode_id,
            frame_id: stamp.render_frame,
            sim_time_ns: stamp.sim_time_ns,
            captured_at_unix_ms: stamp.captured_at_unix_ms,
        },
        camera,
        robot: RobotSelfState {
            joint_positions: measured.positions.clone(),
            joint_velocities: measured.velocities.clone(),
            base_velocity_mps: measured.root_velocity_source,
            projected_gravity: [
                (2. * (w * y - x * z)) as f32,
                (-2. * (y * z + w * x)) as f32,
                (-w * w + x * x + y * y - z * z) as f32,
            ],
        },
    };
    observation.validate().map_err(|error| error.to_string())?;
    Ok(observation)
}

fn reserve_output(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "receipt output must be new and parent must exist ({}): {error}",
                path.display()
            )
        })
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
    let inputs: [(&str, &[u8]); 15] = [
        (
            "src/g1_decision_diagnostic.rs",
            include_bytes!("g1_decision_diagnostic.rs"),
        ),
        (
            "src/bin/g1_visual_decision_probe.rs",
            include_bytes!("bin/g1_visual_decision_probe.rs"),
        ),
        ("src/lib.rs", include_bytes!("lib.rs")),
        ("Cargo.toml", include_bytes!("../Cargo.toml")),
        (
            "../modules/task/src/decision/client.rs",
            include_bytes!("../../modules/task/src/decision/client.rs"),
        ),
        (
            "../modules/task/src/decision/session.rs",
            include_bytes!("../../modules/task/src/decision/session.rs"),
        ),
        (
            "../modules/task/src/decision/worker.rs",
            include_bytes!("../../modules/task/src/decision/worker.rs"),
        ),
        (
            "../modules/task/src/decision/observation.rs",
            include_bytes!("../../modules/task/src/decision/observation.rs"),
        ),
        (
            "../modules/task/src/decision/wire.rs",
            include_bytes!("../../modules/task/src/decision/wire.rs"),
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
        .into_iter()
        .map(|(path, bytes)| FileIdentity {
            path: Path::new(env!("CARGO_MANIFEST_DIR")).join(path),
            sha256: digest(bytes),
            bytes: bytes.len() as u64,
        })
        .collect()
}

fn request_digest(client: &LocalQwenClient, input: &DecisionInput) -> Result<String, String> {
    let body = client
        .request_body(input)
        .map_err(|error| error.to_string())?;
    if body["chat_template_kwargs"]["enable_thinking"] != false
        || body["response_format"]["json_schema"]["strict"] != true
    {
        return Err("client changed the non-thinking strict JSON contract".into());
    }
    serde_json::to_vec(&body)
        .map(|bytes| digest(&bytes))
        .map_err(|error| error.to_string())
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
    use rendering_minigame::g1_camera::{
        G1CameraJointState, G1CameraNativeState, arena_head_camera,
    };
    use robot_minigame::g1::definition::{G1BodyFrame, G1BodyPose, USD_SHA256};
    use task_minigame::decision::DecisionReply;

    // Pure transform/sensor fixture; no native world, renderer or HTTP is created.
    fn fixture() -> (G1CaptureStamp, Vec<u8>) {
        let frame = G1BodyFrame {
            usd_sha256: USD_SHA256.into(),
            episode_id: 7,
            source_tick: 17,
            sim_time: 0.34,
            bodies: (0..53)
                .map(|body| G1BodyPose {
                    body,
                    translation: [0.; 3],
                    rotation_xyzw: [0., 0., 0., 1.],
                })
                .collect(),
        };
        let camera = arena_head_camera(&frame).unwrap();
        let stamp = G1CaptureStamp {
            episode_id: 7,
            capture_sequence: 3,
            render_frame: 77,
            source_ticks: [17; 2],
            interpolation_alpha: 0.,
            sim_time_ns: 340_000_000,
            source: CameraPoseSource::PhysicsBody,
            captured_at_unix_ms: 1000,
            copy_encoded_at_unix_ms: 1001,
            readback_completed_at_unix_ms: 1002,
            camera_translation: camera.world_from_camera.translation.to_array(),
            camera_rotation_xyzw: camera.world_from_camera.rotation.to_array(),
            native_state: Some(G1CameraNativeState {
                body_frame: frame,
                measured_joints: G1CameraJointState {
                    positions: (0..43).map(|i| i as f32 * 0.1).collect(),
                    velocities: (0..43).map(|i| i as f32 * -0.02).collect(),
                    root_rotation_wxyz: [1., 0., 0., 0.],
                    root_angular_velocity_body: [0.1, 0.2, 0.3],
                    root_velocity_source: [0.4, -0.5, 0.6],
                },
            }),
        };
        let png = CameraRgb::from_rgb(
            "fixture",
            EGO_WIDTH,
            EGO_HEIGHT,
            vec![0; (EGO_WIDTH * EGO_HEIGHT * 3) as usize],
        )
        .unwrap()
        .png()
        .to_vec();
        (stamp, png)
    }

    fn session(episode: u64) -> DecisionSession {
        DecisionSession::new(
            episode,
            TaskGoal {
                instruction: "观察或停止".into(),
                profile: TaskProfile::StaticApple,
                public_scene_description: String::new(),
            },
            SkillAvailability::default(),
            DecisionLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn native_measurements_and_original_identity_are_preserved() {
        let (stamp, png) = fixture();
        let observation = snapshot_from_capture(&stamp, png).unwrap();
        assert_eq!(
            observation.stamp,
            ObservationStamp {
                episode_id: 7,
                frame_id: 77,
                sim_time_ns: 340_000_000,
                captured_at_unix_ms: 1000
            }
        );
        let measured = &stamp.native_state.unwrap().measured_joints;
        assert_eq!(observation.robot.joint_positions, measured.positions);
        assert_eq!(observation.robot.joint_velocities, measured.velocities);
        assert_eq!(
            observation.robot.base_velocity_mps,
            measured.root_velocity_source
        );
        assert_eq!(observation.robot.projected_gravity, [0., 0., -1.]);
    }

    #[test]
    fn interpolation_mismatch_fixture_missing_measurements_and_pose_forgery_fail() {
        let (stamp, png) = fixture();
        for invalid in 0..9 {
            let mut altered = stamp.clone();
            match invalid {
                0 => altered.interpolation_alpha = 0.5,
                1 => altered.source_ticks = [16, 17],
                2 => altered.episode_id += 1,
                3 => {
                    altered
                        .native_state
                        .as_mut()
                        .unwrap()
                        .measured_joints
                        .positions
                        .pop()
                        .unwrap();
                }
                4 => altered.camera_translation[0] += 0.01,
                5 => altered.native_state = None,
                6 => altered.readback_completed_at_unix_ms = 999,
                7 => {
                    altered
                        .native_state
                        .as_mut()
                        .unwrap()
                        .measured_joints
                        .root_rotation_wxyz = [0., 0., 0., 1.];
                }
                8 => {
                    altered
                        .native_state
                        .as_mut()
                        .unwrap()
                        .body_frame
                        .bodies
                        .pop()
                        .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(
                snapshot_from_capture(&altered, png.clone()).is_err(),
                "guard {invalid}"
            );
        }
    }

    #[test]
    fn original_wall_age_rejects_before_http_and_delayed_reply_stops() {
        let (stamp, png) = fixture();
        let observation = snapshot_from_capture(&stamp, png).unwrap();
        let mut session = session(stamp.episode_id);
        assert!(
            session
                .prepare(observation.clone(), 999, stamp.sim_time_ns)
                .is_err()
        );
        assert!(
            session
                .prepare(observation.clone(), 1002, stamp.sim_time_ns - 1)
                .is_err()
        );
        assert!(
            session
                .prepare(observation.clone(), 31_001, stamp.sim_time_ns)
                .is_err()
        );
        assert!(!session.is_pending());
        let input = session
            .prepare(observation, 1002, stamp.sim_time_ns)
            .unwrap();
        let reply = DecisionReply {
            request_id: input.request_id,
            stamp: input.observation.stamp,
            elapsed_ms: 29_999,
            result: Ok(ModelDecision {
                episode_id: 7,
                frame_id: 77,
                observed_targets: Vec::new(),
                request: SkillRequest::Observe,
                reason: "观察".into(),
            }),
        };
        assert!(session.accept(reply, 31_001, stamp.sim_time_ns).is_err());
        assert!(matches!(
            session.take_safe_stop(),
            Some(SkillRequest::Stop { .. })
        ));
    }

    #[test]
    fn false_capabilities_reject_physical_skill_and_wrong_episode_reply() {
        let (stamp, png) = fixture();
        let observation = snapshot_from_capture(&stamp, png).unwrap();
        for request in [
            SkillRequest::NavigateAdjustment {
                forward_mps: 0.1,
                lateral_mps: 0.,
                yaw_rps: 0.,
                duration_ms: 100,
            },
            SkillRequest::ExecuteTask {
                task_profile: TaskProfile::StaticApple,
                target_id: "visible_apple".into(),
                destination_id: "visible_destination".into(),
            },
            SkillRequest::Observe,
        ] {
            let mut session = session(7);
            let input = session
                .prepare(observation.clone(), 1002, stamp.sim_time_ns)
                .unwrap();
            let wrong_episode = matches!(&request, SkillRequest::Observe);
            let reply = DecisionReply {
                request_id: input.request_id,
                stamp: input.observation.stamp,
                elapsed_ms: 1,
                result: Ok(ModelDecision {
                    episode_id: if wrong_episode { 8 } else { 7 },
                    frame_id: 77,
                    observed_targets: Vec::new(),
                    request,
                    reason: "测试".into(),
                }),
            };
            assert!(session.accept(reply, 1003, stamp.sim_time_ns).is_err());
            assert!(matches!(
                session.take_safe_stop(),
                Some(SkillRequest::Stop { .. })
            ));
        }
    }

    #[test]
    fn fresh_observe_admits_with_false_capabilities_and_real_schema_shape() {
        let (stamp, png) = fixture();
        let observation = snapshot_from_capture(&stamp, png).unwrap();
        let mut session = session(stamp.episode_id);
        let input = session
            .prepare(observation, 1002, stamp.sim_time_ns)
            .unwrap();
        assert_eq!(input.available_skills, SkillAvailability::default());
        let client = LocalQwenClient::new(LocalQwenConfig::default()).unwrap();
        let body = client.request_body(&input).unwrap();
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        let context: serde_json::Value =
            serde_json::from_str(body["messages"][1]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(context["stamp"]["captured_at_unix_ms"], 1000);
        assert_eq!(context["stamp"]["frame_id"], 77);
        assert_eq!(
            context["available_skills"],
            serde_json::json!({"static_apple":false,"mobile_box":false,"navigate_adjustment":false})
        );
        assert_eq!(
            context["robot_proprioception"].as_object().unwrap().len(),
            4
        );
        assert!(context.get("native_state").is_none());
        assert!(context.get("task_truth").is_none());
        let reply = DecisionReply {
            request_id: input.request_id,
            stamp: input.observation.stamp,
            elapsed_ms: 1,
            result: Ok(ModelDecision {
                episode_id: 7,
                frame_id: 77,
                observed_targets: Vec::new(),
                request: SkillRequest::Observe,
                reason: "观察".into(),
            }),
        };
        let admitted = session.accept(reply, 1003, stamp.sim_time_ns).unwrap();
        assert_eq!(admitted.request, SkillRequest::Observe);
        assert_eq!(admitted.image_to_decision_ms, 3);
        assert!(session.take_safe_stop().is_none());
    }

    #[test]
    fn receipt_reservation_never_replaces_existing_evidence() {
        let path = std::env::temp_dir().join(format!(
            "g1_decision_receipt_{}_{}.json",
            std::process::id(),
            unix_ms().unwrap()
        ));
        let mut first = reserve_output(&path).unwrap();
        first.write_all(b"retained_evidence").unwrap();
        drop(first);
        assert!(reserve_output(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"retained_evidence");
        fs::remove_file(path).unwrap();
    }
}
