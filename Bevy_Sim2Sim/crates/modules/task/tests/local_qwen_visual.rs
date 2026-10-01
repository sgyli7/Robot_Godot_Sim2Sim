//! Explicitly invoked transport/perception check using a captured Bevy frame.
//! A fixture replay is not evidence of robot task completion or live latency.

use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::json;
use task_minigame::{
    decision::{
        CameraRgb, DecisionLimits, DecisionReply, DecisionSession, LocalQwenClient,
        LocalQwenConfig, ObservationSnapshot, TaskGoal,
    },
    types::{ObservationStamp, RobotSelfState, SkillAvailability, TaskProfile},
};

/// Run only after the shared local service is ready. Both fixtures must originate
/// from the same Bevy observation; the output explicitly records fixture replay.
#[test]
#[ignore = "requires SAI_QWEN_CAPTURE_PNG and SAI_QWEN_ROBOT_STATE_JSON from Bevy, plus the local service"]
fn local_qwen_bevy_image_to_structured_decision() {
    let image_path = PathBuf::from(
        std::env::var_os("SAI_QWEN_CAPTURE_PNG")
            .expect("set SAI_QWEN_CAPTURE_PNG to the actual Bevy camera capture"),
    );
    let state_path = PathBuf::from(
        std::env::var_os("SAI_QWEN_ROBOT_STATE_JSON")
            .expect("set SAI_QWEN_ROBOT_STATE_JSON to corresponding Bevy robot proprioception"),
    );
    let robot: RobotSelfState = serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
    let camera =
        CameraRgb::from_png("bevy_capture_fixture", fs::read(&image_path).unwrap()).unwrap();
    let mut session = DecisionSession::new(1, TaskGoal {
        instruction: std::env::var("SAI_QWEN_GOAL").unwrap_or_else(|_| "Observe the current scene and identify visible task objects or placement markers. No execution is qualified in this perception check.".into()),
        profile: TaskProfile::StaticApple,
        public_scene_description: String::new(),
    }, SkillAvailability::default(), DecisionLimits::default()).unwrap();
    let start_unix_ms = unix_ms();
    let input = session
        .prepare(
            ObservationSnapshot {
                stamp: ObservationStamp {
                    episode_id: 1,
                    frame_id: 1,
                    sim_time_ns: 0,
                    captured_at_unix_ms: start_unix_ms,
                },
                camera,
                robot,
            },
            start_unix_ms,
            0,
        )
        .unwrap();
    let client = LocalQwenClient::new(LocalQwenConfig {
        base_url: std::env::var("SAI_QWEN_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8002/v1".into()),
        model: std::env::var("SAI_QWEN_MODEL").unwrap_or_else(|_| "qwen3.8-27b-fp8".into()),
        timeout: Duration::from_secs(30),
        ..Default::default()
    })
    .unwrap();
    let started = Instant::now();
    let decision = client
        .decide(&input)
        .expect("real local Qwen image/schema inference failed");
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let evidence = json!({
        "test_kind":"captured_bevy_image_fixture_replay",
        "task_success_claim":false,
        "capture_path":image_path,
        "proprioception_path":state_path,
        "service_elapsed_ms":elapsed_ms,
        "decision":decision,
    });
    println!("{}", serde_json::to_string_pretty(&evidence).unwrap());
    let admitted = session
        .accept(
            DecisionReply {
                request_id: input.request_id,
                stamp: input.observation.stamp,
                elapsed_ms,
                result: Ok(decision),
            },
            unix_ms(),
            0,
        )
        .expect("model decision failed the ordinary runtime admission checks");
    assert_eq!(admitted.stamp.episode_id, 1);
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
