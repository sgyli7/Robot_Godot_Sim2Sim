use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use task_minigame::{
    decision::{
        CameraRgb, DecisionError, DecisionInput, DecisionLimits, DecisionReply, DecisionSession,
        DecisionWorker, ExecutionFeedback, FeedbackStatus, LocalQwenClient, LocalQwenConfig,
        ModelDecision, ObservationSnapshot, TargetKind, TaskGoal, VisualTarget,
    },
    types::{ObservationStamp, RobotSelfState, SkillAvailability, SkillRequest, TaskProfile},
};

fn goal(profile: TaskProfile) -> TaskGoal {
    TaskGoal {
        instruction: "Place the visible apple in zone B".into(),
        profile,
        public_scene_description: "Placement areas have visible letters.".into(),
    }
}

fn session(profile: TaskProfile, available: bool) -> DecisionSession {
    DecisionSession::new(
        1,
        goal(profile),
        SkillAvailability {
            static_apple: available,
            mobile_box: available,
            navigate_adjustment: available,
        },
        DecisionLimits::default(),
    )
    .unwrap()
}

fn observation(episode_id: u64, frame_id: u64) -> ObservationSnapshot {
    ObservationSnapshot {
        stamp: ObservationStamp {
            episode_id,
            frame_id,
            sim_time_ns: frame_id * 20_000_000,
            captured_at_unix_ms: 1000,
        },
        camera: CameraRgb::from_rgb("head", 2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap(),
        robot: RobotSelfState {
            joint_positions: vec![0.0; 35],
            joint_velocities: vec![0.0; 35],
            base_velocity_mps: [0.0; 3],
            projected_gravity: [0.0, 0.0, -1.0],
        },
    }
}

fn input() -> DecisionInput {
    session(TaskProfile::StaticApple, true)
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap()
}

fn visual(id: &str, kind: TargetKind) -> VisualTarget {
    VisualTarget {
        id: id.into(),
        kind,
        visible_description: format!("visible {id}"),
        bbox_xyxy: [0.1, 0.2, 0.4, 0.5],
        confidence: 0.9,
    }
}

fn decision(request: SkillRequest) -> ModelDecision {
    ModelDecision {
        episode_id: 1,
        frame_id: 1,
        observed_targets: vec![
            visual("apple_1", TargetKind::Apple),
            visual("B", TargetKind::PlacementZone),
        ],
        request,
        reason: "The target and marked destination are visible.".into(),
    }
}

fn execute() -> SkillRequest {
    SkillRequest::ExecuteTask {
        task_profile: TaskProfile::StaticApple,
        target_id: "apple_1".into(),
        destination_id: "B".into(),
    }
}

fn reply(input: &DecisionInput, mut result: ModelDecision) -> DecisionReply {
    result.episode_id = input.observation.stamp.episode_id;
    result.frame_id = input.observation.stamp.frame_id;
    DecisionReply {
        request_id: input.request_id,
        stamp: input.observation.stamp,
        elapsed_ms: 100,
        result: Ok(result),
    }
}

fn completion(decision: &ModelDecision, finish_reason: &str) -> String {
    json!({"choices":[{"message":{"content":serde_json::to_string(decision).unwrap()},"finish_reason":finish_reason}]}).to_string()
}

struct MockServer {
    url: String,
    request: Receiver<Value>,
    handle: thread::JoinHandle<()>,
}

enum ResponseDelivery {
    CompleteAfter(Duration),
    SlowDrip {
        piece_bytes: usize,
        interval: Duration,
    },
}

fn serve(status: &str, body: String, delay: Duration) -> MockServer {
    serve_with_delivery(status, body, ResponseDelivery::CompleteAfter(delay))
}

fn serve_with_delivery(status: &str, body: String, delivery: ResponseDelivery) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let status = status.to_owned();
    let (tx, request) = mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut data = Vec::new();
        let mut buf = [0_u8; 4096];
        let end = loop {
            let length = stream.read(&mut buf).unwrap();
            assert_ne!(length, 0);
            data.extend_from_slice(&buf[..length]);
            if let Some(position) = data.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let headers = std::str::from_utf8(&data[..end]).unwrap();
        assert!(headers.starts_with("POST /v1/chat/completions HTTP/1.1"));
        let length: usize = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|value| value.trim().parse().unwrap())
            })
            .unwrap();
        while data.len() < end + length {
            let length = stream.read(&mut buf).unwrap();
            assert_ne!(length, 0);
            data.extend_from_slice(&buf[..length]);
        }
        tx.send(serde_json::from_slice(&data[end..end + length]).unwrap())
            .unwrap();
        let headers = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        match delivery {
            ResponseDelivery::CompleteAfter(delay) => {
                thread::sleep(delay);
                let _ = stream.write_all(format!("{headers}{body}").as_bytes());
            }
            ResponseDelivery::SlowDrip {
                piece_bytes,
                interval,
            } => {
                if stream.write_all(headers.as_bytes()).is_err() {
                    return;
                }
                for (index, piece) in body.as_bytes().chunks(piece_bytes).enumerate() {
                    if index > 0 {
                        thread::sleep(interval);
                    }
                    // Total-deadline expiry may close the connection mid-body.
                    if stream.write_all(piece).is_err() {
                        break;
                    }
                }
            }
        }
    });
    MockServer {
        url,
        request,
        handle,
    }
}

fn client(url: &str, timeout: Duration) -> LocalQwenClient {
    LocalQwenClient::new(LocalQwenConfig {
        base_url: url.into(),
        timeout,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn only_loopback_http_is_allowed() {
    for url in [
        "https://127.0.0.1/v1",
        "http://example.com/v1",
        "http://192.168.1.1/v1",
        "http://127.0.0.1/v1?token=x",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/other",
    ] {
        assert!(
            LocalQwenClient::new(LocalQwenConfig {
                base_url: url.into(),
                ..Default::default()
            })
            .is_err(),
            "{url}"
        );
    }
    for url in [
        "http://localhost:8002/v1",
        "http://127.0.0.1:8002/v1/",
        "http://[::1]:8002/v1",
    ] {
        assert!(
            LocalQwenClient::new(LocalQwenConfig {
                base_url: url.into(),
                ..Default::default()
            })
            .is_ok(),
            "{url}"
        );
    }
}

#[test]
fn sends_real_png_and_strict_schema_without_privileged_state() {
    let expected = decision(execute());
    let mock = serve("200 OK", completion(&expected, "stop"), Duration::ZERO);
    assert_eq!(
        client(&mock.url, Duration::from_secs(2))
            .decide(&input())
            .unwrap(),
        expected
    );
    let request = mock.request.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(request["response_format"]["type"], "json_schema");
    assert_eq!(request["response_format"]["json_schema"]["strict"], true);
    assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(
        request["response_format"]["json_schema"]["schema"]["properties"]["observed_targets"]["maxItems"],
        16
    );
    let parts = &request["messages"][1]["content"];
    assert!(
        parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,iVBOR")
    );
    let context: Value = serde_json::from_str(parts[0]["text"].as_str().unwrap()).unwrap();
    assert!(context.get("robot_proprioception").is_some());
    assert!(context.get("object_positions").is_none());
    assert_eq!(context["camera"]["width"], 2);
    mock.handle.join().unwrap();
}

#[test]
fn disabled_executor_generation_is_bounded_and_ignoring_it_cannot_enable_execution() {
    let mut session = session(TaskProfile::StaticApple, false);
    let input = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    // An incompatible/malicious service can ignore generation constraints.
    // Ordinary runtime admission must still refuse its physical request.
    let mock = serve(
        "200 OK",
        completion(&decision(execute()), "stop"),
        Duration::ZERO,
    );
    let result = client(&mock.url, Duration::from_secs(2))
        .decide(&input)
        .unwrap();
    assert!(
        session
            .accept(reply(&input, result), 1100, 20_000_000)
            .is_err()
    );
    assert!(session.take_safe_stop().is_some());
    let body = mock.request.recv_timeout(Duration::from_secs(1)).unwrap();
    let schema = &body["response_format"]["json_schema"]["schema"];
    assert_eq!(schema["properties"]["observed_targets"]["maxItems"], 0);
    let skills: Vec<_> = schema["properties"]["request"]["anyOf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|variant| variant["properties"]["skill"]["const"].as_str().unwrap())
        .collect();
    assert_eq!(skills, ["observe", "stop"]);
    mock.handle.join().unwrap();
}

#[test]
fn malformed_unknown_skill_and_truncated_output_are_rejected() {
    for body in [
        "{}".to_owned(),
        json!({"choices":[{"message":{"content":"{\"skill\":\"run_shell\"}"},"finish_reason":"stop"}]}).to_string(),
        completion(&decision(execute()), "length"),
        "x".repeat(256 * 1024 + 1),
    ] {
        let mock = serve("200 OK", body, Duration::ZERO);
        assert!(client(&mock.url, Duration::from_secs(1)).decide(&input()).is_err());
        mock.handle.join().unwrap();
    }
    let mut value = serde_json::to_value(decision(execute())).unwrap();
    value["request"]["shell"] = json!("arbitrary command");
    assert!(serde_json::from_value::<ModelDecision>(value).is_err());
}

#[test]
fn http_failures_and_timeouts_surface_without_retry() {
    for status in [
        "302 Found",
        "429 Too Many Requests",
        "503 Service Unavailable",
    ] {
        let mock = serve(status, "{}".into(), Duration::ZERO);
        assert!(matches!(
            client(&mock.url, Duration::from_secs(1)).decide(&input()),
            Err(DecisionError::Service(_))
        ));
        mock.handle.join().unwrap();
    }
    let mock = serve(
        "200 OK",
        completion(&decision(SkillRequest::Observe), "stop"),
        Duration::from_millis(150),
    );
    let started = Instant::now();
    assert!(
        client(&mock.url, Duration::from_millis(40))
            .decide(&input())
            .is_err()
    );
    assert!(started.elapsed() < Duration::from_millis(120));
    mock.handle.join().unwrap();
}

#[test]
fn slow_drip_response_cannot_extend_the_total_decision_deadline() {
    // A valid completion delivered in about 21 pieces would succeed after
    // >=475ms if the client renewed a 200ms timeout on every blocking Read.
    let body = completion(&decision(SkillRequest::Observe), "stop");
    let piece_bytes = body.len().div_ceil(21);
    assert!(body.len().div_ceil(piece_bytes) >= 20);
    let mock = serve_with_delivery(
        "200 OK",
        body,
        ResponseDelivery::SlowDrip {
            piece_bytes,
            interval: Duration::from_millis(25),
        },
    );
    let client = client(&mock.url, Duration::from_millis(200));
    let input = input();
    let started = Instant::now();
    let result = client.decide(&input);
    let elapsed = started.elapsed();
    mock.request.recv_timeout(Duration::from_secs(1)).unwrap();
    mock.handle.join().unwrap();
    assert!(matches!(result, Err(DecisionError::Service(_))));
    assert!(
        elapsed < Duration::from_millis(450),
        "slow body extended the total request deadline to {elapsed:?}"
    );
}

#[test]
fn invalid_camera_and_robot_sensor_values_are_rejected() {
    assert!(CameraRgb::from_png("head", b"not an image".to_vec()).is_err());
    assert!(CameraRgb::from_rgb("head", 4096, 1, Vec::new()).is_err());
    assert!(CameraRgb::from_rgb("head", 2, 1, vec![0; 5]).is_err());
    let mut frame = observation(1, 1);
    frame.robot.joint_positions[0] = f32::NAN;
    assert!(frame.validate().is_err());
    frame.robot.joint_positions[0] = 0.0;
    frame.robot.joint_velocities.pop();
    assert!(frame.validate().is_err());
}

#[test]
fn accepts_matching_visual_targets_and_never_hidden_target_ids() {
    let mut session = session(TaskProfile::StaticApple, true);
    let input = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    let accepted = session
        .accept(reply(&input, decision(execute())), 1100, 40_000_000)
        .unwrap();
    assert_eq!(accepted.request, execute());
    assert_eq!(session.targets().count(), 2);
    assert_eq!(accepted.image_to_decision_ms, 100);
    let next = session
        .prepare(observation(1, 2), 1200, 40_000_000)
        .unwrap();
    let mut absent = decision(execute());
    absent.observed_targets.clear();
    assert!(
        session
            .accept(reply(&next, absent), 1200, 40_000_000)
            .is_err()
    );
    assert!(matches!(
        session.take_safe_stop(),
        Some(SkillRequest::Stop { .. })
    ));
}

#[test]
fn unqualified_profiles_wrong_object_kinds_and_invented_ids_fail_closed() {
    let mut cases = vec![(false, decision(execute()))];
    let mut wrong_kind = decision(execute());
    wrong_kind.observed_targets[0].kind = TargetKind::Box;
    cases.push((true, wrong_kind));
    let mut unknown = decision(execute());
    unknown.request = SkillRequest::ExecuteTask {
        task_profile: TaskProfile::StaticApple,
        target_id: "hidden_cup".into(),
        destination_id: "B".into(),
    };
    cases.push((true, unknown));
    let mut wrong_profile = decision(execute());
    wrong_profile.request = SkillRequest::ExecuteTask {
        task_profile: TaskProfile::MobileBox,
        target_id: "apple_1".into(),
        destination_id: "B".into(),
    };
    cases.push((true, wrong_profile));
    for (available, decision) in cases {
        let mut session = session(TaskProfile::StaticApple, available);
        let input = session
            .prepare(observation(1, 1), 1000, 20_000_000)
            .unwrap();
        assert!(
            session
                .accept(reply(&input, decision), 1000, 20_000_000)
                .is_err()
        );
        assert_eq!(session.targets().count(), 0);
    }
    assert!(serde_json::from_str::<TaskProfile>("\"universal_g1\"").is_err());
}

#[test]
fn validates_visual_boxes_confidence_and_unique_ids() {
    for mutate in 0..4 {
        let mut invalid = decision(execute());
        match mutate {
            0 => invalid.observed_targets[0].confidence = 0.2,
            1 => invalid.observed_targets[0].bbox_xyxy = [0.8, 0.2, 0.1, 0.4],
            2 => invalid.observed_targets[0].bbox_xyxy[0] = f32::INFINITY,
            3 => invalid.observed_targets[1].id = "apple_1".into(),
            _ => unreachable!(),
        }
        let mut session = session(TaskProfile::StaticApple, true);
        let input = session
            .prepare(observation(1, 1), 1000, 20_000_000)
            .unwrap();
        assert!(
            session
                .accept(reply(&input, invalid), 1000, 20_000_000)
                .is_err()
        );
    }
}

#[test]
fn validates_navigation_velocity_duration_and_profile() {
    for (profile, speed, duration, succeeds) in [
        (TaskProfile::MobileBox, 0.2, 200, true),
        (TaskProfile::StaticApple, 0.2, 200, false),
        (TaskProfile::MobileBox, 0.3, 200, false),
        (TaskProfile::MobileBox, f32::NAN, 200, false),
        (TaskProfile::MobileBox, 0.2, 5000, false),
    ] {
        let mut session = session(profile, true);
        let input = session
            .prepare(observation(1, 1), 1000, 20_000_000)
            .unwrap();
        let decision = decision(SkillRequest::NavigateAdjustment {
            forward_mps: speed,
            lateral_mps: 0.0,
            yaw_rps: 0.0,
            duration_ms: duration,
        });
        assert_eq!(
            session
                .accept(reply(&input, decision), 1000, 20_000_000)
                .is_ok(),
            succeeds
        );
    }
}

#[test]
fn wall_time_and_simulation_time_both_expire_results() {
    for (now, sim) in [
        (31_001, 20_000_000),
        (1000, 30_020_000_001),
        (999, 20_000_000),
        (1000, 19_999_999),
    ] {
        let mut session = session(TaskProfile::StaticApple, true);
        let input = session
            .prepare(observation(1, 1), 1000, 20_000_000)
            .unwrap();
        assert!(
            session
                .accept(reply(&input, decision(execute())), now, sim)
                .is_err()
        );
        assert!(session.take_safe_stop().is_some());
    }
}

#[test]
fn reset_discards_old_response_without_cancelling_new_request() {
    let mut session = session(TaskProfile::StaticApple, true);
    let old = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    session.reset(2, goal(TaskProfile::StaticApple)).unwrap();
    let new = session
        .prepare(observation(2, 1), 1000, 20_000_000)
        .unwrap();
    assert!(
        session
            .accept(reply(&old, decision(execute())), 1000, 20_000_000)
            .is_err()
    );
    assert!(session.is_pending());
    assert!(new.feedback.is_empty());
    assert!(new.remembered_targets.is_empty());
    assert!(new.recent_decisions.is_empty());
    assert!(
        session
            .accept(reply(&new, decision(execute())), 1000, 20_000_000)
            .is_ok()
    );
    assert!(session.reset(2, goal(TaskProfile::StaticApple)).is_err());
}

#[test]
fn pending_expiration_and_operator_stop_invalidate_inflight_decisions() {
    let mut session = session(TaskProfile::StaticApple, true);
    let old = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    assert!(session.expire_pending(40_000, 20_000_000));
    assert!(!session.is_pending());
    assert!(session.take_safe_stop().is_some());
    assert!(
        session
            .accept(reply(&old, decision(execute())), 40_000, 20_000_000)
            .is_err()
    );
    let next = session
        .prepare(observation(1, 2), 1000, 40_000_000)
        .unwrap();
    session.cancel_pending("operator stop");
    assert!(
        session
            .accept(reply(&next, decision(execute())), 1000, 40_000_000)
            .is_err()
    );
}

#[test]
fn failure_feedback_and_decisions_reach_replanning_with_bounded_memory() {
    let mut session = session(TaskProfile::StaticApple, true);
    for frame in 1..=20 {
        let input = session
            .prepare(observation(1, frame), 1000, frame * 20_000_000)
            .unwrap();
        let accepted = session
            .accept(
                reply(&input, decision(SkillRequest::Observe)),
                1000,
                frame * 20_000_000,
            )
            .unwrap();
        session
            .record_feedback(ExecutionFeedback {
                episode_id: 1,
                decision_id: accepted.decision_id,
                status: FeedbackStatus::Failed,
                summary: "Gripper opened without a stable load; reacquire the target.".into(),
                sim_time_ns: frame * 20_000_000,
            })
            .unwrap();
    }
    let next = session
        .prepare(observation(1, 21), 1000, 420_000_000)
        .unwrap();
    assert_eq!(next.feedback.len(), 16);
    assert_eq!(next.recent_decisions.len(), 8);
    assert!(next.feedback.last().unwrap().summary.contains("reacquire"));
    let body = LocalQwenClient::new(Default::default())
        .unwrap()
        .request_body(&next)
        .unwrap();
    assert!(body.to_string().contains("Gripper opened"));
    assert!(session.take_safe_stop().is_some());
    assert!(
        session
            .record_feedback(ExecutionFeedback {
                episode_id: 0,
                decision_id: 20,
                status: FeedbackStatus::Succeeded,
                summary: "old episode".into(),
                sim_time_ns: 0
            })
            .is_err()
    );
}

#[test]
fn repeated_frames_and_forged_model_stamps_are_rejected() {
    let mut session = session(TaskProfile::StaticApple, true);
    let input = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    assert!(matches!(
        session.prepare(observation(1, 2), 1000, 40_000_000),
        Err(DecisionError::Busy)
    ));
    let mut forged = reply(&input, decision(execute()));
    forged.result.as_mut().unwrap().frame_id = 999;
    assert!(session.accept(forged, 1000, 20_000_000).is_err());
    assert!(
        session
            .prepare(observation(1, 1), 1000, 20_000_000)
            .is_err()
    );
}

#[test]
fn worker_is_nonblocking_and_refuses_a_backlog() {
    let mock = serve(
        "200 OK",
        completion(&decision(SkillRequest::Observe), "stop"),
        Duration::from_millis(100),
    );
    let worker = DecisionWorker::spawn(client(&mock.url, Duration::from_secs(1)), 1).unwrap();
    let started = Instant::now();
    worker.try_submit(input()).unwrap();
    assert!(matches!(
        worker.try_submit(input()),
        Err(DecisionError::Busy)
    ));
    assert!(worker.try_recv().unwrap().is_none());
    assert!(started.elapsed() < Duration::from_millis(50));
    mock.handle.join().unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if let Some(reply) = worker.try_recv().unwrap() {
            assert!(reply.result.is_ok());
            break;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!worker.is_busy());
}

#[test]
fn worker_reset_discards_running_old_episode_without_parallel_inference() {
    let mock = serve(
        "200 OK",
        completion(&decision(SkillRequest::Observe), "stop"),
        Duration::from_millis(80),
    );
    let worker = DecisionWorker::spawn(client(&mock.url, Duration::from_secs(1)), 1).unwrap();
    worker.try_submit(input()).unwrap();
    mock.request.recv_timeout(Duration::from_secs(1)).unwrap();
    worker.reset_episode(2).unwrap();
    assert!(worker.try_submit(input()).is_err());
    mock.handle.join().unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while worker.is_busy() {
        assert!(worker.try_recv().unwrap().is_none());
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(worker.try_recv().unwrap().is_none());
}
