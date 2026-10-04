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
        ModelDecision, ObservationSnapshot, RgbTarget, RgbVerifiedTargets, SelectionDecision,
        SelectionInput, SelectionReply, SelectionRequest, TargetKind, TaskGoal, VisualTarget,
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

fn rgb_observation(episode: u64, frame: u64) -> ObservationSnapshot {
    let mut observation = observation(episode, frame);
    observation.camera = CameraRgb::from_rgb("head", 64, 48, vec![127; 64 * 48 * 3]).unwrap();
    observation
}

fn rgb_targets(observation: &ObservationSnapshot) -> RgbVerifiedTargets {
    RgbVerifiedTargets::bind_current_rgb(
        observation,
        vec![
            RgbTarget {
                id: "box_22".into(),
                kind: TargetKind::Box,
                public_description: "brown box marker22".into(),
                marker_id: 22,
                marker_bbox_xyxy: [0.1, 0.2, 0.4, 0.6],
                minimum_edge_px: 12.0,
                reprojection_rms_px: 0.3,
            },
            RgbTarget {
                id: "bin_21".into(),
                kind: TargetKind::PlacementZone,
                public_description: "blue placement zone marker21".into(),
                marker_id: 21,
                marker_bbox_xyxy: [0.5, 0.1, 0.8, 0.4],
                minimum_edge_px: 10.0,
                reprojection_rms_px: 0.1,
            },
        ],
    )
    .unwrap()
}

fn selection(input: &SelectionInput, request: SelectionRequest) -> SelectionReply {
    SelectionReply {
        request_id: input.context.request_id,
        stamp: input.context.observation.stamp,
        elapsed_ms: 100,
        result: Ok(SelectionDecision {
            episode_id: input.context.observation.stamp.episode_id,
            frame_id: input.context.observation.stamp.frame_id,
            request,
            reason: "Visible box and bin".into(),
        }),
    }
}

fn selected_execute() -> SelectionRequest {
    SelectionRequest::ExecuteTask {
        target_id: "box_22".into(),
        destination_id: "bin_21".into(),
    }
}

#[test]
fn rgb_selection_keeps_detector_evidence_distinct_and_carries_execution_feedback() {
    let mut session = session(TaskProfile::MobileBox, true);
    let observation = rgb_observation(1, 1);
    let input = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    let admitted = session
        .accept_selection(selection(&input, selected_execute()), 1100, 20_000_000)
        .unwrap();
    assert!(matches!(
        admitted.request,
        SkillRequest::ExecuteTask {
            task_profile: TaskProfile::MobileBox,
            ..
        }
    ));
    assert_eq!(
        session.targets().count(),
        0,
        "detector measurements must not be minted as model visual claims"
    );
    session
        .record_feedback(ExecutionFeedback {
            episode_id: 1,
            decision_id: admitted.decision_id,
            status: FeedbackStatus::Failed,
            summary: "Visible box moved relative to palms".into(),
            sim_time_ns: 20_000_000,
        })
        .unwrap();
    let next = rgb_observation(1, 2);
    let input = session
        .prepare_selection(next.clone(), rgb_targets(&next), 1200, 40_000_000)
        .unwrap();
    assert_eq!(input.context.feedback.len(), 1);
    assert_eq!(input.context.feedback[0].status, FeedbackStatus::Failed);
    assert_eq!(input.context.recent_decisions.len(), 1);
    assert_eq!(input.verified_targets.stamp().frame_id, 2);
}

#[test]
fn rgb_selection_rejects_camera_substitution_even_with_identical_stamp() {
    let original = rgb_observation(1, 1);
    let targets = rgb_targets(&original);
    let mut substitute = original.clone();
    substitute.camera = CameraRgb::from_rgb("head", 64, 48, vec![126; 64 * 48 * 3]).unwrap();
    let mut session = session(TaskProfile::MobileBox, true);
    assert!(
        session
            .prepare_selection(substitute, targets.clone(), 1000, 20_000_000)
            .is_err()
    );
    assert!(!session.is_pending());
    let mut substitute = original;
    substitute.stamp.frame_id = 2;
    assert!(
        session
            .prepare_selection(substitute, targets, 1000, 20_000_000)
            .is_err()
    );
    assert!(!session.is_pending());
}

#[test]
fn rgb_selection_rejects_invisible_or_incompatible_ids_and_withdrawn_capabilities() {
    for request in [
        SelectionRequest::ExecuteTask {
            target_id: "unseen_box".into(),
            destination_id: "bin_21".into(),
        },
        SelectionRequest::ExecuteTask {
            target_id: "bin_21".into(),
            destination_id: "box_22".into(),
        },
    ] {
        let mut session = session(TaskProfile::MobileBox, true);
        let observation = rgb_observation(1, 1);
        let input = session
            .prepare_selection(
                observation.clone(),
                rgb_targets(&observation),
                1000,
                20_000_000,
            )
            .unwrap();
        assert!(
            session
                .accept_selection(selection(&input, request), 1100, 20_000_000)
                .is_err()
        );
        assert!(session.take_safe_stop().is_some());
    }
    let mut session = session(TaskProfile::MobileBox, false);
    let observation = rgb_observation(1, 1);
    let input = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    assert!(
        session
            .accept_selection(selection(&input, selected_execute()), 1100, 20_000_000)
            .is_err()
    );
}

#[test]
fn rgb_selection_cannot_reuse_expired_reset_or_wrong_wire_results() {
    let limits = DecisionLimits {
        max_frame_age_ms: 20_000,
        max_sim_age_ns: 1_000_000_000,
        target_ttl_ms: 20_000,
    };
    let mut session = DecisionSession::new(
        1,
        goal(TaskProfile::MobileBox),
        SkillAvailability {
            mobile_box: true,
            ..Default::default()
        },
        limits,
    )
    .unwrap();
    let observation = rgb_observation(1, 1);
    let input = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    assert!(session.expire_pending(21_001, 20_000_000));
    assert!(
        session
            .accept_selection(selection(&input, selected_execute()), 21_001, 20_000_000)
            .is_err()
    );
    session.reset(2, goal(TaskProfile::MobileBox)).unwrap();
    assert!(
        session
            .accept_selection(selection(&input, selected_execute()), 1100, 20_000_000)
            .is_err()
    );
    let observation = rgb_observation(2, 1);
    let input = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    assert!(
        session
            .accept(
                reply(&input.context, decision(SkillRequest::Observe)),
                1100,
                20_000_000
            )
            .is_err()
    );
    assert!(!session.is_pending());
    assert!(session.take_safe_stop().is_some());
}

#[test]
fn rgb_selection_http_omits_repeated_detections_and_excludes_unknown_target_ids() {
    let mut session = session(TaskProfile::MobileBox, true);
    let observation = rgb_observation(1, 1);
    let input = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    let selected = selection(&input, selected_execute()).result.unwrap();
    let body = json!({"choices":[{"message":{"content":serde_json::to_string(&selected).unwrap()},"finish_reason":"stop"}]}).to_string();
    let mock = serve("200 OK", body, Duration::ZERO);
    let worker = DecisionWorker::spawn(client(&mock.url, Duration::from_secs(2)), 1).unwrap();
    worker.try_submit_selection(input.clone()).unwrap();
    assert!(matches!(
        worker.try_submit(input.context.clone()),
        Err(DecisionError::Busy)
    ));
    let deadline = Instant::now() + Duration::from_secs(3);
    let reply = loop {
        assert!(
            Instant::now() < deadline,
            "bounded selector worker never returned"
        );
        if let Some(reply) = worker.try_recv_selection().unwrap() {
            break reply;
        }
        thread::sleep(Duration::from_millis(2));
    };
    session.accept_selection(reply, 1100, 20_000_000).unwrap();
    let request = mock.request.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(request["max_tokens"].as_u64().unwrap() <= 128);
    let schema = &request["response_format"]["json_schema"]["schema"];
    assert!(schema["properties"].get("observed_targets").is_none());
    let execute = schema["properties"]["request"]["anyOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["properties"]["action"]["const"] == "execute_task")
        .unwrap();
    assert_eq!(
        execute["properties"]["target_id"]["enum"],
        json!(["box_22"])
    );
    assert_eq!(
        execute["properties"]["destination_id"]["enum"],
        json!(["bin_21"])
    );
    assert!(execute["properties"].get("task_profile").is_none());
    for variant in schema["properties"]["request"]["anyOf"].as_array().unwrap() {
        assert_eq!(
            variant["properties"]
                .as_object()
                .unwrap()
                .keys()
                .next()
                .unwrap(),
            "action",
            "the pinned ordered JSON grammar must choose an action before branch-specific arguments"
        );
    }
    let context: Value = serde_json::from_str(
        request["messages"][1]["content"][0]["text"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        context["rgb_verified_targets"]["image_sha256"],
        input.verified_targets.image_sha256()
    );
    assert!(
        context["rgb_verified_targets"]["targets"][0]
            .get("world_position")
            .is_none()
    );
    assert_eq!(worker.transport_counts(), (1, 1, 0));
    mock.handle.join().unwrap();
}

#[test]
fn delayed_rgb_selection_cannot_cancel_a_new_episode_request() {
    let mut session = session(TaskProfile::MobileBox, true);
    let first = rgb_observation(1, 1);
    let old = session
        .prepare_selection(first.clone(), rgb_targets(&first), 1000, 20_000_000)
        .unwrap();
    session.reset(2, goal(TaskProfile::MobileBox)).unwrap();
    let next = rgb_observation(2, 1);
    let new = session
        .prepare_selection(next.clone(), rgb_targets(&next), 1000, 20_000_000)
        .unwrap();
    assert!(
        session
            .accept_selection(selection(&old, selected_execute()), 1100, 20_000_000)
            .is_err()
    );
    assert!(session.is_pending());
    assert!(
        session
            .accept_selection(selection(&new, selected_execute()), 1100, 20_000_000)
            .is_ok()
    );
}

#[test]
fn rgb_worker_discards_running_old_episode_and_never_starts_parallel_http() {
    let mut session = session(TaskProfile::MobileBox, true);
    let observation = rgb_observation(1, 1);
    let old = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    let decision = selection(&old, selected_execute()).result.unwrap();
    let body = json!({"choices":[{"message":{"content":serde_json::to_string(&decision).unwrap()},"finish_reason":"stop"}]}).to_string();
    let mock = serve("200 OK", body, Duration::from_millis(80));
    let worker = DecisionWorker::spawn(client(&mock.url, Duration::from_secs(2)), 1).unwrap();
    worker.try_submit_selection(old).unwrap();
    mock.request.recv_timeout(Duration::from_secs(1)).unwrap();
    worker.reset_episode(2).unwrap();
    session.reset(2, goal(TaskProfile::MobileBox)).unwrap();
    let observation = rgb_observation(2, 1);
    let new = session
        .prepare_selection(
            observation.clone(),
            rgb_targets(&observation),
            1000,
            20_000_000,
        )
        .unwrap();
    assert!(matches!(
        worker.try_submit_selection(new),
        Err(DecisionError::Busy)
    ));
    mock.handle.join().unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while worker.is_busy() {
        assert!(Instant::now() < deadline);
        assert!(worker.try_recv_selection().unwrap().is_none());
        thread::sleep(Duration::from_millis(2));
    }
    assert!(worker.try_recv_selection().unwrap().is_none());
    assert_eq!(worker.transport_counts(), (1, 1, 1));
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
fn generated_choices_match_current_task_and_capabilities() {
    let client = LocalQwenClient::new(LocalQwenConfig::default()).unwrap();
    for profile in [TaskProfile::StaticApple, TaskProfile::MobileBox] {
        for flags in 0..8 {
            let availability = SkillAvailability {
                static_apple: flags & 1 != 0,
                mobile_box: flags & 2 != 0,
                navigate_adjustment: flags & 4 != 0,
            };
            let mut input = input();
            input.goal.profile = profile;
            input.available_skills = availability;
            let body = client.request_body(&input).unwrap();
            let schema = &body["response_format"]["json_schema"]["schema"];
            let variants = schema["properties"]["request"]["anyOf"].as_array().unwrap();
            let skills: Vec<_> = variants
                .iter()
                .map(|variant| variant["properties"]["skill"]["const"].as_str().unwrap())
                .collect();
            assert!(skills.contains(&"observe") && skills.contains(&"stop"));
            let can_execute = match profile {
                TaskProfile::StaticApple => availability.static_apple,
                TaskProfile::MobileBox => availability.mobile_box,
            };
            let can_navigate = profile == TaskProfile::MobileBox
                && availability.mobile_box
                && availability.navigate_adjustment;
            assert_eq!(
                skills.contains(&"execute_task"),
                can_execute,
                "{profile:?}/{flags}"
            );
            assert_eq!(
                skills.contains(&"navigate_adjustment"),
                can_navigate,
                "{profile:?}/{flags}"
            );
            assert_eq!(
                skills.len(),
                2 + usize::from(can_execute) + usize::from(can_navigate)
            );
            if let Some(execute) = variants
                .iter()
                .find(|variant| variant["properties"]["skill"]["const"] == "execute_task")
            {
                assert_eq!(
                    execute["properties"]["task_profile"]["enum"],
                    json!([profile])
                );
            }
            if !can_execute && !can_navigate {
                assert_eq!(schema["properties"]["observed_targets"]["maxItems"], 0);
            }
        }
    }
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
fn withdrawing_execution_retains_feedback_and_rejects_another_physical_request() {
    let mut session = session(TaskProfile::StaticApple, true);
    let first = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    let admitted = session
        .accept(reply(&first, decision(execute())), 1100, 20_000_000)
        .unwrap();
    session
        .record_feedback(ExecutionFeedback {
            episode_id: 1,
            decision_id: admitted.decision_id,
            status: FeedbackStatus::Running,
            summary: "Motion ended; inspect the next RGB before reporting task success.".into(),
            sim_time_ns: 40_000_000,
        })
        .unwrap();
    session.revoke_physical_skills();
    let next = session
        .prepare(observation(1, 2), 1200, 40_000_000)
        .unwrap();
    assert_eq!(next.available_skills, SkillAvailability::default());
    assert_eq!(next.remembered_targets.len(), 2);
    assert_eq!(next.feedback.len(), 1);
    assert!(
        session
            .accept(reply(&next, decision(execute())), 1300, 40_000_000)
            .is_err()
    );
    assert!(session.take_safe_stop().is_some());
    let last = session
        .prepare(observation(1, 3), 1400, 60_000_000)
        .unwrap();
    let mut observe = decision(SkillRequest::Observe);
    observe.observed_targets.clear();
    assert_eq!(
        session
            .accept(reply(&last, observe), 1500, 60_000_000)
            .unwrap()
            .request,
        SkillRequest::Observe
    );
}

#[test]
fn withdrawing_capabilities_cancels_pending_reply_without_resetting_frame_identity() {
    let mut session = session(TaskProfile::StaticApple, true);
    let pending = session
        .prepare(observation(1, 1), 1000, 20_000_000)
        .unwrap();
    session.revoke_physical_skills();
    assert!(!session.is_pending());
    assert!(session.take_safe_stop().is_some());
    assert!(
        session
            .accept(reply(&pending, decision(execute())), 1100, 20_000_000)
            .is_err()
    );
    assert!(
        session
            .prepare(observation(1, 1), 1100, 20_000_000)
            .is_err()
    );
    let fresh = session
        .prepare(observation(1, 2), 1200, 40_000_000)
        .unwrap();
    assert!(fresh.request_id > pending.request_id);
    assert_eq!(fresh.available_skills, SkillAvailability::default());
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
