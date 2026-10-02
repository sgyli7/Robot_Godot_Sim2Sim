use super::{
    ARENA_ACTION_PERIOD_NS, PolicyActionChunk, PolicyActionError, PolicyObservation,
    PolicyObservationError, StaticOnnxState, profile_contract,
};
use crate::types::{ObservationStamp, TaskProfile};
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::blocking::Client;
use serde::Serialize;
use std::{io::Read, time::Duration};

const MAX_REPLY_BYTES: u64 = 1_048_576;

/// A single bounded inference request. Call from a worker, never a physics system.
#[derive(Clone, Debug)]
pub struct PolicyInferenceRequest {
    pub profile: TaskProfile,
    pub sequence_id: u64,
    pub observation: PolicyObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyInferenceError {
    InvalidEndpoint,
    InvalidTimeout,
    UnsupportedProfile,
    InvalidObservation(PolicyObservationError),
    Transport(String),
    HttpStatus(u16),
    ReplyTooLarge,
    InvalidReply(String),
    ReplyIdentity,
    ReplyContract(PolicyActionError),
}

/// Shared bounded HTTP mechanics; each public client fixes its own profile and wire schema.
pub(super) struct PolicyHttpTransport {
    client: Client,
    endpoint: reqwest::Url,
    timeout: Duration,
}

/// Fixed static-export adapter; it cannot send requests to the mobile profile.
pub struct StaticPolicyClient {
    transport: PolicyHttpTransport,
}

#[derive(Serialize)]
struct StaticWireRequest<'a> {
    schema: &'static str,
    profile: TaskProfile,
    sequence_id: u64,
    observation: &'a ObservationStamp,
    camera_rgb_b64: String,
    state_groups: StaticOnnxState,
}

impl StaticPolicyClient {
    pub fn new(endpoint: &str, timeout: Duration) -> Result<Self, PolicyInferenceError> {
        Ok(Self {
            transport: PolicyHttpTransport::new(endpoint, timeout)?,
        })
    }

    pub fn infer(
        &self,
        request: &PolicyInferenceRequest,
    ) -> Result<PolicyActionChunk, PolicyInferenceError> {
        if request.profile != TaskProfile::StaticApple {
            return Err(PolicyInferenceError::UnsupportedProfile);
        }
        let state_groups = request
            .observation
            .static_onnx_state()
            .map_err(PolicyInferenceError::InvalidObservation)?;
        let wire = StaticWireRequest {
            schema: "unitree_g1_static_observation_v2",
            profile: request.profile,
            sequence_id: request.sequence_id,
            observation: &request.observation.stamp,
            camera_rgb_b64: STANDARD.encode(&request.observation.camera_rgb),
            state_groups,
        };
        self.transport.infer_wire(&wire, request)
    }
}

impl PolicyHttpTransport {
    pub(super) fn new(endpoint: &str, timeout: Duration) -> Result<Self, PolicyInferenceError> {
        let endpoint =
            reqwest::Url::parse(endpoint).map_err(|_| PolicyInferenceError::InvalidEndpoint)?;
        if endpoint.scheme() != "http"
            || endpoint.host_str() != Some("127.0.0.1")
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.path() != "/infer"
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(PolicyInferenceError::InvalidEndpoint);
        }
        if timeout.is_zero() || timeout > Duration::from_secs(20) {
            return Err(PolicyInferenceError::InvalidTimeout);
        }
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .timeout(timeout)
            .build()
            .map_err(|e| PolicyInferenceError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            endpoint,
            timeout,
        })
    }

    pub(super) fn infer_wire(
        &self,
        wire: &impl Serialize,
        request: &PolicyInferenceRequest,
    ) -> Result<PolicyActionChunk, PolicyInferenceError> {
        let response = self
            .client
            .post(self.endpoint.clone())
            // The blocking builder timeout applies again on each Read. Set the
            // request timeout too so the async body retains one total deadline.
            .timeout(self.timeout)
            .json(wire)
            .send()
            .map_err(|e| PolicyInferenceError::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(PolicyInferenceError::HttpStatus(response.status().as_u16()));
        }
        if response
            .content_length()
            .is_some_and(|n| n > MAX_REPLY_BYTES)
        {
            return Err(PolicyInferenceError::ReplyTooLarge);
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_REPLY_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| PolicyInferenceError::Transport(e.to_string()))?;
        if bytes.len() as u64 > MAX_REPLY_BYTES {
            return Err(PolicyInferenceError::ReplyTooLarge);
        }
        let reply: PolicyActionChunk = serde_json::from_slice(&bytes)
            .map_err(|e| PolicyInferenceError::InvalidReply(e.to_string()))?;
        if reply.profile != request.profile
            || reply.sequence_id != request.sequence_id
            || reply.observation != request.observation.stamp
        {
            return Err(PolicyInferenceError::ReplyIdentity);
        }
        let contract = profile_contract(request.profile);
        if reply.model_revision != contract.revision {
            return Err(PolicyInferenceError::ReplyContract(
                PolicyActionError::WrongRevision,
            ));
        }
        if reply.action_period_ns != ARENA_ACTION_PERIOD_NS {
            return Err(PolicyInferenceError::ReplyContract(
                PolicyActionError::WrongPeriod,
            ));
        }
        if reply.frames.len() != contract.action_horizon {
            return Err(PolicyInferenceError::ReplyContract(
                PolicyActionError::WrongHorizon,
            ));
        }
        if reply.frames.iter().any(|frame| {
            !frame
                .joint_targets()
                .iter()
                .chain(&frame.navigate_mps_rps)
                .chain(std::iter::once(&frame.base_height_m))
                .all(|value| value.is_finite())
        }) {
            return Err(PolicyInferenceError::ReplyContract(
                PolicyActionError::NonFinite,
            ));
        }
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyActionFrame;
    use std::{
        io::{BufRead, BufReader, Write},
        net::{TcpListener, TcpStream},
        thread,
    };

    fn request() -> PolicyInferenceRequest {
        PolicyInferenceRequest {
            profile: TaskProfile::StaticApple,
            sequence_id: 17,
            observation: PolicyObservation {
                stamp: ObservationStamp {
                    episode_id: 3,
                    frame_id: 9,
                    sim_time_ns: 40_000_000,
                    captured_at_unix_ms: 100,
                },
                camera_rgb: vec![3; 640 * 480 * 3],
                camera_width: 640,
                camera_height: 480,
                measured_joint_positions_rad: [0.0; 31],
            },
        }
    }

    fn chunk() -> PolicyActionChunk {
        let request = request();
        PolicyActionChunk {
            profile: request.profile,
            observation: request.observation.stamp,
            sequence_id: request.sequence_id,
            model_revision: profile_contract(request.profile).revision.into(),
            action_period_ns: ARENA_ACTION_PERIOD_NS,
            frames: vec![
                PolicyActionFrame {
                    left_arm: [0.0; 7],
                    right_arm: [0.0; 7],
                    left_hand: [0.0; 7],
                    right_hand: [0.0; 7],
                    waist: [0.0; 3],
                    base_height_m: 0.75,
                    navigate_mps_rps: [0.0; 3],
                };
                40
            ],
        }
    }

    fn read_request(stream: &TcpStream) {
        let mut reader = BufReader::new(stream);
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        assert_eq!(header, "POST /infer HTTP/1.1\r\n");
        let mut length = None;
        loop {
            header.clear();
            reader.read_line(&mut header).unwrap();
            if header == "\r\n" {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
            }
        }
        let mut body = vec![0; length.unwrap()];
        reader.read_exact(&mut body).unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(wire["sequence_id"], 17);
        assert_eq!(
            wire["camera_rgb_b64"].as_str().unwrap().len(),
            640 * 480 * 4
        );
    }

    fn transport(
        responder: impl FnOnce(TcpStream) + Send + 'static,
        timeout: Duration,
    ) -> Result<PolicyActionChunk, PolicyInferenceError> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/infer", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            read_request(&stream);
            responder(stream);
        });
        let result = StaticPolicyClient::new(&endpoint, timeout)
            .unwrap()
            .infer(&request());
        server.join().unwrap();
        result
    }

    fn json_reply(chunk: &PolicyActionChunk) -> Result<PolicyActionChunk, PolicyInferenceError> {
        let payload = serde_json::to_vec(chunk).unwrap();
        transport(
            move |mut stream| {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                )
                .unwrap();
                stream.write_all(&payload).unwrap();
            },
            Duration::from_secs(2),
        )
    }

    #[test]
    fn policy_transport_is_loopback_only_without_redirects_or_credentials() {
        for endpoint in [
            "https://127.0.0.1:5557/infer",
            "http://example.com/infer",
            "http://user@127.0.0.1/infer",
            "http://127.0.0.1/infer?next=remote",
        ] {
            assert!(matches!(
                StaticPolicyClient::new(endpoint, Duration::from_secs(1)),
                Err(PolicyInferenceError::InvalidEndpoint)
            ));
        }
        for timeout in [Duration::ZERO, Duration::from_secs(21)] {
            assert!(matches!(
                StaticPolicyClient::new("http://127.0.0.1:5557/infer", timeout),
                Err(PolicyInferenceError::InvalidTimeout)
            ));
        }
    }

    #[test]
    fn real_tcp_transport_validates_profile_sequence_and_complete_stamp() {
        assert_eq!(json_reply(&chunk()).unwrap().frames.len(), 40);
        let mut reply = chunk();
        reply.profile = TaskProfile::MobileBox;
        assert_eq!(
            json_reply(&reply).unwrap_err(),
            PolicyInferenceError::ReplyIdentity
        );
        reply = chunk();
        reply.sequence_id += 1;
        assert_eq!(
            json_reply(&reply).unwrap_err(),
            PolicyInferenceError::ReplyIdentity
        );
        reply = chunk();
        reply.observation.captured_at_unix_ms += 1;
        assert_eq!(
            json_reply(&reply).unwrap_err(),
            PolicyInferenceError::ReplyIdentity
        );
    }

    #[test]
    fn real_tcp_transport_rejects_wrong_revision_period_and_horizon() {
        let mut reply = chunk();
        reply.model_revision = "unfrozen".into();
        assert_eq!(
            json_reply(&reply).unwrap_err(),
            PolicyInferenceError::ReplyContract(PolicyActionError::WrongRevision)
        );
        reply = chunk();
        reply.action_period_ns = 5_000_000;
        assert_eq!(
            json_reply(&reply).unwrap_err(),
            PolicyInferenceError::ReplyContract(PolicyActionError::WrongPeriod)
        );
        reply = chunk();
        reply.frames.pop();
        assert_eq!(
            json_reply(&reply).unwrap_err(),
            PolicyInferenceError::ReplyContract(PolicyActionError::WrongHorizon)
        );
    }

    #[test]
    fn real_tcp_transport_rejects_malformed_frame_shape() {
        let mut reply = serde_json::to_value(chunk()).unwrap();
        reply["frames"][0]["left_arm"] = serde_json::json!([0, 0]);
        let payload = serde_json::to_vec(&reply).unwrap();
        let result = transport(
            move |mut stream| {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                    payload.len()
                )
                .unwrap();
                stream.write_all(&payload).unwrap();
            },
            Duration::from_secs(2),
        );
        assert!(matches!(result, Err(PolicyInferenceError::InvalidReply(_))));
    }

    #[test]
    fn real_tcp_transport_bounds_declared_and_streamed_reply_size() {
        let result = transport(
            |mut stream| {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                    MAX_REPLY_BYTES + 1
                )
                .unwrap();
            },
            Duration::from_secs(2),
        );
        assert_eq!(result.unwrap_err(), PolicyInferenceError::ReplyTooLarge);
        let result = transport(
            |mut stream| {
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                    .unwrap();
                // The client may close immediately after reading its bounded prefix.
                let _ = stream.write_all(&vec![b' '; MAX_REPLY_BYTES as usize + 1]);
            },
            Duration::from_secs(2),
        );
        assert_eq!(result.unwrap_err(), PolicyInferenceError::ReplyTooLarge);
    }

    #[test]
    fn real_tcp_transport_times_out_and_does_not_follow_redirect() {
        let result = transport(
            |_stream| thread::sleep(Duration::from_millis(200)),
            Duration::from_millis(100),
        );
        assert!(matches!(result, Err(PolicyInferenceError::Transport(_))));
        let result = transport(
            |mut stream| {
                stream.write_all(b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://example.com/infer\r\nContent-Length: 0\r\n\r\n").unwrap();
            },
            Duration::from_secs(2),
        );
        assert_eq!(result.unwrap_err(), PolicyInferenceError::HttpStatus(307));
    }

    #[test]
    fn slow_drip_body_cannot_extend_the_total_request_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/infer", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            read_request(&stream);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .unwrap();
            // Every read can complete within 25ms, but the complete body takes
            // >=500ms. A per-read-only 200ms timeout would never reject it.
            for _ in 0..20 {
                if stream.write_all(b" ").is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
        });
        let client = StaticPolicyClient::new(&endpoint, Duration::from_millis(200)).unwrap();
        let started = std::time::Instant::now();
        let result = client.infer(&request());
        let elapsed = started.elapsed();
        server.join().unwrap();
        assert!(matches!(result, Err(PolicyInferenceError::Transport(_))));
        assert!(
            elapsed < Duration::from_millis(450),
            "deadline extended to {elapsed:?}"
        );
    }

    #[test]
    fn mobile_profile_is_rejected_before_network_access() {
        let client =
            StaticPolicyClient::new("http://127.0.0.1:1/infer", Duration::from_secs(1)).unwrap();
        let mut request = request();
        request.profile = TaskProfile::MobileBox;
        assert_eq!(
            client.infer(&request).unwrap_err(),
            PolicyInferenceError::UnsupportedProfile
        );
    }
}
