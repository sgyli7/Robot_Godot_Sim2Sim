use super::client::PolicyHttpTransport;
use super::{PolicyActionChunk, PolicyInferenceError, PolicyInferenceRequest};
use crate::types::{ObservationStamp, TaskProfile};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Serialize;
use std::time::Duration;

/// Matched gn1_6 adapter, currently scoped to the published brown-box task.
/// The mobile state groups retain their named order; no static-export permutation.
pub struct MobilePolicyClient {
    transport: PolicyHttpTransport,
}

#[derive(Serialize)]
struct MobileN16WireState {
    left_arm: [f32; 7],
    right_arm: [f32; 7],
    left_hand: [f32; 7],
    right_hand: [f32; 7],
    waist: [f32; 3],
}

#[derive(Serialize)]
struct MobileWireRequest<'a> {
    schema: &'static str,
    profile: TaskProfile,
    sequence_id: u64,
    observation: &'a ObservationStamp,
    camera_rgb_b64: String,
    state_groups: MobileN16WireState,
}

impl MobilePolicyClient {
    pub fn new(endpoint: &str, timeout: Duration) -> Result<Self, PolicyInferenceError> {
        Ok(Self {
            transport: PolicyHttpTransport::new(endpoint, timeout)?,
        })
    }

    pub fn infer(
        &self,
        request: &PolicyInferenceRequest,
    ) -> Result<PolicyActionChunk, PolicyInferenceError> {
        let wire = self.wire_request(request)?;
        self.transport.infer_wire(&wire, request)
    }

    fn wire_request<'a>(
        &self,
        request: &'a PolicyInferenceRequest,
    ) -> Result<MobileWireRequest<'a>, PolicyInferenceError> {
        if request.profile != TaskProfile::MobileBox {
            return Err(PolicyInferenceError::UnsupportedProfile);
        }
        request
            .observation
            .validate()
            .map_err(PolicyInferenceError::InvalidObservation)?;
        let q = &request.observation.measured_joint_positions_rad;
        Ok(MobileWireRequest {
            schema: "mobile_observation_v1",
            profile: request.profile,
            sequence_id: request.sequence_id,
            observation: &request.observation.stamp,
            camera_rgb_b64: STANDARD.encode(&request.observation.camera_rgb),
            state_groups: MobileN16WireState {
                left_arm: q[..7].try_into().expect("fixed joint range"),
                right_arm: q[7..14].try_into().expect("fixed joint range"),
                left_hand: q[14..21].try_into().expect("fixed joint range"),
                right_hand: q[21..28].try_into().expect("fixed joint range"),
                waist: q[28..31].try_into().expect("fixed joint range"),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{
        PolicyActionError, PolicyActionFrame, PolicyObservation, profile_contract,
    };
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
        thread,
    };

    fn request() -> PolicyInferenceRequest {
        PolicyInferenceRequest {
            profile: TaskProfile::MobileBox,
            sequence_id: 4,
            observation: PolicyObservation {
                stamp: ObservationStamp {
                    episode_id: 1,
                    frame_id: 2,
                    sim_time_ns: 40_000_000,
                    captured_at_unix_ms: 100,
                },
                camera_rgb: vec![3; 640 * 480 * 3],
                camera_width: 640,
                camera_height: 480,
                measured_joint_positions_rad: std::array::from_fn(|index| (index + 1) as f32),
            },
        }
    }

    fn tcp_reply(
        mutate: impl FnOnce(&mut PolicyActionChunk),
    ) -> Result<PolicyActionChunk, PolicyInferenceError> {
        let request = request();
        let mut reply = PolicyActionChunk {
            profile: request.profile,
            observation: request.observation.stamp,
            sequence_id: request.sequence_id,
            model_revision: profile_contract(request.profile).revision.into(),
            action_period_ns: 20_000_000,
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
                50
            ],
        };
        mutate(&mut reply);
        let payload = serde_json::to_vec(&reply).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/infer", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert_eq!(line, "POST /infer HTTP/1.1\r\n");
            let mut length = None;
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        length = Some(value.trim().parse::<usize>().unwrap());
                    }
                }
            }
            let mut body = vec![0; length.unwrap()];
            reader.read_exact(&mut body).unwrap();
            let wire: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(wire["schema"], "mobile_observation_v1");
            assert_eq!(
                wire["state_groups"]["left_hand"],
                serde_json::json!([15., 16., 17., 18., 19., 20., 21.])
            );
            assert_eq!(
                wire["state_groups"]["right_hand"],
                serde_json::json!([22., 23., 24., 25., 26., 27., 28.])
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            )
            .unwrap();
            stream.write_all(&payload).unwrap();
        });
        let result = MobilePolicyClient::new(&endpoint, Duration::from_secs(2))
            .unwrap()
            .infer(&request);
        server.join().unwrap();
        result
    }

    #[test]
    fn mobile_hands_and_waist_preserve_all_distinct_measured_joint_values() {
        let client =
            MobilePolicyClient::new("http://127.0.0.1:5558/infer", Duration::from_secs(2)).unwrap();
        let request = request();
        let wire = client.wire_request(&request).unwrap();
        assert_eq!(wire.state_groups.left_arm, [1., 2., 3., 4., 5., 6., 7.]);
        assert_eq!(
            wire.state_groups.right_arm,
            [8., 9., 10., 11., 12., 13., 14.]
        );
        assert_eq!(
            wire.state_groups.left_hand,
            [15., 16., 17., 18., 19., 20., 21.]
        );
        assert_eq!(
            wire.state_groups.right_hand,
            [22., 23., 24., 25., 26., 27., 28.]
        );
        assert_eq!(wire.state_groups.waist, [29., 30., 31.]);
        let json = serde_json::to_value(wire).unwrap();
        assert_eq!(json["schema"], "mobile_observation_v1");
        assert_eq!(json["profile"], "mobile_box");
        assert!(json.get("instruction").is_none());
    }

    #[test]
    fn mobile_client_rejects_static_profile_before_network_access() {
        let client =
            MobilePolicyClient::new("http://127.0.0.1:1/infer", Duration::from_secs(1)).unwrap();
        let mut request = request();
        request.profile = TaskProfile::StaticApple;
        assert_eq!(
            client.infer(&request).unwrap_err(),
            PolicyInferenceError::UnsupportedProfile
        );
    }

    #[test]
    fn mobile_tcp_transport_preserves_input_and_accepts_exact_50_frame_contract() {
        assert_eq!(tcp_reply(|_| {}).unwrap().frames.len(), 50);
    }

    #[test]
    fn mobile_tcp_transport_rejects_static_horizon_revision_and_stale_identity() {
        assert_eq!(
            tcp_reply(|chunk| {
                chunk.frames.truncate(40);
            })
            .unwrap_err(),
            PolicyInferenceError::ReplyContract(PolicyActionError::WrongHorizon)
        );
        assert_eq!(
            tcp_reply(|chunk| {
                chunk.model_revision = "unfrozen".into();
            })
            .unwrap_err(),
            PolicyInferenceError::ReplyContract(PolicyActionError::WrongRevision)
        );
        assert_eq!(
            tcp_reply(|chunk| {
                chunk.action_period_ns = 5_000_000;
            })
            .unwrap_err(),
            PolicyInferenceError::ReplyContract(PolicyActionError::WrongPeriod)
        );
        assert_eq!(
            tcp_reply(|chunk| {
                chunk.observation.episode_id += 1;
            })
            .unwrap_err(),
            PolicyInferenceError::ReplyIdentity
        );
    }
}
