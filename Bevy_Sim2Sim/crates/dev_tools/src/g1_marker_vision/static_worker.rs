//! Static-only CPU transport. It receives pixels and self sensors, never a world.

use super::digest;
use rendering_minigame::g1_camera::{CameraPoseSource, G1CameraMountProfile, G1CapturedRgb};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Mutex,
        mpsc::{self, Receiver, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use task_minigame::{decision::CameraRgb, types::ObservationStamp};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StaticMarkerVisionConfiguration {
    pub python_path: PathBuf,
    pub python_sha256: String,
    pub worker_path: PathBuf,
    pub worker_sha256: String,
    pub localizer_path: PathBuf,
    pub localizer_sha256: String,
    pub definition_path: PathBuf,
    pub definition_sha256: String,
    pub fiducial_path: PathBuf,
    pub fiducial_sha256: String,
}

impl StaticMarkerVisionConfiguration {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.definition_sha256
            != "571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd"
            || self.fiducial_sha256
                != "1718b8c54289b5dcbb6e3674edc176f03b1f40bdc1e507f62538ae13f336fb74"
            || self.worker_path.parent() != self.localizer_path.parent()
            || self.worker_path.file_name().and_then(|n| n.to_str())
                != Some("unitree_g1_static_vision_worker.py")
            || self.localizer_path.file_name().and_then(|n| n.to_str())
                != Some("unitree_g1_static_vision.py")
        {
            return Err(
                "static CPU port requires its distinct original model/calibration/program".into(),
            );
        }
        for (path, hash, maximum) in [
            (&self.python_path, &self.python_sha256, 64 * 1024 * 1024),
            (&self.worker_path, &self.worker_sha256, 128 * 1024),
            (&self.localizer_path, &self.localizer_sha256, 128 * 1024),
            (
                &self.definition_path,
                &self.definition_sha256,
                32 * 1024 * 1024,
            ),
            (&self.fiducial_path, &self.fiducial_sha256, 16 * 1024),
        ] {
            if !path.is_absolute() || digest(path, maximum)? != *hash {
                return Err("static CPU port path/hash mismatch".into());
            }
        }
        Ok(())
    }
}

struct Pending {
    observation: ObservationStamp,
    image_hash: String,
    input_hash: String,
    started: Instant,
    grip_check: bool,
}

pub(crate) struct StaticMarkerWorker {
    config: StaticMarkerVisionConfiguration,
    child: Child,
    input: ChildStdin,
    replies: Mutex<Receiver<Result<Value, String>>>,
    reader: Option<JoinHandle<()>>,
    root: PathBuf,
    episode_id: u64,
    started: Instant,
    ready: Option<Value>,
    pending: Option<Pending>,
    submitted: bool,
    grip_submitted: bool,
    previous_observation: Option<ObservationStamp>,
}

impl StaticMarkerWorker {
    pub(crate) fn spawn(
        config: StaticMarkerVisionConfiguration,
        root: &Path,
        episode_id: u64,
    ) -> Result<Self, String> {
        config.validate()?;
        if episode_id == 0 {
            return Err("static CPU episode is zero".into());
        }
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("static_marker_worker.log"))
            .map_err(|e| e.to_string())?;
        let started = Instant::now();
        let mut child = Command::new(&config.python_path)
            .arg("-u")
            .arg(&config.worker_path)
            .arg("--definition")
            .arg(&config.definition_path)
            .arg("--fiducials")
            .arg(&config.fiducial_path)
            .arg("--fiducial-sha256")
            .arg(&config.fiducial_sha256)
            .arg("--output-root")
            .arg(&root)
            .arg("--episode")
            .arg(episode_id.to_string())
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log)
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("static CPU stdin absent")?;
        let stdout = child.stdout.take().ok_or("static CPU stdout absent")?;
        let (sender, replies) = mpsc::sync_channel(2);
        let reader = thread::Builder::new()
            .name("g1-static-marker-replies".into())
            .spawn(move || {
                let mut lines = BufReader::new(stdout);
                loop {
                    let mut bytes = Vec::new();
                    let value = match (&mut lines)
                        .take(128 * 1024 + 1)
                        .read_until(b'\n', &mut bytes)
                    {
                        Ok(0) => Err("static CPU reply pipe closed".into()),
                        Ok(_) if bytes.len() > 128 * 1024 || !bytes.ends_with(b"\n") => {
                            Err("static CPU reply exceeds line budget".into())
                        }
                        Ok(_) => serde_json::from_slice(&bytes).map_err(|e| e.to_string()),
                        Err(e) => Err(e.to_string()),
                    };
                    let terminal = value.is_err();
                    if sender.try_send(value).is_err() || terminal {
                        break;
                    }
                }
            })
            .map_err(|e| {
                let _ = child.kill();
                let _ = child.wait();
                e.to_string()
            })?;
        Ok(Self {
            config,
            child,
            input,
            replies: Mutex::new(replies),
            reader: Some(reader),
            root,
            episode_id,
            started,
            ready: None,
            pending: None,
            submitted: false,
            grip_submitted: false,
            previous_observation: None,
        })
    }

    pub(crate) fn ready(&self) -> bool {
        self.ready.is_some()
    }

    pub(crate) fn provenance(&self) -> Value {
        json!({"owned_pid":self.child.id(),"episode_id":self.episode_id,"ready":self.ready,
            "single_capture_submitted":self.submitted,"pending":self.pending.is_some(),
            "grip_capture_submitted":self.grip_submitted,
            "world_or_contact_truth_input":false,"actuation_proposed":false})
    }

    pub(crate) fn poll(&mut self) -> Result<Option<Value>, String> {
        if self
            .pending
            .as_ref()
            .is_some_and(|p| p.started.elapsed() > Duration::from_secs(3))
        {
            return Err("static CPU request deadline exceeded".into());
        }
        let message = match self
            .replies
            .lock()
            .map_err(|_| "static CPU reply slot poisoned")?
            .try_recv()
        {
            Ok(message) => message?,
            Err(TryRecvError::Empty) => {
                if (self.ready.is_none() && self.started.elapsed() > Duration::from_secs(3))
                    || self
                        .pending
                        .as_ref()
                        .is_some_and(|p| p.started.elapsed() > Duration::from_secs(3))
                {
                    return Err("static CPU preparation/request deadline exceeded".into());
                }
                return Ok(None);
            }
            Err(TryRecvError::Disconnected) => return Err("static CPU reader disconnected".into()),
        };
        if message["schema"] != "g1_persistent_static_marker_worker_v1" {
            return Err("foreign static CPU reply protocol".into());
        }
        if message["event"] == "ready" {
            if self.ready.is_some()
                || self.pending.is_some()
                || message["episode_id"] != self.episode_id
                || message["pid"] != self.child.id()
                || message["worker_sha256"] != self.config.worker_sha256
                || message["localizer_sha256"] != self.config.localizer_sha256
                || message["task_qualified"] != false
            {
                return Err("static CPU preparation identity mismatch".into());
            }
            self.ready = Some(message);
            return Ok(None);
        }
        let pending = self.pending.take().ok_or("unsolicited static CPU result")?;
        let observation: ObservationStamp =
            serde_json::from_value(message["observation"].clone()).map_err(|e| e.to_string())?;
        if observation != pending.observation {
            return Err("static CPU result changed original image stamp".into());
        }
        if message["event"] == "error" {
            return Err(format!(
                "static RGB localization rejected: {}",
                message["error"]
            ));
        }
        if message["event"] != "result"
            || message["world_or_contact_truth_input"] != false
            || message["actuation_proposed"] != false
            || message["localization_seconds"]
                .as_f64()
                .is_none_or(|s| !s.is_finite() || !(0. ..=3.).contains(&s))
        {
            return Err("invalid static CPU result envelope".into());
        }
        validate_reply(&message["result"], &pending, &self.config)?;
        let received_at_unix_ms = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_millis(),
        )
        .map_err(|e| e.to_string())?;
        let image_age_ms = received_at_unix_ms
            .checked_sub(observation.captured_at_unix_ms)
            .ok_or("static image wall timestamp is in the future")?;
        Ok(Some(
            json!({"localization":message["result"],"localization_seconds":message["localization_seconds"],
            "roundtrip_seconds":pending.started.elapsed().as_secs_f64(),"provenance":self.provenance(),
            "reply_received_at_unix_ms":received_at_unix_ms,"image_to_localization_ms":image_age_ms,
            "task_qualified":false,"observation_restamped":false}),
        ))
    }

    pub(crate) fn submit_capture(&mut self, frame: &G1CapturedRgb) -> Result<(), String> {
        self.submit(frame, false)
    }

    /// One additional actual image after the fixed lift; apple31 is required.
    /// The initial image still requires both original target identities.
    pub(crate) fn submit_grip_capture(&mut self, frame: &G1CapturedRgb) -> Result<(), String> {
        self.submit(frame, true)
    }

    fn submit(&mut self, frame: &G1CapturedRgb, grip_check: bool) -> Result<(), String> {
        if !self.ready()
            || self.pending.is_some()
            || (!grip_check && self.submitted)
            || (grip_check && (!self.submitted || self.grip_submitted))
            || frame.stamp.episode_id != self.episode_id
            || frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.mount_profile != G1CameraMountProfile::ArenaEgo
            || frame.width != 640
            || frame.height != 480
            || frame.stamp.source_ticks != [frame.stamp.sim_time_ns / 20_000_000; 2]
            || frame.stamp.sim_time_ns % 20_000_000 != 0
        {
            return Err(
                "static CPU input is unready, repeated, foreign or not native ego RGB".into(),
            );
        }
        let state = frame
            .stamp
            .native_state
            .as_ref()
            .ok_or("static CPU self sensors absent")?;
        state.validate()?;
        if state.body_frame.episode_id != frame.stamp.episode_id
            || state.body_frame.source_tick * 20_000_000 != frame.stamp.sim_time_ns
        {
            return Err("static CPU self snapshot changed its native frame".into());
        }
        let observation = ObservationStamp {
            episode_id: self.episode_id,
            frame_id: frame.stamp.capture_sequence,
            sim_time_ns: frame.stamp.sim_time_ns,
            captured_at_unix_ms: frame.stamp.captured_at_unix_ms,
        };
        if observation.frame_id == 0 || observation.captured_at_unix_ms == 0 {
            return Err("static RGB frame identity is zero".into());
        }
        if self.previous_observation.is_some_and(|previous| {
            observation.frame_id <= previous.frame_id
                || observation.sim_time_ns <= previous.sim_time_ns
                || observation.captured_at_unix_ms <= previous.captured_at_unix_ms
        }) {
            return Err("static CPU image identity did not advance".into());
        }
        let directory = self.root.join(if grip_check {
            "static_grip_vision_input"
        } else {
            "static_vision_input"
        });
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let rgb = CameraRgb::from_rgb(
            "native_static_ego",
            frame.width,
            frame.height,
            frame.rgb.clone(),
        )
        .map_err(|e| e.to_string())?;
        fs::write(directory.join("ego.png"), rgb.png()).map_err(|e| e.to_string())?;
        fs::write(directory.join("observation.json"), serde_json::to_vec_pretty(&json!({
            "schema":"g1_static_marker_observation_v1","stamp":observation,
            "camera":{"fx":458.1245526,"fy":458.1245526,"cx":320.,"cy":240.,"near_m":0.1,"far_m":5.,
                "vertical_fov_radians":2. * (240_f64 / 458.1245526).atan()},
            "measured_joints":state.measured_joints,
        })).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let image_hash = digest(&directory.join("ego.png"), 16 * 1024 * 1024)?;
        let input_hash = digest(&directory.join("observation.json"), 128 * 1024)?;
        let mut bytes =
            serde_json::to_vec(&json!({"schema":"g1_persistent_static_marker_request_v1",
            "directory":directory,"observation":observation}))
            .map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        if bytes.len() > 8192 {
            return Err("static CPU request exceeds byte budget".into());
        }
        let started = Instant::now();
        self.input.write_all(&bytes).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        self.pending = Some(Pending {
            observation,
            image_hash,
            input_hash,
            started,
            grip_check,
        });
        self.previous_observation = Some(observation);
        if grip_check {
            self.grip_submitted = true;
        } else {
            self.submitted = true;
        }
        Ok(())
    }
}

fn validate_reply(
    reply: &Value,
    pending: &Pending,
    config: &StaticMarkerVisionConfiguration,
) -> Result<(), String> {
    let fields = [
        "schema",
        "observation",
        "source",
        "image_sha256",
        "input_sha256",
        "definition_sha256",
        "fiducial_sha256",
        "camera_profile",
        "detections",
        "rejected_marker_candidates",
        "world_or_contact_truth_input",
        "actuation_proposed",
        "task_qualified",
    ];
    let object = reply
        .as_object()
        .ok_or("static RGB result is not an object")?;
    if object.len() != fields.len() || fields.iter().any(|f| !object.contains_key(*f)) {
        return Err("static RGB result has foreign fields".into());
    }
    let actual: ObservationStamp =
        serde_json::from_value(reply["observation"].clone()).map_err(|e| e.to_string())?;
    if reply["schema"] != "g1_static_actual_rgb_localization_v1"
        || actual != pending.observation
        || reply["image_sha256"] != pending.image_hash
        || reply["input_sha256"] != pending.input_hash
        || reply["definition_sha256"] != config.definition_sha256
        || reply["fiducial_sha256"] != config.fiducial_sha256
        || reply["camera_profile"] != "arena_ego"
        || reply["world_or_contact_truth_input"] != false
        || reply["actuation_proposed"] != false
        || reply["task_qualified"] != false
    {
        return Err("static RGB result is foreign or not bound to pixels/self sensors".into());
    }
    let detections = reply["detections"]
        .as_array()
        .ok_or("static detections absent")?;
    let mut admitted = [false; 2];
    for detection in detections {
        let index = match (
            detection["marker_id"].as_u64(),
            detection["object_kind"].as_str(),
        ) {
            (Some(31), Some("t1_apple")) => 0,
            (Some(32), Some("t1_plate")) => 1,
            _ => return Err("foreign static target identity".into()),
        };
        let pose: [[f64; 4]; 4] = serde_json::from_value(detection["root_from_object"].clone())
            .map_err(|e| e.to_string())?;
        if admitted[index]
            || pose[3] != [0., 0., 0., 1.]
            || !pose.iter().flatten().all(|v| v.is_finite())
            || detection["minimum_edge_px"]
                .as_f64()
                .is_none_or(|e| !e.is_finite() || e < 8.)
            || detection["reprojection_rms_px"]
                .as_f64()
                .is_none_or(|e| !e.is_finite() || !(0. ..=1.).contains(&e))
        {
            return Err("static target is duplicated or failed its original image gate".into());
        }
        admitted[index] = true;
    }
    // Absence is a perception failure; no actuation or truth-based replacement.
    if !admitted[0] || (!pending.grip_check && !admitted[1]) {
        return Err("actual static RGB did not admit both requested targets".into());
    }
    Ok(())
}

impl Drop for StaticMarkerWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let status = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        if let Ok(file) = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join("static_marker_lifecycle.json"))
        {
            let _ = serde_json::to_writer_pretty(
                file,
                &json!({"owned_pid":self.child.id(),"episode_id":self.episode_id,
                "owned_child_reaped":status.is_ok(),"exit_status":status.ok().map(|s|s.to_string()),
                "single_capture_submitted":self.submitted,"pending_at_shutdown":self.pending.is_some(),
                "grip_capture_submitted":self.grip_submitted,
                "world_or_contact_truth_input":false}),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (StaticMarkerVisionConfiguration, Pending, Value) {
        let config = StaticMarkerVisionConfiguration {
            python_path: PathBuf::new(),
            python_sha256: String::new(),
            worker_path: PathBuf::new(),
            worker_sha256: String::new(),
            localizer_path: PathBuf::new(),
            localizer_sha256: String::new(),
            definition_path: PathBuf::new(),
            definition_sha256: "d".repeat(64),
            fiducial_path: PathBuf::new(),
            fiducial_sha256: "f".repeat(64),
        };
        let pending = Pending {
            observation: ObservationStamp {
                episode_id: 7,
                frame_id: 3,
                sim_time_ns: 0,
                captured_at_unix_ms: 100,
            },
            image_hash: "a".repeat(64),
            input_hash: "b".repeat(64),
            started: Instant::now(),
            grip_check: false,
        };
        let reply = json!({"schema":"g1_static_actual_rgb_localization_v1","observation":pending.observation,
        "source":"actual_rgb_printed_label_pnp_and_original_self_FK","image_sha256":pending.image_hash,
        "input_sha256":pending.input_hash,"definition_sha256":config.definition_sha256,"fiducial_sha256":config.fiducial_sha256,
        "camera_profile":"arena_ego","rejected_marker_candidates":0,"world_or_contact_truth_input":false,
        "actuation_proposed":false,"task_qualified":false,"detections":[
            {"marker_id":31,"object_kind":"t1_apple","minimum_edge_px":16.,"reprojection_rms_px":0.2,
                "root_from_object":[[1.,0.,0.,0.5],[0.,1.,0.,0.2],[0.,0.,1.,0.8],[0.,0.,0.,1.]]},
            {"marker_id":32,"object_kind":"t1_plate","minimum_edge_px":40.,"reprojection_rms_px":0.1,
                "root_from_object":[[1.,0.,0.,0.5],[0.,1.,0.,0.],[0.,0.,1.,0.7],[0.,0.,0.,1.]]}
        ]});
        (config, pending, reply)
    }

    #[test]
    fn static_reply_rejects_mobile_foreign_episode_pixels_and_truth() {
        let (config, pending, reply) = fixture();
        assert!(validate_reply(&reply, &pending, &config).is_ok());
        for (field, value) in [
            ("schema", json!("g1_mobile_actual_rgb_localization_v1")),
            ("image_sha256", json!("c".repeat(64))),
            ("world_or_contact_truth_input", json!(true)),
            ("task_qualified", json!(true)),
        ] {
            let mut foreign = reply.clone();
            foreign[field] = value;
            assert!(validate_reply(&foreign, &pending, &config).is_err());
        }
        let mut foreign = reply.clone();
        foreign["observation"]["episode_id"] = json!(8);
        assert!(validate_reply(&foreign, &pending, &config).is_err());
        let mut foreign = reply;
        foreign["object_world_position"] = json!([0., 0., 0.]);
        assert!(validate_reply(&foreign, &pending, &config).is_err());
    }

    #[test]
    fn static_reply_rejects_missing_duplicate_and_bad_image_targets() {
        let (config, pending, reply) = fixture();
        let mut missing = reply.clone();
        missing["detections"].as_array_mut().unwrap().pop();
        assert!(validate_reply(&missing, &pending, &config).is_err());
        let mut duplicate = reply.clone();
        duplicate["detections"][1] = duplicate["detections"][0].clone();
        assert!(validate_reply(&duplicate, &pending, &config).is_err());
        let mut bad = reply.clone();
        bad["detections"][0]["minimum_edge_px"] = json!(7.9);
        assert!(validate_reply(&bad, &pending, &config).is_err());
        let mut bad = reply;
        bad["detections"][0]["reprojection_rms_px"] = json!(1.01);
        assert!(validate_reply(&bad, &pending, &config).is_err());
    }

    #[test]
    fn grip_reply_requires_current_apple_but_initial_reply_keeps_both_targets() {
        let (config, mut pending, mut reply) = fixture();
        reply["detections"].as_array_mut().unwrap().pop();
        assert!(validate_reply(&reply, &pending, &config).is_err());
        pending.grip_check = true;
        assert!(validate_reply(&reply, &pending, &config).is_ok());
        reply["detections"].as_array_mut().unwrap().clear();
        assert!(validate_reply(&reply, &pending, &config).is_err());
    }
}
