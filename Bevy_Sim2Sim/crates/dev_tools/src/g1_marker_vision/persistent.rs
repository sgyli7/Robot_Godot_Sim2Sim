//! A single bounded CPU process; preparation overlaps initial scene/model load.
//! The sole physics owner never waits on this worker's pipes or file output.

use super::{MarkerVisionConfiguration, digest, worker};
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
    time::{Duration, Instant},
};
use task_minigame::types::ObservationStamp;

struct Pending {
    observation: ObservationStamp,
    image_hash: String,
    input_hash: String,
    placement: bool,
    started: Instant,
}

pub(crate) struct PersistentMarkerWorker {
    config: MarkerVisionConfiguration,
    child: Child,
    input: ChildStdin,
    replies: Mutex<Receiver<Result<Value, String>>>,
    reader: Option<JoinHandle<()>>,
    root: PathBuf,
    episode_id: u64,
    started: Instant,
    ready: Option<Value>,
    pending: Option<Pending>,
    requests: u32,
    last_observation: Option<ObservationStamp>,
}

impl PersistentMarkerWorker {
    pub(crate) fn spawn(
        config: MarkerVisionConfiguration,
        root: &Path,
        episode_id: u64,
    ) -> Result<Self, String> {
        config.validate()?;
        let program = config
            .persistent_worker
            .as_ref()
            .ok_or("continuous marker worker requires a pinned program")?;
        let geometry = config
            .task_geometry
            .as_ref()
            .ok_or("continuous worker requires public geometry")?;
        let fiducials = config
            .fiducial_calibration
            .as_ref()
            .ok_or("continuous worker requires public marker calibration")?;
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("persistent_marker_worker.log"))
            .map_err(|e| e.to_string())?;
        let started = Instant::now();
        let mut child = Command::new(&config.python_path)
            .arg("-u")
            .arg(&program.path)
            .arg("--vision-script")
            .arg(&config.script_path)
            .arg("--vision-sha256")
            .arg(&config.script_sha256)
            .arg("--definition")
            .arg(&config.definition_path)
            .arg("--geometry")
            .arg(&geometry.path)
            .arg("--fiducials")
            .arg(&fiducials.path)
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
        let input = child.stdin.take().ok_or("owned worker stdin absent")?;
        let stdout = child.stdout.take().ok_or("owned worker stdout absent")?;
        let (sender, replies) = mpsc::sync_channel(2);
        let reader = thread::Builder::new()
            .name("g1-marker-replies".into())
            .spawn(move || {
                let mut lines = BufReader::new(stdout);
                loop {
                    let mut bytes = Vec::new();
                    let result = (&mut lines)
                        .take(128 * 1024 + 1)
                        .read_until(b'\n', &mut bytes);
                    let value = match result {
                        Ok(0) => Err("persistent marker process closed its reply pipe".into()),
                        Ok(_) if bytes.len() > 128 * 1024 || !bytes.ends_with(b"\n") => {
                            Err("persistent marker reply exceeds its line budget".into())
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
            requests: 0,
            last_observation: None,
        })
    }

    pub(crate) fn provenance(&self) -> Value {
        json!({"owned_pid":self.child.id(),"episode_id":self.episode_id,
            "ready":self.ready,"request_count":self.requests,"bounded_one_pending_request":true,
            "world_or_contact_truth_input":false})
    }

    pub(crate) fn ready(&self) -> bool {
        self.ready.is_some()
    }

    pub(crate) fn poll(&mut self) -> Result<Option<Value>, String> {
        if self
            .pending
            .as_ref()
            .is_some_and(|p| p.started.elapsed() > Duration::from_secs(3))
        {
            return Err("persistent marker request exceeded its finite deadline".into());
        }
        if self.ready.is_none() && self.started.elapsed() > Duration::from_secs(3) {
            return Err("persistent marker preparation exceeded its finite deadline".into());
        }
        let message = match self
            .replies
            .lock()
            .map_err(|_| "marker reply slot poisoned")?
            .try_recv()
        {
            Ok(message) => message?,
            Err(TryRecvError::Empty) => return Ok(None),
            Err(TryRecvError::Disconnected) => {
                return Err("persistent marker reader disconnected".into());
            }
        };
        if message["schema"] != "g1_persistent_marker_worker_v1" {
            return Err("foreign marker worker reply schema".into());
        }
        if message["event"] == "ready" {
            let program = self.config.persistent_worker.as_ref().unwrap();
            if self.ready.is_some()
                || self.pending.is_some()
                || message["episode_id"] != self.episode_id
                || message["pid"] != self.child.id()
                || message["vision_script_sha256"] != self.config.script_sha256
                || message["worker_sha256"] != program.sha256
            {
                return Err("foreign/repeated marker preparation identity".into());
            }
            self.ready = Some(message);
            return Ok(None);
        }
        let pending = self
            .pending
            .take()
            .ok_or("unsolicited persistent marker result")?;
        let actual: ObservationStamp =
            serde_json::from_value(message["observation"].clone()).map_err(|e| e.to_string())?;
        if actual != pending.observation {
            return Err("persistent marker result changed its actual image stamp".into());
        }
        if message["event"] == "error" {
            return Err(format!(
                "persistent marker localization rejected: {}",
                message["error"]
            ));
        }
        if message["event"] != "result" {
            return Err("unknown persistent marker reply event".into());
        }
        let reply = message["result"].clone();
        worker::validate_reply_for_policy(
            &reply,
            actual,
            &pending.image_hash,
            &pending.input_hash,
            &self.config.definition_sha256,
            None,
            false,
            pending.placement,
        )?;
        let calibration = self.config.fiducial_calibration.as_ref().unwrap();
        if reply["camera_mount_profile"] != "auxiliary_grip_overview"
            || reply["fiducial_calibration_sha256"] != calibration.sha256
            || reply["current_box_palm_center_distance_m"]
                .as_f64()
                .is_none_or(|d| !d.is_finite() || !(0. ..=0.25).contains(&d))
        {
            return Err("unbound auxiliary camera, marker or self FK grip region".into());
        }
        let geometry = self.config.task_geometry.as_ref().unwrap();
        worker::validate_clearance_reply(&reply, actual, &geometry.sha256)?;
        if pending.placement {
            worker::validate_release_reply(&reply, actual, &geometry.sha256)?;
        }
        if !reply["fine_approach_proposal"].is_null() {
            worker::validate_fine_reply(&reply, actual, &geometry.sha256)?;
        }
        Ok(Some(reply))
    }

    pub(crate) fn submit(
        &mut self,
        directory: &Path,
        observation: ObservationStamp,
        placement: bool,
    ) -> Result<(), String> {
        if !self.ready()
            || self.pending.is_some()
            || self.requests >= 12
            || observation.episode_id != self.episode_id
            || self.last_observation.is_some_and(|old| {
                observation.frame_id <= old.frame_id || observation.sim_time_ns <= old.sim_time_ns
            })
        {
            return Err("persistent marker is unready, busy, foreign or exhausted".into());
        }
        let directory = directory.canonicalize().map_err(|e| e.to_string())?;
        if directory.parent() != Some(self.root.as_path()) {
            return Err("marker input is outside its owned episode root".into());
        }
        let image_hash = digest(&directory.join("ego.png"), 16 * 1024 * 1024)?;
        let input = directory.join("observation.json");
        let input_hash = digest(&input, 128 * 1024)?;
        let document: Value = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let actual: ObservationStamp =
            serde_json::from_value(document["stamp"].clone()).map_err(|e| e.to_string())?;
        if actual != observation || document["camera_mount_profile"] != "auxiliary_grip_overview" {
            return Err("persistent marker input changed its actual image identity".into());
        }
        let mut bytes = serde_json::to_vec(&json!({"schema":"g1_persistent_marker_request_v1",
            "directory":directory,"observation":observation,"placement_view_only":placement}))
        .map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        if bytes.len() > 8192 {
            return Err("persistent marker request exceeds byte budget".into());
        }
        self.input.write_all(&bytes).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        self.pending = Some(Pending {
            observation,
            image_hash,
            input_hash,
            placement,
            started: Instant::now(),
        });
        self.requests += 1;
        self.last_observation = Some(observation);
        Ok(())
    }
}

impl Drop for PersistentMarkerWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let status = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        if let Ok(file) = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join("persistent_marker_lifecycle.json"))
        {
            let _ = serde_json::to_writer_pretty(
                file,
                &json!({"owned_pid":self.child.id(),"episode_id":self.episode_id,
                "owned_child_reaped":status.is_ok(),"exit_status":status.ok().map(|s|s.to_string()),
                "request_count":self.requests,"pending_request_at_shutdown":self.pending.as_ref().map(|p|p.observation),
                "world_or_contact_truth_input":false}),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "pinned saved actual RGB round trip through the persistent CPU worker; zero physics/VLA/GPU work"]
    fn saved_persistent_marker_worker_round_trip() -> Result<(), String> {
        let config_path =
            PathBuf::from(std::env::var("G1_MARKER_PERSISTENT_CONFIG").map_err(|e| e.to_string())?);
        let root =
            PathBuf::from(std::env::var("G1_MARKER_PERSISTENT_ROOT").map_err(|e| e.to_string())?);
        let cfg: MarkerVisionConfiguration =
            serde_json::from_slice(&fs::read(config_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let input = root.join("input");
        let doc: Value = serde_json::from_slice(
            &fs::read(input.join("observation.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let observation: ObservationStamp =
            serde_json::from_value(doc["stamp"].clone()).map_err(|e| e.to_string())?;
        let expected: Value = serde_json::from_slice(
            &fs::read(root.join("expected_localization.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let mut worker = PersistentMarkerWorker::spawn(cfg, &root, observation.episode_id)?;
        while !worker.ready() {
            assert!(worker.poll()?.is_none());
            thread::sleep(Duration::from_millis(5));
        }
        worker.submit(&input, observation, false)?;
        assert!(worker.submit(&input, observation, false).is_err());
        let reply = loop {
            if let Some(reply) = worker.poll()? {
                break reply;
            }
            thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(reply, expected);
        assert!(worker.submit(&input, observation, false).is_err());
        let mut foreign = observation;
        foreign.episode_id += 1;
        foreign.frame_id += 1;
        foreign.sim_time_ns += 20_000_000;
        assert!(worker.submit(&input, foreign, false).is_err());
        fs::write(root.join("round_trip_receipt.json"),serde_json::to_vec_pretty(&json!({
            "actual_localization_matches_original_cli":true,"actual_observation":observation,
            "provenance":worker.provenance(),"duplicate_busy_old_episode_rejected":true,
            "physics_integrations":0,"fresh_vla_calls":0,"fresh_camera_frames":0,"qualified":false,
        })).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        drop(worker);
        let lifecycle: Value = serde_json::from_slice(
            &fs::read(root.join("persistent_marker_lifecycle.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        assert_eq!(lifecycle["owned_child_reaped"], true);
        Ok(())
    }
}
