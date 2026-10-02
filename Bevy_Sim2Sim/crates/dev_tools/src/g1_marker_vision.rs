//! One bounded local RGB localization job. This worker never owns physics.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MarkerVisionConfiguration {
    pub python_path: PathBuf,
    pub python_sha256: String,
    pub script_path: PathBuf,
    pub script_sha256: String,
    pub definition_path: PathBuf,
    pub definition_sha256: String,
}

fn digest(path: &Path, maximum: u64) -> Result<String, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > maximum {
        return Err("marker worker file exceeds its byte budget".into());
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(path).map_err(|e| e.to_string())?)
    ))
}

impl MarkerVisionConfiguration {
    pub fn validate(&self) -> Result<(), String> {
        if self.definition_sha256
            != "571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd"
        {
            return Err("marker vision requires the original mobile G1 definition".into());
        }
        for (path, hash, maximum) in [
            (&self.python_path, &self.python_sha256, 64 * 1024 * 1024),
            (&self.script_path, &self.script_sha256, 128 * 1024),
            (
                &self.definition_path,
                &self.definition_sha256,
                // The pinned full original collider definition is17,171,205bytes.
                32 * 1024 * 1024,
            ),
        ] {
            if !path.is_absolute() || digest(path, maximum)? != *hash {
                return Err("marker worker path/hash mismatch".into());
            }
        }
        Ok(())
    }
}

#[cfg(feature = "g1_constraint_diagnostic")]
pub(super) use worker::MarkerVisionJob;

#[cfg(feature = "g1_constraint_diagnostic")]
mod worker {
    use super::*;
    use std::{
        process::{Command, Stdio},
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
            mpsc::{self, Receiver, TryRecvError},
        },
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };
    use task_minigame::types::ObservationStamp;
    pub(crate) struct MarkerVisionJob {
        replies: Mutex<Receiver<Result<serde_json::Value, String>>>,
        cancel: Arc<AtomicBool>,
        thread: Option<JoinHandle<()>>,
        pub observation: ObservationStamp,
        pub started: Instant,
    }

    impl MarkerVisionJob {
        pub fn start(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
        ) -> Result<Self, String> {
            let (sender, receiver) = mpsc::sync_channel(1);
            let cancel = Arc::new(AtomicBool::new(false));
            let stop = cancel.clone();
            let started = Instant::now();
            let thread = thread::Builder::new()
                .name("g1-marker-vision".into())
                .spawn(move || {
                    let result = (|| -> Result<serde_json::Value, String> {
                        config.validate()?;
                        let image = directory.join("ego.png");
                        let input = directory.join("observation.json");
                        let output = directory.join("localization.json");
                        if output.exists() {
                            return Err("marker output must be new".into());
                        }
                        let image_hash = digest(&image, 16 * 1024 * 1024)?;
                        let input_hash = digest(&input, 128 * 1024)?;
                        let log = fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(directory.join("worker.log"))
                            .map_err(|e| e.to_string())?;
                        let mut child = Command::new(&config.python_path)
                            .arg(&config.script_path)
                            .args(["--image"])
                            .arg(&image)
                            .args(["--observation"])
                            .arg(&input)
                            .args(["--definition"])
                            .arg(&config.definition_path)
                            .args(["--output"])
                            .arg(&output)
                            .stdin(Stdio::null())
                            .stdout(log.try_clone().map_err(|e| e.to_string())?)
                            .stderr(log)
                            .spawn()
                            .map_err(|e| e.to_string())?;
                        loop {
                            if stop.load(Ordering::Acquire)
                                || started.elapsed() > Duration::from_secs(3)
                            {
                                let _ = child.kill();
                                let _ = child.wait();
                                return Err(
                                "finite marker localization canceled/timed out; no goal admitted"
                                    .into(),
                            );
                            }
                            match child.try_wait() {
                                Ok(Some(status)) => {
                                    if !status.success() {
                                        return Err(format!(
                                            "marker localization failed: {status}"
                                        ));
                                    }
                                    break;
                                }
                                Ok(None) => thread::sleep(Duration::from_millis(10)),
                                Err(e) => {
                                    let _ = child.kill();
                                    let _ = child.wait();
                                    return Err(e.to_string());
                                }
                            }
                        }
                        if fs::metadata(&output).map_err(|e| e.to_string())?.len() > 128 * 1024 {
                            return Err("marker output exceeds byte budget".into());
                        }
                        let reply: serde_json::Value =
                            serde_json::from_slice(&fs::read(output).map_err(|e| e.to_string())?)
                                .map_err(|e| e.to_string())?;
                        validate_reply(
                            &reply,
                            observation,
                            &image_hash,
                            &input_hash,
                            &config.definition_sha256,
                        )?;
                        Ok(reply)
                    })();
                    let _ = sender.send(result);
                })
                .map_err(|e| e.to_string())?;
            Ok(Self {
                replies: Mutex::new(receiver),
                cancel,
                thread: Some(thread),
                observation,
                started,
            })
        }

        pub fn try_take(&self) -> Option<Result<serde_json::Value, String>> {
            match self.replies.lock().unwrap().try_recv() {
                Ok(reply) => Some(reply),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    Some(Err("marker localization worker disconnected".into()))
                }
            }
        }
    }

    impl Drop for MarkerVisionJob {
        fn drop(&mut self) {
            self.cancel.store(true, Ordering::Release);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn validate_reply(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        image: &str,
        input: &str,
        definition: &str,
    ) -> Result<(), String> {
        let actual: ObservationStamp =
            serde_json::from_value(reply["observation"].clone()).map_err(|e| e.to_string())?;
        if reply["schema"] != "g1_mobile_actual_marker_localization_v1"
            || actual != observation
            || reply["image_sha256"] != image
            || reply["input_sha256"] != input
            || reply["robot_definition_sha256"] != definition
            || reply["actual_rgb_only_object_measurement"] != true
            || reply["world_or_contact_truth_input"] != false
            || reply["task_qualified"] != false
        {
            return Err("foreign/unbound marker localization reply".into());
        }
        let detections = reply["detections"]
            .as_array()
            .ok_or("marker detections absent")?;
        if detections.len() != 2 {
            return Err("both visible box22 and target21 are required".into());
        }
        let mut ids = Vec::new();
        for detection in detections {
            let id = detection["marker_id"]
                .as_u64()
                .ok_or("marker identity absent")?;
            let edge = detection["minimum_edge_px"]
                .as_f64()
                .ok_or("marker edge absent")?;
            let error = detection["reprojection_rms_px"]
                .as_f64()
                .ok_or("marker residual absent")?;
            if !matches!(id, 21 | 22)
                || ids.contains(&id)
                || !edge.is_finite()
                || edge < 8.
                || !error.is_finite()
                || !(0. ..=1.).contains(&error)
            {
                return Err("ambiguous/low-quality marker localization".into());
            }
            ids.push(id);
        }
        let proposal = &reply["navigation_proposal"];
        let stamp: ObservationStamp =
            serde_json::from_value(proposal["observation"].clone()).map_err(|e| e.to_string())?;
        let heading = proposal["heading_yaw_source_rad"]
            .as_f64()
            .ok_or("visual heading absent")?;
        let distance = proposal["relative_distance_m"]
            .as_f64()
            .ok_or("visual distance absent")?;
        if stamp != observation
            || proposal["target_identity"] != 21
            || proposal["source"] != "actual_rgb_marker_pnp_and_original_self_fk"
            || proposal["task_qualified"] != false
            || proposal["automatically_executed"] != false
            || !heading.is_finite()
            || heading.abs() > std::f64::consts::PI
            || !distance.is_finite()
            || !(0.1..=2.5).contains(&distance)
        {
            return Err("invalid visual navigation proposal".into());
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        #[ignore = "requires pinned config and saved actual RGB/self-sensor inputs; no physics/models"]
        fn real_rgb_local_worker_roundtrip() -> Result<(), String> {
            let fixture = std::env::var("G1_MARKER_JOB_FIXTURE").map_err(|e| e.to_string())?;
            let f: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let config: MarkerVisionConfiguration =
                serde_json::from_value(f["configuration"].clone()).map_err(|e| e.to_string())?;
            config.validate()?;
            let directory = PathBuf::from(f["output"].as_str().ok_or("output absent")?);
            fs::create_dir(&directory).map_err(|e| e.to_string())?;
            fs::copy(
                f["image"].as_str().ok_or("image absent")?,
                directory.join("ego.png"),
            )
            .map_err(|e| e.to_string())?;
            fs::copy(
                f["observation"].as_str().ok_or("observation absent")?,
                directory.join("observation.json"),
            )
            .map_err(|e| e.to_string())?;
            let input: serde_json::Value = serde_json::from_slice(
                &fs::read(directory.join("observation.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let stamp =
                serde_json::from_value(input["stamp"].clone()).map_err(|e| e.to_string())?;
            let job = MarkerVisionJob::start(config, directory.clone(), stamp)?;
            let result = loop {
                if let Some(reply) = job.try_take() {
                    break reply?;
                }
                if job.started.elapsed() > Duration::from_secs(4) {
                    return Err("worker did not report finite completion".into());
                }
                thread::sleep(Duration::from_millis(5));
            };
            fs::write(directory.join("roundtrip_receipt.json"),serde_json::to_vec_pretty(&serde_json::json!({
                "actual_saved_rgb_fixture":true,"fresh_vla_calls":0,"actual_integrations":0,
                "qualified":false,"wall_ms":job.started.elapsed().as_secs_f64()*1000.,"reply":result,
            })).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            Ok(())
        }
        fn fixture() -> (serde_json::Value, ObservationStamp) {
            let stamp = ObservationStamp {
                episode_id: 7,
                frame_id: 3,
                sim_time_ns: 20_000_000,
                captured_at_unix_ms: 42,
            };
            let reply = serde_json::json!({
                "schema":"g1_mobile_actual_marker_localization_v1","observation":stamp,
                "image_sha256":"image","input_sha256":"input","robot_definition_sha256":"definition",
                "actual_rgb_only_object_measurement":true,"world_or_contact_truth_input":false,"task_qualified":false,
                "detections":[{"marker_id":21,"minimum_edge_px":12.,"reprojection_rms_px":0.5},
                    {"marker_id":22,"minimum_edge_px":110.,"reprojection_rms_px":0.3}],
                "navigation_proposal":{"observation":stamp,"target_identity":21,
                    "source":"actual_rgb_marker_pnp_and_original_self_fk","task_qualified":false,
                    "automatically_executed":false,"heading_yaw_source_rad":-1.57,"relative_distance_m":1.7},
            });
            (reply, stamp)
        }
        #[test]
        fn rejects_old_episode_and_unbound_image() {
            let (reply, stamp) = fixture();
            assert!(validate_reply(&reply, stamp, "image", "input", "definition").is_ok());
            let mut foreign = stamp;
            foreign.episode_id += 1;
            assert!(validate_reply(&reply, foreign, "image", "input", "definition").is_err());
            assert!(validate_reply(&reply, stamp, "other_image", "input", "definition").is_err());
        }
        #[test]
        fn rejects_missing_duplicate_or_unreliable_target() {
            let (reply, stamp) = fixture();
            for mutation in 0..4 {
                let mut bad = reply.clone();
                match mutation {
                    0 => {
                        bad["detections"].as_array_mut().unwrap().pop();
                    }
                    1 => bad["detections"][1]["marker_id"] = 21.into(),
                    2 => bad["detections"][0]["minimum_edge_px"] = 7.9.into(),
                    _ => bad["detections"][0]["reprojection_rms_px"] = 1.01.into(),
                }
                assert!(validate_reply(&bad, stamp, "image", "input", "definition").is_err());
            }
        }
        #[test]
        fn rejects_truth_claim_and_foreign_navigation_identity() {
            let (reply, stamp) = fixture();
            for mutation in 0..3 {
                let mut bad = reply.clone();
                match mutation {
                    0 => bad["world_or_contact_truth_input"] = true.into(),
                    1 => bad["navigation_proposal"]["target_identity"] = 22.into(),
                    _ => bad["navigation_proposal"]["relative_distance_m"] = 3.0.into(),
                }
                assert!(validate_reply(&bad, stamp, "image", "input", "definition").is_err());
            }
        }
    }
}
