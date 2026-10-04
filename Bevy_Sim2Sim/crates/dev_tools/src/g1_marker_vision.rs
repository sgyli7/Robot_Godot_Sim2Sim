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
    #[serde(default)]
    pub task_geometry: Option<MarkerTaskGeometry>,
    #[serde(default)]
    pub fiducial_calibration: Option<MarkerTaskGeometry>,
    #[serde(default)]
    pub persistent_worker: Option<MarkerPersistentProgram>,
    /// Explicit experimental pair of the two published auxiliary head views.
    #[serde(default)]
    pub same_tick_box_pair: bool,
    /// Separate paired box-only pickup feedback; cannot request navigation.
    #[serde(default)]
    pub same_tick_box_only: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MarkerPersistentProgram {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MarkerTaskGeometry {
    pub path: PathBuf,
    pub sha256: String,
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
        if self.same_tick_box_only && (!self.same_tick_box_pair || self.task_geometry.is_some()) {
            return Err("paired box-only feedback cannot request navigation geometry".into());
        }
        if self.same_tick_box_pair
            && ((self.task_geometry.is_none() && !self.same_tick_box_only)
                || self.fiducial_calibration.is_none()
                || self.persistent_worker.is_some())
        {
            return Err("box pair requires bound geometry/labels and a finite worker".into());
        }
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
        if let Some(geometry) = &self.task_geometry {
            if geometry.sha256 != "19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0"
                || !geometry.path.is_absolute()
                || digest(&geometry.path, 1024 * 1024)? != geometry.sha256
            {
                return Err("marker clearance requires the pinned public task geometry".into());
            }
        }
        if let Some(calibration) = &self.fiducial_calibration {
            if !calibration.path.is_absolute()
                || calibration.sha256.len() != 64
                || digest(&calibration.path, 16 * 1024)? != calibration.sha256
            {
                return Err("fixed marker calibration path/hash mismatch".into());
            }
        }
        if let Some(program) = &self.persistent_worker {
            if !program.path.is_absolute() || digest(&program.path, 128 * 1024)? != program.sha256 {
                return Err("persistent marker program path/hash mismatch".into());
            }
        }
        Ok(())
    }
}

#[cfg(feature = "g1_constraint_diagnostic")]
pub(super) use worker::{MarkerTargetMemory, MarkerVisionJob};
#[cfg(feature = "g1_constraint_diagnostic")]
mod persistent;
#[cfg(feature = "g1_constraint_diagnostic")]
pub(super) use persistent::PersistentMarkerWorker;

#[cfg(feature = "g1_constraint_diagnostic")]
mod mobile_geometry;
#[cfg(feature = "g1_constraint_diagnostic")]
pub(crate) mod mobile_held;
#[cfg(feature = "g1_constraint_diagnostic")]
mod mobile_pair;
mod static_observation;
#[cfg(feature = "g1_constraint_diagnostic")]
pub(crate) use mobile_pair::validate_mobile_pair_inputs;
mod static_worker;
pub(crate) use static_observation::{static_observation, validate_fixed_pair};
pub(crate) use static_worker::{StaticMarkerVisionConfiguration, StaticMarkerWorker};

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

    fn auxiliary_camera_profile(profile: &str) -> bool {
        matches!(
            profile,
            "auxiliary_grip_overview" | "auxiliary_bin_placement"
        )
    }

    fn auxiliary_reply_camera(reply: &serde_json::Value) -> bool {
        reply["camera_mount_profile"]
            .as_str()
            .is_some_and(auxiliary_camera_profile)
    }

    fn validate_public_bin_board_reply(
        reply: &serde_json::Value,
        box_view_only: bool,
    ) -> Result<(), String> {
        let proof = &reply["public_bin_board_pose"];
        if reply["marker_layout_profile"] != "auxiliary_bin_board_targets" || box_view_only {
            return if proof.is_null() {
                Ok(())
            } else {
                Err("unrequested bin board pose proof".into())
            };
        }
        let corners: [[[f64; 2]; 4]; 3] =
            serde_json::from_value(proof["corners_px"].clone()).map_err(|e| e.to_string())?;
        let errors: [f64; 3] =
            serde_json::from_value(proof["per_marker_reprojection_rms_px"].clone())
                .map_err(|e| e.to_string())?;
        let edges: [f64; 3] =
            serde_json::from_value(proof["minimum_edges_px"].clone()).map_err(|e| e.to_string())?;
        let pose: [[f64; 4]; 4] =
            serde_json::from_value(proof["root_from_marker"].clone()).map_err(|e| e.to_string())?;
        let target = reply["detections"]
            .as_array()
            .and_then(|ds| ds.iter().find(|d| d["marker_id"] == 21))
            .ok_or("current bin board target absent")?;
        if proof["schema"] != "g1_actual_rgb_public_bin_board_pose_v1"
            || proof["marker_ids"] != serde_json::json!([21, 23, 24])
            || proof["solver"] != "SQPNP_then_bounded30_iteration_LM"
            || proof["upright_prior_used"] != false
            || proof["world_or_contact_truth_input"] != false
            || proof["task_qualified"] != false
            || target["root_from_marker"] != proof["root_from_marker"]
            || target["corners_px"] != proof["corners_px"][0]
            || target["reprojection_rms_px"].as_f64() != Some(errors[0])
            || target["minimum_edge_px"].as_f64() != Some(edges[0])
            || !pose.iter().flatten().all(|v| v.is_finite())
            || pose[3] != [0., 0., 0., 1.]
        {
            return Err("unbound current RGB bin board pose proof".into());
        }
        for index in 0..3 {
            if !errors[index].is_finite()
                || !(0. ..=1.).contains(&errors[index])
                || !edges[index].is_finite()
                || edges[index] < 8.
                || corners[index].iter().any(|p| {
                    !p[0].is_finite()
                        || !p[1].is_finite()
                        || !(0. ..640.).contains(&p[0])
                        || !(0. ..480.).contains(&p[1])
                })
            {
                return Err("public bin board failed unchanged image/quality gate".into());
            }
            let actual_edge = (0..4)
                .map(|i| {
                    let a = corners[index][i];
                    let b = corners[index][(i + 1) % 4];
                    (a[0] - b[0]).hypot(a[1] - b[1])
                })
                .fold(f64::INFINITY, f64::min);
            if (edges[index] - actual_edge).abs() > 1e-5 {
                return Err("public bin board edge is not its measured corner span".into());
            }
        }
        Ok(())
    }

    #[derive(Clone, serde::Deserialize, serde::Serialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct MarkerTargetMemory {
        pub schema: String,
        pub origin_observation: ObservationStamp,
        pub current_observation: ObservationStamp,
        pub origin_image_sha256: String,
        pub origin_input_sha256: String,
        pub root_from_target_marker: [[f64; 4]; 4],
        pub origin_root_rotation_wxyz: [f32; 4],
        pub self_velocity_displacement_source_m: [f64; 3],
        pub static_target_assumption: bool,
        pub world_or_contact_truth_input: bool,
    }

    impl MarkerTargetMemory {
        pub fn from_localization(reply: &serde_json::Value) -> Result<Self, String> {
            if reply["target_bin_detected"] != true
                || reply["carried_box_detected"] != true
                || reply["world_or_contact_truth_input"] != false
            {
                return Err("target memory requires the admitted actual two-marker image".into());
            }
            let target = reply["detections"]
                .as_array()
                .ok_or("memory marker absent")?
                .iter()
                .find(|d| d["marker_id"] == 21)
                .ok_or("visible target21 absent")?;
            let origin_observation =
                serde_json::from_value(reply["observation"].clone()).map_err(|e| e.to_string())?;
            Ok(Self {
                schema: "g1_mobile_static_target_memory_v1".into(),
                origin_observation,
                current_observation: origin_observation,
                origin_image_sha256: reply["image_sha256"]
                    .as_str()
                    .ok_or("origin image hash absent")?
                    .into(),
                origin_input_sha256: reply["input_sha256"]
                    .as_str()
                    .ok_or("origin input hash absent")?
                    .into(),
                root_from_target_marker: serde_json::from_value(target["root_from_marker"].clone())
                    .map_err(|e| e.to_string())?,
                origin_root_rotation_wxyz: serde_json::from_value(
                    reply["original_self_root_rotation_wxyz"].clone(),
                )
                .map_err(|e| e.to_string())?,
                self_velocity_displacement_source_m: [0.; 3],
                static_target_assumption: true,
                world_or_contact_truth_input: false,
            })
        }
        pub fn validate(&self, observation: ObservationStamp) -> Result<(), String> {
            let old = self.origin_observation;
            let age = observation.sim_time_ns.checked_sub(old.sim_time_ns);
            let wall_age = observation
                .captured_at_unix_ms
                .checked_sub(old.captured_at_unix_ms);
            let hashes = [&self.origin_image_sha256, &self.origin_input_sha256];
            let rotation_norm: f64 = self
                .origin_root_rotation_wxyz
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum();
            if self.schema != "g1_mobile_static_target_memory_v1"
                || self.current_observation != observation
                || !self.static_target_assumption
                || self.world_or_contact_truth_input
                || old.episode_id == 0
                || old.frame_id == 0
                || old.captured_at_unix_ms == 0
                || observation.episode_id != old.episode_id
                || observation.frame_id <= old.frame_id
                || old.sim_time_ns % 20_000_000 != 0
                || observation.sim_time_ns % 20_000_000 != 0
                || age.is_none_or(|ns| ns == 0 || ns > 8_000_000_000)
                || wall_age.is_none_or(|ms| ms == 0 || ms > 12_000)
                || hashes.iter().any(|h| {
                    h.len() != 64
                        || !h
                            .chars()
                            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
                })
                || !self
                    .root_from_target_marker
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite())
                || self.root_from_target_marker[3] != [0., 0., 0., 1.]
                || !rotation_norm.is_finite()
                || (rotation_norm - 1.).abs() > 2e-5
                || !self
                    .self_velocity_displacement_source_m
                    .iter()
                    .all(|v| v.is_finite())
                || self
                    .self_velocity_displacement_source_m
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    > 0.25f64.powi(2)
            {
                return Err("foreign/expired/unbound visual target memory or self motion".into());
            }
            Ok(())
        }
    }
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
            Self::start_with_memory(config, directory, observation, None)
        }
        pub fn start_with_memory(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
            memory: Option<MarkerTargetMemory>,
        ) -> Result<Self, String> {
            Self::start_internal(
                config,
                directory,
                observation,
                memory,
                false,
                false,
                false,
                false,
            )
        }

        /// Current paired RGB feedback for a completed pickup's next carry.
        pub fn start_with_held_box_feedback(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
        ) -> Result<Self, String> {
            if !config.same_tick_box_pair
                || config.same_tick_box_only
                || config.task_geometry.is_none()
            {
                return Err("held-box feedback requires normal current paired geometry".into());
            }
            Self::start_internal(
                config,
                directory,
                observation,
                None,
                false,
                false,
                true,
                false,
            )
        }

        /// Geometry-only diagnostic from current paired RGB and measured self
        /// state. It cannot replace navigation or held-object action admission.
        pub(crate) fn start_with_held_contact_geometry(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
        ) -> Result<Self, String> {
            if !config.same_tick_box_pair
                || config.same_tick_box_only
                || config.task_geometry.is_none()
                || config.persistent_worker.is_some()
            {
                return Err("hand geometry requires bound paired RGB and public geometry".into());
            }
            let helper = config
                .script_path
                .parent()
                .ok_or("hand geometry script has no parent")?
                .join("unitree_g1_mobile_hand_geometry.py");
            if digest(&helper, 128 * 1024)? != super::mobile_geometry::helper_hash() {
                return Err("hand geometry helper differs from the compiled source".into());
            }
            Self::start_internal(
                config,
                directory,
                observation,
                None,
                false,
                false,
                true,
                true,
            )
        }

        pub fn start_box_view(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
        ) -> Result<Self, String> {
            if config.task_geometry.is_some() {
                return Err("box visibility cannot request navigation clearance".into());
            }
            if config.same_tick_box_pair && !config.same_tick_box_only {
                return Err("paired pickup feedback requires its explicit box-only mode".into());
            }
            Self::start_internal(
                config,
                directory,
                observation,
                None,
                true,
                false,
                false,
                false,
            )
        }

        pub fn start_placement_view(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
        ) -> Result<Self, String> {
            if config.task_geometry.is_none() || config.fiducial_calibration.is_none() {
                return Err(
                    "placement view requires bound public geometry and auxiliary labels".into(),
                );
            }
            Self::start_internal(
                config,
                directory,
                observation,
                None,
                false,
                true,
                false,
                false,
            )
        }

        fn start_internal(
            config: MarkerVisionConfiguration,
            directory: PathBuf,
            observation: ObservationStamp,
            memory: Option<MarkerTargetMemory>,
            box_view_only: bool,
            placement_view_only: bool,
            held_box_feedback: bool,
            held_contact_geometry: bool,
        ) -> Result<Self, String> {
            if let Some(memory) = &memory {
                memory.validate(observation)?;
            }
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
                        let input_document: serde_json::Value =
                            serde_json::from_slice(&fs::read(&input).map_err(|e| e.to_string())?)
                                .map_err(|e| e.to_string())?;
                        let camera_profile = input_document["camera_mount_profile"]
                            .as_str()
                            .unwrap_or("arena_ego");
                        let pair = if config.same_tick_box_pair {
                            if memory.is_some() || box_view_only != config.same_tick_box_only {
                                return Err("box pair has an unrequested memory or box-only mode".into());
                            }
                            Some(super::mobile_pair::read_secondary(&directory, &input_document)?)
                        } else {
                            None
                        };
                        if !(camera_profile == "arena_ego" || auxiliary_camera_profile(camera_profile))
                            || (auxiliary_camera_profile(camera_profile)
                                && config.fiducial_calibration.is_none())
                            || (camera_profile == "auxiliary_bin_placement" && (memory.is_some() || box_view_only))
                        {
                            return Err("unbound fixed observation camera profile".into());
                        }
                        let memory_path = directory.join("target_memory.json");
                        let memory_hash = if let Some(memory) = &memory {
                            let mut file = fs::OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&memory_path)
                                .map_err(|e| e.to_string())?;
                            serde_json::to_writer_pretty(&mut file, memory)
                                .map_err(|e| e.to_string())?;
                            Some(digest(&memory_path, 128 * 1024)?)
                        } else {
                            None
                        };
                        let log = fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(directory.join("worker.log"))
                            .map_err(|e| e.to_string())?;
                        let mut command = Command::new(&config.python_path);
                        command
                            .arg(&config.script_path)
                            .args(["--image"])
                            .arg(&image)
                            .args(["--observation"])
                            .arg(&input)
                            .args(["--definition"])
                            .arg(&config.definition_path)
                            .args(["--output"])
                            .arg(&output);
                        if let Some(geometry) = &config.task_geometry {
                            command.arg("--geometry").arg(&geometry.path);
                        }
                        if memory.is_some() {
                            command.arg("--target-memory").arg(&memory_path);
                        }
                        if box_view_only {
                            command.arg("--box-view-only");
                        }
                        if placement_view_only {
                            command.arg("--placement-view-only");
                        }
                        if held_box_feedback {
                            command.arg("--held-box-feedback");
                        }
                        if held_contact_geometry {
                            command.arg("--held-contact-geometry");
                        }
                        if let Some(calibration) = &config.fiducial_calibration {
                            command.arg("--fiducials").arg(&calibration.path);
                        }
                        if pair.is_some() {
                            command.arg("--secondary-image").arg(directory.join("secondary/ego.png"))
                                .arg("--secondary-observation").arg(directory.join("secondary/observation.json"));
                        }
                        let mut child = command
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
                        if held_contact_geometry {
                            super::mobile_geometry::validate_reply(
                                &reply, observation, &image_hash, &input_hash,
                                &config.definition_sha256,
                                &config.task_geometry.as_ref().unwrap().sha256,
                            )?;
                        } else {
                            super::mobile_geometry::reject_unrequested(&reply)?;
                            validate_reply_for_policy(
                            &reply,
                            observation,
                            &image_hash,
                            &input_hash,
                            &config.definition_sha256,
                            memory.as_ref().zip(memory_hash.as_deref()),
                            box_view_only,
                            placement_view_only,
                            )?;
                        }
                        super::mobile_pair::validate_reply(&reply, &input_document, pair.as_ref())?;
                        super::mobile_held::validate_requested_feedback(&reply, observation, held_box_feedback)?;
                        if config
                            .fiducial_calibration
                            .as_ref()
                            .is_some_and(|c| reply["fiducial_calibration_sha256"] != c.sha256)
                            || (config.fiducial_calibration.is_some()
                                && reply["camera_mount_profile"] != camera_profile)
                            || (config.fiducial_calibration.is_none()
                                && (!reply["fiducial_calibration_sha256"].is_null()
                                    || auxiliary_reply_camera(&reply)))
                        {
                            return Err("unbound fixed camera/marker calibration reply".into());
                        }
                        if auxiliary_camera_profile(camera_profile)
                            && reply["current_box_palm_center_distance_m"]
                                .as_f64()
                                .is_none_or(|d| !d.is_finite() || !(0. ..=0.25).contains(&d))
                        {
                            return Err("current auxiliary RGB box is outside the bounded self FK grip region; owner remains paused".into());
                        }
                        if let Some(geometry) = &config.task_geometry {
                            if reply["target_memory_used"] != true && !held_contact_geometry {
                                validate_clearance_reply(&reply, observation, &geometry.sha256)?;
                            }
                            if placement_view_only {
                                validate_release_reply(&reply, observation, &geometry.sha256)?;
                                validate_thumb_preparation_reply(&reply, observation, &geometry.sha256)?;
                            }
                            if !reply["fine_approach_proposal"].is_null() {
                                validate_fine_reply(&reply, observation, &geometry.sha256)?;
                            }
                        } else if !reply["clearance_proposal"].is_null() || !reply["fine_approach_proposal"].is_null() {
                            return Err("unrequested public-geometry clearance reply".into());
                        }
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

    #[cfg(test)]
    fn validate_reply(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        image: &str,
        input: &str,
        definition: &str,
    ) -> Result<(), String> {
        validate_reply_for_memory(reply, observation, image, input, definition, None)
    }

    #[cfg(test)]
    fn validate_reply_for_memory(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        image: &str,
        input: &str,
        definition: &str,
        memory: Option<(&MarkerTargetMemory, &str)>,
    ) -> Result<(), String> {
        validate_reply_for_policy(
            reply,
            observation,
            image,
            input,
            definition,
            memory,
            false,
            false,
        )
    }

    pub(super) fn validate_reply_for_policy(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        image: &str,
        input: &str,
        definition: &str,
        memory: Option<(&MarkerTargetMemory, &str)>,
        box_view_only: bool,
        placement_view_only: bool,
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
        validate_public_bin_board_reply(reply, box_view_only)?;
        let detections = reply["detections"]
            .as_array()
            .ok_or("marker detections absent")?;
        if (memory.is_none() && !box_view_only && detections.len() != 2)
            || (memory.is_some() && !(1..=2).contains(&detections.len()))
            || (box_view_only && !(1..=2).contains(&detections.len()))
        {
            return Err(if box_view_only {
                "current actual box22 visibility is required before view motion"
            } else {
                "both visible box22 and target21 are required"
            }
            .into());
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
        let use_memory = !ids.contains(&21);
        if !ids.contains(&22) {
            return Err("current actual box22 is required".into());
        }
        if box_view_only {
            let proximity = reply["current_box_palm_center_distance_m"]
                .as_f64()
                .ok_or("current RGB/self FK grip proximity absent")?;
            if !proximity.is_finite() || !(0. ..=0.25).contains(&proximity) {
                return Err("current RGB box is outside the bounded self FK grip region; owner remains paused".into());
            }
            if memory.is_some()
                || reply["box_view_only"] != true
                || !reply["navigation_proposal"].is_null()
                || !reply["target_memory_estimate"].is_null()
                || reply["target_memory_used"] == true
                || !reply["clearance_proposal"].is_null()
                || reply["carried_box_detected"] != true
            {
                return Err("box visibility cannot admit navigation or remembered targets".into());
            }
            return Ok(());
        }
        if reply["box_view_only"] == true {
            return Err("box visibility cannot replace a target localization".into());
        }
        if placement_view_only {
            if reply["placement_view_only"] != true
                || !auxiliary_reply_camera(reply)
                || memory.is_some()
                || !reply["navigation_proposal"].is_null()
                || !reply["target_memory_estimate"].is_null()
                || reply["target_memory_used"] == true
                || reply["release_proposal"].is_null()
            {
                return Err(
                    "placement view cannot admit navigation, memory or a foreign camera".into(),
                );
            }
            return Ok(());
        }
        if reply["placement_view_only"] == true || !reply["release_proposal"].is_null() {
            return Err("unrequested release admission reply".into());
        }
        if let Some((memory, hash)) = memory {
            memory.validate(observation)?;
            let estimate = &reply["target_memory_estimate"];
            let stamp: ObservationStamp = serde_json::from_value(estimate["observation"].clone())
                .map_err(|e| e.to_string())?;
            let origin: ObservationStamp =
                serde_json::from_value(estimate["origin_observation"].clone())
                    .map_err(|e| e.to_string())?;
            let matrix: [[f64; 4]; 4] =
                serde_json::from_value(estimate["root_from_marker"].clone())
                    .map_err(|e| e.to_string())?;
            if stamp != observation
                || origin != memory.origin_observation
                || estimate["schema"] != "g1_propagated_static_visual_target_v1"
                || estimate["target_identity"] != 21
                || estimate["memory_sha256"] != hash
                || estimate["origin_image_sha256"] != memory.origin_image_sha256
                || estimate["origin_input_sha256"] != memory.origin_input_sha256
                || estimate["age_sim_ns"]
                    != observation.sim_time_ns - memory.origin_observation.sim_time_ns
                || estimate["age_wall_ms"]
                    != observation.captured_at_unix_ms
                        - memory.origin_observation.captured_at_unix_ms
                || estimate["static_target_assumption"] != true
                || estimate["world_or_contact_truth_input"] != false
                || estimate["task_qualified"] != false
                || reply["target_memory_used"] != use_memory
                || !matrix.iter().flatten().all(|v| v.is_finite())
                || matrix[3] != [0., 0., 0., 1.]
            {
                return Err("foreign/unbound propagated target-memory reply".into());
            }
        } else if !reply["target_memory_estimate"].is_null() || use_memory {
            return Err("unrequested static target memory".into());
        }
        let proposal = &reply["navigation_proposal"];
        if proposal.is_null() {
            let fine = &reply["fine_approach_proposal"];
            if memory.is_some() || fine["state"] != "aligned" || !auxiliary_reply_camera(reply) {
                return Err("missing navigation requires current whole-box alignment".into());
            }
            // The owned worker binds this declared geometry hash to its pinned
            // configuration after this general image/self-state validation.
            let hash = fine["source_geometry_sha256"]
                .as_str()
                .ok_or("alignment geometry absent")?;
            validate_fine_reply(reply, observation, hash)?;
            return Ok(());
        }
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
            || proposal["source"]
                != if use_memory {
                    "actual_box_rgb_and_static_target_memory_self_velocity"
                } else {
                    "actual_rgb_marker_pnp_and_original_self_fk"
                }
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

    pub(super) fn validate_fine_reply(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        geometry_hash: &str,
    ) -> Result<(), String> {
        let p = &reply["fine_approach_proposal"];
        let actual: ObservationStamp =
            serde_json::from_value(p["observation"].clone()).map_err(|e| e.to_string())?;
        let interval: [f64; 2] =
            serde_json::from_value(p["physical_containment_distance_interval_m"].clone())
                .map_err(|e| e.to_string())?;
        let current_margin = p["current_floor_margin_m"]
            .as_f64()
            .ok_or("current fine margin absent")?;
        let margin = p["predicted_floor_margin_m"]
            .as_f64()
            .ok_or("fine margin absent")?;
        if p["schema"] != "g1_visible_marker_fine_carry_v2"
            || actual != observation
            || p["source_geometry_sha256"] != geometry_hash
            || p["world_or_contact_truth_input"] != false
            || p["task_qualified"] != false
            || !auxiliary_reply_camera(reply)
            || !interval.iter().all(|n| n.is_finite())
            || interval[0] < 0.
            // The interval describes complete geometry, not one command. Its
            // first feasible point must fit the existing five 0.15 m steps.
            || interval[0] > 5. * 0.15
            || interval[0] > interval[1]
            || !current_margin.is_finite()
            || p["maximum_physical_step_m"] != 0.15
            || p["new_image_required_after_step"] != true
            || p["existing_navigation_stop_margin_m"] != 0.05
            || !margin.is_finite()
        {
            return Err("foreign/unsafe current whole-object fine interval".into());
        }
        if p["state"] == "aligned" || p["state"] == "blocked" {
            let aligned = current_margin >= 0.02;
            if (p["state"] == "aligned") != aligned
                || !p["goal"].is_null()
                || !p["selected_physical_distance_m"].is_null()
                || p["step_is_final"] != false
                || (margin - current_margin).abs() > 1e-9
                || (!aligned && interval[1] - interval[0] >= 0.04 && interval[1] >= 0.15)
            {
                return Err("fine alignment/block cannot admit movement or bypass geometry".into());
            }
            return Ok(());
        }
        let selected = p["selected_physical_distance_m"]
            .as_f64()
            .ok_or("fine step absent")?;
        let goal: simulation_minigame::g1::mobile_navigation::MobileCarryGoal =
            serde_json::from_value(p["goal"].clone()).map_err(|e| e.to_string())?;
        goal.validate().map_err(|e| e.to_string())?;
        let final_step = selected >= interval[0] && selected <= interval[1];
        if p["state"] != "advance"
            || current_margin >= 0.02
            || interval[1] - interval[0] < 0.04
            || interval[1] < 0.15
            || selected != 0.15
            || p["step_is_final"] != final_step
            || (final_step && margin < 0.02)
            || goal.observation != observation
            || (f64::from(goal.relative_distance_m) + 0.05 - selected).abs() > 1e-6
            || (goal.relative_distance_m - 0.1).abs() > 1e-7
            || (f64::from(goal.heading_yaw_source_rad)
                - reply["navigation_proposal"]["heading_yaw_source_rad"]
                    .as_f64()
                    .ok_or("fine heading absent")?)
            .abs()
                > 1e-6
        {
            return Err("unsafe/unobserved segmented fine approach".into());
        }
        Ok(())
    }

    pub(super) fn validate_release_reply(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        geometry_hash: &str,
    ) -> Result<(), String> {
        let proposal = &reply["release_proposal"];
        let actual: ObservationStamp =
            serde_json::from_value(proposal["observation"].clone()).map_err(|e| e.to_string())?;
        let number = |field| {
            proposal[field]
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| format!("invalid visual release {field}"))
        };
        let margin = number("minimum_signed_floor_margin_m")?;
        let drop = number("estimated_drop_height_m")?;
        let upright = number("visible_bin_upward_cosine")?;
        let speed = number("self_root_speed_m_s")?;
        let required_gap = number("required_measured_palm_gap_m")?;
        let hand_clearance = required_gap <= 0.35;
        let admitted = margin >= 0.02
            && (0.05..=0.4).contains(&drop)
            && upright >= 0.98
            && (0. ..=0.05).contains(&speed)
            && hand_clearance;
        if proposal["schema"] != "g1_visible_marker_release_admission_v2"
            || actual != observation
            || proposal["source_geometry_sha256"] != geometry_hash
            || proposal["world_or_contact_truth_input"] != false
            || proposal["task_qualified"] != false
            || proposal["source"] != "actual_two_marker_rgb_and_public_original_collision_vertices"
            || proposal["minimum_required_floor_margin_m"] != 0.02
            || proposal["opening_profile"] != "gravity_horizontal_original_fingers"
            || proposal["robot_collision_geometry_sha256"] != reply["robot_definition_sha256"]
            || proposal["minimum_hand_clearance_m"] != 0.01
            || !(0. ..=1.).contains(&required_gap)
            || required_gap == 0.
            || proposal["maximum_commanded_palm_gap_m"] != 0.35
            || proposal["hand_clearance_admitted"] != hand_clearance
            || proposal["prediction_is_not_physical_detachment"] != true
            || !(-1. ..=1.).contains(&upright)
            || proposal["release_admitted"] != admitted
        {
            return Err("foreign/unsafe visual release admission".into());
        }
        if admitted {
            let goal: simulation_minigame::g1::mobile_release::MobileReleaseGoal =
                serde_json::from_value(proposal["release_goal"].clone())
                    .map_err(|e| e.to_string())?;
            goal.validate().map_err(|e| e.to_string())?;
            if goal.observation != observation
                || goal.target_palm_gap_m != 0.35
                || goal.duration_ticks != 100
            {
                return Err("release goal detached from its current visual admission".into());
            }
        } else if !proposal["release_goal"].is_null() {
            return Err("rejected visual alignment cannot provide a release goal".into());
        }
        Ok(())
    }

    pub(super) fn validate_thumb_preparation_reply(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        geometry_hash: &str,
    ) -> Result<(), String> {
        use simulation_minigame::g1::mobile_thumb::{
            LEFT_THUMB_CLEARANCE_TARGETS, MAXIMUM_BOUNDED_THUMB_ANGLE_RAD, MobileThumbGoal,
        };
        let proposal = &reply["thumb_preparation_proposal"];
        if proposal.is_null() {
            return Ok(());
        }
        let both_hands = match proposal["schema"].as_str() {
            Some("g1_visible_station_thumb_preparation_v1") => false,
            Some("g1_visible_station_thumb_preparation_v2") => true,
            _ => return Err("foreign thumb preparation schema".into()),
        };
        let release = &reply["release_proposal"];
        let clearance = &proposal["candidate_clearance"];
        let numeric = |value: &serde_json::Value| {
            value
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or("nonfinite thumb proposal".to_owned())
        };
        let gap = numeric(&clearance["required_measured_palm_gap_m"])?;
        let minimum = numeric(&clearance["minimum_certified_separation_at_maximum_gap_m"])?;
        let candidate = gap <= 0.35 && minimum >= 0.016;
        let targets = proposal["target_left_thumb_angles_rad"]
            .as_array()
            .ok_or("thumb targets absent")?;
        let initial = proposal["measured_initial_left_thumb_angles_rad"]
            .as_array()
            .ok_or("initial thumb measurements absent")?;
        if targets.len() != 2 || initial.len() != 2 {
            return Err("thumb angle width mismatch".into());
        }
        for slot in 0..2 {
            if (!both_hands
                && (numeric(&targets[slot])? - f64::from(LEFT_THUMB_CLEARANCE_TARGETS[slot])).abs()
                    > 1e-7)
                || numeric(&targets[slot])?.abs() > f64::from(MAXIMUM_BOUNDED_THUMB_ANGLE_RAD)
                || numeric(&initial[slot])?.abs() > 0.02
            {
                return Err("foreign thumb target or initial posture".into());
            }
        }
        let bounded_targets = if both_hands {
            let right = proposal["target_right_thumb_angles_rad"]
                .as_array()
                .ok_or("right thumb targets absent")?;
            let initial_right = proposal["measured_initial_right_thumb_angles_rad"]
                .as_array()
                .ok_or("right thumb measurements absent")?;
            if right.len() != 2 || initial_right.len() != 2 {
                return Err("right thumb angle width mismatch".into());
            }
            for slot in 0..2 {
                if numeric(&right[slot])?.abs() > f64::from(MAXIMUM_BOUNDED_THUMB_ANGLE_RAD)
                    || numeric(&initial_right[slot])?.abs() > 0.02
                {
                    return Err("foreign right thumb target or initial posture".into());
                }
            }
            let path = &proposal["hypothetical_preparation_path"];
            if !candidate
                || proposal["maximum_absolute_thumb_angle_rad"] != 0.25
                || proposal["planning_clearance_m"] != 0.018
                || proposal["planning_method"] != "fixed_support_planes_bounded_slsqp_v1"
                || path["schema"] != "g1_hypothetical_thumb_support_plane_path_v1"
                || path["samples"] != 51
                || !(0. ..=0.001)
                    .contains(&numeric(&path["maximum_additional_inward_projection_m"])?)
                || path["maximum_allowed_additional_inward_projection_m"] != 0.001
                || path["all_original_hand_hulls"] != 16
                || path["actual_contact_safety_proven"] != false
            {
                return Err("unsafe or unbounded hypothetical thumb path".into());
            }
            Some([
                numeric(&targets[0])? as f32,
                numeric(&targets[1])? as f32,
                numeric(&right[0])? as f32,
                numeric(&right[1])? as f32,
            ])
        } else {
            None
        };
        let actual: ObservationStamp =
            serde_json::from_value(proposal["observation"].clone()).map_err(|e| e.to_string())?;
        if actual != observation
            || reply["camera_mount_profile"] != "auxiliary_bin_placement"
            || reply["placement_view_only"] != true
            || release["release_admitted"] != false
            || release["hand_clearance_admitted"] != false
            || numeric(&release["minimum_signed_floor_margin_m"])? < 0.02
            || !(0.05..=0.4).contains(&numeric(&release["estimated_drop_height_m"])?)
            || numeric(&release["visible_bin_upward_cosine"])? < 0.98
            || !(0. ..=0.05).contains(&numeric(&release["self_root_speed_m_s"])?)
            || proposal["robot_collision_geometry_sha256"] != reply["robot_definition_sha256"]
            || proposal["source_geometry_sha256"] != geometry_hash
            || proposal["world_or_contact_truth_input"] != false
            || proposal["candidate_is_not_measured_state"] != true
            || proposal["fresh_release_observation_required"] != true
            || proposal["release_authorized"] != false
            || proposal["ramp_ticks"] != 50
            || proposal["settling_ticks"] != 50
            || proposal["maximum_tracking_error_rad"] != 0.02
            || proposal["minimum_candidate_clearance_m"] != 0.016
            || clearance["clearance_method"] != "original_convex_vertex_support_plane_v1"
            || clearance["robot_collision_geometry_sha256"] != reply["robot_definition_sha256"]
            || clearance["opening_profile"] != "gravity_horizontal_original_fingers"
            || clearance["minimum_hand_clearance_m"] != 0.01
            || clearance["maximum_commanded_palm_gap_m"] != 0.35
            || clearance["prediction_is_not_physical_detachment"] != true
            || clearance["certified_hand_colliders"] != 16
            || !(0. ..=1.).contains(&gap)
            || gap == 0.
            || !(0. ..=1.).contains(&minimum)
            || clearance["hand_clearance_admitted"] != (gap <= 0.35)
            || proposal["preparation_admitted"] != candidate
        {
            return Err("foreign/unsafe thumb preparation admission".into());
        }
        if candidate {
            let goal: MobileThumbGoal =
                serde_json::from_value(proposal["preparation_goal"].clone())
                    .map_err(|e| e.to_string())?;
            goal.validate().map_err(|e| e.to_string())?;
            if goal.observation != observation || goal.bounded_thumb_targets_rad != bounded_targets
            {
                return Err("thumb preparation goal detached from its current image".into());
            }
        } else if !proposal["preparation_goal"].is_null() {
            return Err("rejected thumb preparation cannot provide a goal".into());
        }
        Ok(())
    }

    pub(super) fn validate_clearance_reply(
        reply: &serde_json::Value,
        observation: ObservationStamp,
        geometry_hash: &str,
    ) -> Result<(), String> {
        let proposal = &reply["clearance_proposal"];
        let stamp: ObservationStamp =
            serde_json::from_value(proposal["observation"].clone()).map_err(|e| e.to_string())?;
        let rise = proposal["required_raise_m"]
            .as_f64()
            .ok_or("clearance rise absent")?;
        if stamp != observation
            || proposal["schema"] != "g1_visible_marker_clearance_v1"
            || proposal["source_geometry_sha256"] != geometry_hash
            || proposal["source"] != "actual_rgb_marker_pose_and_public_original_collision_geometry"
            || proposal["world_or_contact_truth_input"] != false
            || proposal["task_qualified"] != false
            || proposal["minimum_desired_rim_clearance_m"] != 0.1
            || !rise.is_finite()
            || !(-5. ..=5.).contains(&rise)
        {
            return Err("foreign/unbound public-geometry clearance reply".into());
        }
        if !proposal["raise_goal"].is_null() {
            let goal: simulation_minigame::g1::mobile_raise::MobileRaiseGoal =
                serde_json::from_value(proposal["raise_goal"].clone())
                    .map_err(|e| e.to_string())?;
            goal.validate().map_err(|e| e.to_string())?;
            if goal.observation != observation || (f64::from(goal.distance_m) - rise).abs() > 1e-7 {
                return Err("raise goal detached from actual clearance observation".into());
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn rigid_bin_board_proof_cannot_be_missing_foreign_or_lower_quality() {
            let corner = [[100., 100.], [110., 100.], [110., 110.], [100., 110.]];
            let corners = serde_json::json!([corner, corner, corner]);
            let pose = serde_json::json!([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 1.],
                [0., 0., 0., 1.]
            ]);
            let proof = serde_json::json!({
                "schema":"g1_actual_rgb_public_bin_board_pose_v1", "marker_ids":[21,23,24],
                "solver":"SQPNP_then_bounded30_iteration_LM", "upright_prior_used":false,
                "world_or_contact_truth_input":false, "task_qualified":false,
                "corners_px":corners, "minimum_edges_px":[10.,10.,10.],
                "per_marker_reprojection_rms_px":[0.5,0.5,0.5], "root_from_marker":pose,
            });
            let reply = serde_json::json!({
                "marker_layout_profile":"auxiliary_bin_board_targets", "public_bin_board_pose":proof,
                "detections":[{"marker_id":21,"minimum_edge_px":10.,"reprojection_rms_px":0.5,
                    "root_from_marker":pose,"corners_px":corners[0]}],
            });
            assert!(validate_public_bin_board_reply(&reply, false).is_ok());
            assert!(validate_public_bin_board_reply(&reply, true).is_err());
            for mutation in 0..8 {
                let mut bad = reply.clone();
                match mutation {
                    0 => bad["public_bin_board_pose"] = serde_json::Value::Null,
                    1 => {
                        bad["public_bin_board_pose"]["marker_ids"] = serde_json::json!([21, 23, 23])
                    }
                    2 => {
                        bad["public_bin_board_pose"]["per_marker_reprojection_rms_px"][1] =
                            1.01.into()
                    }
                    3 => bad["public_bin_board_pose"]["corners_px"][1][0][0] = (-1.).into(),
                    4 => bad["public_bin_board_pose"]["minimum_edges_px"][2] = 11.0.into(),
                    5 => bad["public_bin_board_pose"]["upright_prior_used"] = true.into(),
                    6 => bad["public_bin_board_pose"]["root_from_marker"][0][3] = 0.1.into(),
                    _ => bad["marker_layout_profile"] = "auxiliary_grip_targets".into(),
                }
                assert!(
                    validate_public_bin_board_reply(&bad, false).is_err(),
                    "mutation {mutation}"
                );
            }
        }
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
            if config.same_tick_box_pair {
                let secondary = directory.join("secondary");
                fs::create_dir(&secondary).map_err(|e| e.to_string())?;
                fs::copy(
                    f["secondary_image"]
                        .as_str()
                        .ok_or("secondary image absent")?,
                    secondary.join("ego.png"),
                )
                .map_err(|e| e.to_string())?;
                fs::copy(
                    f["secondary_observation"]
                        .as_str()
                        .ok_or("secondary observation absent")?,
                    secondary.join("observation.json"),
                )
                .map_err(|e| e.to_string())?;
            }
            let stamp =
                serde_json::from_value(input["stamp"].clone()).map_err(|e| e.to_string())?;
            let memory = if f["memory"].is_null() {
                None
            } else {
                Some(serde_json::from_value(f["memory"].clone()).map_err(|e| e.to_string())?)
            };
            let job = if f["expected_held_contact_geometry"] == true {
                MarkerVisionJob::start_with_held_contact_geometry(config, directory.clone(), stamp)?
            } else if f["expected_held_box_feedback"] == true {
                MarkerVisionJob::start_with_held_box_feedback(config, directory.clone(), stamp)?
            } else if f["placement_view_only"] == true {
                MarkerVisionJob::start_placement_view(config, directory.clone(), stamp)?
            } else if f["box_view_only"] == true {
                MarkerVisionJob::start_box_view(config, directory.clone(), stamp)?
            } else {
                MarkerVisionJob::start_with_memory(config, directory.clone(), stamp, memory)?
            };
            let reply = loop {
                if let Some(reply) = job.try_take() {
                    break reply;
                }
                if job.started.elapsed() > Duration::from_secs(4) {
                    return Err("worker did not report finite completion".into());
                }
                thread::sleep(Duration::from_millis(5));
            };
            if let Some(expected) = f["expected_rejection_substring"].as_str() {
                let error = reply.err().ok_or("fixture unexpectedly admitted a goal")?;
                if !error.contains(expected) {
                    return Err(format!("unexpected rejection: {error}"));
                }
                fs::write(
                    directory.join("roundtrip_receipt.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "actual_saved_rgb_fixture":true,"fresh_vla_calls":0,"actual_integrations":0,
                        "qualified":false,"expected_rejection":error,"no_view_motion_admitted":true,
                        "wall_ms":job.started.elapsed().as_secs_f64()*1000.,
                    }))
                    .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                return Ok(());
            }
            let result = reply?;
            if f["expected_held_box_feedback"] == true
                && result["held_box_feedback"]["schema"] != "g1_current_paired_rgb_relative_grip_v1"
            {
                return Err("requested current held-box feedback absent".into());
            }
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
        fn box_visibility_cannot_admit_target_navigation_or_missing_distant_old_box() {
            let (mut reply, stamp) = fixture();
            reply["detections"].as_array_mut().unwrap().remove(0);
            reply["navigation_proposal"] = serde_json::Value::Null;
            reply["box_view_only"] = true.into();
            reply["carried_box_detected"] = true.into();
            reply["current_box_palm_center_distance_m"] = 0.19.into();
            let check = |r: &serde_json::Value| {
                validate_reply_for_policy(
                    r,
                    stamp,
                    "image",
                    "input",
                    "definition",
                    None,
                    true,
                    false,
                )
            };
            assert!(check(&reply).is_ok());
            assert!(validate_reply(&reply, stamp, "image", "input", "definition").is_err());
            for mutation in 0..5 {
                let mut bad = reply.clone();
                match mutation {
                    0 => bad["detections"] = serde_json::json!([]),
                    1 => bad["current_box_palm_center_distance_m"] = 0.7.into(),
                    2 => bad["observation"]["frame_id"] = 2.into(),
                    3 => bad["navigation_proposal"] = serde_json::json!({"target_identity":21}),
                    _ => bad["target_memory_used"] = true.into(),
                }
                assert!(check(&bad).is_err());
            }
        }
        fn thumb_fixture() -> (serde_json::Value, ObservationStamp) {
            use simulation_minigame::g1::mobile_thumb::LEFT_THUMB_CLEARANCE_TARGETS;
            let (mut reply, stamp) = fixture();
            reply["placement_view_only"] = true.into();
            reply["camera_mount_profile"] = "auxiliary_bin_placement".into();
            reply["release_proposal"] = serde_json::json!({
                "release_admitted":false,"hand_clearance_admitted":false,
                "minimum_signed_floor_margin_m":0.03,"estimated_drop_height_m":0.25,
                "visible_bin_upward_cosine":1.,"self_root_speed_m_s":0.01,
            });
            reply["thumb_preparation_proposal"] = serde_json::json!({
                "schema":"g1_visible_station_thumb_preparation_v1","observation":stamp,
                "source_geometry_sha256":"geometry","robot_collision_geometry_sha256":"definition",
                "world_or_contact_truth_input":false,"candidate_is_not_measured_state":true,
                "fresh_release_observation_required":true,"release_authorized":false,
                "target_left_thumb_angles_rad":LEFT_THUMB_CLEARANCE_TARGETS,
                "measured_initial_left_thumb_angles_rad":[0.,0.],
                "ramp_ticks":50,"settling_ticks":50,"maximum_tracking_error_rad":0.02,
                "minimum_candidate_clearance_m":0.016,"preparation_admitted":true,
                "preparation_goal":{"observation":stamp},
                "candidate_clearance":{
                    "clearance_method":"original_convex_vertex_support_plane_v1",
                    "robot_collision_geometry_sha256":"definition",
                    "opening_profile":"gravity_horizontal_original_fingers",
                    "minimum_hand_clearance_m":0.01,"maximum_commanded_palm_gap_m":0.35,
                    "required_measured_palm_gap_m":0.32,"hand_clearance_admitted":true,
                    "minimum_certified_separation_at_maximum_gap_m":0.018,
                    "certified_hand_colliders":16,"prediction_is_not_physical_detachment":true,
                },
            });
            (reply, stamp)
        }
        #[test]
        fn thumb_preparation_cannot_authorize_release_or_cross_image_and_geometry_guards() {
            let (mut reply, stamp) = thumb_fixture();
            assert!(validate_thumb_preparation_reply(&reply, stamp, "geometry").is_ok());
            for mutation in 0..10 {
                let mut bad = reply.clone();
                match mutation {
                    0 => {
                        bad["thumb_preparation_proposal"]["preparation_goal"]["observation"]["episode_id"] =
                            9.into()
                    }
                    1 => bad["thumb_preparation_proposal"]["observation"]["frame_id"] = 9.into(),
                    2 => bad["camera_mount_profile"] = "auxiliary_grip_overview".into(),
                    3 => bad["thumb_preparation_proposal"]["release_authorized"] = true.into(),
                    4 => {
                        bad["thumb_preparation_proposal"]["candidate_clearance"]["minimum_certified_separation_at_maximum_gap_m"] =
                            0.009.into()
                    }
                    5 => {
                        bad["thumb_preparation_proposal"]["target_left_thumb_angles_rad"][0] =
                            0.1.into()
                    }
                    6 => {
                        bad["thumb_preparation_proposal"]["measured_initial_left_thumb_angles_rad"]
                            [0] = 0.1.into()
                    }
                    7 => bad["release_proposal"]["minimum_signed_floor_margin_m"] = 0.001.into(),
                    8 => {
                        bad["thumb_preparation_proposal"]["candidate_clearance"]["clearance_method"] =
                            "other".into()
                    }
                    _ => {
                        bad["thumb_preparation_proposal"]["candidate_clearance"]["certified_hand_colliders"] =
                            15.into()
                    }
                }
                assert!(
                    validate_thumb_preparation_reply(&bad, stamp, "geometry").is_err(),
                    "mutation {mutation}"
                );
            }
            reply["thumb_preparation_proposal"]["preparation_admitted"] = false.into();
            reply["thumb_preparation_proposal"]["preparation_goal"] = serde_json::Value::Null;
            reply["thumb_preparation_proposal"]["candidate_clearance"]["minimum_certified_separation_at_maximum_gap_m"] =
                0.014.into();
            assert!(validate_thumb_preparation_reply(&reply, stamp, "geometry").is_ok());
        }
        #[test]
        fn both_hand_preparation_binds_four_targets_and_the_finite_outward_path() {
            let (mut reply, stamp) = thumb_fixture();
            let targets = [0.136_114_98, -0.25, 0.062_134_184, 0.181_736_25];
            let p = &mut reply["thumb_preparation_proposal"];
            p["schema"] = "g1_visible_station_thumb_preparation_v2".into();
            p["target_left_thumb_angles_rad"] = serde_json::json!(&targets[..2]);
            p["target_right_thumb_angles_rad"] = serde_json::json!(&targets[2..]);
            p["measured_initial_right_thumb_angles_rad"] = serde_json::json!([0., 0.]);
            p["maximum_absolute_thumb_angle_rad"] = 0.25.into();
            p["planning_clearance_m"] = 0.018.into();
            p["planning_method"] = "fixed_support_planes_bounded_slsqp_v1".into();
            p["preparation_goal"]["bounded_thumb_targets_rad"] = serde_json::json!(targets);
            p["hypothetical_preparation_path"] = serde_json::json!({
                "schema":"g1_hypothetical_thumb_support_plane_path_v1","samples":51,
                "maximum_additional_inward_projection_m":0.0002,
                "maximum_allowed_additional_inward_projection_m":0.001,
                "all_original_hand_hulls":16,"actual_contact_safety_proven":false,
            });
            assert!(validate_thumb_preparation_reply(&reply, stamp, "geometry").is_ok());
            for mutation in 0..13 {
                let mut bad = reply.clone();
                let p = &mut bad["thumb_preparation_proposal"];
                match mutation {
                    0 => p["target_right_thumb_angles_rad"][0] = 0.251.into(),
                    1 => p["target_right_thumb_angles_rad"] = serde_json::json!([0.]),
                    2 => p["measured_initial_right_thumb_angles_rad"][0] = 0.021.into(),
                    3 => p["preparation_goal"]["bounded_thumb_targets_rad"][2] = 0.1.into(),
                    4 => {
                        p["preparation_goal"]
                            .as_object_mut()
                            .unwrap()
                            .remove("bounded_thumb_targets_rad");
                    }
                    5 => {
                        p["hypothetical_preparation_path"]["maximum_additional_inward_projection_m"] =
                            0.0011.into()
                    }
                    6 => {
                        p["hypothetical_preparation_path"]["actual_contact_safety_proven"] =
                            true.into()
                    }
                    7 => p["hypothetical_preparation_path"]["all_original_hand_hulls"] = 15.into(),
                    8 => p["hypothetical_preparation_path"]["samples"] = 50.into(),
                    9 => {
                        p["hypothetical_preparation_path"]["maximum_allowed_additional_inward_projection_m"] =
                            0.002.into()
                    }
                    10 => p["planning_clearance_m"] = 0.01.into(),
                    11 => p["schema"] = "g1_visible_station_thumb_preparation_v1".into(),
                    _ => p["target_left_thumb_angles_rad"][0] = serde_json::Value::Null,
                }
                assert!(
                    validate_thumb_preparation_reply(&bad, stamp, "geometry").is_err(),
                    "mutation {mutation}"
                );
            }
            let (mut legacy, stamp) = thumb_fixture();
            legacy["thumb_preparation_proposal"]["preparation_goal"]["bounded_thumb_targets_rad"] =
                serde_json::json!([0., 0., 0., 0.]);
            assert!(validate_thumb_preparation_reply(&legacy, stamp, "geometry").is_err());
        }
        #[test]
        fn release_cannot_cross_frame_geometry_margin_or_speed_guards() {
            let (mut reply, stamp) = fixture();
            reply["navigation_proposal"] = serde_json::Value::Null;
            reply["placement_view_only"] = true.into();
            reply["camera_mount_profile"] = "auxiliary_grip_overview".into();
            reply["release_proposal"] = serde_json::json!({
                "schema":"g1_visible_marker_release_admission_v2","observation":stamp,
                "source_geometry_sha256":"geometry","world_or_contact_truth_input":false,"task_qualified":false,
                "source":"actual_two_marker_rgb_and_public_original_collision_vertices",
                "minimum_signed_floor_margin_m":0.03,"minimum_required_floor_margin_m":0.02,
                "estimated_drop_height_m":0.25,"visible_bin_upward_cosine":1.,"self_root_speed_m_s":0.01,
                "opening_profile":"gravity_horizontal_original_fingers","robot_collision_geometry_sha256":"definition",
                "minimum_hand_clearance_m":0.01,"required_measured_palm_gap_m":0.32,"maximum_commanded_palm_gap_m":0.35,
                "hand_clearance_admitted":true,"prediction_is_not_physical_detachment":true,
                "release_admitted":true,"release_goal":{"observation":stamp,"target_palm_gap_m":0.35,"duration_ticks":100},
            });
            assert!(validate_release_reply(&reply, stamp, "geometry").is_ok());
            let mut placement_camera = reply.clone();
            placement_camera["camera_mount_profile"] = "auxiliary_bin_placement".into();
            assert!(
                validate_reply_for_policy(
                    &placement_camera,
                    stamp,
                    "image",
                    "input",
                    "definition",
                    None,
                    false,
                    true
                )
                .is_ok()
            );
            assert!(
                validate_reply_for_policy(
                    &reply,
                    stamp,
                    "image",
                    "input",
                    "definition",
                    None,
                    false,
                    true
                )
                .is_ok()
            );
            assert!(validate_reply(&reply, stamp, "image", "input", "definition").is_err());
            for mutation in 0..14 {
                let mut bad = reply.clone();
                let p = &mut bad["release_proposal"];
                match mutation {
                    0 => p["observation"]["frame_id"] = 1.into(),
                    1 => p["source_geometry_sha256"] = "foreign".into(),
                    2 => p["minimum_signed_floor_margin_m"] = 0.019.into(),
                    3 => p["estimated_drop_height_m"] = 0.401.into(),
                    4 => p["self_root_speed_m_s"] = 0.051.into(),
                    5 => p["world_or_contact_truth_input"] = true.into(),
                    6 => p["release_goal"]["observation"]["episode_id"] = 9.into(),
                    7 => p["release_goal"]["duration_ticks"] = 150.into(),
                    8 => p["release_goal"]["target_palm_gap_m"] = 0.3.into(),
                    9 => p["opening_profile"] = "tilted".into(),
                    10 => p["robot_collision_geometry_sha256"] = "foreign".into(),
                    11 => p["required_measured_palm_gap_m"] = 0.351.into(),
                    12 => p["hand_clearance_admitted"] = false.into(),
                    _ => p["prediction_is_not_physical_detachment"] = false.into(),
                }
                assert!(
                    validate_release_reply(&bad, stamp, "geometry").is_err(),
                    "mutation {mutation}"
                );
            }
            reply["release_proposal"]["minimum_signed_floor_margin_m"] = (-0.01).into();
            reply["release_proposal"]["release_admitted"] = false.into();
            reply["release_proposal"]["release_goal"] = serde_json::Value::Null;
            assert!(validate_release_reply(&reply, stamp, "geometry").is_ok());
            reply["release_proposal"]["minimum_signed_floor_margin_m"] = 0.03.into();
            reply["release_proposal"]["required_measured_palm_gap_m"] = 0.36.into();
            reply["release_proposal"]["hand_clearance_admitted"] = false.into();
            assert!(validate_release_reply(&reply, stamp, "geometry").is_ok());
        }
        #[test]
        fn fine_interval_cannot_detach_from_current_image_or_navigation() {
            let (mut reply, stamp) = fixture();
            reply["camera_mount_profile"] = "auxiliary_grip_overview".into();
            reply["fine_approach_proposal"] = serde_json::json!({
                "schema":"g1_visible_marker_fine_carry_v2","observation":stamp,
                "source_geometry_sha256":"geometry","world_or_contact_truth_input":false,"task_qualified":false,
                "physical_containment_distance_interval_m":[0.56,0.68],"selected_physical_distance_m":0.15,
                "state":"advance","current_floor_margin_m":-0.4,"maximum_physical_step_m":0.15,
                "new_image_required_after_step":true,"step_is_final":false,
                "existing_navigation_stop_margin_m":0.05,"predicted_floor_margin_m":-0.25,
                "goal":{"observation":stamp,"heading_yaw_source_rad":-1.57,"relative_distance_m":0.1},
            });
            assert!(validate_fine_reply(&reply, stamp, "geometry").is_ok());
            // Frozen RGB failures had feasible intervals across/beyond 0.70 m.
            // Only the next 0.15 m command is admitted, followed by a new image.
            for interval in [[0.663, 0.775], [0.747, 0.848]] {
                let mut distant = reply.clone();
                distant["fine_approach_proposal"]["physical_containment_distance_interval_m"] =
                    serde_json::json!(interval);
                assert!(validate_fine_reply(&distant, stamp, "geometry").is_ok());
            }
            let mut unreachable = reply.clone();
            unreachable["fine_approach_proposal"]["physical_containment_distance_interval_m"] =
                serde_json::json!([0.751, 0.85]);
            assert!(validate_fine_reply(&unreachable, stamp, "geometry").is_err());
            let mut placement_camera = reply.clone();
            placement_camera["camera_mount_profile"] = "auxiliary_bin_placement".into();
            assert!(validate_fine_reply(&placement_camera, stamp, "geometry").is_ok());
            for mutation in 0..9 {
                let mut bad = reply.clone();
                let p = &mut bad["fine_approach_proposal"];
                match mutation {
                    0 => p["observation"]["episode_id"] = 1.into(),
                    1 => p["source_geometry_sha256"] = "foreign".into(),
                    2 => p["selected_physical_distance_m"] = 0.69.into(),
                    3 => p["existing_navigation_stop_margin_m"] = 0.1.into(),
                    4 => p["step_is_final"] = true.into(),
                    5 => p["goal"]["heading_yaw_source_rad"] = (-1.5).into(),
                    6 => p["new_image_required_after_step"] = false.into(),
                    7 => p["goal"]["relative_distance_m"] = 0.57.into(),
                    _ => p["state"] = "aligned".into(),
                }
                assert!(validate_fine_reply(&bad, stamp, "geometry").is_err());
            }
            let p = &mut reply["fine_approach_proposal"];
            p["state"] = "aligned".into();
            p["current_floor_margin_m"] = 0.022.into();
            p["predicted_floor_margin_m"] = 0.022.into();
            p["selected_physical_distance_m"] = serde_json::Value::Null;
            p["goal"] = serde_json::Value::Null;
            p["physical_containment_distance_interval_m"] = serde_json::json!([0., 0.1]);
            assert!(validate_fine_reply(&reply, stamp, "geometry").is_ok());
            reply["navigation_proposal"] = serde_json::Value::Null;
            assert!(validate_reply(&reply, stamp, "image", "input", "definition").is_ok());
            let mut foreign_camera = reply.clone();
            foreign_camera["camera_mount_profile"] = "arena_ego".into();
            assert!(
                validate_reply(&foreign_camera, stamp, "image", "input", "definition").is_err()
            );
            let mut bad = reply.clone();
            bad["fine_approach_proposal"]["current_floor_margin_m"] = 0.019.into();
            bad["fine_approach_proposal"]["predicted_floor_margin_m"] = 0.019.into();
            assert!(validate_fine_reply(&bad, stamp, "geometry").is_err());
            assert!(validate_reply(&bad, stamp, "image", "input", "definition").is_err());
            bad["fine_approach_proposal"]["state"] = "blocked".into();
            assert!(validate_fine_reply(&bad, stamp, "geometry").is_ok());
        }
        fn memory_fixture() -> (MarkerTargetMemory, ObservationStamp) {
            let current = ObservationStamp {
                episode_id: 7,
                frame_id: 3,
                sim_time_ns: 40_000_000,
                captured_at_unix_ms: 43,
            };
            (
                MarkerTargetMemory {
                    schema: "g1_mobile_static_target_memory_v1".into(),
                    origin_observation: ObservationStamp {
                        frame_id: 2,
                        sim_time_ns: 20_000_000,
                        captured_at_unix_ms: 42,
                        ..current
                    },
                    current_observation: current,
                    origin_image_sha256: "a".repeat(64),
                    origin_input_sha256: "b".repeat(64),
                    root_from_target_marker: [
                        [1., 0., 0., 1.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                    ],
                    origin_root_rotation_wxyz: [1., 0., 0., 0.],
                    self_velocity_displacement_source_m: [0.; 3],
                    static_target_assumption: true,
                    world_or_contact_truth_input: false,
                },
                current,
            )
        }
        #[test]
        fn target_memory_expires_and_cannot_cross_reset_or_replace_current_camera() {
            let (memory, current) = memory_fixture();
            assert!(memory.validate(current).is_ok());
            for mutation in 0..5 {
                let mut bad = memory.clone();
                match mutation {
                    0 => bad.origin_observation.episode_id = 8,
                    1 => bad.origin_observation.frame_id = current.frame_id,
                    2 => {
                        bad.current_observation.sim_time_ns = 8_040_000_000;
                    }
                    3 => bad.self_velocity_displacement_source_m = [0.26, 0., 0.],
                    _ => bad.world_or_contact_truth_input = true,
                }
                assert!(bad.validate(bad.current_observation).is_err());
            }
        }
        #[test]
        fn occluded_target_reply_requires_bound_memory_and_current_box_image() {
            let (memory, current) = memory_fixture();
            let (mut reply, _) = fixture();
            reply["observation"] = serde_json::to_value(current).unwrap();
            reply["navigation_proposal"]["observation"] = serde_json::to_value(current).unwrap();
            reply["detections"].as_array_mut().unwrap().remove(0);
            reply["navigation_proposal"]["source"] =
                "actual_box_rgb_and_static_target_memory_self_velocity".into();
            reply["target_memory_used"] = true.into();
            reply["target_memory_estimate"] = serde_json::json!({
                "schema":"g1_propagated_static_visual_target_v1","observation":current,
                "origin_observation":memory.origin_observation,"target_identity":21,"memory_sha256":"memory",
                "origin_image_sha256":memory.origin_image_sha256,"origin_input_sha256":memory.origin_input_sha256,
                "age_sim_ns":20_000_000,"age_wall_ms":1,"static_target_assumption":true,
                "world_or_contact_truth_input":false,"root_from_marker":memory.root_from_target_marker,"task_qualified":false,
            });
            assert!(
                validate_reply_for_memory(
                    &reply,
                    current,
                    "image",
                    "input",
                    "definition",
                    Some((&memory, "memory"))
                )
                .is_ok()
            );
            assert!(validate_reply(&reply, current, "image", "input", "definition").is_err());
            for mutation in 0..4 {
                let mut bad = reply.clone();
                match mutation {
                    0 => bad["target_memory_estimate"]["memory_sha256"] = "other".into(),
                    1 => bad["target_memory_estimate"]["age_sim_ns"] = 40_000_000.into(),
                    2 => {
                        bad["target_memory_estimate"]["origin_observation"]["episode_id"] = 8.into()
                    }
                    _ => {
                        bad["detections"].as_array_mut().unwrap().clear();
                    }
                }
                assert!(
                    validate_reply_for_memory(
                        &bad,
                        current,
                        "image",
                        "input",
                        "definition",
                        Some((&memory, "memory"))
                    )
                    .is_err()
                );
            }
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

        #[test]
        fn clearance_rejects_old_frame_foreign_geometry_and_unbound_raise() {
            let (mut reply, stamp) = fixture();
            reply["clearance_proposal"] = serde_json::json!({
                "schema":"g1_visible_marker_clearance_v1", "observation":stamp,
                "source_geometry_sha256":"geometry", "source":"actual_rgb_marker_pose_and_public_original_collision_geometry",
                "world_or_contact_truth_input":false,"task_qualified":false,
                "minimum_desired_rim_clearance_m":0.1,"required_raise_m":0.13,
                "raise_goal":{"observation":stamp,"distance_m":0.13,"duration_ticks":130},
            });
            assert!(validate_clearance_reply(&reply, stamp, "geometry").is_ok());
            for mutation in 0..4 {
                let mut bad = reply.clone();
                match mutation {
                    0 => bad["clearance_proposal"]["observation"]["frame_id"] = 2.into(),
                    1 => bad["clearance_proposal"]["source_geometry_sha256"] = "other".into(),
                    2 => bad["clearance_proposal"]["raise_goal"]["distance_m"] = 0.18.into(),
                    _ => bad["clearance_proposal"]["world_or_contact_truth_input"] = true.into(),
                }
                assert!(validate_clearance_reply(&bad, stamp, "geometry").is_err());
            }
        }
    }
}
