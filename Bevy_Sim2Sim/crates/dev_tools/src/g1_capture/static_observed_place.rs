//! One owned bounded CPU pair job, followed by a separately typed classical
//! placement. Only actual RGB, named self sensors and public assets enter it.
use super::*;
use robot_minigame::g1::{definition::G1Definition, policy::bound_bytes};
use simulation_minigame::g1::{
    static_observed_place::{OBSERVED_PLACE_TICKS, StaticObservedPlace, StaticObservedPlaceGoal},
    static_place::StaticPlacementGeometry,
    static_startup::{StaticStartupCommand, StaticStartupExecution},
    static_transfer::StaticLeftPalmKinematics,
    worker::TimedCommand,
};
use std::{
    io::Read,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread::JoinHandle,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Detection {
    object_kind: String,
    marker_id: u32,
    root_from_object: [[f64; 4]; 4],
    reprojection_rms_px: f64,
    per_view_reprojection_rms_px: Vec<f64>,
    #[serde(default)]
    observed_view_indices: Option<Vec<u8>>,
    #[serde(default)]
    pixel_method: Option<String>,
    minimum_edge_px: f64,
    optimizer_evaluations: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PairResult {
    schema: String,
    source: String,
    observations: [ObservationStamp; 2],
    image_sha256: [String; 2],
    input_sha256: [String; 2],
    definition_sha256: String,
    fiducial_sha256: String,
    camera_profiles: [String; 2],
    public_camera_baseline_m: f64,
    detections: Vec<Detection>,
    world_or_contact_truth_input: bool,
    actuation_proposed: bool,
    task_qualified: bool,
}
impl PairResult {
    fn goal(
        &self,
        config: &StaticMarkerVisionConfiguration,
        observations: &[ObservationStamp; 2],
        images: &[String; 2],
        inputs: &[String; 2],
        geometry_sha256: &str,
        role_coverage: bool,
    ) -> Result<StaticObservedPlaceGoal, String> {
        let expected = if role_coverage {
            (
                "g1_static_actual_fixed_pair_role_coverage_localization_v2",
                "public_fixed_pair_explicit_per_role_view_coverage_v2",
            )
        } else {
            (
                "g1_static_actual_fixed_pair_rgb_localization_v1",
                "public_existing_fixed_camera_pair_joint_pattern_v1",
            )
        };
        if self.schema != expected.0
            || self.source != expected.1
            || &self.observations != observations
            || &self.image_sha256 != images
            || &self.input_sha256 != inputs
            || self.definition_sha256 != config.definition_sha256
            || self.fiducial_sha256 != config.fiducial_sha256
            || self.camera_profiles != ["static_placement_overview", "arena_ego"]
            || !self.public_camera_baseline_m.is_finite()
            || (self.public_camera_baseline_m - 0.15).abs() > 1e-7
            || self.detections.len() != 2
            || self.world_or_contact_truth_input
            || self.actuation_proposed
            || self.task_qualified
        {
            return Err("foreign, stale or unbounded fixed-pair result".into());
        }
        for (entry, (kind, marker)) in self
            .detections
            .iter()
            .zip([("t1_apple", 31), ("t1_plate", 32)])
        {
            let raw_arena = role_coverage
                && kind == "t1_apple"
                && entry.observed_view_indices.as_deref() == Some(&[1])
                && entry.pixel_method.as_deref()
                    == Some("standard_dictionary_subpixel_corner_pnp_v1");
            let paired_pattern = role_coverage
                && entry.observed_view_indices.as_deref() == Some(&[0, 1])
                && entry.pixel_method.as_deref() == Some("public_fixed_pair_joint_pattern_v1");
            let coverage_valid = if role_coverage {
                raw_arena || paired_pattern
            } else {
                entry.observed_view_indices.is_none() && entry.pixel_method.is_none()
            };
            if !coverage_valid
                || entry.per_view_reprojection_rms_px.len() != if raw_arena { 1 } else { 2 }
                || entry.object_kind != kind
                || entry.marker_id != marker
                || !entry.reprojection_rms_px.is_finite()
                || !(0.0..=1.0).contains(&entry.reprojection_rms_px)
                || entry
                    .per_view_reprojection_rms_px
                    .iter()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || !entry.minimum_edge_px.is_finite()
                || entry.minimum_edge_px < 8.0
                || if raw_arena {
                    entry.optimizer_evaluations != 0
                } else {
                    !(1..=100).contains(&entry.optimizer_evaluations)
                }
            {
                return Err("fixed-pair detection failed its unchanged pixel/identity gate".into());
            }
        }
        let goal = StaticObservedPlaceGoal {
            observations: self.observations,
            image_sha256: self.image_sha256.clone(),
            input_sha256: self.input_sha256.clone(),
            root_from_apple: self.detections[0].root_from_object,
            root_from_plate: self.detections[1].root_from_object,
            public_geometry_sha256: geometry_sha256.into(),
            rigid_grasp_assumption: true,
            static_target_assumption: true,
        };
        goal.validate().map_err(|e| e.to_string())?;
        Ok(goal)
    }
}

struct PairJob {
    child: Child,
    reply: Mutex<Receiver<Result<Vec<u8>, String>>>,
    received: Option<Vec<u8>>,
    reader: Option<JoinHandle<()>>,
    started: Instant,
    output: PathBuf,
    files: [(PathBuf, String); 4],
    observations: [ObservationStamp; 2],
    images: [String; 2],
    inputs: [String; 2],
}
impl Drop for PairJob {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
pub(super) struct StaticObservedPlaceRoute {
    config: StaticMarkerVisionConfiguration,
    program: BackgroundVisualConfiguration,
    role_coverage: bool,
    kinematics: StaticLeftPalmKinematics,
    geometry: StaticPlacementGeometry,
    job: Option<PairJob>,
}
impl StaticObservedPlaceRoute {
    pub(super) fn new(
        config: &StaticMarkerVisionConfiguration,
        program: BackgroundVisualConfiguration,
        geometry: &simulation_minigame::g1::task_objects::TaskObjectSceneConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let name = program.path.file_name().and_then(|n| n.to_str());
        let role_coverage = name == Some("unitree_g1_static_role_coverage_vision.py");
        if program.path.parent() != config.localizer_path.parent()
            || !matches!(
                name,
                Some(
                    "unitree_g1_static_pair_vision.py"
                        | "unitree_g1_static_role_coverage_vision.py"
                )
            )
        {
            return Err(
                "observed placement requires its distinct pinned fixed-pair program".into(),
            );
        }
        bound_bytes(&program.path, &program.sha256).map_err(|e| e.to_string())?;
        Ok(Self {
            config: config.clone(),
            program,
            role_coverage,
            kinematics: StaticLeftPalmKinematics::new(
                &G1Definition::load(&config.definition_path, &config.definition_sha256)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
            geometry: StaticPlacementGeometry::load_public_definition(
                &geometry.definition,
                &geometry.definition_sha256,
            )
            .map_err(|e| e.to_string())?,
            job: None,
        })
    }
    pub(super) fn reset_episode(&mut self) {
        self.job = None;
    }
    fn start(&mut self, root: &Path) -> Result<(), String> {
        self.config.validate()?;
        bound_bytes(&self.program.path, &self.program.sha256).map_err(|e| e.to_string())?;
        let folders = [
            root.join("static_grip_vision_input"),
            root.join("static_grip_secondary_vision_input"),
        ];
        let values: Vec<serde_json::Value> = folders
            .iter()
            .map(|p| bounded_json(&p.join("observation.json")))
            .collect::<Result<_, _>>()?;
        super::super::g1_marker_vision::validate_fixed_pair(&values[0], &values[1])?;
        let observations = [
            serde_json::from_value(values[0]["stamp"].clone()).map_err(|e| e.to_string())?,
            serde_json::from_value(values[1]["stamp"].clone()).map_err(|e| e.to_string())?,
        ];
        let images = [
            digest(&folders[0].join("ego.png"), 1024 * 1024)?,
            digest(&folders[1].join("ego.png"), 1024 * 1024)?,
        ];
        let inputs = [
            digest(&folders[0].join("observation.json"), 128 * 1024)?,
            digest(&folders[1].join("observation.json"), 128 * 1024)?,
        ];
        let files = [
            (folders[0].join("ego.png"), images[0].clone()),
            (folders[1].join("ego.png"), images[1].clone()),
            (folders[0].join("observation.json"), inputs[0].clone()),
            (folders[1].join("observation.json"), inputs[1].clone()),
        ];
        let output = root.join("static_pair_localization.json");
        if output.exists() {
            return Err("fixed-pair result output is not fresh".into());
        }
        let log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("static_pair_worker.log"))
            .map_err(|e| e.to_string())?;
        let mut child = Command::new(&self.config.python_path)
            .arg("-u")
            .arg(&self.program.path)
            .arg("--first")
            .arg(&folders[0])
            .arg("--second")
            .arg(&folders[1])
            .arg("--definition")
            .arg(&self.config.definition_path)
            .arg("--fiducials")
            .arg(&self.config.fiducial_path)
            .arg("--fiducial-sha256")
            .arg(&self.config.fiducial_sha256)
            .arg("--output")
            .arg(&output)
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(log)
            .spawn()
            .map_err(|e| e.to_string())?;
        let stdout = child.stdout.take().ok_or("fixed-pair stdout absent")?;
        let (sender, reply) = mpsc::sync_channel(1);
        let reader = std::thread::Builder::new()
            .name("g1-static-pair-result".into())
            .spawn(move || {
                let mut bytes = Vec::new();
                let result = stdout
                    .take(128 * 1024 + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())
                    .and_then(|_| {
                        if bytes.len() > 128 * 1024 {
                            Err("fixed-pair stdout exceeds bound".into())
                        } else {
                            Ok(bytes)
                        }
                    });
                let _ = sender.try_send(result);
            })
            .map_err(|e| {
                let _ = child.kill();
                let _ = child.wait();
                e.to_string()
            })?;
        self.job = Some(PairJob {
            child,
            reply: Mutex::new(reply),
            received: None,
            reader: Some(reader),
            started: Instant::now(),
            output,
            files,
            observations,
            images,
            inputs,
        });
        Ok(())
    }
    fn prepare(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .clone()
            .ok_or("observed placement snapshot absent")?;
        if latest.phase != G1WorkerPhase::Paused {
            return Err("paired placement preparation requires paused owner".into());
        }
        if self.job.is_none() {
            self.start(&runtime.output)?;
            return Ok(false);
        }
        let job = self.job.as_mut().unwrap();
        if job.started.elapsed() > Duration::from_secs(3) {
            return Err("fixed-pair CPU deadline exceeded".into());
        }
        if job.received.is_none() {
            job.received = match job
                .reply
                .lock()
                .map_err(|_| "fixed-pair reply slot poisoned")?
                .try_recv()
            {
                Ok(result) => Some(result?),
                Err(TryRecvError::Empty) => return Ok(false),
                Err(TryRecvError::Disconnected) => {
                    return Err("fixed-pair reader disconnected".into());
                }
            };
        }
        match job.child.try_wait().map_err(|e| e.to_string())? {
            None => return Ok(false),
            Some(status) if !status.success() => return Err("fixed-pair CPU process failed".into()),
            Some(_) => {}
        }
        let bytes = job.received.take().unwrap();
        self.config.validate()?;
        bound_bytes(&self.program.path, &self.program.sha256).map_err(|e| e.to_string())?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if bounded_json(&job.output)? != value {
            return Err("fixed-pair stdout/file identities differ".into());
        }
        for (path, hash) in &job.files {
            if digest(path, 1024 * 1024)? != *hash {
                return Err("fixed-pair input changed during computation".into());
            }
        }
        let result: PairResult =
            serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        let goal = result.goal(
            &self.config,
            &job.observations,
            &job.images,
            &job.inputs,
            self.geometry.definition_sha256(),
            self.role_coverage,
        )?;
        let state = latest
            .measurement
            .as_ref()
            .ok_or("observed placement self absent")?;
        let step = latest
            .startup_step
            .as_ref()
            .ok_or("observed placement current command absent")?;
        let StaticStartupExecution::ObservedGrasp(grasp) = &step.execution else {
            return Err(
                "observed placement requires completed original observed-grasp owner".into(),
            );
        };
        if !grasp.completed {
            return Err("observed placement grasp command is incomplete".into());
        }
        let first = bounded_json(
            &runtime
                .output
                .join("static_grip_vision_input/observation.json"),
        )?;
        let sensors: rendering_minigame::g1_camera::G1CameraJointState =
            serde_json::from_value(first["measured_joints"].clone()).map_err(|e| e.to_string())?;
        if sensors.positions != state.joint_positions
            || sensors.velocities != state.joint_velocities
            || sensors.root_rotation_wxyz != state.root_rotation_wxyz
            || sensors.root_velocity_source != state.root_velocity_source
            || sensors.root_angular_velocity_body != state.root_angular_velocity_body
        {
            return Err("paired RGB differs from current original self sensors".into());
        }
        let mut motion =
            StaticObservedPlace::new(goal.clone(), state, grasp.command.clone(), &self.geometry)
                .map_err(|e| e.to_string())?;
        let mut predicted = state.as_ref().clone();
        let mut receipts = Vec::new();
        for _ in 0..OBSERVED_PLACE_TICKS {
            receipts.push(
                motion
                    .update(&predicted, &self.kinematics)
                    .map_err(|e| e.to_string())?,
            );
            predicted.source_tick += 1;
            predicted.sim_time_ns += 20_000_000;
        }
        let target = state.source_tick + OBSERVED_PLACE_TICKS;
        if target > u64::from(runtime.options.ticks) {
            return Err("observed placement exceeds its explicit integration budget".into());
        }
        fs::write(runtime.output.join("static_observed_place_preflight.json"),serde_json::to_vec_pretty(&serde_json::json!({"schema":"g1_live_fixed_pair_observed_place_preflight_v1","target_tick":target,"geometry_ticks":receipts.len(),"physics_integrations":0,"model_calls":0,"world_or_contact_truth_input":false,"task_qualified":false,"steps":receipts})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        fs::write(
            runtime.output.join("static_observed_place_goal.json"),
            serde_json::to_vec_pretty(&goal).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        // Age is checked again after the whole preflight; no wall-clock restamp.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis() as u64;
        if goal
            .observations
            .iter()
            .any(|s| now < s.captured_at_unix_ms || now - s.captured_at_unix_ms > 2000)
        {
            return Err("actual paired RGB expired during placement preparation".into());
        }
        let image_ages_ms = goal
            .observations
            .map(|stamp| now - stamp.captured_at_unix_ms);
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("observed placement owner is foreign".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: target * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(16),
                command: StaticStartupCommand::ObservedPlace(goal),
            })
            .map_err(|e| e.to_string())?;
        let pair_pid = job.child.id();
        let pair_elapsed_ms = job.started.elapsed().as_millis();
        job.child.wait().map_err(|e| e.to_string())?;
        let mut receipt = outcome.0.lock().unwrap();
        receipt.static_pair_localization = Some(value);
        receipt.static_place_handoff = Some(
            serde_json::json!({"phase":"actual_same_tick_fixed_pair_then_bounded_classical_700tick_place","target_tick":target,"owned_pair_pid":pair_pid,"pair_process_reaped":true,"cpu_job_ms":pair_elapsed_ms,"actual_rgb_age_ms_at_admission":image_ages_ms,"pair_program_sha256":self.program.sha256,"original_vla_calls":1,"current_rgb_restamped":false,"world_or_contact_truth_input":false,"rigid_grasp_assumption":true,"static_target_assumption":true,"task_qualified":false}),
        );
        runtime.static_observed_place_target_tick = Some(target);
        self.job.take();
        Ok(true)
    }
}
fn digest(path: &Path, maximum: u64) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err("fixed-pair file is not bounded regular input".into());
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(path).map_err(|e| e.to_string())?)
    ))
}
fn bounded_json(path: &Path) -> Result<serde_json::Value, String> {
    digest(path, 128 * 1024)?;
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
pub(super) fn prepare(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
) -> Result<bool, String> {
    let mut route = runtime
        .static_observed_place_route
        .take()
        .ok_or("observed placement route absent")?;
    let result = route.prepare(runtime, outcome);
    runtime.static_observed_place_route = Some(route);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (serde_json::Value, StaticMarkerVisionConfiguration) {
        let config:StaticMarkerVisionConfiguration=serde_json::from_value(serde_json::json!({
            "python_path":"/python","python_sha256":"a".repeat(64),"worker_path":"/worker","worker_sha256":"b".repeat(64),
            "localizer_path":"/localizer","localizer_sha256":"c".repeat(64),"definition_path":"/definition","definition_sha256":"d".repeat(64),"fiducial_path":"/fiducial","fiducial_sha256":"e".repeat(64)})).unwrap();
        let pose = [
            [1., 0., 0., 0.3],
            [0., 1., 0., 0.1],
            [0., 0., 1., 0.1],
            [0., 0., 0., 1.],
        ];
        let observations = [
            ObservationStamp {
                episode_id: 7,
                frame_id: 3,
                sim_time_ns: 6920000000,
                captured_at_unix_ms: 10000,
            },
            ObservationStamp {
                episode_id: 7,
                frame_id: 4,
                sim_time_ns: 6920000000,
                captured_at_unix_ms: 10010,
            },
        ];
        (
            serde_json::json!({"schema":"g1_static_actual_fixed_pair_rgb_localization_v1","source":"public_existing_fixed_camera_pair_joint_pattern_v1",
            "observations":observations,"image_sha256":["a".repeat(64),"b".repeat(64)],"input_sha256":["c".repeat(64),"d".repeat(64)],
            "definition_sha256":config.definition_sha256,"fiducial_sha256":config.fiducial_sha256,"camera_profiles":["static_placement_overview","arena_ego"],
            "public_camera_baseline_m":0.15,"detections":[{"object_kind":"t1_apple","marker_id":31,"root_from_object":pose,"reprojection_rms_px":0.2,"per_view_reprojection_rms_px":[0.2,0.3],"minimum_edge_px":14.,"optimizer_evaluations":9},
            {"object_kind":"t1_plate","marker_id":32,"root_from_object":pose,"reprojection_rms_px":0.2,"per_view_reprojection_rms_px":[0.2,0.3],"minimum_edge_px":40.,"optimizer_evaluations":9}],"world_or_contact_truth_input":false,"actuation_proposed":false,"task_qualified":false}),
            config,
        )
    }
    #[test]
    fn admits_only_owned_same_tick_geometry_and_refuses_forged_identity_truth_or_pixel_gate() {
        let (value, config) = fixture();
        let good: PairResult = serde_json::from_value(value.clone()).unwrap();
        let geometry = "f".repeat(64);
        let admit = |v: serde_json::Value| -> Result<StaticObservedPlaceGoal, String> {
            serde_json::from_value::<PairResult>(v)
                .map_err(|e| e.to_string())?
                .goal(
                    &config,
                    &good.observations,
                    &good.image_sha256,
                    &good.input_sha256,
                    &geometry,
                    false,
                )
        };
        assert!(admit(value.clone()).is_ok());
        for (key, changed) in [
            ("world_or_contact_truth_input", serde_json::json!(true)),
            ("actuation_proposed", serde_json::json!(true)),
            ("public_camera_baseline_m", serde_json::json!(0.16)),
            ("schema", serde_json::json!("foreign")),
        ] {
            let mut bad = value.clone();
            bad[key] = changed;
            assert!(admit(bad).is_err());
        }
        let mut bad = value.clone();
        bad["observations"][1]["episode_id"] = 8.into();
        assert!(admit(bad).is_err());
        let mut bad = value.clone();
        bad["image_sha256"][0] = "f".repeat(64).into();
        assert!(admit(bad).is_err());
        let mut bad = value.clone();
        bad["detections"][0]["per_view_reprojection_rms_px"][1] = 1.01.into();
        assert!(admit(bad).is_err());
        let mut bad = value.clone();
        bad["detections"][0]["root_from_object"][0][0] = 2.into();
        assert!(admit(bad).is_err());
        let mut bad = value;
        bad["hidden_truth"] = true.into();
        assert!(admit(bad).is_err());
    }
    #[test]
    fn explicit_one_view_method_cannot_impersonate_paired_pattern_or_supply_plate_from_one_view() {
        let (mut value, config) = fixture();
        value["schema"] = "g1_static_actual_fixed_pair_role_coverage_localization_v2".into();
        value["source"] = "public_fixed_pair_explicit_per_role_view_coverage_v2".into();
        for entry in value["detections"].as_array_mut().unwrap() {
            entry["observed_view_indices"] = serde_json::json!([0, 1]);
            entry["pixel_method"] = "public_fixed_pair_joint_pattern_v1".into();
        }
        value["detections"][0]["observed_view_indices"] = serde_json::json!([1]);
        value["detections"][0]["pixel_method"] =
            "standard_dictionary_subpixel_corner_pnp_v1".into();
        value["detections"][0]["per_view_reprojection_rms_px"] = serde_json::json!([0.2]);
        value["detections"][0]["optimizer_evaluations"] = 0.into();
        let good: PairResult = serde_json::from_value(value.clone()).unwrap();
        let geometry = "f".repeat(64);
        let admit =
            |v: serde_json::Value, coverage: bool| -> Result<StaticObservedPlaceGoal, String> {
                serde_json::from_value::<PairResult>(v)
                    .map_err(|e| e.to_string())?
                    .goal(
                        &config,
                        &good.observations,
                        &good.image_sha256,
                        &good.input_sha256,
                        &geometry,
                        coverage,
                    )
            };
        assert!(admit(value.clone(), true).is_ok());
        assert!(admit(value.clone(), false).is_err());
        let mut bad = value.clone();
        bad["detections"][0]["observed_view_indices"] = serde_json::json!([0]);
        assert!(admit(bad, true).is_err());
        let mut bad = value.clone();
        bad["detections"][0]["pixel_method"] = "public_fixed_pair_joint_pattern_v1".into();
        assert!(admit(bad, true).is_err());
        let mut bad = value.clone();
        bad["detections"][1]["observed_view_indices"] = serde_json::json!([1]);
        bad["detections"][1]["pixel_method"] = "standard_dictionary_subpixel_corner_pnp_v1".into();
        bad["detections"][1]["per_view_reprojection_rms_px"] = serde_json::json!([0.2]);
        bad["detections"][1]["optimizer_evaluations"] = 0.into();
        assert!(admit(bad, true).is_err());
        let mut bad = value;
        bad["detections"][0]["per_view_reprojection_rms_px"] = serde_json::json!([0.2, 0.2]);
        assert!(admit(bad, true).is_err());
    }
}
