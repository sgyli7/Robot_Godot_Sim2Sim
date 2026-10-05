//! Explicit finite Qwen dispatch. The station scope additionally admits the
//! fixed profile from initial RGB before any policy or physics update. Target
//! localization and original policy/classical control remain distinct.
use super::*;
#[cfg(feature = "g1_constraint_diagnostic")]
use sha2::{Digest, Sha256};
#[cfg(feature = "g1_constraint_diagnostic")]
use simulation_minigame::g1::{
    mobile_assist::MobileAssistCommand, mobile_navigation::MobileCarryGoal, worker::TimedCommand,
};
#[cfg(feature = "g1_constraint_diagnostic")]
use task_minigame::{
    decision::{
        DecisionLimits, DecisionSession, DecisionWorker, ExecutionFeedback, FeedbackStatus,
        ObservationSnapshot, ProfileStartRequest, RgbTarget, RgbVerifiedTargets, ValidatedDecision,
    },
    types::{SkillAvailability, SkillRequest},
};
use task_minigame::{
    decision::{LocalQwenClient, LocalQwenConfig, TaskGoal},
    interactive::InteractiveDecisionConfig,
};

#[derive(Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    PostgraspSourceMobileTransportV1,
    PostgraspSourceMobileTransportV2RgbVerifiedSelection,
    ScientificStationMobileFromInstructionV1,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    scope: Scope,
    connection: InteractiveDecisionConfig,
    instruction: String,
}

impl Configuration {
    pub(super) fn validate_entry(
        &self,
        mode: CaptureMode,
        config: &CaptureRunnerConfig,
    ) -> Result<(), String> {
        match self.scope {
            Scope::PostgraspSourceMobileTransportV1
            | Scope::PostgraspSourceMobileTransportV2RgbVerifiedSelection
            | Scope::ScientificStationMobileFromInstructionV1 => {}
        }
        let CaptureRunnerConfig::Task(task) = config else {
            return Err("Qwen transport diagnostic requires the matched task owner".into());
        };
        if mode
            != if self.is_station_initial() {
                CaptureMode::StationMobileRelease
            } else {
                CaptureMode::MobileAuxiliaryRelease
            }
            || task.body.profile() != TaskProfile::MobileBox
            || task.max_observation_wall_age_ms != 20_000
            || task.max_observation_age_ns != 1_000_000_000
            || !(1..=20_000).contains(&self.connection.timeout_ms)
            || !(64..=512).contains(&self.connection.max_output_tokens)
            || (self.uses_selection() && self.connection.max_output_tokens > 128)
        {
            return Err("Qwen dispatch requires the explicit source mobile auxiliary release, unchanged20s/1s image limits, bounded20s transport and V1<=512/V2<=128tokens".into());
        }
        self.goal().validate().map_err(|e| e.to_string())?;
        self.client().map(|_| ())
    }

    fn goal(&self) -> TaskGoal {
        TaskGoal {
            instruction: self.instruction.clone(),
            profile: TaskProfile::MobileBox,
            public_scene_description: format!(
                "Finite post-grasp {} development diagnostic, not task qualification. The four original grasp chunks and public-map search already completed. Only the original brown-box-to-blue-bin task is supported. The publicly printed carried-box marker22 names box_marker_22; the target-zone marker21 names bin_marker_21. Use these IDs for the visibly supported box and blue placement zone. No navigation adjustment is available.",
                if self.is_station_initial() {
                    "scientific-station fixed-profile"
                } else {
                    "source-task"
                }
            ),
        }
    }

    pub(super) fn is_station_initial(&self) -> bool {
        matches!(self.scope, Scope::ScientificStationMobileFromInstructionV1)
    }

    fn initial_goal(&self) -> TaskGoal {
        TaskGoal {
            instruction: self.instruction.clone(),
            profile: TaskProfile::MobileBox,
            public_scene_description: "Declared finite scientific-station development profile, not task qualification. Only the fixed original brown-box-to-blue-bin mobile task is supported. Begin admits four original N1.6 grasp chunks and the disclosed public-aisle search. It does not assert that the destination is visible or located. Transport still requires another Qwen decision over both actual RGB marker22(box) and marker21(blue placement zone); release requires separate new calibrated RGB checks. No apple task, arbitrary object/destination or navigation adjustment is available.".into(),
        }
    }

    fn uses_selection(&self) -> bool {
        matches!(
            self.scope,
            Scope::PostgraspSourceMobileTransportV2RgbVerifiedSelection
                | Scope::ScientificStationMobileFromInstructionV1
        )
    }

    fn scope_name(&self) -> &'static str {
        match self.scope {
            Scope::PostgraspSourceMobileTransportV1 => "postgrasp_source_mobile_transport_v1",
            Scope::PostgraspSourceMobileTransportV2RgbVerifiedSelection => {
                "postgrasp_source_mobile_transport_v2_rgb_verified_selection"
            }
            Scope::ScientificStationMobileFromInstructionV1 => {
                "scientific_station_mobile_from_instruction_v1"
            }
        }
    }

    fn client(&self) -> Result<LocalQwenClient, String> {
        LocalQwenClient::new(LocalQwenConfig {
            base_url: self.connection.endpoint.clone(),
            model: self.connection.model.clone(),
            timeout: Duration::from_millis(self.connection.timeout_ms),
            max_output_tokens: self.connection.max_output_tokens,
        })
        .map_err(|e| e.to_string())
    }
}

#[cfg(feature = "g1_constraint_diagnostic")]
struct PendingTransport {
    observation: ObservationStamp,
    qwen_stamp: ObservationStamp,
    localization: serde_json::Value,
    directory: PathBuf,
    image_sha256: String,
    stamp_sha256: String,
    input_sha256: String,
    selection_evidence: Option<RgbVerifiedTargets>,
    started: Instant,
}

#[cfg(feature = "g1_constraint_diagnostic")]
struct PendingInitial {
    stamp: ObservationStamp,
    directory: PathBuf,
    image_sha256: String,
    capture_sha256: String,
    request_sha256: String,
}

#[cfg(feature = "g1_constraint_diagnostic")]
pub(super) struct Dispatch {
    configuration: Configuration,
    worker: DecisionWorker,
    session: Option<DecisionSession>,
    pending: Option<PendingTransport>,
    executed_decision: Option<ValidatedDecision>,
    feedback_pending: bool,
    completed: bool,
    initial_pending: Option<PendingInitial>,
    initial_admitted: bool,
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn unix_ms() -> Result<u64, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis()
        .try_into()
        .map_err(|_| "wall clock overflow".into())
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn input_hash(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn supported_request(decision: &ValidatedDecision) -> Result<(), String> {
    match &decision.request {
        SkillRequest::ExecuteTask { task_profile: TaskProfile::MobileBox, target_id, destination_id }
            if target_id == "box_marker_22" && destination_id == "bin_marker_21" => Ok(()),
        SkillRequest::Stop { reason } => Err(format!("Qwen requested an explicit stop: {reason}")),
        SkillRequest::Observe => Err("Qwen requested observation; this finite diagnostic admits no automatic physical continuation".into()),
        _ => Err("Qwen request does not bind the supported public source-task marker roles".into()),
    }
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn require_marker_roles(reply: &serde_json::Value) -> Result<(), String> {
    let detections = reply["detections"]
        .as_array()
        .ok_or("Qwen binding lacks current RGB detections")?;
    for (kind, id) in [("carried_box", 22_u64), ("target_bin", 21)] {
        if detections
            .iter()
            .filter(|d| d["kind"] == kind && d["marker_id"].as_u64() == Some(id))
            .count()
            != 1
        {
            return Err(
                "Qwen dispatch requires both unique public source-task marker roles in current RGB"
                    .into(),
            );
        }
    }
    if reply["world_or_contact_truth_input"] != false
        || reply["task_qualified"] != false
        || !matches!(
            reply["camera_mount_profile"].as_str(),
            Some("auxiliary_grip_overview" | "auxiliary_bin_placement" | "auxiliary_bin_survey")
        )
    {
        return Err("Qwen source-task geometry has foreign provenance".into());
    }
    Ok(())
}

/// Detector measurements stay distinct from Qwen output. Bounds describe the
/// visible printed marker, not an invented full-object detection or world pose.
#[cfg(feature = "g1_constraint_diagnostic")]
fn current_rgb_targets(
    snapshot: &ObservationSnapshot,
    reply: &serde_json::Value,
    raw_image_hash: &str,
    input_hash: &str,
    control_stamp: ObservationStamp,
) -> Result<RgbVerifiedTargets, String> {
    require_marker_roles(reply)?;
    if reply["actual_rgb_only_object_measurement"] != true
        || reply["image_sha256"] != raw_image_hash
        || reply["input_sha256"] != input_hash
        || reply["observation"] != serde_json::to_value(control_stamp).unwrap()
        || snapshot.stamp.episode_id != control_stamp.episode_id
        || snapshot.stamp.sim_time_ns != control_stamp.sim_time_ns
        || snapshot.stamp.captured_at_unix_ms != control_stamp.captured_at_unix_ms
    {
        return Err("selector detector evidence is not this exact RGB/control acquisition".into());
    }
    let mut targets = Vec::new();
    for (role, marker_id, id, kind, description) in [
        (
            "carried_box",
            22_u64,
            "box_marker_22",
            task_minigame::decision::TargetKind::Box,
            "brown box associated with printed marker22",
        ),
        (
            "target_bin",
            21_u64,
            "bin_marker_21",
            task_minigame::decision::TargetKind::PlacementZone,
            "blue placement zone associated with printed marker21",
        ),
    ] {
        let detection = reply["detections"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["kind"] == role && d["marker_id"] == marker_id)
            .unwrap();
        let corners: [[f32; 2]; 4] =
            serde_json::from_value(detection["corners_px"].clone()).map_err(|e| e.to_string())?;
        let minimum = corners
            .iter()
            .fold([f32::INFINITY; 2], |a, p| [a[0].min(p[0]), a[1].min(p[1])]);
        let maximum = corners.iter().fold([f32::NEG_INFINITY; 2], |a, p| {
            [a[0].max(p[0]), a[1].max(p[1])]
        });
        targets.push(RgbTarget {
            id: id.into(),
            kind,
            public_description: description.into(),
            marker_id: marker_id as u32,
            marker_bbox_xyxy: [
                minimum[0] / snapshot.camera.width() as f32,
                minimum[1] / snapshot.camera.height() as f32,
                maximum[0] / snapshot.camera.width() as f32,
                maximum[1] / snapshot.camera.height() as f32,
            ],
            minimum_edge_px: detection["minimum_edge_px"]
                .as_f64()
                .ok_or("RGB detector edge absent")? as f32,
            reprojection_rms_px: detection["reprojection_rms_px"]
                .as_f64()
                .ok_or("RGB detector residual absent")? as f32,
        });
    }
    RgbVerifiedTargets::bind_current_rgb(snapshot, targets).map_err(|e| e.to_string())
}

#[cfg(feature = "g1_constraint_diagnostic")]
impl Dispatch {
    pub(super) fn new(configuration: Configuration, episode_id: u64) -> Result<Self, String> {
        let worker = DecisionWorker::spawn(configuration.client()?, episode_id)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            configuration,
            worker,
            session: None,
            pending: None,
            executed_decision: None,
            feedback_pending: false,
            completed: false,
            initial_pending: None,
            initial_admitted: false,
        })
    }

    fn record(&self, outcome: &CaptureOutcome, event: serde_json::Value) -> Result<(), String> {
        let mut receipt = outcome
            .0
            .lock()
            .map_err(|_| "Qwen transport receipt poisoned")?;
        merge_mobile_handoff(&mut receipt.mobile_assist_handoff, serde_json::json!({}))?;
        let handoff = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("Qwen transport handoff absent")?;
        if handoff["qwen_postgrasp_transport"].is_null() {
            handoff["qwen_postgrasp_transport"] = serde_json::json!({
                "scope":self.configuration.scope_name(),"rgb_verified_selection_wire":self.configuration.uses_selection(),
                "detector_measurements_are_model_claims":false,"initial_grasp_and_search_qwen_controlled":false,
                "physics_paused_during_qwen":true,"public_marker_roles":{"box_marker_22":22,"bin_marker_21":21},
                "maximum_transport_admissions":1,"world_or_contact_truth_input":false,"task_qualified":false,"events":[],
                "initial_fixed_profile_admission_required":self.configuration.is_station_initial(),
                "initial_grasp_and_search_use_disclosed_fixed_executor":true,
            });
        }
        let events = handoff["qwen_postgrasp_transport"]["events"]
            .as_array_mut()
            .ok_or("Qwen event array absent")?;
        if events.len() >= 8 {
            return Err("Qwen transport event budget exhausted".into());
        }
        events.push(event);
        let (attempts, results, discarded) = self.worker.transport_counts();
        handoff["qwen_postgrasp_transport"]["http_attempts"] = attempts.into();
        handoff["qwen_postgrasp_transport"]["http_results"] = results.into();
        handoff["qwen_postgrasp_transport"]["discarded_results"] = discarded.into();
        Ok(())
    }

    /// No action chunk, standing command or integration may precede admission.
    pub(super) fn initial(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        port: &G1CameraPort,
    ) -> Result<bool, String> {
        if !self.configuration.is_station_initial() || self.initial_admitted {
            return Ok(true);
        }
        let latest = runtime.latest.as_ref().ok_or("initial Qwen lacks owner")?;
        if latest.phase != G1WorkerPhase::Paused
            || latest.episode_id != runtime.episode_id
            || latest.timing.episode_integrations != 0
            || runtime.startup_ticks != 0
            || runtime
                .live_policy
                .as_ref()
                .is_none_or(|p| p.submitted_chunks != 0 || p.pending.is_some())
        {
            return Err("initial Qwen requires the unchanged zero-Tick native boundary before any policy inference".into());
        }
        if let Some(pending) = &self.initial_pending {
            let session = self.session.as_mut().ok_or("initial Qwen session absent")?;
            if session.expire_pending(unix_ms()?, 0) {
                return Err("initial Qwen image expired; zero-Tick owner remains paused".into());
            }
            let Some(reply) = self
                .worker
                .try_recv_profile_start()
                .map_err(|e| e.to_string())?
            else {
                return Ok(false);
            };
            let admitted = session
                .accept_profile_start(reply, unix_ms()?, 0)
                .map_err(|e| e.to_string())?;
            if admitted.stamp != pending.stamp
                || input_hash(&pending.directory.join("ego.png"))? != pending.image_sha256
                || input_hash(&pending.directory.join("audit_stamp.json"))?
                    != pending.capture_sha256
                || input_hash(&pending.directory.join("model_request.json"))?
                    != pending.request_sha256
            {
                return Err("initial Qwen input changed during inference".into());
            }
            self.record(outcome,serde_json::json!({"event":"initial_profile_decision_admitted",
                "model_decision":admitted.decision,"observation":admitted.stamp,
                "image_age_ms":admitted.image_to_decision_ms,"service_elapsed_ms":admitted.service_elapsed_ms,
                "actual_integrations_before_admission":0,"original_policy_calls_before_admission":0,
                "target_localization_claim":false,"release_authorized":false}))?;
            match admitted.decision.request {
                ProfileStartRequest::BeginFixedProfile {} => {}
                ProfileStartRequest::Observe {} => return Err("initial Qwen requested more observation; no policy or physical continuation admitted".into()),
                ProfileStartRequest::Stop {} => return Err(format!("initial Qwen requested an explicit stop: {}",admitted.decision.reason)),
            }
            self.initial_admitted = true;
            self.initial_pending = None;
            self.session = None;
            runtime.requested = false;
            return Ok(true);
        }
        if !runtime.requested {
            port.request_physics_frame(runtime.episode_id, 0)?;
            runtime.requested = true;
            return Ok(false);
        }
        let Some(frame) = port.take() else {
            return Ok(false);
        };
        let frame = frame?;
        if frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.mount_profile != G1CameraMountProfile::ArenaEgo
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.source_ticks != [0; 2]
            || frame.stamp.sim_time_ns != 0
        {
            return Err("initial Qwen RGB is not the actual original camera at zero Tick".into());
        }
        let rgb = task_minigame::decision::CameraRgb::from_rgb(
            "initial_arena_ego",
            frame.width,
            frame.height,
            frame.rgb,
        )
        .map_err(|e| e.to_string())?;
        let directory = runtime.output.join("initial_qwen");
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        fs::write(directory.join("ego.png"), rgb.png()).map_err(|e| e.to_string())?;
        fs::write(
            directory.join("audit_stamp.json"),
            serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let snapshot =
            crate::g1_decision_diagnostic::snapshot_from_capture(&frame.stamp, rgb.png().to_vec())?;
        let mut session = DecisionSession::new(
            runtime.episode_id,
            self.configuration.initial_goal(),
            SkillAvailability {
                mobile_box: true,
                ..Default::default()
            },
            DecisionLimits {
                max_frame_age_ms: 20_000,
                max_sim_age_ns: 1_000_000_000,
                target_ttl_ms: 20_000,
            },
        )
        .map_err(|e| e.to_string())?;
        let input = session
            .prepare_profile_start(snapshot, unix_ms()?, 0)
            .map_err(|e| e.to_string())?;
        let body = self
            .configuration
            .client()?
            .profile_start_request_body(&input)
            .map_err(|e| e.to_string())?;
        fs::write(
            directory.join("model_request.json"),
            serde_json::to_vec_pretty(&body).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        self.worker
            .try_submit_profile_start(input.clone())
            .map_err(|e| e.to_string())?;
        self.initial_pending = Some(PendingInitial {
            stamp: input.context.observation.stamp,
            directory: directory.clone(),
            image_sha256: input_hash(&directory.join("ego.png"))?,
            capture_sha256: input_hash(&directory.join("audit_stamp.json"))?,
            request_sha256: input_hash(&directory.join("model_request.json"))?,
        });
        self.session = Some(session);
        let pending = self.initial_pending.as_ref().unwrap();
        self.record(outcome,serde_json::json!({"event":"initial_profile_request_submitted","observation":pending.stamp,
            "actual_integrations_before_request":0,"original_policy_calls_before_request":0,
            "image_sha256":pending.image_sha256,"capture_sha256":pending.capture_sha256,"request_sha256":pending.request_sha256,
            "target_localization_claim":false,"world_or_contact_truth_input":false}))?;
        Ok(false)
    }

    fn start_transport(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        observation: ObservationStamp,
        localization: serde_json::Value,
    ) -> Result<(), String> {
        let latest = runtime.latest.as_ref().ok_or("Qwen RGB lost owner")?;
        if latest.phase != G1WorkerPhase::Paused
            || latest.episode_id != observation.episode_id
            || latest.timing.episode_integrations * 20_000_000 != observation.sim_time_ns
        {
            return Err("Qwen RGB geometry is not the current paused native boundary".into());
        }
        require_marker_roles(&localization)?;
        let directory = runtime.output.join("visual_approach");
        let stamp: G1CaptureStamp = serde_json::from_slice(
            &fs::read(directory.join("audit_stamp.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let snapshot = crate::g1_decision_diagnostic::snapshot_from_capture(
            &stamp,
            fs::read(directory.join("ego.png")).map_err(|e| e.to_string())?,
        )?;
        if snapshot.stamp.episode_id != observation.episode_id
            || snapshot.stamp.sim_time_ns != observation.sim_time_ns
            || snapshot.stamp.captured_at_unix_ms != observation.captured_at_unix_ms
            || stamp.capture_sequence != observation.frame_id
        {
            return Err("Qwen render-frame identity and control capture identity differ".into());
        }
        let qwen_stamp = snapshot.stamp;
        // These capabilities are confined to this explicitly unqualified test;
        // the interactive lab continues to advertise Observe/Stop only.
        let mut session = DecisionSession::new(
            runtime.episode_id,
            self.configuration.goal(),
            SkillAvailability {
                mobile_box: true,
                ..Default::default()
            },
            DecisionLimits {
                max_frame_age_ms: 20_000,
                max_sim_age_ns: 1_000_000_000,
                target_ttl_ms: 20_000,
            },
        )
        .map_err(|e| e.to_string())?;
        let image_sha256 = input_hash(&directory.join("ego.png"))?;
        let input_sha256 = input_hash(&directory.join("observation.json"))?;
        let selection_evidence = if self.configuration.uses_selection() {
            let evidence = current_rgb_targets(
                &snapshot,
                &localization,
                &image_sha256,
                &input_sha256,
                observation,
            )?;
            let input = session
                .prepare_selection(
                    snapshot,
                    evidence.clone(),
                    unix_ms()?,
                    observation.sim_time_ns,
                )
                .map_err(|e| e.to_string())?;
            self.worker
                .try_submit_selection(input)
                .map_err(|e| e.to_string())?;
            Some(evidence)
        } else {
            let input = session
                .prepare(snapshot, unix_ms()?, observation.sim_time_ns)
                .map_err(|e| e.to_string())?;
            self.worker.try_submit(input).map_err(|e| e.to_string())?;
            None
        };
        self.pending = Some(PendingTransport {
            observation,
            qwen_stamp,
            localization,
            directory: directory.clone(),
            image_sha256,
            stamp_sha256: input_hash(&directory.join("audit_stamp.json"))?,
            input_sha256,
            selection_evidence,
            started: Instant::now(),
        });
        self.session = Some(session);
        let pending = self.pending.as_ref().unwrap();
        self.record(outcome,serde_json::json!({"event":"transport_request_submitted","qwen_observation":qwen_stamp,"control_observation":observation,
            "actual_rgb_only":true,"image_sha256":pending.image_sha256,"capture_stamp_sha256":pending.stamp_sha256,"geometry_input_sha256":pending.input_sha256,
            "rgb_detector_evidence":pending.selection_evidence}))
    }

    fn poll_transport(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .as_ref()
            .ok_or("Qwen transport lost owner")?
            .clone();
        let pending = self
            .pending
            .as_ref()
            .ok_or("Qwen pending transport absent")?;
        if latest.phase != G1WorkerPhase::Paused
            || latest.episode_id != pending.observation.episode_id
            || latest.timing.episode_integrations * 20_000_000 != pending.observation.sim_time_ns
        {
            return Err("Qwen result cannot replace a moving or reset physical boundary".into());
        }
        let session = self.session.as_mut().ok_or("Qwen session absent")?;
        if session.expire_pending(unix_ms()?, pending.observation.sim_time_ns) {
            return Err(
                "Qwen transport image expired; native owner remains explicitly paused".into(),
            );
        }
        let (raw, decision) = if self.configuration.uses_selection() {
            let Some(reply) = self
                .worker
                .try_recv_selection()
                .map_err(|e| e.to_string())?
            else {
                return Ok(false);
            };
            let raw = reply
                .result
                .as_ref()
                .map(|d| serde_json::to_value(d).unwrap())
                .unwrap_or_else(|e| serde_json::json!({"error":e.to_string()}));
            let decision = session
                .accept_selection(reply, unix_ms()?, pending.observation.sim_time_ns)
                .map_err(|e| e.to_string())?;
            (raw, decision)
        } else {
            let Some(reply) = self.worker.try_recv().map_err(|e| e.to_string())? else {
                return Ok(false);
            };
            let raw = reply
                .result
                .as_ref()
                .map(|d| serde_json::to_value(d).unwrap())
                .unwrap_or_else(|e| serde_json::json!({"error":e.to_string()}));
            let decision = session
                .accept(reply, unix_ms()?, pending.observation.sim_time_ns)
                .map_err(|e| e.to_string())?;
            (raw, decision)
        };
        self.record(outcome,serde_json::json!({"event":"transport_decision_admitted","model_decision":raw,"qwen_observation":decision.stamp,
            "image_age_ms":decision.image_to_decision_ms,"service_elapsed_ms":decision.service_elapsed_ms}))?;
        supported_request(&decision)?;
        let pending = self.pending.take().unwrap();
        if decision.stamp != pending.qwen_stamp
            || input_hash(&pending.directory.join("ego.png"))? != pending.image_sha256
            || input_hash(&pending.directory.join("audit_stamp.json"))? != pending.stamp_sha256
            || input_hash(&pending.directory.join("observation.json"))? != pending.input_sha256
        {
            return Err("Qwen dispatch inputs changed while local inference was pending".into());
        }
        // The owner stayed paused during Qwen; preserve its existing20s TTL.
        mobile_held_feedback::check(
            runtime,
            outcome,
            &pending.localization,
            "before_coarse_carry",
            20_000,
        )?;
        let proposal = &pending.localization["navigation_proposal"];
        let distance = proposal["relative_distance_m"]
            .as_f64()
            .ok_or("Qwen RGB coarse distance absent")?
            - 0.65;
        if !(0.1..=1.85).contains(&distance) {
            return Err("Qwen RGB coarse distance outside original bounds".into());
        }
        let goal = MobileCarryGoal {
            observation: pending.observation,
            heading_yaw_source_rad: proposal["heading_yaw_source_rad"]
                .as_f64()
                .ok_or("Qwen RGB heading absent")? as f32,
            relative_distance_m: distance as f32,
        };
        goal.validate().map_err(|e| e.to_string())?;
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("Qwen transport lost sole mobile owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: goal.observation.sim_time_ns + 41_000_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(42),
                command: MobileAssistCommand::ClassicalCarry(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        outcome
            .0
            .lock()
            .unwrap()
            .mobile_assist_handoff
            .as_mut()
            .unwrap()["visual_approach"] = serde_json::json!({
            "actual_localization":pending.localization,"executed_coarse_goal":goal,"reobservation_margin_m":0.65,
            "localization_and_qwen_wall_ms":pending.started.elapsed().as_secs_f64()*1000.,"physics_paused_during_localization":true,
            "same_owner_boundary_tick":latest.timing.episode_integrations,"object_truth_in_command":false,
            "qwen_target_selection":true,"qwen_decision_id":decision.decision_id,"task_qualified":false,
        });
        self.executed_decision = Some(decision);
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.visual_goal_submitted = true;
        assist.submitted_carry_observation = Some(goal.observation);
        runtime.requested = false;
        Ok(false)
    }

    pub(super) fn transport(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        port: &G1CameraPort,
    ) -> Result<bool, String> {
        if self.executed_decision.is_some() {
            return Err("single Qwen transport already admitted".into());
        }
        if self.pending.is_some() {
            return self.poll_transport(runtime, outcome);
        }
        if let Some(job) = runtime
            .mobile_assist
            .as_ref()
            .and_then(|a| a.vision_job.as_ref())
        {
            let Some(reply) = job.try_take() else {
                return Ok(false);
            };
            let observation = job.observation;
            let localization = reply?;
            mobile_held_feedback::check(
                runtime,
                outcome,
                &localization,
                "before_qwen_request",
                2000,
            )?;
            self.start_transport(runtime, outcome, observation, localization)?;
            runtime.mobile_assist.as_mut().unwrap().vision_job.take();
        } else {
            start_marker_job(runtime, port, "visual_approach")?;
        }
        Ok(false)
    }

    pub(super) fn record_failure(
        &self,
        outcome: &CaptureOutcome,
        reason: &str,
    ) -> Result<(), String> {
        self.record(outcome,serde_json::json!({"event":"dispatch_failed","reason":reason,"additional_executor_actions":0,"explicit_pause_required":true}))
    }

    pub(super) fn final_feedback(
        &mut self,
        runtime: &CaptureRuntime,
        outcome: &CaptureOutcome,
    ) -> Result<bool, String> {
        if self.completed {
            return Ok(true);
        }
        let latest = runtime
            .latest
            .as_ref()
            .ok_or("Qwen final feedback lost owner")?;
        let decision = self
            .executed_decision
            .as_ref()
            .ok_or("Qwen transport was not executed")?;
        let sim_time_ns = latest.timing.episode_integrations * 20_000_000;
        if latest.phase != G1WorkerPhase::Paused || latest.episode_id != decision.stamp.episode_id {
            return Err("Qwen final feedback requires its same paused episode".into());
        }
        if !self.feedback_pending {
            let stamp: G1CaptureStamp = serde_json::from_slice(
                &fs::read(runtime.output.join("ego_stamp.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let snapshot = crate::g1_decision_diagnostic::snapshot_from_capture(
                &stamp,
                fs::read(runtime.output.join("ego_640x480.png")).map_err(|e| e.to_string())?,
            )?;
            if snapshot.stamp.episode_id != latest.episode_id
                || snapshot.stamp.sim_time_ns != sim_time_ns
            {
                return Err("Qwen final RGB mismatches completed native owner".into());
            }
            let feedback=ExecutionFeedback {episode_id:latest.episode_id,decision_id:decision.decision_id,status:FeedbackStatus::Running,
                summary:"搬运和开手动作已结束。请根据当前图像复核可见关系；尚未给出物理验收结果，不能仅因动作结束宣称任务成功。诊断执行预算已用完，现在只能观察或停止。".into(),sim_time_ns};
            let session = self.session.as_mut().unwrap();
            session
                .record_feedback(feedback.clone())
                .map_err(|e| e.to_string())?;
            session.revoke_physical_skills();
            if self.configuration.uses_selection() {
                let targets = RgbVerifiedTargets::bind_current_rgb(&snapshot, Vec::new())
                    .map_err(|e| e.to_string())?;
                let input = session
                    .prepare_selection(snapshot, targets, unix_ms()?, sim_time_ns)
                    .map_err(|e| e.to_string())?;
                self.worker
                    .try_submit_selection(input)
                    .map_err(|e| e.to_string())?;
            } else {
                let input = session
                    .prepare(snapshot, unix_ms()?, sim_time_ns)
                    .map_err(|e| e.to_string())?;
                self.worker.try_submit(input).map_err(|e| e.to_string())?;
            }
            self.feedback_pending = true;
            self.record(outcome,serde_json::json!({"event":"final_feedback_submitted","executor_feedback":feedback,"physical_capabilities_revoked":true}))?;
        }
        let session = self.session.as_mut().unwrap();
        if session.expire_pending(unix_ms()?, sim_time_ns) {
            return Err("Qwen final observation expired; explicit pause retained".into());
        }
        let (raw, admitted) = if self.configuration.uses_selection() {
            let Some(reply) = self
                .worker
                .try_recv_selection()
                .map_err(|e| e.to_string())?
            else {
                return Ok(false);
            };
            let raw = reply
                .result
                .as_ref()
                .map(|d| serde_json::to_value(d).unwrap())
                .unwrap_or_else(|e| serde_json::json!({"error":e.to_string()}));
            let admitted = session
                .accept_selection(reply, unix_ms()?, sim_time_ns)
                .map_err(|e| e.to_string())?;
            (raw, admitted)
        } else {
            let Some(reply) = self.worker.try_recv().map_err(|e| e.to_string())? else {
                return Ok(false);
            };
            let raw = reply
                .result
                .as_ref()
                .map(|d| serde_json::to_value(d).unwrap())
                .unwrap_or_else(|e| serde_json::json!({"error":e.to_string()}));
            let admitted = session
                .accept(reply, unix_ms()?, sim_time_ns)
                .map_err(|e| e.to_string())?;
            (raw, admitted)
        };
        if !matches!(
            admitted.request,
            SkillRequest::Observe | SkillRequest::Stop { .. }
        ) {
            return Err("Qwen final feedback restored withdrawn physical skills".into());
        }
        self.record(outcome,serde_json::json!({"event":"final_feedback_admitted","model_decision":raw,"observation":admitted.stamp,
            "image_age_ms":admitted.image_to_decision_ms,"service_elapsed_ms":admitted.service_elapsed_ms,"additional_executor_actions":0}))?;
        self.completed = true;
        Ok(true)
    }
}

#[cfg(all(test, feature = "g1_constraint_diagnostic"))]
mod tests {
    use super::*;
    fn decision(request: SkillRequest) -> ValidatedDecision {
        ValidatedDecision {
            decision_id: 1,
            stamp: ObservationStamp {
                episode_id: 7,
                frame_id: 3,
                sim_time_ns: 0,
                captured_at_unix_ms: 1,
            },
            request,
            reason: "test".into(),
            image_to_decision_ms: 0,
            service_elapsed_ms: 0,
        }
    }
    #[test]
    fn dispatch_binds_only_the_explicit_mobile_public_roles() {
        let request = SkillRequest::ExecuteTask {
            task_profile: TaskProfile::MobileBox,
            target_id: "box_marker_22".into(),
            destination_id: "bin_marker_21".into(),
        };
        assert!(supported_request(&decision(request.clone())).is_ok());
        for (profile, target, destination) in [
            (TaskProfile::StaticApple, "box_marker_22", "bin_marker_21"),
            (TaskProfile::MobileBox, "foreign_box", "bin_marker_21"),
            (TaskProfile::MobileBox, "box_marker_22", "foreign_bin"),
        ] {
            assert!(
                supported_request(&decision(SkillRequest::ExecuteTask {
                    task_profile: profile,
                    target_id: target.into(),
                    destination_id: destination.into()
                }))
                .is_err()
            );
        }
        assert!(supported_request(&decision(SkillRequest::Observe)).is_err());
        assert!(
            supported_request(&decision(SkillRequest::Stop {
                reason: "unsupported".into()
            }))
            .is_err()
        );
    }

    #[test]
    fn selector_binds_detector_capture_sequence_to_the_distinct_render_frame() {
        let control = ObservationStamp {
            episode_id: 7,
            frame_id: 3,
            sim_time_ns: 20_000_000,
            captured_at_unix_ms: 1000,
        };
        let snapshot = ObservationSnapshot {
            stamp: ObservationStamp {
                frame_id: 19,
                ..control
            },
            camera: task_minigame::decision::CameraRgb::from_rgb(
                "aux",
                64,
                48,
                vec![127; 64 * 48 * 3],
            )
            .unwrap(),
            robot: task_minigame::types::RobotSelfState {
                joint_positions: vec![0.0; 43],
                joint_velocities: vec![0.0; 43],
                base_velocity_mps: [0.0; 3],
                projected_gravity: [0.0, 0.0, -1.0],
            },
        };
        let reply = serde_json::json!({
            "detections":[
                {"kind":"carried_box","marker_id":22,"corners_px":[[8,12],[20,12],[20,24],[8,24]],"minimum_edge_px":12,"reprojection_rms_px":0.3},
                {"kind":"target_bin","marker_id":21,"corners_px":[[32,8],[44,8],[44,20],[32,20]],"minimum_edge_px":12,"reprojection_rms_px":0.1}],
            "world_or_contact_truth_input":false,"task_qualified":false,"camera_mount_profile":"auxiliary_grip_overview",
            "actual_rgb_only_object_measurement":true,"image_sha256":"actual_raw_png_hash","input_sha256":"actual_input_hash","observation":control });
        let targets = current_rgb_targets(
            &snapshot,
            &reply,
            "actual_raw_png_hash",
            "actual_input_hash",
            control,
        )
        .unwrap();
        assert_eq!(targets.stamp().frame_id, 19);
        assert_eq!(targets.targets().len(), 2);
        for field in ["image_sha256", "input_sha256"] {
            let mut foreign = reply.clone();
            foreign[field] = "foreign".into();
            assert!(
                current_rgb_targets(
                    &snapshot,
                    &foreign,
                    "actual_raw_png_hash",
                    "actual_input_hash",
                    control
                )
                .is_err()
            );
        }
        for field in ["world_or_contact_truth_input", "task_qualified"] {
            let mut foreign = reply.clone();
            foreign[field] = true.into();
            assert!(
                current_rgb_targets(
                    &snapshot,
                    &foreign,
                    "actual_raw_png_hash",
                    "actual_input_hash",
                    control
                )
                .is_err()
            );
        }
        let mut foreign = reply.clone();
        foreign["observation"]["frame_id"] = 19.into();
        assert!(
            current_rgb_targets(
                &snapshot,
                &foreign,
                "actual_raw_png_hash",
                "actual_input_hash",
                control
            )
            .is_err()
        );
        let mut foreign = reply;
        foreign["detections"][0]["reprojection_rms_px"] = 1.1.into();
        assert!(
            current_rgb_targets(
                &snapshot,
                &foreign,
                "actual_raw_png_hash",
                "actual_input_hash",
                control
            )
            .is_err()
        );
    }
    #[test]
    fn dispatch_requires_both_unique_current_geometry_roles_and_provenance() {
        let valid = serde_json::json!({"detections":[{"kind":"carried_box","marker_id":22},{"kind":"target_bin","marker_id":21}],"world_or_contact_truth_input":false,"task_qualified":false,"camera_mount_profile":"auxiliary_grip_overview"});
        assert!(require_marker_roles(&valid).is_ok());
        for profile in ["auxiliary_bin_placement", "auxiliary_bin_survey"] {
            let mut declared = valid.clone();
            declared["camera_mount_profile"] = profile.into();
            assert!(require_marker_roles(&declared).is_ok());
        }
        for profile in ["arena_ego", "static_placement_overview", "invented_sensor"] {
            let mut foreign = valid.clone();
            foreign["camera_mount_profile"] = profile.into();
            assert!(require_marker_roles(&foreign).is_err());
        }
        let mut absent = valid.clone();
        absent["detections"].as_array_mut().unwrap().pop();
        assert!(require_marker_roles(&absent).is_err());
        let mut duplicate = valid.clone();
        duplicate["detections"]
            .as_array_mut()
            .unwrap()
            .push(valid["detections"][0].clone());
        assert!(require_marker_roles(&duplicate).is_err());
        for key in ["world_or_contact_truth_input", "task_qualified"] {
            let mut changed = valid.clone();
            changed[key] = true.into();
            assert!(require_marker_roles(&changed).is_err());
        }
    }
}
