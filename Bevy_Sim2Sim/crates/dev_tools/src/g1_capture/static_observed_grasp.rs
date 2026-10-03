//! Current preclosure RGB -> disclosed reference geometry -> finite native grasp.
//! Preparation and full joint preflight run outside the physical owner clock.
use super::*;
use rapier3d::na::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use robot_minigame::g1::{
    definition::{G1Definition, SourcePose},
    policy::bound_bytes,
};
use simulation_minigame::g1::{
    static_grasp::{StaticGraspIncrement, StaticObservedGrasp, StaticObservedGraspGoal},
    static_startup::StaticStartupCommand,
    static_transfer::StaticLeftPalmKinematics,
    worker::TimedCommand,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    schema: String,
    definition_sha256: String,
    reference_observation: ObservationStamp,
    reference_image_sha256: String,
    reference_self_input_sha256: String,
    original_hand_model_revision: String,
    palm_from_apple_source: Vec<SourcePose>,
    original_left_hand_targets: Vec<[f32; 7]>,
}
pub(super) struct StaticObservedGraspRoute {
    kinematics: StaticLeftPalmKinematics,
    reference: Reference,
    reference_sha256: String,
    image_submitted: bool,
    target_tick: Option<u64>,
    completed: bool,
}
impl StaticObservedGraspRoute {
    pub(super) fn new(
        definition: &Path,
        hash: &str,
        template: &BackgroundVisualConfiguration,
    ) -> Result<Self, String> {
        let bytes = bound_bytes(&template.path, &template.sha256).map_err(|e| e.to_string())?;
        if bytes.len() > 128 * 1024 {
            return Err("observed grasp template exceeds bound".into());
        }
        let reference: Reference = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let is_hash = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        };
        if reference.schema != "g1_static_observed_rgb_self_grasp_template_v1"
            || reference.definition_sha256 != hash
            || !is_hash(&reference.reference_image_sha256)
            || !is_hash(&reference.reference_self_input_sha256)
            || reference.reference_observation.episode_id == 0
            || reference.reference_observation.frame_id == 0
            || reference.reference_observation.sim_time_ns != 100 * 20_000_000
            || reference.original_hand_model_revision
                != profile_contract(TaskProfile::StaticApple).revision
            || reference.palm_from_apple_source.len() != 41
            || reference.original_left_hand_targets.len() != 40
            || reference
                .original_left_hand_targets
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("foreign observed grasp reference identity".into());
        }
        for p in &reference.palm_from_apple_source {
            if !p
                .position
                .iter()
                .chain(p.rotation_wxyz.iter())
                .all(|v| v.is_finite())
                || Vector3::from(p.position).norm() > 0.3
                || (p.rotation_wxyz.iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 2e-5
            {
                return Err("unbounded observed grasp reference pose".into());
            }
        }
        Ok(Self {
            kinematics: StaticLeftPalmKinematics::new(
                &G1Definition::load(definition, hash).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
            reference,
            reference_sha256: template.sha256.clone(),
            image_submitted: false,
            target_tick: None,
            completed: false,
        })
    }
    pub(super) fn completed(&self) -> bool {
        self.completed
    }
    fn drive(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        port: &G1CameraPort,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .clone()
            .ok_or("observed grasp self snapshot absent")?;
        let tick = latest.timing.episode_integrations;
        if latest.phase == G1WorkerPhase::Failed {
            return Err(latest
                .reason
                .clone()
                .unwrap_or("observed grasp owner failed".into()));
        }
        if let Some(target) = self.target_tick {
            self.completed = tick == target && latest.phase == G1WorkerPhase::Paused;
            return Ok(self.completed);
        }
        if tick < 100 {
            drive_live_policy(runtime, outcome, port)?;
            return Ok(false);
        }
        if tick != 100
            || latest.phase != G1WorkerPhase::Paused
            || runtime
                .live_policy
                .as_ref()
                .is_none_or(|p| p.submitted_chunks != 1)
        {
            return Err("observed grasp requires one complete original100Tick boundary".into());
        }
        if runtime
            .static_marker_activation_render_frame
            .is_none_or(|n| runtime.render_frames < n + 2)
            || runtime
                .static_marker_worker
                .as_ref()
                .is_none_or(|w| !w.ready())
        {
            return Ok(false);
        }
        if !self.image_submitted {
            if !runtime.requested {
                port.request_physics_frame(runtime.episode_id, 100)?;
                runtime.requested = true;
                return Ok(false);
            }
            if let Some(frame) = port.take() {
                let frame = frame?;
                if frame.stamp.source_ticks != [100, 100]
                    || frame.stamp.episode_id != runtime.episode_id
                    || frame.stamp.mount_profile != G1CameraMountProfile::ArenaEgo
                {
                    return Err("observed grasp RGB is stale or foreign".into());
                }
                let state = latest
                    .measurement
                    .as_ref()
                    .ok_or("observed grasp owner sensors absent")?;
                let sensors = &frame
                    .stamp
                    .native_state
                    .as_ref()
                    .ok_or("observed grasp camera sensors absent")?
                    .measured_joints;
                if sensors.positions != state.joint_positions
                    || sensors.velocities != state.joint_velocities
                    || sensors.root_rotation_wxyz != state.root_rotation_wxyz
                    || sensors.root_velocity_source != state.root_velocity_source
                {
                    return Err("observed grasp camera and owner self state disagree".into());
                }
                runtime
                    .static_marker_worker
                    .as_mut()
                    .unwrap()
                    .submit_capture(&frame)?;
                fs::write(
                    runtime.options.output.join("static_preclosure_stamp.json"),
                    serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                runtime.requested = false;
                self.image_submitted = true;
            }
            return Ok(false);
        }
        let Some(localized) = outcome.0.lock().unwrap().static_marker_localization.clone() else {
            return Ok(false);
        };
        let result = &localized["localization"];
        let observation: ObservationStamp =
            serde_json::from_value(result["observation"].clone()).map_err(|e| e.to_string())?;
        let state = latest
            .measurement
            .as_ref()
            .ok_or("observed grasp current sensors absent")?;
        if observation.episode_id != state.episode_id
            || observation.sim_time_ns != state.sim_time_ns
        {
            return Err("observed grasp RGB/self time mismatch".into());
        }
        let apple = result["detections"]
            .as_array()
            .ok_or("observed grasp detections absent")?
            .iter()
            .find(|d| d["object_kind"] == "t1_apple")
            .ok_or("current observed grasp apple absent")?;
        let matrix: [[f64; 4]; 4] =
            serde_json::from_value(apple["root_from_object"].clone()).map_err(|e| e.to_string())?;
        let apple_root = Vector3::new(matrix[0][3], matrix[1][3], matrix[2][3]);
        if !apple_root.iter().all(|v| v.is_finite()) || apple_root.norm() > 1. {
            return Err("unbounded current observed apple position".into());
        }
        let reply: task_minigame::policy::PolicyActionChunk = serde_json::from_slice(
            &fs::read(runtime.options.output.join("live_reply_0001.json"))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if reply.profile != TaskProfile::StaticApple
            || reply.sequence_id != 1
            || reply.frames.len() != 40
            || reply.observation.episode_id != runtime.episode_id
            || reply.observation.sim_time_ns != 60 * 20_000_000
        {
            return Err("observed grasp preceding original command is foreign".into());
        }
        let frame = reply.frames.last().unwrap();
        let command = AgileCommand {
            navigation: frame.navigate_mps_rps,
            pelvis_height: frame.base_height_m,
            upper_positions: std::array::from_fn(|i| match i {
                0..=6 => frame.left_arm[i],
                7..=13 => frame.left_hand[i - 7],
                14..=20 => frame.right_arm[i - 14],
                _ => frame.right_hand[i - 21],
            }),
        };
        let rotation = root_rotation(state).map_err(|e| e.to_string())?;
        let palm = self
            .kinematics
            .commanded_left_palm(state, &command)
            .map_err(|e| e.to_string())?;
        let mut previous = Isometry3::from_parts(
            Translation3::from(rotation * palm.translation.vector),
            rotation * palm.rotation,
        );
        let anchor = rotation * apple_root;
        let mut hand: [f32; 7] = command.upper_positions[7..14].try_into().unwrap();
        let mut increments = Vec::new();
        let mut alignment_ticks = 0;
        for (i, p) in self.reference.palm_from_apple_source.iter().enumerate() {
            let [w, x, y, z] = p.rotation_wxyz;
            let next = Isometry3::from_parts(
                Translation3::from(Vector3::from(p.position) + anchor),
                UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)),
            );
            let next_hand = if i == 0 {
                hand
            } else {
                self.reference.original_left_hand_targets[i - 1]
            };
            let delta = next.translation.vector - previous.translation.vector;
            let angle = (next.rotation * previous.rotation.inverse()).angle();
            let hand_delta = hand
                .iter()
                .zip(next_hand.iter())
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0., f64::max);
            let count = 1usize
                .max((delta.norm() / 0.0016).ceil() as usize)
                .max((angle / 0.01).ceil() as usize)
                .max((hand_delta / 0.024).ceil() as usize);
            if i == 0 {
                alignment_ticks = count;
            }
            if increments.len() + count + 100 > 300 {
                return Err("observed grasp geometry exceeds its400Tick total budget".into());
            }
            let start = previous;
            let start_hand = hand;
            for j in 1..=count {
                let alpha = j as f64 / count as f64;
                let pose = Isometry3::from_parts(
                    Translation3::from(start.translation.vector + delta * alpha),
                    start.rotation.slerp(&next.rotation, alpha),
                );
                hand = std::array::from_fn(|k| {
                    ((1. - alpha) * f64::from(start_hand[k]) + alpha * f64::from(next_hand[k]))
                        as f32
                });
                increments.push(StaticGraspIncrement {
                    translation_source_m: (pose.translation.vector - previous.translation.vector)
                        .into(),
                    rotation_source_rad: (pose.rotation * previous.rotation.inverse())
                        .scaled_axis()
                        .into(),
                    left_hand_targets: hand,
                });
                previous = pose;
            }
        }
        let acquisition_ticks = increments.len();
        for lift in [true, false] {
            for _ in 0..50 {
                increments.push(StaticGraspIncrement {
                    translation_source_m: if lift { [0., 0., 0.001] } else { [0.; 3] },
                    rotation_source_rad: [0.; 3],
                    left_hand_targets: hand,
                });
            }
        }
        let goal = StaticObservedGraspGoal {
            observation,
            image_sha256: result["image_sha256"]
                .as_str()
                .ok_or("observed grasp RGB hash absent")?
                .into(),
            reference_rgb_self_template_sha256: self.reference_sha256.clone(),
            increments,
        };
        let mut preflight =
            StaticObservedGrasp::new(goal.clone(), state, command).map_err(|e| e.to_string())?;
        let mut predicted = state.as_ref().clone();
        let mut receipts = Vec::new();
        for _ in 0..goal.increments.len() {
            receipts.push(
                preflight
                    .update(&predicted, &self.kinematics)
                    .map_err(|e| e.to_string())?,
            );
            predicted.source_tick += 1;
            predicted.sim_time_ns += 20_000_000;
        }
        let target = 100 + goal.increments.len() as u64;
        let maximum = if runtime.static_observed_place {
            400
        } else {
            u64::from(runtime.options.ticks)
        };
        if target > maximum {
            return Err("observed grasp path exceeds its finite pre-placement budget".into());
        }
        let report = serde_json::json!({"schema":"g1_live_observed_grasp_joint_preflight_v1","observation":observation,
            "reference_template_sha256":self.reference_sha256,"alignment_ticks":alignment_ticks,
            "acquisition_ticks":acquisition_ticks,"lift_ticks":50,"hold_ticks":50,"target_tick":target,
            "physics_integrations":0,"model_calls":0,"world_or_contact_truth_input":false,"task_qualified":false,"receipts":receipts});
        fs::write(
            runtime
                .options
                .output
                .join("static_observed_grasp_preflight.json"),
            serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            runtime
                .options
                .output
                .join("static_observed_grasp_goal.json"),
            serde_json::to_vec_pretty(&goal).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("observed grasp owner type is foreign".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: target * 20_000_000,
                valid_until_wall: Instant::now()
                    + Duration::from_nanos((target - 100) * 20_000_000)
                    + Duration::from_secs(2),
                command: StaticStartupCommand::ObservedGrasp(goal),
            })
            .map_err(|e| e.to_string())?;
        outcome.0.lock().unwrap().static_visual_grasp_handoff = Some(serde_json::json!({
            "phase":"one_fresh_original_chunk_then_actual100RGB_and_explicit_classical_template_grasp_lift_hold",
            "markers_activated_at_completed_tick":100,"unmarked_original_vla_calls":1,"first_original_action_tick":60,
            "current_observation_required_tick":100,"geometric_correction_executed":true,"reference_template_sha256":self.reference_sha256,
            "target_tick":target,"task_qualified":false}));
        self.target_tick = Some(target);
        Ok(false)
    }
}
pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let mut route = runtime
        .static_observed_grasp_route
        .take()
        .ok_or("observed grasp route absent")?;
    let result = route.drive(runtime, outcome, port);
    runtime.static_observed_grasp_route = Some(route);
    result
}

fn root_rotation(
    state: &simulation_minigame::g1::runner::G1Measurement,
) -> Result<UnitQuaternion<f64>, robot_minigame::RobotError> {
    let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
    if ![w, x, y, z].iter().all(|v| v.is_finite())
        || (w * w + x * x + y * y + z * z - 1.).abs() > 2e-5
    {
        return Err(robot_minigame::RobotError::Contract(
            "invalid observed grasp self orientation".into(),
        ));
    }
    Ok(UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)))
}
