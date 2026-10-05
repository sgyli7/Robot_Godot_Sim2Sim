//! Paused T1 diagnostic: two original unmarked VLA chunks, actual current RGB,
//! and a distinct classical left-palm transfer in the same physical owner.
use super::*;
use rapier3d::na::{Matrix4, Quaternion, UnitQuaternion, Vector3};
use sha2::{Digest, Sha256};
use simulation_minigame::g1::{
    static_observe::{StaticObservationWithdrawal, StaticObservationWithdrawalGoal},
    static_place::{StaticMemoryPlace, StaticMemoryPlaceGoal, StaticPlacementGeometry},
    static_startup::StaticStartupCommand,
    static_transfer::{StaticLeftPalmKinematics, StaticVisualTransferGoal},
    task_objects::TaskObjectSceneConfig,
    worker::TimedCommand,
};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) struct StaticTransferRoute {
    kinematics: StaticLeftPalmKinematics,
    submitted_image: bool,
    submitted_transfer: bool,
    placement_geometry: Option<StaticPlacementGeometry>,
    submitted_placement: bool,
    memory_image_requested: bool,
    submitted_withdrawal: bool,
    transfer_goal: Option<StaticVisualTransferGoal>,
    grip_image_submitted: bool,
    grip_verified: bool,
}
impl StaticTransferRoute {
    pub(super) fn new(
        path: &Path,
        hash: &str,
        placement: Option<&TaskObjectSceneConfig>,
    ) -> Result<Self, String> {
        let definition = robot_minigame::g1::definition::G1Definition::load(path, hash)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            kinematics: StaticLeftPalmKinematics::new(&definition).map_err(|e| e.to_string())?,
            submitted_image: false,
            submitted_transfer: false,
            placement_geometry: placement
                .map(StaticPlacementGeometry::load)
                .transpose()
                .map_err(|e| e.to_string())?,
            submitted_placement: false,
            memory_image_requested: false,
            submitted_withdrawal: false,
            transfer_goal: None,
            grip_image_submitted: false,
            grip_verified: false,
        })
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
            .ok_or("static transfer owner snapshot absent")?;
        let tick = latest.timing.episode_integrations;
        if latest.phase == G1WorkerPhase::Failed {
            return Err(latest
                .reason
                .clone()
                .unwrap_or("static transfer owner failed".into()));
        }
        if tick < 140
            || runtime
                .live_policy
                .as_ref()
                .is_none_or(|p| p.submitted_chunks < 2)
        {
            drive_live_policy(runtime, outcome, port)?;
            return Ok(false);
        }
        if self.submitted_transfer {
            if runtime.static_regrasp_started {
                let completed = drive_live_policy(runtime, outcome, port)?;
                return Ok(completed && tick == 230 && latest.phase == G1WorkerPhase::Paused);
            }
            if (runtime.static_memory_observe || runtime.static_unheld_regrasp)
                && !self.grip_verified
            {
                return self.drive_grip_verification(runtime, outcome, port);
            }
            if runtime.static_memory_place {
                return self.drive_placement(runtime, outcome, port);
            }
            return Ok(tick == 390 && latest.phase == G1WorkerPhase::Paused);
        }
        if tick != 140 || latest.phase != G1WorkerPhase::Paused {
            return Err("static current perception requires completed140Tick pause".into());
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
        if !self.submitted_image {
            if !runtime.requested {
                port.request_physics_frame(runtime.episode_id, 140)?;
                runtime.requested = true;
                return Ok(false);
            }
            if let Some(frame) = port.take() {
                let frame = frame?;
                if frame.stamp.source_ticks != [140, 140]
                    || frame.stamp.episode_id != runtime.episode_id
                {
                    return Err("static handoff RGB has stale or foreign identity".into());
                }
                runtime
                    .static_marker_worker
                    .as_mut()
                    .unwrap()
                    .submit_capture(&frame)?;
                fs::write(
                    runtime
                        .options
                        .output
                        .join("static_transfer_grasp_stamp.json"),
                    serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                self.submitted_image = true;
                runtime.requested = false;
            }
            return Ok(false);
        }
        let Some(localized) = outcome.0.lock().unwrap().static_marker_localization.clone() else {
            return Ok(false);
        };
        let result = &localized["localization"];
        let observation: ObservationStamp =
            serde_json::from_value(result["observation"].clone()).map_err(|e| e.to_string())?;
        let targets = result["detections"]
            .as_array()
            .ok_or("static targets absent")?;
        let object = |kind: &str| -> Result<[f64; 3], String> {
            let d = targets
                .iter()
                .find(|d| d["object_kind"] == kind)
                .ok_or("static current target absent")?;
            let matrix: [[f64; 4]; 4] =
                serde_json::from_value(d["root_from_object"].clone()).map_err(|e| e.to_string())?;
            Ok([matrix[0][3], matrix[1][3], matrix[2][3]])
        };
        let goal = StaticVisualTransferGoal {
            observation,
            apple_root_source_m: object("t1_apple")?,
            plate_root_source_m: object("t1_plate")?,
        };
        goal.validate().map_err(|e| e.to_string())?;
        let state = latest
            .measurement
            .as_ref()
            .ok_or("static self snapshot absent")?;
        if observation.episode_id != state.episode_id
            || observation.sim_time_ns != state.sim_time_ns
        {
            return Err("static visual goal and current self time disagree".into());
        }
        let chunk: task_minigame::policy::PolicyActionChunk = serde_json::from_slice(
            &fs::read(runtime.output.join("live_reply_0002.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if chunk.profile != TaskProfile::StaticApple
            || chunk.sequence_id != 2
            || chunk.frames.len() != 40
            || chunk.observation.episode_id != runtime.episode_id
        {
            return Err("static preflight original command identity changed".into());
        }
        let frame = chunk.frames.last().unwrap();
        let mut upper = [0.; 28];
        upper[..7].copy_from_slice(&frame.left_arm);
        upper[7..14].copy_from_slice(&frame.left_hand);
        upper[14..21].copy_from_slice(&frame.right_arm);
        upper[21..].copy_from_slice(&frame.right_hand);
        let mut command = AgileCommand {
            navigation: frame.navigate_mps_rps,
            pelvis_height: frame.base_height_m,
            upper_positions: upper,
        };
        let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
        let rotation = UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z));
        let mut delta = rotation
            * (Vector3::from(goal.plate_root_source_m) - Vector3::from(goal.apple_root_source_m));
        delta.z = 0.;
        let mut receipts = Vec::new();
        for n in 0..250 {
            let source = if n < 50 {
                Vector3::new(0., 0., 0.001)
            } else {
                delta / 200.
            };
            let (next, receipt) = self
                .kinematics
                .translate(state, &command, (rotation.inverse() * source).into())
                .map_err(|e| e.to_string())?;
            command = next;
            receipts.push(receipt);
        }
        fs::write(runtime.output.join("static_transfer_current_geometry_preflight.json"),serde_json::to_vec_pretty(&serde_json::json!({"goal":goal,"computed_geometry_points":receipts.len(),"receipts":receipts,"physics_integrations":0,"world_or_contact_truth_input":false,"task_qualified":false})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("static visual transfer selected foreign physical owner".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: if runtime.static_memory_observe
                    || runtime.static_unheld_regrasp
                {
                    190
                } else {
                    390
                } * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(7),
                command: StaticStartupCommand::VisualTransfer(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        self.submitted_transfer = true;
        self.transfer_goal = Some(goal.clone());
        outcome.0.lock().unwrap().static_visual_grasp_handoff = Some(
            serde_json::json!({"phase":"current140Tick_RGB_then_classical_left_palm_transfer","goal":goal,"geometric_correction_executed":true,"requested_classical_ticks":250,"original_vla_calls":2,"whole_path_preflight_outside_physical_tick":true,"world_or_contact_truth_input":false,"task_qualified":false}),
        );
        Ok(false)
    }
    fn drive_grip_verification(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        port: &G1CameraPort,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .clone()
            .ok_or("grip verification self absent")?;
        let tick = latest.timing.episode_integrations;
        if tick < 190 {
            return Ok(false);
        }
        if tick != 190 || latest.phase != G1WorkerPhase::Paused {
            return Err("grip verification requires the completed50Tick lift pause".into());
        }
        if !self.grip_image_submitted {
            if !runtime.requested {
                port.request_physics_frame(runtime.episode_id, 190)?;
                runtime.requested = true;
                return Ok(false);
            }
            if let Some(frame) = port.take() {
                let frame = frame?;
                if frame.stamp.source_ticks != [190, 190]
                    || frame.stamp.episode_id != runtime.episode_id
                {
                    return Err("grip verification RGB is stale or from another episode".into());
                }
                let measured = latest
                    .measurement
                    .as_ref()
                    .ok_or("grip current self absent")?;
                let native = frame
                    .stamp
                    .native_state
                    .as_ref()
                    .ok_or("grip camera self absent")?;
                if native.measured_joints.positions != measured.joint_positions
                    || native.measured_joints.velocities != measured.joint_velocities
                    || native.measured_joints.root_rotation_wxyz != measured.root_rotation_wxyz
                    || native.measured_joints.root_velocity_source != measured.root_velocity_source
                {
                    return Err("grip image and owner joint positions disagree".into());
                }
                let worker = runtime
                    .static_marker_worker
                    .as_mut()
                    .ok_or("grip CPU worker absent")?;
                if runtime.static_unheld_regrasp {
                    worker.submit_unverified_grip_capture(&frame)?;
                } else {
                    worker.submit_grip_capture(&frame)?;
                }
                fs::write(
                    runtime.output.join("static_grip_stamp.json"),
                    serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                runtime.requested = false;
                self.grip_image_submitted = true;
            }
            return Ok(false);
        }
        let (origin, current) = {
            let receipt = outcome.0.lock().unwrap();
            let Some(current) = receipt.static_grip_localization.clone() else {
                return Ok(false);
            };
            (
                receipt
                    .static_marker_localization
                    .clone()
                    .ok_or("grip origin RGB absent")?,
                current,
            )
        };
        let samples = &runtime
            .startup_evidence
            .as_ref()
            .ok_or("grip self stream absent")?
            .static_self_samples;
        let now = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_millis(),
        )
        .map_err(|e| e.to_string())?;
        let apple_visible = current["localization"]["detections"]
            .as_array()
            .is_some_and(|d| d.iter().any(|d| d["marker_id"] == 31));
        let verification = if !apple_visible && runtime.static_unheld_regrasp {
            serde_json::json!({"schema":"g1_static_unverified_lift_absence_v1","accepted":false,
                "observation":current["localization"]["observation"],"reason":"actual lift RGB lacks apple31",
                "world_or_contact_truth_input":false,"task_qualified":false})
        } else {
            super::static_grip_check::verify(
                &origin["localization"],
                &current["localization"],
                samples,
                &self.kinematics,
                now,
            )?
        };
        fs::write(
            runtime.output.join("static_grip_verification.json"),
            serde_json::to_vec_pretty(&verification).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if verification["accepted"] != true {
            if runtime.static_unheld_regrasp {
                let live = runtime
                    .live_policy
                    .as_mut()
                    .ok_or("regrasp policy absent")?;
                if live.submitted_chunks != 2 || live.pending.is_some() || live.max_calls != 3 {
                    return Err("regrasp requires the completed two-chunk boundary".into());
                }
                live.next_boundary_tick = 190;
                runtime.static_regrasp_started = true;
                return Ok(false);
            }
            return Err("actual RGB lift did not verify apple following the palm; horizontal transfer blocked".into());
        }
        if runtime.static_unheld_regrasp {
            return Err(
                "finite regrasp probe requires an unverified lift; verified grasp is not retried"
                    .into(),
            );
        }
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("grip verification selected a foreign physical owner".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: 390 * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(6),
                command: StaticStartupCommand::VisualTransfer(
                    self.transfer_goal
                        .clone()
                        .ok_or("immutable transfer goal absent")?,
                ),
            })
            .map_err(|e| e.to_string())?;
        self.grip_verified = true;
        Ok(false)
    }
    fn drive_placement(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        port: &G1CameraPort,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .clone()
            .ok_or("static placement owner snapshot absent")?;
        let tick = latest.timing.episode_integrations;
        if self.submitted_placement {
            if runtime.static_memory_observe {
                return self.drive_observation_withdrawal(runtime, outcome);
            }
            return Ok(tick == 715 && latest.phase == G1WorkerPhase::Paused);
        }
        if tick < 390 {
            return Ok(false);
        }
        if tick != 390 || latest.phase != G1WorkerPhase::Paused {
            return Err("static placement requires390Tick pause".into());
        }
        if runtime
            .static_auxiliary_activation_frame
            .is_none_or(|n| runtime.render_frames < n + 2)
        {
            return Ok(false);
        }
        if !self.memory_image_requested {
            port.request_physics_frame(runtime.episode_id, 390)?;
            self.memory_image_requested = true;
            runtime.requested = true;
            return Ok(false);
        }
        let Some(frame) = port.take() else {
            return Ok(false);
        };
        let frame = frame?;
        if frame.stamp.source_ticks != [390, 390]
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.mount_profile != G1CameraMountProfile::StaticPlacementOverview
        {
            return Err("placement current RGB identity/mount mismatch".into());
        }
        let current_rgb = task_minigame::decision::CameraRgb::from_rgb(
            "native_ego",
            frame.width,
            frame.height,
            frame.rgb,
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            runtime.output.join("static_memory_current_rgb.png"),
            current_rgb.png(),
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            runtime
                .options
                .output
                .join("static_memory_current_stamp.json"),
            serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        runtime.requested = false;
        let current_observation = ObservationStamp {
            episode_id: frame.stamp.episode_id,
            frame_id: frame.stamp.capture_sequence,
            sim_time_ns: frame.stamp.sim_time_ns,
            captured_at_unix_ms: frame.stamp.captured_at_unix_ms,
        };
        let localized = outcome
            .0
            .lock()
            .unwrap()
            .static_marker_localization
            .clone()
            .ok_or("placement origin RGB absent")?;
        let origin = &localized["localization"];
        let origin_observation: ObservationStamp =
            serde_json::from_value(origin["observation"].clone()).map_err(|e| e.to_string())?;
        let samples = &runtime
            .startup_evidence
            .as_ref()
            .ok_or("placement self stream absent")?
            .static_self_samples;
        if samples.len() != 251
            || samples.iter().enumerate().any(|(i, s)| {
                s.episode_id != runtime.episode_id
                    || s.source_tick != 140 + i as u64
                    || s.sim_time_ns != s.source_tick * 20_000_000
                    || s.root_velocity_source.iter().any(|v| !v.is_finite())
            })
        {
            return Err("placement self stream has gap/reset/nonfinite velocity".into());
        }
        let first = &samples[0];
        let state = latest
            .measurement
            .as_ref()
            .ok_or("placement current self absent")?;
        if origin_observation.episode_id != first.episode_id
            || origin_observation.sim_time_ns != first.sim_time_ns
            || current_observation.episode_id != state.episode_id
            || current_observation.sim_time_ns != state.sim_time_ns
        {
            return Err("static RGB memory and self stream identity mismatch".into());
        }
        let native = frame
            .stamp
            .native_state
            .as_ref()
            .ok_or("placement actual RGB self state absent")?;
        native.validate()?;
        let sensor = &native.measured_joints;
        if sensor.positions != state.joint_positions
            || sensor.velocities != state.joint_velocities
            || sensor.root_rotation_wxyz != state.root_rotation_wxyz
            || sensor.root_angular_velocity_body != state.root_angular_velocity_body
            || sensor.root_velocity_source != state.root_velocity_source
        {
            return Err("placement actual image self sensors differ from owner boundary".into());
        }
        if serde_json::to_value(state.as_ref()).map_err(|e| e.to_string())?
            != serde_json::to_value(&samples[250]).map_err(|e| e.to_string())?
        {
            return Err("placement current self differs from completed self stream".into());
        }
        let object = |kind: &str| -> Result<Matrix4<f64>, String> {
            let m: [[f64; 4]; 4] = serde_json::from_value(
                origin["detections"]
                    .as_array()
                    .ok_or("origin detections absent")?
                    .iter()
                    .find(|d| d["object_kind"] == kind)
                    .ok_or("origin object absent")?["root_from_object"]
                    .clone(),
            )
            .map_err(|e| e.to_string())?;
            Ok(Matrix4::from_fn(|i, j| m[i][j]))
        };
        let apple0 = object("t1_apple")?;
        let plate0 = object("t1_plate")?;
        let palm0 = self
            .kinematics
            .measured_left_palm(first)
            .map_err(|e| e.to_string())?;
        let palm1 = self
            .kinematics
            .measured_left_palm(state)
            .map_err(|e| e.to_string())?;
        let apple = palm1.to_homogeneous() * palm0.inverse().to_homogeneous() * apple0;
        let self_rotation=|s:&simulation_minigame::g1::runner::G1Measurement|->Result<UnitQuaternion<f64>,String>{
            let [w,x,y,z]=s.root_rotation_wxyz.map(f64::from);
            if ![w,x,y,z].iter().all(|v|v.is_finite()) || (w*w+x*x+y*y+z*z-1.).abs()>2e-5 {return Err("invalid placement self orientation".into());}
            Ok(UnitQuaternion::new_normalize(Quaternion::new(w,x,y,z)))
        };
        let r0 = self_rotation(first)?.to_rotation_matrix().into_inner();
        let r1 = self_rotation(state)?.to_rotation_matrix().into_inner();
        let odom = samples[1..].iter().fold(Vector3::<f64>::zeros(), |a, s| {
            a + Vector3::from(s.root_velocity_source.map(f64::from)) * 0.02
        });
        let mut plate = Matrix4::identity();
        plate
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(r1.transpose() * r0 * plate0.fixed_view::<3, 3>(0, 0)));
        plate
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&(r1.transpose() * (r0 * plate0.fixed_view::<3, 1>(0, 3) - odom)));
        let input = serde_json::json!({"schema":"g1_static_live_rgb_and_self_grasp_memory_input_v1","origin_localization":origin,"current_observation":current_observation,"current_image_sha256":format!("{:x}",Sha256::digest(current_rgb.png())),"self_measurements":samples,
            "static_target_assumption":true,"rigid_grasp_assumption":true,"world_or_contact_truth_input":false});
        let bytes = serde_json::to_vec_pretty(&input).map_err(|e| e.to_string())?;
        let input_hash = format!("{:x}", Sha256::digest(&bytes));
        fs::write(
            runtime
                .options
                .output
                .join("static_memory_self_and_rgb_input.json"),
            bytes,
        )
        .map_err(|e| e.to_string())?;
        let geometry_hash = self
            .placement_geometry
            .as_ref()
            .ok_or("static public geometry absent")?
            .definition_sha256()
            .to_owned();
        let goal = StaticMemoryPlaceGoal {
            origin_observation,
            current_observation,
            root_from_apple_estimate: std::array::from_fn(|i| {
                std::array::from_fn(|j| apple[(i, j)])
            }),
            root_from_plate_estimate: std::array::from_fn(|i| {
                std::array::from_fn(|j| plate[(i, j)])
            }),
            memory_input_sha256: input_hash,
            public_geometry_sha256: geometry_hash,
            rigid_grasp_assumption: true,
            static_target_assumption: true,
            current_object_visual_detections: false,
        };
        goal.validate().map_err(|e| e.to_string())?;
        let step = latest
            .startup_step
            .as_ref()
            .ok_or("static completed transfer step absent")?;
        let simulation_minigame::g1::static_startup::StaticStartupExecution::VisualTransfer(step) =
            &step.execution
        else {
            return Err("placement has no typed completed transfer".into());
        };
        if !step.completed {
            return Err("static transfer incomplete".into());
        }
        let mut command = step.command.clone();
        let plan = StaticMemoryPlace::new(
            goal.clone(),
            state,
            command.clone(),
            self.placement_geometry
                .as_ref()
                .ok_or("static public geometry absent")?,
        )
        .map_err(|e| e.to_string())?;
        let delta = Vector3::from(plan.source_displacement_m());
        let rotation = self_rotation(state)?;
        let mut receipts = Vec::new();
        for n in 0..150 {
            let offset = if n < 100 {
                delta / 100.
            } else {
                Vector3::new(0., 0., 0.04 / 50.)
            };
            let (next, receipt) = self
                .kinematics
                .translate(state, &command, (rotation.inverse() * offset).into())
                .map_err(|e| e.to_string())?;
            command = next;
            receipts.push(receipt);
        }
        fs::write(runtime.output.join("static_memory_placement_preflight.json"),serde_json::to_vec_pretty(&serde_json::json!({"goal":goal,"source_displacement_m":delta.as_slice(),"computed_geometry_points":150,"receipts":receipts,"physics_integrations":0,"current_object_visual_detections":false,"world_or_contact_truth_input":false,"task_qualified":false})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("static placement selected foreign owner".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: 715 * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(9),
                command: StaticStartupCommand::MemoryPlace(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        self.submitted_placement = true;
        outcome.0.lock().unwrap().static_visual_grasp_handoff = Some(
            serde_json::json!({"phase":"current140Tick_RGB_transfer_then_disclosed390Tick_grasp_memory_place","goal":goal,
            "geometric_correction_executed":true,"requested_classical_transfer_ticks":250,"requested_classical_place_ticks":325,"original_vla_calls":2,"whole_path_preflight_outside_physical_tick":true,"world_or_contact_truth_input":false,"task_qualified":false}),
        );
        Ok(false)
    }
    fn drive_observation_withdrawal(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .clone()
            .ok_or("observation withdrawal owner absent")?;
        let tick = latest.timing.episode_integrations;
        if self.submitted_withdrawal {
            return Ok(tick == 840 && latest.phase == G1WorkerPhase::Paused);
        }
        if tick < 715 {
            return Ok(false);
        }
        if tick != 715 || latest.phase != G1WorkerPhase::Paused {
            return Err("observation withdrawal requires715Tick pause".into());
        }
        let state = latest
            .measurement
            .as_ref()
            .ok_or("observation withdrawal self absent")?;
        let step = latest
            .startup_step
            .as_ref()
            .ok_or("observation withdrawal placement step absent")?;
        let simulation_minigame::g1::static_startup::StaticStartupExecution::MemoryPlace(place) =
            &step.execution
        else {
            return Err("observation withdrawal requires typed placement".into());
        };
        if !place.completed {
            return Err("observation withdrawal placement incomplete".into());
        }
        let goal = StaticObservationWithdrawalGoal {
            episode_id: runtime.episode_id,
            start_sim_time_ns: 715 * 20_000_000,
        };
        StaticObservationWithdrawal::new(goal.clone(), state, place.command.clone())
            .map_err(|e| e.to_string())?;
        let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
        let rotation = UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z));
        let mut command = place.command.clone();
        let mut receipts = Vec::new();
        for _ in 0..100 {
            let (next, receipt) = self
                .kinematics
                .translate(
                    state,
                    &command,
                    (rotation.inverse() * Vector3::new(0., 0.001, 0.)).into(),
                )
                .map_err(|e| e.to_string())?;
            command = next;
            receipts.push(receipt);
        }
        fs::write(runtime.output.join("static_observation_withdrawal_preflight.json"),serde_json::to_vec_pretty(&serde_json::json!({"goal":goal,"source_displacement_m":[0.,0.1,0.],"computed_geometry_points":100,"receipts":receipts,"physics_integrations":0,"world_or_contact_truth_input":false,"task_qualified":false})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("observation withdrawal selected foreign owner".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: 840 * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(4),
                command: StaticStartupCommand::ObservationWithdrawal(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        self.submitted_withdrawal = true;
        if let Some(handoff) = &mut outcome.0.lock().unwrap().static_visual_grasp_handoff {
            handoff["observation_withdrawal"] = serde_json::json!({"goal":goal,"requested_classical_ticks":125,"source_displacement_m":[0.,0.1,0.],"purpose":"recover actual final target view with unchanged camera","open_hand_self_gate_passed":true,"whole_path_preflight_outside_physical_tick":true,"world_or_contact_truth_input":false,"task_qualified":false});
        }
        Ok(false)
    }
}
pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let mut route = runtime
        .static_transfer_route
        .take()
        .ok_or("static transfer route absent")?;
    let result = route.drive(runtime, outcome, port);
    runtime.static_transfer_route = Some(route);
    result
}
