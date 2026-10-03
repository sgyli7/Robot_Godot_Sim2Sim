//! Paused T1 diagnostic: two original unmarked VLA chunks, actual current RGB,
//! and a distinct classical left-palm transfer in the same physical owner.
use super::*;
use rapier3d::na::{Quaternion, UnitQuaternion, Vector3};
use simulation_minigame::g1::{
    static_startup::StaticStartupCommand,
    static_transfer::{StaticLeftPalmKinematics, StaticVisualTransferGoal},
    worker::TimedCommand,
};

pub(super) struct StaticTransferRoute {
    kinematics: StaticLeftPalmKinematics,
    submitted_image: bool,
    submitted_transfer: bool,
}
impl StaticTransferRoute {
    pub(super) fn new(path: &Path, hash: &str) -> Result<Self, String> {
        let definition = robot_minigame::g1::definition::G1Definition::load(path, hash)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            kinematics: StaticLeftPalmKinematics::new(&definition).map_err(|e| e.to_string())?,
            submitted_image: false,
            submitted_transfer: false,
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
            &fs::read(runtime.options.output.join("live_reply_0002.json"))
                .map_err(|e| e.to_string())?,
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
        fs::write(runtime.options.output.join("static_transfer_current_geometry_preflight.json"),serde_json::to_vec_pretty(&serde_json::json!({"goal":goal,"computed_geometry_points":receipts.len(),"receipts":receipts,"physics_integrations":0,"world_or_contact_truth_input":false,"task_qualified":false})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let CaptureWorker::StaticStartup(worker) = &runtime.worker else {
            return Err("static visual transfer selected foreign physical owner".into());
        };
        worker
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: 390 * 20_000_000,
                valid_until_wall: Instant::now() + Duration::from_secs(7),
                command: StaticStartupCommand::VisualTransfer(goal.clone()),
            })
            .map_err(|e| e.to_string())?;
        self.submitted_transfer = true;
        outcome.0.lock().unwrap().static_visual_grasp_handoff = Some(
            serde_json::json!({"phase":"current140Tick_RGB_then_classical_left_palm_transfer","goal":goal,"geometric_correction_executed":true,"requested_classical_ticks":250,"original_vla_calls":2,"whole_path_preflight_outside_physical_tick":true,"world_or_contact_truth_input":false,"task_qualified":false}),
        );
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
