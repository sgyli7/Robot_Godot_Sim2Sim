//! Two finite unchanged-target holds distinguish load settling from continued slip.
//! Their image verdict never overwrites the failed650-to850 pickup verdict.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution},
    mobile_hold::MobileHoldGoal,
    worker::TimedCommand,
};

pub(super) struct Verification {
    origin: serde_json::Value,
    previous: serde_json::Value,
    goal: MobileHoldGoal,
    submitted_at: Instant,
    checks: Vec<serde_json::Value>,
}

fn submit(runtime: &CaptureRuntime, stamp: ObservationStamp) -> Result<Instant, String> {
    let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
        return Err("loaded hold lost sole physical owner".into());
    };
    let submitted_at = Instant::now();
    owner
        .submit(TimedCommand {
            episode_id: runtime.episode_id,
            valid_until_sim_ns: stamp.sim_time_ns + 2_000_000_000,
            valid_until_wall: submitted_at + Duration::from_secs(4),
            command: MobileAssistCommand::ClassicalPickupHold(MobileHoldGoal {
                observation: stamp,
            }),
        })
        .map_err(|e| e.to_string())?;
    Ok(submitted_at)
}

pub(super) fn begin(runtime: &mut CaptureRuntime, origin: serde_json::Value) -> Result<(), String> {
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&origin)?;
    if stamp.episode_id != runtime.episode_id || stamp.sim_time_ns != 17_000_000_000 {
        return Err("loaded hold needs its own actual850Tick postload image".into());
    }
    let submitted_at = submit(runtime, stamp)?;
    let assist = runtime.mobile_assist.as_mut().unwrap();
    assist.regrasp_loaded = Some(Verification {
        previous: origin.clone(),
        origin,
        goal: MobileHoldGoal { observation: stamp },
        submitted_at,
        checks: Vec::new(),
    });
    assist.vision_job = None;
    runtime.requested = false;
    Ok(())
}

fn vector(reply: &serde_json::Value, key: &str) -> Result<[f64; 3], String> {
    let v: [f64; 3] = serde_json::from_value(reply[key].clone()).map_err(|e| e.to_string())?;
    if v.iter().any(|x| !x.is_finite() || x.abs() > 8.) {
        return Err("loaded stability vector is nonfinite".into());
    }
    Ok(v)
}

fn drift(a: &serde_json::Value, b: &serde_json::Value, key: &str) -> Result<f64, String> {
    let a = vector(a, key)?;
    let b = vector(b, key)?;
    Ok(a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt())
}

fn evaluate(
    origin: &serde_json::Value,
    previous: &serde_json::Value,
    current: &serde_json::Value,
    tick: u64,
    now: u64,
) -> Result<serde_json::Value, String> {
    use crate::g1_marker_vision::mobile_held::paired_stamp;
    let a = paired_stamp(origin)?;
    let p = paired_stamp(previous)?;
    let b = paired_stamp(current)?;
    if a.sim_time_ns != 17_000_000_000
        || !matches!(tick, 950 | 1050)
        || a.episode_id != p.episode_id
        || p.episode_id != b.episode_id
        || p.sim_time_ns + 2_000_000_000 != b.sim_time_ns
        || b.sim_time_ns != tick * 20_000_000
        || b.frame_id <= p.frame_id
        || p.frame_id < a.frame_id
        || b.captured_at_unix_ms <= p.captured_at_unix_ms
        || b.captured_at_unix_ms > now
        || now - b.captured_at_unix_ms > 2000
        || now
            .checked_sub(a.captured_at_unix_ms)
            .is_none_or(|age| age > 12_000)
    {
        return Err("loaded stability observation is stale, repeated, reset or wrong phase".into());
    }
    for reply in [origin, previous, current] {
        if reply["geometry_lift_feedback"] != true
            || reply["world_or_contact_truth_input"] != false
            || !reply["navigation_proposal"].is_null()
            || !reply["clearance_proposal"].is_null()
        {
            return Err("loaded stability requires bound current paired RGB only".into());
        }
    }
    let baseline = drift(origin, current, "current_box_relative_left_palm_m")?;
    let interval = drift(previous, current, "current_box_relative_left_palm_m")?;
    let palm = drift(origin, current, "current_midpalm_gravity_center_m")?;
    let bilateral = super::mobile_regrasp_lift::bilateral_candidate(current);
    Ok(
        serde_json::json!({"schema":"g1_current_paired_rgb_postload_stationary_stability_v1",
        "origin_observation":a,"previous_observation":p,"current_observation":b,
        "postload_relative_displacement_m":baseline,"interval_relative_displacement_m":interval,
        "midpalm_gravity_displacement_m":palm,"bilateral_wholehand_geometry_candidate":bilateral,
        "maximum_relative_displacement_m":0.02,"accepted":baseline<=0.02&&interval<=0.02&&palm<=0.02&&bilateral,
        "preload_failed_verdict_preserved":true,"hand_only_support_proven":false,
        "world_or_contact_truth_input":false,"task_qualified":false}),
    )
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let latest = runtime
        .latest
        .as_ref()
        .ok_or("loaded hold owner absent")?
        .clone();
    if latest.phase != G1WorkerPhase::Paused {
        return Ok(false);
    }
    let tick = latest.timing.episode_integrations;
    let assist = runtime.mobile_assist.as_ref().unwrap();
    let v = assist.regrasp_loaded.as_ref().unwrap();
    let start = v.goal.observation.sim_time_ns / 20_000_000;
    let execution = &latest
        .assist_step
        .as_ref()
        .ok_or("loaded hold execution absent")?
        .execution;
    if tick == start {
        let own = match execution {
            MobileAssistExecution::ClassicalRaise { goal, raising } => {
                start == 850
                    && goal.observation.episode_id == runtime.episode_id
                    && goal.observation.sim_time_ns == 13_000_000_000
                    && raising.completed
                    && raising.raising_ticks == 100
                    && raising.settling_ticks == 100
            }
            MobileAssistExecution::ClassicalHold { goal, holding } => {
                start == 950
                    && goal.observation.episode_id == runtime.episode_id
                    && goal.observation.sim_time_ns == 17_000_000_000
                    && holding.completed
                    && holding.holding_ticks == 100
            }
            _ => false,
        };
        if !own || v.submitted_at.elapsed() >= Duration::from_secs(4) {
            return Err("queued loaded hold lost own predecessor".into());
        }
        return Ok(false);
    }
    if !matches!(execution,MobileAssistExecution::ClassicalHold {goal,holding} if goal==&v.goal && tick==start+100
        && holding.completed && holding.holding_ticks==100 && holding.stable_velocity_ticks>=20)
    {
        return Err("loaded verification exceeded its own100Tick stationary hold".into());
    }
    let directory = format!("regrasp_lift_loaded_{}", v.checks.len() + 1);
    if assist.vision_job.is_none() {
        start_marker_job(runtime, port, &directory)?;
        return Ok(false);
    }
    let job = assist.vision_job.as_ref().unwrap();
    let Some(reply) = job.try_take() else {
        return Ok(false);
    };
    let reply = reply?;
    let stamp = crate::g1_marker_vision::mobile_held::paired_stamp(&reply)?;
    if stamp != job.observation {
        return Err("loaded hold changed actual image identity".into());
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64;
    let verdict = evaluate(&v.origin, &v.previous, &reply, tick, now)?;
    let mut checks = v.checks.clone();
    checks.push(verdict.clone());
    let receipt = serde_json::json!({"mode":"actual_postload_two_stationary_hold_diagnostic",
        "actual_integrations":tick,"checks":checks,"unchanged_original_lifted_posture":true,
        "maximum_holds":2,"maximum_ticks_per_hold":100,"preload_failed_verdict_preserved":true,
        "world_or_contact_truth_input":false,"carry_release_submitted":false,"task_qualified":false});
    fs::write(
        runtime.output.join("regrasp_loaded_stability_receipt.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    outcome
        .0
        .lock()
        .unwrap()
        .mobile_assist_handoff
        .as_mut()
        .unwrap()["loaded_stability"] = receipt;
    if verdict["accepted"] != true {
        return Err(
            "loaded pickup continues to move over original20mm bound; no carry admitted".into(),
        );
    }
    if tick == 1050 {
        let carry_enabled = matches!(&assist.configuration,
            MobileAssistStage::Scan(c) if c.verified_regrasp_pickup);
        if carry_enabled {
            let raise = assist.regrasp_lift.as_ref().unwrap().executed_goal()?;
            let hold = v.goal.clone();
            runtime
                .mobile_assist
                .as_mut()
                .unwrap()
                .pickup
                .accept_loaded_regrasp(raise, hold, reply, &checks)?;
            let assist = runtime.mobile_assist.as_mut().unwrap();
            assist.request_loaded_transport_camera();
            assist.station_hold_submitted = true;
            assist.vision_job = None;
            runtime.requested = false;
            return Ok(false);
        }
        let assist = runtime.mobile_assist.as_mut().unwrap();
        assist.vision_job = None;
        assist.completed = true;
        return Ok(true);
    }
    let submitted_at = submit(runtime, stamp)?;
    let assist = runtime.mobile_assist.as_mut().unwrap();
    let v = assist.regrasp_loaded.as_mut().unwrap();
    v.previous = reply;
    v.goal = MobileHoldGoal { observation: stamp };
    v.submitted_at = submitted_at;
    v.checks = checks;
    assist.vision_job = None;
    runtime.requested = false;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_discards_loaded_admission_and_pending_old_sensor_state() {
        let mut assist = super::super::MobileAssistCaptureRuntime::new(
            MobileAssistStage::Carry(super::super::MobileAssistCaptureConfiguration {
                heading_yaw_source_rad: 0.,
                relative_distance_m: 2.,
            }),
            false,
            false,
            false,
            true,
            true,
            true,
            true,
        );
        assist.request_loaded_transport_camera();
        assist.submitted = true;
        assist.completed = true;
        assist.station_hold_submitted = true;
        assist.station_placement_view_active = true;
        assist.release_submitted = true;
        assist.hold_submitted = true;
        assist.raise_submitted = true;
        assist.fine_alignment_confirmed = true;
        assist.fine_goals_submitted = 3;
        assist.reset_episode();
        assert!(!assist.regrasp_preparation_completed);
        assert!(!assist.submitted && !assist.completed && !assist.station_hold_submitted);
        assert!(!assist.station_placement_view_active && !assist.release_submitted);
        assert!(
            !assist.hold_submitted && !assist.raise_submitted && !assist.fine_alignment_confirmed
        );
        assert_eq!(assist.fine_goals_submitted, 0);
        assert!(
            assist.regrasp.is_none()
                && assist.regrasp_lift.is_none()
                && assist.regrasp_loaded.is_none()
        );
        assert!(assist.pending_box_pair.is_none() && assist.submitted_carry_observation.is_none());
        assert_eq!(
            assist.box_pair_restore_mount,
            Some(G1CameraMountProfile::AuxiliaryGripOverview)
        );
        assert!(
            matches!(assist.configuration, MobileAssistStage::Carry(ref c) if c.relative_distance_m == 2.)
        );
    }

    #[test]
    fn loaded_handoff_requests_render_switch_and_names_the_same_camera() {
        let mut assist = super::super::MobileAssistCaptureRuntime::new(
            MobileAssistStage::Carry(super::super::MobileAssistCaptureConfiguration {
                heading_yaw_source_rad: 0.,
                relative_distance_m: 2.,
            }),
            false,
            false,
            false,
            true,
            true,
            true,
            true,
        );
        assist.box_pair_restore_mount = Some(G1CameraMountProfile::AuxiliaryPregraspOverview);
        assist.box_pair_camera_activation_frame = Some(17);
        assist.request_loaded_transport_camera();
        assert_eq!(
            assist.camera_mount(),
            G1CameraMountProfile::AuxiliaryBinPlacement
        );
        assert_eq!(assist.auxiliary_camera_name(), "auxiliary_bin_placement");
        assert_eq!(assist.box_pair_restore_mount, Some(assist.camera_mount()));
        assert_eq!(assist.box_pair_camera_activation_frame, None);
        // Clearing the acknowledged render request must retain the declared primary.
        assist.box_pair_restore_mount = None;
        assert_eq!(
            assist.camera_mount(),
            G1CameraMountProfile::AuxiliaryBinPlacement
        );
    }

    fn image(tick: u64, frame: u64, wall: u64, x: f64) -> serde_json::Value {
        let s = ObservationStamp {
            episode_id: 1,
            frame_id: frame,
            sim_time_ns: tick * 20_000_000,
            captured_at_unix_ms: wall,
        };
        serde_json::json!({"observation":s,"schema":"g1_mobile_actual_marker_localization_v1",
            "actual_rgb_only_object_measurement":true,"geometry_lift_feedback":true,"world_or_contact_truth_input":false,"task_qualified":false,
            "same_tick_box_pair_pose":{"schema":"g1_actual_rgb_same_tick_box_pair_pose_v1","marker_id":22,
            "observations":[s,ObservationStamp {frame_id:frame+1,captured_at_unix_ms:wall+10,..s}],"world_or_contact_truth_input":false,"task_qualified":false},
            "held_contact_geometry":{"bodies":[{"side":"left","linf_distance_m":0.},{"side":"right","linf_distance_m":0.}]},
            "current_box_relative_left_palm_m":[x,0.,0.],"current_midpalm_gravity_center_m":[0.,0.,0.]})
    }
    #[test]
    fn settling_cannot_erase_accumulated_slip_or_retime_old_images() {
        let a = image(850, 1, 1000, 0.);
        let p = image(950, 3, 3000, 0.015);
        let b = image(1050, 5, 5000, 0.025);
        assert_eq!(evaluate(&a, &p, &b, 1050, 5100).unwrap()["accepted"], false);
        let good = image(1050, 5, 5000, 0.017);
        assert_eq!(
            evaluate(&a, &p, &good, 1050, 5100).unwrap()["accepted"],
            true
        );
        assert!(evaluate(&a, &p, &p, 1050, 5100).is_err());
        let mut reset = good.clone();
        reset["observation"]["episode_id"] = 2.into();
        assert!(evaluate(&a, &p, &reset, 1050, 5100).is_err());
    }

    #[test]
    fn loaded_transport_certificate_requires_both_checks_and_its_final_hold() {
        use robot_minigame::g1::contract::G1Command;
        use simulation_minigame::g1::{mobile_hold::MobileHoldStep, mobile_raise::MobileRaiseGoal};
        let a = image(850, 3, 1000, 0.);
        let p = image(950, 5, 3000, 0.003);
        let b = image(1050, 7, 5000, 0.004);
        let checks = [
            evaluate(&a, &a, &p, 950, 3100).unwrap(),
            evaluate(&a, &p, &b, 1050, 5100).unwrap(),
        ];
        let raise = MobileRaiseGoal {
            observation: ObservationStamp {
                episode_id: 1,
                frame_id: 1,
                sim_time_ns: 13_000_000_000,
                captured_at_unix_ms: 500,
            },
            distance_m: 0.1,
            duration_ticks: 100,
        };
        let hold = MobileHoldGoal {
            observation: crate::g1_marker_vision::mobile_held::paired_stamp(&p).unwrap(),
        };
        let mut pickup = super::super::mobile_pickup::Pickup::default();
        assert!(
            pickup
                .accept_loaded_regrasp(raise.clone(), hold.clone(), b.clone(), &checks[..1])
                .is_err()
        );
        let mut foreign = raise.clone();
        foreign.observation.episode_id = 2;
        assert!(
            pickup
                .accept_loaded_regrasp(foreign, hold.clone(), b.clone(), &checks)
                .is_err()
        );
        pickup
            .accept_loaded_regrasp(raise, hold.clone(), b, &checks)
            .unwrap();
        let mut execution = MobileAssistExecution::ClassicalHold {
            goal: hold,
            holding: MobileHoldStep {
                command: G1Command::default(),
                holding_ticks: 100,
                stable_velocity_ticks: 100,
                self_speed_m_s: 0.,
                completed: true,
            },
        };
        assert_eq!(pickup.completed_hold_ticks(&execution, 1050).unwrap(), 100);
        assert!(pickup.completed_hold_ticks(&execution, 950).is_err());
        if let MobileAssistExecution::ClassicalHold { goal, .. } = &mut execution {
            goal.observation.frame_id += 1;
        }
        assert!(pickup.completed_hold_ticks(&execution, 1050).is_err());
    }
}
