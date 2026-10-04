//! Partial failure evidence must survive an application runner or cleanup wait.

use super::*;
use rendering_minigame::g1_camera::G1CaptureProgress;

pub(super) fn record(
    output: &Path,
    episode_id: u64,
    receipt: &mut G1CaptureReceipt,
    reason: String,
    progress: G1CaptureProgress,
) -> Result<(), String> {
    receipt.failure_reason = Some(reason);
    receipt.final_camera_progress = Some(progress);
    receipt.capture_succeeded = false;
    receipt.task_qualified = false;
    if episode_id == 0 {
        return Err("failure checkpoint lacks the current owner episode".into());
    }
    let checkpoint = serde_json::json!({
        "schema":"g1_capture_failure_checkpoint_v1",
        "episode_id":episode_id,
        "event":"failure_recorded_before_AppExit_signal",
        "capture_completed":false,"app_exit_completed":false,
        "physics_pause_and_shutdown_proven":false,"final_capture_receipt_written":false,
        "recorded_counters_may_be_partial":true,"task_qualified":false,
        "camera_progress_has_current_episode":receipt.final_camera_progress.as_ref().is_some_and(|p| p.episode_id == episode_id),
        "partial_capture_receipt":receipt,
    });
    let bytes = serde_json::to_vec_pretty(&checkpoint).map_err(|e| e.to_string())?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err("partial capture checkpoint exceeds its finite byte budget".into());
    }
    let path = output.join("capture_failure_checkpoint.json");
    if let Ok(metadata) = fs::symlink_metadata(&path) {
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() > 4 * 1024 * 1024
        {
            return Err("existing capture failure checkpoint is not an owned finite file".into());
        }
        let first: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if first["schema"] != checkpoint["schema"]
            || first["partial_capture_receipt"]["scope"]
                != checkpoint["partial_capture_receipt"]["scope"]
            || first["partial_capture_receipt"]["definition_sha256"]
                != checkpoint["partial_capture_receipt"]["definition_sha256"]
            || first["episode_id"] != checkpoint["episode_id"]
        {
            return Err(
                "capture failure checkpoint already belongs to another source/episode".into(),
            );
        }
        // Preserve the first error; a later cleanup error cannot replace it.
        return Ok(());
    }
    let temporary = output.join(".capture_failure_checkpoint.tmp");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    // Install a complete file without overwriting an earlier checkpoint.
    fs::hard_link(&temporary, &path).map_err(|e| e.to_string())?;
    fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    fs::File::open(output)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);

    #[derive(Resource)]
    struct Directory(PathBuf);

    fn receipt() -> G1CaptureReceipt {
        let config: G1RunnerConfig = serde_json::from_value(serde_json::json!({
            "episode_id":1,"definition":"not-loaded.json","definition_sha256":"1".repeat(64),
            "ort_library":"not-loaded.so","ort_sha256":"2".repeat(64),"stand_model":"stand.onnx","walk_model":"walk.onnx",
            "root_pose":{"position":[0.,0.,0.78],"rotation_wxyz":[1.,0.,0.,0.]},
            "robot_contact_friction":0.5,"floor_contact_friction":1.0
        })).unwrap();
        let mut receipt =
            G1CaptureReceipt::initial(3300, &CaptureRunnerConfig::Mobile(config), "fixture");
        receipt.scope = "CPU_failure_signal_fixture_not_native_execution";
        receipt
    }

    fn signal_failure(
        output: Res<Directory>,
        outcome: Res<CaptureOutcome>,
        port: Res<G1CameraPort>,
        mut exit: MessageWriter<AppExit>,
    ) {
        record(
            &output.0,
            port.episode_id(),
            &mut outcome.0.lock().unwrap(),
            "marker localization failed: exit status: 1".into(),
            port.capture_progress(),
        )
        .unwrap();
        exit.write(AppExit::error());
    }

    fn directory() -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "g1_terminal_failure_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        directory
    }

    #[test]
    fn error_signal_preserves_evidence_without_waiting_for_app_runner_or_cleanup() {
        let directory = directory();
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        let outcome = CaptureOutcome(Arc::new(Mutex::new(receipt())));
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_message::<AppExit>()
            .insert_resource(Directory(directory.clone()))
            .insert_resource(port)
            .insert_resource(outcome)
            .add_systems(Update, signal_failure);
        app.update();
        assert!(matches!(app.should_exit(), Some(AppExit::Error(_))));
        assert!(
            directory.join("capture_failure_checkpoint.json").is_file(),
            "error evidence only lives in memory before app.run returns"
        );
        assert!(
            !directory.join("capture_receipt.json").exists(),
            "partial failure must not fabricate the final receipt"
        );
        let checkpoint: serde_json::Value = serde_json::from_slice(
            &fs::read(directory.join("capture_failure_checkpoint.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(checkpoint["app_exit_completed"], false);
        assert_eq!(checkpoint["physics_pause_and_shutdown_proven"], false);
        assert_eq!(
            checkpoint["partial_capture_receipt"]["failure_reason"],
            "marker localization failed: exit status: 1"
        );
        drop(app);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn first_failure_is_preserved_and_cannot_be_reused_by_a_reset_episode() {
        let directory = directory();
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        let mut receipt = receipt();
        receipt.capture_succeeded = true;
        receipt.task_qualified = true;
        record(
            &directory,
            port.episode_id(),
            &mut receipt,
            "first".into(),
            port.capture_progress(),
        )
        .unwrap();
        let path = directory.join("capture_failure_checkpoint.json");
        let bytes = fs::read(&path).unwrap();
        record(
            &directory,
            port.episode_id(),
            &mut receipt,
            "later cleanup error".into(),
            port.capture_progress(),
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(!receipt.capture_succeeded && !receipt.task_qualified);
        port.reset(2).unwrap();
        assert!(
            record(
                &directory,
                port.episode_id(),
                &mut receipt,
                "reset".into(),
                port.capture_progress()
            )
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn failed_write_and_foreign_checkpoint_cannot_fabricate_a_final_receipt() {
        let directory = directory();
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        let mut receipt = receipt();
        assert!(
            record(
                &directory.join("missing"),
                port.episode_id(),
                &mut receipt,
                "filesystem error".into(),
                port.capture_progress()
            )
            .is_err()
        );
        fs::write(
            directory.join("capture_failure_checkpoint.json"),
            b"foreign",
        )
        .unwrap();
        assert!(
            record(
                &directory,
                port.episode_id(),
                &mut receipt,
                "worker error".into(),
                port.capture_progress()
            )
            .is_err()
        );
        assert_eq!(
            fs::read(directory.join("capture_failure_checkpoint.json")).unwrap(),
            b"foreign"
        );
        assert!(!directory.join("capture_receipt.json").exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "saved verified partial count report/real failed RGB fixture; CPU only, not native execution"]
    fn recorded_timeout_counts_survive_the_pre_exit_checkpoint() -> Result<(), String> {
        use sha2::{Digest, Sha256};
        let path = std::env::var("G1_FAILURE_CHECKPOINT_FIXTURE").map_err(|e| e.to_string())?;
        let f: serde_json::Value =
            serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let load = |key: &str| -> Result<serde_json::Value, String> {
            serde_json::from_slice(
                &fs::read(f[key].as_str().ok_or("fixture path absent")?)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())
        };
        let report = load("verified_count_report")?;
        let last = load("last_owner_row")?;
        let observation = load("observation")?;
        let rejection = load("worker_rejection")?;
        if format!(
            "{:x}",
            Sha256::digest(
                fs::read(f["verified_count_report"].as_str().unwrap()).map_err(|e| e.to_string())?
            )
        ) != f["verified_count_report_sha256"].as_str().unwrap()
            || report["partial_trace_verified"] != true
            || report["evidence_verified"] != false
            || report["recorded_integrations"] != 1971
            || report["strict_task_passed"] != false
            || report["trace_sha256"]
                != "f7c9c209c6d25371b4544fef1dafc78fb30012e67bc9d7e1fc18a2ccc59e5a80"
            || last["episode_id"] != 20708
            || last["owner_episode_integrations"] != 1971
            || format!(
                "{:x}",
                Sha256::digest(
                    fs::read(f["last_owner_row"].as_str().ok_or("last row path absent")?)
                        .map_err(|e| e.to_string())?
                )
            ) != f["last_owner_row_sha256"]
                .as_str()
                .ok_or("last row hash absent")?
            || rejection["expected_rejection"] != "marker localization failed: exit status: 1"
        {
            return Err("saved partial count/rejection proof identity differs".into());
        }
        let cfg = load("configuration")?;
        let runner: CaptureRunnerConfig =
            serde_json::from_value(cfg["runner"].clone()).map_err(|e| e.to_string())?;
        let mut receipt = G1CaptureReceipt::initial(
            3300,
            &runner,
            cfg["visual_sha256"].as_str().ok_or("visual hash absent")?,
        );
        receipt.scope = "CPU_recorded_timeout_fixture_replay_not_native_execution";
        receipt.actual_integrations = 1971;
        receipt.actual_torque_updates = last["owner_total_torque_updates"]
            .as_u64()
            .ok_or("torque count absent")?;
        receipt.actual_model_attempts = last["owner_total_model_attempts"]
            .as_u64()
            .ok_or("body model count absent")?;
        receipt.actual_model_successes = last["owner_total_model_successes"]
            .as_u64()
            .ok_or("body model successes absent")?;
        receipt.active_sim_seconds = last["active_sim_seconds"]
            .as_f64()
            .ok_or("active sim absent")?;
        receipt.active_wall_seconds = last["active_wall_seconds"]
            .as_f64()
            .ok_or("active wall absent")?;
        receipt.pending_ticks = last["pending_ticks"].as_u64().ok_or("pending absent")?;
        receipt.control_deadlines_missed = last["control_deadlines_missed"]
            .as_u64()
            .ok_or("deadlines absent")?;
        receipt.owner_step_records = 1971;
        receipt.owner_step_trace_complete = false;
        receipt.live_policy_inference_calls = 4;
        receipt.live_policy_successes = 4;
        let stamp: ObservationStamp =
            serde_json::from_value(observation["stamp"].clone()).map_err(|e| e.to_string())?;
        if stamp.episode_id != 20708 || stamp.sim_time_ns != 1971 * 20_000_000 {
            return Err("saved error RGB differs from recorded count boundary".into());
        }
        let progress = G1CaptureProgress {
            episode_id: stamp.episode_id,
            capture_sequence: stamp.frame_id,
            phase: "CPU_recorded_error_fixture_not_live_camera".into(),
            minimum_physics_tick: Some(1971),
            captured_at_unix_ms: Some(stamp.captured_at_unix_ms),
            ..default()
        };
        let directory = PathBuf::from(f["output"].as_str().ok_or("output absent")?);
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        record(
            &directory,
            stamp.episode_id,
            &mut receipt,
            rejection["expected_rejection"].as_str().unwrap().into(),
            progress,
        )?;
        if directory.join("capture_receipt.json").exists() {
            return Err("CPU replay fabricated final application receipt".into());
        }
        let result: serde_json::Value = serde_json::from_slice(
            &fs::read(directory.join("capture_failure_checkpoint.json"))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if result["partial_capture_receipt"]["actual_integrations"] != 1971
            || result["app_exit_completed"] != false
        {
            return Err("partial counters or exit proof changed".into());
        }
        Ok(())
    }
}
