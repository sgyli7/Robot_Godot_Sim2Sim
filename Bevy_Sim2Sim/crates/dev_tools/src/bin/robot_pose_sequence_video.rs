//! Render recorded native poses from a completed diagnostic run; no pose authoring.

use dev_tools_minigame::visual_preview::{
    RobotPoseSequenceCaptureOptions, RobotPoseSequenceCaptureReceipt,
    RobotPoseSequenceCaptureResources, run_robot_pose_sequence_capture,
};
use rendering_minigame::{
    RobotRenderStyle, RobotVisualModel, StationCameraControl, StationScene, StationView,
    robot_mesh::VerifiedRobotAppearance,
};
use robot_minigame::{body_pose::RobotPoseFrame, definition::RobotDefinition};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path, process::Command, sync::Arc};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn existing_capture(
    directory: &Path,
    poses: &[RobotPoseFrame],
    model_sha256: &str,
    output_fps: u32,
) -> Result<Option<RobotPoseSequenceCaptureReceipt>, Box<dyn Error>> {
    let manifest = directory.join("capture_manifest.json");
    if !manifest.exists() {
        return Ok(None);
    }
    let receipt: RobotPoseSequenceCaptureReceipt = serde_json::from_slice(&fs::read(&manifest)?)?;
    let output_dir = fs::canonicalize(directory)?;
    let indices: Vec<usize> = (0..poses.len())
        .map(|index| index * 60 / output_fps as usize)
        .take_while(|index| *index < poses.len())
        .collect();
    if receipt.model_file_sha256 != model_sha256
        || receipt.declared_source_hz != 60
        || receipt.output_fps != output_fps
        || receipt.frames.len() != indices.len()
    {
        return Err("existing GPU capture identity or frame count differs".into());
    }
    for (index, (&source_index, row)) in indices.iter().zip(&receipt.frames).enumerate() {
        let pose = &poses[source_index];
        let expected_png = output_dir.join(format!("frame_{index:06}.png"));
        if row.output_index != index
            || row.source_index != source_index
            || row.episode_id != pose.episode_id
            || row.global_step != pose.global_step
            || row.episode_step != pose.episode_step
            || row.pose_sha256 != digest(&serde_json::to_vec(pose)?)
            || row.png != expected_png
            || fs::metadata(&row.png)?.len() == 0
        {
            return Err(
                format!("existing GPU capture frame {index} differs from source pose").into(),
            );
        }
    }
    Ok(Some(receipt))
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if !(9..=10).contains(&arguments.len()) {
        return Err("usage: robot_pose_sequence_video MODEL MODEL_SHA APPEARANCE APPEARANCE_SHA ASSETS PHYSICS_REPORT OUTPUT_DIR FPS NEW_RECEIPT.json [NEW_VIDEO.mp4]".into());
    }
    let output = Path::new(&arguments[8]);
    if output.exists() {
        return Err("receipt output must be new".into());
    }
    let definition = Arc::new(match Path::new(&arguments[0])
        .extension()
        .and_then(|v| v.to_str())
    {
        Some("json") => RobotDefinition::load_json(Path::new(&arguments[0]), &arguments[1]),
        Some("ron") => RobotDefinition::load_ron(Path::new(&arguments[0]), &arguments[1]),
        _ => return Err("compiled model must be .json or .ron".into()),
    }?);
    let appearance = Arc::new(match Path::new(&arguments[2])
        .extension()
        .and_then(|v| v.to_str())
    {
        Some("json") => {
            VerifiedRobotAppearance::load_json(Path::new(&arguments[2]), &arguments[3], &definition)
        }
        Some("ron") => {
            VerifiedRobotAppearance::load_ron(Path::new(&arguments[2]), &arguments[3], &definition)
        }
        _ => return Err("appearance must be .json or .ron".into()),
    }?);
    let asset_root = fs::canonicalize(&arguments[4])?;
    let scene = StationScene::load(&asset_root)?;
    let report_bytes = fs::read(&arguments[5])?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    if report["schema"] != "station_robot_60hz_diagnostic_v1"
        || report["scope"] != "single_rapier_world_native_ort_each_60hz_step_p_only_diagnostic_only"
        || report["passed"] != true
        || report["world_count"] != 1
        || report["physics_hz"] != 60
        || report["policy_hz"] != 60
        || report["policy_origin"].as_str().is_none()
        || !report["native_runtime"].is_object()
        || report["policy_model_sha256"]
            .as_str()
            .is_none_or(|hash| hash.len() != 64)
        || report["model_file_sha256"] != definition.file_sha256()
        || report["station_identity"]["model_sha256"] != scene.0.model_sha256
        || report["station_identity"]["manifest_sha256"] != scene.0.manifest_sha256
        || report["station_identity"]["layout_sha256"] != scene.0.layout_sha256
    {
        return Err("physics report, compiled robot or station identity mismatch".into());
    }
    let frames: Vec<RobotPoseFrame> = serde_json::from_value(report["pose_frames"].clone())?;
    if frames.len() < 2
        || report["integration_count"].as_u64() != Some((frames.len() - 1) as u64)
        || report["policy_inference_count"].as_u64() != Some((frames.len() - 1) as u64)
    {
        return Err("pose sequence does not match actual policy/physics step counts".into());
    }
    let trace = report["trace"]
        .as_array()
        .ok_or("physics trace is absent")?;
    let raw_poses = report["pose_frames"]
        .as_array()
        .ok_or("source poses are absent")?;
    if trace.len() + 1 != frames.len() {
        return Err("physics trace and pose sequence lengths differ".into());
    }
    for (index, row) in trace.iter().enumerate() {
        let before = raw_poses[index]["poses"]
            .as_array()
            .and_then(|poses| poses.iter().find(|pose| pose["source_body_id"] == 1))
            .ok_or("source root pose is absent")?;
        let after = raw_poses[index + 1]["poses"]
            .as_array()
            .and_then(|poses| poses.iter().find(|pose| pose["source_body_id"] == 1))
            .ok_or("source root pose is absent")?;
        if row["tick_before"].as_u64() != Some(index as u64)
            || row["root_before"] != before["translation"]
            || row["root_after"] != after["translation"]
            || row["observation"]
                .as_array()
                .is_none_or(|items| items.len() != 61)
            || row["action"]
                .as_array()
                .is_none_or(|items| items.len() != 14)
            || row["torque"]
                .as_array()
                .is_none_or(|items| items.len() != 14)
        {
            return Err(format!("physics trace and source pose disagree at tick {index}").into());
        }
    }
    let first_root = frames[0]
        .poses
        .iter()
        .find(|pose| pose.source_body_id == 1)
        .ok_or("initial root pose is absent")?;
    let mut target = first_root.translation;
    target[1] += 0.02;
    let camera = StationCameraControl {
        view: StationView::Follow,
        target: target.into(),
        yaw: 0.7,
        pitch: 0.2,
        distance: 0.8,
        ..Default::default()
    };
    let model = RobotVisualModel::new(definition, appearance)?
        .with_render_style(RobotRenderStyle::default())?;
    let output_fps: u32 = arguments[7].parse()?;
    if !(1..=60).contains(&output_fps) {
        return Err("video fps must be in 1..=60".into());
    }
    let capture_dir = Path::new(&arguments[6]);
    let receipt = if let Some(existing) = existing_capture(
        capture_dir,
        &frames,
        model.definition().file_sha256(),
        output_fps,
    )? {
        existing
    } else {
        run_robot_pose_sequence_capture(
            RobotPoseSequenceCaptureResources {
                asset_root,
                scene,
                model,
                camera,
                poses_60hz: frames.into_iter().map(Arc::new).collect(),
            },
            RobotPoseSequenceCaptureOptions {
                output_dir: capture_dir.to_path_buf(),
                output_fps,
            },
        )?
    };
    let video = if let Some(video_name) = arguments.get(9) {
        let video_path = Path::new(video_name);
        if video_path.exists() {
            return Err("video output must be new".into());
        }
        let input_pattern = Path::new(&arguments[6]).join("frame_%06d.png");
        let encoded = Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-nostdin")
            .arg("-framerate")
            .arg(&arguments[7])
            .arg("-start_number")
            .arg("0")
            .arg("-i")
            .arg(&input_pattern)
            .arg("-frames:v")
            .arg(receipt.frames.len().to_string())
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-movflags")
            .arg("+faststart")
            .arg(video_path)
            .output()?;
        if !encoded.status.success() {
            return Err(format!(
                "ffmpeg video encoding failed: {}",
                String::from_utf8_lossy(&encoded.stderr)
            )
            .into());
        }
        let video_bytes = fs::read(video_path)?;
        if video_bytes.is_empty() {
            return Err("encoded MP4 is empty".into());
        }
        let inspected = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
            .arg(video_path)
            .args(["-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1"])
            .output()?;
        if !inspected.status.success() {
            return Err("ffmpeg could not decode encoded video frames".into());
        }
        let progress = String::from_utf8(inspected.stdout)?;
        let count = progress
            .lines()
            .filter_map(|line| line.strip_prefix("frame="))
            .last()
            .ok_or("ffmpeg decode did not return a frame count")?
            .parse::<usize>()?;
        if count != receipt.frames.len() {
            return Err(format!(
                "encoded video frame count {count} differs from captured {}",
                receipt.frames.len()
            )
            .into());
        }
        json!({"path":fs::canonicalize(video_path)?,"sha256":digest(&video_bytes),"frames":count,"fps":receipt.output_fps})
    } else {
        Value::Null
    };
    let document = json!({
        "schema":"robot_native_pose_diagnostic_video_v1",
        "scope":"reported_rapier_ort_60hz_pose_gpu_capture_structurally_bound",
        "source_evidence_validation":"trace, pose, model, scene, clock and count structure; producer provenance requires run receipt",
        "source_report_sha256":digest(&report_bytes),
        "source_report":arguments[5],
        "control_law":report["control_law"],
        "station_contact_qualified":false,
        "bam_external_load_qualified":false,
        "skill_qualified":false,
        "capture":receipt,
        "video":video,
    });
    fs::write(output, serde_json::to_vec_pretty(&document)?)?;
    println!(
        "STATUS: success; captured_frames={}; receipt={}",
        document["capture"]["frames"].as_array().map_or(0, Vec::len),
        output.display()
    );
    Ok(())
}
