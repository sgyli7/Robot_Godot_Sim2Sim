//! Validate the source-compiled model before a backend can consume it.

use robot_minigame::definition::RobotDefinition;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if ![2, 4].contains(&args.len()) || (args.len() == 4 && args[2] != "--emit-ron") {
        return Err("expected model path and file SHA256, optionally --emit-ron output.ron".into());
    }
    let path = Path::new(&args[0]);
    let definition = match path.extension().and_then(|extension| extension.to_str()) {
        Some("json") => RobotDefinition::load_json(path, &args[1])?,
        Some("ron") => RobotDefinition::load_ron(path, &args[1])?,
        _ => return Err("expected .json interchange or .ron runtime configuration".into()),
    };
    let emitted = if args.len() == 4 {
        let output_path = Path::new(&args[3]);
        if output_path
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("ron")
        {
            return Err("runtime configuration must have .ron extension".into());
        }
        let ron = definition.to_ron()?;
        let hash = format!("{:x}", Sha256::digest(ron.as_bytes()));
        // Do not overwrite an already recorded candidate's configuration.
        use std::io::Write;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)?;
        output.write_all(ron.as_bytes())?;
        let round_trip = RobotDefinition::load_ron(output_path, &hash)?;
        if serde_json::to_value(round_trip.model())? != serde_json::to_value(definition.model())? {
            return Err("RON round trip changed compiled source data".into());
        }
        Some(serde_json::json!({"path": output_path, "sha256": hash, "round_trip_equal": true}))
    } else {
        None
    };
    let model = definition.model();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope": "compiled_model_validation_only",
            "family": model.family, "file_sha256": definition.file_sha256(),
            "source_report_sha256": model.sha256, "counts": model.counts,
            "emitted_runtime_config": emitted,
            "actuator_joint_ids": definition.actuator_joint_ids(),
            "source_home_body_poses": robot_minigame::kinematics::source_body_poses(&definition, &model.fields.key_qpos[0])?,
            "physics_integrations": 0, "policy_inferences": 0,
            "physical_behavior_accepted": false
        }))?
    );
    Ok(())
}
