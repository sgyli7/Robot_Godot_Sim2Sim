//! Inspect the private source-self registry with zero physics/policy steps.

use std::{collections::HashMap, error::Error, fs, io::Write, path::Path, sync::Arc};

use robot_minigame::{collision_profile::SourceCollisionProfile, definition::RobotDefinition};
use serde_json::{Value, json};
use simulation_minigame::source_collision::SourceCollisionWorld;

type CheckResult<T> = Result<T, Box<dyn Error>>;

fn execute(args: &HashMap<String, String>) -> CheckResult<Value> {
    let required = |key: &str| {
        args.get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}"))
    };
    let definition_path = Path::new(required("--definition")?);
    let definition = Arc::new(
        match definition_path.extension().and_then(|value| value.to_str()) {
            Some("json") => {
                RobotDefinition::load_json(definition_path, required("--definition-sha")?)?
            }
            Some("ron") => {
                RobotDefinition::load_ron(definition_path, required("--definition-sha")?)?
            }
            _ => return Err("definition must be actual JSON or RON".into()),
        },
    );
    let profile_path = Path::new(required("--profile")?);
    let profile = Arc::new(
        match profile_path.extension().and_then(|value| value.to_str()) {
            Some("json") => SourceCollisionProfile::load_json(
                profile_path,
                required("--profile-sha")?,
                &definition,
            )?,
            Some("ron") => SourceCollisionProfile::load_ron(
                profile_path,
                required("--profile-sha")?,
                &definition,
            )?,
            _ => return Err("profile must be actual JSON or RON".into()),
        },
    );
    // Explicit development source-home fixture. This is not a game spawn or
    // target rollout and never calls any physics or controller step.
    let qpos = definition.model().fields.key_qpos[0].clone();
    let world = SourceCollisionWorld::new(definition, profile, &qpos)?;
    let report = world.validate_boundary(world.instance_token()?)?;
    if report.integration_count != 0 || report.torque_update_count != 0 {
        return Err("registration inspection unexpectedly entered a physical boundary".into());
    }
    Ok(
        json!({"passed":true,"scope":"source_self_registry_zero_step_only",
        "fixture":"explicit_source_home","registration":report,"snapshot":world.snapshot(),
        "physics_integrations":0,"policy_inferences":0,"runtime_contact_hook_executed":false,
        "contact_or_ccd_qualified":false,"force_or_bam_qualified":false,
        "environment_supported":false,"target_plant_accepted":false}),
    )
}

fn main() -> CheckResult<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let keys = [
        "--definition",
        "--definition-sha",
        "--profile",
        "--profile-sha",
        "--output",
    ];
    if raw.len() != keys.len() * 2 {
        return Err(
            "expected --definition --definition-sha --profile --profile-sha --output".into(),
        );
    }
    let mut args = HashMap::new();
    for pair in raw.chunks_exact(2) {
        if !keys.contains(&pair[0].as_str())
            || args.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err("unknown or repeated argument".into());
        }
    }
    let result = execute(&args);
    let report = match &result {
        Ok(report) => report.clone(),
        Err(error) => json!({"passed":false,"scope":"source_self_registry_zero_step_only",
            "error":error.to_string(),"contact_or_ccd_qualified":false,"target_plant_accepted":false}),
    };
    let output = args.get("--output").ok_or("missing --output")?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?
        .write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!("{}", json!({"output":output,"passed":result.is_ok()}));
    result.map(|_| ())
}
