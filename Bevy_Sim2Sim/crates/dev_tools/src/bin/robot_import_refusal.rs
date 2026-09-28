//! Exercise malformed imports against an existing native world, without stepping.

use robot_minigame::definition::RobotDefinition;
use serde_json::json;
use simulation_minigame::{SimulationWorld, robot_builder::build_structure};
use std::{error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || args[2] != "--output" {
        return Err("expected malformed model path, SHA256, --output result.json".into());
    }
    let mut simulation = SimulationWorld::foundation();
    let before = simulation.counts();
    let path = Path::new(&args[0]);
    let loaded = match path.extension().and_then(|extension| extension.to_str()) {
        Some("ron") => RobotDefinition::load_ron(path, &args[1]),
        Some("json") => RobotDefinition::load_json(path, &args[1]),
        _ => return Err("expected .ron or .json compiled model".into()),
    };
    let (stage, rejection) = match loaded {
        Err(error) => ("definition", Some(error.to_string())),
        Ok(definition) => (
            "native_import",
            build_structure(
                &mut simulation.world,
                &definition,
                &definition.model().fields.key_qpos[0],
            )
            .err()
            .map(|error| error.to_string()),
        ),
    };
    let after = simulation.counts();
    let passed = rejection.is_some() && before == after;
    let report = json!({
        "scope": "native_import_refusal_only", "input":args[0],
        "expected_file_sha256":args[1], "stage":stage, "rejection":rejection,
        "before":before, "after":after, "preexisting_world_counts_preserved":before == after,
        "physics_integrations":0, "policy_inferences":0, "passed":passed,
    });
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("{}", serde_json::to_string(&report)?);
    if !passed {
        return Err("malformed import was admitted or changed existing world counts".into());
    }
    Ok(())
}
