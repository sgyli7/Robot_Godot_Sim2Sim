//! Print a static engineering-structure report as JSON.
//!
//! Exit 0 when every executed static check passed. Exit 1 when a rule failed
//! or the project manifest could not be read. The report does not claim that
//! rendering, physics, or runtime semantics passed.

#[path = "../structure.rs"]
mod structure;

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let root = match env::args().nth(1) {
        Some(argument) => PathBuf::from(argument),
        None => match env::current_dir() {
            Ok(current) => current,
            Err(error) => {
                let report = structure::configuration_report(format!(
                    "configuration: current directory is unavailable: {error}"
                ));
                println!("{}", structure::to_json(&report));
                return ExitCode::from(1);
            }
        },
    };
    let report = structure::check(&root);
    println!("{}", structure::to_json(&report));
    if report.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
