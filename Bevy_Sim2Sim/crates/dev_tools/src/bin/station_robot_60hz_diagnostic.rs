//! Original bounded, offline 60 Hz ONNX diagnostic entry point.

use dev_tools_minigame::station_robot_diagnostic::{initial_report, run_offline};
use serde_json::json;
use std::{error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 10 {
        return Err("usage: station_robot_60hz_diagnostic MODEL MODEL_SHA QPOS QPOS_SHA ASSETS ONNX POLICY_CONTRACT|--legacy-original NATIVE_ORT_LIB TICKS NEW_REPORT.json".into());
    }
    let output = Path::new(&arguments[9]);
    if output.exists() {
        return Err("report output must be new".into());
    }
    let mut report = initial_report();
    let result = run_offline(&arguments, &mut report);
    if let Err(error) = &result {
        report["error"] = json!(error);
    }
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "STATUS: {}; report={}",
        if result.is_ok() { "success" } else { "failed" },
        output.display()
    );
    result.map_err(|error| error.into())
}
