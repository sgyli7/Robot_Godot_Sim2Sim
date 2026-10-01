//! Offline saved native RGB/self-state to one matched local VLA request.

use dev_tools_minigame::g1_policy_diagnostic::{G1PolicyProbeOptions, run_visual_policy_probe};

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() < 3 {
        return Err("usage: g1_visual_policy_probe EGO.png STAMP.json NEW_RECEIPT.json --profile static_apple|mobile_box --endpoint http://127.0.0.1:PORT/infer --offline-diagnostic".into());
    }
    let options = G1PolicyProbeOptions {
        png: arguments[0].clone().into(),
        stamp: arguments[1].clone().into(),
        output: arguments[2].clone().into(),
        flags: arguments[3..].to_vec(),
    };
    let result = run_visual_policy_probe(&options);
    println!(
        "STATUS: {}; receipt={}",
        if result.is_ok() { "success" } else { "failed" },
        options.output.display()
    );
    result.map(|_| ())
}
