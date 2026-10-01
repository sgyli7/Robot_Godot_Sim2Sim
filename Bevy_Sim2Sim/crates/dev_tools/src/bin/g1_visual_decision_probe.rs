//! Explicit native-camera decision diagnostic; does not start service or physics.

use dev_tools_minigame::g1_decision_diagnostic::{
    G1DecisionProbeOptions, run_visual_decision_probe,
};
use task_minigame::{decision::LocalQwenConfig, types::TaskProfile};

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() < 3 {
        return Err("usage: g1_visual_decision_probe EGO.png STAMP.json NEW_RECEIPT.json [--warmup] [--model NAME] [--endpoint http://127.0.0.1:8002/v1] [--goal TEXT] [--profile static_apple|mobile_box]".into());
    }
    let config = LocalQwenConfig::default();
    let mut options = G1DecisionProbeOptions {
        png: arguments[0].clone().into(),
        stamp: arguments[1].clone().into(),
        output: arguments[2].clone().into(),
        instruction:
            "观察当前本体相机画面。尚未启用实体任务能力，只能观察或停止；不要宣称任务成功。".into(),
        profile: TaskProfile::StaticApple,
        base_url: config.base_url,
        model: config.model,
        warmup: false,
    };
    let mut index = 3;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--warmup" if !options.warmup => {
                options.warmup = true;
                index += 1;
            }
            flag @ ("--model" | "--endpoint" | "--goal" | "--profile") => {
                let value = arguments
                    .get(index + 1)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--model" => options.model = value.clone(),
                    "--endpoint" => options.base_url = value.clone(),
                    "--goal" => options.instruction = value.clone(),
                    "--profile" => {
                        options.profile = match value.as_str() {
                            "static_apple" => TaskProfile::StaticApple,
                            "mobile_box" => TaskProfile::MobileBox,
                            _ => return Err("profile must be static_apple or mobile_box".into()),
                        }
                    }
                    _ => unreachable!(),
                }
                index += 2;
            }
            flag => return Err(format!("unknown or repeated flag: {flag}")),
        }
    }
    let result = run_visual_decision_probe(&options);
    println!(
        "STATUS: {}; report={}",
        if result.is_ok() { "success" } else { "failed" },
        options.output.display()
    );
    result.map(|_| ())
}
