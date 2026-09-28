//! Game startup and CLI assembly; optional verification remains in dev_tools.

use std::{path::PathBuf, process::ExitCode};

use rendering_minigame::{PreviewOptions, StationView};

#[derive(Default)]
struct Arguments {
    scene: Option<String>,
    robot: Option<String>,
    headless: bool,
    verify: bool,
    output: Option<PathBuf>,
    capture: Option<PathBuf>,
    frames: Option<u32>,
    view: StationView,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Bevy_Sim2Sim failed: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let Some(arguments) = parse_arguments(std::env::args().skip(1))? else {
        return Ok(());
    };
    let robot = arguments.robot.as_deref().unwrap_or("none");
    if robot != "none" {
        return Err(format!(
            "real robot mode '{robot}' has no qualified body/policy integration yet; foundation verification cannot substitute for it"
        ));
    }
    let default_scene = if arguments.headless {
        "foundation"
    } else {
        "science_station_preview"
    };
    let scene = arguments.scene.as_deref().unwrap_or(default_scene);
    if arguments.verify {
        if !arguments.headless || scene != "foundation" {
            return Err(
                "this verification phase requires --headless --scene foundation --robot none"
                    .into(),
            );
        }
        if arguments.capture.is_some() || arguments.frames.is_some() {
            return Err(
                "render capture options cannot be used with foundation verification".into(),
            );
        }
        return verify(
            arguments
                .output
                .unwrap_or_else(|| ".scratch/foundation".into()),
        );
    }
    if arguments.output.is_some() {
        return Err("--output currently requires --verify".into());
    }
    if arguments.headless {
        return Err("headless runtime has no qualified policy; use --verify for the explicitly scoped foundation fixture".into());
    }
    if scene != "science_station_preview" {
        return Err(format!(
            "scene '{scene}' has not been implemented/qualified"
        ));
    }
    eprintln!(
        "Starting science-station visual preview; robot behavior and physics qualification are unavailable."
    );
    rendering_minigame::run_preview_with_options(PreviewOptions {
        capture_path: arguments.capture,
        frames: arguments.frames,
        view: arguments.view,
    })
}

#[cfg(feature = "dev_tools")]
fn verify(output: PathBuf) -> Result<(), String> {
    let report =
        dev_tools_minigame::verify_foundation(&output).map_err(|error| error.to_string())?;
    println!(
        "foundation_only passed: physics_steps={}, control_boundaries={}, inference_count={}; report={}",
        report.physics_steps,
        report.control_boundaries,
        report.inference_count,
        output.join("summary.json").display(),
    );
    Ok(())
}

#[cfg(not(feature = "dev_tools"))]
fn verify(_output: PathBuf) -> Result<(), String> {
    Err("verification requires the non-default Cargo feature 'dev_tools'".into())
}

fn parse_arguments(args: impl Iterator<Item = String>) -> Result<Option<Arguments>, String> {
    let mut args = args.peekable();
    let mut options = Arguments::default();
    while let Some(argument) = args.next() {
        let mut value = || {
            args.next()
                .filter(|value| !value.starts_with("--"))
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--help" | "-h" => {
                println!(
                    "Bevy_Sim2Sim foundation and station preview\n\
                     --scene NAME    foundation or science_station_preview\n\
                     --robot NAME    none; real policy modes are not qualified yet\n\
                     --headless      run without a window\n\
                     --verify        scoped foundation check (requires dev_tools feature)\n\
                     --output PATH   verification report directory\n\
                     --capture PATH  preview screenshot file\n\
                     --frames N      exit preview after N displayed frames\n\
                     --view NAME     arrival, overview, towers, samples, berth, hills, follow"
                );
                return Ok(None);
            }
            "--scene" => options.scene = Some(value()?),
            "--robot" => options.robot = Some(value()?),
            "--headless" => options.headless = true,
            "--verify" => options.verify = true,
            "--output" => options.output = Some(value()?.into()),
            "--capture" => options.capture = Some(value()?.into()),
            "--frames" => {
                let frames = value()?
                    .parse::<u32>()
                    .map_err(|_| "--frames needs a positive integer")?;
                if frames == 0 {
                    return Err("--frames must be positive".into());
                }
                options.frames = Some(frames);
            }
            "--view" => {
                options.view = match value()?.as_str() {
                    "arrival" => StationView::Arrival,
                    "overview" => StationView::Overview,
                    "towers" => StationView::Towers,
                    "samples" => StationView::Samples,
                    "berth" => StationView::Berth,
                    "hills" => StationView::Hills,
                    "follow" => StationView::Follow,
                    value => return Err(format!("unknown station view '{value}'")),
                };
            }
            value => return Err(format!("unknown argument '{value}'")),
        }
    }
    Ok(Some(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_rejects_missing_values_and_unknown_options() {
        assert!(parse_arguments(["--robot".to_owned()].into_iter()).is_err());
        assert!(parse_arguments(["--unknown".to_owned()].into_iter()).is_err());
        assert!(parse_arguments(["--frames".to_owned(), "0".to_owned()].into_iter()).is_err());
    }
}
