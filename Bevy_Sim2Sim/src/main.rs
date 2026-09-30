//! Game startup and CLI assembly; optional verification remains in dev_tools.

use std::{
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

use bevy::{app::AppExit, prelude::*};
use rendering_minigame::{
    StationCameraControl, StationLabelBakeStatus, StationRenderHealth, StationScene, StationView,
    StationVisualPlugin, install_station_render_health, validate_render_asset_root,
};

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
    if arguments.capture.is_some() || arguments.frames.is_some() {
        return capture_preview(arguments);
    }
    let asset_root = validate_render_asset_root(&rendering_minigame::default_asset_root())?;
    let scene = StationScene::load(&asset_root)?;
    let mut app = App::new();
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .insert_resource(scene)
        .insert_resource(StationCameraControl {
            view: arguments.view,
            ..default()
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Bevy Science Station — visual preview (no robot)".into(),
                        resolution: (1920, 1080).into(),
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(StationVisualPlugin);
    let health = install_station_render_health(&mut app)?;
    let labels = app.world().resource::<StationLabelBakeStatus>().clone();
    app.insert_resource(WindowLifecycle {
        started: Instant::now(),
        frames: 0,
        first_error_frame: None,
    })
    .add_systems(Update, check_window_health);
    let exit = app.run();
    let status = health.snapshot();
    if let Some(error) = status.error {
        return Err(error);
    }
    if !status.ready || !labels.is_ready() {
        return Err(
            "Renderer exited before required station assets and pipelines were ready".into(),
        );
    }
    match exit {
        AppExit::Success => Ok(()),
        AppExit::Error(code) => Err(format!("Station window exited with error {code}")),
    }
}

#[derive(Resource)]
struct WindowLifecycle {
    started: Instant,
    frames: u32,
    first_error_frame: Option<u32>,
}

fn check_window_health(
    health: Res<StationRenderHealth>,
    labels: Res<StationLabelBakeStatus>,
    mut lifecycle: ResMut<WindowLifecycle>,
    mut exit: MessageWriter<AppExit>,
) {
    lifecycle.frames += 1;
    let status = health.snapshot();
    if let Some(error) = status.error {
        let frame = lifecycle.frames;
        if lifecycle.first_error_frame.is_none() {
            error!("{error}");
        }
        let first = *lifecycle.first_error_frame.get_or_insert(frame);
        if (frame - first >= 8 && status.pending == 0) || frame - first >= 180 {
            exit.write(AppExit::error());
        }
    } else if (!status.ready || !labels.is_ready())
        && lifecycle.started.elapsed() > Duration::from_secs(60)
    {
        error!("Station assets and pipelines did not become ready within 60 seconds");
        exit.write(AppExit::error());
    }
}

#[cfg(feature = "dev_tools")]
fn capture_preview(arguments: Arguments) -> Result<(), String> {
    dev_tools_minigame::visual_preview::run_preview_with_options(
        dev_tools_minigame::visual_preview::PreviewOptions {
            capture_path: arguments.capture,
            frames: arguments.frames,
            view: arguments.view,
        },
    )
}

#[cfg(not(feature = "dev_tools"))]
fn capture_preview(_arguments: Arguments) -> Result<(), String> {
    Err("preview capture and frame limits require the non-default Cargo feature 'dev_tools'".into())
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
                     --capture PATH  preview screenshot file (requires dev_tools)\n\
                     --frames N      exit preview after N displayed frames (requires dev_tools)\n\
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
