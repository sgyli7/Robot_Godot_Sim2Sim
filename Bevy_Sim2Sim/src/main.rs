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
    g1_config: Option<PathBuf>,
    g1_ticks: Option<u32>,
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
    if matches!(
        arguments.scene.as_deref(),
        Some(
            "g1_camera_diagnostic"
                | "g1_mobile_wait_grasp_diagnostic"
                | "g1_task_lab"
                | "g1_mobile_carry_diagnostic"
                | "g1_mobile_assist_diagnostic"
                | "g1_mobile_scan_diagnostic"
                | "g1_mobile_target_view_diagnostic"
                | "g1_mobile_target_approach_diagnostic"
                | "g1_mobile_target_raise_view_diagnostic"
                | "g1_mobile_target_memory_view_diagnostic"
                | "g1_mobile_target_restored_view_diagnostic"
                | "g1_mobile_auxiliary_view_diagnostic"
                | "g1_mobile_auxiliary_approach_diagnostic"
                | "g1_mobile_auxiliary_release_diagnostic"
        )
    ) {
        if arguments.robot.as_deref() != Some("g1")
            || arguments.headless
            || arguments.verify
            || arguments.capture.is_some()
            || arguments.frames.is_some()
        {
            return Err("G1 development scenes require --robot g1, --g1-config and --output; headless/verify/preview capture flags do not apply".into());
        }
        return capture_g1_diagnostic(arguments);
    }
    if arguments.g1_config.is_some() || arguments.g1_ticks.is_some() {
        return Err("G1 diagnostic options require an explicit G1 development scene".into());
    }
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

#[cfg(feature = "dev_tools")]
fn capture_g1_diagnostic(arguments: Arguments) -> Result<(), String> {
    let interactive = arguments.scene.as_deref() == Some("g1_task_lab");
    let mobile_carry = arguments.scene.as_deref() == Some("g1_mobile_carry_diagnostic");
    let mobile_assist = arguments.scene.as_deref() == Some("g1_mobile_assist_diagnostic");
    let mobile_scan = arguments.scene.as_deref() == Some("g1_mobile_scan_diagnostic");
    let mobile_target_view = arguments.scene.as_deref() == Some("g1_mobile_target_view_diagnostic");
    let mobile_target_approach =
        arguments.scene.as_deref() == Some("g1_mobile_target_approach_diagnostic");
    let mobile_target_raise_view =
        arguments.scene.as_deref() == Some("g1_mobile_target_raise_view_diagnostic");
    let mobile_target_memory_view =
        arguments.scene.as_deref() == Some("g1_mobile_target_memory_view_diagnostic");
    let mobile_target_restored_view =
        arguments.scene.as_deref() == Some("g1_mobile_target_restored_view_diagnostic");
    let mobile_auxiliary_view =
        arguments.scene.as_deref() == Some("g1_mobile_auxiliary_view_diagnostic");
    let mobile_auxiliary_approach =
        arguments.scene.as_deref() == Some("g1_mobile_auxiliary_approach_diagnostic");
    let mobile_auxiliary_release =
        arguments.scene.as_deref() == Some("g1_mobile_auxiliary_release_diagnostic");
    if interactive && arguments.g1_ticks.is_some() {
        return Err(
            "g1_task_lab admits UI intentions; --g1-ticks automatic execution is forbidden".into(),
        );
    }
    let path = arguments.g1_config.ok_or("--g1-config is required")?;
    let output = arguments
        .output
        .ok_or("--output must name a new directory")?;
    let options = dev_tools_minigame::g1_capture::G1CaptureOptions {
        output,
        ticks: arguments.g1_ticks.unwrap_or(0),
        timeout: Duration::from_secs(
            if mobile_carry
                || mobile_assist
                || mobile_scan
                || mobile_target_view
                || mobile_target_approach
                || mobile_target_raise_view
                || mobile_target_memory_view
                || mobile_target_restored_view
                || mobile_auxiliary_view
                || mobile_auxiliary_approach
                || mobile_auxiliary_release
            {
                120
            } else {
                75
            },
        ),
    };
    let receipt = if interactive {
        dev_tools_minigame::g1_capture::run_task_lab_from_file(&path, options)
    } else if arguments.scene.as_deref() == Some("g1_mobile_wait_grasp_diagnostic") {
        dev_tools_minigame::g1_capture::run_mobile_wait_grasp_from_file(&path, options)
    } else if mobile_carry {
        dev_tools_minigame::g1_capture::run_mobile_carry_from_file(&path, options)
    } else if mobile_auxiliary_release {
        dev_tools_minigame::g1_capture::run_mobile_auxiliary_release_from_file(&path, options)
    } else if mobile_auxiliary_approach {
        dev_tools_minigame::g1_capture::run_mobile_auxiliary_approach_from_file(&path, options)
    } else if mobile_auxiliary_view {
        dev_tools_minigame::g1_capture::run_mobile_auxiliary_view_from_file(&path, options)
    } else if mobile_target_restored_view {
        dev_tools_minigame::g1_capture::run_mobile_target_restored_view_from_file(&path, options)
    } else if mobile_target_memory_view {
        dev_tools_minigame::g1_capture::run_mobile_target_memory_view_from_file(&path, options)
    } else if mobile_target_raise_view {
        dev_tools_minigame::g1_capture::run_mobile_target_raise_view_from_file(&path, options)
    } else if mobile_target_approach {
        dev_tools_minigame::g1_capture::run_mobile_target_approach_from_file(&path, options)
    } else if mobile_target_view {
        dev_tools_minigame::g1_capture::run_mobile_target_view_from_file(&path, options)
    } else if mobile_scan {
        dev_tools_minigame::g1_capture::run_mobile_scan_from_file(&path, options)
    } else if mobile_assist {
        dev_tools_minigame::g1_capture::run_mobile_assist_from_file(&path, options)
    } else {
        dev_tools_minigame::g1_capture::run_capture_from_file(&path, options)
    }?;
    if interactive {
        println!(
            "native_task_lab window_closed=true, integrations={}, task_qualified=false",
            receipt.actual_integrations
        );
        return Ok(());
    }
    println!(
        "native_camera_diagnostic capture={}, integrations={}, body_policy_calls={}, live_vla_calls={}, task_qualified={}",
        receipt.capture_succeeded,
        receipt.actual_integrations,
        receipt.actual_model_attempts,
        receipt.live_policy_inference_calls,
        receipt.task_qualified,
    );
    Ok(())
}

#[cfg(not(feature = "dev_tools"))]
fn capture_g1_diagnostic(_arguments: Arguments) -> Result<(), String> {
    Err("G1 camera diagnostic requires the non-default Cargo feature 'dev_tools'".into())
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
                     --scene NAME    foundation, science_station_preview, g1_camera_diagnostic, g1_task_lab, g1_mobile_carry_diagnostic g1_mobile_assist_diagnostic g1_mobile_scan_diagnostic g1_mobile_target_view_diagnostic, g1_mobile_target_approach_diagnostic or g1_mobile_target_raise_view_diagnostic or g1_mobile_target_memory_view_diagnostic or g1_mobile_target_restored_view_diagnostic or g1_mobile_auxiliary_view_diagnostic or g1_mobile_auxiliary_approach_diagnostic or g1_mobile_auxiliary_release_diagnostic\n\
                     --robot NAME    none, or g1 for the explicit unqualified camera diagnostic\n\
                     --headless      run without a window\n\
                     --verify        scoped foundation check (requires dev_tools feature)\n\
                     --output PATH   verification or G1 diagnostic directory\n\
                     --g1-config PATH  frozen G1 camera diagnostic JSON (requires dev_tools)\n\
                     --g1-ticks N    camera 0..400 (standing max150); mobile waited grasp1000; mobile carry1000/1500; assisted carry maximum2050\n\
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
            "--g1-config" => options.g1_config = Some(value()?.into()),
            "--g1-ticks" => {
                let ticks = value()?
                    .parse::<u32>()
                    .map_err(|_| "--g1-ticks requires a nonnegative integer")?;
                if ticks > 3150 {
                    return Err("--g1-ticks must be at most3150 before scene validation".into());
                }
                options.g1_ticks = Some(ticks);
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
    if options.scene.as_deref() == Some("g1_mobile_carry_diagnostic") {
        if !matches!(options.g1_ticks, Some(1000 | 1500)) {
            return Err("mobile carry scene requires --g1-ticks1000 or1500".into());
        }
    } else if options.scene.as_deref() == Some("g1_mobile_wait_grasp_diagnostic") {
        if options.g1_ticks != Some(1000) {
            return Err("mobile waited grasp requires --g1-ticks1000 as its finite maximum".into());
        }
    } else if options.scene.as_deref() == Some("g1_mobile_assist_diagnostic") {
        if options.g1_ticks != Some(2050) {
            return Err("mobile assist scene requires --g1-ticks2050 as its finite maximum".into());
        }
    } else if matches!(
        options.scene.as_deref(),
        Some("g1_mobile_scan_diagnostic" | "g1_mobile_auxiliary_view_diagnostic")
    ) {
        if options.g1_ticks != Some(1050) {
            return Err("mobile scan scene requires --g1-ticks1050 as its finite maximum".into());
        }
    } else if matches!(
        options.scene.as_deref(),
        Some(
            "g1_mobile_target_approach_diagnostic"
                | "g1_mobile_target_raise_view_diagnostic"
                | "g1_mobile_target_memory_view_diagnostic"
                | "g1_mobile_target_restored_view_diagnostic"
                | "g1_mobile_auxiliary_approach_diagnostic"
                | "g1_mobile_auxiliary_release_diagnostic"
        )
    ) {
        if options.g1_ticks != Some(3150) {
            return Err(
                "mobile target approach requires --g1-ticks3150 as its finite maximum".into(),
            );
        }
    } else if options.scene.as_deref() == Some("g1_mobile_target_view_diagnostic") {
        if options.g1_ticks != Some(1300) {
            return Err("mobile target view requires --g1-ticks1300 as its finite maximum".into());
        }
    } else if options.g1_ticks.is_some_and(|ticks| ticks > 400) {
        return Err(
            "--g1-ticks must be at most400 outside the explicit mobile development scenes".into(),
        );
    }
    Ok(Some(options))
}

#[cfg(test)]
mod tests {
    #[test]
    fn auxiliary_release_requires_its_fixed_finite_budget() {
        for ticks in ["1050", "2050", "3151"] {
            assert!(
                super::parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_auxiliary_release_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--scene",
                    "g1_mobile_auxiliary_release_diagnostic",
                    "--g1-ticks",
                    "3150"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
    }
    #[test]
    fn auxiliary_approach_cannot_escape_its_fixed_budget() {
        for ticks in ["1050", "1300", "2050", "3151"] {
            assert!(
                super::parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_auxiliary_approach_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--g1-ticks",
                    "3150",
                    "--scene",
                    "g1_mobile_auxiliary_approach_diagnostic"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
    }
    #[test]
    fn auxiliary_view_keeps_the_scan_budget() {
        for ticks in ["1051", "1300", "2050", "3150"] {
            assert!(
                super::parse_arguments(
                    [
                        "--g1-ticks",
                        ticks,
                        "--scene",
                        "g1_mobile_auxiliary_view_diagnostic"
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--scene",
                    "g1_mobile_auxiliary_view_diagnostic",
                    "--g1-ticks",
                    "1050"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
    }
    #[test]
    fn restored_view_keeps_the_bounded_development_budget() {
        for ticks in ["1300", "2050", "3151"] {
            assert!(
                super::parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_target_restored_view_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--scene",
                    "g1_mobile_target_restored_view_diagnostic",
                    "--g1-ticks",
                    "3150"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
    }
    #[test]
    fn target_memory_scene_cannot_widen_other_scene_budgets() {
        for ticks in ["1300", "2050", "3151"] {
            assert!(
                super::parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_target_memory_view_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--scene",
                    "g1_mobile_target_memory_view_diagnostic",
                    "--g1-ticks",
                    "3150"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
    }
    #[test]
    fn raised_camera_scene_has_only_its_explicit_bounded_budget() {
        for ticks in ["1300", "2050", "3151"] {
            assert!(
                super::parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_target_raise_view_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--scene",
                    "g1_mobile_target_raise_view_diagnostic",
                    "--g1-ticks",
                    "3150"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
    }
    #[test]
    fn visual_approach_budget_does_not_widen_other_scenes() {
        for args in [
            [
                "--scene",
                "g1_mobile_target_approach_diagnostic",
                "--g1-ticks",
                "3150",
            ],
            [
                "--g1-ticks",
                "3150",
                "--scene",
                "g1_mobile_target_approach_diagnostic",
            ],
        ] {
            assert!(super::parse_arguments(args.into_iter().map(str::to_owned)).is_ok());
        }
        for scene in [
            "g1_mobile_target_view_diagnostic",
            "g1_mobile_scan_diagnostic",
            "g1_mobile_assist_diagnostic",
            "g1_camera_diagnostic",
        ] {
            assert!(
                super::parse_arguments(
                    ["--scene", scene, "--g1-ticks", "3150"]
                        .into_iter()
                        .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                [
                    "--scene",
                    "g1_mobile_target_approach_diagnostic",
                    "--g1-ticks",
                    "3151"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_err()
        );
    }
    #[test]
    fn target_view_budget_cannot_escape_to_scan_or_carry() {
        assert!(
            super::parse_arguments(
                [
                    "--g1-ticks",
                    "1300",
                    "--scene",
                    "g1_mobile_target_view_diagnostic"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_ok()
        );
        for scene in [
            "g1_camera_diagnostic",
            "g1_mobile_scan_diagnostic",
            "g1_mobile_assist_diagnostic",
        ] {
            assert!(
                super::parse_arguments(
                    ["--scene", scene, "--g1-ticks", "1300"]
                        .into_iter()
                        .map(str::to_owned)
                )
                .is_err()
            );
        }
    }
    #[test]
    fn scan_has_a_separate_turn_only_budget() {
        assert!(
            super::parse_arguments(
                ["--scene", "g1_mobile_scan_diagnostic", "--g1-ticks", "1050"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_ok()
        );
        for scene in [
            "g1_camera_diagnostic",
            "g1_mobile_assist_diagnostic",
            "g1_mobile_carry_diagnostic",
        ] {
            assert!(
                super::parse_arguments(
                    ["--g1-ticks", "1050", "--scene", scene]
                        .into_iter()
                        .map(str::to_owned)
                )
                .is_err()
            );
        }
        assert!(
            super::parse_arguments(
                ["--scene", "g1_mobile_scan_diagnostic", "--g1-ticks", "2050"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err()
        );
    }
    #[test]
    fn mobile_assist_admits_only_its_separate_maximum_in_either_argument_order() {
        for args in [
            [
                "--g1-ticks",
                "2050",
                "--scene",
                "g1_mobile_assist_diagnostic",
            ],
            [
                "--scene",
                "g1_mobile_assist_diagnostic",
                "--g1-ticks",
                "2050",
            ],
        ] {
            assert!(super::parse_arguments(args.into_iter().map(str::to_owned)).is_ok());
        }
        for ticks in ["0", "200", "400", "1000", "1500", "2051"] {
            assert!(
                super::parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_assist_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .into_iter()
                    .map(str::to_owned)
                )
                .is_err()
            );
        }
    }
    use super::*;

    #[test]
    fn cli_rejects_missing_values_and_unknown_options() {
        assert!(parse_arguments(["--robot".to_owned()].into_iter()).is_err());
        assert!(parse_arguments(["--unknown".to_owned()].into_iter()).is_err());
        assert!(parse_arguments(["--frames".to_owned(), "0".to_owned()].into_iter()).is_err());
    }

    #[test]
    fn waited_mobile_grasp_is_an_explicit_finite_scene() {
        assert!(
            parse_arguments(
                [
                    "--scene",
                    "g1_mobile_wait_grasp_diagnostic",
                    "--g1-ticks",
                    "1000"
                ]
                .map(str::to_owned)
                .into_iter()
            )
            .is_ok()
        );
        for ticks in ["200", "999", "1001", "3150"] {
            assert!(
                parse_arguments(
                    [
                        "--scene",
                        "g1_mobile_wait_grasp_diagnostic",
                        "--g1-ticks",
                        ticks
                    ]
                    .map(str::to_owned)
                    .into_iter()
                )
                .is_err()
            );
        }
    }
    #[test]
    fn longer_mobile_budget_cannot_escape_its_scene_or_depend_on_argument_order() {
        for ticks in ["1000", "1500"] {
            for args in [
                ["--g1-ticks", ticks, "--scene", "g1_mobile_carry_diagnostic"],
                ["--scene", "g1_mobile_carry_diagnostic", "--g1-ticks", ticks],
            ] {
                assert_eq!(
                    parse_arguments(args.map(str::to_owned).into_iter())
                        .unwrap()
                        .unwrap()
                        .g1_ticks,
                    Some(ticks.parse().unwrap())
                );
            }
        }
        for args in [
            ["--scene", "g1_camera_diagnostic", "--g1-ticks", "1000"],
            ["--scene", "g1_task_lab", "--g1-ticks", "1000"],
            [
                "--scene",
                "g1_mobile_carry_diagnostic",
                "--g1-ticks",
                "1001",
            ],
            ["--scene", "g1_mobile_carry_diagnostic", "--g1-ticks", "400"],
        ] {
            assert!(parse_arguments(args.map(str::to_owned).into_iter()).is_err());
        }
    }
}
