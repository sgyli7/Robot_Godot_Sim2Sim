//! Explicit GPU evidence check; pure camera contracts live in the module tests.
use rendering_minigame::g1_camera;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    winit::WinitPlugin,
};
use g1_camera::{
    CameraPoseSource, G1CameraInput, G1CameraPlugin, G1CameraPort, G1CameraSourceFrame,
};
use rendering_minigame::{
    StationLabelBakeStatus, StationRenderHealth, StationScene, StationVisualPlugin,
    default_asset_root, install_station_render_health, validate_render_asset_root,
};

#[derive(Resource, Clone, Default)]
struct CaptureOutcome(Arc<Mutex<(bool, bool, Option<String>)>>);

#[derive(Resource)]
struct CaptureHarness {
    output: PathBuf,
    started: Instant,
    frames: u32,
    requested: bool,
}

/// Explicit GPU test: the ego frame uses the published station "samples" shot,
/// not a robot or object transform. Its provenance is StationFixture throughout.
#[test]
#[ignore = "opens a 1920x1080 station window; coordinate the GPU and set SAI_G1_CAMERA_OUTPUT"]
fn station_gpu_ego_rgb_and_full_resolution_main_view() {
    let output = PathBuf::from(
        std::env::var_os("SAI_G1_CAMERA_OUTPUT").expect("set a .scratch evidence directory"),
    );
    std::fs::create_dir_all(&output).unwrap();
    let assets = validate_render_asset_root(&default_asset_root()).unwrap();
    let scene = StationScene::load(&assets).unwrap();
    let shot = scene
        .0
        .layout
        .cameras
        .iter()
        .find(|camera| camera.name == "samples")
        .unwrap();
    let pose = Transform::from_translation(Vec3::from_array(shot.eye))
        .looking_at(Vec3::from_array(shot.target), Vec3::Y);
    let mut app = App::new();
    app.insert_resource(scene)
        .insert_resource(bevy::winit::WinitSettings::continuous())
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: assets.to_string_lossy().into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "G1 camera pipeline — station fixture only".into(),
                        resolution: (1920, 1080).into(),
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(WinitPlugin {
                    run_on_any_thread: true,
                    ..default()
                }),
        )
        .add_plugins((StationVisualPlugin, G1CameraPlugin))
        .insert_resource(G1CameraInput(Some(G1CameraSourceFrame {
            episode_id: 1,
            source_ticks: [0, 0],
            interpolation_alpha: 0.0,
            sim_time_ns: 0,
            source: CameraPoseSource::StationFixture,
            mount_profile: Default::default(),
            world_from_camera: pose,
            native_state: None,
        })))
        .insert_resource(CaptureHarness {
            output,
            started: Instant::now(),
            frames: 0,
            requested: false,
        })
        .init_resource::<CaptureOutcome>()
        .add_systems(Update, capture_fixture);
    app.world().resource::<G1CameraPort>().reset(1).unwrap();
    install_station_render_health(&mut app).unwrap();
    let outcome = app.world().resource::<CaptureOutcome>().clone();
    assert!(matches!(app.run(), AppExit::Success));
    let outcome = outcome.0.lock().unwrap();
    assert!(outcome.0 && outcome.1, "{:?}", outcome.2);
    assert!(outcome.2.is_none(), "{:?}", outcome.2);
}

fn capture_fixture(
    mut commands: Commands,
    mut state: ResMut<CaptureHarness>,
    outcome: Res<CaptureOutcome>,
    health: Res<StationRenderHealth>,
    labels: Res<StationLabelBakeStatus>,
    port: Res<G1CameraPort>,
    mut exit: MessageWriter<AppExit>,
) {
    state.frames += 1;
    if let Some(error) = health.snapshot().error {
        outcome.0.lock().unwrap().2 = Some(error);
    }
    if !state.requested && state.frames > 30 && health.snapshot().ready && labels.is_ready() {
        port.request().unwrap();
        state.requested = true;
        let path = state.output.join("main_1920x1080.png");
        commands.spawn(Screenshot::primary_window()).observe(
            move |event: On<ScreenshotCaptured>, outcome: Res<CaptureOutcome>| {
                let mut status = outcome.0.lock().unwrap();
                if event.image.width() != 1920 || event.image.height() != 1080 {
                    status.2 = Some("main window physical resolution changed".into());
                    return;
                }
                match event
                    .image
                    .clone()
                    .try_into_dynamic()
                    .map_err(|error| error.to_string())
                    .and_then(|image| image.save(&path).map_err(|error| error.to_string()))
                {
                    Ok(()) => status.1 = true,
                    Err(error) => status.2 = Some(error),
                }
            },
        );
    }
    if let Some(result) = port.take() {
        let result = result.and_then(|frame| {
            if frame.width != 640 || frame.height != 480 || frame.rgb.len() != 640 * 480 * 3 {
                return Err("ego output dimensions are wrong".into());
            }
            let min = *frame.rgb.iter().min().unwrap();
            let max = *frame.rgb.iter().max().unwrap();
            if max.saturating_sub(min) < 16 {
                return Err("ego image has no rendered contrast".into());
            }
            let rgba: Vec<u8> = frame
                .rgb
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                .collect();
            Image::new(
                Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                rgba,
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD,
            )
            .try_into_dynamic()
            .map_err(|error| error.to_string())?
            .save(state.output.join("ego_640x480.png"))
            .map_err(|error| error.to_string())?;
            std::fs::write(
                state.output.join("ego_stamp.json"),
                serde_json::to_vec_pretty(&frame.stamp).unwrap(),
            )
            .map_err(|error| error.to_string())?;
            println!(
                "G1_CAMERA_FIXTURE_CAPTURE {}",
                serde_json::to_string(&frame.stamp).unwrap()
            );
            Ok(())
        });
        let mut status = outcome.0.lock().unwrap();
        match result {
            Ok(()) => status.0 = true,
            Err(error) => status.2 = Some(error),
        }
    }
    let mut status = outcome.0.lock().unwrap();
    if state.started.elapsed() > Duration::from_secs(120) {
        status.2 = Some("camera fixture timed out".into());
    }
    if status.2.is_some() || (status.0 && status.1) {
        exit.write(AppExit::Success);
    }
}
