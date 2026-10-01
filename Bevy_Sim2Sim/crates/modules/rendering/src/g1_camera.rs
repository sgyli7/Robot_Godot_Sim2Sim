//! One-shot ego RGB capture from the same rendered world as the main camera.
//!
//! Source poses are supplied by the completed physics snapshot. A capture is
//! stamped when that scene is extracted, then copied after rendering. GPU map
//! completion has its own timestamp and never replaces the capture timestamp.

use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::{
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::FrameCount,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, Extent3d, MapMode, TexelCopyBufferInfo,
            TexelCopyBufferLayout, TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems},
        sync_world::RenderEntity,
        texture::GpuImage,
        view::ViewTarget,
    },
    transform::TransformSystems,
};
use robot_minigame::g1::definition::G1BodyFrame;
use serde::{Deserialize, Serialize};

pub const EGO_WIDTH: u32 = 640;
pub const EGO_HEIGHT: u32 = 480;

/// Arena's published pinhole contract, shared by the two selected ego policies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct G1CameraCalibration {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub vertical_fov_radians: f32,
    pub near_m: f32,
    pub far_m: f32,
}

impl Default for G1CameraCalibration {
    fn default() -> Self {
        Self {
            fx: 458.1245526,
            fy: 458.1245526,
            cx: 320.0,
            cy: 240.0,
            vertical_fov_radians: 0.96512935,
            near_m: 0.1,
            far_m: 5.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraPoseSource {
    PhysicsBody,
    StationFixture,
}

/// Exact camera pose in Bevy world coordinates, with both interpolation sources
/// if present. No object transform or target location is accepted here.
#[derive(Debug, Clone)]
pub struct G1CameraSourceFrame {
    pub episode_id: u64,
    pub source_ticks: [u64; 2],
    pub interpolation_alpha: f32,
    pub sim_time_ns: u64,
    pub source: CameraPoseSource,
    pub world_from_camera: Transform,
}

impl G1CameraSourceFrame {
    pub fn validate(&self) -> Result<(), String> {
        let transform = self.world_from_camera;
        let same_tick = self.source_ticks[0] == self.source_ticks[1];
        if !transform.translation.is_finite()
            || !transform.rotation.is_finite()
            || (transform.rotation.length_squared() - 1.0).abs() > 0.0001
            || transform.scale != Vec3::ONE
            || !self.interpolation_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.interpolation_alpha)
            || (same_tick && self.interpolation_alpha != 0.0)
            || (!same_tick && self.source_ticks[0].checked_add(1) != Some(self.source_ticks[1]))
        {
            return Err("invalid camera pose or interpolation source ticks".into());
        }
        Ok(())
    }
}

/// Arena static release 8b4a3a47 and mobile 7d75c959 share this published
/// head_link camera. The body pose has already undergone C R C^-1; local USD
/// coordinates still need C, and ROS camera +Z/-Y becomes Bevy -Z/+Y.
pub fn arena_head_camera(frame: &G1BodyFrame) -> Result<G1CameraSourceFrame, String> {
    frame.validate().map_err(|error| error.to_string())?;
    if frame.sim_time * 1_000_000_000.0 > u64::MAX as f64 {
        return Err("camera simulation time exceeds timestamp range".into());
    }
    let head = frame
        .bodies
        .iter()
        .find(|body| body.body == 19)
        .ok_or("head_link is absent")?;
    let world_from_head = Quat::from_array(head.rotation_xyzw);
    let source_to_engine = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let head_from_ros_camera = Quat::from_xyzw(-0.62721, 0.62721, -0.32651, 0.32651).normalize();
    let ros_from_bevy_camera = Quat::from_rotation_x(std::f32::consts::PI);
    let transform = Transform {
        translation: Vec3::from_array(head.translation)
            + world_from_head * (source_to_engine * Vec3::new(0.04485, 0.0, 0.35325)),
        rotation: world_from_head * source_to_engine * head_from_ros_camera * ros_from_bevy_camera,
        scale: Vec3::ONE,
    };
    let source = G1CameraSourceFrame {
        episode_id: frame.episode_id,
        source_ticks: [frame.source_tick; 2],
        interpolation_alpha: 0.0,
        sim_time_ns: (frame.sim_time * 1_000_000_000.0).round() as u64,
        source: CameraPoseSource::PhysicsBody,
        world_from_camera: transform,
    };
    source.validate()?;
    Ok(source)
}

/// The runtime writes this before G1CameraSystems::UpdatePose, using the same
/// source frame as robot visual transforms. None disables the ego camera.
#[derive(Resource, Default)]
pub struct G1CameraInput(pub Option<G1CameraSourceFrame>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G1CaptureStamp {
    pub episode_id: u64,
    pub capture_sequence: u64,
    pub render_frame: u64,
    pub source_ticks: [u64; 2],
    pub interpolation_alpha: f32,
    pub sim_time_ns: u64,
    pub source: CameraPoseSource,
    /// CPU scene snapshot acquisition; this is the observation's wall time.
    pub captured_at_unix_ms: u64,
    /// Time at which the copy command was encoded, not a GPU hardware timestamp.
    pub copy_encoded_at_unix_ms: u64,
    pub readback_completed_at_unix_ms: u64,
    pub camera_translation: [f32; 3],
    pub camera_rotation_xyzw: [f32; 4],
}

#[derive(Debug)]
pub struct G1CapturedRgb {
    pub stamp: G1CaptureStamp,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Default)]
struct CaptureSlot {
    episode_id: u64,
    next_sequence: u64,
    requested: bool,
    active: Option<(u64, u64)>,
    completed: Option<Result<G1CapturedRgb, String>>,
}

/// A single shared slot: a pending GPU copy or an unconsumed observation blocks
/// another request. Reset invalidates old output without allocating more copies.
#[derive(Resource, Clone, Default)]
pub struct G1CameraPort(Arc<Mutex<CaptureSlot>>);

impl G1CameraPort {
    pub fn request(&self) -> Result<u64, String> {
        let mut slot = self.0.lock().map_err(|_| "camera slot poisoned")?;
        if slot.requested || slot.active.is_some() || slot.completed.is_some() {
            return Err("camera has a pending or unconsumed observation".into());
        }
        slot.next_sequence = slot
            .next_sequence
            .checked_add(1)
            .ok_or("capture sequence exhausted")?;
        slot.requested = true;
        Ok(slot.next_sequence)
    }

    pub fn take(&self) -> Option<Result<G1CapturedRgb, String>> {
        self.0.lock().ok()?.completed.take()
    }

    pub fn reset(&self, episode_id: u64) -> Result<(), String> {
        let mut slot = self.0.lock().map_err(|_| "camera slot poisoned")?;
        if episode_id <= slot.episode_id {
            return Err("camera episode must increase".into());
        }
        slot.episode_id = episode_id;
        slot.requested = false;
        slot.completed = None;
        // A mapping already in flight is retained until its callback releases
        // it. No second buffer is allocated while that old operation completes.
        Ok(())
    }

    pub fn episode_id(&self) -> u64 {
        self.0.lock().map_or(0, |slot| slot.episode_id)
    }

    fn begin(&self, frame: &ExtractedFrame) -> Option<G1CaptureStamp> {
        let mut slot = self.0.lock().ok()?;
        if !slot.requested
            || slot.active.is_some()
            || slot.completed.is_some()
            || slot.episode_id != frame.source.episode_id
        {
            return None;
        }
        slot.requested = false;
        slot.active = Some((slot.episode_id, slot.next_sequence));
        Some(G1CaptureStamp {
            episode_id: slot.episode_id,
            capture_sequence: slot.next_sequence,
            render_frame: frame.render_frame,
            source_ticks: frame.source.source_ticks,
            interpolation_alpha: frame.source.interpolation_alpha,
            sim_time_ns: frame.source.sim_time_ns,
            source: frame.source.source,
            captured_at_unix_ms: frame.captured_at_unix_ms,
            copy_encoded_at_unix_ms: unix_ms(),
            readback_completed_at_unix_ms: 0,
            camera_translation: frame.source.world_from_camera.translation.to_array(),
            camera_rotation_xyzw: frame.source.world_from_camera.rotation.to_array(),
        })
    }

    fn finish(&self, stamp: G1CaptureStamp, result: Result<Vec<u8>, String>) {
        let Ok(mut slot) = self.0.lock() else {
            return;
        };
        if slot.active != Some((stamp.episode_id, stamp.capture_sequence)) {
            return;
        }
        slot.active = None;
        if slot.episode_id != stamp.episode_id {
            return;
        }
        slot.completed = Some(result.map(|rgb| G1CapturedRgb {
            stamp,
            width: EGO_WIDTH,
            height: EGO_HEIGHT,
            rgb,
        }));
    }
}

#[derive(Component)]
pub struct G1EgoCamera;

#[derive(Resource, Clone)]
pub struct G1EgoImage(pub Handle<Image>);

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum G1CameraSystems {
    UpdatePose,
}

/// Insert after DefaultPlugins. It creates a second target only; no main camera
/// resolution, render layer, station material, light or physical state is changed.
pub struct G1CameraPlugin;

impl Plugin for G1CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<G1CameraInput>()
            .init_resource::<G1CameraPort>()
            .add_systems(Startup, setup_camera)
            .add_systems(
                PostUpdate,
                update_pose
                    .in_set(G1CameraSystems::UpdatePose)
                    .before(TransformSystems::Propagate),
            );
        let port = app.world().resource::<G1CameraPort>().clone();
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .insert_resource(port)
                .init_resource::<ExtractedCapture>()
                .init_resource::<PendingMap>()
                .add_systems(ExtractSchedule, extract_frame)
                .add_systems(
                    RenderGraph,
                    encode_capture
                        .after(RenderGraphSystems::Render)
                        .before(RenderGraphSystems::Submit),
                )
                .add_systems(Render, map_capture.in_set(RenderSystems::Cleanup));
        }
    }
}

fn setup_camera(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_target_texture(
        EGO_WIDTH,
        EGO_HEIGHT,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let image = images.add(image);
    let calibration = G1CameraCalibration::default();
    commands.spawn((
        G1EgoCamera,
        Camera3d::default(),
        Camera {
            order: -1,
            is_active: false,
            ..default()
        },
        RenderTarget::Image(image.clone().into()),
        Msaa::Sample8,
        Tonemapping::None,
        Projection::Perspective(PerspectiveProjection {
            fov: calibration.vertical_fov_radians,
            near: calibration.near_m,
            far: calibration.far_m,
            aspect_ratio: EGO_WIDTH as f32 / EGO_HEIGHT as f32,
            ..default()
        }),
        Transform::IDENTITY,
    ));
    commands.insert_resource(G1EgoImage(image));
}

fn update_pose(
    input: Res<G1CameraInput>,
    mut cameras: Query<(&mut Camera, &mut Transform), With<G1EgoCamera>>,
) {
    for (mut camera, mut transform) in &mut cameras {
        if let Some(frame) = &input.0
            && frame.validate().is_ok()
        {
            *transform = frame.world_from_camera;
            camera.is_active = true;
        } else {
            camera.is_active = false;
        }
    }
}

#[derive(Clone)]
struct ExtractedFrame {
    source: G1CameraSourceFrame,
    render_frame: u64,
    captured_at_unix_ms: u64,
    camera_entity: Entity,
    image: Handle<Image>,
}

#[derive(Resource, Default)]
struct ExtractedCapture(Option<ExtractedFrame>);

fn extract_frame(
    mut extracted: ResMut<ExtractedCapture>,
    input: Extract<Res<G1CameraInput>>,
    image: Extract<Option<Res<G1EgoImage>>>,
    frame_count: Extract<Res<FrameCount>>,
    cameras: Extract<Query<(&RenderEntity, &Camera, &GlobalTransform), With<G1EgoCamera>>>,
) {
    extracted.0 = None;
    let (Some(source), Some(image)) = (&input.0, image.as_ref()) else {
        return;
    };
    let Ok((entity, camera, global)) = cameras.single() else {
        return;
    };
    if !camera.is_active || source.validate().is_err() {
        return;
    }
    let actual = global.compute_transform();
    if (actual.translation - source.world_from_camera.translation).length() > 0.00001
        || actual
            .rotation
            .angle_between(source.world_from_camera.rotation)
            .abs()
            > 0.0001
    {
        return;
    }
    extracted.0 = Some(ExtractedFrame {
        source: source.clone(),
        render_frame: u64::from(frame_count.0),
        captured_at_unix_ms: unix_ms(),
        camera_entity: entity.id(),
        image: image.0.clone(),
    });
}

#[derive(Resource, Default)]
struct PendingMap(Option<(Buffer, G1CaptureStamp)>);

fn encode_capture(
    mut context: RenderContext,
    extracted: Res<ExtractedCapture>,
    images: Res<RenderAssets<GpuImage>>,
    views: Query<&ViewTarget>,
    device: Res<RenderDevice>,
    port: Res<G1CameraPort>,
    mut pending: ResMut<PendingMap>,
) {
    if pending.0.is_some() {
        return;
    }
    let Some(frame) = &extracted.0 else {
        return;
    };
    if views.get(frame.camera_entity).is_err() {
        return;
    }
    let Some(image) = images.get(&frame.image) else {
        return;
    };
    let Some(stamp) = port.begin(frame) else {
        return;
    };
    let bytes_per_row = RenderDevice::align_copy_bytes_per_row((EGO_WIDTH * 4) as usize) as u32;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("g1_single_rgb_readback"),
        size: u64::from(bytes_per_row * EGO_HEIGHT),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context.command_encoder().copy_texture_to_buffer(
        image.texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(EGO_HEIGHT),
            },
        },
        Extent3d {
            width: EGO_WIDTH,
            height: EGO_HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    pending.0 = Some((buffer, stamp));
}

fn map_capture(mut pending: ResMut<PendingMap>, port: Res<G1CameraPort>) {
    let Some((buffer, mut stamp)) = pending.0.take() else {
        return;
    };
    let mapped = buffer.clone();
    let port = port.clone();
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        stamp.readback_completed_at_unix_ms = unix_ms();
        let rgb = match result {
            Ok(()) => {
                let bytes = mapped.slice(..).get_mapped_range();
                let result = unpack_rgb(&bytes, EGO_WIDTH, EGO_HEIGHT);
                drop(bytes);
                mapped.unmap();
                result
            }
            Err(error) => Err(format!("GPU camera readback failed: {error}")),
        };
        port.finish(stamp, rgb);
    });
}

fn unpack_rgb(bytes: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let pitch = RenderDevice::align_copy_bytes_per_row((width * 4) as usize);
    if bytes.len() != pitch * height as usize {
        return Err("unexpected RGBA readback size".into());
    }
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for row in bytes.chunks_exact(pitch) {
        for pixel in row[..width as usize * 4].chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
        }
    }
    Ok(rgb)
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |time| time.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(episode_id: u64) -> ExtractedFrame {
        ExtractedFrame {
            source: G1CameraSourceFrame {
                episode_id,
                source_ticks: [10, 10],
                interpolation_alpha: 0.0,
                sim_time_ns: 200_000_000,
                source: CameraPoseSource::PhysicsBody,
                world_from_camera: Transform::IDENTITY,
            },
            render_frame: 42,
            captured_at_unix_ms: 1234,
            camera_entity: Entity::PLACEHOLDER,
            image: Handle::default(),
        }
    }

    #[test]
    fn capture_slot_has_no_backlog_and_preserves_acquisition_stamp() {
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        port.request().unwrap();
        assert!(port.request().is_err());
        let mut stamp = port.begin(&frame(1)).unwrap();
        stamp.readback_completed_at_unix_ms = 2000;
        assert!(port.begin(&frame(1)).is_none());
        assert!(port.request().is_err());
        port.finish(stamp, Ok(vec![0; (EGO_WIDTH * EGO_HEIGHT * 3) as usize]));
        assert!(port.request().is_err());
        let observation = port.take().unwrap().unwrap();
        assert_eq!(observation.stamp.captured_at_unix_ms, 1234);
        assert_eq!(observation.stamp.readback_completed_at_unix_ms, 2000);
        assert_eq!(observation.stamp.source_ticks, [10, 10]);
        assert!(port.request().is_ok());
    }

    #[test]
    fn reset_drops_old_pixels_and_waits_for_old_gpu_mapping() {
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        port.request().unwrap();
        let stamp = port.begin(&frame(1)).unwrap();
        port.reset(2).unwrap();
        assert!(port.request().is_err());
        port.finish(stamp, Ok(Vec::new()));
        assert!(port.take().is_none());
        port.request().unwrap();
        assert!(port.begin(&frame(1)).is_none());
        assert!(port.begin(&frame(2)).is_some());
    }

    #[test]
    fn padded_rgba_rows_are_unpacked_without_changing_orientation() {
        let mut bytes = vec![0; 512];
        bytes[..8].copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255]);
        bytes[256..264].copy_from_slice(&[7, 8, 9, 255, 10, 11, 12, 255]);
        assert_eq!(
            unpack_rgb(&bytes, 2, 2).unwrap(),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
        assert!(unpack_rgb(&bytes[..511], 2, 2).is_err());
    }

    #[test]
    fn interpolation_requires_two_adjacent_physics_ticks() {
        let mut frame = frame(1).source;
        frame.interpolation_alpha = 0.5;
        assert!(frame.validate().is_err());
        frame.source_ticks = [10, 11];
        assert!(frame.validate().is_ok());
        frame.source_ticks = [10, 12];
        assert!(frame.validate().is_err());
    }

    #[test]
    fn source_head_camera_changes_basis_once_and_preserves_native_tick() {
        use robot_minigame::g1::definition::{G1BodyPose, USD_SHA256};
        let frame = G1BodyFrame {
            usd_sha256: USD_SHA256.into(),
            episode_id: 7,
            source_tick: 12,
            sim_time: 0.24,
            bodies: (0..53)
                .map(|body| G1BodyPose {
                    body,
                    translation: [1.0, 2.0, 3.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                })
                .collect(),
        };
        let camera = arena_head_camera(&frame).unwrap();
        assert!(
            (camera.world_from_camera.translation - Vec3::new(1.04485, 2.35325, 3.0)).length()
                < 0.00001
        );
        let offset = Quat::from_xyzw(-0.62721, 0.62721, -0.32651, 0.32651).normalize();
        let source_forward = offset * Vec3::Z;
        let expected_forward = Vec3::new(source_forward.x, source_forward.z, -source_forward.y);
        assert!(
            (camera.world_from_camera.rotation * Vec3::NEG_Z - expected_forward).length() < 0.00001
        );
        assert_eq!(camera.source_ticks, [12, 12]);
        assert_eq!(camera.sim_time_ns, 240_000_000);
    }
}
