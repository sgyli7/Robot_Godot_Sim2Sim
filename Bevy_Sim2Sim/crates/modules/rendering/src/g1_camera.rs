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
use robot_minigame::g1::{contract::JOINT_COUNT, definition::G1BodyFrame};
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

/// The learned profiles always use ArenaEgo. AuxiliaryGripOverview is a
/// disclosed passive head-mounted sensor for traditional task perception.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum G1CameraMountProfile {
    #[default]
    ArenaEgo,
    AuxiliaryGripOverview,
    /// Declared near-table T1 sensor;150mm above original,+25degree downward view.
    StaticPlacementOverview,
}

impl G1CameraMountProfile {
    fn is_arena(&self) -> bool {
        *self == Self::ArenaEgo
    }
}

#[derive(Resource, Default)]
pub struct G1ActiveCameraMount(pub G1CameraMountProfile);

/// Native measurements from the completed physics boundary, in Arena's 43-joint
/// order. They are not inferred from the visible meshes or inverse kinematics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1CameraJointState {
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub root_rotation_wxyz: [f32; 4],
    pub root_angular_velocity_body: [f32; 3],
    pub root_velocity_source: [f32; 3],
}

/// A single immutable native snapshot drives the meshes, camera and RGB receipt.
/// Tick zero is read from the native initialized world and also requires real
/// joint/root measurements; a render-only initialization cannot substitute.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1CameraNativeState {
    pub body_frame: G1BodyFrame,
    pub measured_joints: G1CameraJointState,
}

impl G1CameraNativeState {
    pub fn validate(&self) -> Result<(), String> {
        self.body_frame
            .validate()
            .map_err(|error| error.to_string())?;
        let expected_time = self.body_frame.source_tick as f64 * 0.02;
        if (self.body_frame.sim_time - expected_time).abs() > 0.000001 {
            return Err("native camera snapshot is not on the 50 Hz grid".into());
        }
        let joints = &self.measured_joints;
        if joints.positions.len() != JOINT_COUNT
            || joints.velocities.len() != JOINT_COUNT
            || !joints
                .positions
                .iter()
                .chain(&joints.velocities)
                .chain(&joints.root_rotation_wxyz)
                .chain(&joints.root_angular_velocity_body)
                .chain(&joints.root_velocity_source)
                .all(|value| value.is_finite())
        {
            return Err("missing or invalid native camera joint measurements".into());
        }
        let root_rotation =
            robot_minigame::basis::source_to_engine_rotation(joints.root_rotation_wxyz)
                .map_err(|error| error.to_string())?;
        let root = self
            .body_frame
            .bodies
            .iter()
            .find(|body| body.body == 0)
            .ok_or("native root body is absent")?;
        if !rotations_agree(
            Quat::from_array(root_rotation),
            Quat::from_array(root.rotation_xyzw),
            0.0001,
        ) {
            return Err("native root sensor and displayed root disagree".into());
        }
        Ok(())
    }
}

/// The runtime supplies the worker's actual completed snapshot here. Installing
/// G1ObservationPlugin makes this the sole input for both robot meshes and ego
/// camera. It does not create, modify or integrate a physical world.
#[derive(Resource, Default)]
pub struct G1BodyObservationInput(pub Option<Arc<G1CameraNativeState>>);

/// Exact camera pose in Bevy world coordinates, with both interpolation sources
/// if present. No object transform or target location is accepted here.
#[derive(Debug, Clone)]
pub struct G1CameraSourceFrame {
    pub episode_id: u64,
    pub source_ticks: [u64; 2],
    pub interpolation_alpha: f32,
    pub sim_time_ns: u64,
    pub source: CameraPoseSource,
    pub mount_profile: G1CameraMountProfile,
    pub world_from_camera: Transform,
    /// Frozen at scene extraction, not looked up when GPU mapping completes.
    pub native_state: Option<Arc<G1CameraNativeState>>,
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
        if let Some(native) = &self.native_state {
            native.validate()?;
            let body = &native.body_frame;
            if self.source != CameraPoseSource::PhysicsBody
                || self.episode_id != body.episode_id
                || self.source_ticks != [body.source_tick; 2]
                || self.interpolation_alpha != 0.0
                || self.sim_time_ns != (body.sim_time * 1_000_000_000.0).round() as u64
            {
                return Err("camera and native state belong to different boundaries".into());
            }
            let expected = native_head_camera(body, self.mount_profile)?.world_from_camera;
            if (transform.translation - expected.translation).length() > 0.00001
                || !rotations_agree(transform.rotation, expected.rotation, 0.0001)
            {
                return Err("camera does not follow the native head pose".into());
            }
        }
        Ok(())
    }
}

// acos(dot(q,q)) magnifies f32 normalization roundoff near zero, and can reject
// the very same quaternion. The sign-invariant chord is 2*sin(angle/4); compare
// it directly while retaining the exact original angular tolerance.
fn rotations_agree(a: Quat, b: Quat, maximum_angle: f32) -> bool {
    let a = a.normalize().to_array().map(f64::from);
    let b = b.normalize().to_array().map(f64::from);
    let difference = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
    let antipodal = a.iter().zip(b).map(|(x, y)| (x + y).powi(2)).sum::<f64>();
    let bound = 2. * (f64::from(maximum_angle) * 0.25).sin();
    difference.min(antipodal) <= bound * bound
}

/// Arena static release 8b4a3a47 and mobile 7d75c959 share this published
/// head_link camera. The body pose has already undergone C R C^-1; local USD
/// coordinates still need C, and ROS camera +Z/-Y becomes Bevy -Z/+Y.
pub fn arena_head_camera(frame: &G1BodyFrame) -> Result<G1CameraSourceFrame, String> {
    native_head_camera(frame, G1CameraMountProfile::ArenaEgo)
}

pub fn native_head_camera(
    frame: &G1BodyFrame,
    mount_profile: G1CameraMountProfile,
) -> Result<G1CameraSourceFrame, String> {
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
    let (vertical_offset, optical_upward) = match mount_profile {
        G1CameraMountProfile::ArenaEgo => (0., Quat::IDENTITY),
        G1CameraMountProfile::AuxiliaryGripOverview => {
            (0.15, Quat::from_rotation_x(std::f32::consts::PI / 12.))
        }
        G1CameraMountProfile::StaticPlacementOverview => {
            (0.15, Quat::from_rotation_x(-25_f32.to_radians()))
        }
    };
    let ros_from_bevy_camera = Quat::from_rotation_x(std::f32::consts::PI);
    let transform = Transform {
        translation: Vec3::from_array(head.translation)
            + world_from_head
                * (source_to_engine * Vec3::new(0.04485, 0.0, 0.35325 + vertical_offset)),
        rotation: world_from_head
            * source_to_engine
            * head_from_ros_camera
            * optical_upward
            * ros_from_bevy_camera,
        scale: Vec3::ONE,
    };
    let source = G1CameraSourceFrame {
        episode_id: frame.episode_id,
        source_ticks: [frame.source_tick; 2],
        interpolation_alpha: 0.0,
        sim_time_ns: (frame.sim_time * 1_000_000_000.0).round() as u64,
        source: CameraPoseSource::PhysicsBody,
        mount_profile,
        world_from_camera: transform,
        native_state: None,
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
    #[serde(default, skip_serializing_if = "G1CameraMountProfile::is_arena")]
    pub mount_profile: G1CameraMountProfile,
    /// CPU scene snapshot acquisition; this is the observation's wall time.
    pub captured_at_unix_ms: u64,
    /// Time at which the copy command was encoded, not a GPU hardware timestamp.
    pub copy_encoded_at_unix_ms: u64,
    pub readback_completed_at_unix_ms: u64,
    pub camera_translation: [f32; 3],
    pub camera_rotation_xyzw: [f32; 4],
    /// Exact rendered body poses and, after integration, native measurements.
    /// Station fixtures have no native state and cannot qualify this pairing.
    pub native_state: Option<G1CameraNativeState>,
}

#[derive(Debug)]
pub struct G1CapturedRgb {
    pub stamp: G1CaptureStamp,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Read-only pipeline timing; never a camera observation or model input.
#[derive(Clone, Debug, Default, Serialize)]
pub struct G1CaptureProgress {
    pub episode_id: u64,
    pub capture_sequence: u64,
    pub phase: String,
    pub minimum_physics_tick: Option<u64>,
    pub last_considered_scene_tick: Option<u64>,
    pub old_scenes_skipped: u64,
    pub copy_scene_tick: Option<u64>,
    pub captured_at_unix_ms: Option<u64>,
    pub copy_encoded_at_unix_ms: Option<u64>,
    pub readback_completed_at_unix_ms: Option<u64>,
}

#[derive(Default)]
struct CaptureSlot {
    episode_id: u64,
    next_sequence: u64,
    requested: bool,
    minimum_physics_tick: Option<u64>,
    active: Option<(u64, u64)>,
    completed: Option<Result<G1CapturedRgb, String>>,
    progress: G1CaptureProgress,
}

/// A single shared slot: a pending GPU copy or an unconsumed observation blocks
/// another request. Reset invalidates old output without allocating more copies.
#[derive(Resource, Clone, Default)]
pub struct G1CameraPort(Arc<Mutex<CaptureSlot>>);

impl G1CameraPort {
    pub fn request(&self) -> Result<u64, String> {
        self.request_inner(None)
    }

    /// Wait for an actual current-episode physics scene at or after this tick.
    /// An older pipelined scene is not copied and its stamp is never rewritten.
    pub fn request_physics_frame(&self, episode_id: u64, minimum_tick: u64) -> Result<u64, String> {
        self.request_inner(Some((episode_id, minimum_tick)))
    }

    fn request_inner(&self, minimum: Option<(u64, u64)>) -> Result<u64, String> {
        let mut slot = self.0.lock().map_err(|_| "camera slot poisoned")?;
        if minimum.is_some_and(|(episode, tick)| {
            episode == 0 || episode != slot.episode_id || tick > u64::MAX / 20_000_000
        }) {
            return Err("camera minimum physics boundary is foreign or invalid".into());
        }
        if slot.requested || slot.active.is_some() || slot.completed.is_some() {
            return Err("camera has a pending or unconsumed observation".into());
        }
        slot.next_sequence = slot
            .next_sequence
            .checked_add(1)
            .ok_or("capture sequence exhausted")?;
        slot.requested = true;
        slot.minimum_physics_tick = minimum.map(|(_, tick)| tick);
        slot.progress = G1CaptureProgress {
            episode_id: slot.episode_id,
            capture_sequence: slot.next_sequence,
            minimum_physics_tick: slot.minimum_physics_tick,
            ..Default::default()
        };
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
        slot.minimum_physics_tick = None;
        slot.completed = None;
        slot.progress = G1CaptureProgress::default();
        // A mapping already in flight is retained until its callback releases
        // it. No second buffer is allocated while that old operation completes.
        Ok(())
    }

    pub fn episode_id(&self) -> u64 {
        self.0.lock().map_or(0, |slot| slot.episode_id)
    }

    /// Read-only diagnostic state; no queue, capture or world is changed.
    pub fn progress(&self) -> &'static str {
        let Ok(slot) = self.0.lock() else {
            return "poisoned";
        };
        if slot.completed.is_some() {
            "completed"
        } else if slot.active.is_some() {
            "gpu_mapping"
        } else if slot.requested {
            "awaiting_render_extraction"
        } else {
            "idle"
        }
    }

    pub fn capture_progress(&self) -> G1CaptureProgress {
        let Ok(slot) = self.0.lock() else {
            return G1CaptureProgress {
                phase: "poisoned".into(),
                ..Default::default()
            };
        };
        let phase = if slot.completed.is_some() {
            "completed"
        } else if slot.active.is_some() {
            "gpu_mapping"
        } else if slot.requested {
            "awaiting_render_extraction"
        } else {
            "idle"
        };
        G1CaptureProgress {
            phase: phase.into(),
            ..slot.progress.clone()
        }
    }

    fn needs_render(&self, source: &G1CameraSourceFrame) -> bool {
        let Ok(slot) = self.0.lock() else {
            return false;
        };
        slot.requested
            && slot.active.is_none()
            && slot.completed.is_none()
            && source.episode_id == slot.episode_id
            && slot.minimum_physics_tick.is_none_or(|tick| {
                source.source == CameraPoseSource::PhysicsBody
                    && source.source_ticks[0] >= tick
                    && source.source_ticks[0] == source.source_ticks[1]
            })
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
        slot.progress.last_considered_scene_tick = Some(frame.source.source_ticks[0]);
        if slot.minimum_physics_tick.is_some_and(|tick| {
            frame.source.source != CameraPoseSource::PhysicsBody
                || frame.source.source_ticks[0] < tick
                || frame.source.source_ticks[0] != frame.source.source_ticks[1]
        }) {
            // Retain the one request; do not allocate a GPU copy for old pixels.
            slot.progress.old_scenes_skipped = slot.progress.old_scenes_skipped.saturating_add(1);
            return None;
        }
        if frame.source.source == CameraPoseSource::PhysicsBody
            && frame.source.native_state.is_none()
        {
            slot.requested = false;
            slot.completed = Some(Err(
                "physical camera capture requires its native measurements".into(),
            ));
            return None;
        }
        slot.requested = false;
        slot.minimum_physics_tick = None;
        slot.active = Some((slot.episode_id, slot.next_sequence));
        let encoded_at = unix_ms();
        slot.progress.copy_scene_tick = Some(frame.source.source_ticks[0]);
        slot.progress.captured_at_unix_ms = Some(frame.captured_at_unix_ms);
        slot.progress.copy_encoded_at_unix_ms = Some(encoded_at);
        Some(G1CaptureStamp {
            episode_id: slot.episode_id,
            capture_sequence: slot.next_sequence,
            render_frame: frame.render_frame,
            source_ticks: frame.source.source_ticks,
            interpolation_alpha: frame.source.interpolation_alpha,
            sim_time_ns: frame.source.sim_time_ns,
            source: frame.source.source,
            mount_profile: frame.source.mount_profile,
            captured_at_unix_ms: frame.captured_at_unix_ms,
            copy_encoded_at_unix_ms: encoded_at,
            readback_completed_at_unix_ms: 0,
            camera_translation: frame.source.world_from_camera.translation.to_array(),
            camera_rotation_xyzw: frame.source.world_from_camera.rotation.to_array(),
            native_state: frame.source.native_state.as_deref().cloned(),
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
        slot.progress.readback_completed_at_unix_ms = Some(stamp.readback_completed_at_unix_ms);
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
    SyncObservation,
    UpdatePose,
}

/// Synchronize both display consumers from the same immutable native message.
/// Use alongside G1CameraPlugin and G1VisualPlugin for a physical body capture;
/// the fixture camera path deliberately does not install this plugin.
pub struct G1ObservationPlugin;

impl Plugin for G1ObservationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<G1BodyObservationInput>()
            .init_resource::<G1ActiveCameraMount>()
            .add_systems(
                PostUpdate,
                sync_body_observation
                    .in_set(G1CameraSystems::SyncObservation)
                    .before(G1CameraSystems::UpdatePose)
                    .before(crate::g1_visual::G1VisualSystems::SyncPose),
            );
    }
}

fn sync_body_observation(
    input: Res<G1BodyObservationInput>,
    mount: Res<G1ActiveCameraMount>,
    mut camera: ResMut<G1CameraInput>,
    mut visual: ResMut<crate::g1_visual::G1VisualInput>,
    mut status: ResMut<crate::g1_visual::G1VisualStatus>,
) {
    if !input.is_changed() && !mount.is_changed() {
        return;
    }
    camera.0 = None;
    visual.0 = None;
    let Some(native) = &input.0 else {
        return;
    };
    let result = native.validate().and_then(|()| {
        let mut source = native_head_camera(&native.body_frame, mount.0)?;
        source.native_state = Some(native.clone());
        source.validate()?;
        Ok(source)
    });
    match result {
        Ok(source) => {
            camera.0 = Some(source);
            visual.0 = Some(native.body_frame.clone());
        }
        Err(error) => status.error = Some(error),
    }
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
    port: Res<G1CameraPort>,
    mut cameras: Query<(&mut Camera, &mut Transform), With<G1EgoCamera>>,
) {
    for (mut camera, mut transform) in &mut cameras {
        if let Some(frame) = &input.0
            && frame.validate().is_ok()
        {
            *transform = frame.world_from_camera;
            // This image target is a one-shot sensor, not a second preview.
            // Keep its exact current pose, but draw only the requested scene.
            camera.is_active = port.needs_render(frame);
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
    visual: Extract<Option<Res<crate::g1_visual::G1VisualStatus>>>,
    task_input: Extract<Option<Res<crate::g1_task_visual::G1TaskVisualInput>>>,
    task_visual: Extract<Option<Res<crate::g1_task_visual::G1TaskVisualStatus>>>,
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
    if source.native_state.is_some()
        && visual.as_ref().is_none_or(|status| {
            status.error.is_some()
                || status.mesh_count != 49
                || status.episode_id != Some(source.episode_id)
                || status.source_tick != Some(source.source_ticks[0])
        })
    {
        return;
    }
    if let Some(task) = task_input.as_ref().and_then(|input| input.0.as_ref()) {
        if task.episode_id != source.episode_id
            || task.source_tick != source.source_ticks[0]
            || task_visual.as_ref().is_none_or(|status| {
                status.error.is_some()
                    || status.episode_id != Some(task.episode_id)
                    || status.source_tick != Some(task.source_tick)
                    || status.visible_objects != task.poses.len()
            })
        {
            return;
        }
    }
    let actual = global.compute_transform();
    if (actual.translation - source.world_from_camera.translation).length() > 0.00001
        || !rotations_agree(actual.rotation, source.world_from_camera.rotation, 0.0001)
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
                source: CameraPoseSource::StationFixture,
                mount_profile: G1CameraMountProfile::ArenaEgo,
                world_from_camera: Transform::IDENTITY,
                native_state: None,
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
        assert!(!port.needs_render(&frame(1).source));
        port.request().unwrap();
        assert!(port.needs_render(&frame(1).source));
        assert!(port.request().is_err());
        let mut stamp = port.begin(&frame(1)).unwrap();
        assert!(!port.needs_render(&frame(1).source));
        stamp.readback_completed_at_unix_ms = 2000;
        assert!(port.begin(&frame(1)).is_none());
        assert!(port.request().is_err());
        port.finish(stamp, Ok(vec![0; (EGO_WIDTH * EGO_HEIGHT * 3) as usize]));
        assert!(port.request().is_err());
        let observation = port.take().unwrap().unwrap();
        assert!(!port.needs_render(&frame(1).source));
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

    fn native(episode_id: u64, tick: u64) -> Arc<G1CameraNativeState> {
        use robot_minigame::g1::definition::{G1BodyPose, USD_SHA256};
        Arc::new(G1CameraNativeState {
            body_frame: G1BodyFrame {
                usd_sha256: USD_SHA256.into(),
                episode_id,
                source_tick: tick,
                sim_time: tick as f64 * 0.02,
                bodies: (0..53)
                    .map(|body| G1BodyPose {
                        body,
                        translation: [0., 0.78, 0.],
                        rotation_xyzw: [0., 0., 0., 1.],
                    })
                    .collect(),
            },
            measured_joints: G1CameraJointState {
                positions: vec![0.; JOINT_COUNT],
                velocities: vec![0.; JOINT_COUNT],
                root_rotation_wxyz: [1., 0., 0., 0.],
                root_angular_velocity_body: [0.; 3],
                root_velocity_source: [0.; 3],
            },
        })
    }

    #[test]
    fn rgb_receipt_keeps_extracted_native_measurements_after_input_advances() {
        let state = native(1, 10);
        let mut extracted = frame(1);
        extracted.source = arena_head_camera(&state.body_frame).unwrap();
        extracted.source.native_state = Some(state);
        extracted.source.validate().unwrap();
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        port.request().unwrap();
        let stamp = port.begin(&extracted).unwrap();
        extracted.source.native_state = Some(native(1, 11));
        port.finish(stamp, Ok(vec![0; (EGO_WIDTH * EGO_HEIGHT * 3) as usize]));
        let captured = port.take().unwrap().unwrap();
        let original = captured.stamp.native_state.unwrap();
        assert_eq!(original.body_frame.source_tick, 10);
        assert_eq!(original.measured_joints.positions.len(), 43);
        assert_eq!(captured.stamp.source_ticks, [10, 10]);
    }

    #[test]
    fn minimum_physics_request_skips_old_scene_without_copy_or_restamp() {
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        assert!(port.request_physics_frame(2, 12).is_err());
        assert!(port.request_physics_frame(1, u64::MAX).is_err());
        port.request_physics_frame(1, 12).unwrap();
        assert!(port.begin(&frame(1)).is_none());
        let mut old = frame(1);
        let state = native(1, 10);
        old.source = arena_head_camera(&state.body_frame).unwrap();
        old.source.native_state = Some(state);
        assert!(!port.needs_render(&old.source));
        assert!(port.begin(&old).is_none());
        assert_eq!(port.progress(), "awaiting_render_extraction");
        let timing = port.capture_progress();
        assert_eq!(timing.minimum_physics_tick, Some(12));
        assert_eq!(timing.old_scenes_skipped, 2);
        assert_eq!(timing.last_considered_scene_tick, Some(10));
        assert!(timing.copy_scene_tick.is_none());
        assert!(port.request().is_err());
        let mut current = frame(1);
        let state = native(1, 13);
        current.source = arena_head_camera(&state.body_frame).unwrap();
        current.source.native_state = Some(state);
        assert!(port.needs_render(&current.source));
        let stamp = port.begin(&current).unwrap();
        assert!(!port.needs_render(&current.source));
        assert_eq!(stamp.source_ticks, [13, 13]);
        assert_eq!(stamp.sim_time_ns, 260_000_000);
        assert_eq!(stamp.captured_at_unix_ms, 1234);
        assert_eq!(
            stamp.native_state.as_ref().unwrap().body_frame.source_tick,
            13
        );
        assert_eq!(port.progress(), "gpu_mapping");
        let timing = port.capture_progress();
        assert_eq!(timing.copy_scene_tick, Some(13));
        assert_eq!(timing.captured_at_unix_ms, Some(1234));
        assert!(timing.readback_completed_at_unix_ms.is_none());
        port.finish(stamp, Ok(Vec::new()));
        assert_eq!(port.take().unwrap().unwrap().stamp.source_ticks, [13, 13]);
        assert_eq!(port.progress(), "idle");
    }

    #[test]
    fn reset_invalidates_minimum_frame_request_without_leaking_its_boundary() {
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        port.request_physics_frame(1, 100).unwrap();
        port.reset(2).unwrap();
        assert!(port.begin(&frame(1)).is_none());
        assert!(port.request_physics_frame(1, 0).is_err());
        assert!(port.capture_progress().minimum_physics_tick.is_none());
        port.request().unwrap();
        assert!(port.begin(&frame(2)).is_some());
    }

    #[test]
    fn native_camera_rotation_accepts_roundoff_and_sign_but_rejects_real_movement() {
        let state = native(1, 10);
        let mut source = arena_head_camera(&state.body_frame).unwrap();
        source.native_state = Some(state);
        source.validate().unwrap();
        source.world_from_camera.rotation = -source.world_from_camera.rotation;
        source.validate().unwrap();
        source.world_from_camera.rotation =
            Quat::from_rotation_y(0.001) * source.world_from_camera.rotation;
        assert!(source.validate().is_err());
    }

    #[test]
    fn physical_capture_rejects_pose_only_and_mismatched_native_boundary() {
        let state = native(1, 10);
        let mut extracted = frame(1);
        extracted.source = arena_head_camera(&state.body_frame).unwrap();
        let port = G1CameraPort::default();
        port.reset(1).unwrap();
        port.request().unwrap();
        assert!(port.begin(&extracted).is_none());
        assert!(port.take().unwrap().is_err());
        extracted.source.native_state = Some(native(1, 11));
        assert!(extracted.source.validate().is_err());
        extracted.source.native_state = Some(state);
        extracted.source.world_from_camera.translation.x += 0.05;
        assert!(extracted.source.validate().is_err());
    }

    #[test]
    fn one_native_input_drives_both_mesh_and_head_camera_without_integration() {
        let state = native(1, 0);
        let mut app = App::new();
        app.init_resource::<G1CameraInput>()
            .init_resource::<crate::g1_visual::G1VisualInput>()
            .init_resource::<crate::g1_visual::G1VisualStatus>()
            .add_plugins(G1ObservationPlugin);
        app.world_mut().resource_mut::<G1BodyObservationInput>().0 = Some(state);
        app.update();
        let camera = app.world().resource::<G1CameraInput>().0.as_ref().unwrap();
        let visual = app
            .world()
            .resource::<crate::g1_visual::G1VisualInput>()
            .0
            .as_ref()
            .unwrap();
        assert_eq!(camera.source_ticks, [0, 0]);
        assert_eq!(camera.episode_id, visual.episode_id);
        assert_eq!(
            camera
                .native_state
                .as_ref()
                .unwrap()
                .body_frame
                .bodies
                .len(),
            53
        );
        app.world_mut().resource_mut::<G1BodyObservationInput>().0 = None;
        app.update();
        assert!(app.world().resource::<G1CameraInput>().0.is_none());
        assert!(
            app.world()
                .resource::<crate::g1_visual::G1VisualInput>()
                .0
                .is_none()
        );
    }

    #[test]
    fn placement_mount_retains_native_identity_and_rejects_foreign_mount_label() {
        let state = native(1, 0);
        let original = arena_head_camera(&state.body_frame).unwrap();
        let mut placement = native_head_camera(
            &state.body_frame, G1CameraMountProfile::StaticPlacementOverview,
        ).unwrap();
        placement.native_state = Some(state);
        placement.validate().unwrap();
        assert_eq!(placement.source_ticks, original.source_ticks);
        assert_eq!(placement.sim_time_ns, original.sim_time_ns);
        assert!(((placement.world_from_camera.translation - original.world_from_camera.translation)
            .length() - 0.15).abs() < 1e-6);
        assert!((placement.world_from_camera.rotation.angle_between(original.world_from_camera.rotation)
            - 25_f32.to_radians()).abs() < 1e-5);
        placement.mount_profile = G1CameraMountProfile::AuxiliaryGripOverview;
        assert!(placement.validate().is_err());
    }

    #[test]
    fn auxiliary_mount_changes_only_camera_and_retains_the_same_native_boundary() {
        let state = native(1, 0);
        let original = arena_head_camera(&state.body_frame).unwrap();
        let mut auxiliary = native_head_camera(
            &state.body_frame,
            G1CameraMountProfile::AuxiliaryGripOverview,
        )
        .unwrap();
        auxiliary.native_state = Some(state.clone());
        auxiliary.validate().unwrap();
        assert_eq!(auxiliary.source_ticks, original.source_ticks);
        assert_eq!(auxiliary.sim_time_ns, original.sim_time_ns);
        assert!(
            ((auxiliary.world_from_camera.translation - original.world_from_camera.translation)
                .length()
                - 0.15)
                .abs()
                < 1e-6
        );
        let mut mislabeled = auxiliary.clone();
        mislabeled.mount_profile = G1CameraMountProfile::ArenaEgo;
        assert!(mislabeled.validate().is_err());
        let mut app = App::new();
        app.init_resource::<G1CameraInput>()
            .init_resource::<crate::g1_visual::G1VisualInput>()
            .init_resource::<crate::g1_visual::G1VisualStatus>()
            .add_plugins(G1ObservationPlugin);
        app.world_mut().resource_mut::<G1BodyObservationInput>().0 = Some(state);
        app.update();
        app.world_mut().resource_mut::<G1ActiveCameraMount>().0 =
            G1CameraMountProfile::AuxiliaryGripOverview;
        app.update();
        let camera = app.world().resource::<G1CameraInput>().0.as_ref().unwrap();
        assert_eq!(
            camera.mount_profile,
            G1CameraMountProfile::AuxiliaryGripOverview
        );
        assert_eq!(camera.source_ticks, [0, 0]);
        assert_eq!(
            app.world()
                .resource::<crate::g1_visual::G1VisualInput>()
                .0
                .as_ref()
                .unwrap()
                .source_tick,
            0
        );
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
