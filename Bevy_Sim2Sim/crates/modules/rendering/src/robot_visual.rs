//! Read-only display of verified MicroDuck meshes and completed physics poses.
//!
//! This module owns no physics, forward kinematics or joint animation. Body
//! roots are independent world poses; their geom children use compiled locals.

use crate::{
    StationScene,
    material::{InkMaterial, StationEnamel, StationMaterial},
    robot_mesh::{
        RobotNormalPolicy, RobotVisualError, VerifiedRobotAppearance,
        build_robot_mesh_buffers_with_normal_policy, effective_geom_rgba,
    },
};
use bevy::{
    asset::RenderAssetUsages, camera::visibility::VisibilitySystems, prelude::*,
    transform::TransformSystems,
};
use robot_minigame::{
    basis::{source_to_engine_rotation, source_to_engine_vector},
    body_pose::RobotPoseFrame,
    definition::RobotDefinition,
};
use std::{collections::HashMap, sync::Arc};

/// Immutable verified model and appearance; insert before adding the plugin.
#[derive(Resource, Clone)]
pub struct RobotVisualModel {
    definition: Arc<RobotDefinition>,
    appearance: Arc<VerifiedRobotAppearance>,
    render_style: RobotRenderStyle,
}
/// Immutable robot-only line treatment; native base geometry remains complete.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct RobotRenderStyle {
    pub ink_enabled: bool,
    pub ink_pixels: f32,
    pub hatch_strength: f32,
    pub ink_casts_shadows: bool,
    pub base_casts_shadows: bool,
    pub base_receives_shadows: bool,
    pub normal_policy: RobotNormalPolicy,
}
impl Default for RobotRenderStyle {
    fn default() -> Self {
        Self {
            ink_enabled: true,
            ink_pixels: 0.72,
            hatch_strength: 0.075,
            ink_casts_shadows: true,
            base_casts_shadows: true,
            base_receives_shadows: true,
            normal_policy: RobotNormalPolicy::NativeUntextured,
        }
    }
}
impl RobotRenderStyle {
    fn validate(self) -> Result<(), RobotVisualError> {
        if !self.ink_pixels.is_finite()
            || !(0.0..=4.0).contains(&self.ink_pixels)
            || (self.ink_enabled && self.ink_pixels == 0.0)
            || !self.hatch_strength.is_finite()
            || !(0.0..=1.0).contains(&self.hatch_strength)
        {
            return Err(RobotVisualError("invalid robot render style".into()));
        }
        Ok(())
    }
}
impl RobotVisualModel {
    pub fn new(
        definition: Arc<RobotDefinition>,
        appearance: Arc<VerifiedRobotAppearance>,
    ) -> Result<Self, RobotVisualError> {
        if appearance.document().model_file_sha256 != definition.file_sha256() {
            return Err(RobotVisualError(
                "visual definition/appearance identity mismatch".into(),
            ));
        }
        Ok(Self {
            definition,
            appearance,
            render_style: RobotRenderStyle::default(),
        })
    }
    /// Configure display before startup; replacing it later requires rebuilding.
    pub fn with_render_style(mut self, style: RobotRenderStyle) -> Result<Self, RobotVisualError> {
        style.validate()?;
        self.render_style = style;
        Ok(self)
    }
    pub fn render_style(&self) -> RobotRenderStyle {
        self.render_style
    }
    pub fn definition(&self) -> &RobotDefinition {
        &self.definition
    }
    pub fn appearance(&self) -> &VerifiedRobotAppearance {
        &self.appearance
    }
    /// Native values remain available in appearance(). PBR mapping is a display
    /// approximation, including derived defaults, not MuJoCo pixel equivalence.
    pub fn material_mapping_report(&self) -> Vec<RobotMaterialMapping> {
        self.appearance
            .document()
            .visible_geom_ids
            .iter()
            .map(|id| material_mapping(self.appearance.document(), *id))
            .collect()
    }
}

/// Explicitly recorded visual mapping differences for each source geom.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RobotMaterialMapping {
    pub geom_id: usize,
    pub source_material_id: i32,
    pub effective_rgba: [f32; 4],
    pub engine_roughness: f32,
    pub engine_metallic: f32,
    pub roughness_derived_from_shininess: bool,
    pub metallic_derived_default: bool,
    pub engine_reflectance: f32,
    /// Native OpenGL specular is used as this style pass's PBR reflectance input;
    /// the two renderers' illumination/BRDF are not equivalent.
    pub native_specular_mapped_to_pbr_reflectance: bool,
    /// Source components are interpreted as sRGBA by the station style pass.
    pub source_rgba_interpreted_as_srgba: bool,
    /// Native planar-reflection rendering is not implemented by this PBR pass.
    pub native_planar_reflectance: f32,
    pub native_planar_reflection_supported: bool,
}
fn material_mapping(
    doc: &crate::robot_mesh::RobotAppearanceDocument,
    geom_id: usize,
) -> RobotMaterialMapping {
    let material_id = doc.geom_matid[geom_id];
    let material = (material_id >= 0).then(|| &doc.materials[material_id as usize]);
    let native_roughness = material.map_or(-1., |m| m.roughness);
    let native_metallic = material.map_or(-1., |m| m.metallic);
    RobotMaterialMapping {
        geom_id,
        source_material_id: material_id,
        effective_rgba: effective_geom_rgba(doc, geom_id),
        engine_roughness: if native_roughness >= 0. {
            native_roughness
        } else {
            // A documented style mapping; OpenGL shininess is not PBR roughness.
            (1. - material.map_or(0.5, |m| m.shininess)).clamp(0.04, 1.)
        },
        engine_metallic: native_metallic.max(0.),
        roughness_derived_from_shininess: native_roughness < 0.,
        metallic_derived_default: native_metallic < 0.,
        engine_reflectance: material.map_or(0.5, |m| m.specular),
        native_specular_mapped_to_pbr_reflectance: true,
        source_rgba_interpreted_as_srgba: true,
        native_planar_reflectance: material.map_or(0., |m| m.reflectance),
        native_planar_reflection_supported: false,
    }
}

#[derive(Clone)]
struct PoseBinding {
    model_file_sha256: String,
    body_count: usize,
    episode_id: u64,
    initial_global_step: u64,
    initial_episode_step: u64,
    handles: Vec<[u32; 2]>,
}
impl PoseBinding {
    fn new(definition: &RobotDefinition, frame: &RobotPoseFrame) -> Result<Self, RobotVisualError> {
        frame
            .validate(definition.file_sha256(), definition.model().counts.nbody)
            .map_err(|e| RobotVisualError(e.to_string()))?;
        let mut handles = vec![[0, 0]; definition.model().counts.nbody];
        for pose in &frame.poses {
            handles[pose.source_body_id] = pose.backend_handle;
        }
        Ok(Self {
            model_file_sha256: definition.file_sha256().into(),
            body_count: definition.model().counts.nbody,
            episode_id: frame.episode_id,
            initial_global_step: frame.global_step,
            initial_episode_step: frame.episode_step,
            handles,
        })
    }
    fn validate(
        &self,
        frame: &RobotPoseFrame,
        previous: Option<&RobotPoseFrame>,
    ) -> Result<(), RobotVisualError> {
        frame
            .validate(&self.model_file_sha256, self.body_count)
            .map_err(|e| RobotVisualError(e.to_string()))?;
        if frame.episode_id != self.episode_id
            || frame
                .poses
                .iter()
                .any(|p| self.handles[p.source_body_id] != p.backend_handle)
        {
            return Err(RobotVisualError(
                "robot episode/assembly changed; explicit rebind required".into(),
            ));
        }
        let global_delta = frame.global_step.checked_sub(self.initial_global_step);
        let episode_delta = frame.episode_step.checked_sub(self.initial_episode_step);
        if global_delta.is_none() || global_delta != episode_delta {
            return Err(RobotVisualError(
                "robot pose clock regressed or episode/global deltas differ".into(),
            ));
        }
        if let Some(previous) = previous {
            if frame.global_step < previous.global_step
                || frame.episode_step < previous.episode_step
            {
                return Err(RobotVisualError("robot pose frame regressed".into()));
            }
            if frame.global_step == previous.global_step && !same_poses(frame, previous) {
                return Err(RobotVisualError(
                    "conflicting robot poses at the same completed step".into(),
                ));
            }
        }
        Ok(())
    }
}
fn same_poses(a: &RobotPoseFrame, b: &RobotPoseFrame) -> bool {
    a.poses.iter().all(|pose| {
        b.poses
            .iter()
            .find(|p| p.source_body_id == pose.source_body_id)
            .is_some_and(|other| {
                pose.backend_handle == other.backend_handle
                    && pose.translation == other.translation
                    && pose.rotation_xyzw == other.rotation_xyzw
            })
    })
}

/// Runtime input is separate from immutable configuration. The application
/// publishes completed frames; the renderer never obtains mutable world access.
#[derive(Resource)]
pub struct RobotVisualInput {
    binding: PoseBinding,
    binding_revision: u64,
    latest: Option<Arc<RobotPoseFrame>>,
    rejected_rebind: Option<String>,
}
impl RobotVisualInput {
    /// The preview checks its complete initial snapshot before opening a window.
    /// Provenance still belongs to the external sole-world frame producer.
    pub(crate) fn validate_for_model(
        &self,
        definition: &RobotDefinition,
    ) -> Result<(), RobotVisualError> {
        if self.binding.model_file_sha256 != definition.file_sha256()
            || self.binding.body_count != definition.model().counts.nbody
        {
            return Err(RobotVisualError(
                "initial robot input/model binding mismatch".into(),
            ));
        }
        if let Some(error) = &self.rejected_rebind {
            return Err(RobotVisualError(error.clone()));
        }
        let frame = self.latest.as_deref().ok_or_else(|| {
            RobotVisualError("robot initialization preview requires a completed pose frame".into())
        })?;
        self.binding.validate(frame, None)
    }
    pub fn new(
        definition: &RobotDefinition,
        initial: Arc<RobotPoseFrame>,
    ) -> Result<Self, RobotVisualError> {
        Ok(Self {
            binding: PoseBinding::new(definition, &initial)?,
            binding_revision: 0,
            latest: Some(initial),
            rejected_rebind: None,
        })
    }
    /// Data is validated as a complete frame by the display synchronization system.
    pub fn publish(&mut self, frame: Arc<RobotPoseFrame>) {
        self.latest = Some(frame);
    }
    /// A missing snapshot must hide the previous robot rather than retain it.
    pub fn withdraw(&mut self) {
        self.latest = None;
    }
    /// Required after a cold reset/assembly rebuild, including reused indices
    /// with a new handle generation. An error invalidates the display input.
    pub fn rebind(
        &mut self,
        definition: &RobotDefinition,
        initial: Arc<RobotPoseFrame>,
    ) -> Result<(), RobotVisualError> {
        let candidate = PoseBinding::new(definition, &initial).and_then(|binding| {
            if binding.model_file_sha256 != self.binding.model_file_sha256
                || binding.body_count != self.binding.body_count
            {
                Err(RobotVisualError(
                    "rebind cannot replace immutable visual model; rebuild visuals explicitly"
                        .into(),
                ))
            } else {
                Ok(binding)
            }
        });
        match candidate {
            Ok(binding) => {
                let Some(revision) = self.binding_revision.checked_add(1) else {
                    let error = RobotVisualError("robot binding revision overflow".into());
                    self.latest = None;
                    self.rejected_rebind = Some(error.to_string());
                    return Err(error);
                };
                self.binding = binding;
                self.binding_revision = revision;
                self.latest = Some(initial);
                self.rejected_rebind = None;
                Ok(())
            }
            Err(error) => {
                self.latest = None;
                self.rejected_rebind = Some(error.to_string());
                Err(error)
            }
        }
    }
}

/// Status reports display readiness only, never controller or physics acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize)]
pub enum RobotVisualPhase {
    #[default]
    AwaitingModel,
    AwaitingFrame,
    Ready {
        episode_id: u64,
        global_step: u64,
        episode_step: u64,
    },
    Failed {
        reason: String,
    },
}
#[derive(Resource, Debug, Default)]
pub struct RobotVisualStatus {
    phase: RobotVisualPhase,
}
impl RobotVisualStatus {
    pub fn phase(&self) -> &RobotVisualPhase {
        &self.phase
    }
}

/// Stable source body identity for inspection; transforms come only from frames.
#[derive(Component, Debug)]
pub struct RobotVisualBody {
    pub source_body_id: usize,
    pub source_name: Option<String>,
}
#[derive(Resource, Default)]
struct RobotVisualState {
    built_identity: Option<(String, String, RobotRenderStyle)>,
    binding_revision: Option<u64>,
    accepted: Option<Arc<RobotPoseFrame>>,
    /// A cold rebind resets handle/episode matching, not the display's history.
    last_sequence: Option<(u64, u64, u64)>,
    terminal_error: Option<String>,
}

/// Applications publish poses before ApplyPoses in PostUpdate. This pass runs
/// before transform/visibility propagation and performs no interpolated motion.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum RobotVisualSystems {
    ApplyPoses,
}

/// Requires StationScene, RobotVisualModel and RobotVisualInput inserted by the
/// application. Add after StationVisualPlugin so its material plugins are shared.
pub struct RobotVisualPlugin;
impl Plugin for RobotVisualPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<MaterialPlugin<StationMaterial>>() {
            app.add_plugins(MaterialPlugin::<StationMaterial>::default());
        }
        if !app.is_plugin_added::<MaterialPlugin<InkMaterial>>() {
            app.add_plugins(MaterialPlugin::<InkMaterial>::default());
        }
        app.init_resource::<RobotVisualStatus>()
            .init_resource::<RobotVisualState>()
            .configure_sets(
                PostUpdate,
                RobotVisualSystems::ApplyPoses
                    .before(TransformSystems::Propagate)
                    .before(VisibilitySystems::VisibilityPropagate),
            )
            .add_systems(Startup, setup_robot_visuals)
            .add_systems(
                PostUpdate,
                apply_robot_poses.in_set(RobotVisualSystems::ApplyPoses),
            );
    }
}

fn compiled_geom_local(
    definition: &RobotDefinition,
    geom: usize,
) -> Result<Transform, RobotVisualError> {
    let fields = &definition.model().fields;
    let position = source_to_engine_vector(fields.geom_pos[geom].map(|v| v as f32));
    let rotation = source_to_engine_rotation(fields.geom_quat[geom].map(|v| v as f32))
        .map_err(|e| RobotVisualError(e.to_string()))?;
    Ok(Transform {
        translation: Vec3::from_array(position),
        rotation: Quat::from_array(rotation),
        scale: Vec3::ONE,
    })
}

fn setup_robot_visuals(
    mut commands: Commands,
    model: Option<Res<RobotVisualModel>>,
    station: Option<Res<StationScene>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StationMaterial>>,
    mut inks: ResMut<Assets<InkMaterial>>,
    mut state: ResMut<RobotVisualState>,
    mut status: ResMut<RobotVisualStatus>,
) {
    let Some(model) = model else {
        state.terminal_error =
            Some("verified RobotVisualModel was not inserted before startup".into());
        status.phase = RobotVisualPhase::Failed {
            reason: state.terminal_error.clone().unwrap(),
        };
        return;
    };
    let Some(station) = station else {
        state.terminal_error =
            Some("immutable StationScene is required for robot ink materials".into());
        status.phase = RobotVisualPhase::Failed {
            reason: state.terminal_error.clone().unwrap(),
        };
        return;
    };
    let buffers = match build_robot_mesh_buffers_with_normal_policy(
        model.definition(),
        model.appearance(),
        model.render_style().normal_policy,
    ) {
        Ok(buffers) => buffers,
        Err(error) => {
            state.terminal_error = Some(error.to_string());
            status.phase = RobotVisualPhase::Failed {
                reason: error.to_string(),
            };
            return;
        }
    };
    let mesh_handles: Vec<_> = buffers
        .into_iter()
        .map(|buffer| {
            let mut mesh = Mesh::new(
                bevy::mesh::PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, buffer.positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, buffer.normals);
            mesh.insert_indices(bevy::mesh::Indices::U32(buffer.indices));
            meshes.add(mesh)
        })
        .collect();
    let definition = model.definition();
    let source = definition.model();
    let doc = model.appearance().document();
    // Build every fallible local transform before spawning any robot entities.
    let locals: Result<Vec<_>, _> = doc
        .visible_geom_ids
        .iter()
        .map(|id| compiled_geom_local(definition, *id))
        .collect();
    let locals = match locals {
        Ok(locals) => locals,
        Err(error) => {
            state.terminal_error = Some(error.to_string());
            status.phase = RobotVisualPhase::Failed {
                reason: error.to_string(),
            };
            return;
        }
    };
    let mut body_entities = vec![None; source.counts.nbody];
    for (id, name) in source.names.body.iter().enumerate().skip(1) {
        body_entities[id] = Some(
            commands
                .spawn((
                    Name::new(format!(
                        "MicroDuck body {id} {}",
                        name.as_deref().unwrap_or("unnamed")
                    )),
                    RobotVisualBody {
                        source_body_id: id,
                        source_name: name.clone(),
                    },
                    Transform::IDENTITY,
                    Visibility::Hidden,
                ))
                .id(),
        );
    }
    let style = model.render_style();
    let ink = style
        .ink_enabled
        .then(|| inks.add(InkMaterial::default().with_pixels(style.ink_pixels)));
    // Cache the actual mesh/material/local-color combination; repeated motors
    // retain separate geom poses while sharing immutable GPU assets.
    let mut combinations = HashMap::new();
    for (id, local) in doc.visible_geom_ids.iter().zip(locals) {
        let mapping = material_mapping(doc, *id);
        let mesh_id = source.fields.geom_dataid[*id] as usize;
        let key = (
            mesh_id,
            mapping.source_material_id,
            mapping.effective_rgba.map(f32::to_bits),
        );
        let (mesh, material) = combinations.entry(key).or_insert_with(|| {
            let rgba = mapping.effective_rgba;
            let native = (mapping.source_material_id >= 0)
                .then(|| &doc.materials[mapping.source_material_id as usize]);
            let emission = native.map_or(0., |m| m.emission);
            let material = materials.add(StationMaterial {
                base: StandardMaterial {
                    base_color: Color::srgba(rgba[0], rgba[1], rgba[2], rgba[3]),
                    perceptual_roughness: mapping.engine_roughness,
                    metallic: mapping.engine_metallic,
                    reflectance: mapping.engine_reflectance,
                    emissive: LinearRgba::new(
                        rgba[0] * emission,
                        rgba[1] * emission,
                        rgba[2] * emission,
                        1.,
                    ),
                    alpha_mode: if rgba[3] < 1. {
                        AlphaMode::Blend
                    } else {
                        AlphaMode::Opaque
                    },
                    ..default()
                },
                extension: StationEnamel::new(0, &station.0.layout)
                    .with_hatch_strength(style.hatch_strength),
            });
            (mesh_handles[mesh_id].clone(), material)
        });
        let geom = commands
            .spawn((
                Name::new(format!(
                    "MicroDuck geom {id} {}",
                    source.names.mesh[mesh_id].as_deref().unwrap_or("unnamed")
                )),
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                local,
                ChildOf(body_entities[source.fields.geom_bodyid[*id]].unwrap()),
            ))
            .id();
        if !style.base_casts_shadows {
            commands.entity(geom).insert(bevy::light::NotShadowCaster);
        }
        if !style.base_receives_shadows {
            commands.entity(geom).insert(bevy::light::NotShadowReceiver);
        }
        if let Some(ink) = &ink {
            let outline = commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(ink.clone()),
                    Transform::IDENTITY,
                    ChildOf(geom),
                    bevy::light::NotShadowReceiver,
                ))
                .id();
            if !style.ink_casts_shadows {
                commands
                    .entity(outline)
                    .insert(bevy::light::NotShadowCaster);
            }
        }
    }
    state.built_identity = Some((
        definition.file_sha256().into(),
        model.appearance().file_sha256().into(),
        style,
    ));
    status.phase = RobotVisualPhase::AwaitingFrame;
}

fn apply_robot_poses(
    model: Option<Res<RobotVisualModel>>,
    input: Option<Res<RobotVisualInput>>,
    mut state: ResMut<RobotVisualState>,
    mut status: ResMut<RobotVisualStatus>,
    mut bodies: Query<(&RobotVisualBody, &mut Transform, &mut Visibility)>,
) {
    let candidate = (|| {
        let model = model
            .as_ref()
            .ok_or_else(|| RobotVisualError("verified robot model is absent".into()))?;
        let identity = (
            model.definition().file_sha256().to_string(),
            model.appearance().file_sha256().to_string(),
            model.render_style(),
        );
        if state.built_identity.as_ref() != Some(&identity) {
            return Err(RobotVisualError(
                "robot visual assets do not match the immutable model/appearance".into(),
            ));
        }
        let input = input.as_ref().ok_or_else(|| {
            RobotVisualError("robot poses require explicit initial binding".into())
        })?;
        if let Some(error) = &input.rejected_rebind {
            return Err(RobotVisualError(error.clone()));
        }
        if state.binding_revision != Some(input.binding_revision) {
            state.binding_revision = Some(input.binding_revision);
            state.accepted = None;
            state.terminal_error = None;
        }
        if let Some(error) = &state.terminal_error {
            return Err(RobotVisualError(error.clone()));
        }
        let Some(frame) = input.latest.as_ref() else {
            return Ok(None);
        };
        input.binding.validate(frame, state.accepted.as_deref())?;
        if let Some((global_step, episode_id, episode_step)) = state.last_sequence {
            if frame.global_step < global_step
                || frame.episode_id < episode_id
                || frame.episode_id == episode_id && frame.episode_step < episode_step
            {
                return Err(RobotVisualError(
                    "robot sequence regressed across explicit rebind".into(),
                ));
            }
        }
        if frame.model_file_sha256 != identity.0 {
            return Err(RobotVisualError(
                "bound poses refer to another visual model".into(),
            ));
        }
        // Preflight the entire ECS coverage before touching a single transform.
        let mut seen = vec![false; model.definition().model().counts.nbody];
        for (body, _, _) in &bodies {
            if body.source_body_id == 0
                || body.source_body_id >= seen.len()
                || seen[body.source_body_id]
            {
                return Err(RobotVisualError(
                    "display body coverage contains a duplicate/invalid source ID".into(),
                ));
            }
            seen[body.source_body_id] = true;
        }
        if seen.iter().skip(1).any(|seen| !seen) {
            return Err(RobotVisualError(
                "display is missing a source robot body".into(),
            ));
        }
        Ok(Some(frame.clone()))
    })();
    match candidate {
        Ok(Some(frame)) => {
            for (body, mut transform, mut visibility) in &mut bodies {
                let pose = frame
                    .poses
                    .iter()
                    .find(|p| p.source_body_id == body.source_body_id)
                    .unwrap();
                transform.translation = Vec3::from_array(pose.translation);
                transform.rotation = Quat::from_array(pose.rotation_xyzw);
                transform.scale = Vec3::ONE;
                *visibility = Visibility::Inherited;
            }
            status.phase = RobotVisualPhase::Ready {
                episode_id: frame.episode_id,
                global_step: frame.global_step,
                episode_step: frame.episode_step,
            };
            state.last_sequence = Some((frame.global_step, frame.episode_id, frame.episode_step));
            state.accepted = Some(frame);
        }
        Ok(None) => {
            for (_, _, mut visibility) in &mut bodies {
                *visibility = Visibility::Hidden;
            }
            status.phase = RobotVisualPhase::AwaitingFrame;
        }
        Err(error) => {
            for (_, _, mut visibility) in &mut bodies {
                *visibility = Visibility::Hidden;
            }
            state.terminal_error = Some(error.to_string());
            status.phase = RobotVisualPhase::Failed {
                reason: error.to_string(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::body_pose::RobotBodyPose;

    #[test]
    fn robot_style_requires_finite_bounded_values() {
        let baseline = RobotRenderStyle::default();
        baseline.validate().unwrap();
        RobotRenderStyle {
            ink_enabled: false,
            ink_pixels: 0.,
            hatch_strength: 0.,
            ..baseline
        }
        .validate()
        .unwrap();
        for invalid in [
            RobotRenderStyle {
                ink_pixels: f32::NAN,
                ..baseline
            },
            RobotRenderStyle {
                ink_pixels: f32::INFINITY,
                ..baseline
            },
            RobotRenderStyle {
                ink_pixels: -0.1,
                ..baseline
            },
            RobotRenderStyle {
                ink_pixels: 4.1,
                ..baseline
            },
            RobotRenderStyle {
                ink_pixels: 0.,
                ..baseline
            },
            RobotRenderStyle {
                hatch_strength: f32::NAN,
                ..baseline
            },
            RobotRenderStyle {
                hatch_strength: -0.1,
                ..baseline
            },
            RobotRenderStyle {
                hatch_strength: 1.1,
                ..baseline
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    // Synthetic boundary data only. These tests make no claim about robot
    // kinematics, a physical integration, material identity or a GPU picture.
    fn frame() -> RobotPoseFrame {
        RobotPoseFrame {
            model_file_sha256: "test_model".into(),
            episode_id: 7,
            global_step: 10,
            episode_step: 3,
            poses: vec![
                RobotBodyPose {
                    source_body_id: 1,
                    backend_handle: [4, 1],
                    translation: [1., 2., 3.],
                    rotation_xyzw: [0., 0., 0., 1.],
                },
                RobotBodyPose {
                    source_body_id: 2,
                    backend_handle: [8, 2],
                    translation: [4., 5., 6.],
                    rotation_xyzw: [0., 0., 0., 1.],
                },
            ],
        }
    }
    fn binding() -> PoseBinding {
        PoseBinding {
            model_file_sha256: "test_model".into(),
            body_count: 3,
            episode_id: 7,
            initial_global_step: 10,
            initial_episode_step: 3,
            handles: vec![[0, 0], [4, 1], [8, 2]],
        }
    }

    /// Opt-in real-export test: ROBOT_VISUAL_RECEIPT names the producer receipt.
    /// It covers exact native buffers and ECS synchronization, with explicitly
    /// synthetic display poses. It does not create a physics world or GPU app.
    #[test]
    #[ignore = "requires a real hash-bound compiled model and native appearance receipt"]
    fn actual_compiled_buffers_and_atomic_display_boundary() {
        let receipt_path = std::env::var("ROBOT_VISUAL_RECEIPT").expect("ROBOT_VISUAL_RECEIPT");
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
        for sidecar in receipt["sidecars"].as_array().unwrap() {
            let definition = Arc::new(
                RobotDefinition::load_json(
                    std::path::Path::new(sidecar["definition"]["path"].as_str().unwrap()),
                    sidecar["definition"]["sha256"].as_str().unwrap(),
                )
                .unwrap(),
            );
            let appearance = Arc::new(
                VerifiedRobotAppearance::load_json(
                    std::path::Path::new(sidecar["path"].as_str().unwrap()),
                    sidecar["sha256"].as_str().unwrap(),
                    &definition,
                )
                .unwrap(),
            );
            // This explicit diagnostic still verifies the source raw arrays.
            // Native effective display normals are checked by the C oracle.
            let buffers = build_robot_mesh_buffers_with_normal_policy(
                &definition,
                &appearance,
                RobotNormalPolicy::RawCompiledDiagnostic,
            )
            .unwrap();
            let fields = &definition.model().fields;
            let mut triangles = 0;
            for buffer in &buffers {
                let mesh_id = buffer.mesh_id;
                assert_eq!(buffer.indices.len(), fields.mesh_facenum[mesh_id] * 3);
                for face in 0..fields.mesh_facenum[mesh_id] {
                    for corner in 0..3 {
                        let source_vertex =
                            fields.mesh_face[fields.mesh_faceadr[mesh_id] + face][corner];
                        let source_normal =
                            appearance.document().meshes[mesh_id].face_normals[face][corner];
                        let display_index = buffer.indices[face * 3 + corner] as usize;
                        assert_eq!(
                            buffer.positions[display_index],
                            source_to_engine_vector(
                                fields.mesh_vert[fields.mesh_vertadr[mesh_id] + source_vertex]
                                    .map(|v| v as f32)
                            )
                        );
                        assert_eq!(
                            buffer.normals[display_index],
                            source_to_engine_vector(
                                appearance.document().meshes[mesh_id].normals[source_normal]
                            )
                        );
                    }
                }
                triangles += buffer.indices.len() / 3;
            }
            let model = RobotVisualModel::new(definition.clone(), appearance.clone()).unwrap();
            assert_eq!(
                model.material_mapping_report().len(),
                appearance.document().visible_geom_ids.len()
            );
            let nbody = definition.model().counts.nbody;
            let initial = Arc::new(RobotPoseFrame {
                model_file_sha256: definition.file_sha256().into(),
                episode_id: 1,
                global_step: 100,
                episode_step: 0,
                poses: (1..nbody)
                    .map(|id| RobotBodyPose {
                        source_body_id: id,
                        backend_handle: [id as u32, 3],
                        translation: [id as f32, 0., 0.],
                        rotation_xyzw: [0., 0., 0., 1.],
                    })
                    .collect(),
            });
            let mut preflight_input = RobotVisualInput::new(&definition, initial.clone()).unwrap();
            preflight_input.validate_for_model(&definition).unwrap();
            preflight_input.withdraw();
            assert!(preflight_input.validate_for_model(&definition).is_err());
            let mut invalid_initial = (*initial).clone();
            invalid_initial.poses.last_mut().unwrap().translation[0] = f32::INFINITY;
            preflight_input.publish(Arc::new(invalid_initial));
            assert!(preflight_input.validate_for_model(&definition).is_err());
            let mut app = App::new();
            app.insert_resource(model)
                .insert_resource(RobotVisualInput::new(&definition, initial.clone()).unwrap())
                .insert_resource(RobotVisualState {
                    built_identity: Some((
                        definition.file_sha256().into(),
                        appearance.file_sha256().into(),
                        RobotRenderStyle::default(),
                    )),
                    ..default()
                })
                .init_resource::<RobotVisualStatus>()
                .add_systems(PostUpdate, apply_robot_poses);
            for id in 1..nbody {
                app.world_mut().spawn((
                    RobotVisualBody {
                        source_body_id: id,
                        source_name: definition.model().names.body[id].clone(),
                    },
                    Transform::IDENTITY,
                    Visibility::Hidden,
                ));
            }
            app.update();
            assert!(matches!(
                app.world().resource::<RobotVisualStatus>().phase(),
                RobotVisualPhase::Ready { .. }
            ));
            let mut bad = (*initial).clone();
            bad.global_step += 1;
            bad.episode_step += 1;
            bad.poses[0].translation[0] += 99.;
            bad.poses.last_mut().unwrap().rotation_xyzw[0] = f32::NAN;
            app.world_mut()
                .resource_mut::<RobotVisualInput>()
                .publish(Arc::new(bad));
            app.update();
            assert!(matches!(
                app.world().resource::<RobotVisualStatus>().phase(),
                RobotVisualPhase::Failed { .. }
            ));
            let mut query = app
                .world_mut()
                .query::<(&RobotVisualBody, &Transform, &Visibility)>();
            for (body, transform, visibility) in query.iter(app.world()) {
                assert_eq!(
                    transform.translation,
                    Vec3::new(body.source_body_id as f32, 0., 0.)
                );
                assert_eq!(*visibility, Visibility::Hidden);
            }
            app.world_mut()
                .resource_mut::<RobotVisualInput>()
                .publish(initial.clone());
            app.update();
            assert!(matches!(
                app.world().resource::<RobotVisualStatus>().phase(),
                RobotVisualPhase::Failed { .. }
            ));
            let mut reset = (*initial).clone();
            reset.episode_id += 1;
            reset.global_step += 1;
            for pose in &mut reset.poses {
                pose.backend_handle[1] += 1;
            }
            app.world_mut()
                .resource_mut::<RobotVisualInput>()
                .rebind(&definition, Arc::new(reset))
                .unwrap();
            app.update();
            assert!(matches!(
                app.world().resource::<RobotVisualStatus>().phase(),
                RobotVisualPhase::Ready { episode_id: 2, .. }
            ));
            for (_, _, visibility) in query.iter(app.world()) {
                assert_eq!(*visibility, Visibility::Inherited);
            }
            let original_model = app.world().resource::<RobotVisualModel>().clone();
            let changed_model = original_model
                .with_render_style(RobotRenderStyle {
                    ink_enabled: false,
                    ..RobotRenderStyle::default()
                })
                .unwrap();
            app.insert_resource(changed_model);
            app.update();
            assert!(matches!(
                app.world().resource::<RobotVisualStatus>().phase(),
                RobotVisualPhase::Failed { .. }
            ));
            for (_, _, visibility) in query.iter(app.world()) {
                assert_eq!(*visibility, Visibility::Hidden);
            }
            // Reject bad group coverage/texture declarations even with otherwise
            // authentic native buffers. These mutations never enter runtime.
            let mut broken = appearance.document().clone();
            broken.visible_geom_ids.pop();
            assert!(crate::robot_mesh::validate_appearance(&broken, &definition).is_err());
            broken = appearance.document().clone();
            broken.texture_count = 1;
            assert!(crate::robot_mesh::validate_appearance(&broken, &definition).is_err());
            broken = appearance.document().clone();
            broken.meshes[0].face_normals[0][0] = usize::MAX;
            assert!(crate::robot_mesh::validate_appearance(&broken, &definition).is_err());
            println!(
                "actual model={} appearance={} body_count={} visible_geoms={} unique_meshes={} source_triangles={} atomic_update/hide/rebind=passed; poses synthetic, no physics/GPU",
                definition.file_sha256(),
                appearance.file_sha256(),
                nbody,
                appearance.document().visible_geom_ids.len(),
                buffers.len(),
                triangles
            );
        }
    }

    #[test]
    fn assembly_generations_and_episodes_cannot_silently_rebind() {
        let binding = binding();
        let mut changed = frame();
        changed.poses[1].backend_handle[1] += 1;
        assert!(
            binding
                .validate(&changed, None)
                .unwrap_err()
                .0
                .contains("explicit rebind")
        );
        changed = frame();
        changed.episode_id += 1;
        assert!(
            binding
                .validate(&changed, None)
                .unwrap_err()
                .0
                .contains("explicit rebind")
        );
    }

    #[test]
    fn unchanged_paused_frame_is_allowed_but_conflicting_or_regressed_frames_fail() {
        let binding = binding();
        let previous = frame();
        let mut same = previous.clone();
        same.poses.reverse();
        binding.validate(&same, Some(&previous)).unwrap();
        same.poses[0].translation[0] += 1.;
        assert!(
            binding
                .validate(&same, Some(&previous))
                .unwrap_err()
                .0
                .contains("conflicting")
        );
        let mut advanced = previous.clone();
        advanced.global_step += 5;
        advanced.episode_step += 5;
        binding.validate(&advanced, Some(&previous)).unwrap();
        assert!(
            binding
                .validate(&previous, Some(&advanced))
                .unwrap_err()
                .0
                .contains("regressed")
        );
        advanced.episode_step -= 1;
        assert!(
            binding
                .validate(&advanced, Some(&previous))
                .unwrap_err()
                .0
                .contains("deltas")
        );
    }

    #[test]
    fn whole_frame_rejects_identity_coverage_and_nonfinite_pose_errors() {
        let binding = binding();
        let mut bad = frame();
        bad.model_file_sha256 = "different_model".into();
        assert!(binding.validate(&bad, None).is_err());
        bad = frame();
        bad.poses.pop();
        assert!(binding.validate(&bad, None).is_err());
        bad = frame();
        bad.poses[1].source_body_id = 1;
        assert!(binding.validate(&bad, None).is_err());
        bad = frame();
        bad.poses[1].translation[2] = f32::NAN;
        assert!(binding.validate(&bad, None).is_err());
        bad = frame();
        bad.poses[1].rotation_xyzw = [0., 0., 0., 2.];
        assert!(binding.validate(&bad, None).is_err());
    }
}
