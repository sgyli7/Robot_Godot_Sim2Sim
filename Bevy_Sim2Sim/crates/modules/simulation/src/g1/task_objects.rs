//! Byte-bound original task convex parts and mass in the owner's Rapier world.
//!
//! Handles and mutation remain private. Render/acceptance consumers receive
//! immutable completed poses; decision clients receive images, never these truth
//! samples. Source cooking is not a claim of full physics or task qualification.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use rapier3d::{
    math::{Pose, Rotation, Vector},
    prelude::*,
};
use robot_minigame::{
    RobotError,
    basis::{engine_to_source_vector, source_to_engine_rotation, source_to_engine_vector},
    g1::{definition::SourcePose, policy::bound_bytes},
};
use serde::{Deserialize, Serialize};

use super::task_background::{
    T2SourceBackground, T2SourceBackgroundConfig, T2SourceBackgroundSample,
};
use super::task_shelf::{T1SourceShelf, T1SourceShelfConfig, T1SourceShelfSample};
use crate::SimulationWorld;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum TaskObjectKind {
    #[serde(rename = "t1_apple")]
    Apple,
    #[serde(rename = "t1_plate")]
    Plate,
    #[serde(rename = "t2_box")]
    BrownBox,
    #[serde(rename = "t2_bin")]
    BlueBin,
}

impl TaskObjectKind {
    pub fn source_sha256(self) -> &'static str {
        match self {
            Self::Apple => "2e0e0462c4345b340e1c6040c11abe1818437ced1db228a53ec0944001bb46d8",
            Self::Plate => "286238c8f957e3267fa21a0003b320868130f903a4b1d991a4ffeb49faaea5a8",
            Self::BrownBox => "50dc139612086b9483770a1abc17dc600445aa4f85323d74fa97069f7c2eb4ed",
            Self::BlueBin => "b9ffec2e70fd009863a3fa8bd699aca808403522eafb259d5638135e63506999",
        }
    }
    fn source_scale(self) -> [f64; 3] {
        match self {
            Self::Apple => [0.009; 3],
            Self::Plate => [0.5; 3],
            Self::BrownBox => [1.; 3],
            Self::BlueBin => [4., 2., 1.],
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Material {
    static_friction: f32,
    dynamic_friction: f32,
    restitution: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConvexPart {
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Object {
    kind: TaskObjectKind,
    usd_sha256: String,
    source_scale_override: [f64; 3],
    source_ccd_enabled: bool,
    mass_kg: f64,
    center_of_mass: [f64; 3],
    principal_inertia: [f64; 3],
    principal_axes_wxyz: [f64; 4],
    material: Material,
    convex_parts: Vec<ConvexPart>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    units: String,
    source_query_sha256: String,
    exporter_sha256: String,
    source_runtime_build: String,
    physics_parity_qualified: bool,
    objects: Vec<Object>,
}

/// Immutable data; no runtime material, shape, mass or placement repair exists.
pub struct TaskObjectsDefinition {
    document: Document,
    file_sha256: String,
}

impl TaskObjectsDefinition {
    /// Published immutable collision vertices only; no poses, contacts or world.
    #[cfg(feature = "g1_constraint_diagnostic")]
    pub(super) fn static_placement_vertices(&self) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let vertices = |kind| self.document.objects.iter().filter(|o| o.kind == kind)
            .flat_map(|o| o.convex_parts.iter()).flat_map(|c| c.points.iter().copied()).collect();
        (vertices(TaskObjectKind::Apple), vertices(TaskObjectKind::Plate))
    }

    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self, RobotError> {
        if std::fs::metadata(path)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > 32 * 1024 * 1024
        {
            return Err(invalid("task object definition exceeds 32 MiB"));
        }
        let bytes = bound_bytes(path, expected_sha256)?;
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(invalid("task object definition exceeds 32 MiB"));
        }
        let document: Document =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        Self::validate(&document)?;
        Ok(Self {
            document,
            file_sha256: expected_sha256.into(),
        })
    }

    fn validate(document: &Document) -> Result<(), RobotError> {
        if document.schema != "native_g1_task_objects_v2"
            || document.units != "metres_kilograms_radians_z_up"
            || document.physics_parity_qualified
            || document.source_runtime_build.is_empty()
            || !sha256_text(&document.source_query_sha256)
            || !sha256_text(&document.exporter_sha256)
            || document.objects.len() != 4
        {
            return Err(invalid(
                "invalid task object identity or qualification claim",
            ));
        }
        let mut kinds = HashSet::new();
        for object in &document.objects {
            if !kinds.insert(object.kind)
                || object.usd_sha256 != object.kind.source_sha256()
                || object.source_scale_override != object.kind.source_scale()
                || object.source_ccd_enabled != (object.kind == TaskObjectKind::Plate)
                || !finite_f32(&[object.mass_kg])
                || object.mass_kg <= 0.
                || !finite_f32(&object.center_of_mass)
                || !finite_f32(&object.principal_inertia)
                || object.principal_inertia.iter().any(|x| *x <= 0.)
                || !finite_f32(&object.principal_axes_wxyz)
                || object.convex_parts.is_empty()
                || object.convex_parts.len() > 1024
                || !object.material.static_friction.is_finite()
                || !object.material.dynamic_friction.is_finite()
                || object.material.static_friction < 0.
                || object.material.dynamic_friction < 0.
                || !object.material.restitution.is_finite()
                || !(0.0..=1.0).contains(&object.material.restitution)
            {
                return Err(invalid("invalid task object geometry/material/mass"));
            }
            source_to_engine_rotation(object.principal_axes_wxyz.map(|x| x as f32))?;
            for part in &object.convex_parts {
                if !(4..=512).contains(&part.points.len())
                    || part.triangles.len() < 4
                    || part.triangles.len() > 2048
                    || part.points.iter().any(|p| !finite_f32(p))
                    || part.triangles.iter().any(|t| {
                        t.iter().any(|i| *i as usize >= part.points.len())
                            || t[0] == t[1]
                            || t[1] == t[2]
                            || t[2] == t[0]
                    })
                {
                    return Err(invalid("invalid original cooked convex part"));
                }
            }
        }
        Ok(())
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    fn object(&self, kind: TaskObjectKind) -> &Object {
        self.document
            .objects
            .iter()
            .find(|object| object.kind == kind)
            .expect("validated four distinct original kinds")
    }
}

/// Placement is performed once before the first integration; there is no set-pose API.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskObjectPlacement {
    pub kind: TaskObjectKind,
    pub root_pose: SourcePose,
}

/// Startup input, kept distinct from image/self-state model observations.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskObjectSceneConfig {
    pub definition: PathBuf,
    pub definition_sha256: String,
    pub placements: Vec<TaskObjectPlacement>,
    #[serde(default)]
    pub source_t1_shelf: Option<T1SourceShelfConfig>,
    #[serde(default)]
    pub source_t2_background: Option<T2SourceBackgroundConfig>,
}

impl TaskObjectSceneConfig {
    pub(super) fn load_in_owner_world(
        &self,
        world: &mut SimulationWorld,
        supported: &[TaskObjectKind],
    ) -> Result<TaskObjectScene, RobotError> {
        if self.placements.len() != supported.len()
            || supported
                .iter()
                .any(|kind| self.placements.iter().filter(|p| p.kind == *kind).count() != 1)
        {
            return Err(invalid(
                "task objects do not match the owner's task profile",
            ));
        }
        let definition = TaskObjectsDefinition::load(&self.definition, &self.definition_sha256)?;
        if let Some(shelf) = &self.source_t1_shelf {
            if supported != [TaskObjectKind::Apple, TaskObjectKind::Plate] {
                return Err(invalid(
                    "original T1 shelf is incompatible with this task profile",
                ));
            }
            shelf.validate()?;
        }
        let background = if let Some(config) = &self.source_t2_background {
            if supported != [TaskObjectKind::BrownBox, TaskObjectKind::BlueBin]
                || self.source_t1_shelf.is_some()
            {
                return Err(invalid(
                    "original T2 background is incompatible with this task profile",
                ));
            }
            Some(config.prepare(world)?)
        } else {
            None
        };
        let mut scene =
            TaskObjectScene::insert_in_owner_world(world, &definition, &self.placements)?;
        if let Some(config) = &self.source_t1_shelf {
            let shelf = T1SourceShelf::insert(world, config)?;
            scene.initial_robot_shelf_overlap = shelf.initial_robot_overlap(world)?;
            scene.source_t1_shelf = Some(shelf);
        }
        scene.source_t2_background = background.map(|prepared| prepared.insert(world));
        Ok(scene)
    }
}

struct Instance {
    kind: TaskObjectKind,
    body: RigidBodyHandle,
    collider: ColliderHandle,
    convex_parts: usize,
}

/// Opaque handles stay with the physics owner that constructed this scene.
pub struct TaskObjectScene {
    definition_sha256: String,
    instances: Vec<Instance>,
    source_t1_shelf: Option<T1SourceShelf>,
    source_t2_background: Option<T2SourceBackground>,
    initial_robot_shelf_overlap: Vec<serde_json::Value>,
}

#[derive(Clone, Copy)]
enum ConvexRepresentation {
    CopiedFaces,
    #[cfg(test)]
    ReconstructedHull,
}

/// Aggregated last-solve contact evidence, never a model observation. A pair
/// can include speculative contacts: distance and impulse must be assessed,
/// rather than treating its presence as proof of a grasp or support.
#[derive(Clone, Debug, Serialize)]
pub struct TaskObjectContactSample {
    pub other_body_handle: Option<[u32; 2]>,
    pub other_robot_body_index: Option<usize>,
    pub other_task_kind: Option<TaskObjectKind>,
    pub other_background_collider_path: Option<String>,
    pub manifold_count: usize,
    pub solver_points: usize,
    pub min_solver_distance_m: Option<f32>,
    /// Read-only shape query at the published post-integration body poses.
    /// Unlike cached solver distances this belongs to this exact sample Tick.
    /// Zero means intersection/touch; None means the query was unsupported.
    pub geometric_distance_after_step_m: Option<f32>,
    /// Legacy sum over all cached manifold points, including points no longer
    /// selected by the solver. It must not establish present support alone.
    pub normal_impulse_n_s: f32,
    /// Legacy cached-normal vector; cached points/normals may be stale.
    /// Tangential/friction impulse is excluded; never a controller input.
    pub normal_impulse_on_object_source: [f32; 3],
    /// Sum over current solver-contact identities only. None means the active
    /// list is unavailable; it is not substituted with cached legacy impulses.
    #[cfg(feature = "g1_constraint_diagnostic")]
    pub active_solver_normal_impulse_n_s: Option<f32>,
    /// Active impulses along stored solver-manifold normals, in source axes.
    /// This is not a post-step surface-normal query.
    #[cfg(feature = "g1_constraint_diagnostic")]
    pub active_solver_normal_impulse_on_object_source: Option<[f32; 3]>,
    /// Diagnostic cache/impulse records. Solver-basis friction components are
    /// not world-space forces, and cached anchors are not fresh shape queries.
    #[cfg(feature = "g1_contact_point_diagnostic")]
    pub diagnostic_solver_contacts: Vec<TaskObjectSolverContactSample>,
}

/// Read-only evidence from active solver-contact identities. These records
/// never enter robot measurements, action admission or task control.
#[cfg(feature = "g1_contact_point_diagnostic")]
#[derive(Clone, Debug, Serialize)]
pub struct TaskObjectSolverContactSample {
    pub manifold_index: usize,
    pub point_index: usize,
    pub cached_contact_new_bit: bool,
    pub normal_impulse_n_s: f32,
    /// None for the simplified rigid-body solver, whose pointwise actual
    /// tangent impulses are not written back. Warmstart data is not substituted.
    pub tangent_impulse_solver_basis_n_s: Option<[f32; 2]>,
    pub effective_friction: f32,
    pub cached_manifold_normal_source: [f32; 3],
    /// Body-CoM-relative world lever arm frozen at the last full pair update.
    pub cached_solver_lever_arm_on_object_source_m: [f32; 3],
    /// Cached body-local anchors resolved at this published body's pose.
    pub cached_anchor_on_object_source_m: [f32; 3],
    pub cached_anchor_on_other_source_m: [f32; 3],
}

/// A render/independent-acceptance sample, never a model observation field.
#[derive(Clone, Debug, Serialize)]
pub struct TaskObjectSample {
    pub kind: TaskObjectKind,
    pub translation_engine: [f32; 3],
    pub rotation_engine_xyzw: [f32; 4],
    pub position_source: [f32; 3],
    pub linear_velocity_source: [f32; 3],
    pub angular_velocity_source: [f32; 3],
    pub mass_kg: f32,
    pub convex_parts: usize,
    pub dynamic: bool,
    pub active_contact_pairs: usize,
    pub source_shelf_contact: bool,
    pub last_solve_contacts: Vec<TaskObjectContactSample>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TaskObjectFrame {
    pub definition_sha256: String,
    pub episode_id: u64,
    pub source_tick: u64,
    pub sim_time: f64,
    pub contact_clustering: bool,
    pub contact_recycling: bool,
    pub world_counts: crate::WorldCounts,
    pub source_t1_shelf: Option<T1SourceShelfSample>,
    pub source_t2_background: Option<T2SourceBackgroundSample>,
    pub initial_robot_shelf_overlap: Vec<serde_json::Value>,
    pub objects: Vec<TaskObjectSample>,
}

impl TaskObjectScene {
    pub(super) fn has_source_t2_background(&self) -> bool {
        self.source_t2_background.is_some()
    }

    /// The ignored diagnostic may isolate this one contact pair class at startup.
    #[cfg(test)]
    pub(super) fn diagnostic_shelf_collider(&self) -> Option<ColliderHandle> {
        self.source_t1_shelf.as_ref().map(|shelf| shelf.collider)
    }

    /// Owner-only startup. All shapes are prepared before adding any body, so a
    /// failed geometry/placement check cannot leave a partially assembled scene.
    pub fn insert_in_owner_world(
        world: &mut SimulationWorld,
        definition: &TaskObjectsDefinition,
        placements: &[TaskObjectPlacement],
    ) -> Result<Self, RobotError> {
        let scene = Self::insert_with_representation(
            world,
            definition,
            placements,
            ConvexRepresentation::CopiedFaces,
        )?;
        // The original plate has 256 cooked convex parts. Clustering its support
        // into four solver points failed the frozen native contact comparison;
        // retaining each part's contacts settled it without changing dt/geometry.
        // Only the world that opts into G1 task objects receives this setting.
        world.world.integration_parameters.contact_clustering = false;
        Ok(scene)
    }

    fn insert_with_representation(
        world: &mut SimulationWorld,
        definition: &TaskObjectsDefinition,
        placements: &[TaskObjectPlacement],
        representation: ConvexRepresentation,
    ) -> Result<Self, RobotError> {
        if world.snapshot().integration_count != 0 || placements.is_empty() {
            return Err(invalid(
                "task objects can only be inserted before integration",
            ));
        }
        let mut kinds = HashSet::new();
        let mut prepared = Vec::new();
        for placement in placements {
            if !kinds.insert(placement.kind) || !finite_f32(&placement.root_pose.position) {
                return Err(invalid("duplicate task kind or invalid source placement"));
            }
            let object = definition.object(placement.kind);
            let mut parts = Vec::with_capacity(object.convex_parts.len());
            for part in &object.convex_parts {
                let vertices: Vec<_> = part.points.iter().copied().map(engine_vector).collect();
                let convex = match representation {
                    ConvexRepresentation::CopiedFaces => {
                        SharedShape::convex_mesh(vertices, &part.triangles)
                    }
                    #[cfg(test)]
                    ConvexRepresentation::ReconstructedHull => SharedShape::convex_hull(&vertices),
                }
                .ok_or_else(|| invalid("cannot construct original cooked convex topology"))?;
                parts.push((Pose::IDENTITY, convex));
            }
            let inertia = object.principal_inertia;
            let mass = MassProperties::with_principal_inertia_frame(
                engine_vector(object.center_of_mass),
                object.mass_kg as f32,
                Vector::new(inertia[0] as f32, inertia[2] as f32, inertia[1] as f32),
                Rotation::from_array(source_to_engine_rotation(
                    object.principal_axes_wxyz.map(|v| v as f32),
                )?),
            );
            let body = RigidBodyBuilder::dynamic()
                .pose(Pose::from_parts(
                    engine_vector(placement.root_pose.position),
                    Rotation::from_array(source_to_engine_rotation(
                        placement.root_pose.rotation_wxyz.map(|v| v as f32),
                    )?),
                ))
                .additional_mass_properties(mass)
                .ccd_enabled(object.source_ccd_enabled)
                .can_sleep(false)
                .linear_damping(0.)
                .angular_damping(0.)
                .additional_solver_iterations(0);
            let collider = ColliderBuilder::compound(parts)
                .density(0.)
                // These four original object materials have equal static/dynamic
                // friction. Rapier has one coefficient; retain dynamic explicitly.
                .friction(object.material.dynamic_friction)
                .friction_combine_rule(CoefficientCombineRule::Average)
                .restitution(object.material.restitution)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_3,
                    Group::ALL,
                    InteractionTestMode::And,
                ));
            prepared.push((placement.kind, object.convex_parts.len(), body, collider));
        }
        let mut instances = Vec::new();
        for (kind, convex_parts, body, collider) in prepared {
            let body = world.world.bodies.insert(body);
            let collider =
                world
                    .world
                    .colliders
                    .insert_with_parent(collider, body, &mut world.world.bodies);
            world.world.bodies[body]
                .recompute_mass_properties_from_colliders(&world.world.colliders);
            instances.push(Instance {
                kind,
                body,
                collider,
                convex_parts,
            });
        }
        Ok(Self {
            definition_sha256: definition.file_sha256.clone(),
            instances,
            source_t1_shelf: None,
            source_t2_background: None,
            initial_robot_shelf_overlap: Vec::new(),
        })
    }

    pub fn frame(&self, world: &SimulationWorld) -> Result<TaskObjectFrame, RobotError> {
        let snapshot = world.snapshot();
        if snapshot.episode_step != snapshot.integration_count {
            return Err(invalid(
                "failed integration has no validated task object frame",
            ));
        }
        let mut objects = Vec::new();
        for instance in &self.instances {
            let body = world
                .world
                .bodies
                .get(instance.body)
                .ok_or_else(|| invalid("stale task body handle"))?;
            let collider = world
                .world
                .colliders
                .get(instance.collider)
                .ok_or_else(|| invalid("stale task collider handle"))?;
            if collider.parent() != Some(instance.body)
                || !body.is_dynamic()
                || !body.translation().is_finite()
                || !body.rotation().is_finite()
                || !body.linvel().is_finite()
                || !body.angvel().is_finite()
            {
                return Err(invalid("task body identity or state changed"));
            }
            let last_solve_contacts: Vec<_> = world
                .world
                .narrow_phase
                .contact_pairs_with(instance.collider)
                .filter(|pair| pair.has_any_active_contact())
                .map(|pair| {
                    let other = if pair.collider1 == instance.collider {
                        pair.collider2
                    } else {
                        pair.collider1
                    };
                    let other_body_handle = world.world.colliders[other].parent().map(|h| {
                        let (index, generation) = h.into_raw_parts();
                        [index, generation]
                    });
                    let solver_points = pair
                        .manifolds
                        .iter()
                        .map(|m| m.data.solver_contacts.len())
                        .sum();
                    let minimum = pair
                        .manifolds
                        .iter()
                        .flat_map(|m| m.data.solver_contacts.iter().map(|p| p.dist))
                        .reduce(f32::min);
                    TaskObjectContactSample {
                        other_background_collider_path: self
                            .source_t2_background
                            .as_ref()
                            .and_then(|b| b.collider_path(other))
                            .map(str::to_owned),
                        other_body_handle,
                        other_robot_body_index: None,
                        other_task_kind: self
                            .instances
                            .iter()
                            .find(|instance| {
                                world.world.colliders[other].parent() == Some(instance.body)
                            })
                            .map(|instance| instance.kind),
                        manifold_count: pair.manifolds.len(),
                        solver_points,
                        min_solver_distance_m: minimum,
                        geometric_distance_after_step_m: rapier3d::parry::query::distance(
                            &(*body.position()
                                * collider.position_wrt_parent().copied().unwrap_or_default()),
                            collider.shape(),
                            &world.world.colliders[other].parent().map_or_else(
                                || *world.world.colliders[other].position(),
                                |parent| {
                                    *world.world.bodies[parent].position()
                                        * world.world.colliders[other]
                                            .position_wrt_parent()
                                            .copied()
                                            .unwrap_or_default()
                                },
                            ),
                            world.world.colliders[other].shape(),
                        )
                        .ok()
                        .filter(|distance| distance.is_finite() && *distance >= 0.),
                        normal_impulse_n_s: pair.total_impulse_magnitude(),
                        normal_impulse_on_object_source: normal_impulse_on_collider(
                            pair,
                            instance.collider,
                        ),
                        #[cfg(feature = "g1_constraint_diagnostic")]
                        active_solver_normal_impulse_n_s: active_solver_normal_impulse(pair),
                        #[cfg(feature = "g1_constraint_diagnostic")]
                        active_solver_normal_impulse_on_object_source:
                            active_solver_normal_vector(pair, instance.collider),
                        #[cfg(feature = "g1_contact_point_diagnostic")]
                        diagnostic_solver_contacts: diagnostic_solver_contacts(
                            pair,
                            instance.collider,
                            &world.world.bodies,
                            &world.world.multibody_joints,
                            world.world.integration_parameters.friction_model,
                        ),
                    }
                })
                .collect();
            objects.push(TaskObjectSample {
                kind: instance.kind,
                translation_engine: body.translation().to_array(),
                rotation_engine_xyzw: body.rotation().to_array(),
                position_source: engine_to_source_vector(body.translation().to_array()),
                linear_velocity_source: engine_to_source_vector(body.linvel().to_array()),
                angular_velocity_source: engine_to_source_vector(body.angvel().to_array()),
                mass_kg: body.mass(),
                convex_parts: instance.convex_parts,
                dynamic: body.is_dynamic(),
                active_contact_pairs: last_solve_contacts.len(),
                source_shelf_contact: last_solve_contacts.iter().any(|contact| {
                    contact
                        .other_background_collider_path
                        .as_ref()
                        .is_some_and(|p| p.contains("/TaskAssets/shelf/"))
                }) || self.source_t1_shelf.as_ref().is_some_and(|shelf| {
                    world
                        .world
                        .narrow_phase
                        .contact_pair(instance.collider, shelf.collider)
                        .is_some_and(|pair| pair.has_any_active_contact())
                }),
                last_solve_contacts,
            });
        }
        Ok(TaskObjectFrame {
            definition_sha256: self.definition_sha256.clone(),
            episode_id: snapshot.episode_id,
            source_tick: snapshot.episode_step,
            sim_time: snapshot.episode_seconds,
            contact_clustering: world.world.integration_parameters.contact_clustering,
            contact_recycling: world.world.integration_parameters.contact_recycling,
            world_counts: world.counts(),
            source_t1_shelf: self
                .source_t1_shelf
                .as_ref()
                .map(|shelf| shelf.sample(world))
                .transpose()?,
            source_t2_background: self
                .source_t2_background
                .as_ref()
                .map(|background| background.sample(world))
                .transpose()?,
            initial_robot_shelf_overlap: self.initial_robot_shelf_overlap.clone(),
            objects,
        })
    }

    /// Test-only label lookup; never changes contact state or controller inputs.
    #[cfg(all(test, feature = "g1_constraint_diagnostic"))]
    pub(super) fn diagnostic_background_path(
        &self,
        collider: rapier3d::prelude::ColliderHandle,
    ) -> Option<&str> {
        self.source_t2_background
            .as_ref()
            .and_then(|b| b.collider_path(collider))
    }
}

fn normal_impulse_on_collider(
    pair: &rapier3d::geometry::ContactPair,
    collider: ColliderHandle,
) -> [f32; 3] {
    // Rapier's constraint uses -manifold.normal on collider1, +normal on2.
    // Both collider insertion orders are checked against gravity contact below.
    let impulse = pair.total_impulse();
    engine_to_source_vector(
        (if pair.collider1 == collider {
            -impulse
        } else {
            impulse
        })
        .to_array(),
    )
}

#[cfg(feature = "g1_constraint_diagnostic")]
pub(super) fn active_solver_normal_impulse(pair: &rapier3d::geometry::ContactPair) -> Option<f32> {
    let mut count = 0;
    let mut total = 0.;
    for manifold in &pair.manifolds {
        for contact in &manifold.data.solver_contacts {
            let index = (contact.contact_id[0] & !rapier3d::geometry::NEW_CONTACT_BIT) as usize;
            let impulse = manifold.points.get(index)?.data.impulse;
            if !impulse.is_finite() || impulse < 0. {
                return None;
            }
            total += impulse;
            count += 1;
        }
    }
    (count > 0 && total.is_finite()).then_some(total)
}

#[cfg(feature = "g1_constraint_diagnostic")]
fn active_solver_normal_vector(
    pair: &rapier3d::geometry::ContactPair,
    object: ColliderHandle,
) -> Option<[f32; 3]> {
    active_solver_normal_impulse(pair)?;
    let mut impulse = Vector::ZERO;
    for manifold in &pair.manifolds {
        let mut magnitude = 0.;
        for contact in &manifold.data.solver_contacts {
            let index = (contact.contact_id[0] & !rapier3d::geometry::NEW_CONTACT_BIT) as usize;
            magnitude += manifold.points.get(index)?.data.impulse;
        }
        impulse += manifold.data.normal * magnitude;
    }
    let sign = if pair.collider1 == object { -1. } else { 1. };
    let result = engine_to_source_vector((impulse * sign).to_array());
    result.iter().all(|v| v.is_finite()).then_some(result)
}

#[cfg(feature = "g1_contact_point_diagnostic")]
fn diagnostic_solver_contacts(
    pair: &rapier3d::geometry::ContactPair,
    object: ColliderHandle,
    bodies: &RigidBodySet,
    multibodies: &MultibodyJointSet,
    friction_model: FrictionModel,
) -> Vec<TaskObjectSolverContactSample> {
    let object_first = pair.collider1 == object;
    pair.manifolds
        .iter()
        .enumerate()
        .flat_map(|(manifold_index, manifold)| {
            let pointwise_friction = friction_model == FrictionModel::Coulomb
                || [manifold.data.rigid_body1, manifold.data.rigid_body2]
                    .into_iter()
                    .flatten()
                    .any(|body| multibodies.rigid_body_link(body).is_some());
            manifold
                .data
                .solver_contacts
                .iter()
                .filter_map(move |contact| {
                    let point_index =
                        (contact.contact_id[0] & !rapier3d::geometry::NEW_CONTACT_BIT) as usize;
                    let point = manifold.points.get(point_index)?;
                    let (anchor1, anchor2) =
                        manifold.data.solver_contact_world_points(contact, bodies);
                    let (object_anchor, other_anchor, lever) = if object_first {
                        (anchor1, anchor2, point.data.solver_dp1)
                    } else {
                        (anchor2, anchor1, point.data.solver_dp2)
                    };
                    Some(TaskObjectSolverContactSample {
                        manifold_index,
                        point_index,
                        cached_contact_new_bit: contact.contact_id[0]
                            & rapier3d::geometry::NEW_CONTACT_BIT
                            != 0,
                        normal_impulse_n_s: point.data.impulse,
                        tangent_impulse_solver_basis_n_s: pointwise_friction.then_some([
                            point.data.tangent_impulse[0],
                            point.data.tangent_impulse[1],
                        ]),
                        effective_friction: manifold.data.friction,
                        cached_manifold_normal_source: engine_to_source_vector(
                            manifold.data.normal.to_array(),
                        ),
                        cached_solver_lever_arm_on_object_source_m: engine_to_source_vector(
                            lever.to_array(),
                        ),
                        cached_anchor_on_object_source_m: engine_to_source_vector(
                            object_anchor.to_array(),
                        ),
                        cached_anchor_on_other_source_m: engine_to_source_vector(
                            other_anchor.to_array(),
                        ),
                    })
                })
        })
        .collect()
}

fn engine_vector(value: [f64; 3]) -> Vector {
    Vector::from_array(source_to_engine_vector(value.map(|v| v as f32)))
}
fn finite_f32(values: &[f64]) -> bool {
    values
        .iter()
        .all(|v| v.is_finite() && (*v as f32).is_finite())
}
fn sha256_text(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::OpenOptions, io::Write};

    #[test]
    fn support_impulse_points_up_for_both_collider_orders() {
        for object_first in [false, true] {
            let mut world = SimulationWorld::with_game_frequency(50).unwrap();
            let object = world
                .world
                .bodies
                .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0., 0.3, 0.)));
            let floor = world.world.bodies.insert(RigidBodyBuilder::fixed());
            let insert_object = |world: &mut SimulationWorld| {
                world.world.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(0.1, 0.1, 0.1).mass(0.1),
                    object,
                    &mut world.world.bodies,
                )
            };
            let insert_floor = |world: &mut SimulationWorld| {
                world.world.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(1., 0.1, 1.),
                    floor,
                    &mut world.world.bodies,
                )
            };
            let object_collider = if object_first {
                let c = insert_object(&mut world);
                insert_floor(&mut world);
                c
            } else {
                insert_floor(&mut world);
                insert_object(&mut world)
            };
            for _ in 0..100 {
                world.step_with_torques(&[]).unwrap();
            }
            let pair = world
                .world
                .narrow_phase
                .contact_pairs_with(object_collider)
                .find(|pair| pair.total_impulse_magnitude() > 0.)
                .expect("gravity support");
            let impulse = normal_impulse_on_collider(pair, object_collider);
            assert!(
                impulse[2] > 0.01,
                "support must oppose gravity: {impulse:?}"
            );
            assert!(impulse[0].abs() < 1e-6 && impulse[1].abs() < 1e-6);
            assert_eq!(world.snapshot().integration_count, 100);
        }
    }

    #[cfg(feature = "g1_contact_point_diagnostic")]
    #[test]
    fn solver_contact_audit_retains_sliding_impulses_for_both_orders() {
        for object_first in [false, true] {
            let mut world = SimulationWorld::with_game_frequency(50).unwrap();
            let object = world
                .world
                .bodies
                .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0., 0.2, 0.)));
            let floor = world.world.bodies.insert(RigidBodyBuilder::fixed());
            let insert_object = |world: &mut SimulationWorld| {
                world.world.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(0.1, 0.1, 0.1)
                        .mass(0.1)
                        .friction(0.5),
                    object,
                    &mut world.world.bodies,
                )
            };
            let insert_floor = |world: &mut SimulationWorld| {
                world.world.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(1., 0.1, 1.).friction(0.5),
                    floor,
                    &mut world.world.bodies,
                )
            };
            let collider = if object_first {
                let c = insert_object(&mut world);
                insert_floor(&mut world);
                c
            } else {
                insert_floor(&mut world);
                insert_object(&mut world)
            };
            for _ in 0..20 {
                world.step_with_torques(&[]).unwrap();
            }
            world.world.bodies[object].set_linvel(Vector::new(0.4, 0., 0.), true);
            world.step_with_torques(&[]).unwrap();
            let pair = world
                .world
                .narrow_phase
                .contact_pairs_with(collider)
                .find(|p| p.total_impulse_magnitude() > 0.)
                .expect("loaded sliding contact");
            let unavailable = diagnostic_solver_contacts(
                pair,
                collider,
                &world.world.bodies,
                &world.world.multibody_joints,
                world.world.integration_parameters.friction_model,
            );
            assert!(!unavailable.is_empty());
            assert!(
                unavailable
                    .iter()
                    .all(|s| s.tangent_impulse_solver_basis_n_s.is_none())
            );
            // This isolated audit test exercises the supported pointwise path;
            // no G1 or default world changes its original friction model.
            world.world.integration_parameters.friction_model = FrictionModel::Coulomb;
            world.step_with_torques(&[]).unwrap();
            let pair = world
                .world
                .narrow_phase
                .contact_pairs_with(collider)
                .find(|p| p.total_impulse_magnitude() > 0.)
                .expect("pointwise sliding contact");
            let samples = diagnostic_solver_contacts(
                pair,
                collider,
                &world.world.bodies,
                &world.world.multibody_joints,
                world.world.integration_parameters.friction_model,
            );
            assert!(!samples.is_empty());
            let normal_sum: f32 = samples.iter().map(|s| s.normal_impulse_n_s).sum();
            assert!((normal_sum - pair.total_impulse_magnitude()).abs() < 1e-6);
            assert!((active_solver_normal_impulse(pair).unwrap() - normal_sum).abs() < 1e-6);
            let active_vector = active_solver_normal_vector(pair, collider).unwrap();
            assert!(active_vector[2] > 0.01);
            // Reproduce a stale unselected manifold point at the actual API
            // seam: legacy aggregation sees it, active solver evidence must not.
            let mut cached = pair.clone();
            let mut stale = cached.manifolds[0].points[0].clone();
            stale.data.impulse = 123.;
            cached.manifolds[0].points.push(stale);
            assert!(cached.total_impulse_magnitude() > 123.);
            assert!((active_solver_normal_impulse(&cached).unwrap() - normal_sum).abs() < 1e-6);
            assert_eq!(
                active_solver_normal_vector(&cached, collider).unwrap(),
                active_vector
            );
            for manifold in &mut cached.manifolds {
                manifold.data.solver_contacts.clear();
            }
            assert!(active_solver_normal_impulse(&cached).is_none());
            assert!(active_solver_normal_vector(&cached, collider).is_none());
            assert!(samples.iter().any(|s| {
                let [x, y] = s
                    .tangent_impulse_solver_basis_n_s
                    .expect("Coulomb writeback");
                x.hypot(y) > 1e-5
            }));
            assert!(samples.iter().all(|s| s.effective_friction == 0.5));
            assert!(world.world.bodies[object].linvel().x < 0.4);
            assert_eq!(world.snapshot().integration_count, 22);
        }
    }

    #[test]
    #[ignore = "requires frozen original cooked objects; performs 150 actual native integrations"]
    fn real_native_task_object_contact_diagnostic() {
        let path = std::env::var("G1_TASK_OBJECTS_DEFINITION").expect("definition path");
        let sha = std::env::var("G1_TASK_OBJECTS_SHA256").expect("definition hash");
        let output = std::env::var("G1_TASK_OBJECTS_OUTPUT").expect("new output");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        let mut receipt = serde_json::json!({"qualified":false,"source_task_physics_parity_proven":false,
            "robot_present":false,"policy_inferences":0,"executor_actions":0,"source_commit":std::env::var("G1_CODE_COMMIT").ok(),"definition_sha256":sha});
        let result = (|| -> Result<(), RobotError> {
            let pgs: usize = std::env::var("G1_TASK_OBJECTS_DIAGNOSTIC_PGS")
                .unwrap_or_else(|_| "4".into())
                .parse()
                .map_err(|_| invalid("invalid diagnostic PGS count"))?;
            let start = std::env::var("G1_TASK_OBJECTS_DIAGNOSTIC_START")
                .unwrap_or_else(|_| "drop_0_3m".into());
            let representation_name = std::env::var("G1_TASK_OBJECTS_DIAGNOSTIC_REPRESENTATION")
                .unwrap_or_else(|_| "copied_faces".into());
            let representation = match representation_name.as_str() {
                "copied_faces" => ConvexRepresentation::CopiedFaces,
                "reconstructed_hull" => ConvexRepresentation::ReconstructedHull,
                _ => return Err(invalid("unsupported convex feature representation")),
            };
            if !matches!(pgs, 4 | 16) || !matches!(start.as_str(), "drop_0_3m" | "near_support_5mm")
            {
                return Err(invalid("unsupported bounded contact comparison"));
            }
            receipt["diagnostic_start"] = serde_json::json!(start);
            receipt["diagnostic_pgs"] = serde_json::json!(pgs);
            receipt["diagnostic_representation"] = serde_json::json!(representation_name);
            let contact_mode = std::env::var("G1_TASK_OBJECTS_DIAGNOSTIC_CONTACT_MODE")
                .unwrap_or_else(|_| "default".into());
            let definition = TaskObjectsDefinition::load(Path::new(&path), &sha)?;
            let mut world =
                SimulationWorld::with_game_frequency(50).map_err(|e| invalid(e.to_string()))?;
            world
                .world
                .integration_parameters
                .num_internal_pgs_iterations = pgs;
            match contact_mode.as_str() {
                "default" | "task_owner" => {}
                "fresh_manifolds" => world.world.integration_parameters.contact_recycling = false,
                "unclustered" => world.world.integration_parameters.contact_clustering = false,
                _ => return Err(invalid("unsupported bounded contact algorithm comparison")),
            }
            let floor = world
                .world
                .bodies
                .insert(RigidBodyBuilder::fixed().translation(Vector::new(0., -0.25, 0.)));
            world.world.colliders.insert_with_parent(
                ColliderBuilder::cuboid(10., 0.25, 10.).friction(1.),
                floor,
                &mut world.world.bodies,
            );
            let kinds = [
                TaskObjectKind::Apple,
                TaskObjectKind::Plate,
                TaskObjectKind::BrownBox,
                TaskObjectKind::BlueBin,
            ];
            let placements: Vec<_> = kinds
                .into_iter()
                .enumerate()
                .map(|(i, kind)| {
                    let initial_z = if start == "near_support_5mm" {
                        -definition
                            .object(kind)
                            .convex_parts
                            .iter()
                            .flat_map(|part| part.points.iter())
                            .map(|p| p[2])
                            .fold(f64::INFINITY, f64::min)
                            + 0.005
                    } else {
                        0.3
                    };
                    TaskObjectPlacement {
                        kind,
                        root_pose: SourcePose {
                            position: [i as f64 - 1.5, 0., initial_z],
                            rotation_wxyz: [1., 0., 0., 0.],
                        },
                    }
                })
                .collect();
            let scene = if contact_mode == "task_owner" {
                if representation_name != "copied_faces" {
                    return Err(invalid("task owner retains original copied convex faces"));
                }
                TaskObjectScene::insert_in_owner_world(&mut world, &definition, &placements)?
            } else {
                TaskObjectScene::insert_with_representation(
                    &mut world,
                    &definition,
                    &placements,
                    representation,
                )?
            };
            receipt["contact_algorithms"] = serde_json::json!({"mode":contact_mode,
                "clustering":world.world.integration_parameters.contact_clustering,
                "recycling":world.world.integration_parameters.contact_recycling,
                "prediction_distance":world.world.integration_parameters.normalized_prediction_distance,
                "allowed_linear_error":world.world.integration_parameters.normalized_allowed_linear_error,
                "recycle_distance":world.world.integration_parameters.normalized_contact_recycle_distance});
            receipt["initial"] = serde_json::to_value(scene.frame(&world)?).unwrap();
            let bin = scene
                .instances
                .iter()
                .find(|i| i.kind == TaskObjectKind::BlueBin)
                .unwrap();
            let box_instance = scene
                .instances
                .iter()
                .find(|i| i.kind == TaskObjectKind::BrownBox)
                .unwrap();
            let bin_center_filled = world.world.colliders[bin.collider]
                .shape()
                .contains_local_point(engine_vector([0., 0., 0.04]));
            let box_center_filled = world.world.colliders[box_instance.collider]
                .shape()
                .contains_local_point(Vector::ZERO);
            receipt["bin_probe_source"] = serde_json::json!([0., 0., 0.04]);
            receipt["bin_open_center_empty"] = serde_json::json!(!bin_center_filled);
            receipt["closed_box_center_filled"] = serde_json::json!(box_center_filled);
            if bin_center_filled || !box_center_filled {
                return Err(invalid(
                    "native task topology sealed the bin or inverted the box",
                ));
            }
            receipt["frames"] = serde_json::json!([]);
            let ticks: u64 = std::env::var("G1_TASK_OBJECTS_DIAGNOSTIC_TICKS")
                .unwrap_or_else(|_| "150".into())
                .parse()
                .map_err(|_| invalid("invalid object Tick budget"))?;
            if !matches!(ticks, 150 | 500) {
                return Err(invalid("object Tick budget must be 150 or 500"));
            }
            receipt["requested_ticks"] = serde_json::json!(ticks);
            let mut settled_ticks = 0u64;
            for _ in 0..ticks {
                let boundary = world.step_with_torques(&[]);
                receipt["integrations"] = serde_json::json!(world.snapshot().integration_count);
                receipt["final_configuration"] =
                    serde_json::to_value(world.configuration()).unwrap();
                receipt["world_counts"] = serde_json::to_value(world.counts()).unwrap();
                boundary.map_err(|e| invalid(e.to_string()))?;
                let frame = scene.frame(&world)?;
                if frame.objects.iter().all(settled) {
                    settled_ticks += 1;
                } else {
                    settled_ticks = 0;
                }
                receipt["frames"]
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::to_value(frame).unwrap());
            }
            receipt["final_configuration"] = serde_json::to_value(world.configuration()).unwrap();
            receipt["world_counts"] = serde_json::to_value(world.counts()).unwrap();
            receipt["integrations"] = serde_json::json!(world.snapshot().integration_count);
            let plate = scene
                .instances
                .iter()
                .find(|i| i.kind == TaskObjectKind::Plate)
                .unwrap();
            let contacts:Vec<_>=world.world.narrow_phase.contact_pairs_with(plate.collider).map(|pair| {
                let summarize=|manifolds:&[ContactManifold]|manifolds.iter().map(|m|serde_json::json!({
                    "normal_source":engine_to_source_vector(m.data.normal.to_array()),
                    "points":m.points.len(),"min_geometric_distance":m.points.iter().map(|p|p.dist).fold(f32::INFINITY,f32::min),
                    "solver_points":m.data.solver_contacts.len(),
                    "min_solver_distance":m.data.solver_contacts.iter().map(|p|p.dist).fold(f32::INFINITY,f32::min)
                })).collect::<Vec<_>>();
                serde_json::json!({"manifolds":summarize(&pair.manifolds),"solver_clusters":summarize(&pair.solver_clusters)})
            }).collect();
            receipt["plate_last_native_solve_contacts"] = serde_json::json!(contacts);
            receipt["contact_samples_are_from_last_solve_not_refreshed_end_pose"] =
                serde_json::json!(true);
            let final_frame = scene.frame(&world)?;
            receipt["final_continuously_settled_ticks"] = serde_json::json!(settled_ticks);
            receipt["final_continuously_settled_seconds"] =
                serde_json::json!(settled_ticks as f64 * 0.02);
            if !final_frame.objects.iter().all(settled)
                || (contact_mode == "task_owner" && settled_ticks < 100)
            {
                return Err(invalid(
                    "original task objects did not settle in physical contact",
                ));
            }
            if world.configuration().num_solver_iterations != 1
                || world.configuration().max_ccd_substeps != 1
                || world.snapshot().integration_count != ticks
            {
                return Err(invalid(
                    "task object diagnostic altered single integration clock",
                ));
            }
            Ok(())
        })();
        receipt["success"] = serde_json::json!(result.is_ok());
        if let Err(error) = &result {
            receipt["error"] = serde_json::json!(error.to_string());
        }
        file.write_all(serde_json::to_string(&receipt).unwrap().as_bytes())
            .unwrap();
        file.write_all(b"\n").unwrap();
        result.unwrap();
    }

    fn settled(o: &TaskObjectSample) -> bool {
        o.active_contact_pairs > 0
            && o.linear_velocity_source
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                < 0.02
            && o.angular_velocity_source
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                < 0.1
    }
}
