//! Byte-bound released T2 background in the same owner world as G1 and props.
//!
//! Source cooking/query provenance is retained. Static authored polygon fans
//! remain an explicit diagnostic; loading these shapes does not qualify PhysX
//! contact parity, task success or the 50 Hz migration.
use crate::SimulationWorld;
use rapier3d::{
    math::{Pose, Rotation, Vector},
    prelude::*,
};
use robot_minigame::{
    RobotError,
    basis::{source_to_engine_rotation, source_to_engine_vector},
    g1::{
        definition::SourcePose,
        policy::bound_bytes,
        task_fixtures::{STATION_PHYSICAL_FIXTURES, T2BackgroundSelection},
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf};

const ARENA: &str = "8b4a3a47fc53de23e8205089d71109a2e2348acd";
const BACKGROUND: &str = "7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051";
const RUNTIME: &str = "6.0.0-rc.22+release.33481.407f3ea1.gl";
const PREFIX: &str = "/World/envs/env_0/galileo_locomanip/";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct T2SourceBackgroundConfig {
    pub definition: PathBuf,
    pub definition_sha256: String,
    /// A single scene-wide offset, also used for the robot, box, bin and floor.
    pub environment_translation_source: [f64; 3],
    /// Default preserves the complete original scene. Station fixtures require
    /// a separately prepared physical environment in the owner factory.
    #[serde(default)]
    pub selection: T2BackgroundSelection,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Material {
    static_friction: f32,
    dynamic_friction: f32,
    restitution: f32,
    friction_combine: String,
    restitution_combine: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    path: String,
    geometry: String,
    material: Material,
    parts: Vec<Part>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    path: String,
    dynamic: bool,
    root_pose: SourcePose,
    linear_velocity: [f64; 3],
    angular_velocity: [f64; 3],
    mass_kg: f64,
    center_of_mass: [f64; 3],
    principal_inertia: [f64; 3],
    principal_axes_wxyz: [f64; 4],
    ccd_enabled: bool,
    linear_damping: f32,
    angular_damping: f32,
    colliders: Vec<Shape>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ground {
    path: String,
    height_source: f64,
    material: Material,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    units: String,
    source_arena_commit: String,
    source_background_sha256: String,
    source_runtime_build: String,
    source_query_sha256: String,
    source_receipt_sha256: String,
    exporter_sha256: String,
    physics_parity_qualified: bool,
    bodies: Vec<Body>,
    existing_ground_plane: Ground,
    audit: serde_json::Value,
}

/// All fallible file, geometry, material and floor checks precede world mutation.
pub(super) struct PreparedBackground {
    selection: T2BackgroundSelection,
    original_owner_floor: ColliderHandle,
    sha256: String,
    query_sha256: String,
    audit: serde_json::Value,
    bodies: Vec<(String, RigidBodyBuilder, Vec<(String, ColliderBuilder)>)>,
}
struct Instance {
    path: String,
    body: RigidBodyHandle,
    colliders: Vec<(String, ColliderHandle)>,
    dynamic: bool,
}
pub(super) struct T2SourceBackground {
    selection: T2BackgroundSelection,
    original_owner_floor: ColliderHandle,
    sha256: String,
    query_sha256: String,
    audit: serde_json::Value,
    instances: Vec<Instance>,
}
#[derive(Clone, Debug, Serialize)]
pub struct T2BackgroundBodySample {
    pub path: String,
    pub translation_engine: [f32; 3],
    pub rotation_engine_xyzw: [f32; 4],
    pub linear_velocity_engine: [f32; 3],
    pub angular_velocity_engine: [f32; 3],
    pub dynamic: bool,
    pub mass_kg: f32,
    pub collider_count: usize,
}
/// Render/independent acceptance only; never a policy observation.
#[derive(Clone, Debug, Serialize)]
pub struct T2SourceBackgroundSample {
    pub definition_sha256: String,
    pub source_query_sha256: String,
    pub physics_parity_qualified: bool,
    pub existing_floor_reused: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub station_task_fixture_selection: Option<serde_json::Value>,
    pub audit: serde_json::Value,
    pub bodies: Vec<T2BackgroundBodySample>,
}

impl T2SourceBackgroundConfig {
    pub(super) fn prepare(
        &self,
        world: &SimulationWorld,
    ) -> Result<PreparedBackground, RobotError> {
        if world.snapshot().integration_count != 0
            || self.environment_translation_source != [0., 0., 0.795]
        {
            return Err(invalid(
                "T2 background is startup-only and requires the original scene-wide floor offset",
            ));
        }
        if std::fs::metadata(&self.definition)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > 32 * 1024 * 1024
        {
            return Err(invalid("T2 background exceeds 32 MiB"));
        }
        let bytes = bound_bytes(&self.definition, &self.definition_sha256)?;
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(invalid("T2 background exceeds 32 MiB"));
        }
        let doc: Document = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        if doc.schema != "native_g1_released_t2_background_v1"
            || doc.units != "metres_kilograms_radians_z_up"
            || doc.source_arena_commit != ARENA
            || doc.source_background_sha256 != BACKGROUND
            || doc.source_runtime_build != RUNTIME
            || doc.physics_parity_qualified
            || ![
                &doc.source_query_sha256,
                &doc.source_receipt_sha256,
                &doc.exporter_sha256,
            ]
            .iter()
            .all(|s| hash(s))
            || doc.bodies.len() != 3
            || doc.bodies.iter().filter(|b| b.dynamic).count() != 2
            || doc.bodies.iter().map(|b| b.colliders.len()).sum::<usize>() != 250
            || doc.audit["physics_parity_qualified"] != false
            || doc.audit["fitted_boxes"] != 130
            || doc.audit["cooked_convex_parts"] != 294
        {
            return Err(invalid(
                "foreign/incomplete T2 background or unsupported qualification claim",
            ));
        }
        let ground = &doc.existing_ground_plane;
        if !ground.path.starts_with(PREFIX)
            || !ground.height_source.is_finite()
            || (ground.height_source + self.environment_translation_source[2]).abs() > 1e-6
        {
            return Err(invalid(
                "original T2 ground plane is incompatible with owner floor",
            ));
        }
        validate_material(&ground.material)?;
        let floors: Vec<_> = world
            .world
            .colliders
            .iter()
            .filter(|(_, c)| c.collision_groups().memberships == Group::GROUP_2)
            .collect();
        if floors.len() != 1 {
            return Err(invalid("T2 background requires one existing owner floor"));
        }
        let floor = floors[0].1;
        let parent = floor
            .parent()
            .and_then(|p| world.world.bodies.get(p))
            .ok_or_else(|| invalid("owner floor missing body"))?;
        let pose = *parent.position() * floor.position_wrt_parent().copied().unwrap_or_default();
        let bounds = floor.shape().compute_aabb(&pose);
        if !parent.is_fixed()
            || bounds.maxs.y.abs() > 1e-6
            || bounds.mins.x > -10.
            || bounds.maxs.x < 10.
            || (floor.friction() - ground.material.dynamic_friction).abs() > 1e-6
            || (floor.restitution() - ground.material.restitution).abs() > 1e-6
        {
            return Err(invalid(
                "owner floor geometry/material differs from released T2 floor",
            ));
        }
        let mut paths = HashSet::new();
        let mut prepared = Vec::new();
        for body in doc.bodies {
            if !body.path.starts_with(PREFIX)
                || !paths.insert(body.path.clone())
                || body.colliders.is_empty()
                || !finite(&body.root_pose.position)
                || !finite(&body.linear_velocity)
                || !finite(&body.angular_velocity)
                || !body.linear_damping.is_finite()
                || body.linear_damping < 0.
                || !body.angular_damping.is_finite()
                || body.angular_damping < 0.
            {
                return Err(invalid("invalid original T2 body identity/state"));
            }
            let position = std::array::from_fn(|i| {
                body.root_pose.position[i] + self.environment_translation_source[i]
            });
            let pose = Pose::from_parts(
                vector(position),
                Rotation::from_array(source_to_engine_rotation(
                    body.root_pose.rotation_wxyz.map(|v| v as f32),
                )?),
            );
            let mut builder = if body.dynamic {
                RigidBodyBuilder::dynamic()
            } else {
                RigidBodyBuilder::fixed()
            };
            builder = builder
                .pose(pose)
                .linear_damping(body.linear_damping)
                .angular_damping(body.angular_damping)
                .ccd_enabled(body.ccd_enabled)
                .additional_solver_iterations(0)
                .can_sleep(false);
            if body.dynamic {
                if !finite(&[body.mass_kg])
                    || body.mass_kg <= 0.
                    || !finite(&body.center_of_mass)
                    || !finite(&body.principal_inertia)
                    || body.principal_inertia.iter().any(|v| *v <= 0.)
                {
                    return Err(invalid("invalid actual T2 background mass/inertia"));
                }
                let i = body.principal_inertia;
                builder = builder
                    .additional_mass_properties(MassProperties::with_principal_inertia_frame(
                        vector(body.center_of_mass),
                        body.mass_kg as f32,
                        Vector::new(i[0] as f32, i[2] as f32, i[1] as f32),
                        Rotation::from_array(source_to_engine_rotation(
                            body.principal_axes_wxyz.map(|v| v as f32),
                        )?),
                    ))
                    .linvel(vector(body.linear_velocity))
                    .angvel(vector(body.angular_velocity));
            } else if body.root_pose.position != [0.; 3]
                || body.mass_kg != 0.
                || body.linear_velocity != [0.; 3]
                || body.angular_velocity != [0.; 3]
            {
                return Err(invalid(
                    "fixed T2 group must have the original world coordinate frame",
                ));
            }
            let mut colliders = Vec::new();
            for shape in body.colliders {
                if !shape.path.starts_with(PREFIX)
                    || !paths.insert(shape.path.clone())
                    || shape.parts.is_empty()
                    || shape.parts.len() > 1024
                {
                    return Err(invalid("invalid original T2 collider coverage"));
                }
                validate_material(&shape.material)?;
                let triangle_mesh = match shape.geometry.as_str() {
                    "triangle_mesh" if !body.dynamic && shape.parts.len() == 1 => true,
                    "convex_parts" => false,
                    _ => return Err(invalid("unsupported T2 collision topology/body type")),
                };
                let mut parts = Vec::new();
                for part in shape.parts {
                    let maximum = if triangle_mesh { 100_000 } else { 512 };
                    if !(3..=maximum).contains(&part.points.len())
                        || part.triangles.is_empty()
                        || part.triangles.len() > 200_000
                        || part.points.iter().any(|v| !finite(v))
                        || part.triangles.iter().any(|t| {
                            t.iter().any(|v| *v as usize >= part.points.len())
                                || t[0] == t[1]
                                || t[1] == t[2]
                                || t[2] == t[0]
                        })
                    {
                        return Err(invalid("malformed original T2 vertices/faces"));
                    }
                    let vertices = part.points.into_iter().map(vector).collect();
                    let mesh = if triangle_mesh {
                        SharedShape::trimesh(vertices, part.triangles)
                            .map_err(|e| invalid(format!("T2 triangle construction: {e:?}")))?
                    } else {
                        SharedShape::convex_mesh(vertices, &part.triangles)
                            .ok_or_else(|| invalid("T2 cooked convex topology rejected"))?
                    };
                    parts.push((Pose::IDENTITY, mesh));
                }
                let shape_geometry = if triangle_mesh {
                    parts.remove(0).1
                } else {
                    SharedShape::compound(parts)
                };
                colliders.push((
                    shape.path,
                    ColliderBuilder::new(shape_geometry)
                        .density(0.)
                        .friction(shape.material.dynamic_friction)
                        .friction_combine_rule(combine(&shape.material.friction_combine)?)
                        .restitution(shape.material.restitution)
                        .restitution_combine_rule(combine(&shape.material.restitution_combine)?)
                        .collision_groups(InteractionGroups::new(
                            Group::GROUP_4,
                            Group::ALL,
                            InteractionTestMode::And,
                        )),
                ));
            }
            prepared.push((body.path, builder, colliders));
        }
        if self.selection == T2BackgroundSelection::StationTaskFixtures {
            for (body, _, colliders) in &mut prepared {
                colliders.retain(|(path, _)| self.selection.physical_fixture(body, path));
            }
            prepared.retain(|(_, _, colliders)| !colliders.is_empty());
            let retained: HashSet<_> = prepared
                .iter()
                .flat_map(|(_, _, shapes)| shapes.iter().map(|(path, _)| path.as_str()))
                .collect();
            if prepared.len() != 2 || retained != HashSet::from(STATION_PHYSICAL_FIXTURES) {
                return Err(invalid(
                    "station task support coverage differs from frozen shelf/table fixtures",
                ));
            }
        }
        Ok(PreparedBackground {
            selection: self.selection,
            original_owner_floor: floors[0].0,
            sha256: self.definition_sha256.clone(),
            query_sha256: doc.source_query_sha256,
            audit: doc.audit,
            bodies: prepared,
        })
    }
}
impl PreparedBackground {
    pub(super) fn insert(self, world: &mut SimulationWorld) -> T2SourceBackground {
        let mut instances = Vec::new();
        for (path, builder, shapes) in self.bodies {
            let body = world.world.bodies.insert(builder);
            let dynamic = world.world.bodies[body].is_dynamic();
            let colliders = shapes
                .into_iter()
                .map(|(path, s)| {
                    (
                        path,
                        world
                            .world
                            .colliders
                            .insert_with_parent(s, body, &mut world.world.bodies),
                    )
                })
                .collect();
            world.world.bodies[body]
                .recompute_mass_properties_from_colliders(&world.world.colliders);
            instances.push(Instance {
                path,
                body,
                colliders,
                dynamic,
            });
        }
        T2SourceBackground {
            selection: self.selection,
            original_owner_floor: self.original_owner_floor,
            sha256: self.sha256,
            query_sha256: self.query_sha256,
            audit: self.audit,
            instances,
        }
    }
}
impl T2SourceBackground {
    pub(super) fn selection(&self) -> T2BackgroundSelection {
        self.selection
    }
    pub(super) fn collider_path(&self, handle: ColliderHandle) -> Option<&str> {
        self.instances
            .iter()
            .flat_map(|b| b.colliders.iter())
            .find(|(_, h)| *h == handle)
            .map(|(path, _)| path.as_str())
    }

    pub(super) fn sample(
        &self,
        world: &SimulationWorld,
    ) -> Result<T2SourceBackgroundSample, RobotError> {
        let mut bodies = Vec::new();
        for instance in &self.instances {
            let b = world
                .world
                .bodies
                .get(instance.body)
                .ok_or_else(|| invalid("stale T2 background body"))?;
            if b.is_dynamic() != instance.dynamic
                || !b.translation().is_finite()
                || !b.rotation().is_finite()
                || !b.linvel().is_finite()
                || !b.angvel().is_finite()
                || instance.colliders.iter().any(|(_, h)| {
                    world
                        .world
                        .colliders
                        .get(*h)
                        .is_none_or(|c| c.parent() != Some(instance.body))
                })
            {
                return Err(invalid("T2 background identity/state changed"));
            }
            bodies.push(T2BackgroundBodySample {
                path: instance.path.clone(),
                translation_engine: b.translation().to_array(),
                rotation_engine_xyzw: b.rotation().to_array(),
                linear_velocity_engine: b.linvel().to_array(),
                angular_velocity_engine: b.angvel().to_array(),
                dynamic: b.is_dynamic(),
                mass_kg: b.mass(),
                collider_count: instance.colliders.len(),
            });
        }
        Ok(T2SourceBackgroundSample {
            definition_sha256: self.sha256.clone(),
            source_query_sha256: self.query_sha256.clone(),
            physics_parity_qualified: false,
            existing_floor_reused: world.world.colliders.contains(self.original_owner_floor),
            station_task_fixture_selection: (self.selection
                == T2BackgroundSelection::StationTaskFixtures)
                .then(|| {
                    serde_json::json!({
                        "selection": self.selection,
                        "retained_collider_paths": STATION_PHYSICAL_FIXTURES,
                        "retained_bodies": 2, "retained_colliders": 6,
                        "omitted_source_bodies": 1, "omitted_source_colliders": 244,
                        "source_material_mass_pose_and_cooking_unchanged": true,
                        "source_task_appearance_parity_qualified": false,
                    })
                }),
            audit: self.audit.clone(),
            bodies,
        })
    }
}
fn vector(v: [f64; 3]) -> Vector {
    Vector::from_array(source_to_engine_vector(v.map(|x| x as f32)))
}
fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite() && (*x as f32).is_finite())
}
fn hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn invalid(s: impl Into<String>) -> RobotError {
    RobotError::Contract(s.into())
}
fn combine(s: &str) -> Result<CoefficientCombineRule, RobotError> {
    match s {
        "average" => Ok(CoefficientCombineRule::Average),
        "min" => Ok(CoefficientCombineRule::Min),
        "max" => Ok(CoefficientCombineRule::Max),
        "multiply" => Ok(CoefficientCombineRule::Multiply),
        _ => Err(invalid("unsupported original T2 material combine rule")),
    }
}
fn validate_material(m: &Material) -> Result<(), RobotError> {
    if !m.static_friction.is_finite()
        || !m.dynamic_friction.is_finite()
        || m.static_friction < 0.
        || m.dynamic_friction < 0.
        || (m.static_friction - m.dynamic_friction).abs() > 1e-6
        || !m.restitution.is_finite()
        || !(0.0..=1.0).contains(&m.restitution)
    {
        return Err(invalid(
            "original T2 material cannot be represented by one native friction coefficient",
        ));
    }
    combine(&m.friction_combine)?;
    combine(&m.restitution_combine)?;
    Ok(())
}
