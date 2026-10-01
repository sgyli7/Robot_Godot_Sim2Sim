//! Original T1 procedural support, constructed once in the robot owner's world.
//!
//! Arena 8b4a3a47 defines a fixed cuboid underneath the original visible shelf.
//! Source contact_offset 5 mm is recorded, not misrepresented as Rapier skin.

use rapier3d::prelude::*;
use robot_minigame::{
    RobotError,
    basis::{engine_to_source_vector, source_to_engine_vector},
};
use serde::{Deserialize, Serialize};

use crate::SimulationWorld;

pub const ARENA_COMMIT: &str = "8b4a3a47fc53de23e8205089d71109a2e2348acd";
pub const ENVIRONMENT_SHA256: &str =
    "50b18ec12fe642ecb478ddf223c9d5c9a3fe265ce4e0b93869c26ece88eeb177";
const SIZE_SOURCE: [f32; 3] = [0.8, 1.5, 0.04];
const CENTER_SOURCE: [f32; 3] = [0.62, 0., -0.05];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct T1SourceShelfConfig {
    pub source_arena_commit: String,
    pub source_environment_sha256: String,
    /// One translation of the entire source environment, not an object correction.
    pub environment_translation_source: [f64; 3],
    /// Declared native material: source procedural support has no authored binding.
    pub native_friction: f32,
}

impl T1SourceShelfConfig {
    pub(super) fn validate(&self) -> Result<(), RobotError> {
        if self.source_arena_commit != ARENA_COMMIT
            || self.source_environment_sha256 != ENVIRONMENT_SHA256
            || self
                .environment_translation_source
                .iter()
                .any(|v| !v.is_finite() || !(*v as f32).is_finite())
            || (self.environment_translation_source[2] - 0.795).abs() > 1e-8
            || !self.native_friction.is_finite()
            || self.native_friction < 0.
        {
            return Err(invalid(
                "invalid original T1 shelf identity, floor translation or declared native material",
            ));
        }
        Ok(())
    }
}

pub(super) struct T1SourceShelf {
    body: RigidBodyHandle,
    pub(super) collider: ColliderHandle,
    native_friction: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct T1SourceShelfSample {
    pub source_arena_commit: &'static str,
    pub source_environment_sha256: &'static str,
    pub translation_engine: [f32; 3],
    pub rotation_engine_xyzw: [f32; 4],
    pub size_engine: [f32; 3],
    pub native_friction: f32,
    pub source_contact_offset_m: f32,
    pub native_prediction_distance_m: f32,
    pub fixed: bool,
}

impl T1SourceShelf {
    pub(super) fn insert(
        world: &mut SimulationWorld,
        config: &T1SourceShelfConfig,
    ) -> Result<Self, RobotError> {
        config.validate()?;
        if world.snapshot().integration_count != 0 {
            return Err(invalid(
                "source shelf can only be constructed before integration",
            ));
        }
        let source = std::array::from_fn(|i| {
            CENTER_SOURCE[i] + config.environment_translation_source[i] as f32
        });
        let body = world.world.bodies.insert(
            RigidBodyBuilder::fixed()
                .translation(Vector::from_array(source_to_engine_vector(source))),
        );
        let collider = world.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(
                SIZE_SOURCE[0] / 2.,
                SIZE_SOURCE[2] / 2.,
                SIZE_SOURCE[1] / 2.,
            )
            .friction(config.native_friction)
            .friction_combine_rule(CoefficientCombineRule::Average)
            .restitution(0.)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_2,
                Group::ALL,
                InteractionTestMode::And,
            )),
            body,
            &mut world.world.bodies,
        );
        Ok(Self {
            body,
            collider,
            native_friction: config.native_friction,
        })
    }

    pub(super) fn sample(
        &self,
        world: &SimulationWorld,
    ) -> Result<T1SourceShelfSample, RobotError> {
        let body = world
            .world
            .bodies
            .get(self.body)
            .ok_or_else(|| invalid("stale source shelf body"))?;
        let collider = world
            .world
            .colliders
            .get(self.collider)
            .ok_or_else(|| invalid("stale source shelf collider"))?;
        if body.body_type() != RigidBodyType::Fixed || collider.parent() != Some(self.body) {
            return Err(invalid("source shelf identity/body type changed"));
        }
        Ok(T1SourceShelfSample {
            source_arena_commit: ARENA_COMMIT,
            source_environment_sha256: ENVIRONMENT_SHA256,
            translation_engine: body.translation().to_array(),
            rotation_engine_xyzw: body.rotation().to_array(),
            size_engine: [SIZE_SOURCE[0], SIZE_SOURCE[2], SIZE_SOURCE[1]],
            native_friction: self.native_friction,
            source_contact_offset_m: 0.005,
            native_prediction_distance_m: world.world.integration_parameters.prediction_distance(),
            fixed: true,
        })
    }

    /// Zero-integration startup evidence. This is not a decision observation.
    pub(super) fn initial_robot_overlap(
        &self,
        world: &SimulationWorld,
    ) -> Result<Vec<serde_json::Value>, RobotError> {
        let shelf = &world.world.colliders[self.collider];
        // Cached collider world poses may await the first pipeline update.
        // Compose current owner body poses with immutable local collider poses.
        let pose = |collider: &Collider| -> Result<rapier3d::math::Pose, RobotError> {
            match collider.parent() {
                Some(parent) => Ok(*world
                    .world
                    .bodies
                    .get(parent)
                    .ok_or_else(|| invalid("stale startup collider parent"))?
                    .position()
                    * *collider
                        .position_wrt_parent()
                        .ok_or_else(|| invalid("missing startup collider local pose"))?),
                None => Ok(*collider.position()),
            }
        };
        let shelf_pose = pose(shelf)?;
        let mut overlaps = Vec::new();
        for (handle, robot) in world.world.colliders.iter() {
            if robot.collision_groups().memberships != Group::GROUP_1 {
                continue;
            }
            let contact = rapier3d::parry::query::contact(
                &pose(robot)?,
                robot.shape(),
                &shelf_pose,
                shelf.shape(),
                0.,
            )
            .map_err(|e| invalid(format!("initial robot/shelf contact query: {e:?}")))?;
            if let Some(contact) = contact {
                if contact.dist < 0. {
                    overlaps.push(serde_json::json!({"robot_collider":format!("{handle:?}"),"geometric_distance_m":contact.dist,"robot_point_source":engine_to_source_vector(contact.point1.to_array()),"support_point_source":engine_to_source_vector(contact.point2.to_array())}));
                }
            }
        }
        Ok(overlaps)
    }
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}
