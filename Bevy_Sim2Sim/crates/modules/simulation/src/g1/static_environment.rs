//! Prepared public environment geometry, installed once in the robot's owner.
//! Engine Y-up coordinates; no robot basis transform and no runtime pose setter.

use crate::SimulationWorld;
use rapier3d::prelude::*;
use robot_minigame::RobotError;
use serde::Serialize;

#[derive(Clone, Debug)]
pub enum StaticEnvironmentShape {
    Convex {
        vertices: Vec<[f32; 3]>,
    },
    Triangles {
        vertices: Vec<[f32; 3]>,
        indices: Vec<[u32; 3]>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct StaticEnvironmentIdentity {
    pub source: String,
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub layout_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct StaticEnvironmentReceipt {
    pub identity: StaticEnvironmentIdentity,
    pub coordinate_frame: &'static str,
    pub collider_count: usize,
    pub vertex_count: usize,
    pub original_broad_floor_removed: bool,
    pub preparation_integrations: u64,
    pub owner_integrations_at_installation: u64,
}

/// Immutable collision shapes prepared before the physical clock starts.
/// Rendering and physics receive geometry from the same checked source read.
#[derive(Clone, Debug)]
pub struct PreparedStaticEnvironment {
    colliders: Vec<Collider>,
    identity: StaticEnvironmentIdentity,
    vertex_count: usize,
}

impl PreparedStaticEnvironment {
    pub fn prepare(
        identity: StaticEnvironmentIdentity,
        shapes: Vec<StaticEnvironmentShape>,
        friction: f32,
    ) -> Result<Self, RobotError> {
        if identity.source.is_empty()
            || [
                &identity.model_sha256,
                &identity.manifest_sha256,
                &identity.layout_sha256,
            ]
            .iter()
            .any(|s| {
                s.len() != 64
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
            || !friction.is_finite()
            || friction < 0.
            || shapes.is_empty()
            || shapes.len() > 8192
        {
            return Err(invalid(
                "invalid public static environment identity/material/budget",
            ));
        }
        let mut colliders = Vec::with_capacity(shapes.len());
        let mut vertex_count = 0usize;
        for shape in shapes {
            let (vertices, indices) = match shape {
                StaticEnvironmentShape::Convex { vertices } => (vertices, None),
                StaticEnvironmentShape::Triangles { vertices, indices } => {
                    (vertices, Some(indices))
                }
            };
            vertex_count = vertex_count
                .checked_add(vertices.len())
                .ok_or_else(|| invalid("environment vertex count overflow"))?;
            if vertices.len() < 3
                || vertex_count > 2_000_000
                || !vertices.iter().flatten().all(|v| v.is_finite())
            {
                return Err(invalid("invalid or excessive public environment vertices"));
            }
            let points = vertices
                .into_iter()
                .map(Vector::from_array)
                .collect::<Vec<_>>();
            let builder = match indices {
                Some(indices) => {
                    if indices.is_empty()
                        || indices.len() > 2_000_000
                        || indices
                            .iter()
                            .flatten()
                            .any(|i| *i as usize >= points.len())
                    {
                        return Err(invalid("invalid public environment triangle coverage"));
                    }
                    ColliderBuilder::trimesh(points, indices)
                        .map_err(|e| invalid(format!("environment triangle mesh: {e:?}")))?
                }
                None => ColliderBuilder::convex_hull(&points)
                    .ok_or_else(|| invalid("invalid public environment convex hull"))?,
            };
            let mut collider = builder
                .friction(friction)
                .restitution(0.)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_2,
                    Group::ALL,
                    InteractionTestMode::And,
                ))
                .build();
            // Materialize immutable mass properties during preparation, not
            // inside the first timed physics boundary.
            collider.set_mass_properties(collider.mass_properties());
            colliders.push(collider);
        }
        Ok(Self {
            colliders,
            identity,
            vertex_count,
        })
    }

    pub(super) fn replace_startup_floor(
        &self,
        world: &mut SimulationWorld,
        floor: RigidBodyHandle,
    ) -> Result<StaticEnvironmentReceipt, RobotError> {
        if world.integration_count != 0
            || world
                .world
                .bodies
                .get(floor)
                .is_none_or(|b| !b.is_fixed() || b.colliders().len() != 1)
        {
            return Err(invalid(
                "environment installation requires the original untouched startup floor",
            ));
        }
        let physics = &mut world.world;
        physics.bodies.remove(
            floor,
            &mut physics.islands,
            &mut physics.colliders,
            &mut physics.impulse_joints,
            &mut physics.multibody_joints,
            true,
        );
        let owner = physics.bodies.insert(RigidBodyBuilder::fixed());
        for collider in &self.colliders {
            physics
                .colliders
                .insert_with_parent(collider.clone(), owner, &mut physics.bodies);
        }
        Ok(StaticEnvironmentReceipt {
            identity: self.identity.clone(),
            coordinate_frame: "right_handed_y_up_meters",
            collider_count: self.colliders.len(),
            vertex_count: self.vertex_count,
            original_broad_floor_removed: true,
            preparation_integrations: 0,
            owner_integrations_at_installation: world.integration_count,
        })
    }
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prepared() -> PreparedStaticEnvironment {
        PreparedStaticEnvironment::prepare(
            StaticEnvironmentIdentity {
                source: "public station fixture".into(),
                model_sha256: "1".repeat(64),
                manifest_sha256: "2".repeat(64),
                layout_sha256: "3".repeat(64),
            },
            vec![StaticEnvironmentShape::Triangles {
                vertices: vec![[-2., 0., -2.], [-2., 0., 2.], [2., 0., 2.], [2., 0., -2.]],
                indices: vec![[0, 1, 2], [0, 2, 3]],
            }],
            0.5,
        )
        .unwrap()
    }
    #[test]
    fn installation_replaces_floor_without_integrating_and_preserves_other_bodies() {
        let mut world = SimulationWorld::with_game_frequency(50).unwrap();
        let floor = world.world.bodies.insert(RigidBodyBuilder::fixed());
        world.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(20., 0.25, 20.),
            floor,
            &mut world.world.bodies,
        );
        let other = world
            .world
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0., 1., 0.)));
        let pose = *world.world.bodies[other].position();
        let receipt = prepared().replace_startup_floor(&mut world, floor).unwrap();
        assert_eq!(world.integration_count, 0);
        assert_eq!(*world.world.bodies[other].position(), pose);
        assert!(!world.world.bodies.contains(floor));
        assert_eq!(world.counts().bodies, 2);
        assert_eq!(world.counts().colliders, 1);
        assert!(receipt.original_broad_floor_removed);
        assert_eq!(world.world.integration_parameters.num_solver_iterations, 1);
    }
    #[test]
    fn imported_public_ground_supports_a_free_body_in_the_same_fifty_hz_world() {
        let mut world = SimulationWorld::with_game_frequency(50).unwrap();
        let floor = world.world.bodies.insert(RigidBodyBuilder::fixed());
        world.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(20., 0.25, 20.),
            floor,
            &mut world.world.bodies,
        );
        prepared().replace_startup_floor(&mut world, floor).unwrap();
        let object = world
            .world
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0., 1., 0.)));
        world.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.1, 0.1, 0.1),
            object,
            &mut world.world.bodies,
        );
        for tick in 1..=60 {
            assert_eq!(
                world.step_with_torques(&[]).unwrap().integration_count,
                tick
            );
        }
        let body = &world.world.bodies[object];
        assert!((0.08..0.12).contains(&body.translation().y));
        assert!(body.linvel().length() < 0.01);
        assert_eq!(world.counts().bodies, 2);
        assert_eq!(world.counts().colliders, 2);
        assert_eq!(world.configuration().physics_hz, 50);
        assert!(
            world
                .world
                .narrow_phase
                .contact_pairs()
                .any(|c| c.has_any_active_contact())
        );
        assert!(prepared().replace_startup_floor(&mut world, floor).is_err());
        assert_eq!(world.integration_count, 60);
    }
}
