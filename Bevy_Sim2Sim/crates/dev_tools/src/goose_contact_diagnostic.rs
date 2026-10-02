//! Explicit fresh-manifold diagnostics; no production dispatcher is replaced.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use rapier3d::{
    geometry::{ContactData, ContactManifold, ContactManifoldData},
    math::{Pose, Real, Vector},
    parry::{
        query::{
            ClosestPoints, Contact, ContactManifoldsWorkspace, DefaultQueryDispatcher,
            NonlinearRigidMotion, PersistentQueryDispatcher, QueryDispatcher, ShapeCastHit,
            ShapeCastOptions, Unsupported, details::NormalConstraints,
        },
        shape::Shape,
    },
};

/// Rebuild raw geometric manifolds and workspaces at every native pair query.
/// Raw manifold warmstart data is consequently discarded too; Rapier's later
/// cluster matching remains enabled. The paired diagnostic must report this.
pub struct FreshContactDispatcher {
    calls: Arc<AtomicU64>,
}

impl FreshContactDispatcher {
    /// Share an actual native query count with the development probe.
    pub fn new(calls: Arc<AtomicU64>) -> Self {
        Self { calls }
    }
}

impl QueryDispatcher for FreshContactDispatcher {
    fn intersection_test(
        &self,
        pose: &Pose,
        a: &dyn Shape,
        b: &dyn Shape,
    ) -> Result<bool, Unsupported> {
        DefaultQueryDispatcher.intersection_test(pose, a, b)
    }
    fn distance(&self, pose: &Pose, a: &dyn Shape, b: &dyn Shape) -> Result<Real, Unsupported> {
        DefaultQueryDispatcher.distance(pose, a, b)
    }
    fn contact(
        &self,
        pose: &Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        DefaultQueryDispatcher.contact(pose, a, b, prediction)
    }
    fn closest_points(
        &self,
        pose: &Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        max_distance: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        DefaultQueryDispatcher.closest_points(pose, a, b, max_distance)
    }
    fn cast_shapes(
        &self,
        pose: &Pose,
        velocity: Vector,
        a: &dyn Shape,
        b: &dyn Shape,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        DefaultQueryDispatcher.cast_shapes(pose, velocity, a, b, options)
    }
    fn cast_shapes_nonlinear(
        &self,
        motion_a: &NonlinearRigidMotion,
        a: &dyn Shape,
        motion_b: &NonlinearRigidMotion,
        b: &dyn Shape,
        start: Real,
        end: Real,
        stop: bool,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        DefaultQueryDispatcher.cast_shapes_nonlinear(motion_a, a, motion_b, b, start, end, stop)
    }
}

impl PersistentQueryDispatcher<ContactManifoldData, ContactData> for FreshContactDispatcher {
    fn contact_manifolds(
        &self,
        pose: &Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold>,
        workspace: &mut Option<ContactManifoldsWorkspace>,
    ) -> Result<(), Unsupported> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        manifolds.clear();
        *workspace = None;
        DefaultQueryDispatcher.contact_manifolds(pose, a, b, prediction, manifolds, workspace)
    }
    fn contact_manifold_convex_convex(
        &self,
        pose: &Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        normals_a: Option<&dyn NormalConstraints>,
        normals_b: Option<&dyn NormalConstraints>,
        prediction: Real,
        manifold: &mut ContactManifold,
    ) -> Result<(), Unsupported> {
        DefaultQueryDispatcher
            .contact_manifold_convex_convex(pose, a, b, normals_a, normals_b, prediction, manifold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapier3d::prelude::SharedShape;

    #[test]
    fn fresh_compound_query_discards_stale_geometry_and_matches_independent_query() {
        let shape =
            SharedShape::compound(vec![(Pose::IDENTITY, SharedShape::cuboid(1.0, 1.0, 1.0))]);
        let calls = Arc::new(AtomicU64::new(0));
        let dispatcher = FreshContactDispatcher::new(calls.clone());
        let mut cached = Vec::<ContactManifold>::new();
        let mut workspace = None;
        DefaultQueryDispatcher
            .contact_manifolds(
                &Pose::from_translation(Vector::X * 2.005),
                shape.as_ref(),
                shape.as_ref(),
                0.02,
                &mut cached,
                &mut workspace,
            )
            .unwrap();
        assert!(workspace.is_some());
        assert!(cached.iter().any(|m| !m.points.is_empty()));
        for point in cached.iter_mut().flat_map(|m| &mut m.points) {
            point.dist = -1.0;
        }
        let pose = Pose::from_translation(Vector::X * 2.01);
        dispatcher
            .contact_manifolds(
                &pose,
                shape.as_ref(),
                shape.as_ref(),
                0.02,
                &mut cached,
                &mut workspace,
            )
            .unwrap();
        let mut independent = Vec::<ContactManifold>::new();
        DefaultQueryDispatcher
            .contact_manifolds(
                &pose,
                shape.as_ref(),
                shape.as_ref(),
                0.02,
                &mut independent,
                &mut None,
            )
            .unwrap();
        let distances: Vec<_> = cached
            .iter()
            .flat_map(|m| &m.points)
            .map(|p| p.dist)
            .collect();
        let expected: Vec<_> = independent
            .iter()
            .flat_map(|m| &m.points)
            .map(|p| p.dist)
            .collect();
        assert!(!distances.is_empty());
        assert_eq!(distances, expected);
        assert!(distances.iter().all(|d| *d > 0.0099 && *d < 0.0101));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
