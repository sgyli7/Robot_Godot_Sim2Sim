//! One Rapier world and the fixed-step boundary shared by game and verification.

pub mod fixed_step_runtime;
pub mod goose;
pub mod robot_builder;
pub mod source_collision;

use std::collections::HashMap;

use bevy_ecs::{entity::Entity, resource::Resource};
use rapier3d::prelude::*;
use serde::Serialize;
use thiserror::Error;

/// Default MicroDuck physical step rate. Other clocks use [`PhysicsClockProfile`].
pub const PHYSICS_HZ: u32 = 60;
/// Maximum overdue physics boundaries consumed by a display frame.
pub const MAX_STEPS_PER_FRAME: u32 = 8;
/// Default MicroDuck step length. Live steps use [`PhysicsClockProfile::dt`].
///
/// Rapier uses f32; episode time is separately derived from integer tick counts.
pub const PHYSICS_DT: f32 = 1.0 / PHYSICS_HZ as f32;

/// Native integration clock selected for one [`SimulationWorld`].
///
/// One `step_with_torques` call is one native step of [`Self::dt`]. Selecting
/// [`Self::Goose50`] does not qualify Goose contact, actuation, or training physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PhysicsClockProfile {
    /// Default MicroDuck clock: one native integration of [`PHYSICS_DT`].
    MicroDuck60,
    /// Goose clock: one native integration of 20 ms.
    Goose50,
}

impl PhysicsClockProfile {
    // Goose native steps per second; one step is 20 ms.
    const GOOSE_HZ: u32 = 50;

    /// Selected steps per second.
    pub const fn rate_hz(self) -> u32 {
        match self {
            Self::MicroDuck60 => PHYSICS_HZ,
            Self::Goose50 => Self::GOOSE_HZ,
        }
    }

    /// Native integration length, in seconds: `1 / rate_hz`.
    pub const fn dt(self) -> f32 {
        1.0 / self.rate_hz() as f32
    }
}

/// A queued contribution to a body's world-space external torque.
#[derive(Clone, Copy, Debug)]
pub struct BodyTorque {
    pub body: RigidBodyHandle,
    pub world_torque: [f32; 3],
}

impl BodyTorque {
    /// Produce equal/opposite contributions without calculating an actuator law.
    pub fn joint_pair(
        parent: RigidBodyHandle,
        child: RigidBodyHandle,
        child_torque: [f32; 3],
    ) -> [Self; 2] {
        [
            Self {
                body: parent,
                world_torque: child_torque.map(|x| -x),
            },
            Self {
                body: child,
                world_torque: child_torque,
            },
        ]
    }
}

/// All persistent backend sets, including the actual joint handles.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct WorldCounts {
    pub bodies: usize,
    pub colliders: usize,
    pub impulse_joints: usize,
    pub multibody_joint_handles: usize,
    pub entity_mappings: usize,
}

/// Evidence of settings that otherwise hide solver time subdivision.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct StepConfiguration {
    /// Profile selected at world construction.
    pub profile: PhysicsClockProfile,
    /// Live native integration length.
    pub dt: f32,
    /// Selected profile rate in hertz, not a rate inferred from a mutated `dt`.
    pub physics_hz: u32,
    pub num_solver_iterations: usize,
    /// Solver convergence passes inside one native step. Rapier does not subdivide `dt` by this count.
    pub num_internal_pgs_iterations: usize,
    pub max_ccd_substeps: usize,
    pub additional_solver_iterations_max: usize,
}

/// Body state uses explicit f32 arrays across the Bevy/Rapier math boundary.
#[derive(Clone, Debug, Serialize)]
pub struct BodySample {
    pub handle: [u32; 2],
    pub entity_bits: Option<u64>,
    pub translation: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub user_force: [f32; 3],
    pub user_torque: [f32; 3],
    pub dynamic: bool,
}

/// A completed physics boundary, without claims about policy inference.
#[derive(Clone, Debug, Serialize)]
pub struct StepSnapshot {
    pub global_step: u64,
    pub episode_id: u64,
    pub episode_step: u64,
    pub global_seconds: f64,
    pub episode_seconds: f64,
    pub integration_count: u64,
    pub torque_update_count: u64,
    /// Live graph pairs; a pair may have no active contacts yet.
    pub contact_pair_count: usize,
    pub active_contact_pair_count: usize,
    /// Raw narrow-phase graph size, including queued removals.
    pub raw_contact_pair_count: usize,
    /// Generation-obsolete graph entries are observable, never exposed as live.
    pub discarded_obsolete_contact_pair_count: usize,
    pub bodies: Vec<BodySample>,
}

/// Rapier's own timers for one completed physics step, sampled only in a
/// development profiling run. Values are nanoseconds. Zero may indicate an
/// inactive or unused stage, or unavailable/disabled `profiler` counters.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RapierCounterSample {
    pub step_ns: u64,
    pub broad_phase_ns: u64,
    pub final_broad_phase_ns: u64,
    pub narrow_phase_ns: u64,
    pub islands_ns: u64,
    pub constraints_ns: u64,
    pub solver_ns: u64,
    /// Legacy stage timer; this pipeline does not populate it. Use `ccd_toi_ns`.
    pub ccd_ns: u64,
    /// Actual TOI/motion-clamping timer used by this Rapier pipeline.
    pub ccd_toi_ns: u64,
    /// Native substep update timer, including mass-property maintenance.
    pub update_ns: u64,
    /// Native collision-detection timer across the pipeline's detection passes.
    pub collision_detection_ns: u64,
    pub user_changes_ns: u64,
}

/// A failed boundary is terminal for that verification candidate.
#[derive(Debug, Error)]
pub enum SimulationError {
    #[error("unknown rigid-body handle {0:?}")]
    UnknownBody(RigidBodyHandle),
    #[error("external torque or its sum is non-finite")]
    NonFiniteTorque,
    #[error("simulation settings violate the selected single-step clock contract")]
    InvalidStepConfiguration,
    #[error("Rapier quarantined a non-finite body or collider")]
    QuarantinedState,
    #[error("body {0:?} has non-finite position or velocity")]
    NonFiniteState(RigidBodyHandle),
    #[error("probe has not been created")]
    MissingProbe,
}

/// Sole physics-world owner. The renderer receives its snapshots, not a world.
#[derive(Resource)]
pub struct SimulationWorld {
    pub world: PhysicsWorld,
    clock_profile: PhysicsClockProfile,
    entity_by_body: HashMap<RigidBodyHandle, Entity>,
    probe: Option<RigidBodyHandle>,
    global_step: u64,
    episode_id: u64,
    episode_step: u64,
    integration_count: u64,
    torque_update_count: u64,
}

impl Default for SimulationWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationWorld {
    /// Opt in to Rapier's existing stage timers for development measurements.
    pub fn enable_rapier_counters(&mut self) {
        self.world.physics_pipeline.counters.enable();
    }

    /// Sample the timers Rapier reset and populated during its latest step.
    pub fn rapier_counter_sample(&self) -> RapierCounterSample {
        let counters = &self.world.physics_pipeline.counters;
        let ns =
            |duration: std::time::Duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        RapierCounterSample {
            step_ns: ns(counters.step_time.time()),
            broad_phase_ns: ns(counters.cd.broad_phase_time.time()),
            final_broad_phase_ns: ns(counters.cd.final_broad_phase_time.time()),
            narrow_phase_ns: ns(counters.cd.narrow_phase_time.time()),
            islands_ns: ns(counters.stages.island_construction_time.time()),
            constraints_ns: ns(counters.stages.island_constraints_collection_time.time()),
            solver_ns: ns(counters.stages.solver_time.time()),
            ccd_ns: ns(counters.stages.ccd_time.time()),
            ccd_toi_ns: ns(counters.ccd.toi_computation_time.time()),
            update_ns: ns(counters.stages.update_time.time()),
            collision_detection_ns: ns(counters.stages.collision_detection_time.time()),
            user_changes_ns: ns(counters.stages.user_changes.time()),
        }
    }

    /// Construct the world on the default MicroDuck 60 Hz clock.
    pub fn new() -> Self {
        Self::new_with_profile(PhysicsClockProfile::MicroDuck60)
    }

    /// Construct the world on an explicit clock. Native `dt` is `1 / rate_hz`.
    ///
    /// Both profiles set one solver iteration and one CCD substep. Rapier
    /// subdivides physical time when those counts, or any body's extra solver
    /// iterations, rise above this single-step guard.
    pub fn new_with_profile(profile: PhysicsClockProfile) -> Self {
        let mut world = PhysicsWorld::new();
        world.integration_parameters.dt = profile.dt();
        world.integration_parameters.num_solver_iterations = 1;
        world.integration_parameters.max_ccd_substeps = 1;
        Self {
            world,
            clock_profile: profile,
            entity_by_body: HashMap::new(),
            probe: None,
            global_step: 0,
            episode_id: 0,
            episode_step: 0,
            integration_count: 0,
            torque_update_count: 0,
        }
    }

    /// Clock chosen at construction. Later edits to integration parameters do not change it.
    pub fn clock_profile(&self) -> PhysicsClockProfile {
        self.clock_profile
    }

    /// Foundation fixture: one fixed floor and one sphere, no MicroDuck policy.
    pub fn foundation() -> Self {
        Self::foundation_with_profile(PhysicsClockProfile::MicroDuck60)
    }

    /// Same fixture bodies as [`Self::foundation`], on an explicit clock profile.
    pub(crate) fn foundation_with_profile(profile: PhysicsClockProfile) -> Self {
        let mut simulation = Self::new_with_profile(profile);
        let floor = simulation.world.bodies.insert(
            RigidBodyBuilder::fixed()
                .translation(Vector::new(0.0, -0.25, 0.0))
                .additional_solver_iterations(0),
        );
        simulation.world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(10.0, 0.25, 10.0),
            floor,
            &mut simulation.world.bodies,
        );
        simulation.probe = Some(simulation.insert_probe());
        simulation
    }

    fn insert_probe(&mut self) -> RigidBodyHandle {
        let handle = self.world.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 1.5, 0.0))
                .can_sleep(false)
                .additional_solver_iterations(0),
        );
        self.world.colliders.insert_with_parent(
            ColliderBuilder::ball(0.25).density(1.0),
            handle,
            &mut self.world.bodies,
        );
        handle
    }

    /// Return the real backend handle for the development fixture.
    pub fn probe_handle(&self) -> Result<RigidBodyHandle, SimulationError> {
        self.probe.ok_or(SimulationError::MissingProbe)
    }

    /// Register a live ECS mapping; stale backend handles cannot be registered.
    pub fn map_entity(
        &mut self,
        body: RigidBodyHandle,
        entity: Entity,
    ) -> Result<(), SimulationError> {
        if self.world.bodies.get(body).is_none() {
            return Err(SimulationError::UnknownBody(body));
        }
        self.entity_by_body.insert(body, entity);
        Ok(())
    }

    /// Look up only an existing backend body, guarding against recycled indices.
    pub fn mapped_entity(&self, body: RigidBodyHandle) -> Option<Entity> {
        self.world.bodies.get(body)?;
        self.entity_by_body.get(&body).copied()
    }

    /// Remove a body, its colliders/joints, mapping, and narrow-phase entries.
    ///
    /// Match `PhysicsWorld::step`'s default empty event handler. Pending collider
    /// removal notifications stay queued for the next genuine broad-phase step.
    pub fn remove_body(&mut self, body: RigidBodyHandle) -> Result<(), SimulationError> {
        let removed_colliders = self
            .world
            .bodies
            .get(body)
            .ok_or(SimulationError::UnknownBody(body))?
            .colliders()
            .to_vec();
        if self.world.remove_body(body).is_none() {
            return Err(SimulationError::UnknownBody(body));
        }
        self.entity_by_body.remove(&body);
        let PhysicsWorld {
            islands,
            narrow_phase,
            colliders,
            bodies,
            ..
        } = &mut self.world;
        narrow_phase.handle_user_changes(
            Some(islands),
            &[],
            &removed_colliders,
            colliders,
            bodies,
            &(),
        );
        Ok(())
    }

    /// Cold-rebuild only the fixture body; floor and other world objects survive.
    pub fn reset_probe(&mut self) -> Result<RigidBodyHandle, SimulationError> {
        let old = self.probe_handle()?;
        self.remove_body(old)?;
        let fresh = self.insert_probe();
        self.probe = Some(fresh);
        self.episode_id += 1;
        self.episode_step = 0;
        Ok(fresh)
    }

    /// Retain observability of the real backend collections across resets.
    pub fn counts(&self) -> WorldCounts {
        WorldCounts {
            bodies: self.world.bodies.len(),
            colliders: self.world.colliders.len(),
            impulse_joints: self.world.impulse_joints.len(),
            multibody_joint_handles: self.world.multibody_joints.iter().count(),
            entity_mappings: self.entity_by_body.len(),
        }
    }

    /// Read actual solver settings, including per-body extra time substeps.
    ///
    /// `physics_hz` is the selected profile rate. `dt` and the PGS count are the
    /// live integration parameters.
    pub fn configuration(&self) -> StepConfiguration {
        let p = &self.world.integration_parameters;
        let profile = self.clock_profile;
        StepConfiguration {
            profile,
            dt: p.dt,
            physics_hz: profile.rate_hz(),
            num_solver_iterations: p.num_solver_iterations,
            num_internal_pgs_iterations: p.num_internal_pgs_iterations,
            max_ccd_substeps: p.max_ccd_substeps,
            additional_solver_iterations_max: self
                .world
                .bodies
                .iter()
                .map(|(_, b)| b.additional_solver_iterations())
                .max()
                .unwrap_or(0),
        }
    }

    fn validate_configuration(&self) -> Result<(), SimulationError> {
        let p = self.configuration();
        if p.dt != self.clock_profile.dt()
            || p.physics_hz != self.clock_profile.rate_hz()
            || p.num_solver_iterations != 1
            || p.max_ccd_substeps != 1
            || p.additional_solver_iterations_max != 0
        {
            return Err(SimulationError::InvalidStepConfiguration);
        }
        Ok(())
    }

    /// Clear old external torques, aggregate this boundary, and integrate once.
    ///
    /// All validation occurs before mutating the force queues. Actuator/BAM and
    /// policy calculations belong to the robot module and are not emulated here.
    pub fn step_with_torques(
        &mut self,
        contributions: &[BodyTorque],
    ) -> Result<StepSnapshot, SimulationError> {
        self.step_with_torques_and_hooks(contributions, &())
    }

    /// Internal seam for the guarded source wrapper. The public development
    /// world retains its original default hooks and the same physical boundary.
    pub(crate) fn step_with_torques_and_hooks(
        &mut self,
        contributions: &[BodyTorque],
        hooks: &dyn PhysicsHooks,
    ) -> Result<StepSnapshot, SimulationError> {
        self.validate_configuration()?;
        let mut summed: HashMap<RigidBodyHandle, Vector> = HashMap::new();
        for contribution in contributions {
            if self.world.bodies.get(contribution.body).is_none() {
                return Err(SimulationError::UnknownBody(contribution.body));
            }
            let torque = Vector::from_array(contribution.world_torque);
            if !torque.is_finite() {
                return Err(SimulationError::NonFiniteTorque);
            }
            let total = summed.entry(contribution.body).or_insert(Vector::ZERO);
            *total += torque;
            if !total.is_finite() {
                return Err(SimulationError::NonFiniteTorque);
            }
        }
        for (_, body) in self.world.bodies.iter_mut() {
            body.reset_torques(true);
        }
        for (handle, torque) in summed {
            self.world.bodies[handle].add_torque(torque, true);
        }
        self.torque_update_count += 1;
        self.world.step_with_events(hooks, &());
        self.integration_count += 1;
        if !self.world.quarantine().is_empty() {
            return Err(SimulationError::QuarantinedState);
        }
        for (handle, body) in self.world.bodies.iter() {
            if !body.position().is_finite()
                || !body.linvel().is_finite()
                || !body.angvel().is_finite()
            {
                return Err(SimulationError::NonFiniteState(handle));
            }
        }
        self.global_step += 1;
        self.episode_step += 1;
        Ok(self.snapshot())
    }

    /// Validate collider generations, current parents, and cached manifold parents.
    ///
    /// A reset may leave removal notifications queued until the next real step.
    /// Filtering does not claim that Rapier's internal graph has been flushed.
    pub fn is_live_contact_pair(&self, pair: &ContactPair) -> bool {
        let Some(first) = self.world.colliders.get(pair.collider1) else {
            return false;
        };
        let Some(second) = self.world.colliders.get(pair.collider2) else {
            return false;
        };
        let parent_is_live = |parent: Option<RigidBodyHandle>| {
            parent.is_none_or(|body| self.world.bodies.get(body).is_some())
        };
        if !parent_is_live(first.parent()) || !parent_is_live(second.parent()) {
            return false;
        }
        pair.manifolds.iter().all(|manifold| {
            manifold.data.rigid_body1 == first.parent()
                && manifold.data.rigid_body2 == second.parent()
                && parent_is_live(manifold.data.rigid_body1)
                && parent_is_live(manifold.data.rigid_body2)
        })
    }

    /// Expose only generation-valid contact pairs to consumers of this world.
    pub fn live_contact_pairs(&self) -> impl Iterator<Item = &ContactPair> {
        self.world
            .contact_pairs()
            .filter(|pair| self.is_live_contact_pair(pair))
    }

    /// Read all body states after a successful step or before the first step.
    pub fn snapshot(&self) -> StepSnapshot {
        let raw_contact_pair_count = self.world.contact_pairs().count();
        let contact_pair_count = self.live_contact_pairs().count();
        let active_contact_pair_count = self
            .live_contact_pairs()
            .filter(|pair| pair.has_any_active_contact())
            .count();
        let mut bodies: Vec<_> = self
            .world
            .bodies
            .iter()
            .map(|(handle, body)| {
                let (index, generation) = handle.into_raw_parts();
                BodySample {
                    handle: [index, generation],
                    entity_bits: self.entity_by_body.get(&handle).map(|e| e.to_bits()),
                    translation: body.translation().to_array(),
                    rotation_xyzw: body.rotation().to_array(),
                    linear_velocity: body.linvel().to_array(),
                    angular_velocity: body.angvel().to_array(),
                    user_force: body.user_force().to_array(),
                    user_torque: body.user_torque().to_array(),
                    dynamic: body.is_dynamic(),
                }
            })
            .collect();
        bodies.sort_by_key(|body| body.handle);
        let rate_hz = f64::from(self.clock_profile.rate_hz());
        StepSnapshot {
            global_step: self.global_step,
            episode_id: self.episode_id,
            episode_step: self.episode_step,
            global_seconds: self.global_step as f64 / rate_hz,
            episode_seconds: self.episode_step as f64 / rate_hz,
            integration_count: self.integration_count,
            torque_update_count: self.torque_update_count,
            contact_pair_count,
            active_contact_pair_count,
            raw_contact_pair_count,
            discarded_obsolete_contact_pair_count: raw_contact_pair_count - contact_pair_count,
            bodies,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torques_aggregate_and_do_not_accumulate_between_ticks() {
        let mut world = SimulationWorld::foundation();
        let body = world.probe_handle().unwrap();
        world
            .step_with_torques(&[
                BodyTorque {
                    body,
                    world_torque: [0.001, 0.0, 0.0],
                },
                BodyTorque {
                    body,
                    world_torque: [0.002, 0.0, 0.0],
                },
            ])
            .unwrap();
        assert!((world.world.bodies[body].user_torque().x - 0.003).abs() < 1e-8);
        world.step_with_torques(&[]).unwrap();
        assert_eq!(world.world.bodies[body].user_torque(), Vector::ZERO);
    }

    #[test]
    fn joint_pair_applies_equal_opposite_torque() {
        let mut world = SimulationWorld::foundation();
        let child = world.probe_handle().unwrap();
        let parent = world.world.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(4.0, 3.0, 0.0))
                .additional_mass(1.0),
        );
        let torques = BodyTorque::joint_pair(parent, child, [0.001, -0.002, 0.0]);
        world.step_with_torques(&torques).unwrap();
        assert_eq!(
            world.world.bodies[parent].user_torque(),
            -world.world.bodies[child].user_torque()
        );
    }

    #[test]
    fn unknown_and_nonfinite_torque_do_not_integrate() {
        let mut world = SimulationWorld::foundation();
        let body = world.probe_handle().unwrap();
        assert!(matches!(
            world.step_with_torques(&[BodyTorque {
                body,
                world_torque: [f32::NAN, 0.0, 0.0]
            }]),
            Err(SimulationError::NonFiniteTorque)
        ));
        world.reset_probe().unwrap();
        assert!(matches!(
            world.step_with_torques(&[BodyTorque {
                body,
                world_torque: [1.0, 0.0, 0.0]
            }]),
            Err(SimulationError::UnknownBody(_))
        ));
        assert_eq!(world.snapshot().integration_count, 0);
    }

    #[test]
    fn twenty_resets_preserve_scene_and_global_time() {
        let mut world = SimulationWorld::foundation();
        let counts = world.counts();
        world.step_with_torques(&[]).unwrap();
        for episode in 1..=20 {
            let previous = world.probe_handle().unwrap();
            let fresh = world.reset_probe().unwrap();
            assert_ne!(fresh, previous);
            assert!(world.world.bodies.get(previous).is_none());
            assert_eq!(world.counts(), counts);
            let state = world.snapshot();
            assert_eq!(state.global_step, 1);
            assert_eq!(state.episode_step, 0);
            assert_eq!(state.episode_id, episode);
            assert_eq!(world.world.bodies[fresh].linvel(), Vector::ZERO);
            assert_eq!(world.world.bodies[fresh].user_torque(), Vector::ZERO);
        }
    }

    #[test]
    fn six_hundred_physics_steps_remain_finite() {
        let mut world = SimulationWorld::foundation();
        for _ in 0..600 {
            world.step_with_torques(&[]).unwrap();
        }
        assert_eq!(world.snapshot().global_step, 600);
        assert_eq!(world.snapshot().integration_count, 600);
        assert_eq!(world.snapshot().torque_update_count, 600);
        assert_eq!(world.configuration().num_solver_iterations, 1);
    }

    #[test]
    fn accidental_time_substeps_are_rejected() {
        let mut world = SimulationWorld::foundation();
        world.world.integration_parameters.num_solver_iterations = 4;
        assert!(matches!(
            world.step_with_torques(&[]),
            Err(SimulationError::InvalidStepConfiguration)
        ));
        assert_eq!(world.snapshot().integration_count, 0);
    }

    #[test]
    fn default_microduck60_and_explicit_goose50_keep_one_native_step() {
        let fresh = SimulationWorld::new();
        assert_eq!(fresh.clock_profile(), PhysicsClockProfile::MicroDuck60);
        let fresh_cfg = fresh.configuration();
        assert_eq!(fresh_cfg.profile, PhysicsClockProfile::MicroDuck60);
        assert_eq!(fresh_cfg.physics_hz, PHYSICS_HZ);
        assert_eq!(fresh_cfg.dt, PHYSICS_DT);
        assert_eq!(fresh_cfg.dt, PhysicsClockProfile::MicroDuck60.dt());
        assert_eq!(fresh_cfg.num_solver_iterations, 1);
        assert_eq!(fresh_cfg.max_ccd_substeps, 1);
        assert_eq!(fresh_cfg.additional_solver_iterations_max, 0);

        let mut micro = SimulationWorld::foundation();
        let micro_pgs = micro.configuration().num_internal_pgs_iterations;
        assert_eq!(micro.clock_profile(), PhysicsClockProfile::MicroDuck60);
        let micro_step = micro.step_with_torques(&[]).unwrap();
        assert_eq!(micro_step.integration_count, 1);
        assert_eq!(micro_step.global_step, 1);
        assert_eq!(micro_step.global_seconds, 1.0 / f64::from(PHYSICS_HZ));
        assert_eq!(micro_step.episode_seconds, 1.0 / f64::from(PHYSICS_HZ));
        assert_eq!(micro.configuration().dt, PHYSICS_DT);
        assert_eq!(micro.configuration().num_solver_iterations, 1);
        assert_eq!(micro.configuration().num_internal_pgs_iterations, micro_pgs);

        let mut goose = SimulationWorld::foundation_with_profile(PhysicsClockProfile::Goose50);
        let goose_cfg = goose.configuration();
        assert_eq!(goose.clock_profile(), PhysicsClockProfile::Goose50);
        assert_eq!(goose_cfg.profile, PhysicsClockProfile::Goose50);
        assert_eq!(goose_cfg.physics_hz, PhysicsClockProfile::Goose50.rate_hz());
        assert_eq!(goose_cfg.physics_hz, 50);
        assert_eq!(goose_cfg.dt, PhysicsClockProfile::Goose50.dt());
        assert_eq!(goose_cfg.dt, 1.0 / 50.0);
        assert!((goose_cfg.dt - 0.02).abs() < 1.0e-6);
        assert_eq!(goose_cfg.num_solver_iterations, 1);
        assert_eq!(goose_cfg.max_ccd_substeps, 1);
        assert_eq!(goose_cfg.additional_solver_iterations_max, 0);
        let visible_pgs = goose
            .world
            .integration_parameters
            .num_internal_pgs_iterations;
        assert_eq!(goose_cfg.num_internal_pgs_iterations, visible_pgs);
        goose
            .world
            .integration_parameters
            .num_internal_pgs_iterations = 16;
        let before_y = goose.world.bodies[goose.probe_handle().unwrap()]
            .translation()
            .y;
        for _ in 0..50 {
            goose.step_with_torques(&[]).unwrap();
        }
        let snap = goose.snapshot();
        assert_eq!(snap.integration_count, 50);
        assert_eq!(snap.torque_update_count, 50);
        assert_eq!(snap.global_step, 50);
        assert_eq!(snap.episode_step, 50);
        assert_eq!(snap.global_seconds, 1.0);
        assert_eq!(snap.episode_seconds, 1.0);
        assert_eq!(goose.configuration().num_internal_pgs_iterations, 16);
        assert_eq!(goose.configuration().num_solver_iterations, 1);
        assert_eq!(goose.configuration().dt, PhysicsClockProfile::Goose50.dt());
        assert_eq!(goose.configuration().physics_hz, 50);
        assert!(
            goose.world.bodies[goose.probe_handle().unwrap()]
                .translation()
                .y
                < before_y
        );
    }

    #[test]
    fn mutated_dt_solver_count_and_per_body_extra_reject_integration() {
        fn expect_reject(world: &mut SimulationWorld) {
            assert!(matches!(
                world.step_with_torques(&[]),
                Err(SimulationError::InvalidStepConfiguration)
            ));
            assert_eq!(world.snapshot().integration_count, 0);
            assert_eq!(world.snapshot().global_step, 0);
            assert_eq!(world.snapshot().global_seconds, 0.0);
        }

        let mut goose = SimulationWorld::foundation_with_profile(PhysicsClockProfile::Goose50);
        goose.world.integration_parameters.dt = PHYSICS_DT;
        assert_eq!(goose.configuration().physics_hz, 50);
        assert_ne!(goose.configuration().dt, goose.clock_profile().dt());
        expect_reject(&mut goose);

        let mut goose = SimulationWorld::foundation_with_profile(PhysicsClockProfile::Goose50);
        goose.world.integration_parameters.num_solver_iterations = 4;
        expect_reject(&mut goose);

        let mut goose = SimulationWorld::foundation_with_profile(PhysicsClockProfile::Goose50);
        let body = goose.probe_handle().unwrap();
        goose.world.bodies[body].set_additional_solver_iterations(1);
        assert_eq!(goose.configuration().additional_solver_iterations_max, 1);
        expect_reject(&mut goose);

        let mut micro = SimulationWorld::foundation();
        micro.world.integration_parameters.dt = PhysicsClockProfile::Goose50.dt();
        assert_eq!(micro.configuration().physics_hz, PHYSICS_HZ);
        expect_reject(&mut micro);
    }

    #[test]
    fn twenty_contact_resets_clean_narrow_phase_and_new_generation_matches_fresh_world() {
        let mut world = SimulationWorld::foundation();
        for _ in 0..120 {
            world.step_with_torques(&[]).unwrap();
        }
        assert_eq!(world.snapshot().active_contact_pair_count, 1);
        let counts = world.counts();
        for episode in 1..=20 {
            let old_body = world.probe_handle().unwrap();
            let old_collider = world.world.bodies[old_body].colliders()[0];
            world.reset_probe().unwrap();
            let fresh_body = world.probe_handle().unwrap();
            let fresh_collider = world.world.bodies[fresh_body].colliders()[0];
            assert_ne!(old_collider, fresh_collider);
            assert!(world.world.colliders.get(old_collider).is_none());
            let reset = world.snapshot();
            assert_eq!(reset.integration_count, 120);
            assert_eq!(reset.episode_id, episode);
            assert_eq!(reset.episode_step, 0);
            assert_eq!(reset.contact_pair_count, 0);
            assert_eq!(reset.raw_contact_pair_count, 0);
            assert_eq!(reset.discarded_obsolete_contact_pair_count, 0);
            assert_eq!(world.live_contact_pairs().count(), 0);
            assert_eq!(world.counts(), counts);
        }
        let fresh_body = world.probe_handle().unwrap();
        let mut reference = SimulationWorld::foundation();
        let reference_body = reference.probe_handle().unwrap();
        for _ in 0..60 {
            world.step_with_torques(&[]).unwrap();
            reference.step_with_torques(&[]).unwrap();
            let actual = &world.world.bodies[fresh_body];
            let expected = &reference.world.bodies[reference_body];
            assert!(
                actual
                    .translation()
                    .abs_diff_eq(expected.translation(), 1e-6)
            );
            assert!(actual.rotation().abs_diff_eq(*expected.rotation(), 1e-6));
            assert!(actual.linvel().abs_diff_eq(expected.linvel(), 1e-6));
            assert!(actual.angvel().abs_diff_eq(expected.angvel(), 1e-6));
            assert_eq!(world.snapshot().discarded_obsolete_contact_pair_count, 0);
        }
        assert_eq!(world.snapshot().integration_count, 180);
        assert_eq!(world.snapshot().episode_step, 60);
    }

    #[test]
    fn public_narrow_phase_removal_handles_queued_pair_without_integration() {
        let mut world = SimulationWorld::foundation();
        for _ in 0..120 {
            world.step_with_torques(&[]).unwrap();
        }
        let old_body = world.probe_handle().unwrap();
        let removed = world.world.bodies[old_body].colliders().to_vec();
        // Deliberately bypass the wrapper to preserve evidence of raw deferred removal.
        world.world.remove_body(old_body).unwrap();
        assert_eq!(world.snapshot().raw_contact_pair_count, 1);
        assert_eq!(world.snapshot().contact_pair_count, 0);
        assert_eq!(world.snapshot().discarded_obsolete_contact_pair_count, 1);
        let PhysicsWorld {
            islands,
            narrow_phase,
            colliders,
            bodies,
            ..
        } = &mut world.world;
        // Preserve removal notifications for the next physics/broad-phase update.
        narrow_phase.handle_user_changes(Some(islands), &[], &removed, colliders, bodies, &());
        assert_eq!(world.snapshot().integration_count, 120);
        assert_eq!(world.snapshot().raw_contact_pair_count, 0);
        assert_eq!(world.snapshot().episode_step, 120);
        // The actual next step must tolerate replaying the pending removal.
        world.step_with_torques(&[]).unwrap();
        assert_eq!(world.snapshot().integration_count, 121);
        assert_eq!(world.snapshot().discarded_obsolete_contact_pair_count, 0);
    }
}
