//! Private, single-world collision admission for the two audited robot families.
//!
//! Environment objects, sensors and custom modifiers have no admitted source
//! contract here. Registration is not contact/CCD/force or target-plant approval.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use rapier3d::prelude::*;
use robot_minigame::{
    RobotError,
    collision_profile::SourceCollisionProfile,
    definition::RobotDefinition,
    joint_feedback::{JointFeedbackFrame, PreviousSolveLoad},
    kinematics::source_body_poses,
};
use serde::Serialize;
use thiserror::Error;

use crate::{
    BodyTorque, SimulationError, SimulationWorld, StepSnapshot, WorldCounts,
    robot_builder::{RobotAssembly, build_structure},
};

static NEXT_WORLD_EPOCH: AtomicU64 = AtomicU64::new(1);
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "sim2sim_observation")]
mod previous_solve;
#[cfg(feature = "sim2sim_observation")]
pub use previous_solve::{
    NativeSolveCandidateChannel, NativeSolveCandidateFrame, PreviousSolveDiagnostic,
    PreviousSolveUnavailable,
};
const SOURCE_HOOKS: ActiveHooks = ActiveHooks::from_bits_retain(
    ActiveHooks::FILTER_CONTACT_PAIRS.bits() | ActiveHooks::FILTER_INTERSECTION_PAIR.bits(),
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RobotInstanceToken {
    world_epoch: u64,
    instance: u64,
}

#[derive(Debug, Error)]
pub enum SourceCollisionError {
    #[error("source collision boundary: {0}")]
    Boundary(String),
    #[error(transparent)]
    Robot(#[from] RobotError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
}

type BoundaryResult<T> = Result<T, SourceCollisionError>;

#[derive(Debug, Serialize)]
pub struct CollisionRegistrationReport {
    pub scope: &'static str,
    pub token: RobotInstanceToken,
    pub family: String,
    pub definition_file_sha256: String,
    pub profile_file_sha256: String,
    pub native_semantic_sha256: String,
    pub counts: WorldCounts,
    pub full_source_geom_pairs: usize,
    pub source_eligible_pairs: usize,
    pub eligible_source_geom_pairs: Vec<[usize; 2]>,
    pub backend_rejected_source_eligible_pairs: usize,
    pub integration_count: u64,
    pub torque_update_count: u64,
    pub environment_supported: bool,
    pub contact_or_ccd_qualified: bool,
    pub force_or_bam_qualified: bool,
    pub target_plant_accepted: bool,
}

struct RegisteredBody {
    token: RobotInstanceToken,
    source_body: usize,
    attached: HashSet<ColliderHandle>,
}

struct RegisteredCollider {
    token: RobotInstanceToken,
    source_geom: usize,
    parent: RigidBodyHandle,
    shape: SharedShape,
    local_pose: Pose,
}

/// Owns its sole SimulationWorld privately. No raw, mutable or consuming world
/// escape hatch exists; callers receive immutable value snapshots and tokens.
pub struct SourceCollisionWorld {
    simulation: SimulationWorld,
    definition: Arc<RobotDefinition>,
    profile: Arc<SourceCollisionProfile>,
    epoch: u64,
    token: Option<RobotInstanceToken>,
    assembly: Option<RobotAssembly>,
    bodies: HashMap<RigidBodyHandle, RegisteredBody>,
    colliders: HashMap<ColliderHandle, RegisteredCollider>,
    #[cfg(test)]
    runtime_audit: Option<Arc<runtime_batch::Audit>>,
    #[cfg(test)]
    late_registry_fault: Option<ColliderHandle>,
}

impl SourceCollisionWorld {
    pub fn new(
        definition: Arc<RobotDefinition>,
        profile: Arc<SourceCollisionProfile>,
        qpos: &[f64],
    ) -> BoundaryResult<Self> {
        if profile.data().definition_file_sha256 != definition.file_sha256()
            || profile.data().family != definition.model().family
        {
            return Err(invalid("profile and definition identity differ"));
        }
        validate_qpos(&definition, qpos)?;
        let epoch = allocate(&NEXT_WORLD_EPOCH)?;
        let mut world = Self {
            simulation: SimulationWorld::new(),
            definition,
            profile,
            epoch,
            token: None,
            assembly: None,
            bodies: HashMap::new(),
            colliders: HashMap::new(),
            #[cfg(test)]
            runtime_audit: None,
            #[cfg(test)]
            late_registry_fault: None,
        };
        world.build_cold(qpos)?;
        Ok(world)
    }

    pub fn instance_token(&self) -> BoundaryResult<RobotInstanceToken> {
        self.token
            .ok_or_else(|| invalid("robot instance is not committed"))
    }

    pub fn snapshot(&self) -> StepSnapshot {
        self.simulation.snapshot()
    }

    /// Read all 14 driven joints from the registered instance and this world's
    /// current tick. Native previous-solve load is explicitly unavailable: the
    /// current read-only Rapier primitives do not qualify BAM feedback.
    pub fn joint_feedback_frame(
        &self,
        token: RobotInstanceToken,
    ) -> BoundaryResult<JointFeedbackFrame> {
        self.validate_boundary(token)?;
        let assembly = self
            .assembly
            .as_ref()
            .ok_or_else(|| invalid("joint feedback assembly is absent"))?;
        let channels =
            assembly.actuator_joint_feedback(&self.simulation.world, &self.definition)?;
        let snapshot = self.simulation.snapshot();
        Ok(JointFeedbackFrame {
            model_file_sha256: self.definition.file_sha256().into(),
            collision_profile_file_sha256: self.profile.file_sha256().into(),
            world_epoch: token.world_epoch,
            robot_instance: token.instance,
            episode_id: snapshot.episode_id,
            global_step: snapshot.global_step,
            episode_step: snapshot.episode_step,
            channels,
            previous_solve_load: PreviousSolveLoad::Unavailable,
        })
    }

    pub fn counts(&self) -> WorldCounts {
        self.simulation.counts()
    }

    pub fn source_body_handle(
        &self,
        token: RobotInstanceToken,
        source_body: usize,
    ) -> BoundaryResult<RigidBodyHandle> {
        self.check_token(token)?;
        self.bodies
            .iter()
            .find_map(|(handle, entry)| (entry.source_body == source_body).then_some(*handle))
            .ok_or_else(|| invalid("source body is absent"))
    }

    /// Full live inventory and all native source pairs, independent of sleep,
    /// enabled state, position, broadphase contacts or hook invocation counts.
    pub fn validate_boundary(
        &self,
        token: RobotInstanceToken,
    ) -> BoundaryResult<CollisionRegistrationReport> {
        self.check_token(token)?;
        self.simulation.validate_configuration()?;
        let world = &self.simulation.world;
        let model = self.definition.model();
        let data = self.profile.data();
        let assembly = self
            .assembly
            .as_ref()
            .ok_or_else(|| invalid("assembly is absent"))?;
        if assembly.source_file_sha256() != self.definition.file_sha256()
            || data.definition_file_sha256 != self.definition.file_sha256()
            || self.bodies.len() != model.counts.nbody - 1
            || world.bodies.len() != self.bodies.len()
            || world.colliders.len() != self.colliders.len()
            || !world.impulse_joints.is_empty()
            || world.multibody_joints.iter().count() != model.counts.njnt - 1
            || self.simulation.counts().entity_mappings != 0
        {
            return Err(invalid(
                "world inventory, assembly or profile is incomplete",
            ));
        }
        let expected_geoms: HashSet<_> = (0..data.counts.ngeom)
            .filter(|geom| {
                data.geom_bodyid[*geom] != 0
                    && (data.geom_contype[*geom] != 0 || data.geom_conaffinity[*geom] != 0)
            })
            .collect();
        let actual_geoms: HashSet<_> = self
            .colliders
            .values()
            .map(|entry| entry.source_geom)
            .collect();
        if actual_geoms != expected_geoms
            || actual_geoms.len() != self.colliders.len()
            || assembly.collision_handles().len() != self.colliders.len()
        {
            return Err(invalid(
                "source geom registration is not complete and unique",
            ));
        }
        for (geom, handle) in assembly.collision_handles() {
            if self
                .colliders
                .get(handle)
                .is_none_or(|entry| entry.source_geom != *geom)
            {
                return Err(invalid(
                    "assembly collider generation or source mapping differs",
                ));
            }
        }
        let mut seen_bodies = HashSet::new();
        for (handle, body) in world.bodies.iter() {
            let entry = self
                .bodies
                .get(&handle)
                .ok_or_else(|| invalid("unregistered live body"))?;
            if entry.token != token
                || entry.source_body == 0
                || entry.source_body >= model.counts.nbody
                || !seen_bodies.insert(entry.source_body)
                || assembly.body_handles()[entry.source_body] != Some(handle)
                || body.user_data != entry.source_body as u128
                || body.body_type() != RigidBodyType::Dynamic
                || !body.is_enabled()
                || body.is_sleeping()
                || body.activation().normalized_linear_threshold != -1.0
                || body.activation().angular_threshold != -1.0
                || body.colliders().iter().copied().collect::<HashSet<_>>() != entry.attached
                || body.colliders().len() != entry.attached.len()
            {
                return Err(invalid(
                    "body generation, owner, inventory or backend flags differ",
                ));
            }
        }
        if seen_bodies != (1..model.counts.nbody).collect::<HashSet<_>>() {
            return Err(invalid("source body registration is incomplete"));
        }
        for (handle, collider) in world.colliders.iter() {
            let entry = self
                .colliders
                .get(&handle)
                .ok_or_else(|| invalid("unregistered live collider"))?;
            let owner = self
                .bodies
                .get(&entry.parent)
                .ok_or_else(|| invalid("collider owner is absent"))?;
            if entry.token != token
                || collider.parent() != Some(entry.parent)
                || data.geom_bodyid[entry.source_geom] != owner.source_body
                || !owner.attached.contains(&handle)
                || collider.user_data != entry.source_geom as u128
                || !collider.is_enabled()
                || collider.is_sensor()
                || collider.active_hooks() != SOURCE_HOOKS
                || collider.collision_groups() != InteractionGroups::all()
                || collider.solver_groups() != InteractionGroups::all()
                || collider.active_collision_types() != ActiveCollisionTypes::default()
                || collider.position_wrt_parent() != Some(&entry.local_pose)
                || !Arc::ptr_eq(&collider.shared_shape().0, &entry.shape.0)
            {
                return Err(invalid(
                    "collider generation, owner or backend configuration differs",
                ));
            }
        }
        let mut seen_joints = HashSet::new();
        for mapping in assembly.joint_mapping() {
            let (multibody, link_id) = world
                .multibody_joints
                .get(mapping.handle)
                .ok_or_else(|| invalid("joint generation is absent"))?;
            let link = multibody
                .link(link_id)
                .ok_or_else(|| invalid("joint link is absent"))?;
            let source = mapping.source_joint;
            if source == 0
                || source >= model.counts.njnt
                || !seen_joints.insert(source)
                || link.rigid_body_handle()
                    != assembly.body_handles()[model.fields.jnt_bodyid[source]]
                        .ok_or_else(|| invalid("joint body is absent"))?
                || link.joint().data.contacts_enabled()
                || !multibody.self_contacts_enabled()
            {
                return Err(invalid(
                    "joint mapping or self-contact configuration differs",
                ));
            }
            let parent = model.fields.body_parentid[model.fields.jnt_bodyid[source]];
            let parent_handle =
                assembly.body_handles()[parent].ok_or_else(|| invalid("joint parent is absent"))?;
            if world
                .multibody_joints
                .joint_between(parent_handle, link.rigid_body_handle())
                .is_none_or(|(handle, _, _)| handle != mapping.handle)
            {
                return Err(invalid("joint topology differs from source"));
            }
        }
        if seen_joints.len() != model.counts.njnt - 1 {
            return Err(invalid("source joint registration is incomplete"));
        }
        let geom_handles: HashMap<_, _> = self
            .colliders
            .iter()
            .map(|(handle, entry)| (entry.source_geom, *handle))
            .collect();
        let mut full_pairs = 0;
        let mut eligible = 0;
        let mut eligible_source_geom_pairs = Vec::new();
        for first in 0..data.counts.ngeom {
            for second in first + 1..data.counts.ngeom {
                full_pairs += 1;
                if self
                    .profile
                    .source_geom_pair_eligibility(first, second)?
                    .is_eligible()
                {
                    eligible += 1;
                    eligible_source_geom_pairs.push([first, second]);
                    let first_handle = *geom_handles
                        .get(&first)
                        .ok_or_else(|| invalid("eligible source geom lacks a collider"))?;
                    let second_handle = *geom_handles
                        .get(&second)
                        .ok_or_else(|| invalid("eligible source geom lacks a collider"))?;
                    if backend_rejects(world, first_handle, second_handle)? {
                        return Err(invalid(
                            "native backend gate rejects a source eligible pair",
                        ));
                    }
                }
            }
        }
        Ok(CollisionRegistrationReport {
            scope: "source_self_collision_registration_and_preintegration_boundary_only",
            token,
            family: data.family.clone(),
            definition_file_sha256: self.definition.file_sha256().into(),
            profile_file_sha256: self.profile.file_sha256().into(),
            native_semantic_sha256: self.profile.native_semantic_sha256().into(),
            counts: self.counts(),
            full_source_geom_pairs: full_pairs,
            source_eligible_pairs: eligible,
            eligible_source_geom_pairs,
            backend_rejected_source_eligible_pairs: 0,
            integration_count: self.simulation.integration_count,
            torque_update_count: self.simulation.torque_update_count,
            environment_supported: false,
            contact_or_ccd_qualified: false,
            force_or_bam_qualified: false,
            target_plant_accepted: false,
        })
    }

    /// The only guarded stepping path; no environment is admitted. Actual
    /// successful stepping/contact qualification is a separate verification gate.
    pub fn step_with_torques(
        &mut self,
        token: RobotInstanceToken,
        contributions: &[BodyTorque],
    ) -> BoundaryResult<StepSnapshot> {
        self.validate_boundary(token)?;
        #[cfg(test)]
        if let Some(handle) = self.late_registry_fault.take() {
            // Explicit test-only fault after successful full admission.
            self.colliders.remove(&handle);
        }
        let integrations_before = self.simulation.integration_count;
        let failure = AtomicBool::new(false);
        let hooks = SourceHooks {
            profile: &self.profile,
            colliders: &self.colliders,
            bodies: &self.bodies,
            token,
            failure: &failure,
            #[cfg(test)]
            audit: self.runtime_audit.as_deref(),
        };
        let result = self
            .simulation
            .step_with_torques_and_hooks(contributions, &hooks);
        if failure.load(Ordering::Relaxed) {
            // The pipeline has already been entered: never report zero steps
            // merely because its callback rejected an inconsistent pair.
            self.token = None;
            return Err(invalid(
                "hook detected an unexpected live registry inconsistency after pipeline entry",
            ));
        }
        if result.is_err() && self.simulation.integration_count != integrations_before {
            self.token = None;
        }
        result.map_err(Into::into)
    }

    pub fn remove_robot(&mut self, token: RobotInstanceToken) -> BoundaryResult<()> {
        self.validate_boundary(token)?;
        self.token = None;
        let assembly = self
            .assembly
            .take()
            .ok_or_else(|| invalid("assembly is absent"))?;
        self.bodies.clear();
        self.colliders.clear();
        for body in assembly.body_handles().iter().rev().flatten() {
            self.simulation.remove_body(*body)?;
        }
        Ok(())
    }

    pub fn rebuild_robot(&mut self, qpos: &[f64]) -> BoundaryResult<RobotInstanceToken> {
        validate_qpos(&self.definition, qpos)?;
        if self.token.is_some()
            || self.assembly.is_some()
            || !self.bodies.is_empty()
            || !self.colliders.is_empty()
            || self.simulation.world.bodies.len() != 0
            || self.simulation.world.colliders.len() != 0
            || !self.simulation.world.impulse_joints.is_empty()
            || self
                .simulation
                .world
                .multibody_joints
                .iter()
                .next()
                .is_some()
        {
            return Err(invalid(
                "cold rebuild requires a completely empty removed instance",
            ));
        }
        let episode = self
            .simulation
            .episode_id
            .checked_add(1)
            .ok_or_else(|| invalid("checked collision episode allocation exhausted"))?;
        self.build_cold(qpos)?;
        self.simulation.episode_id = episode;
        self.simulation.episode_step = 0;
        self.instance_token()
    }

    fn build_cold(&mut self, qpos: &[f64]) -> BoundaryResult<()> {
        let token = RobotInstanceToken {
            world_epoch: self.epoch,
            instance: allocate(&NEXT_INSTANCE)?,
        };
        let assembly = build_structure(&mut self.simulation.world, &self.definition, qpos)?;
        let cleanup_handles = assembly.body_handles().to_vec();
        let result = (|| -> BoundaryResult<()> {
            for (source_body, handle) in assembly.body_handles().iter().enumerate().skip(1) {
                let handle = handle.ok_or_else(|| invalid("source body was not built"))?;
                let attached = self.simulation.world.bodies[handle]
                    .colliders()
                    .iter()
                    .copied()
                    .collect();
                self.bodies.insert(
                    handle,
                    RegisteredBody {
                        token,
                        source_body,
                        attached,
                    },
                );
            }
            for (source_geom, handle) in assembly.collision_handles() {
                let collider = self
                    .simulation
                    .world
                    .colliders
                    .get_mut(*handle)
                    .ok_or_else(|| invalid("source collider was not built"))?;
                collider.set_active_hooks(SOURCE_HOOKS);
                self.colliders.insert(
                    *handle,
                    RegisteredCollider {
                        token,
                        source_geom: *source_geom,
                        parent: collider
                            .parent()
                            .ok_or_else(|| invalid("source collider has no parent"))?,
                        shape: collider.shared_shape().clone(),
                        local_pose: *collider
                            .position_wrt_parent()
                            .ok_or_else(|| invalid("source collider lacks local pose"))?,
                    },
                );
            }
            self.assembly = Some(assembly);
            self.token = Some(token);
            self.validate_boundary(token)?;
            Ok(())
        })();
        if let Err(error) = result {
            self.token = None;
            self.assembly = None;
            self.bodies.clear();
            self.colliders.clear();
            for body in cleanup_handles.iter().rev().flatten() {
                if let Err(cleanup_error) = self.simulation.remove_body(*body) {
                    return Err(invalid(format!(
                        "{error}; rollback failed: {cleanup_error}"
                    )));
                }
            }
            return Err(error);
        }
        Ok(())
    }

    fn check_token(&self, token: RobotInstanceToken) -> BoundaryResult<()> {
        if token.world_epoch != self.epoch || self.token != Some(token) {
            return Err(invalid("stale, foreign or uncommitted robot token"));
        }
        Ok(())
    }
}

fn allocate(counter: &AtomicU64) -> BoundaryResult<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| invalid("checked collision token allocation exhausted"))
}

fn validate_qpos(definition: &RobotDefinition, qpos: &[f64]) -> BoundaryResult<()> {
    if qpos.iter().any(|value| !(*value as f32).is_finite()) {
        return Err(invalid(
            "source qpos cannot be represented at backend float boundary",
        ));
    }
    source_body_poses(definition, qpos)?;
    Ok(())
}

fn backend_rejects(
    world: &PhysicsWorld,
    first: ColliderHandle,
    second: ColliderHandle,
) -> BoundaryResult<bool> {
    let a = world
        .colliders
        .get(first)
        .ok_or_else(|| invalid("first collider generation is absent"))?;
    let b = world
        .colliders
        .get(second)
        .ok_or_else(|| invalid("second collider generation is absent"))?;
    let parent_a = a
        .parent()
        .ok_or_else(|| invalid("first source parent is absent"))?;
    let parent_b = b
        .parent()
        .ok_or_else(|| invalid("second source parent is absent"))?;
    let body_a = world
        .bodies
        .get(parent_a)
        .ok_or_else(|| invalid("first body generation is absent"))?;
    let body_b = world
        .bodies
        .get(parent_b)
        .ok_or_else(|| invalid("second body generation is absent"))?;
    if parent_a == parent_b
        || !a.is_enabled()
        || !b.is_enabled()
        || (!a
            .active_collision_types()
            .test(body_a.body_type(), body_b.body_type())
            && !b
                .active_collision_types()
                .test(body_a.body_type(), body_b.body_type()))
        || !a.collision_groups().test(b.collision_groups())
        || !a.solver_groups().test(b.solver_groups())
        || world
            .impulse_joints
            .joints_between(parent_a, parent_b)
            .any(|(_, joint)| !joint.data.contacts_enabled())
    {
        return Ok(true);
    }
    if let (Some(a), Some(b)) = (
        world.multibody_joints.rigid_body_link(parent_a),
        world.multibody_joints.rigid_body_link(parent_b),
    ) {
        if a.multibody == b.multibody {
            if world
                .multibody_joints
                .get_multibody(a.multibody)
                .is_none_or(|body| !body.self_contacts_enabled())
            {
                return Ok(true);
            }
            if world
                .multibody_joints
                .joint_between(parent_a, parent_b)
                .is_some_and(|(_, _, link)| !link.joint().data.contacts_enabled())
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

struct SourceHooks<'a> {
    profile: &'a SourceCollisionProfile,
    colliders: &'a HashMap<ColliderHandle, RegisteredCollider>,
    bodies: &'a HashMap<RigidBodyHandle, RegisteredBody>,
    token: RobotInstanceToken,
    failure: &'a AtomicBool,
    #[cfg(test)]
    audit: Option<&'a runtime_batch::Audit>,
}

impl SourceHooks<'_> {
    fn allows(&self, context: &PairFilterContext<'_>) -> bool {
        let lookup = |handle: ColliderHandle, parent: Option<RigidBodyHandle>| -> Option<usize> {
            let collider = context.colliders.get(handle)?;
            let entry = self.colliders.get(&handle)?;
            let owner = self.bodies.get(&entry.parent)?;
            if entry.token != self.token
                || owner.token != self.token
                || parent != Some(entry.parent)
                || collider.parent() != parent
                || context.bodies.get(entry.parent).is_none()
                || !owner.attached.contains(&handle)
                || self.profile.data().geom_bodyid.get(entry.source_geom)
                    != Some(&owner.source_body)
            {
                return None;
            }
            Some(entry.source_geom)
        };
        let result = lookup(context.collider1, context.rigid_body1)
            .zip(lookup(context.collider2, context.rigid_body2))
            .and_then(|(first, second)| {
                self.profile
                    .source_geom_pair_eligibility(first, second)
                    .ok()
            });
        match result {
            Some(result) => result.is_eligible(),
            None => {
                self.failure.store(true, Ordering::Relaxed);
                false
            }
        }
    }
}

impl PhysicsHooks for SourceHooks<'_> {
    fn filter_contact_pair(&self, context: &PairFilterContext<'_>) -> Option<SolverFlags> {
        let allowed = self.allows(context);
        #[cfg(test)]
        if let Some(audit) = self.audit {
            audit.record("contact", context, allowed, self.colliders);
        }
        allowed.then_some(SolverFlags::COMPUTE_IMPULSES)
    }
    fn filter_intersection_pair(&self, context: &PairFilterContext<'_>) -> bool {
        let allowed = self.allows(context);
        #[cfg(test)]
        if let Some(audit) = self.audit {
            audit.record("intersection", context, allowed, self.colliders);
        }
        allowed
    }
}

fn invalid(message: impl Into<String>) -> SourceCollisionError {
    SourceCollisionError::Boundary(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::{fs, io::Write, path::PathBuf};

    fn fixture_root() -> PathBuf {
        PathBuf::from(
            std::env::var("SOURCE_COLLISION_FIXTURES")
                .expect("explicit frozen metadata fixture root is required"),
        )
    }

    fn fixtures() -> Vec<(Arc<RobotDefinition>, Arc<SourceCollisionProfile>)> {
        let manifest: Value = serde_json::from_slice(
            &fs::read(fixture_root().join("metadata_manifest.json")).unwrap(),
        )
        .unwrap();
        [0, 2]
            .into_iter()
            .map(|index| {
                let row = &manifest["profiles"][index];
                let definition = Arc::new(
                    RobotDefinition::load_json(
                        std::path::Path::new(row["definition"]["path"].as_str().unwrap()),
                        row["definition"]["sha256"].as_str().unwrap(),
                    )
                    .unwrap(),
                );
                let profile = Arc::new(
                    SourceCollisionProfile::load_json(
                        std::path::Path::new(row["profile"]["path"].as_str().unwrap()),
                        row["profile"]["sha256"].as_str().unwrap(),
                        &definition,
                    )
                    .unwrap(),
                );
                (definition, profile)
            })
            .collect()
    }

    fn new_world(
        definition: &Arc<RobotDefinition>,
        profile: &Arc<SourceCollisionProfile>,
    ) -> SourceCollisionWorld {
        SourceCollisionWorld::new(
            definition.clone(),
            profile.clone(),
            &definition.model().fields.key_qpos[0],
        )
        .unwrap()
    }

    fn write_evidence(name: &str, value: Value) {
        let directory = PathBuf::from(
            std::env::var("SOURCE_COLLISION_REPORT_DIR")
                .expect("explicit scratch evidence directory is required"),
        );
        fs::create_dir_all(&directory).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(name))
            .unwrap()
            .write_all(&serde_json::to_vec_pretty(&value).unwrap())
            .unwrap();
    }

    fn assert_refused_without_step(
        world: &mut SourceCollisionWorld,
        token: RobotInstanceToken,
    ) -> String {
        let before = serde_json::to_value(world.snapshot()).unwrap();
        let counts = world.counts();
        // Prove the preflight has already rejected before even attempting the
        // negative step path, so an unexpectedly accepted candidate never runs.
        assert!(world.validate_boundary(token).is_err());
        let error = world.step_with_torques(token, &[]).unwrap_err().to_string();
        assert_eq!(serde_json::to_value(world.snapshot()).unwrap(), before);
        assert_eq!(world.counts(), counts);
        assert_eq!(world.snapshot().integration_count, 0);
        assert_eq!(world.snapshot().torque_update_count, 0);
        error
    }

    #[test]
    fn checked_token_allocation_never_wraps() {
        let counter = AtomicU64::new(u64::MAX);
        assert!(allocate(&counter).is_err());
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate(&counter).unwrap(), u64::MAX - 1);
        assert!(allocate(&counter).is_err());
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn zero_step_joint_feedback_uses_live_source_articulations() {
        for (definition, profile) in fixtures() {
            let mut world = new_world(&definition, &profile);
            let token = world.instance_token().unwrap();
            let first = world.joint_feedback_frame(token).unwrap();
            let snapshot = world.snapshot();
            let poses = world
                .assembly
                .as_ref()
                .unwrap()
                .pose_frame(&snapshot)
                .unwrap();
            assert_eq!(first.model_file_sha256, definition.file_sha256());
            assert_eq!(first.collision_profile_file_sha256, profile.file_sha256());
            assert_eq!(
                (first.world_epoch, first.robot_instance),
                (token.world_epoch, token.instance)
            );
            assert_eq!(
                (first.global_step, first.episode_id, first.episode_step),
                (0, 0, 0)
            );
            assert_eq!(first.previous_solve_load, PreviousSolveLoad::Unavailable);
            assert_eq!(
                serde_json::to_value(&first).unwrap()["previous_solve_load"],
                "Unavailable"
            );
            assert_eq!(snapshot.integration_count, 0);
            assert_eq!(snapshot.torque_update_count, 0);
            let model = definition.model();
            for (index, channel) in first.channels.iter().enumerate() {
                let joint = definition.actuator_joint_ids()[index];
                let dof = model.fields.jnt_dofadr[joint];
                let body = model.fields.jnt_bodyid[joint];
                let expected = model.fields.key_qpos[0][model.fields.jnt_qposadr[joint]] as f32;
                assert_eq!((channel.source_joint, channel.source_dof), (joint, dof));
                assert!((channel.position - expected).abs() <= 2e-6);
                assert_eq!(channel.velocity, 0.0);
                let pose = poses
                    .poses
                    .iter()
                    .find(|pose| pose.source_body_id == body)
                    .unwrap();
                assert_eq!(
                    (channel.body_handle.index, channel.body_handle.generation),
                    (pose.backend_handle[0], pose.backend_handle[1])
                );
                let mapping = world
                    .assembly
                    .as_ref()
                    .unwrap()
                    .joint_mapping()
                    .iter()
                    .find(|mapping| mapping.source_joint == joint)
                    .unwrap();
                assert_eq!(
                    (channel.joint_handle.index, channel.joint_handle.generation),
                    mapping.handle.into_raw_parts()
                );
                assert_eq!(channel.backend_dof, mapping.backend_dof);
            }

            let mut shifted = model.fields.key_qpos[0].clone();
            let joint = definition.actuator_joint_ids()[0];
            let qpos_index = model.fields.jnt_qposadr[joint];
            shifted[qpos_index] += 0.01;
            world.remove_robot(token).unwrap();
            assert!(world.joint_feedback_frame(token).is_err());
            let replacement = world.rebuild_robot(&shifted).unwrap();
            assert_ne!(replacement, token);
            assert!(world.joint_feedback_frame(token).is_err());
            let second = world.joint_feedback_frame(replacement).unwrap();
            assert_eq!(
                (second.global_step, second.episode_id, second.episode_step),
                (0, 1, 0)
            );
            assert!((second.channels[0].position - shifted[qpos_index] as f32).abs() <= 2e-6);
            assert_eq!(second.previous_solve_load, PreviousSolveLoad::Unavailable);
            assert_eq!(world.snapshot().integration_count, 0);
            assert_eq!(world.snapshot().torque_update_count, 0);
        }
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn complete_native_pairs_and_twenty_cold_resets_are_zero_step() {
        let mut evidence = Vec::new();
        for (definition, profile) in fixtures() {
            let mut world = new_world(&definition, &profile);
            let first_token = world.instance_token().unwrap();
            let first_raw = world
                .source_body_handle(first_token, 1)
                .unwrap()
                .into_raw_parts();
            let report = world.validate_boundary(first_token).unwrap();
            let oracle = fixture_root()
                .parent()
                .unwrap()
                .join("collision_eligibility_native_v1/results")
                .join(format!("{}_robot_only.json", report.family));
            let native: Value = serde_json::from_slice(&fs::read(oracle).unwrap()).unwrap();
            let pairs = native["all_pairs"].as_array().unwrap();
            let admitted: HashSet<_> = report.eligible_source_geom_pairs.iter().copied().collect();
            let mut native_pairs = HashSet::new();
            let mut native_allowed = 0;
            for pair in pairs {
                let ids = [
                    pair["pair"][0].as_u64().unwrap() as usize,
                    pair["pair"][1].as_u64().unwrap() as usize,
                ];
                assert!(native_pairs.insert(ids));
                let expected = pair["native_contact_generated"].as_bool().unwrap();
                native_allowed += usize::from(expected);
                assert_eq!(admitted.contains(&ids), expected);
            }
            assert_eq!(native_pairs.len(), report.full_source_geom_pairs);
            assert_eq!(
                native_allowed,
                native["actual_ncon"].as_u64().unwrap() as usize
            );
            assert_eq!(native_allowed, report.source_eligible_pairs);
            assert_eq!(report.backend_rejected_source_eligible_pairs, 0);
            let mut resets = Vec::new();
            for number in 1..=20 {
                let old = world.instance_token().unwrap();
                let old_handle = world.source_body_handle(old, 1).unwrap();
                world.remove_robot(old).unwrap();
                assert_eq!(world.counts().bodies, 0);
                assert_eq!(world.counts().colliders, 0);
                assert_eq!(world.counts().multibody_joint_handles, 0);
                let empty_rejection = assert_refused_without_step(&mut world, old);
                let fresh = world
                    .rebuild_robot(&definition.model().fields.key_qpos[0])
                    .unwrap();
                assert_ne!(old, fresh);
                let fresh_handle = world.source_body_handle(fresh, 1).unwrap();
                assert_ne!(old_handle, fresh_handle);
                assert_eq!(
                    old_handle.into_raw_parts().0,
                    fresh_handle.into_raw_parts().0
                );
                assert!(world.simulation.world.bodies.get(old_handle).is_none());
                let stale_rejection = assert_refused_without_step(&mut world, old);
                let current = world.validate_boundary(fresh).unwrap();
                assert_eq!(current.counts, report.counts);
                let before_torque_refusal = serde_json::to_value(world.snapshot()).unwrap();
                let old_body_torque_rejection = world
                    .step_with_torques(
                        fresh,
                        &[BodyTorque {
                            body: old_handle,
                            world_torque: [1.0, 2.0, 3.0],
                        }],
                    )
                    .unwrap_err()
                    .to_string();
                assert!(old_body_torque_rejection.contains("unknown rigid-body handle"));
                assert_eq!(
                    serde_json::to_value(world.snapshot()).unwrap(),
                    before_torque_refusal
                );
                assert_eq!(world.snapshot().integration_count, 0);
                assert_eq!(world.snapshot().torque_update_count, 0);
                assert_eq!(
                    current.eligible_source_geom_pairs,
                    report.eligible_source_geom_pairs
                );
                assert_eq!(world.snapshot().episode_id, number);
                assert_eq!(world.snapshot().episode_step, 0);
                resets.push(json!({"reset":number,"old_token":old,"new_token":fresh,
                    "old_body":old_handle.into_raw_parts(),"new_body":fresh_handle.into_raw_parts(),
                    "empty_rejection":empty_rejection,"stale_rejection":stale_rejection,
                    "old_body_torque_rejection":old_body_torque_rejection,
                    "counts":current.counts,"physics_integrations":0}));
            }
            drop(world);
            let mut replacement = new_world(&definition, &profile);
            let fresh = replacement.instance_token().unwrap();
            assert_eq!(
                replacement
                    .source_body_handle(fresh, 1)
                    .unwrap()
                    .into_raw_parts(),
                first_raw
            );
            assert_ne!(fresh.world_epoch, first_token.world_epoch);
            let replacement_rejection = assert_refused_without_step(&mut replacement, first_token);
            evidence.push(
                json!({"initial_registration":report,"native_pair_comparisons":native_pairs.len(),
                "native_allowed_pairs":native_allowed,"resets":resets,
                "replacement_same_raw_handle":first_raw,"replacement_token":fresh,
                "replacement_old_token_rejection":replacement_rejection}),
            );
        }
        write_evidence(
            "registration_and_resets.json",
            json!({"passed":true,"cases":evidence,
            "physics_integrations":0,"policy_inferences":0,"contact_or_ccd_qualified":false,
            "force_or_bam_qualified":false,"target_plant_accepted":false}),
        );
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn all_inventory_faults_refuse_before_mutating_torque_or_counters() {
        let names = [
            "missing_body_registry",
            "missing_collider_registry",
            "stale_body_generation",
            "stale_collider_generation",
            "wrong_owner",
            "wrong_geom",
            "wrong_instance",
            "wrong_world_epoch",
            "extra_body",
            "extra_collider",
            "native_collider_removed",
            "native_body_removed",
            "reparent",
            "disabled_collider",
            "disabled_body",
            "sleeping_body",
            "far_unregistered",
            "sleeping_unregistered",
            "disabled_unregistered",
            "missing_contact_hook",
            "missing_intersection_hook",
            "custom_modifier",
            "sensor",
            "collision_groups",
            "solver_groups",
            "active_collision_types",
            "body_type",
            "self_contacts",
            "joint_contacts",
            "extra_impulse_joint",
            "shape_replaced",
            "local_pose",
            "collider_userdata",
            "body_userdata",
            "dt",
            "extra_solver_iteration",
            "solver_iterations",
            "ccd_steps",
        ];
        let mut evidence = Vec::new();
        for (definition, profile) in fixtures() {
            for name in names {
                let mut world = new_world(&definition, &profile);
                let token = world.instance_token().unwrap();
                let body = world.source_body_handle(token, 1).unwrap();
                let other = world.source_body_handle(token, 2).unwrap();
                let collider = world.assembly.as_ref().unwrap().collision_handles()[0].1;
                let joint = world.assembly.as_ref().unwrap().joint_mapping()[0].handle;
                world.simulation.world.bodies[body].add_torque(Vector::new(1.0, 2.0, 3.0), false);
                match name {
                    "missing_body_registry" => {
                        world.bodies.remove(&body);
                    }
                    "missing_collider_registry" => {
                        world.colliders.remove(&collider);
                    }
                    "stale_body_generation" => {
                        let entry = world.bodies.remove(&body).unwrap();
                        let (index, generation) = body.into_raw_parts();
                        world.bodies.insert(
                            RigidBodyHandle::from_raw_parts(index, generation + 1),
                            entry,
                        );
                    }
                    "stale_collider_generation" => {
                        let entry = world.colliders.remove(&collider).unwrap();
                        let (index, generation) = collider.into_raw_parts();
                        world
                            .colliders
                            .insert(ColliderHandle::from_raw_parts(index, generation + 1), entry);
                    }
                    "wrong_owner" => world.colliders.get_mut(&collider).unwrap().parent = other,
                    "wrong_geom" => {
                        world.colliders.get_mut(&collider).unwrap().source_geom = usize::MAX
                    }
                    "wrong_instance" => {
                        world.colliders.get_mut(&collider).unwrap().token.instance += 1
                    }
                    "wrong_world_epoch" => {
                        world.bodies.get_mut(&body).unwrap().token.world_epoch += 1
                    }
                    "extra_body" => {
                        world
                            .simulation
                            .world
                            .bodies
                            .insert(RigidBodyBuilder::dynamic());
                    }
                    "extra_collider" => {
                        world.simulation.world.colliders.insert_with_parent(
                            ColliderBuilder::ball(0.1),
                            body,
                            &mut world.simulation.world.bodies,
                        );
                    }
                    "native_collider_removed" => {
                        world.simulation.world.remove_collider(collider);
                    }
                    "native_body_removed" => {
                        world.simulation.world.remove_body(other);
                    }
                    "reparent" => world.simulation.world.colliders.set_parent(
                        collider,
                        Some(other),
                        &mut world.simulation.world.bodies,
                    ),
                    "disabled_collider" => {
                        world.simulation.world.colliders[collider].set_enabled(false)
                    }
                    "disabled_body" => world.simulation.world.bodies[body].set_enabled(false),
                    "sleeping_body" => world.simulation.world.bodies[body].sleep(),
                    "far_unregistered" | "sleeping_unregistered" | "disabled_unregistered" => {
                        let extra = world.simulation.world.bodies.insert(
                            RigidBodyBuilder::dynamic()
                                .translation(Vector::splat(1.0e8))
                                .sleeping(name == "sleeping_unregistered"),
                        );
                        let extra_collider = world.simulation.world.colliders.insert_with_parent(
                            ColliderBuilder::ball(0.1),
                            extra,
                            &mut world.simulation.world.bodies,
                        );
                        if name == "disabled_unregistered" {
                            world.simulation.world.colliders[extra_collider].set_enabled(false);
                        }
                    }
                    "missing_contact_hook" => world.simulation.world.colliders[collider]
                        .set_active_hooks(ActiveHooks::FILTER_INTERSECTION_PAIR),
                    "missing_intersection_hook" => world.simulation.world.colliders[collider]
                        .set_active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS),
                    "custom_modifier" => world.simulation.world.colliders[collider]
                        .set_active_hooks(SOURCE_HOOKS | ActiveHooks::MODIFY_SOLVER_CONTACTS),
                    "sensor" => world.simulation.world.colliders[collider].set_sensor(true),
                    "collision_groups" => world.simulation.world.colliders[collider]
                        .set_collision_groups(InteractionGroups::none()),
                    "solver_groups" => world.simulation.world.colliders[collider]
                        .set_solver_groups(InteractionGroups::none()),
                    "active_collision_types" => world.simulation.world.colliders[collider]
                        .set_active_collision_types(ActiveCollisionTypes::empty()),
                    "body_type" => world.simulation.world.bodies[body]
                        .set_body_type(RigidBodyType::Fixed, false),
                    "self_contacts" | "joint_contacts" => {
                        let (multibody, link) = world
                            .simulation
                            .world
                            .multibody_joints
                            .get_mut(joint)
                            .unwrap();
                        if name == "self_contacts" {
                            multibody.set_self_contacts_enabled(false);
                        } else {
                            multibody
                                .link_mut(link)
                                .unwrap()
                                .joint
                                .data
                                .set_contacts_enabled(true);
                        }
                    }
                    "extra_impulse_joint" => {
                        world.simulation.world.impulse_joints.insert(
                            body,
                            other,
                            FixedJointBuilder::new(),
                            false,
                        );
                    }
                    "shape_replaced" => {
                        world.simulation.world.colliders[collider].set_shape(SharedShape::ball(0.1))
                    }
                    "local_pose" => world.simulation.world.colliders[collider]
                        .set_position_wrt_parent(Pose::from_translation(Vector::X)),
                    "collider_userdata" => {
                        world.simulation.world.colliders[collider].user_data = u128::MAX
                    }
                    "body_userdata" => world.simulation.world.bodies[body].user_data = u128::MAX,
                    "dt" => world.simulation.world.integration_parameters.dt = 0.01,
                    "extra_solver_iteration" => {
                        world.simulation.world.bodies[body].set_additional_solver_iterations(1)
                    }
                    "solver_iterations" => {
                        world
                            .simulation
                            .world
                            .integration_parameters
                            .num_solver_iterations = 2
                    }
                    "ccd_steps" => {
                        world
                            .simulation
                            .world
                            .integration_parameters
                            .max_ccd_substeps = 2
                    }
                    _ => panic!("unknown fault"),
                }
                let rejection = assert_refused_without_step(&mut world, token);
                evidence.push(
                    json!({"family":definition.model().family,"fault":name,"rejection":rejection,
                    "snapshot_and_torque_queues_unchanged":true,"physics_integrations":0}),
                );
            }
        }
        write_evidence(
            "inventory_refusals.json",
            json!({"passed":true,"actual_refusals":evidence.len(),
            "cases":evidence,"physics_integrations":0,"policy_inferences":0}),
        );
    }

    #[test]
    #[ignore = "requires explicit hash-checked frozen native source fixtures"]
    fn invalid_build_rebuild_and_foreign_tokens_preserve_boundary() {
        let mut evidence = Vec::new();
        for (definition, profile) in fixtures() {
            let mut world = new_world(&definition, &profile);
            let token = world.instance_token().unwrap();
            let before = serde_json::to_value(world.snapshot()).unwrap();
            assert!(
                world
                    .rebuild_robot(&definition.model().fields.key_qpos[0])
                    .is_err()
            );
            assert_eq!(before, serde_json::to_value(world.snapshot()).unwrap());
            let mut invalid_qpos = definition.model().fields.key_qpos[0].clone();
            invalid_qpos[7] = f64::INFINITY;
            assert!(
                SourceCollisionWorld::new(definition.clone(), profile.clone(), &invalid_qpos)
                    .is_err()
            );
            let foreign = RobotInstanceToken {
                world_epoch: token.world_epoch + 1,
                instance: token.instance,
            };
            let wrong_epoch = assert_refused_without_step(&mut world, foreign);
            let foreign = RobotInstanceToken {
                world_epoch: token.world_epoch,
                instance: token.instance + 1,
            };
            let wrong_instance = assert_refused_without_step(&mut world, foreign);
            world.remove_robot(token).unwrap();
            let empty = serde_json::to_value(world.snapshot()).unwrap();
            assert!(world.rebuild_robot(&invalid_qpos).is_err());
            assert_eq!(empty, serde_json::to_value(world.snapshot()).unwrap());
            let fresh = world
                .rebuild_robot(&definition.model().fields.key_qpos[0])
                .unwrap();
            assert!(world.validate_boundary(fresh).is_ok());
            evidence.push(json!({"family":definition.model().family,"second_instance_rejected":true,
                "invalid_new_qpos_rejected":true,"invalid_rebuild_preserved_empty_state":true,
                "wrong_epoch":wrong_epoch,"wrong_instance":wrong_instance,"physics_integrations":0}));
        }
        write_evidence(
            "lifecycle_refusals.json",
            json!({"passed":true,"cases":evidence,
            "physics_integrations":0,"policy_inferences":0}),
        );
    }
}

/// Explicitly invoked bounded CPU fixtures; none run in the default test suite.
#[cfg(test)]
mod runtime_batch {
    use super::*;
    use robot_minigame::collision_profile::{GeomMasks, masks_allow};
    use serde_json::{Value, json};
    use std::{fs, io::Write, path::PathBuf, sync::Mutex};

    #[derive(Default)]
    pub(super) struct Audit {
        calls: Mutex<Vec<Value>>,
    }
    impl Audit {
        pub(super) fn record(
            &self,
            kind: &str,
            context: &PairFilterContext<'_>,
            allowed: bool,
            entries: &HashMap<ColliderHandle, RegisteredCollider>,
        ) {
            self.calls.lock().unwrap().push(json!({
                "kind":kind,"allowed":allowed,
                "caller_backtrace":format!("{:?}",std::backtrace::Backtrace::force_capture()),
                "colliders":[raw(context.collider1),raw(context.collider2)],
                "bodies":[context.rigid_body1.map(raw),context.rigid_body2.map(raw)],
                "source_geoms":[entries.get(&context.collider1).map(|e|e.source_geom),
                    entries.get(&context.collider2).map(|e|e.source_geom)],
                "collider_positions_bits":[context.colliders.get(context.collider1)
                    .map(|c|bits(c.translation().to_array())),context.colliders.get(context.collider2)
                    .map(|c|bits(c.translation().to_array()))]
            }));
        }
        fn take(&self) -> Vec<Value> {
            std::mem::take(&mut *self.calls.lock().unwrap())
        }
    }
    fn raw<T>(handle: T) -> [u32; 2]
    where
        T: IntoRaw,
    {
        handle.parts()
    }
    trait IntoRaw {
        fn parts(self) -> [u32; 2];
    }
    impl IntoRaw for ColliderHandle {
        fn parts(self) -> [u32; 2] {
            let (a, b) = self.into_raw_parts();
            [a, b]
        }
    }
    impl IntoRaw for RigidBodyHandle {
        fn parts(self) -> [u32; 2] {
            let (a, b) = self.into_raw_parts();
            [a, b]
        }
    }
    fn bits<const N: usize>(value: [f32; N]) -> Vec<u32> {
        value.into_iter().map(f32::to_bits).collect()
    }
    fn root() -> PathBuf {
        PathBuf::from(std::env::var("COLLISION_RUNTIME_OUTPUT").unwrap())
    }
    fn save(name: &str, value: &Value) {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root().join(name))
            .unwrap();
        file.write_all(serde_json::to_string_pretty(value).unwrap().as_bytes())
            .unwrap();
        file.sync_all().unwrap();
    }
    fn read(path: impl AsRef<std::path::Path>) -> Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }
    fn fixtures() -> Vec<(Arc<RobotDefinition>, Arc<SourceCollisionProfile>)> {
        let manifest = read(
            ".scratch/foundation_engineering/collision_profile_typed_v1/metadata_manifest.json",
        );
        [0, 2]
            .into_iter()
            .map(|i| {
                let row = &manifest["profiles"][i];
                let def = Arc::new(
                    RobotDefinition::load_json(
                        std::path::Path::new(row["definition"]["path"].as_str().unwrap()),
                        row["definition"]["sha256"].as_str().unwrap(),
                    )
                    .unwrap(),
                );
                let profile = Arc::new(
                    SourceCollisionProfile::load_json(
                        std::path::Path::new(row["profile"]["path"].as_str().unwrap()),
                        row["profile"]["sha256"].as_str().unwrap(),
                        &def,
                    )
                    .unwrap(),
                );
                (def, profile)
            })
            .collect()
    }
    fn limits(def: &RobotDefinition, q: &[f64]) -> bool {
        let f = &def.model().fields;
        q.len() == def.model().counts.nq
            && q.iter().all(|x| x.is_finite())
            && (1..def.model().counts.njnt).all(|j| {
                !f.jnt_limited[j] || {
                    let v = q[f.jnt_qposadr[j]];
                    v >= f.jnt_range[j][0] && v <= f.jnt_range[j][1]
                }
            })
    }
    fn overlap(world: &SourceCollisionWorld) -> Vec<Value> {
        let w = &world.simulation.world;
        let mut geoms: Vec<_> = world.colliders.iter().collect();
        geoms.sort_by_key(|(_, r)| r.source_geom);
        let mut result = Vec::new();
        for a in 0..geoms.len() {
            for b in a + 1..geoms.len() {
                let (ha, ra) = geoms[a];
                let (hb, rb) = geoms[b];
                if !world
                    .profile
                    .source_geom_pair_eligibility(ra.source_geom, rb.source_geom)
                    .unwrap()
                    .is_eligible()
                {
                    continue;
                }
                let ca = &w.colliders[*ha];
                let cb = &w.colliders[*hb];
                match rapier3d::parry::query::contact(
                    ca.position(),
                    ca.shape(),
                    cb.position(),
                    cb.shape(),
                    0.0,
                ) {
                    Ok(Some(c)) if c.dist >= -0.003 && c.dist <= -0.00005 => result
                        .push(json!({"geoms":[ra.source_geom,rb.source_geom],"distance":c.dist})),
                    _ => {}
                }
            }
        }
        result
    }
    #[test]
    #[ignore = "explicit bounded plan preparation, zero integrations"]
    fn prepare() {
        let mut families = Vec::new();
        for (def, profile) in fixtures() {
            let mut candidates = Vec::new();
            let mut selected = None;
            let home = def.model().fields.key_qpos[0].clone();
            for i in 0..64 {
                let mut q = home.clone();
                if i != 0 {
                    for j in 1..def.model().counts.njnt {
                        let f = &def.model().fields;
                        if f.jnt_limited[j] {
                            let t = ((i * (j * 2 + 1) + j * 13) % 17) as f64 / 16.0;
                            q[f.jnt_qposadr[j]] = (f.jnt_range[j][0]
                                + t * (f.jnt_range[j][1] - f.jnt_range[j][0]))
                                .clamp(f.jnt_range[j][0], f.jnt_range[j][1]);
                        }
                    }
                }
                assert!(limits(&def, &q));
                let world = SourceCollisionWorld::new(def.clone(), profile.clone(), &q).unwrap();
                let found = overlap(&world);
                if selected.is_none() && !found.is_empty() {
                    selected = Some(i);
                }
                candidates.push(json!({"id":i,"qpos":q,"joint_limits_passed":true,"shallow_eligible_overlaps":found}));
            }
            let mut smoke = Vec::new();
            for i in 0..4 {
                let mut q = home.clone();
                if i == 1 {
                    q[0] = 1.25;
                    q[1] = 0.34;
                    q[2] = -0.4;
                }
                if i == 2 {
                    let angle = 0.15f64;
                    q[3] = angle.cos();
                    q[4] = angle.sin();
                }
                if i == 3 {
                    for j in 1..def.model().counts.njnt {
                        let f = &def.model().fields;
                        if f.jnt_limited[j] {
                            q[f.jnt_qposadr[j]] = (q[f.jnt_qposadr[j]] + 0.01)
                                .clamp(f.jnt_range[j][0], f.jnt_range[j][1]);
                        }
                    }
                }
                assert!(limits(&def, &q));
                smoke.push(json!({"id":i,"qpos":q,"joint_limits_passed":true}));
            }
            families.push(json!({"family":def.model().family,"definition_sha":def.file_sha256(),"profile_sha":profile.file_sha256(),
                "joint_ranges":def.model().fields.jnt_range,"joint_limited":def.model().fields.jnt_limited,
                "candidates":candidates,"selection_rule":"first by id with eligible original-shape contact distance in [-0.003,-0.00005]",
                "selected":selected,"smoke":smoke}));
        }
        save(
            "prepared.json",
            &json!({"scope":"preintegrations_static_original_geometry_selection","physics_integrations":0,
            "planned_entries":118,"supplementary_limit":10,"hard_pipeline_entry_limit":128,"policy_inferences":0,
            "families":families,"dt_bits":(1.0f32/60.0).to_bits(),"solver_iterations":1,"additional_iterations":0,"max_ccd_substeps":1,
            "source_pairs":[[8,18],[6,18],[25,28]],"mask_cases":[[1,0,0,1],[0,1,1,0],[1,2,2,0],[1,2,0,1],[1,1,1,1],[1,1,2,2],[0,0,3,3],[3,3,1,2]],
            "contact_spheres":{"radius":0.5,"centers":[[-0.45,0.0,0.0],[0.45,0.0,0.0]],"gravity":[0,0,0]},
            "ccd_spheres":{"radius":0.5,"moving_x":-3,"target_x":0,"velocity_x":360},
            "warmstart_spheres":{"radius":0.5,"moving_y":0.99,"fixed_y":0,"gravity":[0,-9.81,0]},
            "production_environment":false,"production_sensors":false}),
        );
    }

    struct Ledger {
        file: fs::File,
        used: usize,
        offset: usize,
    }
    impl Ledger {
        fn new() -> Self {
            assert!(
                root().join("frozen_before_entries.json").exists(),
                "all recipes and sources must be frozen first"
            );
            let path = root().join("pipeline_ledger.jsonl");
            let used = if path.exists() {
                fs::read_to_string(&path)
                    .unwrap()
                    .lines()
                    .filter(|l| read_line(l)["phase"] == "before")
                    .count()
            } else {
                0
            };
            let offset = std::env::var("COLLISION_RUNTIME_ENTRY_OFFSET")
                .ok()
                .map(|text| text.parse::<usize>().unwrap())
                .unwrap_or(0);
            assert!(offset <= 128 && offset + used <= 128);
            Self {
                file: fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap(),
                used,
                offset,
            }
        }
        fn append(&mut self, row: Value) {
            writeln!(self.file, "{}", serde_json::to_string(&row).unwrap()).unwrap();
            self.file.flush().unwrap();
            self.file.sync_data().unwrap();
        }
        fn before(&mut self, name: &str, scope: &str, input: Value) -> usize {
            assert!(
                self.offset + self.used < 128,
                "persistent pipeline-entry budget exhausted"
            );
            self.used += 1;
            let global_entry = self.offset + self.used;
            self.append(json!({"phase":"before","entry":global_entry,"local_entry":self.used,"case":name,"scope":scope,"inputs":input,
                "remaining":128-global_entry,"dt_bits":(1.0f32/60.0).to_bits(),"solver":1,"additional":0,"max_ccd":1}));
            global_entry
        }
        fn after(&mut self, id: usize, row: Value) {
            self.append(json!({"phase":"after","entry":id,"observed":row}));
        }
    }
    fn read_line(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }
    fn contacts(world: &PhysicsWorld) -> Value {
        let mut rows = Vec::new();
        for pair in world.contact_pairs() {
            let a = world
                .colliders
                .get(pair.collider1)
                .map(|c| c.user_data as usize);
            let b = world
                .colliders
                .get(pair.collider2)
                .map(|c| c.user_data as usize);
            for (kind, manifolds) in [
                ("manifold", &pair.manifolds),
                ("solver_cluster", &pair.solver_clusters),
            ] {
                for m in manifolds {
                    for p in &m.points {
                        rows.push(json!({"geoms":[a,b],"handles":[raw(pair.collider1),raw(pair.collider2)],"kind":kind,
                        "normal_bits":bits(m.data.normal.to_array()),"local_p1_bits":bits(p.local_p1.to_array()),
                        "local_p2_bits":bits(p.local_p2.to_array()),"dist_bits":p.dist.to_bits(),"impulse_bits":p.data.impulse.to_bits(),
                        "warmstart_bits":p.data.warmstart_impulse.to_bits(),"tangent_bits":bits([p.data.tangent_impulse[0],p.data.tangent_impulse[1]]),
                        "warmstart_tangent_world_bits":bits(p.data.warmstart_tangent_world.to_array()),
                        "warmstart_positive":p.data.warmstart_impulse.is_finite() && p.data.warmstart_impulse>0.0}));
                    }
                }
            }
        }
        rows.sort_by_key(|r| r.to_string());
        json!(rows)
    }
    fn body_bits(world: &PhysicsWorld) -> Value {
        let mut rows:Vec<_>=world.bodies.iter().map(|(_,b)|json!({"source_body":b.user_data,
            "p":bits(b.translation().to_array()),"r":bits(b.rotation().to_array()),"v":bits(b.linvel().to_array()),"w":bits(b.angvel().to_array())})).collect();
        rows.sort_by_key(|r| r["source_body"].as_u64());
        json!(rows)
    }
    fn canonical_contacts(value: &Value) -> Value {
        let mut rows = value.as_array().unwrap().clone();
        for r in &mut rows {
            r.as_object_mut().unwrap().remove("handles");
        }
        rows.sort_by_key(|r| r.to_string());
        json!(rows)
    }
    fn warm(v: &Value) -> bool {
        v.as_array()
            .unwrap()
            .iter()
            .any(|r| r["warmstart_positive"] == true)
    }
    fn source_step(
        ledger: &mut Ledger,
        name: &str,
        world: &mut SourceCollisionWorld,
        torques: &[BodyTorque],
    ) -> Value {
        let token = world.instance_token().unwrap();
        let admission = world.validate_boundary(token).unwrap();
        let audit = Arc::new(Audit::default());
        world.runtime_audit = Some(audit.clone());
        let before = world.snapshot();
        let id=ledger.before(name,"original_full_source_wrapper",json!({"token":token,"snapshot":before,"admission":admission,
            "torques":torques.iter().map(|t|json!({"body":raw(t.body),"bits":bits(t.world_torque)})).collect::<Vec<_>>()}));
        let result = world.step_with_torques(token, torques);
        let after = world.snapshot();
        let calls = audit.take();
        let row = json!({"returned_ok":result.is_ok(),"error":result.err().map(|e|e.to_string()),"snapshot":after,
            "integration_delta":after.integration_count-before.integration_count,
            "completed_tick_delta":after.global_step-before.global_step,"torque_delta":after.torque_update_count-before.torque_update_count,
            "token_valid":world.instance_token().is_ok(),"calls":calls,"body_bits":body_bits(&world.simulation.world),
            "contact_bits":contacts(&world.simulation.world),"joint_state_bits":world.assembly.as_ref().map(|a|a.joint_state(&world.simulation.world).ok().map(|rows|rows.into_iter().map(|(j,q,qd)|json!([j,q.to_bits(),qd.to_bits()])).collect::<Vec<_>>())),
            "quarantine":world.simulation.world.quarantine().is_empty()});
        ledger.after(id, row.clone());
        row
    }
    #[derive(Default)]
    struct Events(Mutex<Vec<Value>>);
    impl EventHandler for Events {
        fn handle_collision_event(
            &self,
            _: &RigidBodySet,
            colliders: &ColliderSet,
            event: CollisionEvent,
            _: Option<&ContactPair>,
        ) {
            let (kind, a, b, flags) = match event {
                CollisionEvent::Started(a, b, f) => ("Started", a, b, f),
                CollisionEvent::Stopped(a, b, f) => ("Stopped", a, b, f),
            };
            self.0.lock().unwrap().push(
                json!({"kind":kind,"colliders":[raw(a),raw(b)],"flags":flags.bits(),
                "currently_live":[colliders.get(a).is_some(),colliders.get(b).is_some()]}),
            );
        }
        fn handle_contact_force_event(
            &self,
            _: Real,
            _: &RigidBodySet,
            _: &ColliderSet,
            _: &ContactPair,
            _: Real,
        ) {
        }
    }
    struct Fixture {
        sim: SimulationWorld,
        profile: Arc<SourceCollisionProfile>,
        registry: HashMap<ColliderHandle, RegisteredCollider>,
        owners: HashMap<RigidBodyHandle, RegisteredBody>,
        token: RobotInstanceToken,
        bodies: [RigidBodyHandle; 2],
        geoms: [ColliderHandle; 2],
        audit: Audit,
        events: Events,
        entries: usize,
    }
    impl Fixture {
        fn new(
            profile: Arc<SourceCollisionProfile>,
            pair: [usize; 2],
            reverse: bool,
            sensor: bool,
            ccd: bool,
            warmstart: bool,
        ) -> Self {
            let mut sim = SimulationWorld::new();
            sim.world.gravity = if warmstart {
                Vector::new(0., -9.81, 0.)
            } else {
                Vector::ZERO
            };
            let token = RobotInstanceToken {
                world_epoch: allocate(&NEXT_WORLD_EPOCH).unwrap(),
                instance: allocate(&NEXT_INSTANCE).unwrap(),
            };
            let mut handles = [None, None];
            let mut geoms = [None, None];
            let mut owners = HashMap::new();
            let mut registry = HashMap::new();
            for side in if reverse { [1, 0] } else { [0, 1] } {
                let builder = if (ccd || warmstart) && side == 1 {
                    RigidBodyBuilder::fixed()
                } else {
                    RigidBodyBuilder::dynamic().can_sleep(false)
                };
                let p = if ccd {
                    Vector::new(if side == 0 { -3. } else { 0. }, 0., 0.)
                } else if warmstart {
                    Vector::new(0., if side == 0 { 0.99 } else { 0. }, 0.)
                } else {
                    Vector::new(if side == 0 { -0.45 } else { 0.45 }, 0., 0.)
                };
                let builder = builder
                    .translation(p)
                    .user_data(profile.data().geom_bodyid[pair[side]] as u128);
                let builder = if ccd && side == 0 {
                    builder.linvel(Vector::new(360., 0., 0.)).ccd_enabled(true)
                } else {
                    builder
                };
                let body = sim.world.bodies.insert(builder);
                let collider = sim.world.colliders.insert_with_parent(
                    ColliderBuilder::ball(0.5)
                        .density(1.0)
                        .sensor(sensor && side == 1)
                        .active_hooks(SOURCE_HOOKS)
                        .active_events(ActiveEvents::COLLISION_EVENTS)
                        .user_data(pair[side] as u128)
                        .build(),
                    body,
                    &mut sim.world.bodies,
                );
                let c = &sim.world.colliders[collider];
                registry.insert(
                    collider,
                    RegisteredCollider {
                        token,
                        source_geom: pair[side],
                        parent: body,
                        shape: c.shared_shape().clone(),
                        local_pose: *c.position_wrt_parent().unwrap(),
                    },
                );
                owners.insert(
                    body,
                    RegisteredBody {
                        token,
                        source_body: profile.data().geom_bodyid[pair[side]],
                        attached: HashSet::from([collider]),
                    },
                );
                handles[side] = Some(body);
                geoms[side] = Some(collider);
            }
            Self {
                sim,
                profile,
                registry,
                owners,
                token,
                bodies: handles.map(Option::unwrap),
                geoms: geoms.map(Option::unwrap),
                audit: Audit::default(),
                events: Events::default(),
                entries: 0,
            }
        }
        fn step(
            &mut self,
            ledger: &mut Ledger,
            name: &str,
            mask: Option<(GeomMasks, GeomMasks)>,
        ) -> Value {
            let id=ledger.before(name,if mask.is_some(){"asymmetric_mask_sphere_hook"}else{"source_hooks_sphere_surrogate"},json!({
                "token":self.token,"body_bits":body_bits(&self.sim.world),"source_geoms":self.geoms.map(|g|self.sim.world.colliders[g].user_data),
                "colliders":self.geoms.map(raw),"mask_override":mask,"gravity_bits":bits(self.sim.world.gravity.to_array()),
                "sensor":self.geoms.map(|g|self.sim.world.colliders[g].is_sensor()),
                "active_hooks":self.geoms.map(|g|self.sim.world.colliders[g].active_hooks().bits()),
                "ccd_enabled":self.bodies.map(|b|self.sim.world.bodies[b].is_ccd_enabled())}));
            let failure = AtomicBool::new(false);
            let hooks = SourceHooks {
                profile: &self.profile,
                colliders: &self.registry,
                bodies: &self.owners,
                token: self.token,
                failure: &failure,
                audit: Some(&self.audit),
            };
            if let Some((a, b)) = mask {
                struct Masks<'a> {
                    a: GeomMasks,
                    b: GeomMasks,
                    audit: &'a Audit,
                    registry: &'a HashMap<ColliderHandle, RegisteredCollider>,
                }
                impl PhysicsHooks for Masks<'_> {
                    fn filter_contact_pair(
                        &self,
                        c: &PairFilterContext<'_>,
                    ) -> Option<SolverFlags> {
                        let allowed = masks_allow(self.a, self.b);
                        self.audit.record("mask_contact", c, allowed, self.registry);
                        allowed.then_some(SolverFlags::COMPUTE_IMPULSES)
                    }
                }
                self.sim.world.step_with_events(
                    &Masks {
                        a,
                        b,
                        audit: &self.audit,
                        registry: &self.registry,
                    },
                    &self.events,
                );
            } else {
                self.sim.world.step_with_events(&hooks, &self.events);
            }
            self.entries += 1;
            let calls = self.audit.take();
            let events = std::mem::take(&mut *self.events.0.lock().unwrap());
            let intersection = self
                .sim
                .world
                .narrow_phase
                .intersection_pair(self.geoms[0], self.geoms[1]);
            let row = json!({"returned_ok":true,"pipeline_entries_in_world":self.entries,"body_bits":body_bits(&self.sim.world),
                "positions":self.bodies.map(|b|self.sim.world.bodies[b].translation().to_array()),"calls":calls,"events":events,
                "contact_bits":contacts(&self.sim.world),"active_contacts":self.sim.world.contact_pairs().filter(|p|p.has_any_active_contact()).count(),
                "intersection":intersection,"lookup_failure":failure.load(Ordering::Relaxed),"quarantine":self.sim.world.quarantine().is_empty(),
                "ccd_active":self.bodies.map(|b|self.sim.world.bodies[b].is_ccd_active())});
            ledger.after(id, row.clone());
            row
        }
        fn rebuild_dynamic(&mut self, sensor: bool, warmstart: bool) -> Value {
            let old_body = self.bodies[0];
            let old_geom = self.geoms[0];
            self.sim.remove_body(old_body).unwrap();
            self.registry.remove(&old_geom);
            self.owners.remove(&old_body);
            let body = self.sim.world.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .can_sleep(false)
                    .translation(if warmstart {
                        Vector::new(0., 0.99, 0.)
                    } else {
                        Vector::new(-0.45, 0., 0.)
                    })
                    .user_data(self.profile.data().geom_bodyid[8] as u128),
            );
            let g = self.sim.world.colliders.insert_with_parent(
                ColliderBuilder::ball(0.5)
                    .density(1.0)
                    .sensor(sensor)
                    .active_hooks(SOURCE_HOOKS)
                    .active_events(ActiveEvents::COLLISION_EVENTS)
                    .user_data(8)
                    .build(),
                body,
                &mut self.sim.world.bodies,
            );
            let c = &self.sim.world.colliders[g];
            self.registry.insert(
                g,
                RegisteredCollider {
                    token: self.token,
                    source_geom: 8,
                    parent: body,
                    shape: c.shared_shape().clone(),
                    local_pose: *c.position_wrt_parent().unwrap(),
                },
            );
            self.owners.insert(
                body,
                RegisteredBody {
                    token: self.token,
                    source_body: self.profile.data().geom_bodyid[8],
                    attached: HashSet::from([g]),
                },
            );
            self.bodies[0] = body;
            self.geoms[0] = g;
            json!({"old_body":raw(old_body),"new_body":raw(body),"old_geom":raw(old_geom),"new_geom":raw(g),
                "retired_absent":self.sim.world.bodies.get(old_body).is_none() && self.sim.world.colliders.get(old_geom).is_none(),"snapshot":self.sim.snapshot()})
        }
    }
    fn list(v: &Value, key: &str) -> Vec<Value> {
        v[key].as_array().unwrap().clone()
    }
    fn calls_allowed(v: &Value, kind: &str, allowed: bool) -> bool {
        v["calls"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["kind"] == kind && r["allowed"] == allowed)
    }
    fn gate(name: &str, scope: &str, passed: bool, data: Value) -> Value {
        json!({"case":name,"scope":scope,"passed":passed,"observed":data})
    }

    #[test]
    #[ignore = "explicit single bounded runtime batch; persistent 128-entry ledger"]
    fn execute() {
        let prepared = read(root().join("prepared.json"));
        let mut ledger = Ledger::new();
        let start_entries = ledger.used;
        let mut results = Vec::new();
        let mut original = Vec::new();
        let mut uncovered = Vec::new();
        let all = fixtures();
        for (index, (def, profile)) in all.iter().enumerate() {
            let recipe = &prepared["families"][index];
            let family = &def.model().family;
            for smoke in recipe["smoke"].as_array().unwrap() {
                let q: Vec<f64> = serde_json::from_value(smoke["qpos"].clone()).unwrap();
                let mut world =
                    SourceCollisionWorld::new(def.clone(), profile.clone(), &q).unwrap();
                let name = format!("A_{family}_cold_{}", smoke["id"]);
                let row = source_step(&mut ledger, &name, &mut world, &[]);
                results.push(gate(
                    &name,
                    "original_full_wrapper_cold",
                    row["returned_ok"] == true
                        && row["integration_delta"] == 1
                        && row["completed_tick_delta"] == 1
                        && row["torque_delta"] == 1,
                    row,
                ));
            }
            let Some(selected) = recipe["selected"].as_u64() else {
                uncovered.push(json!({"case":format!("A_{family}_original_warmstart"),"reason":"no shallow eligible original-geometry pair in 64 frozen legal qpos candidates"}));
                continue;
            };
            let q: Vec<f64> =
                serde_json::from_value(recipe["candidates"][selected as usize]["qpos"].clone())
                    .unwrap();
            let mut world = SourceCollisionWorld::new(def.clone(), profile.clone(), &q).unwrap();
            let old_token = world.instance_token().unwrap();
            let old_handles: Vec<_> = world.colliders.keys().copied().collect();
            let mut history = Vec::new();
            for i in 0..8 {
                if world.instance_token().is_err() {
                    break;
                }
                let row = source_step(
                    &mut ledger,
                    &format!("A_{family}_history_{i}"),
                    &mut world,
                    &[],
                );
                let good = row["returned_ok"] == true;
                history.push(row);
                if !good {
                    break;
                }
            }
            let actual_warm = history.iter().any(|r| warm(&r["contact_bits"]));
            let consecutive = history.windows(2).any(|w| {
                let pairs = |v: &Value| -> HashSet<String> {
                    v["contact_bits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|p| p["warmstart_positive"] == true)
                        .map(|p| p["geoms"].to_string())
                        .collect()
                };
                !pairs(&w[0]).is_disjoint(&pairs(&w[1]))
            });
            let actual_positive = history.iter().any(|r| calls_allowed(r, "contact", true));
            if !actual_warm || !consecutive || !actual_positive {
                uncovered.push(json!({"case":format!("A_{family}_original_warmstart"),"reason":"original history did not reach all actual positive callback/nonzero and persistent native warmstart gates","positive_callback":actual_positive,"actual_nonzero_warmstart":actual_warm,"consecutive_pair":consecutive}));
            }
            if world.instance_token().is_err() {
                original.push(json!({"family":family,"history":history,"rebuild_uncovered":"history failed terminally"}));
                continue;
            }
            // Clone admission faults after a real history, not a sleep-based hook count proxy.
            let mut faults = Vec::new();
            for fault in [
                "far_extra",
                "sleep_extra",
                "disabled_extra",
                "source_sleep",
                "source_disabled",
                "missing_contact_hook",
                "missing_intersection_hook",
            ] {
                let before = world.snapshot();
                let mut undo_body = None;
                let mut undo_collider = None;
                let source = *world.bodies.keys().next().unwrap();
                let collider = *world.colliders.keys().next().unwrap();
                match fault {
                    "far_extra" | "sleep_extra" | "disabled_extra" => {
                        let b = world.simulation.world.bodies.insert(
                            RigidBodyBuilder::dynamic()
                                .translation(Vector::new(10000., 10000., 10000.)),
                        );
                        let g = world.simulation.world.colliders.insert_with_parent(
                            ColliderBuilder::ball(0.1),
                            b,
                            &mut world.simulation.world.bodies,
                        );
                        if fault == "sleep_extra" {
                            world.simulation.world.bodies[b].sleep();
                        }
                        if fault == "disabled_extra" {
                            world.simulation.world.colliders[g].set_enabled(false);
                        }
                        undo_body = Some(b);
                    }
                    "source_sleep" => world.simulation.world.bodies[source].sleep(),
                    "source_disabled" => {
                        world.simulation.world.colliders[collider].set_enabled(false);
                        undo_collider = Some(collider);
                    }
                    "missing_contact_hook" => world.simulation.world.colliders[collider]
                        .set_active_hooks(ActiveHooks::FILTER_INTERSECTION_PAIR),
                    _ => world.simulation.world.colliders[collider]
                        .set_active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS),
                }
                let fault_before = serde_json::to_value(world.snapshot()).unwrap();
                let torque_before: Vec<_> = world
                    .simulation
                    .world
                    .bodies
                    .iter()
                    .map(|(h, b)| (raw(h), bits(b.user_torque().to_array())))
                    .collect();
                let err = world
                    .step_with_torques(old_token, &[])
                    .err()
                    .map(|e| e.to_string());
                let unchanged = fault_before == serde_json::to_value(world.snapshot()).unwrap()
                    && torque_before
                        == world
                            .simulation
                            .world
                            .bodies
                            .iter()
                            .map(|(h, b)| (raw(h), bits(b.user_torque().to_array())))
                            .collect::<Vec<_>>();
                faults.push(json!({"fault":fault,"error":err,"unchanged":unchanged,"integration_delta":world.snapshot().integration_count-before.integration_count}));
                if let Some(b) = undo_body {
                    world.simulation.remove_body(b).unwrap();
                }
                if let Some(g) = undo_collider {
                    world.simulation.world.colliders[g].set_enabled(true);
                }
                if fault == "source_sleep" {
                    world.simulation.world.bodies[source].wake_up(true);
                }
                world.simulation.world.colliders[collider].set_active_hooks(SOURCE_HOOKS);
            }
            // The above tests preserve entry counters; reject-only body sleep changes are
            // restored before removal, not passed off as unchanged physical state.
            let remove_before = world.snapshot();
            world.remove_robot(old_token).unwrap();
            let removed = world.snapshot();
            let rejected = world.step_with_torques(old_token, &[]).is_err();
            let fresh_token = world.rebuild_robot(&q).unwrap();
            let new_handles: Vec<_> = world.colliders.keys().copied().collect();
            let retire_absent = old_handles
                .iter()
                .all(|h| world.simulation.world.colliders.get(*h).is_none());
            let mut fresh = SourceCollisionWorld::new(def.clone(), profile.clone(), &q).unwrap();
            let mut comparison = Vec::new();
            for i in 0..6 {
                if world.instance_token().is_err() || fresh.instance_token().is_err() {
                    break;
                }
                let a = source_step(
                    &mut ledger,
                    &format!("A_{family}_rebuilt_{i}"),
                    &mut world,
                    &[],
                );
                let b = source_step(
                    &mut ledger,
                    &format!("A_{family}_fresh_{i}"),
                    &mut fresh,
                    &[],
                );
                let exact = a["body_bits"] == b["body_bits"]
                    && a["joint_state_bits"] == b["joint_state_bits"]
                    && canonical_contacts(&a["contact_bits"])
                        == canonical_contacts(&b["contact_bits"]);
                comparison.push(
                    json!({"tick":i+1,"exact_body_joint_contact_bits":exact,"rebuilt":a,"fresh":b}),
                );
            }
            let pass = actual_warm
                && consecutive
                && actual_positive
                && history.len() == 8
                && comparison.len() == 6
                && comparison
                    .iter()
                    .all(|r| r["exact_body_joint_contact_bits"] == true)
                && retire_absent
                && rejected
                && old_token != fresh_token
                && removed.integration_count == remove_before.integration_count;
            results.push(gate(&format!("A_{family}_warmstart_reset"),"original_full_source_geometry",pass,json!({"history":history,"faults":faults,
                "old_token":old_token,"new_token":fresh_token,"old_collider_handles":old_handles.into_iter().map(raw).collect::<Vec<_>>(),
                "new_collider_handles":new_handles.into_iter().map(raw).collect::<Vec<_>>(),"after_remove":removed,"retired_absent":retire_absent,
                "old_token_rejected_without_pipeline":rejected,"comparison":comparison})));
            original.push(json!({"family":family,"original_geometry_warmstart_coverage":actual_warm&&consecutive&&actual_positive}));
        }
        let profile = all[0].1.clone();
        let source_pairs = [[8, 18], [6, 18], [25, 28]];
        let native = read(
            ".scratch/foundation_engineering/collision_eligibility_native_v1/results/leg_allcollisions_robot_only.json",
        );
        for pair in source_pairs {
            let eligible = native["eligible_pairs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p == &json!(pair));
            assert_eq!(
                eligible,
                pair == [8, 18],
                "independent native oracle must qualify fixture selection"
            );
        }
        // B: real source hooks on explicit sphere surrogates.
        for pair in source_pairs {
            for reverse in [false, true] {
                let mut f = Fixture::new(profile.clone(), pair, reverse, false, false, false);
                let name = format!("B_contact_{pair:?}_{reverse}");
                let row = f.step(&mut ledger, &name, None);
                let allowed = pair == [8, 18];
                let pass = calls_allowed(&row, "contact", allowed)
                    && !row["lookup_failure"].as_bool().unwrap()
                    && if allowed {
                        row["active_contacts"].as_u64().unwrap() > 0
                            && row["contact_bits"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|p| p["impulse_bits"].as_u64() != Some(0))
                    } else {
                        row["active_contacts"] == 0 && !warm(&row["contact_bits"])
                    };
                results.push(gate(&name, "source_hooks_sphere_surrogate", pass, row));
                if allowed {
                    let name = format!("B_contact_allowed_second_{reverse}");
                    let row = f.step(&mut ledger, &name, None);
                    results.push(gate(
                        &name,
                        "source_hooks_sphere_surrogate",
                        calls_allowed(&row, "contact", true),
                        row,
                    ));
                }
            }
        }
        for pair in [[6, 18], [25, 28]] {
            let mut f = Fixture::new(profile.clone(), pair, false, false, false, false);
            for g in f.geoms {
                f.sim.world.colliders[g].set_active_hooks(ActiveHooks::empty());
            }
            let name = format!("B_no_flags_{pair:?}");
            let row = f.step(&mut ledger, &name, None);
            results.push(gate(
                &name,
                "unfiltered_sphere_control",
                row["active_contacts"].as_u64().unwrap() > 0 && list(&row, "calls").is_empty(),
                row,
            ));
        }
        for same_body in [true, false] {
            let mut f = Fixture::new(profile.clone(), [8, 18], false, false, false, false);
            if same_body {
                f.sim.world.colliders.set_parent(
                    f.geoms[1],
                    Some(f.bodies[0]),
                    &mut f.sim.world.bodies,
                );
            } else {
                f.sim.world.impulse_joints.insert(
                    f.bodies[0],
                    f.bodies[1],
                    FixedJointBuilder::new().contacts_enabled(false),
                    true,
                );
            }
            let name = format!("B_native_early_gate_samebody_{same_body}");
            let row = f.step(&mut ledger, &name, None);
            results.push(gate(
                &name,
                "backend_early_gate_sphere_control",
                list(&row, "calls").is_empty() && row["active_contacts"] == 0,
                row,
            ));
        }
        for masks in [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 2, 2, 0],
            [1, 2, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [0, 0, 3, 3],
            [3, 3, 1, 2],
        ] {
            let expected=read(format!(".scratch/foundation_engineering/collision_eligibility_native_v1/results/mask_{}_{}_{}_{}.json",masks[0],masks[1],masks[2],masks[3]))["actual_ncon"].as_u64().unwrap()>0;
            let a = GeomMasks {
                contype: masks[0],
                conaffinity: masks[1],
            };
            let b = GeomMasks {
                contype: masks[2],
                conaffinity: masks[3],
            };
            let mut f = Fixture::new(profile.clone(), [8, 18], false, false, false, false);
            let name = format!("B_asymmetric_mask_{masks:?}");
            let row = f.step(&mut ledger, &name, Some((a, b)));
            let pass = calls_allowed(&row, "mask_contact", expected)
                && (row["active_contacts"].as_u64().unwrap() > 0) == expected;
            results.push(gate(&name,"asymmetric_mask_test_hook_not_admitted_source_profile",pass,json!({"native_expected":expected,"and_result":a.contype&b.conaffinity!=0 && b.contype&a.conaffinity!=0,"runtime":row})));
        }
        // C: intersection hooks and cold flags. Sensors never enter qualified world.
        for pair in source_pairs {
            for reverse in [false, true] {
                let mut f = Fixture::new(profile.clone(), pair, reverse, true, false, false);
                let name = format!("C_intersection_{pair:?}_{reverse}");
                let row = f.step(&mut ledger, &name, None);
                let expected = pair == [8, 18];
                let started = list(&row, "events").iter().any(|e| e["kind"] == "Started");
                results.push(gate(
                    &name,
                    "sensor_sphere_test_only",
                    calls_allowed(&row, "intersection", expected)
                        && row["intersection"] == expected
                        && started == expected,
                    row,
                ));
            }
        }
        for pair in [[6, 18], [25, 28]] {
            let mut f = Fixture::new(profile.clone(), pair, false, true, false, false);
            for g in f.geoms {
                f.sim.world.colliders[g].set_active_hooks(ActiveHooks::empty());
            }
            let name = format!("C_no_flags_{pair:?}");
            let row = f.step(&mut ledger, &name, None);
            results.push(gate(
                &name,
                "unfiltered_sensor_control",
                row["intersection"] == true && list(&row, "calls").is_empty(),
                row,
            ));
        }
        {
            let mut f = Fixture::new(profile.clone(), [6, 18], false, true, false, false);
            for g in f.geoms {
                f.sim.world.colliders[g].set_active_hooks(ActiveHooks::empty());
            }
            let first = f.step(&mut ledger, "C_sleep_hot_attach_setup", None);
            for b in f.bodies {
                f.sim.world.bodies[b].sleep();
            }
            for g in f.geoms {
                f.sim.world.colliders[g].set_active_hooks(SOURCE_HOOKS);
            }
            let second = f.step(&mut ledger, "C_sleep_hot_attach_probe", None);
            let skip = list(&second, "calls").is_empty() && second["intersection"] == true;
            if !skip {
                uncovered.push(json!({"case":"C_sleep_hot_attach","reason":"actual native sleep/update masks did not satisfy skip control","first":first,"second":second}));
            }
            results.push(gate(
                "C_sleep_hot_attach",
                "unsupported_hot_attach_mechanism_control",
                skip,
                json!({"first":first,"second":second}),
            ));
        }
        {
            let mut f = Fixture::new(profile.clone(), [8, 18], false, true, false, false);
            let first = f.step(&mut ledger, "C_sensor_generation_setup", None);
            let old = f.geoms[1];
            let body = f.bodies[1];
            f.sim.remove_body(body).unwrap();
            f.registry.remove(&old);
            f.owners.remove(&body);
            let new_body = f.sim.world.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .can_sleep(false)
                    .translation(Vector::new(0.45, 0., 0.))
                    .user_data(profile.data().geom_bodyid[18] as u128),
            );
            let new_geom = f.sim.world.colliders.insert_with_parent(
                ColliderBuilder::ball(0.5)
                    .sensor(true)
                    .active_hooks(SOURCE_HOOKS)
                    .active_events(ActiveEvents::COLLISION_EVENTS)
                    .user_data(18),
                new_body,
                &mut f.sim.world.bodies,
            );
            let c = &f.sim.world.colliders[new_geom];
            f.registry.insert(
                new_geom,
                RegisteredCollider {
                    token: f.token,
                    source_geom: 18,
                    parent: new_body,
                    shape: c.shared_shape().clone(),
                    local_pose: *c.position_wrt_parent().unwrap(),
                },
            );
            f.owners.insert(
                new_body,
                RegisteredBody {
                    token: f.token,
                    source_body: profile.data().geom_bodyid[18],
                    attached: HashSet::from([new_geom]),
                },
            );
            f.geoms[1] = new_geom;
            f.bodies[1] = new_body;
            let second = f.step(&mut ledger, "C_sensor_generation_rebuilt", None);
            let live_started = list(&second, "events")
                .iter()
                .filter(|e| e["kind"] == "Started")
                .all(|e| e["currently_live"] == json!([true, true]));
            results.push(gate("C_sensor_generation","sensor_sphere_test_only",first["intersection"]==true && second["intersection"]==true && old!=new_geom && live_started,json!({"old":raw(old),"new":raw(new_geom),"first":first,"second":second,"old_stopped_events_permitted":true})));
        }
        // D: first-entry CCD and pseudo events (the latter use contact hook only).
        for sensor in [false, true] {
            for pair in source_pairs {
                for reverse in [false, true] {
                    let mut f = Fixture::new(profile.clone(), pair, reverse, sensor, true, false);
                    let name = format!("D_first_ccd_sensor_{sensor}_{pair:?}_{reverse}");
                    let row = f.step(&mut ledger, &name, None);
                    let expected = pair == [8, 18];
                    let x = row["positions"][0][0].as_f64().unwrap();
                    let events = list(&row, "events");
                    let calls = list(&row, "calls");
                    let ccd_call = calls.iter().any(|c| {
                        c["kind"] == "contact"
                            && c["allowed"] == expected
                            && c["caller_backtrace"]
                                .as_str()
                                .unwrap_or("")
                                .contains("sweep")
                    });
                    let response = if sensor {
                        x > 1.0
                            && if expected {
                                events.iter().any(|e| e["kind"] == "Started")
                                    && events.iter().any(|e| e["kind"] == "Stopped")
                            } else {
                                events.is_empty()
                            }
                    } else if expected {
                        x > -3. && x < 0.
                    } else {
                        x > 1.
                    };
                    results.push(gate(&name,"ccd_sphere_test_only",ccd_call && response && !row["lookup_failure"].as_bool().unwrap(),json!({"runtime":row,"real_ccd_callback_stack":ccd_call,"sensor_events_via_contact_hook_only":sensor})));
                }
            }
        }
        for reverse in [false, true] {
            let mut f = Fixture::new(profile.clone(), [6, 18], reverse, false, true, false);
            for g in f.geoms {
                f.sim.world.colliders[g].set_active_hooks(ActiveHooks::empty());
            }
            let name = format!("D_no_flags_solid_{reverse}");
            let row = f.step(&mut ledger, &name, None);
            let x = row["positions"][0][0].as_f64().unwrap();
            results.push(gate(
                &name,
                "ccd_unfiltered_sphere_control",
                x > -3. && x < 0. && list(&row, "calls").is_empty(),
                row,
            ));
            let mut f = Fixture::new(profile.clone(), [8, 18], reverse, false, true, false);
            f.sim.world.bodies[f.bodies[0]].enable_ccd(false);
            f.sim.world.bodies[f.bodies[1]].set_body_type(RigidBodyType::Dynamic, true);
            let name = format!("D_nonbullet_dynamic_target_{reverse}");
            let row = f.step(&mut ledger, &name, None);
            let swept = list(&row, "calls").iter().any(|c| {
                c["caller_backtrace"]
                    .as_str()
                    .unwrap_or("")
                    .contains("sweep")
            });
            results.push(gate(
                &name,
                "native_ccd_target_tier_control",
                !swept,
                json!({"runtime":row,"no_ccd_hook_call":!swept}),
            ));
        }
        // E: cache-isolation primitive baseline with real nonzero native warmstart.
        {
            let mut f = Fixture::new(profile.clone(), [8, 18], false, false, false, true);
            let mut history = Vec::new();
            for i in 0..6 {
                history.push(f.step(&mut ledger, &format!("E_warm_history_{i}"), None));
            }
            let nonzero = history
                .windows(2)
                .any(|v| warm(&v[0]["contact_bits"]) && warm(&v[1]["contact_bits"]));
            let removed = f.rebuild_dynamic(false, true);
            let mut fresh = Fixture::new(profile.clone(), [8, 18], false, false, false, true);
            let mut comparison = Vec::new();
            for i in 0..6 {
                let a = f.step(&mut ledger, &format!("E_rebuilt_{i}"), None);
                let b = fresh.step(&mut ledger, &format!("E_fresh_{i}"), None);
                let exact = a["body_bits"] == b["body_bits"]
                    && canonical_contacts(&a["contact_bits"])
                        == canonical_contacts(&b["contact_bits"]);
                comparison.push(json!({"tick":i+1,"exact_bits":exact,"rebuilt":a,"fresh":b}));
            }
            results.push(gate("E_actual_warmstart_generation_reset","sphere_cache_mechanism_only",nonzero && removed["retired_absent"]==true && comparison.iter().all(|r|r["exact_bits"]==true),json!({"actual_nonzero_warmstart":nonzero,"history":history,"removal":removed,"comparison":comparison})));
        }
        // F: actual public guarded-step failure accounting, never relabelled zero steps.
        for (index, (def, profile)) in all.iter().enumerate() {
            let recipe = &prepared["families"][index];
            let family = &def.model().family;
            if let Some(selected) = recipe["selected"].as_u64() {
                let q: Vec<f64> =
                    serde_json::from_value(recipe["candidates"][selected as usize]["qpos"].clone())
                        .unwrap();
                let mut world =
                    SourceCollisionWorld::new(def.clone(), profile.clone(), &q).unwrap();
                let geom =
                    recipe["candidates"][selected as usize]["shallow_eligible_overlaps"][0]["geoms"]
                        [0]
                    .as_u64()
                    .unwrap() as usize;
                world.late_registry_fault = world
                    .colliders
                    .iter()
                    .find_map(|(h, r)| (r.source_geom == geom).then_some(*h));
                let old_token = world.instance_token().unwrap();
                let row = source_step(
                    &mut ledger,
                    &format!("F_{family}_late_registry"),
                    &mut world,
                    &[],
                );
                let follow_before = world.snapshot().integration_count;
                let follow = world.step_with_torques(old_token, &[]).is_err();
                let pass = row["returned_ok"] == false
                    && row["integration_delta"] == 1
                    && row["torque_delta"] == 1
                    && row["token_valid"] == false
                    && follow
                    && world.snapshot().integration_count == follow_before;
                if !pass {
                    uncovered.push(json!({"case":format!("F_{family}_late_registry"),"reason":"the injected original-geom entry did not trigger an actual missing-registry callback"}));
                }
                results.push(gate(
                    &format!("F_{family}_late_registry"),
                    "original_full_wrapper_test_only_late_fault",
                    pass,
                    row,
                ));
            } else {
                uncovered.push(json!({"case":format!("F_{family}_late_registry"),"reason":"no shallow original-geom selected pose"}));
            }
            let q = def.model().fields.key_qpos[0].clone();
            let mut world = SourceCollisionWorld::new(def.clone(), profile.clone(), &q).unwrap();
            let body = world
                .source_body_handle(world.instance_token().unwrap(), 1)
                .unwrap();
            let row = source_step(
                &mut ledger,
                &format!("F_{family}_natural_quarantine"),
                &mut world,
                &[BodyTorque {
                    body,
                    world_torque: [f32::MAX, 0., 0.],
                }],
            );
            let pass = row["returned_ok"] == false
                && row["integration_delta"] == 1
                && row["torque_delta"] == 1
                && row["token_valid"] == false;
            if !pass {
                uncovered.push(json!({"case":format!("F_{family}_natural_quarantine"),"reason":"finite extreme torque did not produce the expected native poststep error"}));
            }
            results.push(gate(
                &format!("F_{family}_natural_quarantine"),
                "original_source_poststep_accounting_control",
                pass,
                row,
            ));
        }
        let failed: Vec<_> = results
            .iter()
            .filter(|r| r["passed"] != true)
            .map(|r| r["case"].clone())
            .collect();
        save(
            "runtime_results.json",
            &json!({"scope":"bounded_runtime_backend_mechanism_validation","planned_entries":118,"entries_before_run":start_entries,
            "entries_this_run":ledger.used-start_entries,"persistent_total_entries":ledger.used,"hard_limit":128,"results":results,"failed_cases":failed,
            "original_geometry":original,"uncovered":uncovered,"passed_all_measured_gates":failed.is_empty(),
            "policy_inferences":0,"BAM_calls":0,"GPU_frames":0,"learning_steps":0,"production_environment_admitted":false,
            "production_sensors_admitted":false,"full_BAM_external_load_qualified":false,"target_plant_accepted":false}),
        );
        assert!(
            failed.is_empty(),
            "actual measured gates failed; immutable report retained: {failed:?}"
        );
    }

    /// Eight fixed first-step closing-velocity controls. This is separate from
    /// the immutable 118-entry run and does not alter the production SourceHooks.
    #[test]
    #[ignore = "single approved supplementary batch, global entries 119 through 126"]
    fn supplement_closing_velocity() {
        let recipe = read(root().join("supplement_recipe.json"));
        let cases = recipe["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 8);
        let prior = read(root().join("frozen_before_entries.json"));
        assert_eq!(prior["prior_pipeline_entries"], 118);
        let mut ledger = Ledger::new();
        assert_eq!(ledger.offset, 118);
        assert_eq!(ledger.used, 0, "this approved batch is single-use");
        let profile = fixtures()[0].1.clone();
        let mut results = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            let pair: [usize; 2] = serde_json::from_value(case["source_pair"].clone()).unwrap();
            let reverse = case["reverse"].as_bool().unwrap();
            let no_hooks = case["no_hooks"].as_bool().unwrap();
            assert_eq!(case["local_case"], index + 1);
            assert_eq!(
                pair,
                match index % 4 {
                    0 | 3 => [8, 18],
                    1 => [6, 18],
                    _ => [25, 28],
                }
            );
            assert_eq!(reverse, index >= 4);
            assert_eq!(no_hooks, index % 4 == 3);
            assert_eq!(case["velocity_x_mps"], json!([1.0, -1.0]));
            let mut f = Fixture::new(profile.clone(), pair, reverse, false, false, false);
            // Same sphere, density, mass, positions, gravity and solver as B.
            // Only the first-step finite closing velocities change.
            f.sim.world.bodies[f.bodies[0]].set_linvel(Vector::new(1.0, 0.0, 0.0), true);
            f.sim.world.bodies[f.bodies[1]].set_linvel(Vector::new(-1.0, 0.0, 0.0), true);
            if no_hooks {
                for geom in f.geoms {
                    f.sim.world.colliders[geom].set_active_hooks(ActiveHooks::empty());
                }
            }
            let name = format!("supplement_closing_{index}_{pair:?}_{reverse}_{no_hooks}");
            let row = f.step(&mut ledger, &name, None);
            let impulse_positive = row["contact_bits"].as_array().unwrap().iter().any(|point| {
                point["impulse_bits"].as_u64().is_some_and(|raw| {
                    let impulse = f32::from_bits(raw as u32);
                    impulse.is_finite() && impulse > 0.0
                })
            });
            let pass = if pair == [8, 18] {
                let hook = if no_hooks {
                    row["calls"].as_array().unwrap().is_empty()
                } else {
                    calls_allowed(&row, "contact", true)
                };
                hook && row["active_contacts"].as_u64().unwrap_or(0) > 0 && impulse_positive
            } else {
                calls_allowed(&row, "contact", false)
                    && row["active_contacts"] == 0
                    && !impulse_positive
            };
            results.push(gate(&name, if no_hooks {"unfiltered_closing_velocity_sphere_control"}
                else {"source_hooks_closing_velocity_sphere_surrogate"}, pass,
                json!({"runtime":row,"actual_positive_normal_impulse":impulse_positive,"source_pair":pair,"reverse":reverse,
                    "no_hooks":no_hooks,"first_step_velocities_mps":[[1.0,0.0,0.0],[-1.0,0.0,0.0]]})));
        }
        let failed: Vec<_> = results
            .iter()
            .filter(|r| r["passed"] != true)
            .map(|r| r["case"].clone())
            .collect();
        save(
            "supplement_results.json",
            &json!({"scope":"separate_closing_velocity_sphere_mechanism_only",
            "original_first_batch_source_and_three_failures_unchanged":true,
            "global_entry_ids":[119,120,121,122,123,124,125,126],
            "entries_this_batch":ledger.used,"total_entries_across_batches":ledger.offset+ledger.used,
            "hard_limit":128,"results":results,"failed_cases":failed,"all_eight_gates_passed":failed.is_empty(),
            "production_sensor_or_environment_admitted":false,"original_geometry_or_force_equivalence_qualified":false,
            "hot_attach_sleep_coverage":false,"policy_inferences":0,"BAM_calls":0,"GPU_frames":0,"learning_steps":0}),
        );
        assert!(
            failed.is_empty(),
            "supplementary nonzero impulse or exclusion gate failed; original result retained: {failed:?}"
        );
    }
}
