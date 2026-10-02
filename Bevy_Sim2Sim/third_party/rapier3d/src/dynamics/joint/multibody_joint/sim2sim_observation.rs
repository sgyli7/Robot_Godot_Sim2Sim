//! Local addition: read-only observations of existing multibody solver operands.
// This module never feeds data back into physical simulation.

use crate::alloc_prelude::*;
use crate::dynamics::solver::GenericContactConstraint;
use crate::dynamics::solver::solver_contact_graph::ContactRef;
use crate::dynamics::solver::{GenericJointConstraint, WritebackId};
use crate::dynamics::{Multibody, MultibodyIndex, RigidBodyHandle};
use crate::dynamics::{MultibodyJointSet, MultibodyLinkId};
use crate::math::Real;
#[cfg(feature = "sim2sim-limit-row-trace")]
use crate::math::SPATIAL_DIM;
use alloc::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct ContactSideOwner {
    pub body: RigidBodyHandle,
    pub multibody: MultibodyIndex,
    pub link_id: usize,
    pub ndofs: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ContactConstraintIdentity {
    pub reference: ContactRef,
    pub contact_ids: Vec<u8>,
    pub sides: [Option<ContactSideOwner>; 2],
    pub generic_sides: [bool; 2],
    pub rigid_solver_ids: [Option<u32>; 2],
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ContactObservationManifest {
    pub complete: bool,
    pub constraints: Vec<ContactConstraintIdentity>,
}

/// A read-only checkpoint in the staged solver. These samples are diagnostic
/// operands, not MuJoCo forces or a qualified BAM external load.
#[cfg(feature = "sim2sim-limit-row-trace")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitRowTracePhase {
    /// The position-bias pass has solved the limit row.
    AfterBiasedSolve,
    /// The biased velocity has been integrated into the coordinate.
    AfterPositionIntegration,
    /// The relaxation pass has removed the position-bias RHS.
    AfterUnbiasedSolve,
}

#[cfg(feature = "sim2sim-limit-row-trace")]
impl LimitRowTracePhase {
    /// A stable label for diagnostic reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AfterBiasedSolve => "after_biased_solve",
            Self::AfterPositionIntegration => "after_position_integration",
            Self::AfterUnbiasedSolve => "after_unbiased_solve",
        }
    }
}

/// One internal limit row at one checkpoint within a solver substep.
#[cfg(feature = "sim2sim-limit-row-trace")]
#[derive(Clone, Copy, Debug)]
pub struct LimitRowTraceSample {
    /// Point within one staged solver substep.
    pub phase: LimitRowTracePhase,
    /// Solver substep ordinal, starting at zero.
    pub substep_id: usize,
    /// Index in the generic joint constraint vector.
    pub row_index: usize,
    /// Row's DOF index within the owning joint.
    pub joint_local_dof: usize,
    /// Row's DOF index within the multibody articulation.
    pub backend_dof: usize,
    /// Signed unit Jacobian entry; predictive lower stops use -1.
    pub jacobian_sign: Real,
    /// Generalized coordinate at this checkpoint.
    pub coordinate: Real,
    /// Generalized solver velocity at this checkpoint.
    pub generalized_velocity: Real,
    /// Current constraint RHS, including bias before relaxation.
    pub rhs: Real,
    /// RHS retained when the position bias is removed.
    pub rhs_without_bias: Real,
    /// Raw solver lambda; the generalized impulse is `-J * lambda` on side 2.
    pub impulse: Real,
    /// Nonnegative upper-limit or nonpositive lower-limit impulse bounds.
    pub impulse_bounds: [Real; 2],
}

/// One original impulse-joint row acting on a multibody's second-side block.
/// For a same-owner loop this block is the actual relative Jacobian J2-J1.
/// Only fixed/empty first blocks are supported by this diagnostic.
#[cfg(feature = "sim2sim-limit-row-trace")]
#[derive(Clone, Debug)]
pub struct NativeJointRowTraceSample {
    /// Existing staged barrier, without an added solve or integration.
    pub phase: LimitRowTracePhase,
    /// Existing temporal substep ordinal.
    pub substep_id: usize,
    /// Original solver row index.
    pub row_index: usize,
    /// Original impulse-joint graph index.
    pub joint_index: usize,
    /// Original impulse-joint DOF index, before any numerical row transform.
    pub joint_local_dof: usize,
    /// Raw accumulated native lambda at this barrier.
    pub impulse: Real,
    /// Generalized impulse -J2^T*lambda on this owned second-side block.
    pub generalized_impulse_side2: Vec<Real>,
    /// Original second-side Jacobian at this barrier.
    pub jacobian_side2: Vec<Real>,
    /// Native inverse-mass response to that original Jacobian.
    pub weighted_jacobian_side2: Vec<Real>,
    /// Original solver velocities, before position integration at the biased barrier.
    pub solver_velocity_side2: Vec<Real>,
    /// Original RHS at this barrier, including any remaining position bias.
    pub rhs: Real,
    /// Native constraint force mixing coefficient in the impulse update.
    pub cfm_gain: Real,
    /// Native reciprocal row inertia, including constraint force mixing.
    pub inverse_row_inertia: Real,
}

/// Read-only operands immediately before and after one native generic row update.
/// This diagnostic supports a multibody second side and an empty first side.
#[cfg(feature = "sim2sim-limit-row-trace")]
#[derive(Clone, Debug)]
pub struct NativeGenericJointUpdateSample {
    /// Existing solver row ordinal; repeated ordinals delimit existing passes.
    pub row_index: usize,
    /// Whether the existing relaxation pass removed position bias.
    pub without_bias: bool,
    /// Original writeback kind, copied before any classification by the consumer.
    pub writeback_kind: &'static str,
    /// Original graph joint identity; None identifies an internal row.
    pub joint_index: Option<usize>,
    /// Native lambda immediately before this update.
    pub impulse_before: Real,
    /// Native lambda immediately after this update.
    pub impulse_after: Real,
    /// Original permitted native lambda interval.
    pub impulse_bounds: [Real; 2],
    /// Original unweighted second-side Jacobian.
    pub jacobian: Vec<Real>,
    /// Original native inverse-mass response to that Jacobian.
    pub weighted_jacobian: Vec<Real>,
    /// Native generalized velocities immediately before this row update.
    pub velocity_before: Vec<Real>,
    /// Native generalized velocities immediately after this row update.
    pub velocity_after: Vec<Real>,
    /// Native row RHS used by this update.
    pub rhs: Real,
    /// Native CFM gain used by this update.
    pub cfm_gain: Real,
    /// Native reciprocal row inertia used by this update.
    pub inverse_row_inertia: Real,
}

/// The native operands of one contact row whose two sides share an articulation.
/// Contact Jacobians are already signed: its relative row is J1 + J2.
/// Computed responses use f64 accumulation of the actual stored solver operands;
/// they exclude contact softness and do not add a solve or modify an impulse.
#[cfg(feature = "sim2sim-limit-row-trace")]
#[derive(Clone, Debug)]
pub struct NativeContactMassTraceSample {
    /// Existing biased-solve barrier, before position integration.
    pub phase: LimitRowTracePhase,
    /// Existing temporal substep ordinal.
    pub substep_id: usize,
    /// Native contact graph edge and solver manifold ordinal.
    pub manifold_reference: [u32; 2],
    /// Original manifold point identity.
    pub contact_id: u8,
    /// Zero is normal; one and two are the original tangential rows.
    pub row_kind: usize,
    /// Original rigid bodies owning the two signed row blocks.
    pub bodies: [RigidBodyHandle; 2],
    /// Original link identities within the shared owner.
    pub link_ids: [usize; 2],
    /// Width of each original shared-owner row block.
    pub ndofs: usize,
    /// Native r coefficient, before the separate contact softness factor.
    pub native_reciprocal_response: Real,
    /// J1 dot WJ1 and J2 dot WJ2, from the actual row blocks.
    pub side_response: [f64; 2],
    /// J1 dot WJ2 and J2 dot WJ1; absent in a side-only denominator.
    pub cross_response: [f64; 2],
    /// (J1 + J2) dot (WJ1 + WJ2), the shared-velocity unit-impulse response.
    pub combined_response: f64,
    /// Original accumulated impulse at this barrier, including warmstart accounting.
    pub total_impulse: Real,
}

/// Partial measurements from one completed full pipeline step.
///
/// The partial getter retains stage-A validity; complete native contact data needs
/// the explicit contact-complete getter. Neither getter qualifies BAM external load.
/// Parallel-backend collection is not qualified in stage A and returns no valid data.
#[derive(Clone, Debug, Default)]
pub struct MultibodyObservation {
    /// Epoch of the owning joint set's full pipeline step, never a solver offset.
    pub epoch: u64,
    /// Topology epoch associated with the measurements.
    pub topology_epoch: u32,
    /// True only after finite, single-step data has completed all phase-A gates.
    pub valid: bool,
    /// True only after the independent native contact manifest and final rows match.
    pub contact_coverage: bool,
    /// Measured inertial projection Jᵀ(m a_bias, gyro + I α_bias), before M⁻¹.
    pub inertial_projection: Vec<Real>,
    /// Jᵀ(effective linear force − user force, 0), before M⁻¹.
    /// This records the actual effective-force subtraction, including its rounding.
    pub gravity_projection: Vec<Real>,
    /// Jᵀ of the world-space user force and torque actually queued on bodies.
    /// This read-only diagnostic does not change the articulation solve.
    pub user_force_projection: Vec<Real>,
    /// Sum of signed original, unweighted generic-joint J λ rows only.
    pub generic_joint_impulse: Vec<Real>,
    /// Subset of generic_joint_impulse from native articulation dry-friction rows.
    pub own_dry_friction_impulse: Vec<Real>,
    /// Number of recorded generic-joint row sides for this articulation.
    pub generic_joint_row_side_count: usize,
    /// Stage-specific internal limit rows, only when the separate trace feature is enabled.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub limit_row_timing: Vec<LimitRowTraceSample>,
    /// Original impulse-joint rows at the same diagnostic barriers.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub joint_row_timing: Vec<NativeJointRowTraceSample>,
    /// Per-row update operands, only with the separate serial trace feature.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub generic_joint_updates: Vec<NativeGenericJointUpdateSample>,
    /// Original shared-owner contact row responses at the biased-solve barrier.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub contact_mass_timing: Vec<NativeContactMassTraceSample>,
    /// Every eligible row matched the native contact ownership manifest and layout.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub contact_mass_trace_complete: bool,
    /// Number of recorded native dry-friction row sides.
    pub own_dry_friction_row_side_count: usize,
    /// Whether the existing implicit-Coriolis energy guard was evaluated.
    pub energy_guard_evaluated: bool,
    /// Whether that original guard entered its plain-mass fallback.
    pub energy_guard_fallback: bool,
    /// Whether the original last-chance fallback cleared acceleration.
    pub energy_guard_acceleration_cleared: bool,
    /// A separate diagnostic request bypassed the implicit velocity-dependent matrix.
    #[cfg(feature = "sim2sim-plain-mass-probe")]
    pub plain_mass_probe_selected: bool,
    /// Sum of native signed normal-contact J times final total impulse.
    /// Experimental combined same-owner rows use a zero first block and the
    /// relative second block; this is an owner total, not per-link attribution.
    pub contact_normal_impulse: Vec<Real>,
    /// Sum of both actual original tangential-contact rows times final total impulse.
    pub contact_tangent_impulse: Vec<Real>,
    /// Number of active point sides expected by the pre-integration audit.
    pub contact_expected_side_count: usize,
    /// Number of active point sides actually reconciled and collected.
    pub contact_measured_side_count: usize,
    /// Number of original normal row sides collected.
    pub contact_normal_row_side_count: usize,
    /// Number of original tangential row sides collected (two per 3D point side).
    pub contact_tangent_row_side_count: usize,
    /// Whether the actual final contact collector ran for this active owner.
    pub contact_collector_completed: bool,
    /// Whether every expected contact owner/identity/layout was reconciled.
    pub contact_ownership_complete: bool,
    contact_manifest: Option<Arc<ContactObservationManifest>>,
    pub(crate) full_step_dt: Real,
    pub(crate) solver_assignment_count: usize,
    pub(crate) solver_offset: Option<u32>,
    pub(crate) single_temporal_step: bool,
    pub(crate) bias_recorded: bool,
    pub(crate) joint_rows_recorded: bool,
    pub(crate) row_ownership_complete: bool,
}

impl MultibodyObservation {
    /// The actual full-pipeline time step associated with these solver operands.
    /// It is zero for an invalidated cold state, never an inferred display rate.
    pub fn full_step_dt(&self) -> Real {
        self.full_step_dt
    }

    pub(crate) fn begin(epoch: u64, topology_epoch: u32, dt: Real) -> Self {
        Self {
            epoch,
            topology_epoch,
            full_step_dt: dt,
            row_ownership_complete: true,
            ..Self::default()
        }
    }

    pub(crate) fn allocate(&mut self, ndofs: usize) {
        self.inertial_projection.resize(ndofs, 0.0);
        self.inertial_projection.fill(0.0);
        self.gravity_projection.resize(ndofs, 0.0);
        self.gravity_projection.fill(0.0);
        self.user_force_projection.resize(ndofs, 0.0);
        self.user_force_projection.fill(0.0);
        self.generic_joint_impulse.resize(ndofs, 0.0);
        self.generic_joint_impulse.fill(0.0);
        self.own_dry_friction_impulse.resize(ndofs, 0.0);
        self.own_dry_friction_impulse.fill(0.0);
        self.contact_normal_impulse.resize(ndofs, 0.0);
        self.contact_normal_impulse.fill(0.0);
        self.contact_tangent_impulse.resize(ndofs, 0.0);
        self.contact_tangent_impulse.fill(0.0);
    }

    pub(crate) fn finish(&mut self, topology_epoch: u32, one_ccd_step: bool, finite_world: bool) {
        self.valid = self.epoch != 0
            && !cfg!(feature = "parallel")
            && self.topology_epoch == topology_epoch
            && one_ccd_step
            && finite_world
            && self.single_temporal_step
            && self.solver_assignment_count == 1
            && self.bias_recorded
            && self.joint_rows_recorded
            && self.row_ownership_complete
            && self
                .inertial_projection
                .iter()
                .chain(&self.gravity_projection)
                .chain(&self.user_force_projection)
                .chain(&self.generic_joint_impulse)
                .chain(&self.own_dry_friction_impulse)
                .all(|value| value.is_finite());
        self.contact_coverage = self.valid
            && self.contact_collector_completed
            && self.contact_ownership_complete
            && self
                .contact_manifest
                .as_ref()
                .is_some_and(|manifest| manifest.complete)
            && self.contact_expected_side_count == self.contact_measured_side_count
            && self.contact_normal_row_side_count == self.contact_expected_side_count
            && self.contact_expected_side_count.checked_mul(2)
                == Some(self.contact_tangent_row_side_count)
            && self.contact_normal_impulse.len() == self.inertial_projection.len()
            && self.contact_tangent_impulse.len() == self.inertial_projection.len()
            && self
                .contact_normal_impulse
                .iter()
                .chain(&self.contact_tangent_impulse)
                .all(|value| value.is_finite());
    }
}

impl MultibodyJointSet {
    /// Copy an existing row update without adding a solve or modifying velocities.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub(crate) fn observe_generic_joint_update(
        &mut self,
        roots: &[MultibodyLinkId],
        constraint: &GenericJointConstraint,
        row_index: usize,
        without_bias: bool,
        impulse_before: Real,
        velocity_before: Vec<Real>,
        jacobians: &[Real],
        velocities_after: &[Real],
    ) {
        let Some(root) = roots.iter().find(|root| {
            let mb = &self.multibodies[root.multibody.0];
            mb.solver_id == constraint.solver_vel2 && mb.ndofs() == constraint.ndofs2
        }) else {
            return;
        };
        let offset = constraint.j_id2;
        let ndofs = constraint.ndofs2;
        let velocity_offset = constraint.solver_vel2 as usize;
        let sample = NativeGenericJointUpdateSample {
            row_index,
            without_bias,
            writeback_kind: match constraint.writeback_id {
                WritebackId::Dof(_) => "dof",
                WritebackId::Limit(_) => "limit",
                WritebackId::Motor(_) => "motor",
                WritebackId::Friction(_) => "friction",
            },
            joint_index: (constraint.joint_id != usize::MAX).then_some(constraint.joint_id),
            impulse_before,
            impulse_after: constraint.impulse,
            impulse_bounds: constraint.impulse_bounds,
            jacobian: jacobians[offset..offset + ndofs].to_vec(),
            weighted_jacobian: jacobians[offset + ndofs..offset + 2 * ndofs].to_vec(),
            velocity_before,
            velocity_after: velocities_after[velocity_offset..velocity_offset + ndofs].to_vec(),
            rhs: constraint.rhs,
            cfm_gain: constraint.cfm_gain,
            inverse_row_inertia: constraint.inv_lhs,
        };
        self.multibodies[root.multibody.0]
            .sim2sim_observation
            .generic_joint_updates
            .push(sample);
    }

    /// Capture one checkpoint without changing constraint state or solver velocity.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub(crate) fn observe_limit_row_timing(
        &mut self,
        roots: &[MultibodyLinkId],
        constraints: &[GenericJointConstraint],
        jacobians: &[Real],
        solver_velocities: &[Real],
        phase: LimitRowTracePhase,
        substep_id: usize,
    ) {
        for (row_index, constraint) in constraints.iter().enumerate() {
            if let WritebackId::Dof(joint_local_dof) = constraint.writeback_id
                && constraint.joint_id != usize::MAX
                && !constraint.is_rigid_body2
                && constraint.ndofs1 == 0
                && constraint.solver_vel1 == u32::MAX
                && let Some(root) = roots.iter().find(|root| {
                    let multibody = &self.multibodies[root.multibody.0];
                    multibody.solver_id == constraint.solver_vel2
                        && multibody.ndofs() == constraint.ndofs2
                })
                && let Some(row) = constraint
                    .j_id2
                    .checked_add(constraint.ndofs2)
                    .and_then(|end| jacobians.get(constraint.j_id2..end))
            {
                let sample = NativeJointRowTraceSample {
                    phase,
                    substep_id,
                    row_index,
                    joint_index: constraint.joint_id,
                    joint_local_dof,
                    impulse: constraint.impulse,
                    generalized_impulse_side2: row
                        .iter()
                        .map(|j| -j * constraint.impulse)
                        .collect(),
                    jacobian_side2: row.to_vec(),
                    weighted_jacobian_side2: jacobians
                        .get(
                            constraint.j_id2 + constraint.ndofs2
                                ..constraint.j_id2 + 2 * constraint.ndofs2,
                        )
                        .unwrap_or(&[])
                        .to_vec(),
                    solver_velocity_side2: solver_velocities
                        .get(
                            constraint.solver_vel2 as usize
                                ..constraint.solver_vel2 as usize + constraint.ndofs2,
                        )
                        .unwrap_or(&[])
                        .to_vec(),
                    rhs: constraint.rhs,
                    cfm_gain: constraint.cfm_gain,
                    inverse_row_inertia: constraint.inv_lhs,
                };
                self.multibodies[root.multibody.0]
                    .sim2sim_observation
                    .joint_row_timing
                    .push(sample);
            }
            let WritebackId::Limit(joint_local_dof) = constraint.writeback_id else {
                continue;
            };
            if constraint.joint_id != usize::MAX || constraint.is_rigid_body2 {
                continue;
            }
            // Internal motor rows currently carry a Limit writeback tag too;
            // their symmetric bounds distinguish them from a one-sided limit.
            if constraint.impulse_bounds[0] < 0.0 && constraint.impulse_bounds[1] > 0.0 {
                continue;
            }
            let Some(root) = roots.iter().find(|root| {
                let multibody = &self.multibodies[root.multibody.0];
                multibody.solver_id == constraint.solver_vel2
                    && multibody.ndofs() == constraint.ndofs2
            }) else {
                continue;
            };
            let Some(row) = constraint
                .j_id2
                .checked_add(constraint.ndofs2)
                .and_then(|end| jacobians.get(constraint.j_id2..end))
            else {
                continue;
            };
            let Some(backend_dof) = row.iter().position(|value| value.abs() == 1.0) else {
                continue;
            };
            // Native internal limit rows contain exactly one unit Jacobian entry.
            if row
                .iter()
                .enumerate()
                .any(|(index, value)| index != backend_dof && *value != 0.0)
            {
                continue;
            }
            let multibody = &self.multibodies[root.multibody.0];
            let Some(link) = multibody.links().find(|link| {
                link.assembly_id() <= backend_dof
                    && backend_dof < link.assembly_id() + link.joint().ndofs()
            }) else {
                continue;
            };
            let local_dof = backend_dof - link.assembly_id();
            let locked_bits = link.joint().data.locked_axes.bits();
            let Some(axis) = (0..SPATIAL_DIM)
                .filter(|axis| locked_bits & (1 << axis) == 0)
                .nth(local_dof)
            else {
                continue;
            };
            let coordinate = link.joint().coords()[axis];
            let Some(&generalized_velocity) =
                solver_velocities.get(constraint.solver_vel2 as usize + backend_dof)
            else {
                continue;
            };
            let sample = LimitRowTraceSample {
                phase,
                substep_id,
                row_index,
                joint_local_dof,
                backend_dof,
                jacobian_sign: row[backend_dof],
                coordinate,
                generalized_velocity,
                rhs: constraint.rhs,
                rhs_without_bias: constraint.rhs_wo_bias,
                impulse: constraint.impulse,
                impulse_bounds: constraint.impulse_bounds,
            };
            self.multibodies[root.multibody.0]
                .sim2sim_observation
                .limit_row_timing
                .push(sample);
        }
    }

    /// Invalidates diagnostics after a caller-managed cold state reset, without stepping.
    /// Topology insertion/removal invokes this automatically.
    pub fn invalidate_sim2sim_observations(&mut self) {
        for (_, multibody) in &mut self.multibodies {
            multibody.sim2sim_observation = MultibodyObservation::begin(
                self.sim2sim_observation_epoch,
                self.topology_epoch,
                0.0,
            );
        }
    }

    pub(crate) fn begin_sim2sim_observation_step(&mut self, dt: Real) {
        self.sim2sim_observation_epoch = self.sim2sim_observation_epoch.wrapping_add(1).max(1);
        for (_, multibody) in &mut self.multibodies {
            multibody.sim2sim_observation = MultibodyObservation::begin(
                self.sim2sim_observation_epoch,
                self.topology_epoch,
                dt,
            );
        }
    }

    pub(crate) fn finish_sim2sim_observation_step(
        &mut self,
        one_ccd_step: bool,
        finite_world: bool,
    ) {
        for (_, multibody) in &mut self.multibodies {
            multibody
                .sim2sim_observation
                .finish(self.topology_epoch, one_ccd_step, finite_world);
        }
    }

    pub(crate) fn observe_generic_joint_rows(
        &mut self,
        roots: &[MultibodyLinkId],
        constraints: &[GenericJointConstraint],
        jacobians: &[Real],
    ) {
        // Rebuilt from this solve's actual roots; solver offsets are never stable identities.
        let owners: Vec<_> = roots
            .iter()
            .map(|root| {
                let multibody = &self.multibodies[root.multibody.0];
                (multibody.solver_id, multibody.ndofs(), root.multibody)
            })
            .collect();
        for root in roots {
            let multibody = &mut self.multibodies[root.multibody.0];
            multibody.sim2sim_observation.joint_rows_recorded = true;
            if multibody.sim2sim_observation.solver_offset != Some(multibody.solver_id) {
                multibody.sim2sim_observation.row_ownership_complete = false;
            }
        }
        for constraint in constraints {
            let own_friction = constraint.joint_id == usize::MAX
                && matches!(constraint.writeback_id, WritebackId::Friction(_));
            for (is_rigid, offset, ndofs, j_id, sign) in [
                (
                    constraint.is_rigid_body1,
                    constraint.solver_vel1,
                    constraint.ndofs1,
                    constraint.j_id1,
                    1.0,
                ),
                (
                    constraint.is_rigid_body2,
                    constraint.solver_vel2,
                    constraint.ndofs2,
                    constraint.j_id2,
                    -1.0,
                ),
            ] {
                if is_rigid || ndofs == 0 {
                    continue;
                }
                let owner = owners
                    .iter()
                    .find(|(start, width, _)| *start == offset && *width == ndofs);
                let row = j_id
                    .checked_add(ndofs)
                    .and_then(|end| jacobians.get(j_id..end));
                let Some((_, _, owner)) =
                    owner.filter(|_| row.is_some() && constraint.impulse.is_finite())
                else {
                    for root in roots {
                        self.multibodies[root.multibody.0]
                            .sim2sim_observation
                            .row_ownership_complete = false;
                    }
                    continue;
                };
                let observation = &mut self.multibodies[owner.0].sim2sim_observation;
                // Original J begins at j_id; weighted M^-1 J begins at j_id + ndofs and is not read.
                for (index, value) in row.unwrap().iter().enumerate() {
                    let signed_impulse = sign * *value * constraint.impulse;
                    observation.generic_joint_impulse[index] += signed_impulse;
                    if own_friction {
                        observation.own_dry_friction_impulse[index] += signed_impulse;
                    }
                }
                observation.generic_joint_row_side_count += 1;
                if own_friction {
                    observation.own_dry_friction_row_side_count += 1;
                }
            }
        }
    }
}

impl Multibody {
    /// Completed native normal/two-tangent contact collection, in addition to stage-A validity.
    /// This is not a BAM external-load API, and asserts no source torsion/rolling compatibility.
    pub fn sim2sim_contact_complete_observation(&self) -> Option<&MultibodyObservation> {
        let observation = &self.sim2sim_observation;
        (observation.valid && observation.contact_coverage).then_some(observation)
    }
}

impl MultibodyJointSet {
    pub(crate) fn prepare_contact_observation(&mut self, manifest: ContactObservationManifest) {
        let manifest = Arc::new(manifest);
        for (index, multibody) in &mut self.multibodies {
            let observation = &mut multibody.sim2sim_observation;
            observation.contact_ownership_complete = manifest.complete;
            let count = manifest
                .constraints
                .iter()
                .try_fold(0_usize, |total, constraint| {
                    let owner_sides = constraint
                        .sides
                        .iter()
                        .filter(|side| side.as_ref().is_some_and(|side| side.multibody.0 == index))
                        .count();
                    total.checked_add(constraint.contact_ids.len().checked_mul(owner_sides)?)
                });
            observation.contact_expected_side_count = count.unwrap_or(0);
            observation.contact_ownership_complete &= count.is_some();
            observation.contact_manifest = Some(Arc::clone(&manifest));
        }
    }

    fn reject_contact_observation(&mut self, roots: &[MultibodyLinkId]) {
        for root in roots {
            self.multibodies[root.multibody.0]
                .sim2sim_observation
                .contact_ownership_complete = false;
        }
    }

    /// Read existing contact operands only at the completed biased-solve barrier.
    #[cfg(feature = "sim2sim-limit-row-trace")]
    pub(crate) fn observe_contact_mass_timing(
        &mut self,
        roots: &[MultibodyLinkId],
        constraints: &[GenericContactConstraint],
        jacobians: &[Real],
        substep_id: usize,
    ) {
        let manifest = roots.first().and_then(|root| {
            self.multibodies[root.multibody.0]
                .sim2sim_observation
                .contact_manifest
                .clone()
        });
        for root in roots {
            self.multibodies[root.multibody.0]
                .sim2sim_observation
                .contact_mass_trace_complete = manifest.as_ref().is_some_and(|m| m.complete);
        }
        let Some(manifest) = manifest.filter(|m| m.complete) else {
            return;
        };
        for constraint in constraints {
            let Some(expected) = manifest
                .constraints
                .iter()
                .find(|entry| entry.reference == constraint.manifold_id)
            else {
                for root in roots {
                    self.multibodies[root.multibody.0]
                        .sim2sim_observation
                        .contact_mass_trace_complete = false;
                }
                return;
            };
            let [Some(side1), Some(side2)] = &expected.sides else {
                continue;
            };
            if side1.multibody != side2.multibody {
                continue;
            }
            let owner = side1.multibody;
            let multibody = &self.multibodies[owner.0];
            let n = multibody.ndofs();
            let count = constraint.num_contacts as usize;
            let matched = roots.iter().any(|root| root.multibody == owner)
                && n != 0
                && constraint.ndofs1 == n
                && constraint.ndofs2 == n
                && constraint.solver_vel1 == multibody.solver_id
                && constraint.solver_vel2 == multibody.solver_id
                && constraint.generic_constraint_mask == 0b11
                && expected.generic_sides == [true, true]
                && side1.ndofs == n
                && side2.ndofs == n
                && count <= constraint.manifold_contact_id.len()
                && constraint.manifold_contact_id[..count] == expected.contact_ids
                && [side1, side2].iter().all(|side| {
                    self.rigid_body_link(side.body)
                        .is_some_and(|link| link.multibody == owner && link.id == side.link_id)
                });
            if !matched {
                self.multibodies[owner.0]
                    .sim2sim_observation
                    .contact_mass_trace_complete = false;
                continue;
            }
            for k in 0..count {
                for row_kind in 0..3 {
                    let start = constraint.j_id + (k * 3 + row_kind) * 4 * n;
                    let Some(blocks) = jacobians.get(start..start + 4 * n) else {
                        self.multibodies[owner.0]
                            .sim2sim_observation
                            .contact_mass_trace_complete = false;
                        continue;
                    };
                    let j1 = &blocks[..n];
                    let wj1 = &blocks[n..2 * n];
                    let j2 = &blocks[2 * n..3 * n];
                    let wj2 = &blocks[3 * n..];
                    let dot = |left: &[Real], right: &[Real]| -> f64 {
                        left.iter()
                            .zip(right)
                            .map(|(a, b)| (*a as f64) * (*b as f64))
                            .sum()
                    };
                    let combined_response = (0..n)
                        .map(|i| (j1[i] as f64 + j2[i] as f64) * (wj1[i] as f64 + wj2[i] as f64))
                        .sum();
                    let (native_reciprocal_response, total_impulse) = if row_kind == 0 {
                        (
                            constraint.normal_part[k].r,
                            constraint.normal_part[k].total_impulse(),
                        )
                    } else {
                        (
                            constraint.tangent_part[k].r[row_kind - 1],
                            constraint.tangent_part[k].total_impulse()[row_kind - 1],
                        )
                    };
                    let sample = NativeContactMassTraceSample {
                        phase: LimitRowTracePhase::AfterBiasedSolve,
                        substep_id,
                        manifold_reference: [
                            constraint.manifold_id.edge,
                            constraint.manifold_id.manifold,
                        ],
                        contact_id: constraint.manifold_contact_id[k],
                        row_kind,
                        bodies: [side1.body, side2.body],
                        link_ids: [side1.link_id, side2.link_id],
                        ndofs: n,
                        native_reciprocal_response,
                        side_response: [dot(j1, wj1), dot(j2, wj2)],
                        cross_response: [dot(j1, wj2), dot(j2, wj1)],
                        combined_response,
                        total_impulse,
                    };
                    if !blocks.iter().all(|value| value.is_finite())
                        || !sample.native_reciprocal_response.is_finite()
                        || !sample.combined_response.is_finite()
                        || !sample.total_impulse.is_finite()
                    {
                        self.multibodies[owner.0]
                            .sim2sim_observation
                            .contact_mass_trace_complete = false;
                        continue;
                    }
                    self.multibodies[owner.0]
                        .sim2sim_observation
                        .contact_mass_timing
                        .push(sample);
                }
            }
        }
    }

    pub(crate) fn observe_contact_rows(
        &mut self,
        roots: &[MultibodyLinkId],
        constraints: &[GenericContactConstraint],
        jacobians: &[Real],
    ) {
        let Some(manifest) = roots.first().and_then(|root| {
            self.multibodies[root.multibody.0]
                .sim2sim_observation
                .contact_manifest
                .clone()
        }) else {
            self.reject_contact_observation(roots);
            return;
        };
        let owners: Vec<_> = roots
            .iter()
            .map(|root| {
                let multibody = &self.multibodies[root.multibody.0];
                (multibody.solver_id, multibody.ndofs(), root.multibody)
            })
            .collect();
        for root in roots {
            self.multibodies[root.multibody.0]
                .sim2sim_observation
                .contact_collector_completed = true;
        }
        if !manifest.complete || constraints.len() != manifest.constraints.len() {
            self.reject_contact_observation(roots);
            return;
        }
        let mut seen = Vec::new();
        let mut next_original_j_id = 0_usize;
        for constraint in constraints {
            if constraint.j_id != next_original_j_id {
                self.reject_contact_observation(roots);
                return;
            }
            if seen.contains(&constraint.manifold_id) {
                self.reject_contact_observation(roots);
                return;
            }
            seen.push(constraint.manifold_id);
            let Some(expected) = manifest
                .constraints
                .iter()
                .find(|expected| expected.reference == constraint.manifold_id)
            else {
                self.reject_contact_observation(roots);
                return;
            };
            let count = constraint.num_contacts as usize;
            if count != expected.contact_ids.len()
                || count > constraint.manifold_contact_id.len()
                || constraint.manifold_contact_id[..count] != expected.contact_ids
            {
                self.reject_contact_observation(roots);
                return;
            }
            let d = match constraint.ndofs1.checked_add(constraint.ndofs2) {
                Some(d) => d,
                None => {
                    self.reject_contact_observation(roots);
                    return;
                }
            };
            let stride = match d.checked_mul(6) {
                Some(stride) => stride,
                None => {
                    self.reject_contact_observation(roots);
                    return;
                }
            };
            let required_end = count
                .checked_mul(stride)
                .and_then(|width| constraint.j_id.checked_add(width));
            if required_end.is_none_or(|end| end > jacobians.len()) {
                self.reject_contact_observation(roots);
                return;
            }
            for k in 0..count {
                if !constraint.normal_part[k].total_impulse().is_finite()
                    || !constraint.tangent_part[k]
                        .total_impulse()
                        .iter()
                        .all(|impulse| impulse.is_finite())
                {
                    self.reject_contact_observation(roots);
                    return;
                }
            }
            next_original_j_id = required_end.unwrap();
            for (side, (ndofs, offset)) in [
                (constraint.ndofs1, constraint.solver_vel1),
                (constraint.ndofs2, constraint.solver_vel2),
            ]
            .into_iter()
            .enumerate()
            {
                let actual_generic = constraint.generic_constraint_mask & (1 << side) != 0;
                if actual_generic != expected.generic_sides[side]
                    || constraint.generic_constraint_mask & !0b11 != 0
                {
                    self.reject_contact_observation(roots);
                    return;
                }
                let Some(owner) = &expected.sides[side] else {
                    if ndofs != 0 || expected.rigid_solver_ids[side] != Some(offset) {
                        self.reject_contact_observation(roots);
                        return;
                    }
                    continue;
                };
                let owner_matches = owners.iter().any(|(start, width, index)| {
                    *start == offset && *width == ndofs && *index == owner.multibody
                });
                let link_matches = self.rigid_body_link(owner.body).is_some_and(|link| {
                    link.multibody == owner.multibody && link.id == owner.link_id
                });
                if !owner_matches
                    || !link_matches
                    || ndofs != owner.ndofs
                    || constraint.generic_constraint_mask & (1 << side) == 0
                {
                    self.reject_contact_observation(roots);
                    return;
                }
                for k in 0..count {
                    let side_offset = if side == 0 {
                        Some(0)
                    } else {
                        constraint.ndofs1.checked_mul(2)
                    };
                    for row in 0_usize..3 {
                        let start = k
                            .checked_mul(stride)
                            .and_then(|start| constraint.j_id.checked_add(start))
                            .and_then(|start| {
                                row.checked_mul(d.checked_mul(2)?)
                                    .and_then(|row| start.checked_add(row))
                            })
                            .and_then(|start| start.checked_add(side_offset?));
                        let original = start.and_then(|start| {
                            start
                                .checked_add(ndofs)
                                .and_then(|end| jacobians.get(start..end))
                        });
                        let impulse = if row == 0 {
                            constraint.normal_part[k].total_impulse()
                        } else {
                            constraint.tangent_part[k].total_impulse()[row - 1]
                        };
                        let Some(original) = original.filter(|original| {
                            impulse.is_finite() && original.iter().all(|value| value.is_finite())
                        }) else {
                            self.reject_contact_observation(roots);
                            return;
                        };
                        let observation =
                            &mut self.multibodies[owner.multibody.0].sim2sim_observation;
                        let destination = if row == 0 {
                            &mut observation.contact_normal_impulse
                        } else {
                            &mut observation.contact_tangent_impulse
                        };
                        if destination.len() != ndofs {
                            self.reject_contact_observation(roots);
                            return;
                        }
                        for (index, value) in original.iter().enumerate() {
                            // Each contact J side is already signed at generation; do not negate side two.
                            destination[index] += *value * impulse;
                        }
                        if row == 0 {
                            observation.contact_normal_row_side_count += 1;
                        } else {
                            observation.contact_tangent_row_side_count += 1;
                        }
                    }
                    self.multibodies[owner.multibody.0]
                        .sim2sim_observation
                        .contact_measured_side_count += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[cfg(feature = "sim2sim-limit-row-trace")]
    fn shared_contact_fixture(
        combined: bool,
        centers: [Vector; 2],
        slide_axis: Vector,
    ) -> (PhysicsWorld, [RigidBodyHandle; 2]) {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        world.integration_parameters.dt = 0.02;
        world.integration_parameters.num_solver_iterations = 1;
        world.integration_parameters.max_ccd_substeps = 0;
        world.integration_parameters.combine_same_multibody_contacts = combined;
        let root = world.insert_body(RigidBodyBuilder::dynamic().additional_mass_properties(
            MassProperties::new(Vector::ZERO, 1.0, Vector::splat(0.1)),
        ));
        world.bodies[root].recompute_mass_properties_from_colliders(&world.colliders);
        let mut children = Vec::new();
        for anchor in centers {
            let child = world.insert_body(
                RigidBodyBuilder::dynamic()
                    .translation(anchor)
                    .additional_mass_properties(MassProperties::new(
                        Vector::ZERO,
                        1.0,
                        Vector::splat(0.1),
                    )),
            );
            world.insert_collider(
                ColliderBuilder::ball(0.1).density(0.0).friction(0.0),
                Some(child),
            );
            world.bodies[child].recompute_mass_properties_from_colliders(&world.colliders);
            world
                .insert_multibody_joint(
                    root,
                    child,
                    PrismaticJointBuilder::new(slide_axis)
                        .local_anchor1(anchor)
                        .contacts_enabled(true),
                )
                .unwrap();
            children.push(child);
        }
        assert_eq!(
            world
                .multibody_joints
                .rigid_body_link(children[0])
                .unwrap()
                .multibody,
            world
                .multibody_joints
                .rigid_body_link(children[1])
                .unwrap()
                .multibody,
        );
        (world, [children[0], children[1]])
    }

    #[cfg(feature = "sim2sim-limit-row-trace")]
    #[test]
    fn actual_shared_owner_contact_mass_includes_cross_response() {
        for combined in [false, true] {
            let (mut world, children) =
                shared_contact_fixture(combined, [-Vector::Y * 0.09, Vector::Y * 0.09], Vector::X);
            world.step();
            let owner = *world.multibody_joints.rigid_body_link(children[0]).unwrap();
            let observation = world
                .multibody_joints
                .get_multibody(owner.multibody)
                .unwrap()
                .sim2sim_observation()
                .unwrap();
            assert!(observation.contact_mass_trace_complete);
            let normal = observation
                .contact_mass_timing
                .iter()
                .find(|row| row.row_kind == 0)
                .unwrap();
            // Three 1 kg bodies with child slides along X: along the Y contact
            // normal all move together. Opposite forces at the same point cancel.
            if combined {
                assert!(normal.side_response.iter().all(|response| *response == 0.0));
                assert_eq!(normal.native_reciprocal_response, 0.0);
                assert_eq!(normal.total_impulse, 0.0);
            } else {
                assert!(
                    (normal.side_response.iter().sum::<f64>() - 2.0 / 3.0).abs() < 1.0e-5,
                    "{normal:?}"
                );
                assert!((normal.cross_response.iter().sum::<f64>() + 2.0 / 3.0).abs() < 1.0e-5);
                assert!(normal.combined_response.abs() < 1.0e-5);
                assert!((normal.native_reciprocal_response - 1.5).abs() < 1.0e-5);
            }
            assert_eq!(normal.phase, LimitRowTracePhase::AfterBiasedSolve);
            assert_eq!(normal.substep_id, 0);
            assert_eq!(observation.contact_mass_timing.len(), 3);
            world.multibody_joints.invalidate_sim2sim_observations();
            assert!(
                world
                    .multibody_joints
                    .get_multibody(owner.multibody)
                    .unwrap()
                    .sim2sim_observation()
                    .is_none()
            );
        }
    }

    #[cfg(feature = "sim2sim-limit-row-trace")]
    #[test]
    fn combined_shared_contact_resolves_nonzero_row_in_one_pass() {
        let normal = Vector::new(1.0, 1.0, 0.0).normalize();
        let mut closing = Vec::new();
        for combined in [false, true] {
            let (mut world, children) =
                shared_contact_fixture(combined, [-normal * 0.1, normal * 0.1], Vector::Y);
            world.integration_parameters.contact_softness = SpringCoefficients::new(1.0e6, 1.0);
            let owner = *world.multibody_joints.rigid_body_link(children[0]).unwrap();
            for (child, velocity) in children.into_iter().zip([1.0, -1.0]) {
                let link = *world.multibody_joints.rigid_body_link(child).unwrap();
                let mb = world
                    .multibody_joints
                    .get_multibody_mut(link.multibody)
                    .unwrap();
                let dof = mb.link(link.id).unwrap().assembly_id();
                mb.generalized_velocity_mut()[dof] = velocity;
            }
            world.step();
            let observation = world
                .multibody_joints
                .get_multibody(owner.multibody)
                .unwrap()
                .sim2sim_observation()
                .unwrap();
            assert!(observation.contact_mass_trace_complete);
            let row = observation
                .contact_mass_timing
                .iter()
                .find(|row| row.row_kind == 0)
                .unwrap();
            assert!(row.combined_response > 0.1);
            if combined {
                assert!(
                    (row.native_reciprocal_response as f64 * row.combined_response - 1.0).abs()
                        < 1.0e-5
                );
                assert!(row.total_impulse > 0.0);
            }
            let mb = world
                .multibody_joints
                .get_multibody(owner.multibody)
                .unwrap();
            let slide_dofs = children.map(|child| {
                let link = world.multibody_joints.rigid_body_link(child).unwrap();
                mb.link(link.id).unwrap().assembly_id()
            });
            // Read the completed native generalized velocity, rather than the
            // rigid-body velocity cache from the pre-integration link update.
            let velocity = mb.generalized_velocity();
            closing.push((normal.y * (velocity[slide_dofs[0]] - velocity[slide_dofs[1]])).abs());
        }
        // The only normal contact has no restitution or friction. A correct
        // relative mass removes its initial closing velocity in the native pass.
        assert!(closing[0] > 0.01, "native closing: {closing:?}");
        assert!(closing[1] < 1.0e-5, "combined closing: {closing:?}");
    }

    fn fixture() -> (
        MultibodyJointSet,
        Vec<MultibodyLinkId>,
        GenericContactConstraint,
        Vec<Real>,
    ) {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        world.integration_parameters.num_solver_iterations = 1;
        let root = world.insert_body(RigidBodyBuilder::fixed());
        let body = world.insert_body(RigidBodyBuilder::dynamic().additional_mass_properties(
            MassProperties::new(Vector::ZERO, 1.0, Vector::splat(0.1)),
        ));
        world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
        let handle = world
            .insert_multibody_joint(root, body, PrismaticJointBuilder::new(Vector::Y))
            .unwrap();
        world.step();
        let initial = world
            .multibody_joints
            .get(handle)
            .unwrap()
            .0
            .sim2sim_contact_complete_observation()
            .unwrap();
        assert_eq!(initial.contact_expected_side_count, 0);
        assert_eq!(initial.contact_measured_side_count, 0);
        assert!(
            initial
                .contact_normal_impulse
                .iter()
                .chain(&initial.contact_tangent_impulse)
                .all(|value| *value == 0.0)
        );
        let owner = *world.multibody_joints.rigid_body_link(body).unwrap();
        let mut constraint = GenericContactConstraint::invalid();
        constraint.manifold_id = ContactRef {
            edge: 0,
            manifold: 0,
        };
        constraint.num_contacts = 1;
        constraint.manifold_contact_id[0] = 0;
        constraint.j_id = 0;
        constraint.ndofs1 = 1;
        constraint.ndofs2 = 0;
        constraint.solver_vel1 = world.multibody_joints.get(handle).unwrap().0.solver_id;
        constraint.solver_vel2 = u32::MAX;
        constraint.generic_constraint_mask = 0b11;
        constraint.normal_part[0].impulse = 0.5;
        constraint.normal_part[0].impulse_accumulator = 0.5;
        constraint.tangent_part[0].impulse = TangentImpulse::new(0.25, -0.25);
        constraint.tangent_part[0].impulse_accumulator = TangentImpulse::new(0.25, -0.25);
        world
            .multibody_joints
            .prepare_contact_observation(ContactObservationManifest {
                complete: true,
                constraints: vec![ContactConstraintIdentity {
                    reference: constraint.manifold_id,
                    contact_ids: vec![0],
                    generic_sides: [true, true],
                    rigid_solver_ids: [None, Some(u32::MAX)],
                    sides: [
                        Some(ContactSideOwner {
                            body,
                            multibody: owner.multibody,
                            link_id: owner.id,
                            ndofs: 1,
                        }),
                        None,
                    ],
                }],
            });
        let observation =
            &mut world.multibody_joints.multibodies[owner.multibody.0].sim2sim_observation;
        observation.contact_collector_completed = false;
        observation.contact_coverage = false;
        observation.contact_measured_side_count = 0;
        observation.contact_normal_row_side_count = 0;
        observation.contact_tangent_row_side_count = 0;
        (
            world.multibody_joints,
            vec![owner],
            constraint,
            vec![2.0, 10_000.0, 3.0, 10_000.0, 4.0, 10_000.0],
        )
    }

    #[test]
    fn original_rows_final_total_and_two_same_owner_sides() {
        let (mut set, roots, mut constraint, mut jacobians) = fixture();
        set.observe_contact_rows(&roots, &[constraint], &jacobians);
        let observation = &set.multibodies[roots[0].multibody.0].sim2sim_observation;
        assert!(observation.contact_ownership_complete);
        assert_eq!(observation.contact_normal_impulse, vec![2.0]);
        assert_eq!(observation.contact_tangent_impulse, vec![-0.5]);
        assert_eq!(observation.contact_normal_row_side_count, 1);
        assert_eq!(observation.contact_tangent_row_side_count, 2);
        let (mut set, roots, _, _) = fixture();
        let manifest = Arc::make_mut(
            set.multibodies[roots[0].multibody.0]
                .sim2sim_observation
                .contact_manifest
                .as_mut()
                .unwrap(),
        );
        manifest.constraints[0].sides[1] = manifest.constraints[0].sides[0].clone();
        manifest.constraints[0].rigid_solver_ids[1] = None;
        set.multibodies[roots[0].multibody.0]
            .sim2sim_observation
            .contact_expected_side_count = 2;
        constraint.ndofs2 = 1;
        constraint.solver_vel2 = constraint.solver_vel1;
        jacobians = vec![
            2.0, 10_000.0, -2.0, 10_000.0, 3.0, 10_000.0, -3.0, 10_000.0, 4.0, 10_000.0, -4.0,
            10_000.0,
        ];
        set.observe_contact_rows(&roots, &[constraint], &jacobians);
        let observation = &set.multibodies[roots[0].multibody.0].sim2sim_observation;
        assert!(observation.contact_ownership_complete);
        assert_eq!(observation.contact_normal_impulse, vec![0.0]);
        assert_eq!(observation.contact_tangent_impulse, vec![0.0]);
        assert_eq!(observation.contact_measured_side_count, 2);
    }

    #[test]
    fn actual_inactive_reactivated_and_reset_contact_lifecycle() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        world.integration_parameters.num_solver_iterations = 1;
        let root = world.insert_body(RigidBodyBuilder::fixed());
        let child = world.insert_body(RigidBodyBuilder::dynamic().additional_mass_properties(
            MassProperties::new(Vector::ZERO, 1.0, Vector::splat(0.1)),
        ));
        world.bodies[child].recompute_mass_properties_from_colliders(&world.colliders);
        let handle = world
            .insert_multibody_joint(root, child, PrismaticJointBuilder::new(Vector::Y))
            .unwrap();
        world.step();
        assert!(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .sim2sim_contact_complete_observation()
                .is_some()
        );
        world.bodies[child].set_enabled(false);
        world.step();
        assert!(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .sim2sim_contact_complete_observation()
                .is_none()
        );
        assert!(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .sim2sim_observation()
                .is_none()
        );
        world.bodies[child].set_enabled(true);
        world.step();
        let data = world
            .multibody_joints
            .get(handle)
            .unwrap()
            .0
            .sim2sim_contact_complete_observation()
            .unwrap();
        assert_eq!(data.epoch, 3);
        assert_eq!(data.contact_measured_side_count, 0);
        world.multibody_joints.invalidate_sim2sim_observations();
        assert!(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .sim2sim_contact_complete_observation()
                .is_none()
        );
        world.step();
        assert!(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .sim2sim_contact_complete_observation()
                .is_some()
        );
        world.integration_parameters.num_solver_iterations = 4;
        world.step();
        assert!(
            world
                .multibody_joints
                .get(handle)
                .unwrap()
                .0
                .sim2sim_contact_complete_observation()
                .is_none()
        );
    }

    #[test]
    fn failed_contact_completeness_preserves_valid_partial_getter() {
        let (mut set, roots, _, _) = fixture();
        let topology = set.topology_epoch;
        let multibody = &mut set.multibodies[roots[0].multibody.0];
        multibody.sim2sim_observation.contact_ownership_complete = false;
        multibody.sim2sim_observation.finish(topology, true, true);
        assert!(multibody.sim2sim_observation().is_some());
        assert!(multibody.sim2sim_contact_complete_observation().is_none());
    }

    #[test]
    fn malformed_contact_diagnostics_fail_closed() {
        for case in 0..16 {
            let (mut set, roots, mut constraint, mut jacobians) = fixture();
            let mut rows = vec![constraint];
            match case {
                0 => rows.clear(),
                1 => rows.push(constraint),
                2 => constraint.manifold_id.edge = 1,
                3 => constraint.manifold_contact_id[0] = 1,
                4 => constraint.ndofs1 = 2,
                5 => constraint.solver_vel1 += 1,
                6 => constraint.generic_constraint_mask = 0,
                7 => jacobians.truncate(1),
                8 => constraint.j_id = usize::MAX,
                9 => constraint.normal_part[0].impulse = Real::NAN,
                10 => jacobians[0] = Real::NAN,
                11 => {
                    Arc::make_mut(
                        set.multibodies[roots[0].multibody.0]
                            .sim2sim_observation
                            .contact_manifest
                            .as_mut()
                            .unwrap(),
                    )
                    .complete = false
                }
                12 => {
                    let manifest = Arc::make_mut(
                        set.multibodies[roots[0].multibody.0]
                            .sim2sim_observation
                            .contact_manifest
                            .as_mut()
                            .unwrap(),
                    );
                    let side = manifest.constraints[0].sides[0].as_mut().unwrap();
                    let (index, generation) = side.multibody.0.into_raw_parts();
                    side.multibody = MultibodyIndex(crate::data::Index::from_raw_parts(
                        index,
                        generation.wrapping_add(1),
                    ));
                }
                13 => constraint.ndofs1 = usize::MAX,
                14 => constraint.solver_vel2 = 0,
                15 => constraint.generic_constraint_mask |= 0b100,
                _ => unreachable!(),
            }
            if case >= 2 {
                rows[0] = constraint;
            }
            set.observe_contact_rows(&roots, &rows, &jacobians);
            let observation = &set.multibodies[roots[0].multibody.0].sim2sim_observation;
            assert!(!observation.contact_ownership_complete, "case {case}");
        }
    }
}
