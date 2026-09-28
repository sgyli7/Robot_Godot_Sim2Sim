//! Read-only Rapier solver diagnostics for a registered source articulation.
//!
//! These values are native generalized forces and impulses. The residual below
//! is an arithmetic candidate only: it is not a MuJoCo `qfrc_bias`,
//! `qfrc_constraint`, or qualified BAM `external_load` measurement. The prior
//! applied actuator is not captured here either.

use rapier3d::prelude::Multibody;
use robot_minigame::{ACTION_DIMENSION, joint_feedback::NativeHandleIdentity};
use serde::Serialize;
use thiserror::Error;

use super::{BoundaryResult, RobotInstanceToken, SourceCollisionWorld, invalid};

/// Why no complete native candidate can be read at the current tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PreviousSolveUnavailable {
    NoCompletedStep,
    UnsupportedBackend,
    NativeObservationInvalid,
    ContactCoverageIncomplete,
}

/// A diagnostic read; an unavailable read never substitutes fourteen zeros.
#[derive(Clone, Debug, Serialize)]
pub enum PreviousSolveDiagnostic {
    Unavailable(PreviousSolveUnavailable),
    Candidate(NativeSolveCandidateFrame),
}

/// One source-driven DOF, with raw Rapier terms in N·m and N·m·s.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct NativeSolveCandidateChannel {
    pub source_joint: usize,
    pub source_dof: usize,
    pub backend_dof: usize,
    pub joint_handle: NativeHandleIdentity,
    pub body_handle: NativeHandleIdentity,
    pub inertial_projection_nm: f32,
    pub gravity_projection_nm: f32,
    pub generic_joint_impulse_nms: f32,
    /// This is a subset of `generic_joint_impulse_nms`, never an extra row.
    pub own_dry_friction_impulse_nms: f32,
    pub contact_normal_impulse_nms: f32,
    pub contact_tangent_impulse_nms: f32,
    pub owner_generic_joint_row_side_count: usize,
    pub owner_own_dry_friction_row_side_count: usize,
    pub owner_contact_expected_side_count: usize,
    pub owner_contact_measured_side_count: usize,
    pub owner_energy_guard_evaluated: bool,
    pub owner_energy_guard_fallback: bool,
    pub owner_energy_guard_acceleration_cleared: bool,
    /// `gravity - inertial + (generic - own + normal + tangent) / actual_dt`.
    /// This is only a Rapier diagnostic residual, not BAM external load.
    pub residual_excluding_own_friction_nm: f32,
}

/// A complete native diagnostic tied to exactly one registered instance/tick.
/// The production `JointFeedbackFrame.previous_solve_load` remains unavailable.
#[derive(Clone, Debug, Serialize)]
pub struct NativeSolveCandidateFrame {
    pub scope: &'static str,
    pub model_file_sha256: String,
    pub collision_profile_file_sha256: String,
    pub world_epoch: u64,
    pub robot_instance: u64,
    pub episode_id: u64,
    pub global_step: u64,
    pub episode_step: u64,
    pub integration_count: u64,
    pub native_observation_epoch: u64,
    pub native_topology_epoch: u32,
    /// Read from the completed Rapier observation, not inferred from a rate.
    pub actual_step_dt_seconds: f32,
    pub channels: [NativeSolveCandidateChannel; ACTION_DIMENSION],
    pub bam_input_qualified: bool,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DiagnosticBridgeError {
    #[error("native diagnostic time step must be finite and positive")]
    InvalidStepDt,
    #[error("native diagnostic contains a non-finite force or impulse")]
    NonFiniteComponent,
    #[error("native diagnostic residual exceeds finite f32 range")]
    NonFiniteResidual,
    #[error("native observation epoch is absent or stale")]
    StaleObservationEpoch,
    #[error("native observation time step differs from the integration setting")]
    StepDtMismatch,
    #[error("native observation topology epochs differ across driven DOFs")]
    TopologyMismatch,
}

/// Pure unit conversion and subset subtraction; no physics or BAM law runs.
pub fn native_residual_excluding_own_friction(
    inertial_nm: f32,
    gravity_nm: f32,
    generic_joint_impulse_nms: f32,
    own_dry_friction_impulse_nms: f32,
    contact_normal_impulse_nms: f32,
    contact_tangent_impulse_nms: f32,
    actual_step_dt_seconds: f32,
) -> Result<f32, DiagnosticBridgeError> {
    if !actual_step_dt_seconds.is_finite() || actual_step_dt_seconds <= 0.0 {
        return Err(DiagnosticBridgeError::InvalidStepDt);
    }
    let values = [
        inertial_nm,
        gravity_nm,
        generic_joint_impulse_nms,
        own_dry_friction_impulse_nms,
        contact_normal_impulse_nms,
        contact_tangent_impulse_nms,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DiagnosticBridgeError::NonFiniteComponent);
    }
    let impulse_without_own_friction = f64::from(generic_joint_impulse_nms)
        - f64::from(own_dry_friction_impulse_nms)
        + f64::from(contact_normal_impulse_nms)
        + f64::from(contact_tangent_impulse_nms);
    let residual = f64::from(gravity_nm) - f64::from(inertial_nm)
        + impulse_without_own_friction / f64::from(actual_step_dt_seconds);
    let residual = residual as f32;
    if !residual.is_finite() {
        return Err(DiagnosticBridgeError::NonFiniteResidual);
    }
    Ok(residual)
}

/// Reject a warm-looking value from a previous tick, topology, or step size.
fn validate_observation_identity(
    integration_count: u64,
    configured_dt: f32,
    observed_epoch: u64,
    observed_dt: f32,
    first_topology_epoch: Option<u32>,
    observed_topology_epoch: u32,
) -> Result<(), DiagnosticBridgeError> {
    if integration_count == 0 || observed_epoch != integration_count {
        return Err(DiagnosticBridgeError::StaleObservationEpoch);
    }
    if !configured_dt.is_finite() || configured_dt <= 0.0 {
        return Err(DiagnosticBridgeError::InvalidStepDt);
    }
    if !observed_dt.is_finite()
        || observed_dt <= 0.0
        || observed_dt.to_bits() != configured_dt.to_bits()
    {
        return Err(DiagnosticBridgeError::StepDtMismatch);
    }
    if first_topology_epoch.is_some_and(|first| first != observed_topology_epoch) {
        return Err(DiagnosticBridgeError::TopologyMismatch);
    }
    Ok(())
}

impl SourceCollisionWorld {
    /// Read a completed Rapier diagnostic after the guarded source step.
    /// It never changes production BAM eligibility, even when all rows exist.
    pub fn diagnostic_previous_solve(
        &self,
        token: RobotInstanceToken,
    ) -> BoundaryResult<PreviousSolveDiagnostic> {
        let feedback = self.joint_feedback_frame(token)?;
        let snapshot = self.simulation.snapshot();
        if snapshot.episode_step == 0 || snapshot.integration_count == 0 {
            return Ok(PreviousSolveDiagnostic::Unavailable(
                PreviousSolveUnavailable::NoCompletedStep,
            ));
        }
        if !Multibody::sim2sim_observation_backend_supported() {
            return Ok(PreviousSolveDiagnostic::Unavailable(
                PreviousSolveUnavailable::UnsupportedBackend,
            ));
        }

        let assembly = self
            .assembly
            .as_ref()
            .ok_or_else(|| invalid("diagnostic source assembly is absent"))?;
        let mut channels = Vec::with_capacity(ACTION_DIMENSION);
        let mut native_epoch = None;
        let mut topology_epoch = None;
        let mut actual_dt = None;
        for channel in feedback.channels {
            let mapping = assembly
                .joint_mapping()
                .iter()
                .find(|mapping| mapping.source_joint == channel.source_joint)
                .ok_or_else(|| invalid("diagnostic source joint mapping is absent"))?;
            let (multibody, link_id) =
                self.simulation
                    .world
                    .multibody_joints
                    .get(mapping.handle)
                    .ok_or_else(|| invalid("diagnostic articulation generation is stale"))?;
            let link = multibody
                .link(link_id)
                .ok_or_else(|| invalid("diagnostic source link is absent"))?;
            if mapping.backend_dof != channel.backend_dof
                || link.assembly_id() != channel.backend_dof
                || link.rigid_body_handle().into_raw_parts()
                    != (channel.body_handle.index, channel.body_handle.generation)
                || mapping.handle.into_raw_parts()
                    != (channel.joint_handle.index, channel.joint_handle.generation)
            {
                return Err(invalid("diagnostic source DOF or handle identity changed"));
            }
            let (epoch, valid, coverage) = multibody.sim2sim_observation_status();
            if !valid {
                return Ok(PreviousSolveDiagnostic::Unavailable(
                    PreviousSolveUnavailable::NativeObservationInvalid,
                ));
            }
            if !coverage {
                return Ok(PreviousSolveDiagnostic::Unavailable(
                    PreviousSolveUnavailable::ContactCoverageIncomplete,
                ));
            }
            let observation = multibody
                .sim2sim_contact_complete_observation()
                .ok_or_else(|| invalid("diagnostic observation status disagrees with getter"))?;
            validate_observation_identity(
                snapshot.integration_count,
                self.simulation.world.integration_parameters.dt,
                epoch,
                observation.full_step_dt(),
                topology_epoch,
                observation.topology_epoch,
            )
            .map_err(|error| invalid(error.to_string()))?;
            if native_epoch.is_some_and(|first| first != epoch) {
                return Err(invalid(
                    "diagnostic native epochs differ across driven DOFs",
                ));
            }
            if actual_dt
                .is_some_and(|first: f32| first.to_bits() != observation.full_step_dt().to_bits())
            {
                return Err(invalid("diagnostic native dt differs across driven DOFs"));
            }
            native_epoch = Some(epoch);
            topology_epoch = Some(observation.topology_epoch);
            actual_dt = Some(observation.full_step_dt());

            let dof = channel.backend_dof;
            let terms = [
                observation.inertial_projection.get(dof),
                observation.gravity_projection.get(dof),
                observation.generic_joint_impulse.get(dof),
                observation.own_dry_friction_impulse.get(dof),
                observation.contact_normal_impulse.get(dof),
                observation.contact_tangent_impulse.get(dof),
            ];
            let [
                Some(&inertial),
                Some(&gravity),
                Some(&generic),
                Some(&own),
                Some(&normal),
                Some(&tangent),
            ] = terms
            else {
                return Err(invalid("diagnostic native DOF vector is incomplete"));
            };
            if observation.own_dry_friction_row_side_count
                > observation.generic_joint_row_side_count
            {
                return Err(invalid("diagnostic own-friction rows exceed generic rows"));
            }
            let residual = native_residual_excluding_own_friction(
                inertial,
                gravity,
                generic,
                own,
                normal,
                tangent,
                observation.full_step_dt(),
            )
            .map_err(|error| invalid(error.to_string()))?;
            channels.push(NativeSolveCandidateChannel {
                source_joint: channel.source_joint,
                source_dof: channel.source_dof,
                backend_dof: dof,
                joint_handle: channel.joint_handle,
                body_handle: channel.body_handle,
                inertial_projection_nm: inertial,
                gravity_projection_nm: gravity,
                generic_joint_impulse_nms: generic,
                own_dry_friction_impulse_nms: own,
                contact_normal_impulse_nms: normal,
                contact_tangent_impulse_nms: tangent,
                owner_generic_joint_row_side_count: observation.generic_joint_row_side_count,
                owner_own_dry_friction_row_side_count: observation.own_dry_friction_row_side_count,
                owner_contact_expected_side_count: observation.contact_expected_side_count,
                owner_contact_measured_side_count: observation.contact_measured_side_count,
                owner_energy_guard_evaluated: observation.energy_guard_evaluated,
                owner_energy_guard_fallback: observation.energy_guard_fallback,
                owner_energy_guard_acceleration_cleared: observation
                    .energy_guard_acceleration_cleared,
                residual_excluding_own_friction_nm: residual,
            });
        }
        let channels: [NativeSolveCandidateChannel; ACTION_DIMENSION] = channels
            .try_into()
            .map_err(|_| invalid("diagnostic driven channel count changed"))?;
        Ok(PreviousSolveDiagnostic::Candidate(
            NativeSolveCandidateFrame {
                scope: "rapier_native_previous_solve_diagnostic_only",
                model_file_sha256: feedback.model_file_sha256,
                collision_profile_file_sha256: feedback.collision_profile_file_sha256,
                world_epoch: feedback.world_epoch,
                robot_instance: feedback.robot_instance,
                episode_id: feedback.episode_id,
                global_step: feedback.global_step,
                episode_step: feedback.episode_step,
                integration_count: snapshot.integration_count,
                native_observation_epoch: native_epoch
                    .ok_or_else(|| invalid("diagnostic native epoch is absent"))?,
                native_topology_epoch: topology_epoch
                    .ok_or_else(|| invalid("diagnostic native topology is absent"))?,
                actual_step_dt_seconds: actual_dt
                    .ok_or_else(|| invalid("diagnostic native dt is absent"))?,
                channels,
                bam_input_qualified: false,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_subtracts_own_friction_from_generic_subset_once() {
        let result =
            native_residual_excluding_own_friction(3.0, 5.0, 0.75, 0.25, 0.5, -0.25, 0.25).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn bridge_uses_actual_dt_and_rejects_invalid_values() {
        let first =
            native_residual_excluding_own_friction(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1).unwrap();
        let second =
            native_residual_excluding_own_friction(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2).unwrap();
        assert_eq!(first, 10.0);
        assert_eq!(second, 5.0);
        assert_eq!(
            native_residual_excluding_own_friction(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Err(DiagnosticBridgeError::InvalidStepDt)
        );
        assert_eq!(
            native_residual_excluding_own_friction(0.0, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.1),
            Err(DiagnosticBridgeError::NonFiniteComponent)
        );
        assert_eq!(
            native_residual_excluding_own_friction(0.0, 0.0, f32::MAX, 0.0, 0.0, 0.0, 0.01),
            Err(DiagnosticBridgeError::NonFiniteResidual)
        );
    }

    #[test]
    fn identity_rejects_cold_stale_dt_and_topology() {
        assert_eq!(
            validate_observation_identity(0, 0.1, 0, 0.1, None, 4),
            Err(DiagnosticBridgeError::StaleObservationEpoch)
        );
        assert_eq!(
            validate_observation_identity(2, 0.1, 1, 0.1, None, 4),
            Err(DiagnosticBridgeError::StaleObservationEpoch)
        );
        assert_eq!(
            validate_observation_identity(2, 0.1, 2, 0.2, None, 4),
            Err(DiagnosticBridgeError::StepDtMismatch)
        );
        assert_eq!(
            validate_observation_identity(2, 0.1, 2, 0.1, Some(3), 4),
            Err(DiagnosticBridgeError::TopologyMismatch)
        );
        assert_eq!(
            validate_observation_identity(2, 0.1, 2, 0.1, Some(4), 4),
            Ok(())
        );
    }
}
