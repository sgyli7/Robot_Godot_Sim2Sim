//! Bounded read-only native motor evidence for a recorded mechanics replay.
//! A token binds the existing owner, generation and start Tick. No motor,
//! policy observation, contact, gain or solver state is modified here.

use super::{G1ActuatorBackend, G1Runner, error};
use rapier3d::dynamics::MultibodyJointHandle;
use robot_minigame::{
    RobotError,
    g1::{actuator, contract::JOINT_NAMES},
};
use serde_json::{Value, json};
use std::sync::Arc;

/// The existing Rapier trace permits one session per owner: 20 raw Ticks,
/// followed by aggregates up to150. Reset requires a new owner and token.
pub struct G1MotorEffortWindow {
    episode_id: u64,
    start_tick: u64,
    mapping: [(MultibodyJointHandle, usize); 43],
    owner: Arc<()>,
}

impl G1Runner {
    /// Register the already implemented finite motor-row observer. This does
    /// not substitute cached GenericJoint motor impulses, whose internal
    /// multibody writeback is unavailable, or external PD reference efforts.
    pub fn start_motor_effort_window(&mut self) -> Result<G1MotorEffortWindow, RobotError> {
        if self.actuator_backend != G1ActuatorBackend::NativeForceBased
            || self.halted
            || self.simulation.configuration().physics_hz != 50
            || self.simulation.configuration().dt.to_bits() != 0.02_f32.to_bits()
        {
            return Err(error(
                "motor effort window requires live original50Hz native motors",
            ));
        }
        let mapping = self
            .assembly
            .driven_motor_trace_pairs()
            .ok_or_else(|| error("motor effort window requires43 generation-bound joints"))?;
        if !self
            .simulation
            .world
            .multibody_joints
            .sim2sim_start_motor_row_trace(mapping[0].0)
        {
            return Err(error(
                "motor effort window already started or unsupported owner",
            ));
        }
        Ok(G1MotorEffortWindow {
            episode_id: self.episode_id,
            start_tick: self.simulation.integration_count,
            mapping,
            owner: self.motor_effort_owner.clone(),
        })
    }

    /// Copy this completed boundary's actual biased/integrated/unbiased rows.
    /// Returns an error for expired, repeated-owner, stale or incomplete data.
    /// Reading it cannot integrate or advance an inference history.
    pub fn motor_effort_window_snapshot(
        &self,
        token: &G1MotorEffortWindow,
    ) -> Result<Value, RobotError> {
        let elapsed = self
            .simulation
            .integration_count
            .checked_sub(token.start_tick)
            .filter(|tick| (1..=150).contains(tick))
            .ok_or_else(|| error("motor effort window has no current bounded Tick"))?;
        if !Arc::ptr_eq(&self.motor_effort_owner, &token.owner)
            || self.episode_id != token.episode_id
            || self.assembly.driven_motor_trace_pairs() != Some(token.mapping)
        {
            return Err(error(
                "motor effort window token belongs to a different owner",
            ));
        }
        let (tree, _) = self
            .simulation
            .world
            .multibody_joints
            .get(token.mapping[0].0)
            .ok_or_else(|| error("motor effort window has a stale articulation"))?;
        let observation = tree
            .sim2sim_motor_trace_observation()
            .ok_or_else(|| error("motor effort window has no actual solver receipt"))?;
        let trace = &observation.motor_row_trace;
        let aggregate = &trace.aggregate;
        if !trace.active
            || trace.session_tick as u64 != elapsed
            || trace.dt.to_bits() != 0.02_f32.to_bits()
            || aggregate.rows_per_phase != [43; 3]
            || aggregate.cap_violation
            || aggregate.nonfinite
            || aggregate.overflow
            || aggregate.ownership_incomplete
            || aggregate.temporal_step_invalid
        {
            return Err(error("motor effort window current solver coverage invalid"));
        }
        let mut rows = Vec::with_capacity(trace.raw_rows.len());
        let mut coverage = [[false; 43]; 3];
        for row in &trace.raw_rows {
            let index = token
                .mapping
                .iter()
                .position(|(_, slot)| *slot == row.backend_dof)
                .ok_or_else(|| error("motor effort row has no source joint identity"))?;
            let phase = match row.phase {
                rapier3d::dynamics::MotorRowTracePhase::AfterBiasedSolve => 0,
                rapier3d::dynamics::MotorRowTracePhase::AfterPositionIntegration => 1,
                rapier3d::dynamics::MotorRowTracePhase::AfterUnbiasedSolve => 2,
            };
            let parameters = actuator::parameters(index)?;
            if row.substep_id != 0
                || row.joint_local_dof != 0
                || !row.force_based
                || coverage[phase][index]
                || row.stiffness.to_bits() != parameters.stiffness.to_bits()
                || row.damping.to_bits() != parameters.damping.to_bits()
                || row.max_force.to_bits() != parameters.effort_limit.to_bits()
            {
                return Err(error(
                    "motor effort row changed source gains, mapping or temporal step",
                ));
            }
            coverage[phase][index] = true;
            rows.push(json!({"joint_name":JOINT_NAMES[index], "wbc_index":index,
                "phase":row.phase.as_str(), "backend_dof":row.backend_dof,
                "substep_id":row.substep_id, "coordinate_rad":row.coordinate,
                "generalized_velocity_rad_s":row.generalized_velocity,
                "target_position_rad":row.target_position,
                "actual_motor_impulse_Nm_s":row.impulse,
                "actual_impulse_bounds_Nm_s":row.impulse_bounds,
                "equivalent_boundary_mean_torque_Nm":row.equivalent_mean_torque,
                "effort_cap_Nm":row.max_force, "stiffness":row.stiffness,
                "damping":row.damping, "rhs":row.rhs, "rhs_without_bias":row.rhs_without_bias}));
        }
        if elapsed <= 20 && !coverage.iter().flatten().all(|covered| *covered)
            || elapsed > 20 && !rows.is_empty()
        {
            return Err(error("motor effort raw evidence coverage changed"));
        }
        Ok(
            json!({"schema":"g1_native_motor_effort_window_v1", "episode_id":self.episode_id,
            "source_tick":self.simulation.integration_count, "start_tick":token.start_tick,
            "session_tick":trace.session_tick, "dt":trace.dt, "raw_rows":rows,
            "raw_rows_available":elapsed<=20, "actual_rows_per_phase":aggregate.rows_per_phase,
            "maximum_absolute_motor_impulse_Nm_s":aggregate.max_abs_impulse,
            "maximum_absolute_boundary_mean_torque_Nm":aggregate.max_abs_equivalent_mean_torque,
            "ordinary_observation_valid":observation.valid,
            "ordinary_contact_coverage":observation.contact_coverage,
            "energy_guard_evaluated":observation.energy_guard_evaluated,
            "energy_guard_fallback":observation.energy_guard_fallback,
            "diagnostic_only":true, "model_or_controller_input":false,
            "continuous_torque_waveform_proven":false, "task_qualified":false}),
        )
    }
}
