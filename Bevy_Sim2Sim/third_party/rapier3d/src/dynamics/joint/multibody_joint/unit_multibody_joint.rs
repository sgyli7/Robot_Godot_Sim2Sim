#![allow(missing_docs)] // For downcast.

use crate::dynamics::integration_parameters::SpringCoefficients;
use crate::dynamics::joint::MultibodyLink;
use crate::dynamics::solver::{GenericJointConstraint, WritebackId};
use crate::dynamics::{IntegrationParameters, JointMotor, Multibody};
use crate::math::{DVector, Real};

/// Parameters from the frozen compiled source model, present only in a
/// manually selected, single-axis experiment. The source's current frozen
/// impedance uses midpoint 0.5 and power 2; other curves are refused.
#[cfg(feature = "sim2sim-source-limit-probe")]
#[derive(Copy, Clone, Debug)]
pub(crate) struct SourceLimitProbe {
    pub solref: [Real; 2],
    pub solimp: [Real; 5],
    pub margin: Real,
    pub dof_invweight0: Real,
}

/// Diagnostic counterpart to the source's one active unilateral limit row.
/// Rapier's internal row uses +e_dof with positive impulse opposing an upper
/// violation. The source row uses -e_dof with positive force. A velocity
/// impulse is dt times source force, with R in the row denominator as CFM.
/// Keeping rhs_wo_bias equal to rhs prevents Rapier's later stabilization
/// pass from cancelling the physical soft-limit impulse after integration.
#[cfg(feature = "sim2sim-source-limit-probe")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn unit_joint_source_limit_probe_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    limits: [Real; 2],
    curr_pos: Real,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
    probe: &SourceLimitProbe,
) {
    let ndofs = multibody.ndofs();
    let backend_dof = dof_id + link.assembly_id;
    let initial_velocity = multibody.generalized_velocity()[backend_dof];
    let upper_depth = curr_pos - limits[1] + probe.margin;
    let lower_depth = limits[0] - curr_pos + probe.margin;
    let (direction, depth, impulse_bounds) = if upper_depth > 0.0 {
        (1.0, upper_depth, [0.0, Real::MAX])
    } else if lower_depth > 0.0 {
        (-1.0, lower_depth, [-Real::MAX, 0.0])
    } else {
        (0.0, 0.0, [0.0, 0.0])
    };

    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    let dof_j_id = *j_id + backend_dof;
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));
    let lhs = jacobians[dof_j_id + ndofs];

    // This interpolation is the frozen source's (midpoint=.5, power=2)
    // quadratic solimp curve. refsafe is enabled in the frozen source model.
    let x = (depth / probe.solimp[2]).clamp(0.0, 1.0);
    let s = if x <= 0.5 {
        2.0 * x * x
    } else {
        1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    };
    let d = probe.solimp[0] + (probe.solimp[1] - probe.solimp[0]) * s;
    let tau = probe.solref[0].max(2.0 * params.dt);
    let b = 2.0 / (probe.solimp[1] * tau);
    let k = d / (probe.solimp[1] * probe.solimp[1] * tau * tau * probe.solref[1] * probe.solref[1]);
    let aref = k * depth + b * direction * initial_velocity;
    let r = (1.0 - d) / d * probe.dof_invweight0;
    // At the beginning of the solve, v = v0 + dt*a_free. This RHS makes the
    // row impulse direction*dt*(aref - J*a_free)/(lhs + R), J=-direction.
    let rhs = if direction != 0.0 {
        params.dt * direction * aref - initial_velocity
    } else {
        0.0
    };

    constraints[*insert_at] = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,
        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX,
        impulse: 0.0,
        impulse_bounds,
        inv_lhs: crate::utils::inv(lhs + r),
        rhs,
        rhs_wo_bias: rhs,
        cfm_coeff: 0.0,
        cfm_gain: r,
        writeback_id: WritebackId::Limit(dof_id),
    };
    *insert_at += 1;
    *j_id += 2 * ndofs;
}

/// Two unilateral rows enforce lower <= q + dt * v <= upper before the
/// existing Euler coordinate integration. Both remain available while q is
/// inside its range, so a motor/contact cannot cross a bound before activation.
/// Bounds enter the physical velocity solve, including its stabilization pass;
/// no coordinate projection, pose write, narrowed range or extra step is used.
#[cfg(feature = "g1-predictive-limits")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn unit_joint_predictive_limit_constraints(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    limits: [Real; 2],
    curr_pos: Real,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
) {
    let ndofs = multibody.ndofs();
    let backend_dof = dof_id + link.assembly_id;
    for (bound, impulse_bounds) in [
        (limits[0], [-Real::MAX, 0.0]),
        (limits[1], [0.0, Real::MAX]),
    ] {
        let dof_j_id = *j_id + backend_dof;
        jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
        jacobians[dof_j_id] = 1.0;
        jacobians[dof_j_id + ndofs] = 1.0;
        multibody
            .inv_augmented_mass()
            .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));
        let lhs = jacobians[dof_j_id + ndofs];
        let rhs = (curr_pos - bound) / params.dt;
        constraints[*insert_at] = GenericJointConstraint {
            is_rigid_body1: false,
            solver_vel1: u32::MAX,
            ndofs1: 0,
            j_id1: 0,
            is_rigid_body2: false,
            solver_vel2: multibody.solver_id,
            ndofs2: ndofs,
            j_id2: *j_id,
            joint_id: usize::MAX,
            impulse: 0.0,
            impulse_bounds,
            inv_lhs: crate::utils::inv(lhs),
            rhs,
            rhs_wo_bias: rhs,
            cfm_coeff: 0.0,
            cfm_gain: 0.0,
            writeback_id: WritebackId::Limit(dof_id),
        };
        *insert_at += 1;
        *j_id += 2 * ndofs;
    }
}

/// Initializes and generate the velocity constraints applicable to the multibody links attached
/// to this multibody_joint.
pub fn unit_joint_limit_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    limits: [Real; 2],
    curr_pos: Real,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
    softness: SpringCoefficients<Real>,
) {
    let ndofs = multibody.ndofs();
    let min_enabled = curr_pos < limits[0];
    let max_enabled = limits[1] < curr_pos;

    // Compute per-joint ERP and CFM
    let erp_inv_dt = softness.erp_inv_dt(params.dt);
    let cfm_coeff = softness.cfm_coeff(params.dt);

    let rhs_bias = ((curr_pos - limits[1]).max(0.0) - (limits[0] - curr_pos).max(0.0)) * erp_inv_dt;
    let rhs_wo_bias = 0.0;

    let dof_j_id = *j_id + dof_id + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    let lhs = jacobians[dof_j_id + ndofs]; // = J^t * M^-1 J
    let impulse_bounds = [
        min_enabled as u32 as Real * -Real::MAX,
        max_enabled as u32 as Real * Real::MAX,
    ];
    let cfm_gain = lhs * cfm_coeff;

    let constraint = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,

        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX, // TODO: we don’t support impulse writeback for internal constraints yet.
        impulse: 0.0,
        impulse_bounds,
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        cfm_coeff,
        cfm_gain,
        writeback_id: WritebackId::Limit(dof_id),
    };

    constraints[*insert_at] = constraint;
    *insert_at += 1;

    *j_id += 2 * ndofs;
}

/// Generates the dry-friction (MuJoCo `frictionloss`) velocity constraint for
/// one generalized DoF.
///
/// `softness` supplies the CFM compliance (MuJoCo's `solreffriction`); only CFM
/// applies, never ERP, since there is no position error for a bias to chase.
#[allow(clippy::too_many_arguments)]
pub fn unit_joint_friction_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    friction: Real,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
    softness: SpringCoefficients<Real>,
) {
    let ndofs = multibody.ndofs();
    let cfm_coeff = softness.cfm_coeff(params.dt);

    let dof_j_id = *j_id + dof_id + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    let lhs = jacobians[dof_j_id + ndofs]; // = J^t * M^-1 J
    let max_impulse = friction * params.dt;
    let cfm_gain = lhs * cfm_coeff;

    let constraint = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,
        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX, // TODO: we don’t support impulse writeback for internal constraints yet.
        impulse: 0.0,
        impulse_bounds: [-max_impulse, max_impulse],
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: 0.0,
        rhs_wo_bias: 0.0,
        cfm_coeff,
        cfm_gain,
        writeback_id: WritebackId::Friction(dof_id),
    };

    constraints[*insert_at] = constraint;
    *insert_at += 1;

    *j_id += 2 * ndofs;
}

/// Initializes and generate the velocity constraints applicable to the multibody links attached
/// to this multibody_joint.
pub fn unit_joint_motor_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    motor: &JointMotor,
    curr_pos: Real,
    limits: Option<[Real; 2]>,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
) {
    let inv_dt = params.inv_dt();
    let ndofs = multibody.ndofs();
    let motor_params = motor.motor_params(params.dt);

    let dof_j_id = *j_id + dof_id + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    let lhs = jacobians[dof_j_id + ndofs]; // = J^t * M^-1 J
    let impulse_bounds = [-motor_params.max_impulse, motor_params.max_impulse];
    let cfm_gain = lhs * motor_params.cfm_coeff + motor_params.cfm_gain;

    let mut rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        rhs_wo_bias += (curr_pos - motor_params.target_pos) * motor_params.erp_inv_dt;
    }

    let mut target_vel = motor_params.target_vel;
    if let Some(limits) = limits {
        target_vel = target_vel.clamp(
            (limits[0] - curr_pos) * inv_dt,
            (limits[1] - curr_pos) * inv_dt,
        );
    };

    rhs_wo_bias += -target_vel;

    let constraint = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,

        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX, // TODO: we don’t support impulse writeback for internal constraints yet.
        impulse: 0.0,
        impulse_bounds,
        cfm_coeff: motor_params.cfm_coeff,
        cfm_gain,
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        writeback_id: WritebackId::Limit(dof_id),
    };

    constraints[*insert_at] = constraint;
    *insert_at += 1;

    *j_id += 2 * ndofs;
}
