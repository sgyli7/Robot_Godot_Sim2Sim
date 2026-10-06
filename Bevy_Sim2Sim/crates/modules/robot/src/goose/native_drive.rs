//! Signed effort limits and history around upstream native position motors.
//!
//! Rapier ForceBased and MuJoCo implicitfast are different numerical drivers.
//! This target revision preserves the source SI limits, not source qualification.

use super::{
    contract::{GOOSE_DT, GooseControlContract, GooseNativeState},
    gravity::GooseNominalGravity,
};
use crate::RobotError;
use serde::Serialize;

pub const GOOSE_RAPIER_DRIVE_REVISION: &str = "goose_rapier_force_based_drive_v1";

/// One queued update. Axis 5 is a true rotor effort; the other 17 are motors.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GooseNativeMotorCommand {
    pub update: u64,
    pub target_rad: [f64; 18],
    pub motor_target_rad: [f64; 18],
    pub symmetric_limit_nm: [f64; 18],
    pub external_effort_nm: [f64; 18],
    pub lower_limit_nm: [f64; 18],
    pub upper_limit_nm: [f64; 18],
    pub pre_tick_motor_velocity_rad_s: [f64; 18],
    pub motoring_power_bound_w: f64,
}

/// A cold instance starts with source neutral targets and zero action/heat/phase.
pub struct GooseNativeDrive {
    contract: GooseControlContract,
    gravity: GooseNominalGravity,
    strength: f64,
    delayed: bool,
    target_rad: [f64; 18],
    previous_action: [f64; 18],
    squared_torque_ewma: [f64; 18],
    phase: f64,
    control_count: u64,
    pending: Option<(GooseNativeMotorCommand, [f64; 18])>,
}

impl GooseNativeDrive {
    pub fn new(
        contract: GooseControlContract,
        gravity: GooseNominalGravity,
        strength: f64,
        delayed: bool,
    ) -> Result<Self, RobotError> {
        contract.validate()?;
        if contract.candidate != "goose_task_proxy_11_rigid_braking_v1"
            || contract.joints.iter().any(|j| j.kp_nm_rad <= 0.0)
            || !strength.is_finite()
            || strength <= 0.0
            || strength > 1.0
        {
            return Err(invalid(
                "Native target drive requires the frozen braking source and positive stiffness/strength",
            ));
        }
        let target_rad = std::array::from_fn(|i| contract.joints[i].q_neutral_rad);
        Ok(Self {
            contract,
            gravity,
            strength,
            delayed,
            target_rad,
            previous_action: [0.0; 18],
            squared_torque_ewma: [0.0; 18],
            phase: 0.0,
            control_count: 0,
            pending: None,
        })
    }

    /// Motor speed uses the input rotor on axis 5; Actor feedback still uses jaw.
    pub fn prepare(
        &mut self,
        state: &GooseNativeState,
        motor_velocity: [f64; 18],
        action: [f64; 18],
    ) -> Result<GooseNativeMotorCommand, RobotError> {
        finite(&state.joint_position_rad)?;
        finite(&state.joint_velocity_rad_s)?;
        finite(&motor_velocity)?;
        finite(&action)?;
        let neck = std::array::from_fn(|i| state.joint_position_rad[i]);
        let feed = self
            .gravity
            .torque_nm(neck, state.root_rotation_world_wxyz)?;
        self.prepare_with_feed(state, motor_velocity, action, feed)
    }

    fn prepare_with_feed(
        &mut self,
        state: &GooseNativeState,
        motor_velocity: [f64; 18],
        action: [f64; 18],
        feed: [f64; 5],
    ) -> Result<GooseNativeMotorCommand, RobotError> {
        if self.pending.is_some() {
            return Err(invalid(
                "Previous native drive update has not completed a real step",
            ));
        }
        let action = action.map(|v| v.clamp(-1.0, 1.0));
        let delayed_action = if self.delayed {
            self.previous_action
        } else {
            action
        };
        let mut target_rad = self.target_rad;
        let mut peak = [0.0; 18];
        let mut motoring = [0.0; 18];
        for i in 0..18 {
            let j = &self.contract.joints[i];
            let desired = (j.q_neutral_rad + j.action_scale_rad * delayed_action[i])
                .clamp(j.range_rad[0], j.range_rad[1]);
            target_rad[i] += (desired - target_rad[i]).clamp(
                -j.speed_limit_rad_s * GOOSE_DT,
                j.speed_limit_rad_s * GOOSE_DT,
            );
            let continuous = j.continuous_design_limit_nm * self.strength;
            peak[i] = if self.squared_torque_ewma[i] > continuous * continuous {
                continuous
            } else {
                j.torque_peak_limit_nm * self.strength
            };
            motoring[i] = peak[i]
                * (1.0 - motor_velocity[i].abs() / (j.speed_limit_rad_s * 1.3)).clamp(0.0, 1.0);
        }
        let bound = (0..18)
            .map(|i| motoring[i] * motor_velocity[i].abs())
            .sum::<f64>();
        let power_scale =
            (self.contract.positive_mechanical_power_limit_w / bound.max(1e-12)).min(1.0);
        let mut packet = GooseNativeMotorCommand {
            update: self.control_count + 1,
            target_rad,
            motor_target_rad: target_rad,
            symmetric_limit_nm: [0.0; 18],
            external_effort_nm: [0.0; 18],
            lower_limit_nm: [0.0; 18],
            upper_limit_nm: [0.0; 18],
            pre_tick_motor_velocity_rad_s: motor_velocity,
            motoring_power_bound_w: bound * power_scale,
        };
        for i in 0..18 {
            let positive = motoring[i] * power_scale;
            let lower = -if motor_velocity[i] < 0.0 {
                positive
            } else {
                peak[i]
            };
            let upper = if motor_velocity[i] > 0.0 {
                positive
            } else {
                peak[i]
            };
            packet.lower_limit_nm[i] = lower;
            packet.upper_limit_nm[i] = upper;
            if i == 5 {
                let j = &self.contract.joints[i];
                packet.external_effort_nm[i] = (j.kp_nm_rad
                    * (target_rad[i] - state.joint_position_rad[i])
                    - j.kd_nm_s_rad * state.joint_velocity_rad_s[i])
                    .clamp(lower, upper);
            } else {
                // b + clamp(PD + feed - b, -r, r) == clamp(PD + feed, lower, upper).
                let bias = (upper + lower) * 0.5;
                packet.external_effort_nm[i] = bias;
                packet.symmetric_limit_nm[i] = (upper - lower) * 0.5;
                let gravity = if i < 5 { feed[i] } else { 0.0 };
                packet.motor_target_rad[i] += (gravity - bias) / self.contract.joints[i].kp_nm_rad;
            }
        }
        self.pending = Some((packet.clone(), action));
        Ok(packet)
    }

    /// Complete only after one real step, using solved motor impulse/dt plus bias.
    pub fn finish(
        &mut self,
        packet: &GooseNativeMotorCommand,
        actual_torque_nm: [f64; 18],
    ) -> Result<(), RobotError> {
        finite(&actual_torque_nm)?;
        let Some((expected, action)) = &self.pending else {
            return Err(invalid("No pending native motor step"));
        };
        if expected != packet
            || (0..18).any(|i| {
                actual_torque_nm[i] < packet.lower_limit_nm[i] - 2e-4
                    || actual_torque_nm[i] > packet.upper_limit_nm[i] + 2e-4
            })
        {
            return Err(invalid(
                "Native motor ticket or actual signed effort bound mismatch",
            ));
        }
        for (heat, torque) in self.squared_torque_ewma.iter_mut().zip(actual_torque_nm) {
            *heat += GOOSE_DT / 2.0 * (torque * torque - *heat);
        }
        self.target_rad = packet.target_rad;
        self.previous_action = *action;
        self.phase = (self.phase
            + std::f64::consts::TAU * self.contract.phase_frequency_hz * GOOSE_DT)
            % std::f64::consts::TAU;
        self.control_count += 1;
        self.pending = None;
        Ok(())
    }
    pub fn contract(&self) -> &GooseControlContract {
        &self.contract
    }
    pub fn previous_action(&self) -> [f64; 18] {
        self.previous_action
    }
    pub fn phase(&self) -> f64 {
        self.phase
    }
    pub fn control_count(&self) -> u64 {
        self.control_count
    }
    pub fn squared_torque_ewma(&self) -> [f64; 18] {
        self.squared_torque_ewma
    }
}

fn finite(values: &[f64]) -> Result<(), RobotError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(RobotError::NonFinite("Goose native drive"))
    }
}
fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}
