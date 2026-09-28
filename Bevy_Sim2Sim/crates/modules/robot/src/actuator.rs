//! BAM m6 firmware voltage, motor torque and native-friction budgets.
//!
//! Equations follow the pinned Rhoban BAM mjlab implementation. The plant must
//! solve dry friction as a constraint; this module never substitutes tanh drag.

use crate::contract::finite;
use crate::{ACTION_DIMENSION, RobotError};
use serde::{Deserialize, Serialize};

/// Values exported from the actual BAM model, before narrowing scalar operands.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BamParameters {
    pub kt: f64,
    pub resistance: f64,
    pub error_gain: f64,
    pub firmware_kp: f64,
    pub max_pwm: f64,
    pub max_current: Option<f64>,
    pub armature: f64,
    pub friction_base: f64,
    pub friction_stribeck: f64,
    pub friction_viscous: f64,
    pub dtheta_stribeck: f64,
    pub alpha: f64,
    pub load_friction_motor: f64,
    pub load_friction_external: f64,
    pub load_friction_motor_stribeck: f64,
    pub load_friction_external_stribeck: f64,
    pub load_friction_motor_quad: f64,
    pub load_friction_external_quad: f64,
}

impl BamParameters {
    pub fn validate(&self) -> Result<(), RobotError> {
        let values = [
            self.kt,
            self.resistance,
            self.error_gain,
            self.firmware_kp,
            self.max_pwm,
            self.armature,
            self.friction_base,
            self.friction_stribeck,
            self.friction_viscous,
            self.dtheta_stribeck,
            self.alpha,
            self.load_friction_motor,
            self.load_friction_external,
            self.load_friction_motor_stribeck,
            self.load_friction_external_stribeck,
            self.load_friction_motor_quad,
            self.load_friction_external_quad,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > f64::from(f32::MAX))
            || self.resistance <= 0.0
            || self.dtheta_stribeck <= 0.0
            || self.kt <= 0.0
            || self.max_pwm <= 0.0
            || self.max_current.is_some_and(|value| {
                !value.is_finite() || value <= 0.0 || value > f64::from(f32::MAX)
            })
        {
            return Err(RobotError::Contract("invalid BAM parameters".into()));
        }
        Ok(())
    }
}

/// Startup-randomized supply and per-episode gain/friction scales.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BamEnvironment {
    pub supply_voltage: f32,
    pub voltage_drop_gain: f32,
    pub minimum_voltage: f32,
    pub kp_scale: f32,
    pub kd_scale: f32,
    pub friction_scale: f32,
}

/// Two different histories are required by the upstream model.
pub struct BamInput<'a> {
    pub target: &'a [f32; ACTION_DIMENSION],
    pub position: &'a [f32; ACTION_DIMENSION],
    pub velocity: &'a [f32; ACTION_DIMENSION],
    pub previous_motor_computed: &'a [f32; ACTION_DIMENSION],
    pub previous_actuator_applied: &'a [f32; ACTION_DIMENSION],
    /// Previous constraint load excluding our dry friction, minus bias.
    pub external_load: &'a [f32; ACTION_DIMENSION],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BamOutput {
    /// Computed electrical torque, before the plant actuator force range.
    pub motor_torque: [f32; ACTION_DIMENSION],
    /// Nonnegative generalized dry-friction force budget in Nm.
    pub friction_budget: [f32; ACTION_DIMENSION],
    pub viscous_damping: f32,
    pub effective_voltage: f32,
}

pub fn compute(
    parameters: &BamParameters,
    environment: BamEnvironment,
    input: BamInput<'_>,
) -> Result<BamOutput, RobotError> {
    parameters.validate()?;
    for values in [
        input.target,
        input.position,
        input.velocity,
        input.previous_motor_computed,
        input.previous_actuator_applied,
        input.external_load,
    ] {
        finite(values, "BAM input")?;
    }
    finite(
        &[
            environment.supply_voltage,
            environment.voltage_drop_gain,
            environment.minimum_voltage,
            environment.kp_scale,
            environment.kd_scale,
            environment.friction_scale,
        ],
        "BAM environment",
    )?;
    if environment.supply_voltage <= 0.0
        || environment.minimum_voltage <= 0.0
        || environment.voltage_drop_gain < 0.0
        || environment.kp_scale < 0.0
        || environment.kd_scale < 0.0
        || environment.friction_scale < 0.0
    {
        return Err(RobotError::Contract("invalid BAM environment".into()));
    }
    let load: f32 = input
        .previous_motor_computed
        .iter()
        .map(|value| value.abs())
        .sum();
    let voltage = (environment.supply_voltage - environment.voltage_drop_gain * load)
        .max(environment.minimum_voltage);
    let kt = parameters.kt as f32;
    let resistance = parameters.resistance as f32;
    // Upstream computes this scalar power in Python before the tensor operation.
    let kt_squared = (parameters.kt * parameters.kt) as f32;
    let kp = parameters.firmware_kp as f32 * environment.kp_scale;
    let mut result = BamOutput {
        motor_torque: [0.0; ACTION_DIMENSION],
        friction_budget: [0.0; ACTION_DIMENSION],
        viscous_damping: parameters.friction_viscous as f32,
        effective_voltage: voltage,
    };
    for index in 0..ACTION_DIMENSION {
        let velocity = input.velocity[index];
        let scaled_velocity = velocity * environment.kd_scale;
        let mut duty =
            (input.target[index] - input.position[index]) * kp * parameters.error_gain as f32;
        if let Some(max_current) = parameters.max_current {
            let center = kt * scaled_velocity / voltage;
            // R*I is evaluated as a scalar before division by the voltage tensor.
            let span = (parameters.resistance * max_current) as f32 / voltage;
            duty = duty.clamp(center - span, center + span);
        }
        duty = duty.clamp(-(parameters.max_pwm as f32), parameters.max_pwm as f32);
        let control = voltage * duty;
        result.motor_torque[index] =
            kt * control / resistance - kt_squared * scaled_velocity / resistance;
        let stribeck = (-(velocity.abs() / parameters.dtheta_stribeck as f32)
            .powf(parameters.alpha as f32))
        .exp();
        let motor = input.previous_actuator_applied[index];
        let external = input.external_load[index];
        let gearbox = (external * parameters.load_friction_external as f32
            - motor * parameters.load_friction_motor as f32)
            .abs();
        let gearbox_stribeck = (external * parameters.load_friction_external_stribeck as f32
            - motor * parameters.load_friction_motor_stribeck as f32)
            .abs();
        // Match the source tensor operations, including strict tie handling and
        // both masked products. An overflowing inactive branch must be rejected
        // as non-finite, rather than silently changing the upstream arithmetic.
        let drive_mask = if motor.abs() > external.abs() {
            1.0
        } else {
            0.0
        };
        let backdrive_mask = 1.0 - drive_mask;
        let quadratic =
            drive_mask * parameters.load_friction_external_quad as f32 * external.abs().powi(2)
                + backdrive_mask * parameters.load_friction_motor_quad as f32 * motor.abs().powi(2);
        result.friction_budget[index] = (parameters.friction_base as f32
            + stribeck * parameters.friction_stribeck as f32
            + gearbox
            + stribeck * gearbox_stribeck
            + stribeck * quadratic)
            * environment.friction_scale;
    }
    finite(&result.motor_torque, "BAM torque")?;
    finite(&result.friction_budget, "BAM friction")?;
    finite(&[voltage, result.viscous_damping], "BAM passive output")?;
    Ok(result)
}
