//! Goose policy boundaries, deliberately independent of MicroDuck dimensions.

use serde::{Deserialize, Serialize};

use crate::RobotError;

pub const GOOSE_ACTION_DIMENSION: usize = 18;
pub const GOOSE_OBSERVATION_DIMENSION: usize = 65;
pub const GOOSE_PICKUP_OBSERVATION_DIMENSION: usize = 82;
pub const GOOSE_HZ: u32 = 50;
pub const GOOSE_DT: f64 = 0.02;
pub const GOOSE_JOINT_ORDER: [&str; GOOSE_ACTION_DIMENSION] = [
    "neck_yaw",
    "neck_pitch",
    "neck_mid_pitch",
    "head_pitch",
    "head_roll",
    "beak_hinge",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
];

/// The unchanged active-axis meaning plus an explicitly derived timing version.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseControlContract {
    pub schema: String,
    pub candidate: String,
    pub robot: String,
    pub joint_order: Vec<String>,
    pub joints: Vec<GooseJointControl>,
    pub physics_dt_s: f64,
    pub torque_dt_s: f64,
    pub policy_dt_s: f64,
    pub observation_size: usize,
    pub action_size: usize,
    pub positive_mechanical_power_limit_w: f64,
    pub phase_frequency_hz: f64,
    pub model_sha256: String,
}

/// Axis limits are SI values from the same neutral mechanical contract.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseJointControl {
    pub name: String,
    pub q_neutral_rad: f64,
    pub action_scale_rad: f64,
    pub range_rad: [f64; 2],
    pub kp_nm_rad: f64,
    pub kd_nm_s_rad: f64,
    pub torque_peak_limit_nm: f64,
    pub continuous_design_limit_nm: f64,
    pub speed_limit_rad_s: f64,
    pub actuation_joint: Option<String>,
}

impl GooseControlContract {
    /// Reject the delivered high-rate contract rather than quietly retiming it.
    pub fn validate(&self) -> Result<(), RobotError> {
        let fail = |message: &str| RobotError::Contract(message.to_owned());
        let identity_valid = matches!(
            (self.schema.as_str(), self.candidate.as_str()),
            (
                "goose_50hz_candidate_si_v1",
                "goose_460_full50_v1" | "goose_460_condensed50_v1"
            ) | ("goose_50hz_experimental_si_v2", "goose_460_full50_be_v2")
        );
        if self.robot != "Goose_V0.1"
            || !identity_valid
            || self.action_size != GOOSE_ACTION_DIMENSION
            || self.observation_size != GOOSE_OBSERVATION_DIMENSION
            || [self.physics_dt_s, self.torque_dt_s, self.policy_dt_s]
                .iter()
                .any(|dt| *dt != GOOSE_DT)
        {
            return Err(fail("Goose candidate, 65/18 or single-step 50 Hz mismatch"));
        }
        if self.joints.len() != GOOSE_ACTION_DIMENSION
            || self
                .joint_order
                .iter()
                .map(String::as_str)
                .ne(GOOSE_JOINT_ORDER)
            || self
                .joints
                .iter()
                .map(|j| j.name.as_str())
                .ne(GOOSE_JOINT_ORDER)
        {
            return Err(fail("Goose active-axis order mismatch"));
        }
        if self.model_sha256.len() != 64
            || !self
                .model_sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(fail("Goose model SHA256 must be lowercase hexadecimal"));
        }
        if !self.positive_mechanical_power_limit_w.is_finite()
            || self.positive_mechanical_power_limit_w <= 0.0
            || !self.phase_frequency_hz.is_finite()
            || self.phase_frequency_hz <= 0.0
        {
            return Err(fail("Invalid Goose power or phase frequency"));
        }
        for joint in &self.joints {
            let values = [
                joint.q_neutral_rad,
                joint.action_scale_rad,
                joint.range_rad[0],
                joint.range_rad[1],
                joint.kp_nm_rad,
                joint.kd_nm_s_rad,
                joint.torque_peak_limit_nm,
                joint.continuous_design_limit_nm,
                joint.speed_limit_rad_s,
            ];
            finite(&values)?;
            if joint.range_rad[0] >= joint.range_rad[1]
                || !(joint.range_rad[0]..=joint.range_rad[1]).contains(&joint.q_neutral_rad)
                || joint.action_scale_rad <= 0.0
                || joint.kp_nm_rad < 0.0
                || joint.kd_nm_s_rad < 0.0
                || joint.continuous_design_limit_nm <= 0.0
                || joint.torque_peak_limit_nm < joint.continuous_design_limit_nm
                || joint.speed_limit_rad_s <= 0.0
            {
                return Err(fail("Invalid Goose axis range or effort limits"));
            }
        }
        if self.joints[5].actuation_joint.as_deref() != Some("beak_input_rotor") {
            return Err(fail("Goose jaw action must drive the actual input rotor"));
        }
        Ok(())
    }

    /// The 65-field layout is kept in the original SI ordering and scales.
    pub fn observation(
        &self,
        state: &GooseNativeState,
        command: [f64; 3],
        previous_action: [f64; 18],
        phase: f64,
    ) -> Result<[f32; 65], RobotError> {
        self.validate()?;
        finite(&command)?;
        finite(&previous_action)?;
        finite(&state.gyro_body_rad_s)?;
        finite(&state.projected_gravity)?;
        finite(&state.joint_position_rad)?;
        finite(&state.joint_velocity_rad_s)?;
        finite(&[phase])?;
        let mut observation = [0.0_f32; 65];
        for i in 0..3 {
            observation[i] = clip(state.gyro_body_rad_s[i] * 0.25);
            observation[3 + i] = clip(state.projected_gravity[i]);
            observation[6 + i] = clip(command[i]);
        }
        for i in 0..18 {
            observation[9 + i] = clip(state.joint_position_rad[i] - self.joints[i].q_neutral_rad);
            observation[27 + i] = clip(state.joint_velocity_rad_s[i] * 0.1);
            observation[45 + i] = clip(previous_action[i].clamp(-1.0, 1.0));
        }
        observation[63] = phase.sin() as f32;
        observation[64] = phase.cos() as f32;
        Ok(observation)
    }

    /// Goals come from the selected game object, never privileged hidden mass.
    pub fn pickup_observation(
        &self,
        base: [f32; 65],
        state: &GooseNativeState,
        goal: &TaskGoal,
        stage: PickupStage,
    ) -> Result<[f32; 82], RobotError> {
        self.validate()?;
        if !base.iter().all(|x| x.is_finite()) {
            return Err(RobotError::NonFinite("Goose base observation"));
        }
        let mut observation = [0.0; 82];
        observation[..65].copy_from_slice(&base);
        observation[75 + stage.index()] = 1.0;
        if !goal.valid {
            return Ok(observation);
        }
        if goal.object_id.is_empty() {
            return Err(RobotError::Contract(
                "Valid Goose goal needs an object identity".into(),
            ));
        }
        finite(&state.root_position_world_m)?;
        finite(&goal.position_world_m)?;
        finite(&goal.placement_world_m)?;
        let root = unit_quaternion(state.root_rotation_world_wxyz)?;
        let object = unit_quaternion(goal.rotation_world_wxyz)?;
        let inverse = [root[0], -root[1], -root[2], -root[3]];
        let relative_object = rotate(
            inverse,
            std::array::from_fn(|i| goal.position_world_m[i] - state.root_position_world_m[i]),
        );
        let relative_place = rotate(
            inverse,
            std::array::from_fn(|i| goal.placement_world_m[i] - state.root_position_world_m[i]),
        );
        let mut orientation = multiply(inverse, object);
        // Resolve quaternion sign deterministically, including exactly 180 deg.
        if orientation
            .iter()
            .find(|value| value.abs() > 1e-12)
            .is_some_and(|value| *value < 0.0)
        {
            orientation = orientation.map(|x| -x);
        }
        for i in 0..3 {
            observation[65 + i] = clip(relative_object[i]);
            observation[72 + i] = clip(relative_place[i]);
        }
        for i in 0..4 {
            observation[68 + i] = orientation[i] as f32;
        }
        observation[81] = 1.0;
        Ok(observation)
    }
}

/// All robot values are at the same physical boundary, in the neutral SI frame.
#[derive(Clone, Debug)]
pub struct GooseNativeState {
    pub gyro_body_rad_s: [f64; 3],
    pub projected_gravity: [f64; 3],
    pub joint_position_rad: [f64; 18],
    pub joint_velocity_rad_s: [f64; 18],
    pub root_position_world_m: [f64; 3],
    pub root_rotation_world_wxyz: [f64; 4],
}

/// The explicit selected-object interface, independent of a vision provider.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskGoal {
    pub object_id: String,
    pub position_world_m: [f64; 3],
    pub rotation_world_wxyz: [f64; 4],
    pub placement_world_m: [f64; 3],
    pub valid: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum PickupStage {
    Approach,
    BendAlign,
    Close,
    Lift,
    Carry,
    Place,
}

impl PickupStage {
    pub const fn index(self) -> usize {
        match self {
            Self::Approach => 0,
            Self::BendAlign => 1,
            Self::Close => 2,
            Self::Lift => 3,
            Self::Carry => 4,
            Self::Place => 5,
        }
    }
}

/// Persistent target/thermal/action state is retained across skill switches.
#[derive(Clone, Debug)]
pub struct GooseActuatorState {
    pub target_rad: [f64; 18],
    pub squared_torque_ewma: [f64; 18],
    pub previous_action: [f64; 18],
    pub control_count: u64,
}

/// An effort result; the builder applies beak effort to the input rotor.
#[derive(Clone, Debug)]
pub struct GooseEffort {
    pub torque_nm: [f64; 18],
    pub positive_mechanical_power_w: f64,
    pub saturated_axes: usize,
}

impl GooseActuatorState {
    pub fn new(contract: &GooseControlContract) -> Result<Self, RobotError> {
        contract.validate()?;
        Ok(Self {
            target_rad: std::array::from_fn(|i| contract.joints[i].q_neutral_rad),
            squared_torque_ewma: [0.0; 18],
            previous_action: [0.0; 18],
            control_count: 0,
        })
    }

    /// Exactly one 20 ms drive update, matching the delivered torque law.
    ///
    /// Nominal neck gravity compensation is supplied from encoder/IMU values;
    /// it must not be calculated from randomized simulator truth.
    pub fn update(
        &mut self,
        contract: &GooseControlContract,
        action: [f64; 18],
        q: [f64; 18],
        qd: [f64; 18],
        nominal_neck_gravity_nm: [f64; 5],
        strength: f64,
        delay_one_policy_step: bool,
    ) -> Result<GooseEffort, RobotError> {
        contract.validate()?;
        finite(&action)?;
        finite(&q)?;
        finite(&qd)?;
        finite(&nominal_neck_gravity_nm)?;
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) || strength == 0.0 {
            return Err(RobotError::Contract(
                "Invalid Goose strength multiplier".into(),
            ));
        }
        let action = action.map(|x| x.clamp(-1.0, 1.0));
        let command = if delay_one_policy_step {
            self.previous_action
        } else {
            action
        };
        let mut torque = [0.0; 18];
        let mut saturated = 0;
        for i in 0..18 {
            let joint = &contract.joints[i];
            let desired = (joint.q_neutral_rad + joint.action_scale_rad * command[i])
                .clamp(joint.range_rad[0], joint.range_rad[1]);
            let slew = joint.speed_limit_rad_s * GOOSE_DT;
            self.target_rad[i] += (desired - self.target_rad[i]).clamp(-slew, slew);
            let requested = joint.kp_nm_rad * (self.target_rad[i] - q[i])
                - joint.kd_nm_s_rad * qd[i]
                + if i < 5 {
                    nominal_neck_gravity_nm[i]
                } else {
                    0.0
                };
            let peak = joint.torque_peak_limit_nm * strength;
            let continuous = joint.continuous_design_limit_nm * strength;
            let speed_cap =
                peak * (1.0 - qd[i].abs() / (joint.speed_limit_rad_s * 1.3)).clamp(0.0, 1.0);
            let thermal_cap = if self.squared_torque_ewma[i] > continuous * continuous {
                continuous
            } else {
                peak
            };
            let cap = speed_cap.min(thermal_cap);
            saturated += usize::from(requested.abs() > cap);
            torque[i] = requested.clamp(-cap, cap);
        }
        let positive_power = (0..18).map(|i| (torque[i] * qd[i]).max(0.0)).sum::<f64>();
        if positive_power > contract.positive_mechanical_power_limit_w {
            let fraction = contract.positive_mechanical_power_limit_w / positive_power;
            for i in 0..18 {
                if torque[i] * qd[i] > 0.0 {
                    torque[i] *= fraction;
                }
            }
        }
        for i in 0..18 {
            self.squared_torque_ewma[i] +=
                GOOSE_DT / 2.0 * (torque[i] * torque[i] - self.squared_torque_ewma[i]);
        }
        self.previous_action = action;
        self.control_count += 1;
        Ok(GooseEffort {
            torque_nm: torque,
            positive_mechanical_power_w: (0..18).map(|i| (torque[i] * qd[i]).max(0.0)).sum(),
            saturated_axes: saturated,
        })
    }
}

fn finite(values: &[f64]) -> Result<(), RobotError> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(RobotError::NonFinite("Goose SI state"))
    }
}
fn clip(value: f64) -> f32 {
    value.clamp(-20.0, 20.0) as f32
}
fn unit_quaternion(q: [f64; 4]) -> Result<[f64; 4], RobotError> {
    finite(&q)?;
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return Err(RobotError::Contract("Zero Goose quaternion".into()));
    }
    Ok(q.map(|x| x / norm))
}
fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let result = multiply(
        multiply(q, [0.0, v[0], v[1], v[2]]),
        [q[0], -q[1], -q[2], -q[3]],
    );
    [result[1], result[2], result[3]]
}

#[cfg(test)]
mod tests {
    use super::*;
    fn contract() -> GooseControlContract {
        GooseControlContract {
            schema: "goose_50hz_candidate_si_v1".into(),
            candidate: "goose_460_full50_v1".into(),
            robot: "Goose_V0.1".into(),
            joint_order: GOOSE_JOINT_ORDER.map(str::to_owned).to_vec(),
            joints: GOOSE_JOINT_ORDER
                .iter()
                .enumerate()
                .map(|(i, name)| GooseJointControl {
                    name: (*name).into(),
                    q_neutral_rad: 0.0,
                    action_scale_rad: 1.0,
                    range_rad: [-1.0, 1.0],
                    kp_nm_rad: 100.0,
                    kd_nm_s_rad: 1.0,
                    torque_peak_limit_nm: 12.0,
                    continuous_design_limit_nm: 6.4,
                    speed_limit_rad_s: 1.0,
                    actuation_joint: (i == 5).then(|| "beak_input_rotor".into()),
                })
                .collect(),
            physics_dt_s: GOOSE_DT,
            torque_dt_s: GOOSE_DT,
            policy_dt_s: GOOSE_DT,
            observation_size: 65,
            action_size: 18,
            positive_mechanical_power_limit_w: 350.0,
            phase_frequency_hz: 1.2,
            model_sha256: "a".repeat(64),
        }
    }

    #[test]
    fn experimental_candidate_keeps_axis_contract_and_requires_its_schema_pair() {
        let mut candidate = contract();
        candidate.candidate = "goose_460_full50_be_v2".into();
        assert!(candidate.validate().is_err());
        candidate.schema = "goose_50hz_experimental_si_v2".into();
        assert!(candidate.validate().is_ok());
        candidate.candidate = "goose_460_full50_v1".into();
        assert!(candidate.validate().is_err());
    }
    fn state() -> GooseNativeState {
        GooseNativeState {
            gyro_body_rad_s: [4.0, 8.0, 12.0],
            projected_gravity: [0.0, 0.0, -1.0],
            joint_position_rad: [0.0; 18],
            joint_velocity_rad_s: [2.0; 18],
            root_position_world_m: [2.0, 3.0, 0.0],
            root_rotation_world_wxyz: [1.0, 0.0, 0.0, 0.0],
        }
    }
    #[test]
    fn rejects_old_timing_and_wrong_motor_mapping() {
        let mut c = contract();
        c.physics_dt_s = 0.0001;
        assert!(c.validate().is_err());
        c.physics_dt_s = GOOSE_DT;
        c.joints[5].actuation_joint = None;
        assert!(c.validate().is_err());
    }
    #[test]
    fn golden_observation_and_relative_selected_goal() {
        let c = contract();
        let s = state();
        let obs = c
            .observation(&s, [0.3, -0.1, 0.6], [0.25; 18], 0.0)
            .unwrap();
        assert_eq!(&obs[..9], &[1.0, 2.0, 3.0, 0.0, 0.0, -1.0, 0.3, -0.1, 0.6]);
        assert!(obs[27..45].iter().all(|v| *v == 0.2));
        assert_eq!(&obs[63..], &[0.0, 1.0]);
        let goal = TaskGoal {
            object_id: "cup".into(),
            position_world_m: [3.0, 5.0, 0.0],
            rotation_world_wxyz: [-1.0, 0.0, 0.0, 0.0],
            placement_world_m: [4.0, 3.0, 0.0],
            valid: true,
        };
        let pickup = c
            .pickup_observation(obs, &s, &goal, PickupStage::Lift)
            .unwrap();
        assert_eq!(
            &pickup[65..75],
            &[1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(pickup[78], 1.0);
        assert_eq!(pickup[81], 1.0);
    }
    #[test]
    fn drive_updates_once_with_twenty_ms_slew_and_delay() {
        let c = contract();
        let mut drive = GooseActuatorState::new(&c).unwrap();
        let first = drive
            .update(&c, [1.0; 18], [0.0; 18], [0.0; 18], [0.0; 5], 1.0, true)
            .unwrap();
        assert_eq!(first.torque_nm, [0.0; 18]);
        let second = drive
            .update(&c, [1.0; 18], [0.0; 18], [0.0; 18], [0.0; 5], 1.0, true)
            .unwrap();
        assert_eq!(second.torque_nm, [2.0; 18]);
        assert_eq!(drive.target_rad, [0.02; 18]);
        assert_eq!(drive.control_count, 2);
        assert!((drive.squared_torque_ewma[0] - 0.04).abs() < 1e-12);
    }
    #[test]
    fn thermal_speed_power_caps_and_invalid_action_are_enforced() {
        let mut c = contract();
        c.positive_mechanical_power_limit_w = 5.0;
        let mut drive = GooseActuatorState::new(&c).unwrap();
        drive.target_rad = [1.0; 18];
        let efforts = drive
            .update(&c, [1.0; 18], [0.0; 18], [0.2; 18], [0.0; 5], 1.0, false)
            .unwrap();
        assert!((efforts.positive_mechanical_power_w - 5.0).abs() < 1e-12);
        let stopped = drive
            .update(&c, [1.0; 18], [0.0; 18], [2.0; 18], [0.0; 5], 1.0, false)
            .unwrap();
        assert_eq!(stopped.torque_nm, [0.0; 18]);
        assert!(
            drive
                .update(
                    &c,
                    [f64::NAN; 18],
                    [0.0; 18],
                    [0.0; 18],
                    [0.0; 5],
                    1.0,
                    false
                )
                .is_err()
        );
    }
}
