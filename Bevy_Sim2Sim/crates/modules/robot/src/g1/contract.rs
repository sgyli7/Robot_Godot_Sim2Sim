//! Arena homie_v2 boundary at revision 7d75c95934c51a0318c957a8831e862ca43c53b5.
//!
//! Joint order follows loco_manip_g1_joints_order_43dof.yaml. The network sees
//! 29 body joints, excluding fingers, and controls the 15 leg/waist joints.
//! Upper-body targets contain the remaining 28 joints in the same named order.

use crate::RobotError;

pub const JOINT_COUNT: usize = 43;
pub const BODY_COUNT: usize = 29;
pub const LOWER_COUNT: usize = 15;
pub const UPPER_COUNT: usize = 28;
pub const FRAME_SIZE: usize = 86;
pub const HISTORY_SIZE: usize = 6;
pub const OBSERVATION_SIZE: usize = FRAME_SIZE * HISTORY_SIZE;
pub const CONTROL_HZ: u32 = 50;
pub const JOINT_NAMES: [&str; JOINT_COUNT] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
];
pub const BODY_INDICES: [usize; BODY_COUNT] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 29, 30, 31, 32,
    33, 34, 35,
];
pub const LOWER_HOME: [f32; LOWER_COUNT] = [
    -0.1, 0., 0., 0.3, -0.2, 0., -0.1, 0., 0., 0.3, -0.2, 0., 0., 0., 0.,
];

/// Physical SI commands in the source Z-up base frame; upper targets are radians.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1Command {
    pub navigation: [f32; 3],
    pub pelvis_height: f32,
    pub torso_rpy: [f32; 3],
    pub upper_positions: [f32; UPPER_COUNT],
}

impl Default for G1Command {
    fn default() -> Self {
        Self {
            navigation: [0.; 3],
            // Arena's action term overrides the YAML's 0.74 with a 0.75 goal.
            pelvis_height: 0.75,
            torso_rpy: [0.; 3],
            upper_positions: [0.; UPPER_COUNT],
        }
    }
}

impl G1Command {
    pub fn validate(&self) -> Result<(), RobotError> {
        finite(&self.navigation)?;
        finite(&[self.pelvis_height])?;
        finite(&self.torso_rpy)?;
        finite(&self.upper_positions)
    }

    /// Source selects stand below a strict Euclidean 0.05 command threshold.
    pub fn standing(&self) -> bool {
        self.navigation
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt()
            < 0.05
    }
}

/// Native joint measurements and root-link sensors, never rendered poses.
#[derive(Clone, Debug)]
pub struct G1State {
    pub positions: [f32; JOINT_COUNT],
    pub velocities: [f32; JOINT_COUNT],
    pub root_rotation_wxyz: [f32; 4],
    pub root_angular_velocity_body: [f32; 3],
}

impl G1State {
    pub fn validate(&self) -> Result<(), RobotError> {
        finite(&self.positions)?;
        finite(&self.velocities)?;
        finite(&self.root_rotation_wxyz)?;
        finite(&self.root_angular_velocity_body)?;
        let norm = self
            .root_rotation_wxyz
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>();
        if (norm - 1.).abs() > 1e-5 {
            return Err(RobotError::Contract(
                "G1 root quaternion is not unit length".into(),
            ));
        }
        Ok(())
    }
}

/// Stateful history is shared across standing/walking policy changes, as upstream.
pub struct HomieHistory {
    observation: [f32; OBSERVATION_SIZE],
    previous_action: [f32; LOWER_COUNT],
}

impl Default for HomieHistory {
    fn default() -> Self {
        Self {
            observation: [0.; OBSERVATION_SIZE],
            previous_action: [0.; LOWER_COUNT],
        }
    }
}

impl HomieHistory {
    /// Append one current frame, keeping startup history left-padded with zeros.
    pub fn observe(
        &mut self,
        state: &G1State,
        command: &G1Command,
    ) -> Result<&[f32; OBSERVATION_SIZE], RobotError> {
        state.validate()?;
        command.validate()?;
        let mut frame = [0.; FRAME_SIZE];
        for i in 0..3 {
            frame[i] = command.navigation[i] * [2., 2., 0.5][i];
            frame[4 + i] = command.torso_rpy[i];
            frame[7 + i] = state.root_angular_velocity_body[i] * 0.5;
        }
        frame[3] = command.pelvis_height;
        let [w, x, y, z] = state.root_rotation_wxyz.map(f64::from);
        // Exact inverse quaternion rotation of source gravity [0,0,-1].
        frame[10] = (2. * (w * y - x * z)) as f32;
        frame[11] = (-2. * (y * z + w * x)) as f32;
        frame[12] = (-w * w + x * x + y * y - z * z) as f32;
        for (i, &joint) in BODY_INDICES.iter().enumerate() {
            let home = LOWER_HOME.get(i).copied().unwrap_or(0.);
            frame[13 + i] = (f64::from(state.positions[joint]) - f64::from(home)) as f32;
            frame[13 + BODY_COUNT + i] = (f64::from(state.velocities[joint]) * 0.05) as f32;
        }
        frame[13 + 2 * BODY_COUNT..].copy_from_slice(&self.previous_action);
        self.observation.copy_within(FRAME_SIZE.., 0);
        self.observation[OBSERVATION_SIZE - FRAME_SIZE..].copy_from_slice(&frame);
        Ok(&self.observation)
    }

    /// Commit only a successful, finite network output. No action clipping exists upstream.
    pub fn accept_action(
        &mut self,
        action: [f32; LOWER_COUNT],
        command: &G1Command,
    ) -> Result<[f32; JOINT_COUNT], RobotError> {
        finite(&action)?;
        command.validate()?;
        let mut targets = [0.; JOINT_COUNT];
        for i in 0..LOWER_COUNT {
            // Upstream NumPy scales in f32, then adds f32 defaults.
            targets[i] = action[i] * 0.25 + LOWER_HOME[i];
        }
        targets[LOWER_COUNT..].copy_from_slice(&command.upper_positions);
        finite(&targets)?;
        self.previous_action = action;
        Ok(targets)
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

pub(crate) fn finite(values: &[f32]) -> Result<(), RobotError> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(RobotError::NonFinite("G1 boundary"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> G1State {
        let mut positions = [0.; JOINT_COUNT];
        positions[..LOWER_COUNT].copy_from_slice(&LOWER_HOME);
        G1State {
            positions,
            velocities: [0.; JOINT_COUNT],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_angular_velocity_body: [0.; 3],
        }
    }

    #[test]
    fn history_warmup_switch_and_reset_follow_source() {
        let mut h = HomieHistory::default();
        let c = G1Command::default();
        let first = *h.observe(&home(), &c).unwrap();
        assert!(first[..430].iter().all(|&x| x == 0.));
        assert_eq!(&first[440..443], &[0., -0., -1.]);
        h.accept_action([0.4; LOWER_COUNT], &c).unwrap();
        let moving = G1Command {
            navigation: [0.2, 0., 0.],
            ..c.clone()
        };
        let next = h.observe(&home(), &moving).unwrap();
        assert_eq!(&next[344..430], &first[430..]);
        assert_eq!(&next[501..], &[0.4; LOWER_COUNT]);
        h.reset();
        assert_eq!(*h.observe(&home(), &c).unwrap(), first);
    }

    #[test]
    fn fingers_are_excluded_from_observation_and_upper_targets_pass_through() {
        let mut state = home();
        state.positions[22] = 2.;
        state.positions[29] = 0.75;
        let mut command = G1Command::default();
        command.upper_positions[22 - LOWER_COUNT] = 0.7;
        let mut h = HomieHistory::default();
        let obs = h.observe(&state, &command).unwrap();
        assert_eq!(obs[430 + 13 + 22], 0.75);
        let targets = h.accept_action([0.; LOWER_COUNT], &command).unwrap();
        assert_eq!(targets[22], 0.7);
        assert_eq!(&targets[..LOWER_COUNT], &LOWER_HOME);
    }

    #[test]
    fn bad_sensors_do_not_advance_history() {
        let mut h = HomieHistory::default();
        let mut state = home();
        state.velocities[42] = f32::NAN;
        assert!(h.observe(&state, &G1Command::default()).is_err());
        assert!(h.observation.iter().all(|&v| v == 0.));
        assert!(
            !G1Command {
                navigation: [0.05, 0., 0.],
                ..Default::default()
            }
            .standing()
        );
    }
}
