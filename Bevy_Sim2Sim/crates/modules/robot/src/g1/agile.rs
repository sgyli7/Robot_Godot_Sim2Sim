//! Independent T1 AGILE recurrent-student boundary at Arena 8b4a3a47fc53.
//!
//! Matches the pinned `g1_agile.yaml` and `g1_agile_policy.py`. In particular,
//! this is an 80-wide observation with raw 12-action feedback, not Homie's
//! 516-wide history. The caller supplies the actual source/native default q;
//! static-task shoulder defaults must not be replaced with Homie's home pose.

use super::{
    actuator::IdealPd,
    contract::{JOINT_COUNT, UPPER_COUNT, finite},
};
use crate::RobotError;

mod policy;
pub use policy::{AgilePolicy, AgileResult};

pub const CONTRACT_VERSION: &str = "arena_8b4a3_agile_student_v1";
pub const MODEL_SHA256: &str = "c8e30ec353bbba464298aeb051963cee17e1f1a484081c726ebb1d7513570da6";
pub const OBSERVATION_SIZE: usize = 80;
pub const ACTION_SIZE: usize = 12;
pub const RECURRENT_SIZE: usize = 256;
pub const INPUT_INDICES: [usize; 29] = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 29, 4, 10, 16, 30, 5, 11, 17, 31, 18, 32, 19, 33, 20,
    34, 21, 35,
];
pub const OUTPUT_INDICES: [usize; ACTION_SIZE] = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11];
pub const ACTION_SCALE: [f32; ACTION_SIZE] = [
    0.5475464463233948,
    0.5475464463233948,
    0.3506614565849304,
    0.3506614565849304,
    0.5475464463233948,
    0.5475464463233948,
    0.3506614565849304,
    0.3506614565849304,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
];
pub const ACTION_OFFSET: [f32; ACTION_SIZE] =
    [-0.1, -0.1, 0., 0., 0., 0., 0.3, 0.3, -0.2, -0.2, 0., 0.];

/// Sensors use AGILE's source convention. COM-frame angular velocity and
/// projected gravity are explicit rather than silently borrowed from Homie.
#[derive(Clone, Debug)]
pub struct AgileState {
    pub positions: [f32; JOINT_COUNT],
    pub velocities: [f32; JOINT_COUNT],
    pub root_angular_velocity_com: [f32; 3],
    pub projected_gravity_body: [f32; 3],
}

impl AgileState {
    pub fn validate(&self) -> Result<(), RobotError> {
        finite(&self.positions)?;
        finite(&self.velocities)?;
        finite(&self.root_angular_velocity_com)?;
        finite(&self.projected_gravity_body)?;
        let norm = self
            .projected_gravity_body
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>();
        if (norm - 1.).abs() > 1e-5 {
            return Err(RobotError::Contract(
                "AGILE projected gravity must be a unit vector".into(),
            ));
        }
        Ok(())
    }
}

/// Source SI velocity-height command; AGILE has no torso-RPY command channel.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgileCommand {
    pub navigation: [f32; 3],
    pub pelvis_height: f32,
    pub upper_positions: [f32; UPPER_COUNT],
}

impl AgileCommand {
    pub fn validate(&self) -> Result<(), RobotError> {
        finite(&self.navigation)?;
        finite(&[self.pelvis_height])?;
        finite(&self.upper_positions)
    }
}

pub struct AgileHistory {
    default_positions: [f32; JOINT_COUNT],
    observation: [f32; OBSERVATION_SIZE],
    previous_raw_action: [f32; ACTION_SIZE],
}

impl AgileHistory {
    pub fn new(default_positions: [f32; JOINT_COUNT]) -> Result<Self, RobotError> {
        finite(&default_positions)?;
        Ok(Self {
            default_positions,
            observation: [0.; OBSERVATION_SIZE],
            previous_raw_action: [0.; ACTION_SIZE],
        })
    }

    pub fn observe(
        &mut self,
        state: &AgileState,
        command: &AgileCommand,
    ) -> Result<&[f32; OBSERVATION_SIZE], RobotError> {
        state.validate()?;
        command.validate()?;
        self.observation[..3].copy_from_slice(&command.navigation);
        self.observation[3] = command.pelvis_height;
        self.observation[4..7].copy_from_slice(&state.root_angular_velocity_com);
        self.observation[7..10].copy_from_slice(&state.projected_gravity_body);
        for (i, &joint) in INPUT_INDICES.iter().enumerate() {
            // Upstream casts q, dq and default_q to f32 before these operations.
            self.observation[10 + i] = state.positions[joint] - self.default_positions[joint];
            self.observation[39 + i] = state.velocities[joint] * 0.1;
        }
        self.observation[68..].copy_from_slice(&self.previous_raw_action);
        finite(&self.observation)?;
        Ok(&self.observation)
    }

    pub fn accept_action(
        &mut self,
        raw: [f32; ACTION_SIZE],
        command: &AgileCommand,
    ) -> Result<[f32; JOINT_COUNT], RobotError> {
        finite(&raw)?;
        command.validate()?;
        let mut targets = [0.; JOINT_COUNT];
        for i in 0..ACTION_SIZE {
            targets[OUTPUT_INDICES[i]] = raw[i].clamp(-6., 6.) * ACTION_SCALE[i] + ACTION_OFFSET[i];
        }
        // Waist stays at zero; only the 28 named upper joints pass through.
        targets[15..].copy_from_slice(&command.upper_positions);
        finite(&targets)?;
        self.previous_raw_action = raw;
        Ok(targets)
    }

    pub fn reset(&mut self) {
        self.observation.fill(0.);
        self.previous_raw_action.fill(0.);
    }
}

/// Source AGILE actuator metadata; this does not change any world's backend.
pub fn parameters(index: usize) -> Result<IdealPd, RobotError> {
    let mut p = super::actuator::parameters(index)?;
    let name = super::contract::JOINT_NAMES[index];
    let family = if name.contains("hip_pitch") || name.contains("hip_yaw") {
        Some(0.010177520_f64)
    } else if name.contains("hip_roll") || name.contains("knee_") {
        Some(0.025101925_f64)
    } else if name.contains("ankle_") {
        Some(2. * 0.003609725_f64)
    } else {
        None
    };
    if let Some(armature) = family {
        let omega = 10. * 2. * std::f64::consts::PI;
        p.armature = armature as f32;
        p.stiffness = (armature * omega.powi(2)) as f32;
        p.damping = (4. * armature * omega) as f32;
    } else if name.starts_with("waist_") {
        p.stiffness = 300.;
        p.damping = 5.;
        p.armature = 0.03;
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> AgileState {
        AgileState {
            positions: [0.; 43],
            velocities: [0.; 43],
            root_angular_velocity_com: [0.; 3],
            projected_gravity_body: [0., 0., -1.],
        }
    }
    fn command() -> AgileCommand {
        AgileCommand {
            navigation: [0.2, -0.1, 0.3],
            pelvis_height: 0.75,
            upper_positions: [0.; 28],
        }
    }

    #[test]
    fn named_source_permutation_retains_task_defaults_and_unscaled_sensors() {
        let default = std::array::from_fn(|i| i as f32 * 0.01);
        let mut s = state();
        s.positions = std::array::from_fn(|i| default[i] + i as f32 * 0.25);
        s.velocities = std::array::from_fn(|i| i as f32);
        s.root_angular_velocity_com = [0.1, -0.2, 0.3];
        let mut history = AgileHistory::new(default).unwrap();
        let obs = history.observe(&s, &command()).unwrap();
        assert_eq!(&obs[..7], &[0.2, -0.1, 0.3, 0.75, 0.1, -0.2, 0.3]);
        for (i, &joint) in INPUT_INDICES.iter().enumerate() {
            assert_eq!(obs[10 + i], s.positions[joint] - default[joint]);
            assert_eq!(obs[39 + i], joint as f32 * 0.1);
        }
        assert_eq!(
            super::super::contract::JOINT_NAMES[INPUT_INDICES[15]],
            "left_shoulder_roll_joint"
        );
        assert_eq!(
            super::super::contract::JOINT_NAMES[INPUT_INDICES[19]],
            "left_shoulder_yaw_joint"
        );
    }

    #[test]
    fn clipped_targets_keep_raw_feedback_and_reset_clears_only_episode_state() {
        let mut history = AgileHistory::new([0.25; 43]).unwrap();
        let mut cmd = command();
        cmd.upper_positions = std::array::from_fn(|i| i as f32 * 0.01);
        let raw = std::array::from_fn(|i| if i % 2 == 0 { 8. } else { -8. });
        let targets = history.accept_action(raw, &cmd).unwrap();
        assert_eq!(targets[0], 6. * ACTION_SCALE[0] - 0.1);
        assert_eq!(targets[6], -6. * ACTION_SCALE[1] - 0.1);
        assert_eq!(&targets[12..15], &[0.; 3]);
        assert_eq!(&targets[15..], &cmd.upper_positions);
        assert_eq!(&history.observe(&state(), &cmd).unwrap()[68..], &raw);
        history.reset();
        let obs = history.observe(&state(), &cmd).unwrap();
        assert_eq!(&obs[68..], &[0.; 12]);
        assert_eq!(obs[10], -0.25);
    }

    #[test]
    fn rejects_invalid_inputs_before_mutating_raw_feedback() {
        assert!(AgileHistory::new([f32::NAN; 43]).is_err());
        let mut history = AgileHistory::new([0.; 43]).unwrap();
        history.accept_action([0.2; 12], &command()).unwrap();
        assert!(
            history
                .accept_action([f32::INFINITY; 12], &command())
                .is_err()
        );
        let mut invalid = state();
        invalid.projected_gravity_body = [0.; 3];
        assert!(history.observe(&invalid, &command()).is_err());
        assert_eq!(
            &history.observe(&state(), &command()).unwrap()[68..],
            &[0.2; 12]
        );
    }

    #[test]
    fn agile_actuators_keep_physical_limits_without_mutating_homie() {
        assert!((parameters(0).unwrap().stiffness - 40.17924).abs() < 1e-5);
        assert!((parameters(1).unwrap().armature - 0.025101925).abs() < 1e-8);
        assert_eq!(parameters(3).unwrap().effort_limit, 139.);
        assert!((parameters(4).unwrap().stiffness - 28.501247).abs() < 1e-5);
        assert_eq!(parameters(12).unwrap().stiffness, 300.);
        assert_eq!(parameters(22).unwrap().stiffness, 4.);
        assert_eq!(
            super::super::actuator::parameters(0).unwrap().stiffness,
            150.
        );
        assert!(parameters(43).is_err());
    }

    #[test]
    #[ignore = "requires G1_AGILE_ORACLE and G1_AGILE_ORACLE_SHA256 from actual pinned upstream Python"]
    fn upstream_transform_parity() {
        let bytes = super::super::policy::bound_bytes(
            std::path::Path::new(&std::env::var("G1_AGILE_ORACLE").unwrap()),
            &std::env::var("G1_AGILE_ORACLE_SHA256").unwrap(),
        )
        .unwrap();
        let receipt: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            receipt["arena_commit"],
            "8b4a3a47fc53de23e8205089d71109a2e2348acd"
        );
        assert_eq!(receipt["weight_sha256"], MODEL_SHA256);
        assert_eq!(receipt["physics_integrations"], 0);
        let mut history = None;
        let mut previous_episode = None;
        let mut count = 0;
        for frame in receipt["frames"].as_array().unwrap() {
            let vector = |name: &str| {
                frame[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let episode = frame["episode"].as_u64().unwrap();
            if previous_episode != Some(episode) {
                history = Some(
                    AgileHistory::new(vector("default_positions").try_into().unwrap()).unwrap(),
                );
                previous_episode = Some(episode);
            }
            let history = history.as_mut().unwrap();
            let state = AgileState {
                positions: vector("positions").try_into().unwrap(),
                velocities: vector("velocities").try_into().unwrap(),
                root_angular_velocity_com: vector("root_angular_velocity_body").try_into().unwrap(),
                projected_gravity_body: vector("projected_gravity_body").try_into().unwrap(),
            };
            let command = AgileCommand {
                navigation: vector("navigation").try_into().unwrap(),
                pelvis_height: frame["pelvis_height"].as_f64().unwrap() as f32,
                upper_positions: vector("upper_positions").try_into().unwrap(),
            };
            let actual = history.observe(&state, &command).unwrap();
            for (a, b) in actual.iter().zip(vector("observation")) {
                assert!(
                    (a - b).abs() <= 1e-6,
                    "frame {count}: observation {a} vs {b}"
                );
            }
            let targets = history
                .accept_action(vector("raw_action").try_into().unwrap(), &command)
                .unwrap();
            for (a, b) in targets.iter().zip(vector("targets")) {
                assert!((a - b).abs() <= 1e-6, "frame {count}: target {a} vs {b}");
            }
            count += 1;
        }
        assert_eq!(count, 12);
    }
}
