//! Matched decoded VLA -> physical-controller command boundary.
//!
//! Both original Arena WBC configs assign waist joints to the lower policy.
//! Decoded waist joint angles therefore never become torso-orientation commands.
//! This adapter grants no physical or task qualification.

use robot_minigame::g1::{
    agile::AgileCommand,
    contract::{G1Command, UPPER_COUNT},
};
use task_minigame::{
    policy::{ActionLimits, PolicyActionError, PolicyActionFrame},
    types::TaskProfile,
};

/// Distinct commands prevent a T1 frame from accidentally selecting Homie.
#[derive(Clone, Debug)]
pub enum ArenaControllerCommand {
    StaticAgile(AgileCommand),
    MobileHomieV2(G1Command),
}

#[derive(Clone, Debug)]
pub struct ArenaCommandAdmission {
    pub controller: ArenaControllerCommand,
    /// Retained for evidence. The original lower-body policy owns actual waist q.
    pub decoded_waist_targets_rad: [f32; 3],
}

/// Validate physical bounds before constructing a force-controller command.
pub fn controller_command(
    profile: TaskProfile,
    frame: &PolicyActionFrame,
    limits: &ActionLimits,
) -> Result<ArenaCommandAdmission, PolicyActionError> {
    frame.validate(limits)?;
    let mut upper_positions = [0.; UPPER_COUNT];
    // Canonical physical order: left arm, left hand, right arm, right hand.
    upper_positions[..7].copy_from_slice(&frame.left_arm);
    upper_positions[7..14].copy_from_slice(&frame.left_hand);
    upper_positions[14..21].copy_from_slice(&frame.right_arm);
    upper_positions[21..28].copy_from_slice(&frame.right_hand);
    let controller = match profile {
        TaskProfile::StaticApple => ArenaControllerCommand::StaticAgile(AgileCommand {
            navigation: frame.navigate_mps_rps,
            pelvis_height: frame.base_height_m,
            upper_positions,
        }),
        TaskProfile::MobileBox => ArenaControllerCommand::MobileHomieV2(G1Command {
            navigation: frame.navigate_mps_rps,
            pelvis_height: frame.base_height_m,
            // Original build_gr00t_action_np explicitly supplies zero torso RPY.
            torso_rpy: [0.; 3],
            upper_positions,
        }),
    };
    Ok(ArenaCommandAdmission {
        controller,
        decoded_waist_targets_rad: frame.waist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::g1::contract::{JOINT_NAMES, LOWER_COUNT};
    use task_minigame::policy::ACTION_JOINT_NAMES;

    fn fixture() -> (PolicyActionFrame, ActionLimits) {
        (
            PolicyActionFrame {
                left_arm: std::array::from_fn(|i| i as f32 * 0.01),
                right_arm: std::array::from_fn(|i| (i + 7) as f32 * 0.01),
                left_hand: std::array::from_fn(|i| (i + 14) as f32 * 0.01),
                right_hand: std::array::from_fn(|i| (i + 21) as f32 * 0.01),
                waist: [0.28, 0.29, 0.30],
                base_height_m: 0.75,
                navigate_mps_rps: [0.1, -0.2, 0.3],
            },
            ActionLimits {
                joint_min_rad: [-1.; 31],
                joint_max_rad: [1.; 31],
                base_height_min_m: 0.4,
                base_height_max_m: 0.9,
                navigation_abs_max: [1.; 3],
            },
        )
    }

    #[test]
    fn both_profiles_preserve_all_named_upper_joints_and_original_base_semantics() {
        let (frame, limits) = fixture();
        let canonical = frame.joint_targets();
        for profile in [TaskProfile::StaticApple, TaskProfile::MobileBox] {
            let admitted = controller_command(profile, &frame, &limits).unwrap();
            assert_eq!(admitted.decoded_waist_targets_rad, frame.waist);
            let positions = match admitted.controller {
                ArenaControllerCommand::StaticAgile(command) => {
                    assert_eq!(profile, TaskProfile::StaticApple);
                    assert_eq!(command.navigation, frame.navigate_mps_rps);
                    assert_eq!(command.pelvis_height, frame.base_height_m);
                    command.upper_positions
                }
                ArenaControllerCommand::MobileHomieV2(command) => {
                    assert_eq!(profile, TaskProfile::MobileBox);
                    assert_eq!(command.torso_rpy, [0.; 3]);
                    assert_eq!(command.navigation, frame.navigate_mps_rps);
                    assert_eq!(command.pelvis_height, frame.base_height_m);
                    command.upper_positions
                }
            };
            for (name, actual) in JOINT_NAMES[LOWER_COUNT..].iter().zip(positions) {
                let index = ACTION_JOINT_NAMES
                    .iter()
                    .position(|candidate| candidate == name)
                    .unwrap();
                assert_eq!(
                    actual, canonical[index],
                    "Incorrect physical target for {name}"
                );
            }
        }
    }

    #[test]
    fn invalid_decoded_waist_cannot_bypass_full_frame_admission() {
        let (mut frame, limits) = fixture();
        frame.waist[2] = f32::NAN;
        assert!(matches!(
            controller_command(TaskProfile::StaticApple, &frame, &limits),
            Err(PolicyActionError::NonFinite)
        ));
        frame.waist[2] = 1.1;
        assert!(matches!(
            controller_command(TaskProfile::MobileBox, &frame, &limits),
            Err(PolicyActionError::JointLimit { index: 30 })
        ));
    }
}
