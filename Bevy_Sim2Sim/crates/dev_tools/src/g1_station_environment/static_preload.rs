//! Finite diagnostic finger preload from original joint encoders only.
//! Position error is a load proxy, never proof of object contact or retention.

use robot_minigame::g1::agile::AgileCommand;
use simulation_minigame::g1::runner::G1Measurement;
use task_minigame::policy::ActionLimits;

pub(super) struct StaticFingerPreload {
    command: AgileCommand,
    limits: ActionLimits,
    episode_id: u64,
    updates: u64,
}

impl StaticFingerPreload {
    pub(super) fn new(
        command: AgileCommand,
        limits: ActionLimits,
        measurement: &G1Measurement,
    ) -> Result<Self, String> {
        command.validate().map_err(|e| e.to_string())?;
        limits.validate().map_err(|e| format!("{e:?}"))?;
        if measurement.source_tick != 140 || measurement.episode_id == 0 {
            return Err("preload requires the explicit140Tick mechanical boundary".into());
        }
        Ok(Self {
            command,
            limits,
            episode_id: measurement.episode_id,
            updates: 0,
        })
    }

    pub(super) fn next(&mut self, measurement: &G1Measurement) -> Result<AgileCommand, String> {
        if self.updates >= 20
            || measurement.episode_id != self.episode_id
            || measurement.source_tick != 140 + self.updates
            || measurement.sim_time_ns != measurement.source_tick * 20_000_000
            || measurement.joint_positions.len() != 43
            || !measurement.joint_positions.iter().all(|q| q.is_finite())
        {
            return Err("preload encoder episode/time/count/finite gate failed".into());
        }
        // Original source stiffness remains4Nm/rad.0.08rad is a bounded
        // position-load proxy (~0.32Nm before damping), not a torque sensor.
        // Close at1rad/s on the unchanged50Hz clock, within original bounds.
        // Keep thumb opposition, arms, right hand, navigation and height fixed.
        for (digit, direction) in [(0, -1.), (1, -1.), (2, -1.), (3, -1.), (5, 1.), (6, 1.)] {
            let upper = 7 + digit;
            let measured = measurement.joint_positions[15 + upper];
            let desired = self.command.upper_positions[upper];
            if direction * (desired - measured) < 0.08 {
                self.command.upper_positions[upper] = (desired + direction * 0.02).clamp(
                    self.limits.joint_min_rad[14 + digit],
                    self.limits.joint_max_rad[14 + digit],
                );
            }
        }
        self.updates += 1;
        Ok(self.command.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (AgileCommand, ActionLimits, G1Measurement) {
        (
            AgileCommand {
                navigation: [0.; 3],
                pelvis_height: 0.75,
                upper_positions: [0.; 28],
            },
            ActionLimits {
                joint_min_rad: [-0.1; 31],
                joint_max_rad: [0.1; 31],
                base_height_min_m: 0.4,
                base_height_max_m: 0.9,
                navigation_abs_max: [1.; 3],
            },
            G1Measurement {
                episode_id: 7,
                source_tick: 140,
                sim_time_ns: 2_800_000_000,
                joint_positions: vec![0.; 43],
                joint_velocities: vec![0.; 43],
                root_rotation_wxyz: [1., 0., 0., 0.],
                root_angular_velocity_body: [0.; 3],
                root_velocity_source: [0.; 3],
            },
        )
    }

    #[test]
    fn encoder_tracking_closes_only_flexion_with_original_limits_and_twenty_tick_bound() {
        let (origin, limits, mut selfstate) = fixture();
        let mut preload = StaticFingerPreload::new(origin.clone(), limits, &selfstate).unwrap();
        for tick in 0..20 {
            selfstate.source_tick = 140 + tick;
            selfstate.sim_time_ns = selfstate.source_tick * 20_000_000;
            let command = preload.next(&selfstate).unwrap();
            assert_eq!(command.upper_positions[..7], origin.upper_positions[..7]);
            assert_eq!(command.upper_positions[11], 0.);
            assert_eq!(command.upper_positions[14..], origin.upper_positions[14..]);
            assert_eq!(command.navigation, origin.navigation);
            assert_eq!(command.pelvis_height, origin.pelvis_height);
            assert!(command.upper_positions.iter().all(|q| q.abs() <= 0.1));
            selfstate.joint_positions[15..].copy_from_slice(&command.upper_positions);
        }
        assert!(preload.next(&selfstate).is_err());
    }

    #[test]
    fn loaded_encoders_stop_closure_and_stale_or_nonfinite_states_are_rejected() {
        let (origin, limits, mut selfstate) = fixture();
        for (digit, sign) in [(0, -1.), (1, -1.), (2, -1.), (3, -1.), (5, 1.), (6, 1.)] {
            selfstate.joint_positions[22 + digit] = -sign * 0.1;
        }
        let mut preload = StaticFingerPreload::new(origin.clone(), limits, &selfstate).unwrap();
        let mut invalid = selfstate.clone();
        invalid.episode_id += 1;
        assert!(preload.next(&invalid).is_err());
        invalid = selfstate.clone();
        invalid.joint_positions[22] = f32::NAN;
        assert!(preload.next(&invalid).is_err());
        assert_eq!(
            preload.next(&selfstate).unwrap().upper_positions,
            origin.upper_positions
        );
        assert!(preload.next(&selfstate).is_err());
    }
}
