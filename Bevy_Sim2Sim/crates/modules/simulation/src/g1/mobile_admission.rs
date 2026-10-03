//! Keep image acquisition and actual owner execution clocks distinct.
//! Continuous admission is constructed only from the owner's bounded self
//! history during explicit standing. No object/world/contact input is accepted.

use super::runner::G1Measurement;
use robot_minigame::RobotError;
use serde::Serialize;
use task_minigame::types::ObservationStamp;

#[derive(Clone, Debug, Serialize)]
pub struct MobileImageAdmission {
    observation: ObservationStamp,
    execution_start_sim_ns: u64,
    image_age_ns: u64,
    camera_to_owner_self_displacement_source_m: [f64; 3],
    maximum_measured_palm_motion_m: f64,
    self_rotation_change_rad: f64,
    continuous_stationary_wait: bool,
}

impl MobileImageAdmission {
    pub(super) fn navigation_target_at_execution(&self, heading: f32, distance: f32) -> (f32, f32) {
        if !self.continuous_stationary_wait {
            return (heading, distance);
        }
        let x = f64::from(distance) * f64::from(heading).cos()
            - self.camera_to_owner_self_displacement_source_m[0];
        let y = f64::from(distance) * f64::from(heading).sin()
            - self.camera_to_owner_self_displacement_source_m[1];
        (y.atan2(x) as f32, x.hypot(y) as f32)
    }
    pub(super) fn at_current_boundary(
        observation: ObservationStamp,
        state: &G1Measurement,
    ) -> Result<Self, RobotError> {
        if observation.episode_id != state.episode_id
            || observation.sim_time_ns != state.sim_time_ns
            || state.source_tick.checked_mul(20_000_000) != Some(state.sim_time_ns)
        {
            return Err(invalid("image is not the current paused owner boundary"));
        }
        Ok(Self {
            observation,
            execution_start_sim_ns: state.sim_time_ns,
            image_age_ns: 0,
            camera_to_owner_self_displacement_source_m: [0.; 3],
            maximum_measured_palm_motion_m: 0.,
            self_rotation_change_rad: 0.,
            continuous_stationary_wait: false,
        })
    }

    pub(super) fn from_stationary_wait(
        observation: ObservationStamp,
        state: &G1Measurement,
        displacement: [f64; 3],
        palm_motion_m: f64,
        rotation_change_rad: f64,
        maximum_image_age_ns: u64,
    ) -> Result<Self, RobotError> {
        let age = state
            .sim_time_ns
            .checked_sub(observation.sim_time_ns)
            .ok_or_else(|| invalid("stationary image is from a future owner clock"))?;
        if observation.episode_id != state.episode_id
            || observation.frame_id == 0
            || observation.captured_at_unix_ms == 0
            || observation.sim_time_ns % 20_000_000 != 0
            || state.source_tick.checked_mul(20_000_000) != Some(state.sim_time_ns)
            || maximum_image_age_ns == 0
            || age > maximum_image_age_ns
            || !displacement.iter().all(|v| v.is_finite())
            || displacement.iter().map(|v| v * v).sum::<f64>().sqrt() > 0.005
            || !palm_motion_m.is_finite()
            || !(0. ..=0.005).contains(&palm_motion_m)
            || !rotation_change_rad.is_finite()
            || !(0. ..=0.01).contains(&rotation_change_rad)
        {
            return Err(invalid(&format!(
                "stationary image rejected: age_ns={age}, self_displacement_m={displacement:?}, palm_motion_m={palm_motion_m}, rotation_change_rad={rotation_change_rad}; bounds={maximum_image_age_ns}ns/0.005m/0.01rad"
            )));
        }
        Ok(Self {
            observation,
            execution_start_sim_ns: state.sim_time_ns,
            image_age_ns: age,
            camera_to_owner_self_displacement_source_m: displacement,
            maximum_measured_palm_motion_m: palm_motion_m,
            self_rotation_change_rad: rotation_change_rad,
            continuous_stationary_wait: true,
        })
    }

    pub(super) fn matches(&self, observation: ObservationStamp, state: &G1Measurement) -> bool {
        self.observation == observation
            && state.episode_id == observation.episode_id
            && self.execution_start_sim_ns == state.sim_time_ns
            && state.source_tick.checked_mul(20_000_000) == Some(state.sim_time_ns)
    }
}

fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(tick: u64) -> G1Measurement {
        G1Measurement {
            episode_id: 1,
            source_tick: tick,
            sim_time_ns: tick * 20_000_000,
            joint_positions: vec![0.; 43],
            joint_velocities: vec![0.; 43],
            root_rotation_wxyz: [1., 0., 0., 0.],
            root_velocity_source: [0.; 3],
            root_angular_velocity_body: [0.; 3],
        }
    }
    #[test]
    fn clocks_remain_distinct_and_proof_cannot_move_to_another_boundary() {
        let stamp = ObservationStamp {
            episode_id: 1,
            frame_id: 2,
            sim_time_ns: 20_000_000,
            captured_at_unix_ms: 3,
        };
        assert!(MobileImageAdmission::at_current_boundary(stamp, &state(2)).is_err());
        let a = MobileImageAdmission::from_stationary_wait(
            stamp,
            &state(2),
            [0.001, 0., 0.],
            0.002,
            0.001,
            1_000_000_000,
        )
        .unwrap();
        assert_eq!(a.observation, stamp);
        assert_eq!(a.execution_start_sim_ns, 40_000_000);
        assert!(a.matches(stamp, &state(2)));
        assert!(!a.matches(stamp, &state(3)));
        let mut foreign = stamp;
        foreign.episode_id = 2;
        assert!(!a.matches(foreign, &state(2)));
        for (d, p, r, age) in [
            ([0.006, 0., 0.], 0., 0., 1_000_000_000),
            ([0.; 3], 0.006, 0., 1_000_000_000),
            ([0.; 3], 0., 0.011, 1_000_000_000),
            ([0.; 3], 0., 0., 1),
        ] {
            assert!(
                MobileImageAdmission::from_stationary_wait(stamp, &state(2), d, p, r, age).is_err()
            );
        }
    }
}
