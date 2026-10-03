//! Bounded traditional AGILE placement from disclosed grasp/target memory.
//! Inputs: past RGB, current self sensors, public collision vertices. No truth.
use super::{
    runner::G1Measurement,
    static_transfer::{StaticCartesianReceipt, StaticLeftPalmKinematics, self_rotation},
};
use rapier3d::na::{Matrix3, Vector3};
use robot_minigame::{RobotError, g1::agile::AgileCommand};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

pub const STATIC_PLACE_TICKS: u64 = 325;
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticMemoryPlaceGoal {
    pub origin_observation: ObservationStamp,
    pub current_observation: ObservationStamp,
    pub root_from_apple_estimate: [[f64; 4]; 4],
    pub root_from_plate_estimate: [[f64; 4]; 4],
    pub memory_input_sha256: String,
    pub public_geometry_sha256: String,
    pub rigid_grasp_assumption: bool,
    pub static_target_assumption: bool,
    pub current_object_visual_detections: bool,
}
impl StaticMemoryPlaceGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let a = self.origin_observation;
        let b = self.current_observation;
        if a.episode_id == 0
            || a.episode_id != b.episode_id
            || a.frame_id == 0
            || b.frame_id <= a.frame_id
            || a.sim_time_ns != 140 * 20_000_000
            || b.sim_time_ns != 390 * 20_000_000
            || a.captured_at_unix_ms == 0
            || b.captured_at_unix_ms < a.captured_at_unix_ms
            || b.captured_at_unix_ms - a.captured_at_unix_ms > 8000
            || !self.rigid_grasp_assumption
            || !self.static_target_assumption
            || self.current_object_visual_detections
            || !sha(&self.memory_input_sha256)
            || !sha(&self.public_geometry_sha256)
        {
            return Err(invalid(
                "static placement requires bounded disclosed140-to390Tick memory",
            ));
        }
        matrix(&self.root_from_apple_estimate)?;
        matrix(&self.root_from_plate_estimate)?;
        Ok(())
    }
}
pub struct StaticPlacementGeometry {
    apple: Vec<[f64; 3]>,
    plate: Vec<[f64; 3]>,
    sha256: String,
}
impl StaticPlacementGeometry {
    pub(super) fn load(
        config: &super::task_objects::TaskObjectSceneConfig,
    ) -> Result<Self, RobotError> {
        let d = super::task_objects::TaskObjectsDefinition::load(
            &config.definition,
            &config.definition_sha256,
        )?;
        let (apple, plate) = d.static_placement_vertices();
        if apple.is_empty() || plate.is_empty() {
            return Err(invalid("missing original static public geometry"));
        }
        Ok(Self {
            apple,
            plate,
            sha256: config.definition_sha256.clone(),
        })
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct StaticMemoryPlaceStep {
    pub goal: StaticMemoryPlaceGoal,
    pub placement_ticks: u32,
    pub phase: &'static str,
    pub completed: bool,
    pub source_displacement_m: [f64; 3],
    pub classical_geometry: Option<StaticCartesianReceipt>,
    pub command: AgileCommand,
    pub world_or_contact_truth_input: bool,
    pub task_qualified: bool,
}
pub struct StaticMemoryPlace {
    goal: StaticMemoryPlaceGoal,
    delta: Vector3<f64>,
    ticks: u32,
    command: AgileCommand,
    closed_hand: [f32; 7],
}
impl StaticMemoryPlace {
    pub(super) fn new(
        goal: StaticMemoryPlaceGoal,
        state: &G1Measurement,
        command: AgileCommand,
        geometry: &StaticPlacementGeometry,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if state.episode_id != goal.current_observation.episode_id
            || state.source_tick != 390
            || state.sim_time_ns != goal.current_observation.sim_time_ns
            || goal.public_geometry_sha256 != geometry.sha256
        {
            return Err(invalid(
                "placement memory does not match390Tick owner/public geometry",
            ));
        }
        let root = self_rotation(state)?.to_rotation_matrix().into_inner();
        let apple_r = root * matrix(&goal.root_from_apple_estimate)?;
        let plate_r = root * matrix(&goal.root_from_plate_estimate)?;
        let low = geometry
            .apple
            .iter()
            .map(|p| (apple_r * Vector3::from(*p)).z)
            .fold(f64::INFINITY, f64::min);
        let high = geometry
            .plate
            .iter()
            .map(|p| (plate_r * Vector3::from(*p)).z)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut target = root * translation(&goal.root_from_plate_estimate);
        target.z += high - low + 0.005;
        let delta = target - root * translation(&goal.root_from_apple_estimate);
        if !delta.iter().all(|v| v.is_finite())
            || !(-0.15..=0.).contains(&delta.z)
            || delta.fixed_rows::<2>(0).norm() > 0.035
            || delta.norm() / 100. > 0.0016
        {
            return Err(invalid("static memory lowering path exceeds frozen bounds"));
        }
        let closed_hand = command.upper_positions[7..14].try_into().unwrap();
        Ok(Self {
            goal,
            delta,
            ticks: 0,
            command,
            closed_hand,
        })
    }
    pub fn goal(&self) -> &StaticMemoryPlaceGoal {
        &self.goal
    }
    pub(super) fn update(
        &mut self,
        state: &G1Measurement,
        fk: &StaticLeftPalmKinematics,
    ) -> Result<StaticMemoryPlaceStep, RobotError> {
        if self.ticks >= 325
            || state.episode_id != self.goal.current_observation.episode_id
            || state.source_tick != 390 + u64::from(self.ticks)
            || state.sim_time_ns != state.source_tick * 20_000_000
        {
            return Err(invalid("static placement repeated/foreign/excessiveTick"));
        }
        let (phase, receipt) = if self.ticks < 100 {
            let (c, r) = fk.translate(
                state,
                &self.command,
                (self_rotation(state)?.inverse() * (self.delta / 100.)).into(),
            )?;
            self.command = c;
            ("lower", Some(r))
        } else if self.ticks < 150 {
            let remaining = 1. - (self.ticks - 100 + 1) as f32 / 50.;
            for i in 0..7 {
                self.command.upper_positions[7 + i] = self.closed_hand[i] * remaining;
            }
            ("open", None)
        } else if self.ticks < 200 {
            let (c, r) = fk.translate(
                state,
                &self.command,
                (self_rotation(state)?.inverse() * Vector3::new(0., 0., 0.04 / 50.)).into(),
            )?;
            self.command = c;
            ("retract", Some(r))
        } else {
            ("settle", None)
        };
        self.command.validate()?;
        self.ticks += 1;
        Ok(StaticMemoryPlaceStep {
            goal: self.goal.clone(),
            placement_ticks: self.ticks,
            phase,
            completed: self.ticks == 325,
            source_displacement_m: self.delta.into(),
            classical_geometry: receipt,
            command: self.command.clone(),
            world_or_contact_truth_input: false,
            task_qualified: false,
        })
    }
}
fn matrix(m: &[[f64; 4]; 4]) -> Result<Matrix3<f64>, RobotError> {
    if !m.iter().flatten().all(|v| v.is_finite())
        || m[3] != [0., 0., 0., 1.]
        || translation(m).norm() > 2.
    {
        return Err(invalid("invalid bounded memory pose"));
    }
    let r = Matrix3::from_fn(|i, j| m[i][j]);
    if (r.transpose() * r - Matrix3::identity()).abs().max() > 1e-6
        || (r.determinant() - 1.).abs() > 1e-6
    {
        return Err(invalid("memory pose is not a rigid rotation"));
    }
    Ok(r)
}
fn translation(m: &[[f64; 4]; 4]) -> Vector3<f64> {
    Vector3::new(m[0][3], m[1][3], m[2][3])
}
fn sha(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn invalid(s: impl Into<String>) -> RobotError {
    RobotError::Contract(s.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn memory() -> StaticMemoryPlaceGoal {
        let pose = [
            [1., 0., 0., 0.3],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.1],
            [0., 0., 0., 1.],
        ];
        StaticMemoryPlaceGoal {
            origin_observation: ObservationStamp {
                episode_id: 7,
                frame_id: 3,
                sim_time_ns: 2_800_000_000,
                captured_at_unix_ms: 10_000,
            },
            current_observation: ObservationStamp {
                episode_id: 7,
                frame_id: 4,
                sim_time_ns: 7_800_000_000,
                captured_at_unix_ms: 15_144,
            },
            root_from_apple_estimate: pose,
            root_from_plate_estimate: pose,
            memory_input_sha256: "a".repeat(64),
            public_geometry_sha256: "b".repeat(64),
            rigid_grasp_assumption: true,
            static_target_assumption: true,
            current_object_visual_detections: false,
        }
    }
    #[test]
    fn refuses_reset_stale_relabelled_or_nonrigid_memory() {
        let good = memory();
        assert!(good.validate().is_ok());
        let mut bad = good.clone();
        bad.current_observation.episode_id += 1;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.current_observation.captured_at_unix_ms = 19_000;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.current_object_visual_detections = true;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.root_from_apple_estimate[0][0] = 2.;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.root_from_plate_estimate[0][3] = f64::NAN;
        assert!(bad.validate().is_err());
        let mut json = serde_json::to_value(good).unwrap();
        json["world_position"] = serde_json::json!([0., 0., 1.]);
        assert!(serde_json::from_value::<StaticMemoryPlaceGoal>(json).is_err());
    }
}
