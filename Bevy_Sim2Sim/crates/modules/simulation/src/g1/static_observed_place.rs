//! Separate one-chunk traditional placement after corrected grasp and paired RGB.
//! Published geometry, measured self and disclosed rigid-grasp/static-target
//! assumptions only. World/contact truth is unavailable to this module.
use super::{
    runner::G1Measurement,
    static_place::StaticPlacementGeometry,
    static_transfer::{StaticCartesianReceipt, StaticLeftPalmKinematics, self_rotation},
};
use rapier3d::na::Vector3;
use robot_minigame::{RobotError, g1::agile::AgileCommand};
use serde::{Deserialize, Serialize};
use task_minigame::types::ObservationStamp;

pub const OBSERVED_PLACE_TICKS: u64 = 700;
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StaticObservedPlaceGoal {
    pub observations: [ObservationStamp; 2],
    pub image_sha256: [String; 2],
    pub input_sha256: [String; 2],
    pub root_from_apple: [[f64; 4]; 4],
    pub root_from_plate: [[f64; 4]; 4],
    pub public_geometry_sha256: String,
    pub rigid_grasp_assumption: bool,
    pub static_target_assumption: bool,
}
impl StaticObservedPlaceGoal {
    pub fn validate(&self) -> Result<(), RobotError> {
        let [a, b] = self.observations;
        if a.episode_id == 0
            || a.episode_id != b.episode_id
            || a.frame_id == 0
            || a.frame_id >= b.frame_id
            || a.sim_time_ns != b.sim_time_ns
            || a.sim_time_ns % 20_000_000 != 0
            || !(101..=500).contains(&(a.sim_time_ns / 20_000_000))
            || a.captured_at_unix_ms == 0
            || a.captured_at_unix_ms >= b.captured_at_unix_ms
            || b.captured_at_unix_ms - a.captured_at_unix_ms > 2000
            || !self.rigid_grasp_assumption
            || !self.static_target_assumption
            || self
                .image_sha256
                .iter()
                .chain(self.input_sha256.iter())
                .chain(std::iter::once(&self.public_geometry_sha256))
                .any(|s| !sha(s))
            || self.image_sha256[0] == self.image_sha256[1]
        {
            return Err(invalid(
                "observed placement requires a bounded same-Tick fixed RGB pair",
            ));
        }
        super::static_place::matrix(&self.root_from_apple)?;
        super::static_place::matrix(&self.root_from_plate)?;
        Ok(())
    }
    pub fn start_tick(&self) -> u64 {
        self.observations[1].sim_time_ns / 20_000_000
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StaticObservedPlaceStep {
    pub placement_ticks: u32,
    pub phase: &'static str,
    pub completed: bool,
    pub source_horizontal_displacement_m: [f64; 3],
    pub source_lowering_displacement_m: [f64; 3],
    pub classical_geometry: Option<StaticCartesianReceipt>,
    pub command: AgileCommand,
    pub original_vla_output: bool,
    pub world_or_contact_truth_input: bool,
    pub task_qualified: bool,
}
pub struct StaticObservedPlace {
    goal: StaticObservedPlaceGoal,
    horizontal: Vector3<f64>,
    lowering: Vector3<f64>,
    closed_hand: [f32; 7],
    command: AgileCommand,
    ticks: u64,
}
impl StaticObservedPlace {
    pub fn new(
        goal: StaticObservedPlaceGoal,
        state: &G1Measurement,
        command: AgileCommand,
        geometry: &StaticPlacementGeometry,
    ) -> Result<Self, RobotError> {
        goal.validate()?;
        command.validate()?;
        if state.episode_id != goal.observations[1].episode_id
            || state.source_tick != goal.start_tick()
            || state.sim_time_ns != goal.observations[1].sim_time_ns
            || goal.public_geometry_sha256 != geometry.definition_sha256()
            || command.navigation.iter().any(|v| v.abs() > 0.01)
        {
            return Err(invalid(
                "observed placement pair differs from its owner/public geometry",
            ));
        }
        let rotation = self_rotation(state)?;
        let apple = rotation * super::static_place::translation(&goal.root_from_apple);
        let plate = rotation * super::static_place::translation(&goal.root_from_plate);
        let mut horizontal = plate - apple;
        horizontal.z = 0.;
        if !(0.05..=0.30).contains(&horizontal.norm()) || horizontal.norm() / 200. > 0.0016 {
            return Err(invalid(
                "observed placement horizontal path exceeds its original frozen bounds",
            ));
        }
        let lowering = geometry.lowering_delta(
            state,
            &goal.root_from_apple,
            &goal.root_from_plate,
            horizontal + Vector3::new(0., 0., 0.05),
        )?;
        let closed_hand = command.upper_positions[7..14].try_into().unwrap();
        Ok(Self {
            goal,
            horizontal,
            lowering,
            closed_hand,
            command,
            ticks: 0,
        })
    }
    pub fn goal(&self) -> &StaticObservedPlaceGoal {
        &self.goal
    }
    pub fn update(
        &mut self,
        state: &G1Measurement,
        fk: &StaticLeftPalmKinematics,
    ) -> Result<StaticObservedPlaceStep, RobotError> {
        if self.ticks >= OBSERVED_PLACE_TICKS
            || state.episode_id != self.goal.observations[1].episode_id
            || state.source_tick != self.goal.start_tick() + self.ticks
            || state.sim_time_ns != state.source_tick * 20_000_000
        {
            return Err(invalid(
                "observed placement refuses repeated, stale or exhausted Tick",
            ));
        }
        let (phase, offset) = match self.ticks {
            0..50 => ("lift", Some(Vector3::new(0., 0., 0.001))),
            50..250 => ("transfer", Some(self.horizontal / 200.)),
            250..350 => ("lower", Some(self.lowering / 100.)),
            350..400 => {
                let remaining = 1. - (self.ticks - 350 + 1) as f32 / 50.;
                for i in 0..7 {
                    self.command.upper_positions[7 + i] = self.closed_hand[i] * remaining;
                }
                ("open", None)
            }
            400..450 => ("retract", Some(Vector3::new(0., 0., 0.04 / 50.))),
            450..575 => ("settle", None),
            575..675 => ("observation_withdrawal", Some(Vector3::new(0., 0.001, 0.))),
            675..700 => ("observation_hold", None),
            _ => return Err(invalid("observed placement phase exhausted")),
        };
        let receipt = if let Some(offset) = offset {
            let (command, receipt) = fk.translate(
                state,
                &self.command,
                (self_rotation(state)?.inverse() * offset).into(),
            )?;
            self.command = command;
            Some(receipt)
        } else {
            None
        };
        self.command.validate()?;
        fk.commanded_left_palm(state, &self.command)?;
        self.ticks += 1;
        Ok(StaticObservedPlaceStep {
            placement_ticks: self.ticks as u32,
            phase,
            completed: self.ticks == OBSERVED_PLACE_TICKS,
            source_horizontal_displacement_m: self.horizontal.into(),
            source_lowering_displacement_m: self.lowering.into(),
            classical_geometry: receipt,
            command: self.command.clone(),
            original_vla_output: false,
            world_or_contact_truth_input: false,
            task_qualified: false,
        })
    }
}
fn sha(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn invalid(s: impl std::fmt::Display) -> RobotError {
    RobotError::Contract(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn goal() -> StaticObservedPlaceGoal {
        let pose = [
            [1., 0., 0., 0.3],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.1],
            [0., 0., 0., 1.],
        ];
        StaticObservedPlaceGoal {
            observations: [
                ObservationStamp {
                    episode_id: 7,
                    frame_id: 3,
                    sim_time_ns: 6_920_000_000,
                    captured_at_unix_ms: 10000,
                },
                ObservationStamp {
                    episode_id: 7,
                    frame_id: 4,
                    sim_time_ns: 6_920_000_000,
                    captured_at_unix_ms: 10010,
                },
            ],
            image_sha256: ["a".repeat(64), "b".repeat(64)],
            input_sha256: ["c".repeat(64), "d".repeat(64)],
            root_from_apple: pose,
            root_from_plate: pose,
            public_geometry_sha256: "e".repeat(64),
            rigid_grasp_assumption: true,
            static_target_assumption: true,
        }
    }
    #[test]
    fn refuses_old_contract_reset_stale_nonrigid_and_truth() {
        let good = goal();
        assert!(good.validate().is_ok());
        let mut bad = good.clone();
        bad.observations[1].episode_id += 1;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.observations[1].sim_time_ns += 20_000_000;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.observations[1].captured_at_unix_ms += 2001;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.root_from_apple[0][0] = 2.;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.rigid_grasp_assumption = false;
        assert!(bad.validate().is_err());
        let mut bad = serde_json::to_value(good).unwrap();
        bad["contact_truth"] = serde_json::json!(true);
        assert!(serde_json::from_value::<StaticObservedPlaceGoal>(bad).is_err());
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SelfState {
        episode_id: u64,
        source_tick: u64,
        sim_time_ns: u64,
        joint_positions: Vec<f32>,
        joint_velocities: Vec<f32>,
        root_rotation_wxyz: [f32; 4],
        root_angular_velocity_body: [f32; 3],
        root_velocity_source: [f32; 3],
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        definition: std::path::PathBuf,
        definition_sha256: String,
        geometry: std::path::PathBuf,
        geometry_sha256: String,
        state: SelfState,
        command: AgileCommand,
        goal: StaticObservedPlaceGoal,
        output: std::path::PathBuf,
    }
    #[test]
    #[ignore = "explicit saved actual pairedRGB/self whole-path preflight;0native/model/image calls"]
    fn preflight_actual_paired_rgb_placement() -> Result<(), RobotError> {
        use robot_minigame::g1::{definition::G1Definition, policy::bound_bytes};
        let path = std::env::var("G1_OBSERVED_PLACE_PREFLIGHT").map_err(invalid)?;
        let hash = std::env::var("G1_OBSERVED_PLACE_PREFLIGHT_SHA256").map_err(invalid)?;
        let bytes = bound_bytes(std::path::Path::new(&path), &hash)?;
        if bytes.len() > 128 * 1024 {
            return Err(invalid("observed place preflight fixture exceeds bound"));
        }
        let f: Fixture = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let definition = G1Definition::load(&f.definition, &f.definition_sha256)?;
        let fk = StaticLeftPalmKinematics::new(&definition)?;
        let geometry =
            StaticPlacementGeometry::load_public_definition(&f.geometry, &f.geometry_sha256)?;
        let s = f.state;
        let mut state = G1Measurement {
            episode_id: s.episode_id,
            source_tick: s.source_tick,
            sim_time_ns: s.sim_time_ns,
            joint_positions: s.joint_positions,
            joint_velocities: s.joint_velocities,
            root_rotation_wxyz: s.root_rotation_wxyz,
            root_angular_velocity_body: s.root_angular_velocity_body,
            root_velocity_source: s.root_velocity_source,
        };
        let initial = f.command.clone();
        let mut motion = StaticObservedPlace::new(f.goal, &state, f.command, &geometry)?;
        let mut steps = Vec::new();
        let mut failure = None;
        for _ in 0..OBSERVED_PLACE_TICKS {
            match motion.update(&state, &fk) {
                Ok(step) => {
                    assert_eq!(
                        step.command.upper_positions[14..],
                        initial.upper_positions[14..]
                    );
                    assert_eq!(step.command.navigation, initial.navigation);
                    assert_eq!(step.command.pelvis_height, initial.pelvis_height);
                    steps.push(step);
                    state.source_tick += 1;
                    state.sim_time_ns += 20_000_000;
                }
                Err(e) => {
                    failure = Some(e.to_string());
                    break;
                }
            }
        }
        let summary = serde_json::json!({"schema":"g1_actual_fixed_pair_static_placement_joint_preflight_v1","geometry_ticks":steps.len(),"failure":failure,"physics_integrations":0,"model_calls":0,"fresh_images":0,"world_or_contact_truth_input":false,"task_qualified":false,"steps":steps});
        std::fs::write(
            f.output,
            serde_json::to_vec_pretty(&summary).map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        assert!(failure.is_none(), "{failure:?}");
        assert_eq!(steps.len(), 700);
        Ok(())
    }
}
