//! Typed handoff from original mobile VLA chunks to disclosed traditional grip
//! and self-state navigation on the same sole native physics owner.

use std::time::{SystemTime, UNIX_EPOCH};

use robot_minigame::{
    RobotError,
    g1::{contract::G1Command, definition::G1Definition},
};
use serde::Serialize;
use task_minigame::{
    policy::{ActionLimits, profile_contract},
    types::TaskProfile,
};

use super::{
    mobile_grip::{MobileGripCalibration, MobileGripReceipt},
    mobile_navigation::{MobileCarryGoal, MobileCarryNavigator, MobileNavigationStep},
    runner::{G1Measurement, G1ProgressCounts},
    task_objects::TaskObjectFrame,
    task_policy::{ArenaControllerCommand, controller_command},
    task_runner::{
        ArenaBodyStep, ArenaTaskBodyConfig, ArenaTaskCommand, ArenaTaskExecution, ArenaTaskRunner,
        ArenaTaskRunnerConfig,
    },
};

#[derive(Clone, Debug)]
pub enum MobileAssistCommand {
    OriginalVla(ArenaTaskCommand),
    ClassicalCarry(MobileCarryGoal),
}

impl MobileAssistCommand {
    pub(crate) fn validate(&self) -> Result<(), RobotError> {
        match self {
            Self::OriginalVla(command) => {
                command.validate()?;
                if command.chunk.profile != TaskProfile::MobileBox {
                    return Err(invalid("assisted mobile owner rejects static VLA"));
                }
                Ok(())
            }
            Self::ClassicalCarry(goal) => goal.validate(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MobileAssistExecution {
    OriginalVla(ArenaTaskExecution),
    ClassicalCarry {
        goal: MobileCarryGoal,
        grip: MobileGripReceipt,
        navigation: MobileNavigationStep,
        command: G1Command,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct MobileAssistStep {
    pub execution: MobileAssistExecution,
    pub body: ArenaBodyStep,
}

/// No second world, physical pose setter, task-object sensor or model client.
pub struct MobileAssistRunner {
    owner: ArenaTaskRunner,
    calibration: MobileGripCalibration,
    limits: ActionLimits,
    maximum_observation_wall_age_ms: u64,
    last_vla_execution: Option<ArenaTaskExecution>,
    last_vla_command: Option<G1Command>,
    carry: Option<CarryState>,
    halted: bool,
}

struct CarryState {
    navigator: MobileCarryNavigator,
    command: G1Command,
    grip: MobileGripReceipt,
}

impl MobileAssistRunner {
    /// Opt-in factory only: original complete T2 scene/50Hz/native motors with
    /// the already measured4PGS candidate. Normal task loading is unchanged.
    pub fn load(config: &ArenaTaskRunnerConfig) -> Result<Self, RobotError> {
        let ArenaTaskBodyConfig::MobileHomieV2(body) = &config.body else {
            return Err(invalid("mobile assist requires Homie_v2/N1.6"));
        };
        let definition = G1Definition::load(&body.definition, &body.definition_sha256)?;
        Ok(Self {
            owner: ArenaTaskRunner::load_mobile_constraint_diagnostic(config)?,
            calibration: MobileGripCalibration::new(&definition)?,
            limits: config.limits.clone(),
            maximum_observation_wall_age_ms: config.max_observation_wall_age_ms,
            last_vla_execution: None,
            last_vla_command: None,
            carry: None,
            halted: false,
        })
    }
    pub fn measurement(&self) -> Result<G1Measurement, RobotError> {
        self.owner.measurement()
    }
    pub fn initial_frame(&self) -> Result<robot_minigame::g1::definition::G1BodyFrame, RobotError> {
        self.owner.initial_frame()
    }
    pub fn task_object_frame(&self) -> Result<Option<TaskObjectFrame>, RobotError> {
        self.owner.task_object_frame()
    }
    pub fn progress_counts(&self) -> G1ProgressCounts {
        let mut counts = self.owner.progress_counts();
        counts.halted |= self.halted;
        counts
    }
    pub fn completed_carry(&self) -> bool {
        self.carry
            .as_ref()
            .is_some_and(|carry| carry.navigator.completed())
    }

    pub fn step_with_guard(
        &mut self,
        command: &MobileAssistCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<MobileAssistStep, RobotError> {
        if self.halted {
            return Err(invalid("mobile assist halted; reset required"));
        }
        let result = self.step_inner(command, guard);
        if result.is_err() {
            self.halted = true;
        }
        result
    }
    fn step_inner(
        &mut self,
        command: &MobileAssistCommand,
        guard: &mut dyn FnMut() -> Result<(), RobotError>,
    ) -> Result<MobileAssistStep, RobotError> {
        guard()?;
        command.validate()?;
        match command {
            MobileAssistCommand::OriginalVla(command) => {
                if self
                    .carry
                    .as_ref()
                    .is_some_and(|carry| !carry.navigator.completed())
                {
                    return Err(invalid(
                        "cannot replace active classical carry with a VLA chunk",
                    ));
                }
                let step = self.owner.step_with_guard(command, guard)?;
                let admitted = controller_command(
                    TaskProfile::MobileBox,
                    &command.chunk.frames[step.execution.frame_index],
                    &self.limits,
                )
                .map_err(|e| invalid(format!("{e:?}")))?;
                let ArenaControllerCommand::MobileHomieV2(original) = admitted.controller else {
                    return Err(invalid("foreign controller in mobile handoff"));
                };
                self.last_vla_command = Some(original);
                self.last_vla_execution = Some(step.execution.clone());
                self.carry = None;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::OriginalVla(step.execution),
                    body: step.body,
                })
            }
            MobileAssistCommand::ClassicalCarry(goal) => {
                let state = self.owner.measurement()?;
                if self.carry.is_none() {
                    let previous = self
                        .last_vla_execution
                        .as_ref()
                        .ok_or_else(|| invalid("carry has no original VLA predecessor"))?;
                    if previous.frame_index + 1
                        != profile_contract(TaskProfile::MobileBox).action_horizon
                    {
                        return Err(invalid(
                            "carry handoff requires an original complete VLA chunk",
                        ));
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_err(|e| invalid(e.to_string()))?
                        .as_millis() as u64;
                    let age = now
                        .checked_sub(goal.observation.captured_at_unix_ms)
                        .ok_or_else(|| invalid("carry observation is from a future wall clock"))?;
                    if age > self.maximum_observation_wall_age_ms {
                        return Err(invalid("carry observation expired before owner handoff"));
                    }
                    let navigator = MobileCarryNavigator::new(goal.clone(), &state)?;
                    let original = self
                        .last_vla_command
                        .as_ref()
                        .ok_or_else(|| invalid("carry lost its original upper command"))?;
                    let correction = self.calibration.correct(&state, original)?;
                    self.carry = Some(CarryState {
                        navigator,
                        command: correction.command,
                        grip: correction.receipt,
                    });
                }
                let carry = self.carry.as_mut().unwrap();
                if carry.navigator.goal() != goal {
                    return Err(invalid(
                        "carry goal changed while executing; stop/reset required",
                    ));
                }
                let navigation = carry.navigator.update(&state)?;
                carry.command.navigation = navigation.navigation;
                let body = self
                    .owner
                    .step_mobile_assist_with_guard(&carry.command, guard)?;
                Ok(MobileAssistStep {
                    execution: MobileAssistExecution::ClassicalCarry {
                        goal: goal.clone(),
                        grip: carry.grip.clone(),
                        navigation,
                        command: carry.command.clone(),
                    },
                    body: ArenaBodyStep::MobileHomieV2(Box::new(body)),
                })
            }
        }
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use serde::Deserialize;
    use std::{fs, io::Write, path::Path, sync::Arc};
    use task_minigame::{policy::PolicyActionChunk, types::ObservationStamp};
    #[derive(Deserialize)]
    struct Sequence {
        schema: String,
        chunks: Vec<PolicyActionChunk>,
    }
    fn read(name: &str) -> Result<Vec<u8>, RobotError> {
        let path = std::env::var(name).map_err(|e| invalid(e.to_string()))?;
        let hash = std::env::var(format!("{name}_SHA256")).map_err(|e| invalid(e.to_string()))?;
        bound_bytes(Path::new(&path), &hash)
    }
    #[test]
    #[ignore = "requires frozen mobileconfig/four saved native replies/new output; <=2050real50Hzsteps,0VLA"]
    fn real_mobile_typed_assist_diagnostic() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
        {
            return Err(invalid(
                "typedassist diagnostic requires exactlyfour native reply fixtures",
            ));
        }
        let path = std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut receipt = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| invalid(e.to_string()))?;
        let mut trace = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&path).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let mut owner = MobileAssistRunner::load(&config)?;
        let mut current = None;
        let mut actual_ticks = 0;
        let mut minimum_upright = 1_f32;
        let mut last_root = None;
        let mut walk_origin = None;
        let mut final_displacement = 0_f32;
        let mut maximum_walk_displacement = 0_f32;
        let result = (|| -> Result<(), RobotError> {
            for tick in 1..=2050 {
                if tick <= 200 && (tick - 1) % 50 == 0 {
                    let mut chunk = sequence.chunks[((tick - 1) / 50) as usize].clone();
                    // Offline wall-admission rebinding only. Original raw
                    // fixture stamps/action bytes remain in the bound input.
                    chunk.observation.captured_at_unix_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_err(|e| invalid(e.to_string()))?
                        .as_millis()
                        as u64;
                    current = Some(MobileAssistCommand::OriginalVla(ArenaTaskCommand {
                        chunk: Arc::new(chunk),
                        scheduled_start_sim_ns: None,
                    }));
                } else if tick == 201 {
                    let state = owner.measurement()?;
                    current = Some(MobileAssistCommand::ClassicalCarry(MobileCarryGoal {
                        observation: ObservationStamp {
                            episode_id: state.episode_id,
                            frame_id: 5,
                            sim_time_ns: state.sim_time_ns,
                            captured_at_unix_ms: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map_err(|e| invalid(e.to_string()))?
                                .as_millis()
                                as u64,
                        },
                        heading_yaw_source_rad: std::f32::consts::FRAC_PI_2,
                        relative_distance_m: 2.,
                    }));
                }
                let step = owner.step_with_guard(current.as_ref().unwrap(), &mut || Ok(()))?;
                actual_ticks = tick;
                let ArenaBodyStep::MobileHomieV2(body) = &step.body else {
                    return Err(invalid("typed mobile diagnostic changed body"));
                };
                if body.integration_count != tick
                    || body.inference.inference_count != tick
                    || body.torque_update_count != tick
                    || body.step_configuration.dt != 0.02
                    || body.step_configuration.physics_hz != 50
                    || body.step_configuration.num_solver_iterations != 1
                    || body.step_configuration.num_internal_pgs_iterations != 4
                    || body.step_configuration.max_ccd_substeps != 1
                {
                    return Err(invalid("typed mobile owner changed counts/timing"));
                }
                minimum_upright = minimum_upright.min(body.root_upright_cosine);
                let (phase, command) = match &step.execution {
                    MobileAssistExecution::OriginalVla(_) => ("saved_grasp_fixture", None),
                    MobileAssistExecution::ClassicalCarry {
                        navigation,
                        command,
                        ..
                    } => {
                        let phase = match navigation.phase {
                            super::super::mobile_navigation::MobileNavigationPhase::Turn => {
                                "feedback_fixed_grip_turn"
                            }
                            super::super::mobile_navigation::MobileNavigationPhase::Walk => {
                                "feedback_fixed_grip_walk"
                            }
                            super::super::mobile_navigation::MobileNavigationPhase::Stop => {
                                "feedback_fixed_grip_stop"
                            }
                        };
                        (phase, Some(command))
                    }
                };
                if phase == "feedback_fixed_grip_walk" && walk_origin.is_none() {
                    walk_origin = last_root;
                }
                if let Some(origin) = walk_origin {
                    let origin: [f32; 3] = origin;
                    final_displacement = (body.root_position_source[0] - origin[0])
                        .hypot(body.root_position_source[1] - origin[1]);
                    if phase == "feedback_fixed_grip_walk" {
                        maximum_walk_displacement =
                            maximum_walk_displacement.max(final_displacement);
                    }
                }
                last_root = Some(body.root_position_source);
                serde_json::to_writer(&mut trace,&serde_json::json!({"phase":phase,"manual_body_command":command,"body":step.body,"execution":step.execution,"diagnostic_saved_grasp_fixture":true,"autonomous_execution":false})).map_err(|e|invalid(e.to_string()))?;
                writeln!(trace).map_err(|e| invalid(e.to_string()))?;
                if minimum_upright < 0.95 {
                    return Err(invalid("independent diagnostic fall guard"));
                }
                if owner.completed_carry() {
                    break;
                }
            }
            Ok(())
        })();
        trace.flush().map_err(|e| invalid(e.to_string()))?;
        let passed = result.is_ok()
            && owner.completed_carry()
            && maximum_walk_displacement >= 2.
            && final_displacement >= 2.;
        serde_json::to_writer_pretty(&mut receipt,&serde_json::json!({"schema":"g1_mobile_typed_assist_diagnostic_v1","qualified":false,"autonomous_execution":false,"task_success_verified":false,"actual_vla_inferences":0,"original_saved_reply_count":4,"actual_ticks":actual_ticks,"counts":owner.progress_counts(),"physics_hz":50,"integrations_per_tick":1,"completed_carry":owner.completed_carry(),"minimum_upright":minimum_upright,"maximum_walk_displacement_m":maximum_walk_displacement,"final_displacement_m":final_displacement,"mechanical_only_passed":passed,"error":result.as_ref().err().map(ToString::to_string),"scope":"savednativegrasp/runtimeclassicalgrip/typedselfstatenav;notliveRGB,Qwen,release,or1x"})).map_err(|e|invalid(e.to_string()))?;
        receipt.flush().map_err(|e| invalid(e.to_string()))?;
        result?;
        if !passed {
            return Err(invalid("typed mechanical carry didnotmeet2mwalkandstop"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod worker_diagnostic {
    use super::super::worker::{G1WorkerPhase, MobileAssistWorker, TimedCommand};
    use super::*;
    use robot_minigame::g1::policy::bound_bytes;
    use serde::Deserialize;
    use std::{
        fs,
        io::Write,
        path::Path,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };
    use task_minigame::{policy::PolicyActionChunk, types::ObservationStamp};
    #[derive(Deserialize)]
    struct Sequence {
        schema: String,
        chunks: Vec<PolicyActionChunk>,
    }
    fn read(name: &str) -> Result<Vec<u8>, RobotError> {
        bound_bytes(
            Path::new(&std::env::var(name).map_err(|e| invalid(e.to_string()))?),
            &std::env::var(format!("{name}_SHA256")).map_err(|e| invalid(e.to_string()))?,
        )
    }
    #[test]
    #[ignore = "requires frozenmobileconfig/fourreplies/newoutput;one real50Hz backgroundhandoff,<=2050Ticks/50s/0VLA"]
    fn real_mobile_typed_worker_handoff() -> Result<(), RobotError> {
        let config: ArenaTaskRunnerConfig =
            serde_json::from_slice(&read("G1_MOBILE_REPLAY_CONFIG")?)
                .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = serde_json::from_slice(&read("G1_MOBILE_REPLAY_ACTIONS")?)
            .map_err(|e| invalid(e.to_string()))?;
        if sequence.schema != "g1_saved_native_mobile_action_sequence_v1"
            || sequence.chunks.len() != 4
        {
            return Err(invalid("typedworkerrequiresfourboundednativefixtures"));
        }
        let path = std::env::var("G1_MOBILE_REPLAY_OUTPUT").map_err(|e| invalid(e.to_string()))?;
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|e| invalid(e.to_string()))?;
        let mut trace_file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(Path::new(&path).with_extension("jsonl"))
            .map_err(|e| invalid(e.to_string()))?;
        let worker = MobileAssistWorker::spawn_mobile_assist(config.clone())?;
        let trace = worker.subscribe_steps(2048)?;
        let started = Instant::now();
        let mut latest = None;
        let mut next_chunk = 0;
        let mut goal_submitted = false;
        let mut completed_at = None;
        let mut observed_steps = 0_u64;
        let result = (|| -> Result<(), RobotError> {
            loop {
                for record in trace.drain() {
                    observed_steps += 1;
                    let (phase, command) = match &record.step.execution {
                        MobileAssistExecution::OriginalVla(_) => ("saved_grasp_fixture", None),
                        MobileAssistExecution::ClassicalCarry {
                            navigation,
                            command,
                            ..
                        } => {
                            let phase = match navigation.phase {
                                super::super::mobile_navigation::MobileNavigationPhase::Turn => {
                                    "feedback_fixed_grip_turn"
                                }
                                super::super::mobile_navigation::MobileNavigationPhase::Walk => {
                                    "feedback_fixed_grip_walk"
                                }
                                super::super::mobile_navigation::MobileNavigationPhase::Stop => {
                                    "feedback_fixed_grip_stop"
                                }
                            };
                            (phase, Some(command))
                        }
                    };
                    serde_json::to_writer(&mut trace_file,&serde_json::json!({"phase":phase,"manual_body_command":command,"body":record.step.body,"execution":record.step.execution,"episode_id":record.episode_id,"generation":record.generation,"timing":record.timing,"diagnostic_saved_grasp_fixture":true,"autonomous_execution":false})).map_err(|e|invalid(e.to_string()))?;
                    writeln!(trace_file).map_err(|e| invalid(e.to_string()))?;
                }
                if let Some(snapshot) = worker.take_latest() {
                    latest = Some(snapshot);
                }
                if let Some(snapshot) = &latest {
                    if snapshot.phase == G1WorkerPhase::Failed {
                        return Err(invalid(format!(
                            "backgroundownerfailed: {:?}",
                            snapshot.reason
                        )));
                    }
                    if snapshot.phase == G1WorkerPhase::Paused && snapshot.frame.is_some() {
                        let ticks = snapshot.timing.episode_integrations;
                        if next_chunk<4 && ticks==next_chunk as u64*50 {
                            let mut chunk=sequence.chunks[next_chunk].clone();
                            chunk.observation.captured_at_unix_ms=SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|invalid(e.to_string()))?.as_millis() as u64;
                            worker.submit(TimedCommand {episode_id:config.body.episode_id(),valid_until_sim_ns:(next_chunk as u64+1)*1_000_000_000,valid_until_wall:Instant::now()+Duration::from_secs(2),command:MobileAssistCommand::OriginalVla(ArenaTaskCommand {chunk:Arc::new(chunk),scheduled_start_sim_ns:None})})?;
                            next_chunk+=1;
                        } else if next_chunk==4 && ticks==200 && !goal_submitted {
                            let goal=MobileCarryGoal {observation:ObservationStamp {episode_id:snapshot.episode_id,frame_id:5,sim_time_ns:4_000_000_000,captured_at_unix_ms:SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|invalid(e.to_string()))?.as_millis() as u64},heading_yaw_source_rad:std::f32::consts::FRAC_PI_2,relative_distance_m:2.};
                            worker.submit(TimedCommand {episode_id:snapshot.episode_id,valid_until_sim_ns:41_000_000_000,valid_until_wall:Instant::now()+Duration::from_secs(40),command:MobileAssistCommand::ClassicalCarry(goal)})?;goal_submitted=true;
                        } else if goal_submitted && snapshot.step.as_ref().is_some_and(|step|matches!(&step.execution,MobileAssistExecution::ClassicalCarry {navigation,..} if navigation.completed)) {
                            let entry=completed_at.get_or_insert((Instant::now(),ticks));
                            if ticks!=entry.1 {return Err(invalid("completed background skill performed an extra integration"));}
                            if entry.0.elapsed()>=Duration::from_secs(1){break;}
                        }
                    }
                }
                if started.elapsed() > Duration::from_secs(50) {
                    return Err(invalid("finite background handoff deadline missed"));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Ok(())
        })();
        trace_file.flush().map_err(|e| invalid(e.to_string()))?;
        let timing = latest.as_ref().map(|s| s.timing.clone());
        let dropped = trace.dropped_records();
        let counts = timing.as_ref().map(|t| t.episode_integrations).unwrap_or(0);
        let passed =
            result.is_ok() && completed_at.is_some() && dropped == 0 && observed_steps == counts;
        serde_json::to_writer_pretty(&mut file,&serde_json::json!({"schema":"g1_mobile_typed_background_handoff_v1","qualified":false,"autonomous_execution":false,"task_success_verified":false,"formal_one_x_qualified":false,"actual_vla_inferences":0,"original_saved_reply_count":4,"actual_ticks":counts,"observed_completed_steps":observed_steps,"trace_dropped":dropped,"timing":timing,"paused_after_completed_skill":completed_at.is_some(),"one_second_after_completion_adds_no_ticks":completed_at.is_some(),"background_handoff_only_passed":passed,"error":result.as_ref().err().map(ToString::to_string),"wall_seconds_including_load_pause_evidence":started.elapsed().as_secs_f64(),"scope":"savednativegrasp/actualbackgroundclock/typedclassicalcarry;notliveRGB,render/Qwen/release/formal1x"})).map_err(|e|invalid(e.to_string()))?;
        file.flush().map_err(|e| invalid(e.to_string()))?;
        worker.shutdown()?;
        result?;
        if !passed {
            return Err(invalid("backgroundcarrydidnotcompletewithfulltrace"));
        }
        Ok(())
    }
}
