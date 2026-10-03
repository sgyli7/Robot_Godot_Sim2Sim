//! Finite source-scene RGB route with explicit standing during perception.
//! Receives the camera port and typed self/owner phase identities; never reads
//! task-object poses or contacts. Physical acceptance remains an independent
//! consumer of the full owner trace.

use super::*;
use simulation_minigame::g1::{
    mobile_assist::{MobileAssistCommand, MobileAssistExecution, MobileObservedSkill},
    mobile_hold::MobileHoldGoal,
    mobile_navigation::{MobileCarryGoal, MobileScanGoal},
    mobile_release::MobileReleaseGoal,
    worker::TimedCommand,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Grasp,
    Scan,
    Coarse,
    Fine,
    Placement,
    Finish,
}

pub(super) struct ContinuousMobileRoute {
    stage: Stage,
    vision: MarkerVisionConfiguration,
    search_heading: f32,
    expected_wait_request: u64,
    image_min_tick: Option<u64>,
    job: Option<MarkerVisionJob>,
    fine_goals: u32,
    images: u32,
    completed: bool,
}

impl ContinuousMobileRoute {
    pub(super) fn new(config: &MobileScanCaptureConfiguration) -> Result<Self, String> {
        let vision = config
            .vision
            .clone()
            .ok_or("continuous route requires the bound marker worker")?;
        vision.validate()?;
        Ok(Self {
            stage: Stage::Grasp,
            vision,
            search_heading: config.heading_yaw_source_rad,
            expected_wait_request: 4,
            image_min_tick: None,
            job: None,
            fine_goals: 0,
            images: 0,
            completed: false,
        })
    }
    pub(super) fn needs_auxiliary_view(&self) -> bool {
        self.stage != Stage::Grasp
    }
    pub(super) fn completed(&self) -> bool {
        self.completed
    }

    fn drive(
        &mut self,
        runtime: &mut CaptureRuntime,
        outcome: &CaptureOutcome,
        port: &G1CameraPort,
    ) -> Result<bool, String> {
        let latest = runtime
            .latest
            .as_ref()
            .ok_or("continuous route owner absent")?
            .clone();
        let tick = latest.timing.episode_integrations;
        if tick > u64::from(runtime.options.ticks) || latest.phase == G1WorkerPhase::Failed {
            return Err(latest
                .reason
                .clone()
                .unwrap_or("continuous route owner failed/budget exceeded".into()));
        }
        if tick > 0 && latest.phase == G1WorkerPhase::Paused {
            if self.stage==Stage::Finish && latest.assist_step.as_ref().is_some_and(|s| matches!(&s.execution,
                MobileAssistExecution::ClassicalModelWait {goal,waiting} if goal.request_id==self.expected_wait_request && waiting.completed)) {
                self.completed=true;
                if let Some(handoff)=&mut outcome.0.lock().unwrap().mobile_assist_handoff {handoff["completed_at_actual_tick"]=tick.into();handoff["completed"]=true.into();}
                return Ok(true);
            }
            return Err("continuous route exhausted a finite owner wait; explicit pause".into());
        }
        let ready=latest.assist_step.as_ref().is_some_and(|s| matches!(&s.execution,
            MobileAssistExecution::ClassicalModelWait {goal,waiting} if goal.request_id==self.expected_wait_request && waiting.observation_ready && !waiting.completed));
        if self.stage == Stage::Grasp {
            if runtime
                .live_policy
                .as_ref()
                .is_some_and(|p| p.submitted_chunks == 4)
                && ready
            {
                self.stage = Stage::Scan;
                outcome.0.lock().unwrap().mobile_assist_handoff = Some(serde_json::json!({
                    "mode":"continuous_source_rgb_classical_route_not_qualified","original_vla_grasp_chunks":4,
                    "marker_activation_requires_actual_fourth_chunk_completion":true,
                    "actual_grasp_wait_ready_tick":tick,"qwen_target_selection":false,"world_or_contact_truth_input":false,
                    "traditional_public_map_heading_rad":self.search_heading,"events":[],"task_qualified":false,
                }));
                return Ok(false);
            }
            drive_waited_mobile_policy(runtime, outcome, port)?;
            return Ok(false);
        }
        if self.stage == Stage::Finish || !ready {
            return Ok(false);
        }
        if let Some(job) = &self.job {
            let Some(reply) = job.try_take() else {
                return Ok(false);
            };
            let reply = reply?;
            let observation = job.observation;
            self.record(outcome,serde_json::json!({"event":"current_rgb_localization","stage":format!("{:?}",self.stage),"observation":observation,
                "arrival_display_tick":tick,"localization_wall_ms":job.started.elapsed().as_secs_f64()*1000.,
                "physics_paused_during_localization":false,"actual_localization":reply}))?;
            if reply["camera_mount_profile"] != "auxiliary_grip_overview"
                || observation.episode_id != runtime.episode_id
            {
                return Err("continuous marker reply has foreign camera/episode".into());
            }
            let skill = match self.stage {
                Stage::Coarse => {
                    let proposal = &reply["navigation_proposal"];
                    let distance = proposal["relative_distance_m"]
                        .as_f64()
                        .ok_or("coarse RGB distance absent")?
                        - 0.65;
                    if !(0.1..=1.85).contains(&distance) {
                        return Err("coarse RGB waypoint outside bounded range".into());
                    }
                    self.stage = Stage::Fine;
                    MobileObservedSkill::Carry(MobileCarryGoal {
                        observation,
                        heading_yaw_source_rad: proposal["heading_yaw_source_rad"]
                            .as_f64()
                            .ok_or("coarse RGB heading absent")?
                            as f32,
                        relative_distance_m: distance as f32,
                    })
                }
                Stage::Fine => {
                    if reply["clearance_proposal"]["required_raise_m"]
                        .as_f64()
                        .is_none_or(|r| r > 0.)
                    {
                        return Err("current fine RGB requires unsupported clearance adjustment; explicit pause".into());
                    }
                    match reply["fine_approach_proposal"]["state"]
                        .as_str()
                        .ok_or("fine RGB containment state absent")?
                    {
                        "aligned" => {
                            self.stage = Stage::Placement;
                            MobileObservedSkill::Hold(MobileHoldGoal { observation })
                        }
                        "advance" if self.fine_goals < 5 => {
                            let goal: MobileCarryGoal = serde_json::from_value(
                                reply["fine_approach_proposal"]["goal"].clone(),
                            )
                            .map_err(|e| e.to_string())?;
                            if goal.observation != observation || goal.relative_distance_m != 0.1 {
                                return Err(
                                    "fine RGB goal mutated its image or fixed short distance"
                                        .into(),
                                );
                            }
                            self.fine_goals += 1;
                            MobileObservedSkill::Carry(goal)
                        }
                        _ => {
                            return Err(
                                "fine RGB containment rejected or five-goal budget exhausted"
                                    .into(),
                            );
                        }
                    }
                }
                Stage::Placement => {
                    if reply["release_proposal"]["release_admitted"] != true {
                        return Err(
                            "current placement-only RGB envelope rejected; explicit pause".into(),
                        );
                    }
                    let goal: MobileReleaseGoal =
                        serde_json::from_value(reply["release_proposal"]["release_goal"].clone())
                            .map_err(|e| e.to_string())?;
                    if goal.observation != observation {
                        return Err("release proposal changed its actual RGB identity".into());
                    }
                    self.stage = Stage::Finish;
                    MobileObservedSkill::Release(goal)
                }
                _ => return Err("unexpected continuous localization phase".into()),
            };
            self.submit(runtime, outcome, skill)?;
            self.job.take();
            return Ok(false);
        }
        let Some((observation, directory)) = self.capture(runtime, port)? else {
            return Ok(false);
        };
        if self.stage == Stage::Scan {
            self.submit(
                runtime,
                outcome,
                MobileObservedSkill::Scan(MobileScanGoal {
                    observation,
                    heading_yaw_source_rad: self.search_heading,
                }),
            )?;
            self.stage = Stage::Coarse;
        } else {
            self.job = Some(if self.stage == Stage::Placement {
                MarkerVisionJob::start_placement_view(self.vision.clone(), directory, observation)?
            } else {
                MarkerVisionJob::start(self.vision.clone(), directory, observation)?
            });
        }
        Ok(false)
    }

    fn submit(
        &mut self,
        runtime: &CaptureRuntime,
        outcome: &CaptureOutcome,
        skill: MobileObservedSkill,
    ) -> Result<(), String> {
        let observation = match &skill {
            MobileObservedSkill::Carry(g) => g.observation,
            MobileObservedSkill::Scan(g) => g.observation,
            MobileObservedSkill::Hold(g) => g.observation,
            MobileObservedSkill::Release(g) => g.observation,
        };
        match &skill {
            MobileObservedSkill::Carry(g) => g.validate(),
            MobileObservedSkill::Scan(g) => g.validate(),
            MobileObservedSkill::Hold(g) => g.validate(),
            MobileObservedSkill::Release(g) => g.validate(),
        }
        .map_err(|e| e.to_string())?;
        let command = MobileAssistCommand::ObservedSkillThenWait(skill);
        let CaptureWorker::AssistedMobile(owner) = &runtime.worker else {
            return Err("continuous skill lost its unique owner".into());
        };
        owner
            .submit(TimedCommand {
                episode_id: runtime.episode_id,
                valid_until_sim_ns: u64::from(runtime.options.ticks) * ARENA_ACTION_PERIOD_NS,
                valid_until_wall: Instant::now() + Duration::from_secs(45),
                command,
            })
            .map_err(|e| e.to_string())?;
        self.expected_wait_request = observation.frame_id;
        self.record(outcome,serde_json::json!({"event":"observed_skill_submitted","next_stage":format!("{:?}",self.stage),"observation":observation,
            "arrival_display_tick":runtime.latest.as_ref().unwrap().timing.episode_integrations,"execution_start":"actual_owner_step_image_admission",
            "original_image_restamped":false,"following_traditional_wait_maximum_ticks":200,"world_or_contact_truth_input":false}))
    }

    fn capture(
        &mut self,
        runtime: &CaptureRuntime,
        port: &G1CameraPort,
    ) -> Result<Option<(ObservationStamp, PathBuf)>, String> {
        let tick = runtime.latest.as_ref().unwrap().timing.episode_integrations;
        if self.image_min_tick.is_none() {
            if self.images >= 12 {
                return Err("continuous current-image budget exhausted".into());
            }
            port.request_physics_frame(runtime.episode_id, tick)?;
            self.image_min_tick = Some(tick);
            return Ok(None);
        }
        let Some(frame) = port.take() else {
            return Ok(None);
        };
        let frame = frame?;
        let image_tick = frame.stamp.source_ticks[0];
        if frame.stamp.source != CameraPoseSource::PhysicsBody
            || frame.stamp.mount_profile != G1CameraMountProfile::AuxiliaryGripOverview
            || frame.stamp.episode_id != runtime.episode_id
            || frame.stamp.source_ticks != [image_tick; 2]
            || frame.stamp.sim_time_ns != image_tick * ARENA_ACTION_PERIOD_NS
            || image_tick < self.image_min_tick.unwrap()
            || image_tick > tick
        {
            return Err(
                "continuous current RGB is old, foreign or incoherent; no restamping".into(),
            );
        }
        frame
            .stamp
            .native_state
            .as_ref()
            .ok_or("continuous RGB self state absent")?
            .validate()?;
        let input = marker_observation(&frame.stamp)?;
        let observation: ObservationStamp =
            serde_json::from_value(input["stamp"].clone()).map_err(|e| e.to_string())?;
        let directory = runtime
            .options
            .output
            .join(format!("continuous_{:?}_{:02}", self.stage, self.images).to_lowercase());
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let rgb = task_minigame::decision::CameraRgb::from_rgb(
            "native_ego",
            frame.width,
            frame.height,
            frame.rgb,
        )
        .map_err(|e| e.to_string())?;
        fs::write(directory.join("ego.png"), rgb.png()).map_err(|e| e.to_string())?;
        fs::write(
            directory.join("observation.json"),
            serde_json::to_vec_pretty(&input).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            directory.join("audit_stamp.json"),
            serde_json::to_vec_pretty(&frame.stamp).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        self.images += 1;
        self.image_min_tick = None;
        Ok(Some((observation, directory)))
    }
    fn record(&self, outcome: &CaptureOutcome, event: serde_json::Value) -> Result<(), String> {
        let mut receipt = outcome.0.lock().unwrap();
        let events = receipt
            .mobile_assist_handoff
            .as_mut()
            .ok_or("continuous provenance absent")?["events"]
            .as_array_mut()
            .ok_or("continuous event array absent")?;
        if events.len() >= 32 {
            return Err("continuous provenance budget exhausted".into());
        }
        events.push(event);
        Ok(())
    }
}

pub(super) fn drive(
    runtime: &mut CaptureRuntime,
    outcome: &CaptureOutcome,
    port: &G1CameraPort,
) -> Result<bool, String> {
    let mut route = runtime
        .continuous_route
        .take()
        .ok_or("continuous route absent")?;
    let result = route.drive(runtime, outcome, port);
    runtime.continuous_route = Some(route);
    result
}
