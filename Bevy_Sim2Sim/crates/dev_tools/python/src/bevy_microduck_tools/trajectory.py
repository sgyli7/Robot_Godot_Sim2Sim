"""Lossless-in-scope source capture, including the true terminal before reset."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from .serialization import identity, sha256_file, write_json
from .source_adapter import JOINT_ORDER, SKILL_TASKS
from .timing import delay_diagnostics, install_delays
from .provenance import describe_origin, model_state, restored_origin
from .workflow import Rejection, verify_claimed_learning, verify_claimed_runtime, require_verified_runtime


def _capture_source(cfg, adoption: dict, source: dict, output: Path, *, steps: int | None = 60,
                   device: str = "cuda:0", make_runner: bool = True, video: bool = False,
                   learning_iterations: int = 0, resume: Path | None = None,
                   resume_origin: Path | None = None, _learning_binding: dict | None = None, _verified_runtime=None,
                   _profile_manifest: dict | None = None, _profile_token=None, _phase_journal=None,
                   _profile_manifest_path: Path | None = None) -> dict:
    if learning_iterations:
        verify_claimed_learning(_learning_binding, iterations=learning_iterations, seed=cfg.seed, checkpoint=resume)
        if _verified_runtime is None:
            _verified_runtime=verify_claimed_runtime(_learning_binding,Path(source["root"]))
        require_verified_runtime(_verified_runtime,_learning_binding,Path(source["root"]))
    if _profile_manifest is not None:
        from .profile_identity import require_zero_token
        if learning_iterations != 0 or steps is not None or not video or resume is None or resume_origin is None or _phase_journal is None:
            raise Rejection("Source profile discovery must be full-natural zero-update restored inference with video")
        if _profile_manifest_path is None or _profile_token.runtime_verified is not True:
            raise Rejection("Actual discovered source requires fully verified manifest bytes")
        require_zero_token(_profile_token, _profile_manifest, manifest_path=_profile_manifest_path)
        _phase_journal.mark("source_capture_begin", source_commit=source["commit"], adoption_sha256=adoption["sha256"])
    import mujoco
    import mujoco_warp as mjwarp
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    hooks = None
    producer = None
    if _profile_manifest is not None:
        from .profile_observer import ProducerJournal, warp_observation
        producer = ProducerJournal()
        hooks = warp_observation(_phase_journal, producer)
        hooks.__enter__()
        _phase_journal.mark("env_initialization", device=device)

    if cfg.scene.num_envs != 1 or (steps is not None and steps < 1):
        raise ValueError("Diagnostic capture requires one world and at least one policy step")
    origin_input = restored_origin(resume_origin, resume) if resume_origin is not None and resume is not None else None
    if resume_origin is not None and resume is None:
        raise ValueError("Origin provided without a checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    full_episode_requested = steps is None
    steps = env.max_episode_length + 1 if steps is None else steps
    install_delays(env, adoption["timing_plan"])
    env.sim.expand_model_fields(("dof_frictionloss", "dof_damping"))
    env.sim.create_graph()
    if _phase_journal is not None:
        properties = torch.cuda.get_device_properties(device)
        _phase_journal.mark("graph_creation", physics_hz=60, policy_hz=60, source_substeps=cfg.decimation,
                            platform={"kernel_release": __import__("os").uname().release,
                                      "gpu_name": properties.name, "gpu_compute_capability": [properties.major, properties.minor],
                                      "gpu_total_memory": properties.total_memory,
                                      "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                                      "torch_cuda_version": torch.version.cuda,
                                      "mujoco_version": mujoco.__version__,
                                      "driver_internal_bytes": "opaque_platform_input_not_content_hashed"})
    model = env.sim.mj_model
    cpu_data = mujoco.MjData(model)
    names = lambda kind, count: [mujoco.mj_id2name(model, kind, i) for i in range(count)]
    body_order = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_order = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_order = names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if tuple(name.split("/")[-1] for name in actuator_order) != JOINT_ORDER:
        raise ValueError("Compiled scene actuator mapping differs from canonical order")
    raw_path = output / "source_trajectory.jsonl"
    integration_count = 0
    observation = None
    controller = []
    records = 0
    terminal_reason = None
    learning_active = False
    learning_elapsed = 0.0
    timing_checks = {}
    video_report = None
    policy_trace = None
    policy_call_count = 0
    checkpoint_origin = None
    frozen_before = None
    frozen_after = None
    episode_epoch = -1
    episode_origin_tick = 0
    reset_in_progress = False
    native_force_measurement = {"phase": "environment_initialization_before_capture",
                                "episode_epoch": None, "physics_tick": None,
                                "executed_integration_interval": None}
    termination_measurement = {"phase": "manager_initialization", "episode_epoch": None,
                               "physics_tick": None}
    recorder = None
    if video:
        from .video import SourceVideo
        root_body = next(i for i, name in enumerate(body_order) if name and name.endswith("trunk_base"))
        recorder = SourceVideo(model, output / "source_video.mp4", physics_dt=env.physics_dt, root_body=root_body)

    def first_world(value):
        if hasattr(value, "_tensor"):
            value = value[...]
        return value.detach().cpu().tolist()[0]

    stream = raw_path.open("w")
    def emit(value):
        nonlocal records
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        records += 1

    emit({"kind": "header", "schema": "microduck_4d_raw_v2", "engine": "mujoco_warp",
          "skill": adoption["skill"], "adoption_sha256": adoption["sha256"],
          "body_order": body_order, "joint_order": joint_order, "actuator_order": actuator_order,
          "canonical_actuator_order": list(JOINT_ORDER), "control_dt": env.step_dt,
          "integration_dt": env.physics_dt, "source_substeps": cfg.decimation,
          "seed": cfg.seed, "auto_reset": False, "units": "SI; quaternions wxyz; Z-up",
          "full_episode_requested": full_episode_requested,
          "actual_max_episode_length": env.max_episode_length,
          "missing_channels": ["target Rapier state"] + ([] if video else ["real video (capture not rendered)"]),
          "qualification": "bounded source diagnostic; no skill certification",
          "measurement_semantics": "Only physics_step has an executed integration interval/impulse. Forward-only snapshots carry instantaneous native forces. Manager terminal results and last inference are explicitly stamped caches.",
          "learning_iterations": learning_iterations, "resume_checkpoint": str(resume.resolve()) if resume else None,
          "checkpoint_origin_input": origin_input,
          "initialization": "restored_checkpoint" if resume is not None else "fresh_initialization"})

    def capture(kind, *, reason=None):
        mjwarp.get_data_into(cpu_data, model, env.sim.wp_data, world_id=0)
        contacts = []
        for index in range(cpu_data.ncon):
            contact = cpu_data.contact[index]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, cpu_data, index, wrench)
            contacts.append({"geom_ids": contact.geom.tolist(), "position": contact.pos.tolist(),
                             "frame": contact.frame.tolist(), "distance": float(contact.dist),
                             "force_contact_frame": wrench.tolist(),
                             "impulse_contact_frame": (wrench * env.physics_dt).tolist() if kind == "physics_step" else None,
                             "efc_address": int(contact.efc_address)})
        # Recompute body/geom poses from post-integration qpos, without re-solving forces.
        mujoco.mj_kinematics(model, cpu_data)
        mujoco.mj_comPos(model, cpu_data)
        mujoco.mj_comVel(model, cpu_data)
        body_velocities = []
        for body_id in range(model.nbody):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, cpu_data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            body_velocities.append(velocity.tolist())
        robot = env.scene["robot"].data
        frame = {"kind": kind, "time_seconds": integration_count * env.physics_dt,
                 "physics_tick": integration_count, "policy_tick": int(env.common_step_counter),
                 "episode_epoch": episode_epoch, "episode_physics_tick": integration_count - episode_origin_tick,
                 "episode_time_seconds": (integration_count - episode_origin_tick) * env.physics_dt,
                 "qpos": cpu_data.qpos.tolist(), "qvel": cpu_data.qvel.tolist(),
                 "body_positions": cpu_data.xpos.tolist(), "body_quaternions": cpu_data.xquat.tolist(),
                 "body_velocities_angular_linear_world": body_velocities,
                 "geom_positions": cpu_data.geom_xpos.tolist(), "geom_matrices": cpu_data.geom_xmat.tolist(),
                 "raw_action": first_world(env.action_manager.action),
                 "joint_targets": first_world(robot.joint_pos_target),
                 "joint_positions": first_world(robot.joint_pos), "joint_velocities": first_world(robot.joint_vel),
                 "qfrc_actuator_applied": cpu_data.qfrc_actuator.tolist(),
                 "qfrc_constraint": cpu_data.qfrc_constraint.tolist(), "qfrc_bias": cpu_data.qfrc_bias.tolist(),
                 "ctrl_motor_torque": cpu_data.ctrl.tolist(), "contacts": contacts,
                 "contact_interval": [(integration_count - 1) * env.physics_dt, integration_count * env.physics_dt] if kind == "physics_step" else None,
                 "contact_measurement": {"phase": "executed_integration_solve" if kind == "physics_step" else "instantaneous_snapshot_no_integration",
                                         "native_source": native_force_measurement.copy(),
                                         "body_poses": "kinematics refreshed from current qpos, without re-solving contacts"},
                 "native_force_measurement": native_force_measurement.copy(),
                 "commands": {name: first_world(env.command_manager.get_command(name)) for name in env.cfg.commands},
                 "termination_terms": {name: bool(env.termination_manager.get_term(name)[0]) for name in env.cfg.terminations},
                 "termination_measurement": {**termination_measurement,
                                             "belongs_to_current_episode": termination_measurement["episode_epoch"] == episode_epoch,
                                             "computed_at_current_physics_tick": termination_measurement["physics_tick"] == integration_count},
                 "objects": {name: {"position": first_world(entity.data.root_link_pos_w),
                                    "quaternion": first_world(entity.data.root_link_quat_w),
                                    "linear_velocity": first_world(entity.data.root_link_lin_vel_w),
                                    "angular_velocity": first_world(entity.data.root_link_ang_vel_w)}
                             for name, entity in env.scene.entities.items() if name != "robot"},
                 "controller_inputs_and_outputs": controller.copy(),
                 "reason": reason, "input_events": [],
                 "policy_input_and_output_at_call": policy_trace if kind == "policy_call" else None,
                 "last_policy_call": policy_trace,
                 "policy_call_for_this_integration": policy_trace["call_id"] if kind == "physics_step" and policy_trace is not None and policy_trace["episode_epoch"] == episode_epoch else None,
                 "post_step_observation": first_world(observation["actor"]) if observation is not None and kind in {"post_policy", "terminal", "pre_reset"} else None,
                 "reset_observation": first_world(observation["actor"]) if observation is not None and kind in {"reset_initial", "reset_after_terminal"} else None}
        emit(frame)
        if recorder is not None and kind in {"reset_initial", "physics_step"}:
            recorder.capture(cpu_data, integration_count)

    for actuator in env.scene["robot"].actuators:
        original_compute = actuator.compute
        def compute(cmd, actuator=actuator, original_compute=original_compute):
            qfrc_bias = actuator._as_tensor(actuator._data.qfrc_bias)
            qfrc_constraint = actuator._as_tensor(actuator._data.qfrc_constraint)
            own_friction = actuator._dof_friction_force(qfrc_bias.shape[-1])
            ids = actuator._dof_ids
            controller.clear()
            row = {"feedback_position": first_world(cmd.pos), "feedback_velocity": first_world(cmd.vel),
                   "computation_phase": "reset_warmup_before_forward" if reset_in_progress else "actuator_write_before_integration",
                   "feedback_state_measurement": {"episode_epoch": episode_epoch, "physics_tick": integration_count,
                                                  "episode_physics_tick": integration_count - episode_origin_tick,
                                                  "phase": "new_reset_state_before_forward" if reset_in_progress else "current_pre_integration_state"},
                   "input_load_measurement": native_force_measurement.copy(),
                   "delayed_position_target": first_world(cmd.position_target),
                   "previous_applied_actuator": first_world(actuator._as_tensor(actuator._data.qfrc_actuator)[:, ids]),
                   "external_without_own_friction": first_world((-qfrc_bias + qfrc_constraint - own_friction)[:, ids]),
                   "qfrc_bias": first_world(qfrc_bias[:, ids]), "qfrc_constraint": first_world(qfrc_constraint[:, ids]),
                   "qfrc_own_friction": first_world(own_friction[:, ids]),
                   "previous_computed_motor": first_world(actuator._prev_motor_torque)}
            result = original_compute(cmd)
            row["computed_motor_torque"] = first_world(result)
            row["frictionloss"] = first_world(actuator._as_tensor(actuator._mjwarp_model.dof_frictionloss)[:, ids])
            row["damping"] = first_world(actuator._as_tensor(actuator._mjwarp_model.dof_damping)[:, ids])
            controller.append(row)
            return result
        actuator.compute = compute
    original_reset = env.sim.reset
    def sim_reset(env_ids=None):
        nonlocal episode_epoch, episode_origin_tick, reset_in_progress
        result = original_reset(env_ids)
        episode_epoch += 1
        episode_origin_tick = integration_count
        reset_in_progress = True
        # Warp reset clears force inputs and solver/contact counts, but leaves
        # qfrc_bias/constraint/actuator derived arrays until the next forward.
        # Keep their actual preceding measurement stamp for BAM warmup reads.
        return result
    env.sim.reset = sim_reset
    original_forward = env.sim.forward
    def forward():
        nonlocal native_force_measurement, reset_in_progress
        result = original_forward()
        native_force_measurement = {"phase": "reset_forward" if reset_in_progress else "post_policy_forward",
                                    "episode_epoch": episode_epoch, "physics_tick": integration_count,
                                    "executed_integration_interval": None}
        reset_in_progress = False
        return result
    env.sim.forward = forward
    original_termination_compute = env.termination_manager.compute
    def termination_compute():
        nonlocal termination_measurement
        result = original_termination_compute()
        termination_measurement = {"phase": "real_policy_step_compute_before_post_policy_forward",
                                   "episode_epoch": episode_epoch, "physics_tick": integration_count}
        return result
    env.termination_manager.compute = termination_compute
    original_step = env.sim.step
    def step():
        nonlocal integration_count, native_force_measurement
        original_step()
        integration_count += 1
        native_force_measurement = {"phase": "integration_solve", "episode_epoch": episode_epoch,
                                    "physics_tick": integration_count,
                                    "executed_integration_interval": [(integration_count - 1) * env.physics_dt, integration_count * env.physics_dt]}
        capture("physics_step")
    env.sim.step = step
    original_policy_step = env.step
    def policy_step(action):
        nonlocal observation
        result = original_policy_step(action)
        observation = result[0]
        capture("post_policy")
        if learning_active and (bool(result[2][0]) or bool(result[3][0])):
            reason = "terminated" if bool(result[2][0]) else "time_limit"
            capture("terminal", reason=reason)
            capture("pre_reset", reason=reason)
            observation, _ = env.reset()
            capture("reset_after_terminal")
            capture("reset_initial")
            result = (observation, *result[1:])
        return result
    env.step = policy_step
    runner = None
    checkpoint = output / "source_initial.pt"
    onnx_path = output / "source_initial.onnx"
    try:
        if make_runner:
            wrapper = RslRlVecEnvWrapper(env)
            agent = dataclasses.asdict(source["registry"].load_rl_cfg(SKILL_TASKS[adoption["skill"]]))
            agent.update(seed=cfg.seed, logger="tensorboard", upload_model=False)
            runner_cls = source["registry"].load_runner_cls(SKILL_TASKS[adoption["skill"]])
            runner = runner_cls(wrapper, agent, str(output / "runner_logs"), device=device)
            # Upstream mjlab 1.3.0 expects logger_type; rsl-rl 5.0.1 calls it
            # cfg["logger"]. Keep the upstream saver/metadata path operational.
            if not hasattr(runner.logger, "logger_type"):
                runner.logger.logger_type = agent["logger"]
            if resume is not None:
                runner.load(str(resume), map_location=device)
            if _phase_journal is not None:
                _phase_journal.mark("runner_restore", checkpoint_sha256=sha256_file(resume) if resume is not None else None)
            checkpoint_origin = describe_origin(runner, resume, origin_input, learning_iterations)
            write_json(output / "checkpoint_origin.json", checkpoint_origin)
            policy = runner.get_inference_policy(device=device)
            original_act = runner.alg.act
            def sampled_act(inputs):
                nonlocal policy_trace, policy_call_count
                frozen = inputs["actor"].detach().clone()
                output_action = original_act(inputs)
                policy_call_count += 1
                policy_trace = {"call_id": policy_call_count, "episode_epoch": episode_epoch,
                                "physics_tick_before_step": integration_count,
                                "time_seconds": integration_count * env.physics_dt,
                                "phase": "learning", "stochastic": True,
                                "input_61": first_world(frozen), "output_14": first_world(output_action.detach().clone()),
                                "actor_mean_14": first_world(runner.alg.actor.output_mean.detach().clone()),
                                "learning_iteration": runner.current_learning_iteration}
                capture("policy_call")
                return output_action
            runner.alg.act = sampled_act
            observation = env.obs_buf
        else:
            observation, _ = env.reset()
        if observation["actor"].shape[-1] != 61:
            raise ValueError(f"Actual actor observation dimension {observation['actor'].shape}")
        capture("reset_initial")
        timing_checks["initial"] = delay_diagnostics(env)
        if runner is not None:
            wrapper.get_observations()
            if delay_diagnostics(env) != timing_checks["initial"]:
                raise ValueError("Same-timestamp observation read advanced causal history")
        if learning_iterations:
            if runner is None or not 1 <= learning_iterations <= 2:
                raise ValueError("This structural smoke accepts one or two actual upstream PPO updates")
            learning_active = True
            learning_started = time.monotonic()
            runner.learn(num_learning_iterations=learning_iterations, init_at_random_ep_len=False)
            learning_elapsed = time.monotonic() - learning_started
            learning_active = False
            capture("terminal", reason="bounded_learning_horizon_truncation")
            capture("pre_reset", reason="bounded_learning_horizon_truncation")
            with torch.inference_mode():
                observation, _ = env.reset()
            capture("reset_after_terminal")
            capture("reset_initial")
            policy = runner.get_inference_policy(device=device)
        if runner is not None:
            frozen_before = model_state(runner)
        if _phase_journal is not None:
            _phase_journal.mark("deterministic_inference", policy_updates_this_run=0)
        for tick in range(steps):
            with torch.inference_mode():
                if runner is None:
                    action = torch.zeros((1, 14), device=device)
                else:
                    inputs = wrapper.get_observations()
                    frozen = inputs["actor"].detach().clone()
                    action = policy(inputs)
                    policy_call_count += 1
                    policy_trace = {"call_id": policy_call_count, "episode_epoch": episode_epoch,
                                    "physics_tick_before_step": integration_count,
                                    "time_seconds": integration_count * env.physics_dt,
                                    "phase": "inference", "stochastic": False,
                                    "input_61": first_world(frozen), "output_14": first_world(action.detach().clone()),
                                    "actor_mean_14": first_world(action.detach().clone())}
                    capture("policy_call")
                observation, reward, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                terminal_reason = "terminated" if bool(terminated[0]) else "time_limit"
                break
        terminal_reason = terminal_reason or "diagnostic_horizon_truncation"
        if _phase_journal is not None:
            _phase_journal.mark("natural_terminal_reset", terminal_reason=terminal_reason,
                                physics_ticks=integration_count)
        timing_checks["terminal"] = delay_diagnostics(env)
        capture("terminal", reason=terminal_reason)
        capture("pre_reset", reason=terminal_reason)
        if runner is not None:
            frozen_after = model_state(runner)
            if frozen_before != frozen_after:
                raise ValueError("Policy parameters, normalizer, optimizer or iteration changed during inference")
            write_json(output / "policy_freeze.json", {"schema": "actual_policy_freeze_v1", "before": frozen_before,
                       "after": frozen_after, "identical": True, "method": "all actor/critic state_dict tensor bytes including normalizer buffers and optimizer steps; actual inference episode boundaries"})
            runner.save(str(checkpoint))
            if producer is not None:
                producer.record_generated(checkpoint, producer="upstream_runner.save",
                    recipe={"actual_frozen_policy_state_identity": identity(frozen_before),
                            "checkpoint_origin_identity": identity(checkpoint_origin),
                            "ppo_updates_this_run": 0})
            # Use the authoritative export method and validate existence, not its warning-only save wrapper.
            runner.export_policy_to_onnx(str(output), onnx_path.name)
            if producer is not None:
                producer.record_generated(onnx_path, producer="upstream_runner.export_policy_to_onnx",
                    recipe={"checkpoint_sha256": sha256_file(checkpoint),
                            "actual_frozen_policy_state_identity": identity(frozen_before),
                            "torch_export_runtime": "actual upstream runner method"})
            before = env.common_step_counter
            if producer is not None:
                producer.before_load(checkpoint, expected_producer="upstream_runner.save")
            runner.load(str(checkpoint), map_location=device)
            if env.common_step_counter != before:
                raise ValueError("Upstream resume did not preserve curriculum step counter")
            import onnx
            if producer is not None:
                producer.before_load(onnx_path, expected_producer="upstream_runner.export_policy_to_onnx")
            onnx.checker.check_model(onnx.load(onnx_path))
            if _profile_manifest is not None:
                post_export_state = model_state(runner)
                write_json(output / "policy_post_export_freeze.json",
                           {"schema": "actual_policy_post_export_freeze_v1", "before": frozen_before,
                            "after_save_export_reload": post_export_state,
                            "identical": post_export_state == frozen_before,
                            "ppo_updates_this_run": 0})
                if post_export_state != frozen_before:
                    raise ValueError("Zero-update actor/critic/normalizer/optimizer state changed after save/export/reload")
        if _phase_journal is not None:
            _phase_journal.mark("save_resume_export", onnx_sha256=sha256_file(onnx_path),
                                actor_frozen=frozen_before == frozen_after)
        with torch.inference_mode():
            observation, _ = env.reset()
        timing_checks["post_reset"] = delay_diagnostics(env)
        capture("reset_after_terminal")
        from .model_export import export_observation_action_reference
        boundary = export_observation_action_reference(env, output / "observation_action_reference.json")
        if _profile_manifest is not None and runner is not None:
            final_state = model_state(runner)
            freeze_path = output / "policy_post_export_freeze.json"
            freeze = json.loads(freeze_path.read_text())
            freeze["after_terminal_reset"] = final_state
            freeze["identical_after_terminal_reset"] = final_state == frozen_before
            write_json(freeze_path, freeze)
            if final_state != frozen_before:
                raise ValueError("Zero-update model or optimizer changed after terminal reset")
    finally:
        stream.close()
        if recorder is not None:
            video_report = recorder.close()
        env.close()
        if _phase_journal is not None:
            _phase_journal.mark("video_closed", video_sha256=sha256_file(output / "source_video.mp4") if video_report is not None else None)
        if hooks is not None:
            hooks.__exit__(None, None, None)
        if producer is not None:
            write_json(output / "jit_producer_journal.json", {"schema": "microduck_profile_producer_journal_v1",
                       "generated": producer.produced, "loaded": producer.loaded,
                       "scope": "actual Warp CUDA and upstream runner checkpoint/ONNX build/load wrappers; CPU JIT and driver internal cache not asserted complete"})
    report = {"status": "captured", "skill": adoption["skill"], "records": records,
              "integration_count": integration_count, "policy_count": integration_count // cfg.decimation,
              "elapsed_seconds": time.monotonic() - started, "terminal_reason": terminal_reason,
              "full_episode_requested": full_episode_requested,
              "natural_task_terminal": terminal_reason in {"terminated", "time_limit"},
              "source_commit": source["commit"], "adoption_sha256": adoption["sha256"],
              "trajectory": {"path": str(raw_path.resolve()), "sha256": sha256_file(raw_path)},
              "checkpoint": str(checkpoint.resolve()) if checkpoint.exists() else None,
              "onnx": str(onnx_path.resolve()) if onnx_path.exists() else None,
              "learned_this_invocation": bool(learning_iterations), "learning_iterations": learning_iterations,
              "ppo_iterations_this_invocation": learning_iterations,
              "checkpoint_origin": checkpoint_origin,
              "policy_frozen_during_inference": frozen_before == frozen_after if runner is not None else None,
              "learning_elapsed_seconds": learning_elapsed,
              "skill_qualified": False, "video_review_complete": False}
    report["timing_checks"] = timing_checks
    report["action_scale"] = boundary["action_scale"]
    report["home"] = boundary["home"]
    report["video"] = video_report
    write_json(output / "capture_report.json", report)
    return report


def capture_source(*args, learning_iterations: int = 0, **kwargs) -> dict:
    """Public lower-level capture also forbids bypassing bounded learning."""
    if type(learning_iterations) is not int or learning_iterations != 0:
        raise Rejection("Public capture_source cannot learn; use the authorized bounded training entry")
    if "_learning_binding" in kwargs or "_verified_runtime" in kwargs:
        raise Rejection("Private learning binding is not a public capture argument")
    return _capture_source(*args, learning_iterations=0, **kwargs)
