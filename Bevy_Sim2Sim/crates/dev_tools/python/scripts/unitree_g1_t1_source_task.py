#!/usr/bin/env python3
"""Bounded original T1 scene/camera rollout; optional VLA or expert diagnostic.

Keeps the original task layout, six-second episode, reset events, 200/50 Hz
PhysX/WBC configuration and AGILE implementation. Only cached asset paths are
redirected. Source performance never qualifies native Rapier task performance.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

from unitree_g1_t1_source_shelf import ARENA_REV, LAB_REV, USD_SHA, URDF_SHA, AGILE_SHA
from unitree_g1_source_expert import load_sequence


BACKGROUND_SHA = '7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051'
APPLE_SHA = '2e0e0462c4345b340e1c6040c11abe1818437ced1db228a53ec0944001bb46d8'
PLATE_SHA = '286238c8f957e3267fa21a0003b320868130f903a4b1d991a4ffeb49faaea5a8'
STATIC_REV = '7f78bebf1a90131e7304beacfcd47eb27bad16ab'
ARM_SUFFIXES = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
HAND_SUFFIXES = ('index_0', 'index_1', 'middle_0', 'middle_1', 'thumb_0', 'thumb_1', 'thumb_2')


def take_prefetched_chunk(pending, tick, consumed_frames):
    """Exact simulated boundary; neither drop predecessor frames nor rebase RGB."""
    if (pending is None or tick <= 0 or tick % 40 != 0 or consumed_frames != 40
            or pending['observed_tick'] != tick - 30
            or len(pending['frames']) != 40):
        raise ValueError('Prefetched source chunk requires all40 predecessor frames and exact30-Tick image age')
    return pending['frames'], pending['sequence'], pending['observed_tick']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def policy_reply(path, body):
    data = json.dumps(body, allow_nan=False).encode()
    if len(data) > 2_097_152:
        raise ValueError('Source observation exceeds bounded service request')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(30)
        connection.connect(str(path))
        connection.sendall(len(data).to_bytes(4, 'big') + data)
        def exact(length):
            value = bytearray()
            while len(value) < length:
                part = connection.recv(length - len(value))
                if not part:
                    raise ValueError('Local policy bridge disconnected')
                value.extend(part)
            return bytes(value)
        size = int.from_bytes(exact(4), 'big')
        if not 0 < size <= 1_048_576:
            raise ValueError('Policy reply exceeds finite bound')
        result = json.loads(exact(size))
    if result.get('error'):
        raise ValueError(f"Local policy failed: {result['error']}")
    for key in ('profile', 'sequence_id', 'observation'):
        if result[key] != body[key]:
            raise ValueError(f'Policy reply identity changed: {key}')
    if result['model_revision'] != STATIC_REV or result['action_period_ns'] != 20_000_000 or len(result['frames']) != 40:
        raise ValueError('Policy reply changed matched T1 contract')
    return result


def renderer_reply(path, body):
    """Private diagnostic IPC, distinct from the RGB/self-state policy wire."""
    data = json.dumps(body, allow_nan=False).encode()
    if not 0 < len(data) <= 2_097_152:
        raise ValueError('Renderer pose request exceeds finite bound')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(90)
        connection.connect(str(path))
        connection.sendall(len(data).to_bytes(4, 'big') + data)
        def exact(length):
            value = bytearray()
            while len(value) < length:
                part = connection.recv(length - len(value))
                if not part:
                    raise ValueError('Diagnostic renderer disconnected')
                value.extend(part)
            return bytes(value)
        size = int.from_bytes(exact(4), 'big')
        if not 0 < size <= 2_097_152:
            raise ValueError('Renderer reply exceeds finite bound')
        result = json.loads(exact(size))
    if result.get('error'):
        raise ValueError(f"Diagnostic renderer failed: {result['error']}")
    if (result['schema'] != 'g1_source_renderer_rgb_v1'
            or result['episode_id'] != body['episode_id']
            or result['control_tick'] != body['control_tick']
            or not result['pose_correspondence_passed']
            or result['native_integrations'] != 0
            or result['native_controller_updates'] != 0
            or result['native_policy_calls'] != 0):
        raise ValueError('Renderer identity/zero-integration contract changed')
    now = time.time_ns() // 1_000_000
    if not 0 <= now - result['captured_at_unix_ms'] <= 15_000:
        raise ValueError('Renderer returned an old/future image')
    return result


def run(args, receipt, output):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    parser = get_isaaclab_arena_cli_parser()
    from isaaclab_arena_environments.galileo_g1_static_pick_and_place_environment import GalileoG1StaticPickAndPlaceEnvironment
    GalileoG1StaticPickAndPlaceEnvironment.add_cli_args(parser)
    launch = parser.parse_args(['--headless', '--enable_cameras', '--no-solve-relations', '--embodiment', 'g1_wbc_agile_joint'])
    launch.seed = args.seed
    sys.argv = [sys.argv[0]]
    with SimulationAppContext(launch):
        try:
            import numpy as np
            import torch
            import warp as wp
            from PIL import Image
            from isaaclab_arena.assets.background_library import GalileoLocomanipBackground
            from isaaclab_arena.assets.object_library import Apple01ObjaverseRobolab, ClayPlatesHot3DRobolab
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as action_module
            import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as model_module
            torch._C._jit_override_can_fuse_on_gpu(False)
            def tensor(value):
                return wp.to_torch(value) if isinstance(value, wp.array) else value
            def measured(value):
                return tensor(value)[0].detach().cpu().numpy()

            original_factory = action_module.get_wbc_policy
            def local_factory(name, model, config, count):
                if config.wbc_version != 'agile':
                    raise ValueError('T1 body controller must retain AGILE')
                config.wbc_model_path = str(args.agile)
                return original_factory(name, model, config, count)
            action_module.get_wbc_policy = local_factory
            def local_robot_path(url, **kwargs):
                if url.endswith('/g1_29dof_with_hand.urdf'):
                    return str(args.urdf)
                if url.endswith('/Arena/wbc_policy/robot_model/g1/'):
                    return str(args.urdf.parent)
                raise ValueError(f'Unexpected source robot asset: {url}')
            model_module.retrieve_file_path = local_robot_path
            GalileoLocomanipBackground.usd_path = str(args.background)
            Apple01ObjaverseRobolab.usd_path = str(args.apple)
            ClayPlatesHot3DRobolab.usd_path = str(args.plate)
            description = GalileoG1StaticPickAndPlaceEnvironment().get_env(launch)
            description.embodiment.scene_config.robot.spawn.usd_path = str(args.usd)
            env = ArenaEnvBuilder(description, launch).make_registered()
            raw = env.unwrapped
            if raw.cfg.sim.dt != .005 or raw.cfg.decimation != 4 or raw.cfg.episode_length_s != 6.0:
                raise ValueError('Original T1 frequency/episode contract changed')
            robot = raw.scene['robot']
            camera = raw.scene['robot_head_cam']
            term = raw.action_manager.get_term('g1_action')
            names = list(robot.joint_names)
            groups = {f'{side}_arm': [f'{side}_{suffix}_joint' for suffix in ARM_SUFFIXES] for side in ('left', 'right')}
            groups.update({f'{side}_hand': [f'{side}_hand_{suffix}_joint' for suffix in HAND_SUFFIXES] for side in ('left', 'right')})
            groups['waist'] = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
            indices = {key: [names.index(name) for name in joints] for key, joints in groups.items()}
            receipt.update(runtime_build=Path('/isaac-sim/VERSION').read_text().strip(),
                torch_version=torch.__version__, torch_gpu_jit_fusion_enabled=torch._C._jit_can_fuse_on_gpu(),
                source_physics_hz=200, source_control_hz=50, source_episode_length_s=6.0,
                joint_names=names, camera_config=camera.cfg.to_dict(),
                task_assets=list(description.scene.assets), action_shape=list(env.action_space.shape))
            env.reset(seed=args.seed)
            if args.reset_camera_refresh:
                # Lab's RTX dedup key contains only sim id + physics step. A
                # reset at that same step can re-read the pre-reset annotator
                # despite new q. Pump a render-only update and invalidate the
                # sensor cache; assert that no physical state/clock advanced.
                import carb
                import omni.kit.app
                capture_dir = args.output.with_suffix('.reset_camera')
                capture_dir.mkdir()
                before_rgb = measured(camera.data.output['rgb'])[:, :, :3].copy()
                Image.fromarray(before_rgb).save(capture_dir / 'before.png')
                def physical_state():
                    return {
                        'q': measured(robot.data.joint_pos).copy(),
                        'dq': measured(robot.data.joint_vel).copy(),
                        'root': measured(robot.data.root_link_pose_w).copy(),
                        **{name: measured(raw.scene[name].data.root_state_w).copy()
                           for name in (launch.object, launch.destination)}}
                before = physical_state()
                counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                setting = '/app/player/playSimulations'
                previous = carb.settings.get_settings().get(setting)
                try:
                    raw.sim.physics_manager.forward()
                    raw.sim.set_setting(setting, False)
                    omni.kit.app.get_app().update()
                finally:
                    raw.sim.set_setting(setting, previous)
                camera.reset()
                after_rgb = measured(camera.data.output['rgb'])[:, :, :3].copy()
                Image.fromarray(after_rgb).save(capture_dir / 'after.png')
                after = physical_state()
                error = max(float(np.abs(after[key] - value).max()) for key, value in before.items())
                receipt['reset_camera_refresh'] = {
                    'render_only_pumps': 1, 'before_sdk_counters': counters,
                    'after_sdk_counters': (int(raw._sim_step_counter), int(raw.sim._physics_step_count)),
                    'physical_state_max_abs_change': error,
                    'before_rgb_sha256': hashlib.sha256(before_rgb.tobytes()).hexdigest(),
                    'after_rgb_sha256': hashlib.sha256(after_rgb.tobytes()).hexdigest(),
                    'changed_rgb_bytes': int(np.count_nonzero(before_rgb != after_rgb)),
                    'camera_frame': camera.frame.detach().cpu().tolist()}
                if error != 0 or counters != (int(raw._sim_step_counter), int(raw.sim._physics_step_count)):
                    raise ValueError('Render-only reset refresh changed physical state or integration counters')
            physics_counter_at_reset = int(raw._sim_step_counter)
            expert = load_sequence(args.expert_sequence, args.expert_sha256) if args.expert_sequence else None
            if expert:
                actual_initial = measured(robot.data.joint_pos)
                initial_error = max(abs(float(actual_initial[names.index(name)]) - target)
                    for name, target in zip(expert['joint_names'], expert['initial_joint_positions']))
                receipt.update(expert_sequence_sha256=args.expert_sha256,
                    expert_dataset_revision=expert['dataset_revision'], expert_frames=len(expert['frames']),
                    expert_initial_joint_max_error_rad=initial_error,
                    expert_object_initial_pose_available=False,
                    expert_after_sequence=expert['after_sequence'])
                if initial_error > 1e-3:
                    raise ValueError('Expert diagnostic initial robot pose does not match original reset')
            action = torch.zeros(env.action_space.shape, device=raw.device)
            action[:, :43] = tensor(robot.data.default_joint_pos)
            action[:, -4] = .75
            frames = []
            frame_index = 0
            sequence = 0
            active_sequence = 0
            pending_chunk = None
            last_action_sha256 = None
            trace = args.output.with_suffix('.jsonl')
            captures = args.output.with_suffix('.captures')
            captures.mkdir()
            with trace.open('x') as log:
                for tick in range(args.ticks + 1):
                    q = measured(robot.data.joint_pos)
                    root = measured(robot.data.root_link_pose_w)
                    gravity = measured(robot.data.projected_gravity_b)
                    if not all(np.isfinite(x).all() for x in (q, root, gravity)):
                        raise ValueError(f'Nonfinite original T1 state at {tick}')
                    sample = {'control_tick': tick,
                        'physics_steps_since_reset': int(raw._sim_step_counter) - physics_counter_at_reset,
                        'root_link_pose_w': root.tolist(), 'joint_positions': q.tolist(),
                        'processed_joint_targets': term.processed_actions[0].detach().cpu().tolist(),
                        'upright': float(-gravity[2]), 'policy_sequence': active_sequence,
                        'last_action_frame_sha256': last_action_sha256,
                        'last_action_frame': min(tick - 1, len(expert['frames']) - 1) if expert and tick else
                            (frame_index - 1 if frames else None),
                        'acceptance_truth_only': {name: {
                            'pose': measured(raw.scene[name].data.root_pose_w).tolist(),
                            'velocity': measured(raw.scene[name].data.root_vel_w).tolist()}
                            for name in (launch.object, launch.destination)}}
                    log.write(json.dumps(sample, allow_nan=False) + '\n'); log.flush()
                    receipt['last_state'] = sample
                    receipt['completed_control_ticks'] = tick
                    if args.joint_kinematics_audit and tick % 40 == 0:
                        # Independent SDK quantities only. No coordinate writes,
                        # extra physics steps, controller or task-policy inputs.
                        counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                        jacobian = measured(robot.root_view.get_jacobians())
                        poses = measured(robot.data.body_link_pose_w)
                        if (jacobian.shape != (len(robot.body_names), 6, len(names) + 6)
                                or poses.shape != (len(robot.body_names), 7)
                                or not np.isfinite(jacobian).all() or not np.isfinite(poses).all()):
                            raise ValueError('Unexpected floating-base SDK joint-kinematics quantities')
                        audit = {'schema': 'g1_source_joint_kinematics_v1',
                            'control_tick': tick, 'source_physics_steps_since_reset': tick * 4,
                            'before_sdk_counters': counters,
                            'after_sdk_counters': (int(raw._sim_step_counter), int(raw.sim._physics_step_count)),
                            'joint_names': names, 'body_names': list(robot.body_names),
                            'joint_positions': q.tolist(), 'body_link_pose_w': poses.tolist(),
                            'jacobian_shape': list(jacobian.shape),
                            'world_jacobian_linear_then_angular': jacobian.tolist(),
                            'coordinate_writes': 0, 'extra_integrations': 0,
                            'scope': 'read-only SDK world Jacobian/FK; independent diagnostic truth only'}
                        if audit['before_sdk_counters'] != audit['after_sdk_counters']:
                            raise ValueError('Read-only kinematics audit advanced SDK counters')
                        (captures / f'kinematics_{tick:04}.json').write_text(
                            json.dumps(audit, allow_nan=False))
                    if root[2] + .795 < .35 or sample['upright'] < .5:
                        raise ValueError(f'Original T1 fall guard at {tick}')
                    early_observation = (args.policy_prefetch_after_ticks is not None
                        and tick % 40 == 10 and tick + 30 < args.ticks)
                    if tick % 40 == 0 or early_observation:
                        rgb = measured(camera.data.output['rgb'])[:, :, :3]
                        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                            raise ValueError('Original camera did not produce 640x480 uint8 RGB')
                        Image.fromarray(rgb).save(captures / f'head_{tick:04}.png')
                        captured_at_unix_ms = time.time_ns() // 1_000_000
                        if args.renderer_socket and tick < args.ticks:
                            # Measured simulator truth goes only to the separate
                            # renderer, never into the task-policy request. The
                            # original source physics remains the sole task
                            # owner and is blocked without advancing here.
                            def renderer_state():
                                return {
                                    'q': measured(robot.data.joint_pos).copy(),
                                    'dq': measured(robot.data.joint_vel).copy(),
                                    'root': measured(robot.data.root_link_pose_w).copy(),
                                    **{name: measured(raw.scene[name].data.root_state_w).copy()
                                       for name in (launch.object, launch.destination)}}
                            before = renderer_state()
                            counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                            render_request = {'schema': 'g1_source_renderer_pose_v1',
                                'episode_id': args.episode_id, 'control_tick': tick,
                                'environment_translation_source': [0., 0., .795],
                                'joint_kinematics': audit,
                                'task_objects': sample['acceptance_truth_only']}
                            render_started = time.monotonic()
                            rendered = renderer_reply(args.renderer_socket, render_request)
                            after = renderer_state()
                            error = max(float(np.abs(after[key] - value).max())
                                        for key, value in before.items())
                            after_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                            if error != 0 or counters != after_counters:
                                raise ValueError('Renderer request advanced or changed source physical state')
                            pixels = base64.b64decode(rendered.pop('camera_rgb_b64'), validate=True)
                            if len(pixels) != 640 * 480 * 3 or hashlib.sha256(pixels).hexdigest() != rendered['rgb_sha256']:
                                raise ValueError('Renderer RGB dimensions/hash changed')
                            rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(480, 640, 3)
                            Image.fromarray(rgb).save(captures / f'bevy_head_{tick:04}.png')
                            captured_at_unix_ms = rendered['captured_at_unix_ms']
                            rendered.update(source_before_sdk_counters=counters,
                                source_after_sdk_counters=after_counters,
                                source_physical_state_max_abs_change=error,
                                render_request_seconds=time.monotonic() - render_started)
                            (captures / f'renderer_{tick:04}.json').write_text(json.dumps(rendered, allow_nan=False))
                            receipt.setdefault('renderer_requests', []).append(rendered)
                        state = {key: q[value].tolist() for key, value in indices.items()}
                        # Use the numerically verified training order; published
                        # preprocess_state element_names incorrectly say thumb first.
                        stamp = {'episode_id': args.episode_id, 'frame_id': tick,
                            'sim_time_ns': tick * 20_000_000, 'captured_at_unix_ms': captured_at_unix_ms}
                        make_request = (args.policy_prefetch_after_ticks is None
                            or tick == 0 or early_observation)
                        if make_request:
                            sequence += 1
                        request = {'schema': 'unitree_g1_static_observation_v2', 'profile': 'static_apple',
                            'sequence_id': sequence, 'observation': stamp,
                            'camera_rgb_b64': base64.b64encode(rgb.tobytes()).decode(), 'state_groups': state}
                        if make_request:
                            (captures / f'request_{tick:04}.json').write_text(json.dumps(request, allow_nan=False))
                        submit_policy = (args.policy_socket and tick < args.ticks
                            and (args.policy_prefetch_after_ticks is None or tick == 0 or early_observation))
                        if submit_policy:
                            started = time.monotonic()
                            reply = policy_reply(args.policy_socket, request)
                            (captures / f'reply_{tick:04}.json').write_text(json.dumps(reply, allow_nan=False))
                            if early_observation:
                                if pending_chunk is not None or frame_index != 10:
                                    raise ValueError('Source prefetch exceeded one pending chunk or changed trigger')
                                pending_chunk = {'frames': reply['frames'], 'sequence': sequence,
                                    'observed_tick': tick}
                            else:
                                frames = reply['frames']; frame_index = 0
                                active_sequence = sequence
                                receipt.setdefault('installed_chunks', []).append({
                                    'sequence': sequence, 'observation_control_tick': tick,
                                    'start_control_tick': tick, 'observation_age_ticks': 0})
                            receipt['policy_calls'] += 1
                            receipt.setdefault('policy_call_seconds', []).append(time.monotonic() - started)
                        print(f'G1_T1_TASK tick={tick} policy_calls={receipt["policy_calls"]} root_z={root[2]:.6f}', flush=True)
                    if tick == args.ticks:
                        break
                    if args.policy_prefetch_after_ticks is not None and tick > 0 and tick % 40 == 0:
                        frames, active_sequence, observed_tick = take_prefetched_chunk(
                            pending_chunk, tick, frame_index)
                        pending_chunk = None
                        frame_index = 0
                        receipt.setdefault('installed_chunks', []).append({
                            'sequence': active_sequence, 'observation_control_tick': observed_tick,
                            'start_control_tick': tick, 'observation_age_ticks': tick - observed_tick})
                    if frames or expert:
                        frame = expert['frames'][min(tick, len(expert['frames']) - 1)] if expert else frames[frame_index]
                        last_action_sha256 = hashlib.sha256(json.dumps(
                            frame, sort_keys=True, allow_nan=False).encode()).hexdigest()
                        # Original remap_policy_joints_to_sim_joints_np starts
                        # with zero for unpredicted leg joint targets; WBC then
                        # fills those joints from the lower-body policy.
                        action.zero_()
                        for key, ids in indices.items():
                            target = np.asarray(frame[key], dtype=np.float32)
                            if target.shape != (len(ids),) or not np.isfinite(target).all():
                                raise ValueError('Invalid decoded original task action')
                            action[0, ids] = torch.as_tensor(target, device=raw.device)
                        action[0, 43:46] = torch.as_tensor(frame['navigate_mps_rps'], device=raw.device)
                        action[0, 46] = frame['base_height_m']
                        action[0, 47:50] = 0  # Original GR00T does not emit torso RPY.
                        if not expert:
                            frame_index += 1
                    _, _, terminated, truncated, info = env.step(action)
                    receipt['completed_control_ticks'] = tick + 1
                    receipt['measured_physics_steps_since_reset'] = int(raw._sim_step_counter) - physics_counter_at_reset
                    if receipt['measured_physics_steps_since_reset'] != (tick + 1) * 4:
                        raise ValueError('SDK physical integration counter diverged from original decimation')
                    if bool(terminated.any()) or bool(truncated.any()):
                        receipt['episode_termination'] = {'tick': tick + 1,
                            'terminated': bool(terminated.any()), 'truncated': bool(truncated.any()),
                            'original_terms': {name: bool(raw.termination_manager.get_term(name).any())
                                for name in raw.termination_manager.active_terms},
                            'reset_metrics': info.get('log', {}),
                            'terminal_step_auto_reset': True}
                        break
            receipt['source_scene_rollout_completed'] = True
            receipt['source_vla_executed'] = receipt['policy_calls'] > 0
            receipt['source_expert_executed'] = expert is not None
            receipt['trace_sha256'] = digest(trace)
            env.close()
        except BaseException as error:
            receipt['error'] = repr(error)
            raise
        finally:
            output.seek(0); output.truncate(); json.dump(receipt, output, indent=2, allow_nan=False, default=str)
            output.write('\n'); output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('usd', 'urdf', 'agile', 'background', 'apple', 'plate', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode-id', type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--policy-socket', type=Path)
    mode.add_argument('--expert-sequence', type=Path)
    parser.add_argument('--policy-prefetch-after-ticks', type=int, choices=[10],
        help='Simulated image-age diagnostic only: observe at10, apply at next40 boundary; inference still pauses source physics')
    parser.add_argument('--renderer-socket', type=Path,
                        help='Diagnostic only: source physics plus zero-Tick Bevy render of its measured poses')
    parser.add_argument('--expert-sha256')
    parser.add_argument('--joint-kinematics-audit', action='store_true',
                        help='Read-only SDK Jacobian/body-pose evidence at existing 40-Tick boundaries')
    parser.add_argument('--reset-camera-refresh', action='store_true', default=True,
                        help='Render-only source reset cache diagnostic; assert unchanged physical state/counters')
    args = parser.parse_args()
    if not 1 <= args.ticks <= 300 or args.episode_id < 1:
        parser.error('Use 1..300 ticks within the original six-second episode and a positive episode identity')
    if bool(args.expert_sequence) != bool(args.expert_sha256):
        parser.error('Expert diagnostics require both sequence and its SHA-256 identity')
    if args.policy_prefetch_after_ticks is not None and (
            not args.policy_socket or args.renderer_socket or args.ticks % 40 != 0
            or not 80 <= args.ticks <= 240):
        parser.error('Image-age comparison requires originalRTX policy mode and2..6 whole40-frame chunks')
    if args.renderer_socket:
        if not args.policy_socket:
            parser.error('Renderer comparison requires the original local policy bridge')
        args.joint_kinematics_audit = True
    if args.expert_sequence:
        load_sequence(args.expert_sequence, args.expert_sha256)
    receipt = {'schema': 'g1_t1_original_source_task_v1', 'qualified': False,
        'arena_commit': ARENA_REV, 'lab_commit': LAB_REV, 'harness_sha256': digest(__file__),
        'source_scene_rollout_completed': False, 'source_vla_executed': False, 'source_expert_executed': False,
        'task_success_verified': False, 'rendered_materials_verified': False,
        'policy_calls': 0, 'requested_control_ticks': args.ticks, 'seed': args.seed,
        'episode_id': args.episode_id, 'environment_translation': [0, 0, 0],
        'scope': 'original full T1 scene/camera; optional local ONNX or expert diagnostic; no native qualification'}
    if args.policy_prefetch_after_ticks is not None:
        receipt.update(scope='original source200/50 simulated30-Tick observation-to-action delay; source inference pauses; not a continuous1x or native qualification',
            simulated_image_age_diagnostic=True, source_inference_pauses=True,
            observation_trigger_after_ticks=10, scheduled_image_age_ticks=30)
    if args.renderer_socket:
        receipt.update(scope='original source physics/control with external zero-Tick Bevy-rendered observations; causal renderer diagnostic only',
                       external_renderer_diagnostic=True, source_renderer=str(args.renderer_socket))
    with args.output.open('x') as output:
        for name, expected in [('usd', USD_SHA), ('urdf', URDF_SHA), ('agile', AGILE_SHA),
                ('background', BACKGROUND_SHA), ('apple', APPLE_SHA), ('plate', PLATE_SHA)]:
            if digest(getattr(args, name)) != expected:
                raise ValueError(f'Frozen original artifact changed: {name}')
        json.dump(receipt, output, indent=2); output.flush()
        run(args, receipt, output)


if __name__ == '__main__':
    main()
