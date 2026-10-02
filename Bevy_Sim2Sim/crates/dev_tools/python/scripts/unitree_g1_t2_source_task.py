#!/usr/bin/env python3
"""Finite original T2/Homie scene diagnostic, separate from T1/N1.7.

The default only checks frozen files. --run-source starts the original 200/50
PhysX/Homie environment. With --policy-port it consumes the original N1.6
fifty-frame replies; inference pauses source physics. No native qualification.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time
import urllib.request

from unitree_g1_policy_prepare import PROFILES
from unitree_g1_source_stand import ARENA_REV, LAB_REV, FROZEN_ASSETS, digest, source_check


PROFILE = PROFILES['mobile_box']
TASK_FILES = {
    'background': ('Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd',
                   '7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051'),
    'box': ('Arena/assets/object_library/brown_box/brown_box.usd',
            '50dc139612086b9483770a1abc17dc600445aa4f85323d74fa97069f7c2eb4ed'),
    'bin': ('Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd',
            'b9ffec2e70fd009863a3fa8bd699aca808403522eafb259d5638135e63506999'),
}
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
HAND = ('index_0', 'index_1', 'middle_0', 'middle_1', 'thumb_0', 'thumb_1', 'thumb_2')
GROUP_NAMES = {f'{side}_arm': [f'{side}_{suffix}_joint' for suffix in ARM] for side in ('left', 'right')}
GROUP_NAMES.update({f'{side}_hand': [f'{side}_hand_{suffix}_joint' for suffix in HAND] for side in ('left', 'right')})
GROUP_NAMES['waist'] = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
SOURCE_PROFILES = {
    'development_0_3': (ARENA_REV, LAB_REV, '6.1.0.0'),
    'release_0_2_1': ('8b4a3a47fc53de23e8205089d71109a2e2348acd',
                      'e57379c634b42db5a0fe9f754341be6e2a7c7c43',
                      '6.0.0-rc.22+release.33481.407f3ea1.gl'),
}


def verify_source_receipt(path, expected_sha, roots, profile):
    """Verify a host-frozen source tree inside a read-only container mount."""
    if digest(path) != expected_sha:
        raise ValueError('T2 source-tree receipt identity changed')
    data = json.loads(path.read_text())
    arena_rev, lab_rev, _ = SOURCE_PROFILES[profile]
    if (data.get('schema') != 'g1_t2_source_tree_v1' or data.get('source_profile') != profile
            or data.get('commits') != {'arena': arena_rev, 'lab': lab_rev}):
        raise ValueError('T2 source-tree receipt belongs to another source stack')
    seen = set()
    for item in data['files']:
        root = roots.get(item['root'])
        relative = Path(item['path'])
        key = (item['root'], item['path'])
        if root is None or relative.is_absolute() or '..' in relative.parts or key in seen:
            raise ValueError('Invalid or duplicate T2 source-tree path')
        local = root / relative
        if not local.resolve().is_relative_to(root.resolve()) or digest(local) != item['sha256']:
            raise ValueError(f'T2 source-tree file changed: {key}')
        seen.add(key)
    if any(not any(key[0] == name for key in seen) for name in roots):
        raise ValueError('T2 source-tree receipt omitted a source root')
    return len(seen)


def validate_reply(request, reply):
    """Reject foreign profiles/stamps, altered timing, partial/nonfinite frames."""
    if request.get('profile') != 'mobile_box' or type(reply) is not dict:
        raise ValueError('T2 accepts only the mobile-box contract')
    stamp = reply.get('observation')
    if (type(stamp) is not dict
            or set(stamp) != {'episode_id', 'frame_id', 'sim_time_ns', 'captured_at_unix_ms'}
            or any(type(v) is not int or not 0 <= v < 2**64 for v in stamp.values())
            or type(reply.get('sequence_id')) is not int):
        raise ValueError('Invalid T2 reply stamp or sequence type')
    for key in ('profile', 'sequence_id', 'observation'):
        if reply.get(key) != request[key]:
            raise ValueError(f'T2 policy reply changed {key}')
    frames = reply.get('frames')
    if (reply.get('model_revision') != PROFILE['revision']
            or type(reply.get('action_period_ns')) is not int
            or reply.get('action_period_ns') != 20_000_000
            or type(frames) is not list or len(frames) != 50):
        raise ValueError('T2 reply changed original N1.6 revision/horizon/period')
    widths = {key: len(names) for key, names in GROUP_NAMES.items()}
    widths['navigate_mps_rps'] = 3
    for frame in frames:
        if type(frame) is not dict or set(frame) != {*widths, 'base_height_m'}:
            raise ValueError('T2 action groups differ from original contract')
        for key, width in widths.items():
            values = frame[key]
            if type(values) is not list or len(values) != width:
                raise ValueError(f'Wrong T2 group width: {key}')
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.402823466e38 for v in values):
                raise ValueError(f'Invalid T2 group value: {key}')
        height = frame['base_height_m']
        if type(height) not in (int, float) or not math.isfinite(height) or abs(height) > 3.402823466e38:
            raise ValueError('Invalid T2 base height')
    return frames


def policy_reply(port, request, socket_path=None):
    data = json.dumps(request, allow_nan=False).encode()
    if len(data) > 2_097_152:
        raise ValueError('T2 observation exceeds request bound')
    if socket_path:
        from unitree_g1_source_http_bridge import exact
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(35); connection.connect(str(socket_path))
            connection.sendall(len(data).to_bytes(4, 'big') + data)
            size = int.from_bytes(exact(connection, 4), 'big')
            if not 0 < size <= 1_048_576:
                raise ValueError('T2 Unix transport reply exceeds bounded size')
            payload = exact(connection, size)
    else:
        wire = urllib.request.Request(f'http://127.0.0.1:{port}/infer', data=data,
                                      headers={'Content-Type': 'application/json'})
        # A process-local opener has no proxy, so RGB/self-state cannot leave loopback.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(wire, timeout=30) as response:
            payload = response.read(1_048_577)
    if len(payload) > 1_048_576:
        raise ValueError('T2 reply exceeds bounded size')
    result = json.loads(payload)
    validate_reply(request, result)
    return result


def body_contract_frame(term, robot, measured, before_q, before_dq, tick):
    """Read the actual original WBC buffers after its one control call.

    No reconstructed observation, second inference, or physical truth is used.
    Joint targets must be the same values handed to the real articulation.
    """
    import numpy as np
    lower = term.get_wbc_policy.lower_body_policy
    order = term.wbc_g1_joints_order
    canonical_names = sorted(order, key=order.get)
    if sorted(order.values()) != list(range(43)) or len(canonical_names) != 43:
        raise ValueError('Original Homie canonical joint order changed')
    ids = [robot.joint_names.index(name) for name in canonical_names]
    obs = lower.observation
    if (not np.array_equal(obs['q'][0], before_q[ids])
            or not np.array_equal(obs['dq'][0], before_dq[ids])):
        raise ValueError('Captured Homie input is not the pre-control robot state')
    targets = measured(term.processed_actions)[ids]
    applied = measured(robot.data.joint_pos_target)[ids]
    if not np.array_equal(targets, applied):
        raise ValueError('Source processed targets differ from articulation targets')
    raw_targets = measured(term.raw_actions)[ids]
    goal = term.get_wbc_goal
    frame = {'control_tick': tick, 'positions': obs['q'][0].tolist(),
        'velocities': obs['dq'][0].tolist(),
        'root_rotation_wxyz': obs['floating_base_pose'][0, 3:7].tolist(),
        'root_angular_velocity_body': obs['floating_base_vel'][0, 3:6].tolist(),
        'navigation': goal['navigate_cmd'][0].tolist(),
        'pelvis_height': float(goal['base_height_command'][0, 0]),
        'torso_rpy': goal['torso_orientation_rpy_cmd'][0].tolist(),
        'upper_positions': raw_targets[15:].tolist(),
        'observation': lower.obs_buffer[0].tolist(), 'action': lower.action[0].tolist(),
        'lower_targets': targets[:15].tolist(), 'targets': targets.tolist(),
        'articulation_targets': applied.tolist(),
        'policy': 'stand' if np.linalg.norm(lower.cmd) < .05 else 'walk'}
    widths = {'positions': 43, 'velocities': 43, 'root_rotation_wxyz': 4,
        'root_angular_velocity_body': 3, 'navigation': 3, 'torso_rpy': 3,
        'upper_positions': 28, 'observation': 516, 'action': 15,
        'lower_targets': 15, 'targets': 43, 'articulation_targets': 43}
    if any(len(frame[key]) != width or not np.isfinite(frame[key]).all()
           for key, width in widths.items()):
        raise ValueError('Nonfinite or malformed actual source Homie buffers')
    return frame, canonical_names


def run_source(args, receipt, save, task_paths, mesh_assets):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    release_source = args.source_profile == 'release_0_2_1'
    parser = get_isaaclab_arena_cli_parser()
    if release_source:
        from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import GalileoG1LocomanipPickAndPlaceEnvironment
        GalileoG1LocomanipPickAndPlaceEnvironment.add_cli_args(parser)
        launch = parser.parse_args(['--headless', '--enable_cameras', '--no-solve-relations',
                                   '--embodiment', 'g1_wbc_joint', '--object', 'brown_box'])
        launch.seed = args.seed
    else:
        launch = parser.parse_args(['--enable_cameras'])
    sys.argv = [sys.argv[0]]
    if args.startup_trace:
        import faulthandler
        sys.argv.append('--info')
        faulthandler.dump_traceback_later(45, file=sys.__stderr__)
    receipt['stage'] = 'starting_original_sdk'; save()
    with SimulationAppContext(launch):
        if args.startup_trace:
            faulthandler.cancel_dump_traceback_later()
        receipt['stage'] = 'building_original_task'; save()
        try:
            import importlib.metadata
            import numpy as np
            import torch
            import warp as wp
            from PIL import Image
            from isaaclab_arena.assets.background_library import GalileoLocomanipBackground
            from isaaclab_arena.assets.object_library import BrownBox, BlueSortingBin
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import GalileoG1LocomanipPickAndPlaceEnvironment
            if not release_source:
                from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
                from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import GalileoG1LocomanipPickAndPlaceEnvironmentCfg
            import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as action_module
            import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as model_module
            import isaaclab
            for module, root in ((sys.modules[ArenaEnvBuilder.__module__], args.arena_source),
                                 (isaaclab, args.lab_source)):
                if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                    raise ValueError('T2 simulation import escaped the selected frozen source')
            torch._C._jit_override_can_fuse_on_gpu(False)
            def measured(value):
                if hasattr(value, 'torch'):
                    value = value.torch
                elif isinstance(value, wp.array):
                    value = wp.to_torch(value)
                return value[0].detach().cpu().numpy()

            original_factory = action_module.get_wbc_policy
            def local_factory(name, model, config, count):
                if config.wbc_version != 'homie_v2':
                    raise ValueError('T2 must retain original Homie v2 controller')
                config.wbc_model_path = ','.join(str(args.homie_assets / file) for file in ('stand.onnx', 'walk.onnx'))
                return original_factory(name, model, config, count)
            action_module.get_wbc_policy = local_factory
            def local_robot_path(url, **kwargs):
                if url.endswith('/g1_29dof_with_hand.urdf'):
                    return str(args.homie_assets / 'g1_29dof_with_hand.urdf')
                if url.endswith('/Arena/wbc_policy/robot_model/g1/'):
                    return str(args.homie_assets)
                if url in mesh_assets:
                    return str(mesh_assets[url])
                raise ValueError(f'Unexpected T2 kinematic asset: {url}')
            model_module.retrieve_file_path = local_robot_path
            GalileoLocomanipBackground.usd_path = str(task_paths['background'])
            BrownBox.usd_path = str(task_paths['box'])
            BlueSortingBin.usd_path = str(task_paths['bin'])
            if release_source:
                description = GalileoG1LocomanipPickAndPlaceEnvironment().get_env(launch)
            else:
                description = GalileoG1LocomanipPickAndPlaceEnvironment().build(
                    GalileoG1LocomanipPickAndPlaceEnvironmentCfg(enable_cameras=True, embodiment='g1_wbc_joint'))
            if description.task.task_description != PROFILE['reference_instruction']:
                raise ValueError('Original T2 language instruction changed')
            description.embodiment.scene_config.robot.spawn.usd_path = str(args.homie_assets / 'g1_29dof_with_hand_rev_1_0.usd')
            if args.scene_overlap_audit:
                original_cfg_callback = description.env_cfg_callback
                def query_cfg_callback(cfg):
                    cfg = original_cfg_callback(cfg)
                    cfg.sim.enable_scene_query_support = True
                    cfg.sim.physics.enable_scene_query_support = True
                    return cfg
                description.env_cfg_callback = query_cfg_callback
            builder_cfg = launch if release_source else ArenaEnvBuilderCfg(seed=args.seed, solve_relations=False)
            env = ArenaEnvBuilder(description, builder_cfg).make_registered()
            raw = env.unwrapped
            if raw.cfg.sim.dt != .005 or raw.cfg.decimation != 4 or raw.cfg.episode_length_s != 30.0:
                raise ValueError('Original T2 frequency/episode contract changed')
            robot = raw.scene['robot']; camera = raw.scene['robot_head_cam']
            term = raw.action_manager.get_term('g1_action')
            names = list(robot.joint_names)
            indices = {key: [names.index(name) for name in joints] for key, joints in GROUP_NAMES.items()}
            runtime_versions = {key: importlib.metadata.version(key) for key in ('torch', 'warp-lang')}
            if release_source:
                runtime_versions['isaacsim_standalone_build'] = Path('/isaac-sim/VERSION').read_text().strip()
                if runtime_versions['isaacsim_standalone_build'] != SOURCE_PROFILES[args.source_profile][2]:
                    raise ValueError('Released T2 source SDK build changed')
            else:
                runtime_versions['isaacsim'] = importlib.metadata.version('isaacsim')
                if runtime_versions['isaacsim'] != SOURCE_PROFILES[args.source_profile][2]:
                    raise ValueError('Development T2 source SDK build changed')
            receipt.update(runtime_versions=runtime_versions,
                           joint_names=names, policy_joint_groups=GROUP_NAMES,
                           camera_config=camera.cfg.to_dict(), action_shape=list(env.action_space.shape))
            if tuple(env.action_space.shape) != (1, 50):
                raise ValueError('Original T2 WBC action space changed')
            env.reset(seed=args.seed)
            receipt['stage'] = 'reset_camera_refresh'; save()
            # Ensure the reset pose is rendered without advancing physics. Preserve
            # both the before/after physical state and SDK counters as evidence.
            import carb
            import omni.kit.app
            def physical_state():
                return {'q': measured(robot.data.joint_pos).copy(), 'dq': measured(robot.data.joint_vel).copy(),
                        'root': measured(robot.data.root_link_pose_w).copy(),
                        **{name: measured(raw.scene[name].data.root_state_w).copy() for name in ('brown_box', 'blue_sorting_bin')}}
            before = physical_state()
            counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
            settings = carb.settings.get_settings(); previous = settings.get('/app/player/playSimulations')
            try:
                raw.sim.physics_manager.forward()
                raw.sim.set_setting('/app/player/playSimulations', False)
                omni.kit.app.get_app().update()
            finally:
                raw.sim.set_setting('/app/player/playSimulations', previous)
            camera.reset()
            after = physical_state()
            error = max(float(np.abs(after[key] - value).max()) for key, value in before.items())
            after_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
            receipt['reset_camera_refresh'] = {'render_only_pumps': 1, 'before_sdk_counters': counters,
                'after_sdk_counters': after_counters, 'physical_state_max_abs_change': error}
            if error != 0 or counters != after_counters:
                raise ValueError('T2 render refresh changed physical state or counters')
            reset_counter = int(raw._sim_step_counter)
            captures = args.output.with_suffix('.captures'); captures.mkdir()
            if args.scene_render_audit:
                from unitree_g1_t2_render_query import query_render_inputs
                before=physical_state();counters=(int(raw._sim_step_counter),int(raw.sim._physics_step_count))
                receipt['actual_render_inputs']=query_render_inputs(raw,captures/'render_inputs.json')
                after=physical_state();delta=max(float(np.abs(after[k]-v).max()) for k,v in before.items())
                receipt['actual_render_inputs'].update(physical_state_max_abs_change=delta,before_sdk_counters=counters,
                    after_sdk_counters=[int(raw._sim_step_counter),int(raw.sim._physics_step_count)])
                if delta!=0 or counters!=(int(raw._sim_step_counter),int(raw.sim._physics_step_count)):
                    raise ValueError('Render input query changed actual scene state/counters')
                save()
            if args.scene_light_causal_probe:
                from unitree_g1_t2_render_query import probe_legacy_light_inputs
                before=physical_state();counters=(int(raw._sim_step_counter),int(raw.sim._physics_step_count))
                receipt['light_causal_probe']=probe_legacy_light_inputs(raw,camera,captures)
                after=physical_state();delta=max(float(np.abs(after[k]-v).max()) for k,v in before.items())
                receipt['light_causal_probe'].update(physical_state_max_abs_change=delta,before_sdk_counters=counters,
                    after_sdk_counters=[int(raw._sim_step_counter),int(raw.sim._physics_step_count)])
                if delta!=0 or counters!=(int(raw._sim_step_counter),int(raw.sim._physics_step_count)):
                    raise ValueError('Light-only renderer probe changed actual physical state/counters')
                save()
            if args.contact_settings_audit:
                from unitree_g1_t2_contact_query import query_contacts
                before=physical_state();counters=(int(raw._sim_step_counter),int(raw.sim._physics_step_count))
                receipt['actual_contact_parameters']=query_contacts(raw,measured,captures/'contact_parameters.json')
                after=physical_state();delta=max(float(np.abs(after[k]-v).max()) for k,v in before.items())
                receipt['actual_contact_parameters'].update(physical_state_max_abs_change=delta,before_sdk_counters=counters,
                    after_sdk_counters=[int(raw._sim_step_counter),int(raw.sim._physics_step_count)])
                if delta!=0 or counters!=(int(raw._sim_step_counter),int(raw.sim._physics_step_count)):
                    raise ValueError('Contact query changed actual scene state/counters')
                save()
            if args.background_owner_audit:
                from unitree_g1_t2_background_query import query_background
                before_query = physical_state()
                before_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                receipt['background_owner_query'] = query_background(raw, measured, captures / 'background_owners.json')
                after_query = physical_state()
                error = max(float(np.abs(after_query[key] - value).max()) for key, value in before_query.items())
                after_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                receipt['background_owner_query'].update(physical_state_max_abs_change=error,
                    before_sdk_counters=before_counters, after_sdk_counters=after_counters)
                if error != 0 or before_counters != after_counters:
                    raise ValueError('Background owner query changed physical state or counters')
                save()
            if args.scene_overlap_audit:
                from omni.physx import get_physx_scene_query_interface
                box_pose = measured(raw.scene['brown_box'].data.root_link_pose_w).tolist()
                before_overlap = physical_state()
                overlap_hits = []
                def hit_report(hit):
                    overlap_hits.append({'collision': str(hit.collision), 'rigid_body': str(hit.rigid_body)})
                    return True
                hit_count = get_physx_scene_query_interface().overlap_box(
                    (.0999, .0999, .0999), tuple(box_pose[:3]), tuple(box_pose[3:]), hit_report, False)
                after_overlap = physical_state()
                if any(not np.array_equal(value, after_overlap[key]) for key, value in before_overlap.items()):
                    raise ValueError('Scene overlap read changed physical state')
                receipt['actual_scene_overlap_query'] = {'box_half_extent_m': [.0999]*3, 'box_pose_xyzw': box_pose,
                    'count': int(hit_count), 'hits': overlap_hits, 'query_support_only_enabled': True,
                    'physics_state_changed': False, 'sdk_counters': [int(raw._sim_step_counter), int(raw.sim._physics_step_count)]}
                save()
            if args.initial_collision_audit:
                from unitree_g1_initial_collision_query import query_initial_collisions
                before_query = physical_state()
                before_body_poses = measured(robot.data.body_link_pose_w).copy()
                before_query_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                receipt['initial_collision_query'] = query_initial_collisions(
                    raw, measured, captures / 'initial_collision_geometry.json')
                after_query = physical_state()
                query_error = max(float(np.abs(after_query[key] - value).max()) for key, value in before_query.items())
                query_error = max(query_error, float(np.abs(measured(robot.data.body_link_pose_w) - before_body_poses).max()))
                after_query_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                receipt['initial_collision_query'].update(physical_state_max_abs_change=query_error,
                    before_sdk_counters=before_query_counters, after_sdk_counters=after_query_counters)
                if query_error != 0 or before_query_counters != after_query_counters:
                    raise ValueError('Initial collision query changed physics state/counters')
                save()
            contact_events = []; contact_subscription = None; replay_contact_view = None
            if args.reset_contact_audit or args.native_action_replay:
                from omni.physx import get_physx_simulation_interface
                from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                actor_prefixes = ('/World/envs/env_0/brown_box', '/World/envs/env_0/Robot/') if args.native_action_replay else ('/World/envs/env_0/brown_box',)
                actors = [prim for prim in stage.Traverse()
                          if str(prim.GetPath()).startswith(actor_prefixes)
                          and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                reporting_actors = [prim for prim in actors if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)]
                existing_report_api = bool(reporting_actors) if args.native_action_replay else bool(actors) and len(reporting_actors) == len(actors)
                if not existing_report_api and not release_source:
                    raise ValueError('Original T2 box has no existing contact report API; no schema is added')
                receipt['observed_contact_report_actors'] = [str(prim.GetPath()) for prim in actors]
                receipt['existing_contact_report_actors'] = [str(prim.GetPath()) for prim in reporting_actors]
                receipt['original_contact_report_api_available'] = existing_report_api
                receipt['contact_report_schema_writes'] = 0
                receipt['contact_report_overflow'] = False
                def contacts(headers, data):
                    for header in headers:
                        paths = {key: str(PhysicsSchemaTools.intToSdfPath(getattr(header, key)))
                                 for key in ('actor0', 'actor1', 'collider0', 'collider1')}
                        if not any('/brown_box' in paths[key] for key in ('actor0', 'actor1')):
                            continue
                        if len(contact_events) >= (4096 if args.native_action_replay else 1024):
                            receipt['contact_report_overflow'] = True
                            continue
                        values = []
                        for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                            point = data[index]
                            values.append({key: [float(getattr(point, key)[i]) for i in range(3)]
                                           for key in ('position', 'normal', 'impulse')})
                            values[-1]['separation_m'] = float(point.separation)
                        contact_events.append({**paths, 'event': str(header.type),
                            'sdk_physics_count_at_callback': int(raw.sim._physics_step_count),
                            'env_counter_at_callback': int(raw._sim_step_counter), 'points': values})
                if existing_report_api:
                    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
                if args.native_action_replay:
                    from unitree_g1_t2_contact_query import make_box_hand_contact_view, sample_box_hand_contacts
                    before = physical_state()
                    counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                    replay_contact_view, metadata = make_box_hand_contact_view(raw)
                    after = physical_state()
                    delta = max(float(np.abs(after[key] - value).max()) for key, value in before.items())
                    metadata.update(physical_state_max_abs_change=delta, before_sdk_counters=counters,
                                    after_sdk_counters=[int(raw._sim_step_counter), int(raw.sim._physics_step_count)])
                    if delta != 0 or counters != (int(raw._sim_step_counter), int(raw.sim._physics_step_count)):
                        raise ValueError('Contact tensor view changed physical state/counters')
                    receipt['acceptance_contact_tensor_view'] = metadata
            action = torch.zeros(env.action_space.shape, device=raw.device)
            frames = []; frame_index = 0; sequence = 0; last_sha = None
            body_frames = []
            with args.output.with_suffix('.jsonl').open('x') as trace:
                for tick in range(args.ticks + 1):
                    q = measured(robot.data.joint_pos); root = measured(robot.data.root_link_pose_w)
                    gravity = measured(robot.data.projected_gravity_b)
                    if not all(np.isfinite(value).all() for value in (q, root, gravity)):
                        raise ValueError(f'Nonfinite T2 physical state at {tick}')
                    actual_steps = int(raw._sim_step_counter) - reset_counter
                    if actual_steps != tick * 4:
                        raise ValueError('T2 source integration count changed')
                    sample = {'control_tick': tick, 'physics_steps_since_reset': actual_steps,
                        'root_link_pose_w': root.tolist(), 'joint_positions': q.tolist(),
                        'upright': float(-gravity[2]), 'policy_sequence': sequence,
                        'last_action_frame': frame_index - 1 if frames else None,
                        'last_action_frame_sha256': last_sha,
                        'acceptance_truth_only': {name: {'pose': measured(raw.scene[name].data.root_pose_w).tolist(),
                            'velocity': measured(raw.scene[name].data.root_vel_w).tolist()}
                            for name in ('brown_box', 'blue_sorting_bin')}}
                    if args.native_action_replay:
                        # Actual measured SDK link poses distinguish tracking
                        # from shape/contact differences. Acceptance-only;
                        # replay runs zero VLA calls and no policy reads this.
                        sample['acceptance_truth_only']['robot_body_link_poses_xyzw'] = measured(robot.data.body_link_pose_w).tolist()
                        receipt['acceptance_body_names'] = list(robot.body_names)
                        sample['acceptance_truth_only']['box_hand_contact_tensor'] = sample_box_hand_contacts(replay_contact_view, .005)
                    if (args.reset_contact_audit or args.native_action_replay) and 'contact_sensor_brown_box' in raw.scene.sensors:
                        sensor = raw.scene['contact_sensor_brown_box']
                        sample['acceptance_truth_only']['existing_box_contact_sensor'] = {
                            'net_forces_w': measured(sensor.data.net_forces_w).tolist(),
                            'force_matrix_w': measured(sensor.data.force_matrix_w).tolist(),
                            'filter_prim_paths_expr': sensor.cfg.filter_prim_paths_expr}
                    trace.write(json.dumps(sample, allow_nan=False) + '\n'); trace.flush()
                    receipt['last_state'] = sample; receipt['completed_control_ticks'] = tick
                    receipt['actual_sdk_integrations'] = actual_steps
                    if root[2] + .795 < .35 or sample['upright'] < .5:
                        raise ValueError(f'Original T2 fall guard at {tick}')
                    if tick % 50 == 0 or tick == args.ticks:
                        rgb = measured(camera.data.output['rgb'])[:, :, :3].copy()
                        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                            raise ValueError('Original T2 camera shape/type changed')
                        Image.fromarray(rgb).save(captures / f'head_{tick:04}.png')
                        groups = {key: q[ids].tolist() for key, ids in indices.items()}
                        request = {'schema': 'mobile_observation_v1', 'profile': 'mobile_box', 'sequence_id': sequence + 1,
                            'observation': {'episode_id': args.episode_id, 'frame_id': tick,
                                'sim_time_ns': tick * 20_000_000, 'captured_at_unix_ms': time.time_ns() // 1_000_000},
                            'camera_rgb_b64': base64.b64encode(rgb.tobytes()).decode(), 'state_groups': groups}
                        (captures / f'request_{tick:04}.json').write_text(json.dumps(request, allow_nan=False))
                        np.savez_compressed(captures / f'observation_{tick:04}.npz', ego_view=rgb[None],
                                            **{key: np.asarray(values, dtype=np.float32)[None] for key, values in groups.items()})
                        if tick < args.ticks and (args.policy_port or args.policy_socket):
                            started = time.monotonic(); reply = policy_reply(args.policy_port, request, args.policy_socket)
                            (captures / f'reply_{tick:04}.json').write_text(json.dumps(reply, allow_nan=False))
                            frames = reply['frames']; frame_index = 0; sequence += 1
                            receipt['policy_calls'] += 1
                            receipt.setdefault('policy_call_seconds', []).append(time.monotonic() - started)
                        elif args.native_action_replay and tick < args.ticks:
                            frozen=json.loads(args.native_action_replay.read_text())
                            if digest(args.native_action_replay)!=args.native_action_replay_sha256:
                                raise ValueError('Frozen native diagnostic action bytes changed')
                            frames=frozen['chunks'][tick//50]['frames'];frame_index=0;sequence+=1
                        elif args.reset_contact_audit and tick == 0:
                            # Explicit ten-control diagnostic, using only reset
                            # self-state. This is neither a VLA reply nor a wait
                            # controller, and is never installed in native runtime.
                            frame = {**groups, 'navigate_mps_rps': [0.0, 0.0, 0.0], 'base_height_m': .75}
                            frames = [dict(frame) for _ in range(10)]
                            if args.body_contract_audit:
                                # Finite named diagnostic commands exercise both
                                # real Homie networks. Never a VLA/wait substitute.
                                for walk_frame in frames[5:]:
                                    walk_frame['navigate_mps_rps'] = [.1, 0.0, 0.0]
                            receipt['diagnostic_selfstate_targets'] = frame
                        print(f'G1_T2_TASK tick={tick} calls={receipt["policy_calls"]} upright={sample["upright"]:.6f}', flush=True)
                    if tick == args.ticks:
                        break
                    if not frames:
                        raise ValueError('T2 motion requires actual original N1.6 policy frames')
                    frame = frames[frame_index]
                    last_sha = hashlib.sha256(json.dumps(frame, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    action.zero_()
                    for key, ids in indices.items():
                        action[0, ids] = torch.as_tensor(frame[key], device=raw.device)
                    action[0, 43:46] = torch.as_tensor(frame['navigate_mps_rps'], device=raw.device)
                    action[0, 46] = frame['base_height_m']
                    frame_index += 1
                    if args.body_contract_audit:
                        before_q = measured(robot.data.joint_pos).copy()
                        before_dq = measured(robot.data.joint_vel).copy()
                    _, _, terminated, truncated, info = env.step(action)
                    receipt['completed_control_ticks'] = tick + 1
                    receipt['actual_sdk_integrations'] = int(raw._sim_step_counter) - reset_counter
                    if receipt['actual_sdk_integrations'] != (tick + 1) * 4:
                        raise ValueError('T2 source integration count changed after step')
                    if args.body_contract_audit:
                        before_read = physical_state()
                        before_read_counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
                        body_frame, canonical_names = body_contract_frame(
                            term, robot, measured, before_q, before_dq, tick)
                        if (any(not np.array_equal(value, physical_state()[key]) for key, value in before_read.items())
                                or before_read_counters != (int(raw._sim_step_counter), int(raw.sim._physics_step_count))):
                            raise ValueError('Body-contract reading changed physical state/counters')
                        body_frames.append(body_frame)
                        body_file = captures / 'body_contract.json'
                        body_file.write_text(json.dumps({'schema': 'g1_released_homie_actual_contract_v1',
                            'arena_commit': receipt['arena_commit'], 'lab_commit': receipt['lab_commit'],
                            'source_profile': args.source_profile, 'body_assets_sha256': FROZEN_ASSETS,
                            'source_tree_sha256': args.source_tree_sha256,
                            'source_runtime': receipt['runtime_versions'], 'joint_names': canonical_names,
                            'body_input_truth_fields': [], 'extra_inferences': 0,
                            'reading_changed_physics': False, 'frames': body_frames}, allow_nan=False))
                        receipt['body_contract_capture'] = {'file': str(body_file), 'sha256': digest(body_file),
                            'frames': len(body_frames), 'stand_controls': 5, 'walk_controls': 5,
                            'extra_inferences': 0, 'reading_changed_physics': False}
                    if bool(terminated.any()) or bool(truncated.any()):
                        receipt['episode_termination'] = {'tick': tick + 1, 'terminated': bool(terminated.any()),
                            'truncated': bool(truncated.any()), 'terminal_step_auto_reset': True,
                            'original_terms': {name: bool(raw.termination_manager.get_term(name).any())
                                for name in raw.termination_manager.active_terms}, 'reset_metrics': info.get('log', {})}
                        break
            receipt['source_scene_rollout_completed'] = True
            receipt['source_vla_executed'] = receipt['policy_calls'] > 0
            receipt['stage'] = 'completed_finite_source_diagnostic'
            receipt['trace_sha256'] = digest(args.output.with_suffix('.jsonl'))
            if args.reset_contact_audit or args.native_action_replay:
                (captures / 'initial_contacts.json').write_text(json.dumps(contact_events, allow_nan=False))
                receipt['observed_contact_report_events'] = len(contact_events)
                contact_subscription = None
            env.close()
        except BaseException as error:
            receipt['error'] = repr(error)
            raise
        finally:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arena-source', type=Path, required=True)
    parser.add_argument('--lab-source', type=Path, required=True)
    parser.add_argument('--homie-assets', type=Path, required=True)
    parser.add_argument('--task-assets', type=Path, required=True, help='Frozen Assets/Isaac/6.1/Isaac/IsaacLab directory')
    parser.add_argument('--source-profile', choices=SOURCE_PROFILES, default='development_0_3',
                        help='Independently pinned source stack; both profiles keep original T2 N1.6/Homie only')
    parser.add_argument('--source-tree-receipt', type=Path,
                        help='Optional hash-frozen host-verified source tree for read-only container mounts without accessible Git metadata')
    parser.add_argument('--source-tree-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-source', action='store_true')
    parser.add_argument('--ticks', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode-id', type=int, required=True)
    parser.add_argument('--policy-port', type=int)
    parser.add_argument('--policy-socket', type=Path, help='Private byte-preserving transport to the original local HTTP model owner')
    parser.add_argument('--contact-settings-audit', action='store_true', help='Released zero-step effective SDK contact/material/actuator inputs, no writes')
    parser.add_argument('--scene-render-audit', action='store_true', help='Read actual released light/material/render settings; exclusive zero-step diagnostic')
    parser.add_argument('--scene-light-causal-probe', action='store_true', help='Four fixed released zero-step renders distinguish legacy/modern light intensity; temporary lighting overrides restored')
    parser.add_argument('--native-action-replay', type=Path, help='One fixed 100-control source-only causal diagnostic, never task autonomy')
    parser.add_argument('--native-action-replay-sha256')
    parser.add_argument('--startup-trace', action='store_true', help='Finite startup stack/log diagnostic only')
    parser.add_argument('--reset-contact-audit', action='store_true',
                        help='Ten original source controls with fixed reset self-state targets; no VLA or native fallback')
    parser.add_argument('--initial-collision-audit', action='store_true',
                        help='Read original SDK cooking with actual reset body poses; only zero-tick or ten-control no-VLA diagnostic')
    parser.add_argument('--scene-overlap-audit', action='store_true',
                        help='Explicit source query-support diagnostic; original ten controls, no VLA or qualification')
    parser.add_argument('--body-contract-audit', action='store_true',
                        help='Released-source ten controls: five stand and five walk; read actual WBC input/output/articulation targets, no extra inference')
    parser.add_argument('--background-owner-audit', action='store_true',
                        help='Read all released background shapes/materials and actual dynamic owner mass/inertia; zero controls only')
    args = parser.parse_args()
    if (not 0 <= args.ticks <= 1500 or args.episode_id < 1 or not 0 <= args.seed < 2**32
            or (args.policy_port is not None and not 1 <= args.policy_port <= 65535)
            or (args.policy_port is not None and args.policy_socket is not None)
            or (args.ticks > 0 and not args.policy_port and not args.policy_socket and not args.reset_contact_audit and not args.native_action_replay)
            or (args.reset_contact_audit and (args.ticks != 10 or args.policy_port is not None or args.policy_socket is not None))):
        parser.error('Use0..1500 controls, positive episode and valid seed/loopback port; motion requires N1.6 service or explicit ten-control contact audit')
    if args.initial_collision_audit and (not args.run_source or args.policy_port or args.policy_socket or (args.ticks != 0 and not args.reset_contact_audit)):
        parser.error('Initial collision reading requires zero ticks or the original ten-control no-VLA diagnostic')
    if args.scene_overlap_audit and (not args.run_source or not args.reset_contact_audit or args.source_profile != 'development_0_3'):
        parser.error('Scene overlap diagnostic requires the original ten-control no-VLA fixture')
    if args.body_contract_audit and (not args.run_source or not args.reset_contact_audit
            or args.source_profile != 'release_0_2_1' or args.initial_collision_audit or args.scene_overlap_audit):
        parser.error('Body contract reading requires released-source ten-control no-VLA fixture without collision/query changes')
    if args.background_owner_audit and (not args.run_source or args.ticks != 0
            or args.source_profile != 'release_0_2_1' or args.policy_port or args.policy_socket
            or args.initial_collision_audit or args.scene_overlap_audit or args.body_contract_audit):
        parser.error('Background owner reading requires released zero-control fixture without other queries or models')
    if bool(args.native_action_replay) != bool(args.native_action_replay_sha256):
        parser.error('Native action diagnostic file/hash must be paired')
    replay = None
    if args.native_action_replay:
        if (not args.run_source or args.source_profile != 'release_0_2_1' or args.ticks != 100
                or args.policy_port or args.policy_socket or args.reset_contact_audit or args.body_contract_audit
                or args.initial_collision_audit or args.background_owner_audit or args.scene_overlap_audit
                or args.native_action_replay.stat().st_size > 256*1024
                or digest(args.native_action_replay) != args.native_action_replay_sha256):
            parser.error('Replay is exactly one hash-bound released 100-control source-only diagnostic without models/queries')
        replay=json.loads(args.native_action_replay.read_text())
        if (replay.get('schema') != 'g1_native_t2_actions_diagnostic_v1' or replay.get('qualified') is not False
                or len(replay.get('chunks',[]))!=2):
            raise ValueError('Wrong frozen native action diagnostic identity')
        for chunk in replay['chunks']:
            validate_reply({key:chunk[key] for key in ('profile','sequence_id','observation')},chunk)
    if args.contact_settings_audit and (not args.run_source or args.source_profile!='release_0_2_1' or args.ticks!=0
            or args.policy_port or args.policy_socket or args.reset_contact_audit or args.initial_collision_audit
            or args.background_owner_audit or args.scene_overlap_audit or args.body_contract_audit or args.native_action_replay):
        parser.error('Effective contact query requires exclusive released zero-step scene')
    if args.scene_render_audit and (not args.run_source or args.source_profile!='release_0_2_1' or args.ticks!=0
            or args.policy_port or args.policy_socket or args.reset_contact_audit or args.initial_collision_audit
            or args.background_owner_audit or args.scene_overlap_audit or args.body_contract_audit
            or args.native_action_replay or args.contact_settings_audit):
        parser.error('Render input query requires exclusive released zero-step scene')
    if args.scene_light_causal_probe and not args.scene_render_audit:
        parser.error('Light-only causal probe requires the exclusive released zero-step render audit')
    arena_rev, lab_rev, expected_runtime = SOURCE_PROFILES[args.source_profile]
    if bool(args.source_tree_receipt) != bool(args.source_tree_sha256):
        parser.error('Source-tree receipt and its frozen SHA256 must be provided together')
    verified_source_files = None
    if args.source_tree_receipt:
        verified_source_files = verify_source_receipt(args.source_tree_receipt, args.source_tree_sha256,
            {'arena': args.arena_source, 'lab': args.lab_source}, args.source_profile)
    else:
        source_check(args.arena_source, arena_rev); source_check(args.lab_source, lab_rev)
    for name, expected in FROZEN_ASSETS.items():
        if digest(args.homie_assets / name) != expected:
            raise ValueError(f'Frozen T2 body/controller changed: {name}')
    mesh_receipt = args.homie_assets / 'mesh_receipt.json'
    mesh_identity = json.loads(mesh_receipt.read_text())
    if mesh_identity['parent_urdf_sha256'] != FROZEN_ASSETS['g1_29dof_with_hand.urdf']:
        raise ValueError('T2 mesh receipt belongs to another URDF')
    mesh_assets = {}
    for item in mesh_identity['files']:
        local = (args.homie_assets / item['file']).resolve()
        if not local.is_relative_to(args.homie_assets.resolve()) or digest(local) != item['sha256']:
            raise ValueError('Frozen T2 kinematic mesh changed or escaped its root')
        mesh_assets[item['url']] = local
    import xml.etree.ElementTree as ET
    original_meshes = {mesh.get('filename') for mesh in ET.parse(args.homie_assets / 'g1_29dof_with_hand.urdf').getroot().iter('mesh')}
    if original_meshes != {item['file'] for item in mesh_identity['files']}:
        raise ValueError('Frozen T2 mesh receipt does not cover the original URDF')
    task_paths = {name: args.task_assets / value[0] for name, value in TASK_FILES.items()}
    for name, (_, expected) in TASK_FILES.items():
        if digest(task_paths[name]) != expected:
            raise ValueError(f'Frozen T2 task asset changed: {name}')
    # Check the actual original N1.6 group names, without using a T1 joint/config file.
    import yaml
    joint_file = args.arena_source / 'isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml'
    original_groups = yaml.safe_load(joint_file.read_text())['joints']
    if any(original_groups[key] != names for key, names in GROUP_NAMES.items()):
        raise ValueError('Original T2 policy joint order changed')
    receipt = {'schema': 'g1_t2_original_source_task_v1', 'qualified': False,
        'source_scene_rollout_completed': False, 'source_vla_executed': False, 'task_success_verified': False,
        'arena_commit': arena_rev, 'lab_commit': lab_rev, 'source_profile': args.source_profile,
        'expected_source_runtime': expected_runtime, 'model_revision': PROFILE['revision'],
        'source_tree_sha256': args.source_tree_sha256, 'verified_source_tree_files': verified_source_files,
        'body_backend': 'homie_v2', 'action_horizon': 50, 'action_period_ns': 20_000_000,
        'source_physics_hz': 200, 'source_control_hz': 50, 'source_inference_pauses': True,
        'formal_native_frequency_changed': False, 'harness_sha256': digest(Path(__file__)),
        'policy_joint_file_sha256': digest(joint_file), 'body_assets_sha256': FROZEN_ASSETS,
        'kinematic_mesh_receipt_sha256': digest(mesh_receipt), 'verified_kinematic_meshes': len(mesh_assets),
        'task_assets_sha256': {name: expected for name, (_, expected) in TASK_FILES.items()},
        'fixed_instruction': PROFILE['reference_instruction'], 'policy_calls': 0,
        'seed': args.seed, 'episode_id': args.episode_id, 'requested_control_ticks': args.ticks,
        'scope': 'original T2 brown-box-to-blue-bin scene with optional original N1.6 loopback service; source200/50 only; no native qualification'}
    if args.reset_contact_audit:
        receipt.update(scope='test-private ten-control initial-contact comparison; fixed reset self-state targets and original Homie; no VLA, wait fallback, native or task qualification',
                       reset_contact_diagnostic=True, installed_native_fallback=False)
    if args.body_contract_audit:
        receipt.update(scope='test-private released Homie numerical contract: five stand and five walk controls, fixed reset upper targets; no VLA, wait fallback or native qualification',
                       body_contract_diagnostic=True)
    if replay is not None:
        receipt.update(scope='source-only identical native action replay for causal comparison; not autonomous/VLA success',
            native_action_replay_sha256=args.native_action_replay_sha256, action_replay_controls=100,
            action_replay_chunks=replay['chunks'])
    with args.output.open('x') as output:
        def save():
            output.seek(0); output.truncate()
            json.dump(receipt, output, indent=2, allow_nan=False, default=str)
            output.write('\n'); output.flush()
        save()
        if args.run_source:
            try:
                run_source(args, receipt, save, task_paths, mesh_assets)
            except BaseException as error:
                receipt['error'] = repr(error); save()
                raise


if __name__ == '__main__':
    main()
