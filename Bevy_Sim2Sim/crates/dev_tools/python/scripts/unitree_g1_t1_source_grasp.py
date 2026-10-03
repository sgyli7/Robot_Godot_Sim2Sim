#!/usr/bin/env python3
"""Finite saved-command grasp comparison in original200/50Hz AGILE/PhysX.

Original robot, shelf, apple/plate USD and fixed initial poses are used in a
reduced scene. Saved native commands are a mechanical diagnostic, never a new
VLA forward, source expert demonstration or autonomous task success. No contact
or object truth is passed into the controller. Native formal50Hz is unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

from unitree_g1_t1_source_shelf import ARENA_REV, LAB_REV, USD_SHA, URDF_SHA, AGILE_SHA
from unitree_g1_t1_source_task import APPLE_SHA, PLATE_SHA, ARM_SUFFIXES, HAND_SUFFIXES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixture(path, expected):
    if digest(path) != expected:
        raise ValueError('Mechanical fixture SHA changed')
    value = json.loads(Path(path).read_text())
    if (value['schema'] != 'g1_t1_source_grasp_mechanical_v1'
            or value['autonomous_task_qualified'] is not False
            or value['fresh_vla_calls'] != 0
            or value['startup_ticks'] != 60
            or len(value['commands']) != 100):
        raise ValueError('Mechanical fixture scope/budget changed')
    import math
    for c in value['commands']:
        if (set(c) != {'upper_positions', 'navigation', 'pelvis_height'}
                or len(c['upper_positions']) != 28 or len(c['navigation']) != 3
                or not all(math.isfinite(x) for x in c['upper_positions'] + c['navigation'] + [c['pelvis_height']])
                or max(abs(x) for x in c['navigation']) > .01):
            raise ValueError('Mechanical original command shape/finite/navigation gate failed')
    for key in ('apple', 'plate'):
        if (len(value[key]['position']) != 3 or len(value[key]['rotation_wxyz']) != 4
                or not all(math.isfinite(x) for x in value[key]['position'] + value[key]['rotation_wxyz'])):
            raise ValueError('Mechanical original initial pose invalid')
    return value


def run(args, fixture, receipt, output):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    launch = get_isaaclab_arena_cli_parser().parse_args(['--headless', '--no-solve-relations'])
    launch.enable_cameras = False
    launch.seed = args.seed
    sys.argv = [sys.argv[0]]
    with SimulationAppContext(launch):
        try:
            import numpy as np
            import torch
            import warp as wp
            from isaaclab import sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_base import ObjectType
            from isaaclab_arena.assets.object_library import GroundPlane, Apple01ObjaverseRobolab, ClayPlatesHot3DRobolab
            from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
            from isaaclab_arena.scene.scene import Scene
            from isaaclab_arena.utils.pose import Pose
            from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
                G1_STATIC_OPEN_ARM_JOINT_POS, G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
                G1_STATIC_FINGER_STATIC_FRICTION, G1_STATIC_FINGER_DYNAMIC_FRICTION,
                G1_STATIC_FINGER_PRIM_NAME_MARKERS)
            import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as action_module
            import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as model_module
            torch._C._jit_override_can_fuse_on_gpu(False)
            def tensor(value):
                return wp.to_torch(value) if isinstance(value, wp.array) else value
            def measured(value):
                return tensor(value)[0].detach().cpu().numpy()
            factory = action_module.get_wbc_policy
            def local_factory(name, model, config, count):
                if config.wbc_version != 'agile':
                    raise ValueError('Mechanical source must retain original AGILE')
                config.wbc_model_path = str(args.agile)
                return factory(name, model, config, count)
            action_module.get_wbc_policy = local_factory
            def local_robot_path(url, **kwargs):
                if url.endswith('/g1_29dof_with_hand.urdf'):
                    return str(args.urdf)
                if url.endswith('/Arena/wbc_policy/robot_model/g1/'):
                    return str(args.urdf.parent)
                raise ValueError(f'Unexpected source asset path: {url}')
            model_module.retrieve_file_path = local_robot_path
            class SourceShelf(Object):
                def __init__(self):
                    self.spawner_cfg = sim_utils.CuboidCfg(size=(.8, 1.5, .04),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005), visible=False)
                    super().__init__(name='static_pick_place_shelf_support',
                        prim_path='{ENV_REGEX_NS}/static_pick_place_shelf_support',
                        object_type=ObjectType.SPAWNER,
                        initial_pose=Pose(position_xyz=(.62, 0., .745), rotation_xyzw=(0., 0., 0., 1.)),
                        tags=['background', 'procedural'])
            embodiment = G1WBCAgileJointEmbodiment(enable_cameras=False, lock_waist=True)
            embodiment.set_initial_pose(Pose(position_xyz=(.25, .08, .795), rotation_xyzw=(0., 0., 0., 1.)))
            embodiment.set_joint_initial_pos(G1_STATIC_OPEN_ARM_JOINT_POS)
            embodiment.scene_config.robot.spawn.usd_path = str(args.usd)
            embodiment.set_finger_contact_friction(material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
                static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
                dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
                prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS)
            Apple01ObjaverseRobolab.usd_path = str(args.apple)
            ClayPlatesHot3DRobolab.usd_path = str(args.plate)
            apple = Apple01ObjaverseRobolab(scale=(.009,) * 3)
            plate = ClayPlatesHot3DRobolab(scale=(.5,) * 3)
            for name, asset in [('apple', apple), ('plate', plate)]:
                pose = fixture[name]; w, x, y, z = pose['rotation_wxyz']
                asset.set_initial_pose(Pose(position_xyz=tuple(pose['position']), rotation_xyzw=(x, y, z, w)))
            def configure(cfg):
                if cfg.sim.dt != .005 or cfg.decimation != 4:
                    raise ValueError('Original source200/50Hz changed')
                cfg.episode_length_s = 6.
                cfg.seed = args.seed
                return cfg
            description = IsaacLabArenaEnvironment(name='g1_t1_source_saved_grasp_probe',
                scene=Scene(assets=[GroundPlane(), SourceShelf(), apple, plate]),
                embodiment=embodiment, env_cfg_callback=configure)
            env = ArenaEnvBuilder(description, launch).make_registered()
            raw = env.unwrapped; robot = raw.scene['robot']; term = raw.action_manager.get_term('g1_action')
            names = list(robot.joint_names)
            groups = {f'{side}_arm': [names.index(f'{side}_{s}_joint') for s in ARM_SUFFIXES] for side in ('left', 'right')}
            groups.update({f'{side}_hand': [names.index(f'{side}_hand_{s}_joint') for s in HAND_SUFFIXES] for side in ('left', 'right')})
            receipt.update(runtime_build=Path('/isaac-sim/VERSION').read_text().strip(),
                source_physics_hz=200, source_control_hz=50, source_physics_steps_per_control=4,
                joint_names=names, body_names=list(robot.body_names),
                original_actuators={key: value.to_dict() for key, value in robot.cfg.actuators.items()},
                ground_material=raw.cfg.sim.physics_material.to_dict(),
                actual_source_body_properties={key: measured(getattr(robot.data, key)).tolist()
                    for key in ('body_mass', 'body_inertia', 'body_com_pose_b')})
            env.reset(seed=args.seed)
            counter0 = int(raw._sim_step_counter)
            defaults = tensor(robot.data.default_joint_pos)
            action = torch.zeros(env.action_space.shape, device=raw.device)
            trace = args.output.with_suffix('.jsonl')
            with trace.open('x') as log:
                for tick in range(161):
                    applied = None
                    if tick:
                        action.zero_()
                        if tick <= 60:
                            action[:, :43] = defaults
                            action[0, 46] = .75
                        else:
                            applied = fixture['commands'][tick - 61]
                            values = np.asarray(applied['upper_positions'], dtype=np.float32)
                            for key, lo, hi in [('left_arm', 0, 7), ('left_hand', 7, 14), ('right_arm', 14, 21), ('right_hand', 21, 28)]:
                                action[0, groups[key]] = torch.as_tensor(values[lo:hi], device=raw.device)
                            action[0, 43:46] = torch.as_tensor(applied['navigation'], device=raw.device)
                            action[0, 46] = applied['pelvis_height']
                        _, _, terminated, truncated, _ = env.step(action)
                        receipt['actual_control_ticks'] = tick
                        receipt['actual_physics_steps'] = int(raw._sim_step_counter) - counter0
                        if receipt['actual_physics_steps'] != 4 * tick:
                            raise ValueError('Source original integration count diverged')
                        if bool(terminated.any()) or bool(truncated.any()):
                            raise ValueError(f'Unexpected source mechanical auto-reset at{tick}')
                    state = {key: measured(getattr(robot.data, key)) for key in
                        ('root_link_pose_w', 'root_link_vel_w', 'joint_pos', 'joint_vel', 'projected_gravity_b', 'body_link_pose_w')}
                    if not all(np.isfinite(value).all() for value in state.values()):
                        raise ValueError('Nonfinite source mechanical state')
                    sample = {'control_tick': tick, 'source_physics_steps_since_reset': 4 * tick,
                        **{key: value.tolist() for key, value in state.items()},
                        'upright': float(-state['projected_gravity_b'][2]),
                        'processed_joint_targets': term.processed_actions[0].detach().cpu().tolist(),
                        'applied_offline_command': applied,
                        'acceptance_truth_only': {name: {
                            'pose': measured(raw.scene[asset.name].data.root_pose_w).tolist(),
                            'velocity': measured(raw.scene[asset.name].data.root_vel_w).tolist()}
                            for name, asset in [('apple', apple), ('plate', plate)]}}
                    log.write(json.dumps(sample, allow_nan=False) + '\n');log.flush()
                    if tick % 40 == 0:
                        print(f'G1_T1_SOURCE_GRASP tick={tick} physics={4*tick}', flush=True)
                    if state['root_link_pose_w'][2] < .35 or sample['upright'] < .5:
                        raise ValueError(f'Source mechanical fall at{tick}')
            receipt['finite_source_comparison_completed'] = True
            receipt['trace_sha256'] = digest(trace)
            env.close()
        except BaseException as error:
            receipt['error'] = repr(error)
            raise
        finally:
            output.seek(0);output.truncate();json.dump(receipt,output,indent=2,allow_nan=False,default=str);output.write('\n');output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('usd', 'urdf', 'agile', 'apple', 'plate', 'fixture', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--fixture-sha256', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args();fixture = load_fixture(args.fixture, args.fixture_sha256)
    receipt = {'schema':'g1_t1_source_saved_grasp_comparison_v1','autonomous_task_qualified':False,
        'native_task_qualified':False,'fresh_vla_calls':0,'fresh_images':0,
        'source_expert_executed':False,'saved_command_diagnostic':True,
        'requested_control_ticks':160,'maximum_source_physics_steps':640,
        'actual_control_ticks':0,'actual_physics_steps':0,'finite_source_comparison_completed':False,
        'arena_commit':ARENA_REV,'lab_commit':LAB_REV,'harness_sha256':digest(__file__),
        'fixture_sha256':args.fixture_sha256,'original_finger_friction_preserved':True,
        'environment_translation_source':[0.,0.,.795],
        'scope':'reduced original ground/robot/source shelf/original apple and plate; no full background, cameras or task policy; source200/50 original, native50unchanged'}
    with args.output.open('x') as output:
        for name, expected in [('usd',USD_SHA),('urdf',URDF_SHA),('agile',AGILE_SHA),('apple',APPLE_SHA),('plate',PLATE_SHA)]:
            if digest(getattr(args,name)) != expected:
                raise ValueError(f'Frozen source artifact changed:{name}')
        json.dump(receipt,output,indent=2);output.flush();run(args,fixture,receipt,output)

if __name__ == '__main__':
    main()
