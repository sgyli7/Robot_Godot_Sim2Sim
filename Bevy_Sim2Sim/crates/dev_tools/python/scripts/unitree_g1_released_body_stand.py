#!/usr/bin/env python3
"""One released-source flat-floor Homie standing reference, never task success.

Fixed 1500 controls, original 200/50 PhysX/IdealPD. Native counterpart uses
the same reset joint/root/command/material configuration at 50/50. No VLA,
training, runtime fallback or parameter search exists in this diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from unitree_g1_source_stand import FROZEN_ASSETS, digest
from unitree_g1_t2_source_task import SOURCE_PROFILES, verify_source_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arena-source', 'lab-source', 'homie-assets', 'source-tree-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-tree-sha256', required=True)
    args = parser.parse_args()
    profile = 'release_0_2_1'
    count = verify_source_receipt(args.source_tree_receipt, args.source_tree_sha256,
        {'arena': args.arena_source, 'lab': args.lab_source}, profile)
    for name, expected in FROZEN_ASSETS.items():
        if digest(args.homie_assets / name) != expected:
            raise ValueError('Original released body/weights changed: ' + name)
    receipt = {'schema': 'g1_released_homie_source_stand30_v1', 'qualified': False,
        'source_profile': profile, 'arena_commit': SOURCE_PROFILES[profile][0],
        'lab_commit': SOURCE_PROFILES[profile][1], 'source_tree_sha256': args.source_tree_sha256,
        'verified_source_files': count, 'body_assets_sha256': FROZEN_ASSETS,
        'harness_sha256': digest(Path(__file__)), 'requested_controls': 1500,
        'source_physics_hz': 200, 'source_control_hz': 50, 'policy_calls': 0,
        'vla_calls': 0, 'completed_controls': 0, 'actual_integrations': 0,
        'navigation': [0., 0., 0.], 'height_command': .75,
        'root_position_source': [0., .18, .795], 'root_rotation_xyzw': [0., 0., 0., 1.],
        'upper_targets': [0.] * 28, 'stage': 'verified_source',
        'scope': 'flat floor without original background/objects; source reference only; no T0/task/1x qualification'}
    with args.output.open('x') as output:
        def save():
            output.seek(0); output.truncate()
            json.dump(receipt, output, indent=2, allow_nan=False, default=str)
            output.write('\n'); output.flush()
        save()
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
        launch = get_isaaclab_arena_cli_parser().parse_args(['--headless', '--no-solve-relations'])
        sys.argv = [sys.argv[0]]
        with SimulationAppContext(launch):
            try:
                import numpy as np
                import torch
                import warp as wp
                from isaaclab_arena.assets.object_library import GroundPlane
                from isaaclab_arena.embodiments.g1.g1 import G1WBCJointEmbodiment
                from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
                from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
                from isaaclab_arena.scene.scene import Scene
                from isaaclab_arena.utils.pose import Pose
                import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as action_module
                import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as model_module
                import isaaclab
                for module, root in ((sys.modules[ArenaEnvBuilder.__module__], args.arena_source), (isaaclab, args.lab_source)):
                    if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                        raise ValueError('Released body import escaped verified source')
                build = Path('/isaac-sim/VERSION').read_text().strip()
                if build != SOURCE_PROFILES[profile][2]:
                    raise ValueError('Released body SDK identity changed')
                receipt['runtime_build'] = build
                original_factory = action_module.get_wbc_policy
                def local_factory(name, model, config, count):
                    if config.wbc_version != 'homie_v2':
                        raise ValueError('Released body requires original Homie only')
                    config.wbc_model_path = ','.join(str(args.homie_assets / name) for name in ('stand.onnx', 'walk.onnx'))
                    return original_factory(name, model, config, count)
                action_module.get_wbc_policy = local_factory
                def local_robot(url, **kwargs):
                    if url.endswith('/g1_29dof_with_hand.urdf'):
                        return str(args.homie_assets / 'g1_29dof_with_hand.urdf')
                    if url.endswith('/Arena/wbc_policy/robot_model/g1/'):
                        return str(args.homie_assets)
                    raise ValueError('Unexpected released kinematic URL: ' + url)
                model_module.retrieve_file_path = local_robot
                torch._C._jit_override_can_fuse_on_gpu(False)
                ground = GroundPlane()
                material = ground.spawner_cfg.physics_material
                if (material.static_friction, material.dynamic_friction, material.restitution) != (.5, .5, 0.):
                    raise ValueError('Original flat-floor material differs from native reference')
                receipt['ground_material_cfg'] = material.to_dict()
                body = G1WBCJointEmbodiment(enable_cameras=False,
                    initial_pose=Pose(position_xyz=(0., .18, .795), rotation_xyzw=(0., 0., 0., 1.)))
                body.scene_config.robot.spawn.usd_path = str(args.homie_assets / 'g1_29dof_with_hand_rev_1_0.usd')
                description = IsaacLabArenaEnvironment(name='g1_released_homie_flat_stand30',
                    scene=Scene(assets=[ground]), embodiment=body)
                env = ArenaEnvBuilder(description, launch).make_registered()
                raw = env.unwrapped; robot = raw.scene['robot']; term = raw.action_manager.get_term('g1_action')
                if raw.cfg.sim.dt != .005 or raw.cfg.decimation != 4 or raw.cfg.episode_length_s <= 30:
                    raise ValueError('Original released body frequency/horizon changed')
                def measured(value):
                    if hasattr(value, 'torch'): value = value.torch
                    elif isinstance(value, wp.array): value = wp.to_torch(value)
                    return value[0].detach().cpu().numpy()
                env.reset(seed=42)
                initial_counter = int(raw._sim_step_counter)
                initial_root = measured(robot.data.root_link_pose_w)
                if not np.allclose(initial_root[:3], [0., .18, .795], rtol=0, atol=1e-7):
                    raise ValueError('Source reset root differs from frozen native reference')
                receipt.update(joint_names=list(robot.joint_names), body_names=list(robot.body_names),
                    initial_q=measured(robot.data.joint_pos).tolist(), initial_root=initial_root.tolist(),
                    authored_actuators={name: cfg.to_dict() for name, cfg in robot.cfg.actuators.items()},
                    compiled_plant={name: measured(getattr(robot.data, prop)).tolist() for name, prop in (
                        ('mass', 'body_mass'), ('com_pose_xyzw', 'body_com_pose_b'), ('inertia_matrix', 'body_inertia'))},
                    compiled_plant_body_order='body_names via reordered public data properties',
                    stage='source_standing')
                save()
                with args.output.with_suffix('.jsonl').open('x') as trace:
                    for tick in range(1501):
                        root = measured(robot.data.root_link_pose_w)
                        q = measured(robot.data.joint_pos); dq = measured(robot.data.joint_vel)
                        gravity = measured(robot.data.projected_gravity_b)
                        if not all(np.isfinite(value).all() for value in (root, q, dq, gravity)):
                            raise ValueError('Nonfinite source body state')
                        actual_steps = int(raw._sim_step_counter) - initial_counter
                        if actual_steps != tick * 4:
                            raise ValueError('Source standing integration count changed')
                        sample = {'control_tick': tick, 'physics_steps': actual_steps,
                            'root_link_pose_w': root.tolist(), 'joint_positions': q.tolist(), 'joint_velocities': dq.tolist(),
                            'root_link_velocity_w': measured(robot.data.root_link_vel_w).tolist(),
                            'upright': float(-gravity[2]), 'processed_targets': measured(term.processed_actions).tolist()}
                        trace.write(json.dumps(sample, allow_nan=False) + '\n'); trace.flush()
                        receipt.update(completed_controls=tick, actual_integrations=actual_steps, last_state=sample)
                        if root[2] < .35 or sample['upright'] < .5:
                            receipt['termination'] = 'fell'; break
                        if tick == 1500:
                            receipt['termination'] = 'completed'; break
                        action = torch.zeros(env.action_space.shape, device=raw.device); action[0, 46] = .75
                        _, _, terminated, truncated, _ = env.step(action)
                        receipt['policy_calls'] += 1
                        if bool(terminated.any()) or bool(truncated.any()):
                            raise ValueError('Unexpected source standing auto-reset')
                receipt.update(stage='completed_finite_source_reference', trace_sha256=digest(args.output.with_suffix('.jsonl')))
                save(); env.close()
            except BaseException as error:
                receipt['error'] = repr(error); raise
            finally:
                save()


if __name__ == '__main__':
    main()
