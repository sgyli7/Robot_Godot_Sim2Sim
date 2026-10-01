#!/usr/bin/env python3
"""Bounded original T1 AGILE/PhysX standing comparison with source shelf.

Uses the pinned T1 Arena and Lab in the original ARM64 6.0.0-dev2 image. This
reduced scene uniformly translates source ground/robot/shelf by +0.795 m. It is
not a full original visual task, VLA rollout, or Bevy task qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


ARENA_REV = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
LAB_REV = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
USD_SHA = "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"
URDF_SHA = "3dcb9c361753f464fa1f0238cdf800af842909628fd153733075607881c12d62"
AGILE_SHA = "c8e30ec353bbba464298aeb051963cee17e1f1a484081c726ebb1d7513570da6"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, receipt, output):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    launch = get_isaaclab_arena_cli_parser().parse_args(["--headless", "--no-solve-relations"])
    launch.enable_cameras = False
    launch.seed = args.seed
    sys.argv = [sys.argv[0]]
    with SimulationAppContext(launch):
        try:
            import numpy as np
            import torch
            import warp as wp

            # The original cu128 image's GPU TorchScript fusion requests an
            # unsupported GB10 NVRTC architecture. Keep the original operators
            # and disable only that process-local GPU fusion optimization.
            torch._C._jit_override_can_fuse_on_gpu(False)
            receipt["torch_gpu_jit_fusion_enabled"] = torch._C._jit_can_fuse_on_gpu()
            receipt["torch_runtime"] = {"version": torch.__version__, "path": torch.__file__,
                "cuda": torch.version.cuda, "device_capability": list(torch.cuda.get_device_capability())}

            def tensor(value):
                return wp.to_torch(value) if isinstance(value, wp.array) else value

            def first_numpy(value):
                return tensor(value)[0].detach().cpu().numpy()
            from isaaclab import sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_base import ObjectType
            from isaaclab_arena.assets.object_library import GroundPlane
            from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
            from isaaclab_arena.scene.scene import Scene
            from isaaclab_arena.utils.pose import Pose
            from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
                G1_STATIC_FINGER_DYNAMIC_FRICTION, G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
                G1_STATIC_FINGER_PRIM_NAME_MARKERS, G1_STATIC_FINGER_STATIC_FRICTION,
                G1_STATIC_OPEN_ARM_JOINT_POS,
            )
            import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as action_module
            import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as model_module

            # Redirect only original byte-verified storage locations. Original
            # policy construction, robot-model class, observations and gains remain.
            original_factory = action_module.get_wbc_policy
            def local_policy_factory(robot_name, robot_model, config, num_envs):
                if config.wbc_version != "agile":
                    raise ValueError("T1 source must retain the AGILE contract")
                config.wbc_model_path = str(args.agile)
                return original_factory(robot_name, robot_model, config, num_envs)
            action_module.get_wbc_policy = local_policy_factory
            def local_robot_path(url, **kwargs):
                if url.endswith("/g1_29dof_with_hand.urdf"):
                    return str(args.urdf)
                if url.endswith("/Arena/wbc_policy/robot_model/g1/"):
                    return str(args.urdf.parent)
                raise ValueError(f"Unexpected source robot asset resolution: {url}")
            model_module.retrieve_file_path = local_robot_path

            class SourceShelf(Object):
                def __init__(self):
                    self.spawner_cfg = sim_utils.CuboidCfg(
                        size=(0.8, 1.5, 0.04),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                        visible=False,
                    )
                    super().__init__(name="static_pick_place_shelf_support",
                        prim_path="{ENV_REGEX_NS}/static_pick_place_shelf_support",
                        object_type=ObjectType.SPAWNER,
                        initial_pose=Pose(position_xyz=(0.62, 0., 0.745), rotation_xyzw=(0., 0., 0., 1.)),
                        tags=["background", "procedural"])

            embodiment = G1WBCAgileJointEmbodiment(enable_cameras=False, lock_waist=True)
            embodiment.set_initial_pose(Pose(position_xyz=(0.25, 0.08, 0.795), rotation_xyzw=(0., 0., 0., 1.)))
            embodiment.set_joint_initial_pos(G1_STATIC_OPEN_ARM_JOINT_POS)
            embodiment.scene_config.robot.spawn.usd_path = str(args.usd)
            embodiment.set_finger_contact_friction(
                material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
                static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
                dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
                prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS,
            )
            def configure(cfg):
                if cfg.sim.dt != 0.005 or cfg.decimation != 4:
                    raise ValueError("Original source must use 200 Hz physics / 50 Hz WBC")
                cfg.episode_length_s = (args.ticks + 100) / 50
                cfg.seed = args.seed
                return cfg
            assets = [GroundPlane()]
            if args.shelf:
                assets.append(SourceShelf())
            description = IsaacLabArenaEnvironment(name="g1_t1_source_shelf_stand_probe",
                scene=Scene(assets=assets), embodiment=embodiment, env_cfg_callback=configure)
            env = ArenaEnvBuilder(description, launch).make_registered()
            raw = env.unwrapped
            robot = raw.scene["robot"]
            term = raw.action_manager.get_term("g1_action")
            policy = term.wbc_policy.lower_body_policy
            if digest(policy.session.session._model_path) != AGILE_SHA:
                raise ValueError("Loaded source AGILE model identity changed")
            receipt.update(source_physics_hz=1 / raw.cfg.sim.dt, source_control_hz=1 / (raw.cfg.sim.dt * raw.cfg.decimation),
                physics_steps_per_control=raw.cfg.decimation, joint_names=list(robot.joint_names), body_names=list(robot.body_names),
                original_actuators={name: value.to_dict() for name, value in robot.cfg.actuators.items()},
                ground_material=raw.cfg.sim.physics_material.to_dict(),
                runtime_build=Path("/isaac-sim/VERSION").read_text().strip())
            limits = first_numpy(robot.data.joint_pos_limits)
            receipt["joint_limits"] = limits.tolist()
            env.reset(seed=args.seed)
            action = torch.zeros(env.action_space.shape, device=raw.device)
            action[:, :robot.num_joints] = tensor(robot.data.default_joint_pos)
            action[:, -4] = 0.75
            receipt["frozen_action"] = action[0].detach().cpu().tolist()
            receipt["default_joint_positions"] = first_numpy(robot.data.default_joint_pos).tolist()
            trace = args.output.with_suffix(".jsonl")
            with trace.open("x") as log:
                for tick in range(args.ticks + 1):
                    if tick:
                        _, _, terminated, truncated, _ = env.step(action)
                        if bool(terminated.any()) or bool(truncated.any()):
                            raise ValueError(f"Unexpected source episode reset at tick {tick}")
                    fields = {key: first_numpy(getattr(robot.data, key)) for key in
                        ("root_link_pose_w", "root_link_vel_w", "joint_pos", "joint_vel", "projected_gravity_b",
                         "computed_torque", "applied_torque", "body_link_pose_w")}
                    if not all(np.isfinite(value).all() for value in fields.values()):
                        raise ValueError(f"Nonfinite source state at tick {tick}")
                    violations = np.flatnonzero((fields["joint_pos"] < limits[:,0] - .001) | (fields["joint_pos"] > limits[:,1] + .001))
                    sample = {"control_tick":tick, "source_physics_steps_since_reset":tick * raw.cfg.decimation,
                        **{key:value.tolist() for key,value in fields.items()},
                        "upright":float(-fields["projected_gravity_b"][2]),
                        "processed_joint_targets":term.processed_actions[0].detach().cpu().tolist(),
                        "joint_limit_violations":[{"joint":robot.joint_names[i],"q":float(fields["joint_pos"][i]),
                            "limits":limits[i].tolist()} for i in violations]}
                    log.write(json.dumps(sample,allow_nan=False)+'\n');log.flush()
                    receipt["completed_control_ticks"] = tick
                    receipt["last_state"] = sample
                    if tick == 0:
                        receipt["initial_state"] = sample
                    if tick % 100 == 0:
                        print(f"G1_T1_SOURCE tick={tick} shelf={args.shelf} height={fields['root_link_pose_w'][2]:.6f}",flush=True)
                    if violations.size or fields["root_link_pose_w"][2] < .35 or sample["upright"] < .5:
                        receipt["first_failure"] = sample
                        raise ValueError(f"Source stand finite/fall/limit guard failed at tick {tick}")
            receipt["source_standing_budget_passed"] = True
            receipt["trace_sha256"] = digest(trace)
            env.close()
        except BaseException as error:
            receipt["error"] = repr(error)
            raise
        finally:
            output.seek(0);output.truncate();json.dump(receipt,output,indent=2,allow_nan=False,default=str)
            output.write('\n');output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("usd", "urdf", "agile", "output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--ticks",type=int,default=1500)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--shelf",action="store_true")
    args = parser.parse_args()
    if not 1 <= args.ticks <= 1500:
        parser.error("Source comparison is bounded to 1..1500 control ticks")
    receipt = {"schema":"g1_t1_original_source_shelf_stand_v1","qualified":False,"source_task_rollout_verified":False,
        "source_standing_budget_passed":False,"arena_commit":ARENA_REV,"lab_commit":LAB_REV,
        "original_image_tag":"6.0.0-dev2","environment_translation_source":[0.,0.,.795],
        "shelf_enabled":args.shelf,"requested_control_ticks":args.ticks,"harness_sha256":digest(__file__),
        "scope":"reduced translated ground/robot/source_shelf only; no full background, props, cameras or VLA"}
    with args.output.open("x") as output:
        try:
            for path,expected in [(args.usd,USD_SHA),(args.urdf,URDF_SHA),(args.agile,AGILE_SHA)]:
                if digest(path)!=expected:
                    raise ValueError(f"Frozen source artifact changed: {path}")
            json.dump(receipt,output,indent=2);output.flush()
            run(args,receipt,output)
        except BaseException as error:
            receipt["error"] = repr(error)
            output.seek(0);output.truncate();json.dump(receipt,output,indent=2,default=str);output.flush()
            raise


if __name__=="__main__":
    main()
