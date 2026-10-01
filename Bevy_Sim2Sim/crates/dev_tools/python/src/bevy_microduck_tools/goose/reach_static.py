"""Bounded Goose FK/payload static screen. This never qualifies free-root motion.

Runs without integration or collision solving. The root is prescribed for this
screen, and the reported force demand still needs actual foot-contact support.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from .artifacts import JOINT_ORDER, sha256, write_json


def generalized_gravity(model, data, *, payload_mass_kg=0., grip_site_id=None):
    """Required generalized holding force from published SI masses and COMs."""
    torque = np.zeros(model.nv)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    gravity = np.asarray(model.opt.gravity)
    for bid in range(1, model.nbody):
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, bid)
        torque -= jacp.T @ (model.body_mass[bid] * gravity)
    if payload_mass_kg:
        if grip_site_id is None:
            raise ValueError("Payload requires a real contract grip site")
        mujoco.mj_jacSite(model, data, jacp, jacr, grip_site_id)
        torque -= jacp.T @ (payload_mass_kg * gravity)
    return torque


def reduce_named_torques(model, generalized):
    """Ideal equal-crank virtual-work reduction: rotor+jaw-coupler on axis5."""
    result = np.array([generalized[int(model.joint(n).dofadr[0])] for n in JOINT_ORDER])
    result[5] += generalized[int(model.joint("beak_input_rotor").dofadr[0])]
    result[5] -= generalized[int(model.joint("beak_coupler_link").dofadr[0])]
    return result


def _convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1]+upper[:-1])


def inside_support(point, hull):
    edges = np.roll(hull, -1, axis=0)-hull
    relative = np.asarray(point)-hull
    cross = edges[:, 0]*relative[:, 1]-edges[:, 1]*relative[:, 0]
    return bool((cross >= -1e-12).all())


def screen(model_path: Path, contract_path: Path, output: Path, *, symmetric_squat=False):
    contract = json.loads(contract_path.read_text())
    if sha256(model_path) != contract["model_sha256"]:
        raise ValueError("Static screen model/contract mismatch")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ids = np.array([int(model.joint(n).qposadr[0]) for n in JOINT_ORDER])
    neutral = np.array([j["q_neutral_rad"] for j in contract["joints"]])
    continuous = np.array([j["continuous_design_limit_nm"] for j in contract["joints"]])
    grip_id = model.site("grip").id
    hull = _convex_hull(list(itertools.chain.from_iterable(contract["contact_hulls_world_at_zero_m"].values())))
    axes = [np.linspace(*contract["joints"][index]["range_rad"], 7) for index in (1, 2, 3)]
    rows = []
    squat_values = np.linspace(0., .9, 7) if symmetric_squat else [0.]
    for pose_index, pose in enumerate(itertools.product(squat_values, itertools.product(*axes))):
        squat, angles = pose
        mujoco.mj_resetData(model, data)
        q = neutral.copy()
        q[[1, 2, 3]] = angles
        q[5] = .15  # Prescribed jaw pose, not an executed clamp or grasp.
        # Mirror angular coordinates because the published left/right pitch
        # axes are opposite. Hip/knee/ankle sum to zero in world pitch.
        q[[8, 9, 10]] = [-squat, 2*squat, -squat]
        q[[14, 15, 16]] = [squat, -2*squat, squat]
        ranges = np.asarray([j["range_rad"] for j in contract["joints"]])
        if not ((q >= ranges[:,0]).all() and (q <= ranges[:,1]).all()):
            raise ValueError("Screen pose outside named active-axis limits")
        data.qpos[ids] = q
        for link in contract["passive_linkage_joints"]:
            src = int(model.joint(link["mimic_joint"]).qposadr[0])
            dest = int(model.joint(link["name"]).qposadr[0])
            data.qpos[dest] = link["mimic_multiplier"]*data.qpos[src]+link["mimic_offset_rad"]
        mujoco.mj_kinematics(model, data)
        foot_offsets = []
        for joint_index in (11, 17):
            name = JOINT_ORDER[joint_index]
            bid = model.body(name).id
            target = np.asarray(contract["joints"][joint_index]["pivot_world_at_zero_m"])
            foot_offsets.append(target-data.xpos[bid])
            if np.max(np.abs(data.xmat[bid].reshape(3,3)-np.eye(3))) > 1e-10:
                raise ValueError("Symmetric squat did not preserve foot orientation")
        if np.linalg.norm(foot_offsets[0]-foot_offsets[1]) > 1e-10:
            raise ValueError("Symmetric squat cannot retain both foot origins")
        data.qpos[:3] += np.mean(foot_offsets,axis=0)
        mujoco.mj_kinematics(model,data)
        mujoco.mj_comPos(model, data)
        grip = data.site_xpos[grip_id].copy()
        first_moment = (model.body_mass[:, None]*data.xipos).sum(axis=0)
        robot_mass = float(model.body_mass.sum())
        loads = []
        for payload in (.1, .2, .3):
            generalized = generalized_gravity(model, data, payload_mass_kg=payload, grip_site_id=grip_id)
            torque = reduce_named_torques(model, generalized)
            combined_com = (first_moment+payload*grip)/(robot_mass+payload)
            margins = continuous-np.abs(torque)
            loads.append({"payload_mass_kg": payload, "required_joint_holding_torque_nm": torque.tolist(),
                          "continuous_margin_nm": margins.tolist(), "within_continuous_caps": bool((margins >= 0).all()),
                          "combined_com_world_m": combined_com.tolist(), "com_projection_inside_double_foot_hull": inside_support(combined_com[:2], hull),
                          "required_ground_support_force_n": (robot_mass+payload)*9.81,
                          "root_generalized_holding_wrench": generalized[:6].tolist()})
        rows.append({"pose_index": pose_index, "q_rad": q.tolist(), "grip_world_m": grip.tolist(), "squat_rad": float(squat), "prescribed_root_translation_m": data.qpos[:3].tolist(), "loads": loads})
    # Keep every tested pose; a useful low-height result still cannot claim
    # interference-free motion or actual payload contact in either engine.
    low = [r for r in rows if .005 <= r["grip_world_m"][2] <= .05]
    best = sorted(rows, key=lambda r: abs(r["grip_world_m"][2]-.025))[:10]
    report = {"schema": "goose_reach_static_screen_v1", "candidate": contract["candidate"],
              "model_sha256": sha256(model_path), "contract_sha256": sha256(contract_path),
              "screen_code_sha256": sha256(Path(__file__)), "engine": "mujoco_kinematics_only", "engine_version": mujoco.__version__,
              "poses_checked": len(rows), "root_prescribed": True,
              "symmetric_squat": symmetric_squat, "foot_frame_preservation": "both ankle_roll origins and orientations retained by prescribed root translation", "physics_integrations": 0, "optimizer_updates": 0,
              "grip_authority": contract["sites"]["grip"], "support_hull_world_xy_m": hull.tolist(),
              "minimum_grip_z_m": min(r["grip_world_m"][2] for r in rows),
              "low_grip_pose_count_5_to_50mm": len(low),
              "low_grip_static_candidates": {str(payload): sum(all((r["loads"][index]["within_continuous_caps"],r["loads"][index]["com_projection_inside_double_foot_hull"])) for r in low) for index,payload in enumerate((.1,.2,.3))},
              "closest_25mm_poses": best, "rows": rows, "qualified": False,
              "collision_interference": "not_checked; original model action interference qualification has not passed",
              "free_root_ground_reach": "not_checked", "physical_contact_support": "not_checked",
              "object_grasp_lift_carry_place": "not_checked", "jaw_loop_dynamics": "not_checked",
              "jaw_torque_reduction": "ideal virtual work only; no gearbox loss/thermal/hardware claim",
              "load_path_scope": "all published body gravity plus point payload at actual grip; no base support in a dynamic run"}
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symmetric-squat", action="store_true")
    args = parser.parse_args(argv)
    model, contract, output = (p.resolve() for p in (args.model, args.contract, args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    result = screen(model, contract, output, symmetric_squat=args.symmetric_squat)
    print(f"STATUS STATIC_SCREEN_ONLY poses={result['poses_checked']} low={result['low_grip_pose_count_5_to_50mm']} {output}")


if __name__ == "__main__":
    main()
