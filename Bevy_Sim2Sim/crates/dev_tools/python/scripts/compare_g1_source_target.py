#!/usr/bin/env python3
"""Read-only Homie source/target evidence comparison; never starts a simulator.

Authored means configuration or a derivation from authored geometry. Compiled
means an existing native-state receipt (including calculations from its poses).
Unknown never becomes zero. This tool cannot award standing qualification: it
reports the frozen harness verdict, drift and limit violations independently.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re


ARENA_REV = "7d75c95934c51a0318c957a8831e862ca43c53b5"
USD_SHA = "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"
MODEL_SHA = "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18"
PHYSICS_SHA = "571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd"
PARAMETERS = ("stiffness", "damping", "effort_limit", "velocity_limit", "armature", "friction")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path, expected=None):
    actual = digest(path)
    if expected and actual != expected:
        raise ValueError(f"Evidence identity differs: {path}: {actual}")
    return json.loads(path.read_text()), {"path": str(path.resolve()), "sha256": actual}


def decode_ast(node):
    """Read constructor kwargs without importing or executing upstream code."""
    if isinstance(node, ast.Call):
        return {arg.arg: decode_ast(arg.value) for arg in node.keywords}
    if isinstance(node, ast.Dict):
        return {decode_ast(k): decode_ast(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [decode_ast(value) for value in node.elts]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return {"unevaluated_source": ast.unparse(node)}


def authored_contract(arena, workspace):
    g1_path = arena / "isaaclab_arena/embodiments/g1/g1.py"
    order_path = arena / "isaaclab_arena_g1/g1_env/config/loco_manip_g1_joints_order_43dof.yaml"
    contract_path = workspace / "crates/modules/robot/src/g1/contract.rs"
    actuator_path = workspace / "crates/modules/robot/src/g1/actuator.rs"
    cfg_nodes = [n.value for n in ast.parse(g1_path.read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "G1_CFG" for t in n.targets)]
    if len(cfg_nodes) != 1:
        raise ValueError("Expected exactly one authored G1_CFG")
    cfg = decode_ast(cfg_nodes[0])
    order = sorted((int(i), name) for name, i in re.findall(r"^'([^']+)':\s*(\d+)$", order_path.read_text(), re.M))
    if [i for i, _ in order] != list(range(43)):
        raise ValueError("Source WBC order must name each of 43 joints once")
    names = [name for _, name in order]
    target_text = contract_path.read_text()
    target_names = re.findall(r'"([^"]+)"', target_text.split("pub const JOINT_NAMES:", 1)[1].split("= [", 1)[1].split("];", 1)[0])
    if target_names != names:
        raise ValueError("Target WBC order differs from source")
    lower = target_text.split("pub const LOWER_HOME:", 1)[1].split("= [", 1)[1].split("];", 1)[0]
    target_q = [float(x) for x in lower.split(",") if x.strip()] + [0.0] * 28
    target_actuator = actuator_path.read_text()
    branches = re.findall(r'(?:if|else if) ([^{]+)\{\s*\(([^)]+)\)', target_actuator)
    armature = float(re.search(r"armature:\s*([\d.]+),", target_actuator).group(1))
    profile = []
    source_q = []
    for i, name in enumerate(names):
        groups = [g for g in cfg["actuators"].values() if any(re.fullmatch(p, name) for p in g["joint_names_expr"])]
        if len(groups) != 1:
            raise ValueError(f"Ambiguous source actuator for {name}")
        source = {}
        for key in PARAMETERS:
            value = groups[0].get(key)
            if isinstance(value, dict):
                matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous authored {name}.{key}")
                value = matches[0]
            source[key] = value
        target = None
        for condition, values in branches:
            if any(s in name for s in re.findall(r'name.contains\("([^"]+)"\)', condition)):
                target = dict(zip(PARAMETERS[:4], [float(x) for x in values.split(",")]))
                break
        if target is None or len(target) != 4:
            raise ValueError(f"Unsupported target actuator expression for {name}")
        target["armature"] = armature
        # The native generalized friction array is currently filled with zero;
        # the separate Rust profile field only preserves source metadata.
        target["friction"] = 0.0
        initial_values = [v for pattern, v in cfg["init_state"]["joint_pos"].items() if re.fullmatch(pattern, name)]
        source_q.append(initial_values[0] if initial_values else 0.0)
        profile.append({"joint": name, "source_authored": source, "target_authored": target,
                        "pd_armature_velocity_metadata_match": all(source[k] == target[k] for k in PARAMETERS[:5]),
                        "source_initial_q_defaulted_to_zero": not initial_values})
    if len(target_q) != 43:
        raise ValueError("Unexpected target initial q width")
    return {"joint_names": names, "source_initial_q": source_q, "target_initial_q": target_q,
            "initial_dq": [0.0] * 43, "profiles": profile,
            "files": [{"path": str(p), "sha256": digest(p)} for p in (g1_path, order_path, contract_path, actuator_path)]}


def rotate(q, v):
    length = math.sqrt(sum(x * x for x in q))
    if not math.isfinite(length) or abs(length - 1) > 1e-4:
        raise ValueError("Invalid recorded unit quaternion")
    x, y, z, w = [a / length for a in q]
    cross = [y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]]
    again = [y * cross[2] - z * cross[1], z * cross[0] - x * cross[2], x * cross[1] - y * cross[0]]
    return [v[i] + 2 * (w * cross[i] + again[i]) for i in range(3)]


def target_root_pose(frame):
    body = next(b for b in frame["bodies"] if b["body"] == 0)
    x, y, z = body["translation"]
    qx, qy, qz, qw = body["rotation_xyzw"]
    return [x, -z, y, qx, -qz, qy, qw]


def projected_gravity(pose):
    x, y, z, w = pose[3:]
    return rotate([-x, -y, -z, w], [0, 0, -1])


def target_sole_heights(initial, definition):
    poses = {b["body"]: b for b in initial["bodies"]}
    rows = []
    for collider in definition["collisions"]:
        if collider["shape"] != "sphere":
            continue
        body_name = definition["bodies"][collider["body"]]["name"]
        if body_name not in ("left_ankle_roll_link", "right_ankle_roll_link"):
            continue
        pose = poses[collider["body"]]
        x, y, z = collider["local_pose"]["position"]
        offset = rotate(pose["rotation_xyzw"], [x, z, -y])
        rows.append({"body": body_name, "collider": collider["path"],
                     "z_min_m": pose["translation"][1] + offset[1] - collider["radius"]})
    return rows


def violation_metrics(steps, names, limits):
    maximum = {}
    first = None
    for step in steps:
        if len(step["joint_positions"]) != len(names):
            raise ValueError("Target joint position width differs from WBC order")
        for name, value in zip(names, step["joint_positions"]):
            low, high = limits[name]
            excess = max(low - value, value - high, 0.0)
            if excess > maximum.get(name, {}).get("excess_rad", 0):
                maximum[name] = {"excess_rad": excess, "control_tick": step["frame"]["source_tick"], "q": value}
            if excess > 0.001 and first is None:
                first = {"joint": name, **maximum[name]}
    return {"tolerance_rad": 0.001, "first_beyond_tolerance": first,
            "joints_beyond_tolerance": {k: v for k, v in maximum.items() if v["excess_rad"] > 0.001}}


def target_metrics(target, definition, names):
    steps = target["steps"]
    if not steps:
        raise ValueError("Target has no completed steps")
    limits = {j["name"]: j["limits"] for j in definition["joints"] if j["kind"] != "fixed"}
    initial = target_root_pose(target["initial"])
    final = target_root_pose(steps[-1]["frame"])
    violations = violation_metrics(steps, names, limits)
    return {"frozen_harness_passed": target["passed"], "frozen_termination": target["termination"],
            "completed_ticks": len(steps), "duration_seconds": steps[-1]["frame"]["sim_time"],
            "initial_root_pose_source": initial, "last_root_pose_source": final,
            "root_xy_drift_m": math.hypot(final[0] - initial[0], final[1] - initial[1]),
            "minimum_root_height_m": min(s["root_position_source"][2] for s in steps),
            "max_tilt_degrees": max(math.degrees(math.acos(max(-1, min(1, s["root_upright_cosine"])))) for s in steps),
            "max_abs_joint_velocity": max(abs(v) for s in steps for v in s["joint_velocities"]),
            "limit_violations": violations,
            "interpretation": "No-fall is not standing qualification. Zero-command drift and finger limit violations remain independent adverse signals; no new XY threshold is retroactively assigned."}


def reorder(values, input_names, names):
    if len(values) != len(input_names) or len(set(input_names)) != len(input_names):
        raise ValueError("Invalid source joint vector or duplicate names")
    named = dict(zip(input_names, values))
    if set(named) != set(names):
        raise ValueError("Source/target driven joint sets differ")
    return [named[name] for name in names]


def max_delta(a, b):
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("Invalid comparison vectors")
    return max(abs(x - y) for x, y in zip(a, b))


def compare_boundaries(source, trace, target, names):
    if not source.get("rollouts"):
        return {"status": "unknown", "reason": "Source receipt has no native rollouts", "rows": []}
    input_names = source["joint_names"]
    source_rows = trace or [dict(state, repeat=run["repeat"]) for run in source["rollouts"]
                            for state in (run["initial_state"], run.get("last_state", run["initial_state"]))]
    targets = {s["frame"]["source_tick"]: s for s in target["steps"]}
    seen = set()
    result = []
    for row in source_rows:
        tick = row["control_tick"]
        key = (row["repeat"], tick)
        if key in seen:
            continue
        seen.add(key)
        if row["sample_phase"] != "control_boundary" or row["physics_tick"] != 4 * tick:
            raise ValueError("Source row is not the declared 200/50 control boundary")
        if not math.isclose(row["sim_time_seconds"], tick / 50, abs_tol=1e-7):
            raise ValueError("Source timestamp mismatch")
        # Tick zero q/dq is authored in the target, not recorded native state.
        if tick == 0 or tick not in targets:
            continue
        step = targets[tick]
        if not math.isclose(step["frame"]["sim_time"], row["sim_time_seconds"], abs_tol=1e-7):
            raise ValueError("Target timestamp mismatch")
        pose = target_root_pose(step["frame"])
        result.append({"repeat": row["repeat"], "control_tick": tick, "source_physics_tick": 4 * tick,
                       "target_physics_tick": tick,
                       "max_joint_q_delta_rad": max_delta(reorder(row["joint_pos"], input_names, names), step["joint_positions"]),
                       "max_joint_dq_delta_rad_s": max_delta(reorder(row["joint_vel"], input_names, names), step["joint_velocities"]),
                       "max_root_xyz_delta_m": max_delta(row["root_link_pose_w"][:3], pose[:3]),
                       "max_projected_gravity_delta": max_delta(row["projected_gravity_b"], projected_gravity(pose)),
                       "target_gravity_origin": "derived from completed native root pose",
                       "root_gyro_delta": None,
                       "torque_delta": None})
    return {"status": "compiled", "sampling": "full trace" if trace else "receipt endpoints only", "rows": result,
            "limitations": ["Target receipt did not record per-tick root gyro", "Source explicit PD updates each physics step; target recorded torque precedes its single step, so end-boundary torques are not directly paired", "Differences describe distinct 200/50 versus 50/50 trajectories, not causal attribution"]}


def compare_plant(source, definition, tensor_oracle=None):
    plant = source.get("compiled_plant")
    if not plant:
        return {"status": "unknown", "reason": "No compiled PhysX plant readback"}
    names = source["body_names"]
    authored = {b["name"]: b for b in definition["bodies"]}
    if len(set(names)) != len(names):
        raise ValueError("Duplicate compiled body names")
    semantics = source.get("compiled_plant_semantics", {}).get("inertia_matrix", {})
    tensor_compatible = all(semantics.get(key) == value for key, value in {
        "reference_point": "center_of_mass", "expressed_in": "rigid_body_prim_frame",
        "layout": "column_major_3x3", "unit": "kg*m^2"}.items())
    native_tensors = {b["name"]: b["native_tensor_row_major"] for b in tensor_oracle["bodies"]} if tensor_oracle else {}
    result = {"status": "compiled", "missing_source_bodies": sorted(set(authored) - set(names)),
              "extra_source_bodies": sorted(set(names) - set(authored)), "rows": [],
              "inertia_comparison": "compiled link-frame tensors at COM; convert target C^T I_engine C" if tensor_compatible and native_tensors
                  else "unknown: requires declared source tensor-frame semantics and existing native target tensor oracle",
              "source_semantics": source.get("compiled_plant_semantics")}
    for key in ("mass", "com_pose_xyzw", "inertia_matrix"):
        if len(plant[key]["values"]) != len(names):
            raise ValueError(f"Compiled {key} body count mismatch")
    for i, name in enumerate(names):
        body = authored.get(name)
        row = {"name": name, **{k: plant[k]["values"][i] for k in plant}, "target_basis": "authored export, not new native readback"}
        if body:
            mass = row["mass"]
            row["mass_delta_vs_authored"] = mass - body["mass"] if isinstance(mass, (int, float)) else None
            row["authored_mass_frame_fallback"] = body["diagnostic_mass_defaults"]
            com = row["com_pose_xyzw"]
            row["com_position_delta_vs_authored"] = (max_delta(com[:3], body["center_of_mass"])
                if not body["diagnostic_mass_defaults"] and all(v is not None for v in com[:3]) else None)
        if tensor_compatible and name in native_tensors:
            raw = row["inertia_matrix"]
            if len(raw) != 9:
                raise ValueError("Expected nine column-major inertia entries")
            if all(v is not None for v in raw):
                # C maps source xyz -> engine xz(-y); axes below are its columns.
                axes = [(0, 1), (2, -1), (1, 1)]
                native = native_tensors[name]
                source_tensor = [[raw[3 * col + r] for col in range(3)] for r in range(3)]
                target_tensor = [[native[a][b] * sa * sb for b, sb in axes] for a, sa in axes]
                error = max(abs(source_tensor[r][c] - target_tensor[r][c]) for r in range(3) for c in range(3))
                scale = max(abs(v) for v in raw)
                row["tensor_max_abs_delta_native_target"] = error
                row["tensor_relative_delta_native_target"] = error / scale if scale else None
                row["tensor_target_basis"] = "existing zero-integration native tensor receipt, converted to source link axes"
        result["rows"].append(row)
    return result


def source_rollout_metrics(source, trace, contract, definition):
    """Keep endpoint-only metrics explicitly distinct from full-trace extrema."""
    results = []
    limits = {j["name"]: j["limits"] for j in definition["joints"] if j["kind"] != "fixed"}
    for run in source.get("rollouts", []):
        initial, final = run["initial_state"], run.get("last_state", run["initial_state"])
        rows = ([row for row in trace if row["repeat"] == run["repeat"]] if trace
                else [initial, final])
        q = reorder(initial["joint_pos"], source["joint_names"], contract["joint_names"])
        dq = reorder(initial["joint_vel"], source["joint_names"], contract["joint_names"])
        source_limits = reorder(source["joint_limits"], source["joint_names"], contract["joint_names"])
        foot_names = initial.get("foot_link_names", [])
        foot_poses = initial.get("foot_link_poses", [])
        sole_heights = []
        if foot_names:
            if len(foot_names) != len(foot_poses) or len(set(foot_names)) != len(foot_names):
                raise ValueError("Invalid source foot pose names")
            named_poses = dict(zip(foot_names, foot_poses))
            for collider in definition["collisions"]:
                name = definition["bodies"][collider["body"]]["name"]
                if collider["shape"] == "sphere" and name in named_poses:
                    pose = named_poses[name]
                    offset = rotate(pose[3:], collider["local_pose"]["position"])
                    sole_heights.append({"body": name, "collider": collider["path"],
                                         "z_min_m": pose[2] + offset[2] - collider["radius"]})
        results.append({"repeat": run["repeat"], "frozen_harness_passed": run.get("passed"),
                        "initial_root_pose_source": initial["root_link_pose_w"],
                        "initial_q_max_delta_vs_target_authored": max_delta(q, contract["target_initial_q"]),
                        "initial_dq_max_delta_vs_target_authored": max_delta(dq, contract["initial_dq"]),
                        "joint_limit_max_delta_vs_target_authored": max(abs(pair[k] - limits[name][k])
                            for name, pair in zip(contract["joint_names"], source_limits) for k in range(2)),
                        "root_xy_drift_m": math.hypot(*(final["root_link_pose_w"][i] - initial["root_link_pose_w"][i] for i in range(2))),
                        "extrema_sampling": "full trace" if trace else "receipt endpoints only",
                        "minimum_sampled_height_m": min(row["root_link_pose_w"][2] for row in rows),
                        "maximum_sampled_tilt_degrees": max(math.degrees(row["tilt_radians"]) for row in rows),
                        "first_limit_violation": run.get("first_joint_limit_violation"),
                        "initial_foot_link_names": initial.get("foot_link_names"),
                        "initial_foot_link_poses": initial.get("foot_link_poses"),
                        "initial_sole_geometry": sole_heights,
                        "sole_geometry_basis": "compiled named foot poses + authored USD sphere geometry; empty if names unavailable"})
    return results


def comparison_table(source, target, contract, soles):
    has_native = bool(source.get("rollouts"))
    return [
        ["Sampling", "authored", "200 Hz physics, 50 Hz WBC; 4 integrations/control", "compiled", "50/50; 1 integration/control", "Pair control boundaries, not equal physics tick numbers"],
        ["Joint q/dq order", "compiled" if has_native else "unknown", "Receipt joint_names mapped to WBC43 when present", "authored", "WBC43 exactly matches pinned YAML", "Native order must be remapped by name"],
        ["Initial q/dq", "authored", "lower15 [-.1,0,0,.3,-.2,0] twice + waist0; upper28=0; dq=0", "authored", "Same 43 values; initial native q/dq absent from frozen receipt", "Source unspecified wrist/hand q uses default zero; compiled confirmation pending"],
        ["Initial base pose", "compiled" if has_native else "authored", str(source["rollouts"][0]["initial_state"]["root_link_pose_w"]) if has_native else "Requested [0,0,.78], xyzw [0,0,0,1]", "compiled", str(target_root_pose(target["initial"])), "Source actual reset pose comes from initial_state"],
        ["Projected gravity", "compiled" if has_native else "unknown", "root-link body sensor", "compiled", "Derived from completed native pose; initial [0,0,-1]", "Derived pose quantity labelled separately from recorded sensor"],
        ["Root gyro", "compiled" if has_native else "unknown", "root_link_ang_vel_b, not COM-frame AGILE sensor", "unknown", "Per-tick absent; cc4 current generalized-root feedback has separate pulse proof", "No finite-difference substitute"],
        ["Explicit PD", "authored", "IdealPD clip(Kp(qtarget-q)-Kd*dq, effort), every 200 Hz", "authored", "Same 43 gain/effort values, every 50 Hz", "Same gains does not mean same feedback cadence; never compare PhysX drive Kp as IdealPD Kp"],
        ["Armature", "authored", "0.03 per driven joint; any compiled readback separately retained", "authored", "0.03 each native generalized joint slot; no receipt readback", "Source backend joint_armature mapped by name when present"],
        ["Velocity limits", "authored", "Metadata per actuator; compiled solver cap not yet recorded", "authored", "Metadata matches, no native joint speed cap implemented", "Source velocity_limit legacy semantics require compiled confirmation"],
        ["Joint friction", "authored", "Feet/waist/arms/hands friction .03; hips/knees unspecified; backend properties separate", "authored", "Native generalized friction zero", "Pinned Lab API: IsaacSim>=5 static/dynamic friction are effort (Nm for hinges); older versions used unitless coefficient. Actual backend components still require readback; no compensating torque added"],
        ["Contact friction", "authored", "Floor static/dynamic .5/.5; robot binding and effective combine unknown", "authored", f"Robot {target['robot_friction']}, floor {target['floor_friction']}; Rapier Average => .75 (derived)", "PhysX configured combine=None is not proof of actual mode; no separate target static/dynamic"],
        ["Floor/contact shape", "authored", "USD Plane Z at 0; source convex cooking unknown", "authored", "Cuboid top at source Z0; eight sole spheres radius .005; Rapier convex hulls", "Matching top height alone does not establish contact parity"],
        ["Initial sole lower bound", "unknown", "Need source actual foot poses + same sphere geometry", "compiled", f"Native initial link poses + authored spheres: min Z {min(s['z_min_m'] for s in soles):.9f} m", "Mixed derivation; negative means geometric penetration, not measured impulse"],
        ["Mass/COM/inertia", "compiled" if "compiled_plant" in source else "unknown", "compiled_plant and fixed sensor presence pending or retained in JSON", "authored", "34.39423835 kg USD; four diagnostic sensor COM/frame fallbacks", "Mass compare by body name; tensors require declared link-frame-at-COM semantics and existing target native oracle"],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-trace", type=Path)
    parser.add_argument("--target-tensor-oracle", type=Path)
    parser.add_argument("--target-tensor-oracle-sha256")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--arena-source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists() or args.output == markdown:
        parser.error("Choose new JSON/Markdown outputs; existing evidence is never overwritten")
    target, target_file = load_json(args.target_report, args.target_sha256)
    source, source_file = load_json(args.source_report, args.source_sha256)
    definition, definition_file = load_json(args.definition, PHYSICS_SHA)
    if (source["arena_revision"] != ARENA_REV or definition["usd_sha256"] != USD_SHA
        or source["assets_sha256"]["g1_29dof_with_hand_rev_1_0.usd"] != USD_SHA
        or source["assets_sha256"]["stand.onnx"] != MODEL_SHA
        or target["usd_sha256"] != USD_SHA or target["stand_sha256"] != MODEL_SHA
        or target["definition_sha256"] != PHYSICS_SHA):
        raise ValueError("Source/target model identity mismatch")
    if (source["source_physics_hz"], source["source_wbc_hz"], target["physics_hz"], target["control_hz"]) != (200, 50, 50, 50):
        raise ValueError("Unexpected comparison timing")
    contract = authored_contract(args.arena_source, args.workspace)
    trace = None
    inputs = [source_file, target_file, definition_file, *contract.pop("files")]
    tensor_oracle = None
    if args.target_tensor_oracle:
        if not args.target_tensor_oracle_sha256:
            parser.error("An existing target tensor oracle requires its SHA256")
        tensor_oracle, oracle_file = load_json(args.target_tensor_oracle, args.target_tensor_oracle_sha256)
        if (tensor_oracle["definition_sha256"] != PHYSICS_SHA or tensor_oracle["usd_sha256"] != USD_SHA
            or tensor_oracle["integration_count"] != 0):
            raise ValueError("Target tensor oracle is not the matching zero-integration receipt")
        inputs.append(oracle_file)
    if args.source_trace:
        if digest(args.source_trace) != source.get("trace_sha256"):
            raise ValueError("Source trace does not match receipt trace_sha256")
        trace = [json.loads(line) for line in args.source_trace.read_text().splitlines() if line.strip()]
        inputs.append({"path": str(args.source_trace.resolve()), "sha256": digest(args.source_trace)})
    soles = target_sole_heights(target["initial"], definition)
    report = {"schema": "g1_read_only_source_target_comparison_v1", "no_dynamics_executed": True,
              "inputs": inputs, "script_sha256": digest(Path(__file__)), "target_code_commit": target["code_commit"],
              "classes": {"authored": "Source/config value or its mathematical derivation", "compiled": "Existing native-state readback or explicitly labelled derived pose quantity", "unknown": "Evidence unavailable or semantics unresolved"},
              "source_rollout_count": len(source.get("rollouts", [])),
              "source_frozen_harness_verdict": source.get("source_stand_verified"),
              "source_verdict_is_not_stationary_stand_proof": True,
              "authored_contract": contract, "target_metrics": target_metrics(target, definition, contract["joint_names"]),
              "target_initial_sole_geometry": soles,
              "boundary_comparison": compare_boundaries(source, trace, target, contract["joint_names"]),
              "source_rollout_metrics": source_rollout_metrics(source, trace, contract, definition),
              "compiled_source_vs_authored_export": compare_plant(source, definition, tensor_oracle),
              "source_backend_joint_parameters": {key: reorder(values, source["joint_names"], contract["joint_names"])
                                                  for key, values in source.get("backend_joint_parameters", {}).items()},
              "source_backend_joint_friction_properties": {"joint_names": source.get("joint_names"), "values": source.get("backend_joint_friction_properties"), "semantics": source.get("backend_joint_friction_semantics")},
              "source_sensor_presence": source.get("compiled_sensor_link_presence"),
              "comparison_table": comparison_table(source, target, contract, soles)}
    metrics = report["target_metrics"]
    lines = ["# G1 source/target read-only comparison", "", "No simulator, inference or integration was run by this report.", "",
             f"Target frozen verdict: **{metrics['frozen_harness_passed']}**; {metrics['duration_seconds']} s; XY drift **{metrics['root_xy_drift_m']:.6f} m**; max tilt {metrics['max_tilt_degrees']:.6f} deg.",
             f"Source native rollouts available: **{report['source_rollout_count']}**. No-fall is not stationary standing qualification.", "",
             "| Item | Source class | Source | Target class | Target | Boundary / limitation |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in report["comparison_table"]]
    lines += ["", "Joint limit failures (tolerance 0.001 rad):", ""]
    lines += [f"- {name}: excess {value['excess_rad']:.9f} rad at tick {value['control_tick']}."
              for name, value in metrics["limit_violations"]["joints_beyond_tolerance"].items()]
    lines += ["", "Input identities:", ""] + [f"- `{entry['sha256']}` — {entry['path']}" for entry in inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(args.output.resolve()), "markdown": str(markdown.resolve()),
                      "source_native_rollouts": report["source_rollout_count"], "target_metrics": metrics}, allow_nan=False))


if __name__ == "__main__":
    main()
