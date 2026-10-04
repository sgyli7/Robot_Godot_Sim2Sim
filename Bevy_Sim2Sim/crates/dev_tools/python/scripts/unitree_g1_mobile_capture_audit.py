#!/usr/bin/env python3
"""Independent read-only T2 route/placement audit, never a control input.

Scores the disclosed fixed-profile executor with camera/model pauses. Passing
this benchmark does not qualify continuous 1x execution or arbitrary targets.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from unitree_g1_mobile_placement_audit import evaluate

REVISION = "dfe74af855007f26093f362cd2d7a2f404b64b93"
EXPECTED_EVENTS = ["initial_profile_request_submitted", "initial_profile_decision_admitted",
                   "transport_request_submitted", "transport_decision_admitted",
                   "final_feedback_submitted", "final_feedback_admitted"]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def horizontal(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_evidence(definition, rows):
    hands = {i for i, body in enumerate(definition["bodies"]) if "_hand_" in body["name"]}
    feet = {i for i, body in enumerate(definition["bodies"])
            if body["name"] in {"left_ankle_roll_link", "right_ankle_roll_link"}}
    held, violations = [], []
    phases = Counter()
    bodies, boxes = [], []
    for tick, row in enumerate(rows, 1):
        body = row["body"]["mobile_homie_v2"]
        if not (row["owner_episode_integrations"] == body["integration_count"]
                == body["torque_update_count"] == body["inference"]["inference_count"] == tick):
            raise ValueError("Physics/body/motor counter gap or repeated integration")
        step = body["step_configuration"]
        if (step["physics_hz"] != 50 or np.float32(step["dt"]) != np.float32(.02)
                or step["num_solver_iterations"] != 1 or step["max_ccd_substeps"] != 1
                or step["num_internal_pgs_iterations"] != 4
                or step["additional_solver_iterations_max"] != 0):
            raise ValueError("Formal 50Hz single integration/explicit solver contract changed")
        environment = body["native_static_environment"]
        if environment["collider_count"] != 2553 or not environment["original_broad_floor_removed"]:
            raise ValueError("Foreign physical station or hidden broad floor")
        phase = next(iter(row["execution"]))
        phases[phase] += 1
        box = next(o for o in body["task_objects"]["objects"] if o["kind"] == "t2_box")
        if tick > 200 and phase != "classical_release":
            contacts = [c for c in box["last_solve_contacts"]
                        if (c["active_solver_normal_impulse_n_s"] or 0) > 1e-7]
            held.append(bool(contacts) and all(c["other_robot_body_index"] in hands for c in contacts))
        for contact in body["native_station_contacts"]:
            if (contact["other_body_fixed"]
                    and (contact["active_solver_normal_impulse_n_s"] or 0) > 1e-7
                    and contact["robot_body_index"] not in feet):
                violations.append(tick)
        bodies.append(body)
        boxes.append(box)
    if not rows:
        raise ValueError("Empty physical route")
    standing = all(b["root_upright_cosine"] > .95 and b["root_position_source"][2] >= .35
                   for b in bodies)
    return {"phases": dict(phases), "standing_all_ticks": standing,
            "minimum_upright": min(b["root_upright_cosine"] for b in bodies),
            "all_postgrasp_pre_release_hand_only_support": bool(held) and all(held),
            "held_samples": len(held), "nonfoot_positive_fixed_world_contact_ticks": violations,
            "box_horizontal_displacement_m": horizontal(boxes[0]["position_source"], boxes[-1]["position_source"]),
            "robot_horizontal_displacement_m": horizontal(bodies[0]["root_position_source"], bodies[-1]["root_position_source"])}


def qwen_evidence(native, receipt, episode):
    transport = receipt["mobile_assist_handoff"]["qwen_postgrasp_transport"]
    events = transport["events"]
    if [e["event"] for e in events] != EXPECTED_EVENTS:
        return False
    initial, selection, final = events[1], events[3], events[5]
    if (initial["model_decision"]["request"]["action"] != "begin_fixed_profile"
            or initial["actual_integrations_before_admission"] != 0
            or initial["original_policy_calls_before_admission"] != 0
            or initial["target_localization_claim"] or initial["release_authorized"]
            or selection["model_decision"]["request"] != {
                "action": "execute_task", "target_id": "box_marker_22", "destination_id": "bin_marker_21"}
            or final["model_decision"]["request"]["action"] not in {"observe", "stop"}
            or final["additional_executor_actions"] != 0
            or transport["http_attempts"] != 3 or transport["http_results"] != 3
            or transport["discarded_results"] != 0 or transport["world_or_contact_truth_input"]):
        return False
    stamps = [e.get("observation", e.get("qwen_observation")) for e in (initial, selection, final)]
    if (any(s["episode_id"] != episode for s in stamps)
            or len({s["frame_id"] for s in stamps}) != 3
            or stamps[0]["sim_time_ns"] != 0
            or stamps[0]["sim_time_ns"] >= stamps[1]["sim_time_ns"]
            or stamps[1]["sim_time_ns"] >= stamps[2]["sim_time_ns"]
            or any(e["image_age_ms"] > 20000 or e["service_elapsed_ms"] > 20000
                   for e in (initial, selection, final))):
        return False
    wire = json.loads((native / "initial_qwen/model_request.json").read_text())
    context = json.loads(wire["messages"][1]["content"][0]["text"])
    if (set(context) != {"stamp", "camera", "robot_proprioception", "public_goal", "available_skills",
                         "remembered_visual_targets", "recent_decisions", "execution_feedback"}
            or set(context["robot_proprioception"]) != {
                "joint_positions", "joint_velocities", "base_velocity_mps", "projected_gravity"}
            or context["stamp"] != stamps[0] or context["remembered_visual_targets"]
            or context["recent_decisions"] or context["execution_feedback"]
            or wire["max_tokens"] != 128 or wire["chat_template_kwargs"]["enable_thinking"]):
        return False
    png = base64.b64decode(wire["messages"][1]["content"][1]["image_url"]["url"].split(",")[1], validate=True)
    return hashlib.sha256(png).hexdigest() == events[0]["image_sha256"]


def audit(case, seed):
    case = Path(case)
    native = case / "native"
    receipt = json.loads((native / "capture_receipt.json").read_text())
    cfg = json.loads((case / "config.json").read_text())
    body_cfg = cfg["runner"]["body"]["mobile_homie_v2"]
    episode = body_cfg["episode_id"]
    definition_path = Path(body_cfg["definition"])
    object_path = Path(body_cfg["task_objects"]["definition"])
    if (digest(definition_path) != body_cfg["definition_sha256"]
            or digest(object_path) != body_cfg["task_objects"]["definition_sha256"]):
        raise ValueError("Bound robot or object collision definition changed")
    rows = [json.loads(line) for line in (native / "owner_steps.jsonl").open()]
    if len(rows) != receipt["actual_integrations"] or any(r["episode_id"] != episode for r in rows):
        raise ValueError("Capture count or episode identity disagrees with owner trace")
    if not rows:
        return {"evidence_verified": True, "strict_task_passed": False,
                "actual_integrations": 0, "fresh_vla_calls": receipt["live_policy_inference_calls"],
                "qwen_http_results": receipt.get("mobile_assist_handoff", {}).get("qwen_postgrasp_transport", {}).get("http_results", 0),
                "failure_reason": receipt["failure_reason"], "full_task_qualified": False}
    route = route_evidence(json.loads(definition_path.read_text()), rows)
    placement = evaluate(json.loads(object_path.read_text()), rows)
    (case / "independent_placement_audit.json").write_text(json.dumps(placement, indent=2) + "\n")
    captures = sorted((case / "policy_captures").glob(f"e{episode}_f*_q*"))
    if len(captures) != receipt["live_policy_successes"]:
        raise ValueError("Original model captures/counter disagree")
    for capture in captures:
        record = json.loads((capture / "receipt.json").read_text())
        identity = record["policy_identity"]
        if (record["stamp"]["episode_id"] != episode or record["model_revision"] != REVISION
                or identity["initial_seed"] != seed or identity["inference_backend"] != "official_gr00t_n1d6_pytorch"
                or not record["full_forward_verified"]
                or digest(capture / "observation.npz") != record["observation_sha256"]
                or digest(capture / "actions.npz") != record["actions_sha256"]):
            raise ValueError("Policy weights/seed/actual capture identity mismatch")
        sequence = record["sequence_id"]
        with np.load(capture / "observation.npz") as observation:
            rgb = np.asarray(Image.open(native / f"live_ego_{sequence:04}.png").convert("RGB"))
            if not np.array_equal(observation["ego_view"], rgb.reshape(1, 1, 480, 640, 3)):
                raise ValueError("Actual VLA input differs from native captured RGB")
    handoff = receipt["mobile_assist_handoff"]
    prep = handoff.get("visual_thumb_preparation", {}).get("executed_goal", {}).get("observation")
    release = handoff.get("visual_release_alignment", {}).get("executed_goal", {}).get("observation")
    fresh = False
    if prep and release:
        observation = json.loads((native / "visual_release_alignment_after_thumb/observation.json").read_text())
        fresh = (observation["stamp"] == release and release["episode_id"] == episode
                 and release["sim_time_ns"] > prep["sim_time_ns"] and release["frame_id"] > prep["frame_id"])
    qwen_passed = qwen_evidence(native, receipt, episode)
    poses = {p["kind"]: p["root_pose"]["position"] for p in body_cfg["task_objects"]["placements"]}
    source_target_distance = horizontal(poses["t2_box"], poses["t2_bin"])
    complete = (receipt["capture_succeeded"] and receipt["owner_step_trace_complete"]
                and receipt["owner_step_trace_dropped"] == 0 and receipt["live_policy_inference_calls"] == 4
                and receipt["live_policy_successes"] == 4 and route["phases"].get("original_vla") == 200
                and route["phases"].get("classical_carry", 0) > 0
                and route["phases"].get("classical_release", 0) >= 101)
    passed = (complete and route["standing_all_ticks"]
              and route["all_postgrasp_pre_release_hand_only_support"]
              and not route["nonfoot_positive_fixed_world_contact_ticks"]
              and route["box_horizontal_displacement_m"] >= 1.8
              and route["robot_horizontal_displacement_m"] >= .8 and source_target_distance >= 1.8
              and fresh and qwen_passed and placement["final_continuous_placement_seconds"] >= 2.)
    return {**route, "evidence_verified": True, "strict_task_passed": bool(passed),
            "actual_integrations": len(rows), "fresh_vla_calls": receipt["live_policy_inference_calls"],
            "qwen_http_results": handoff["qwen_postgrasp_transport"]["http_results"],
            "three_fresh_qwen_decisions_verified": qwen_passed, "fresh_after_thumb_release_rgb": fresh,
            "source_target_distance_m": source_target_distance,
            "final_continuous_placement_seconds": placement["final_continuous_placement_seconds"],
            "placement_window_passed": placement["diagnostic_release_window_passed"],
            "final_placement": placement["samples"][-1], "capture_succeeded": receipt["capture_succeeded"],
            "failure_reason": receipt["failure_reason"], "trace_sha256": digest(native / "owner_steps.jsonl"),
            "active_sim_wall_ratio": receipt["active_sim_seconds"] / receipt["active_wall_seconds"],
            "pending_ticks": receipt["pending_ticks"], "control_deadlines_missed": receipt["control_deadlines_missed"],
            "pauses_for_camera_and_policy": receipt["pauses_for_camera_and_policy"],
            "continuous_real_time_qualified": False, "full_task_qualified": False}
