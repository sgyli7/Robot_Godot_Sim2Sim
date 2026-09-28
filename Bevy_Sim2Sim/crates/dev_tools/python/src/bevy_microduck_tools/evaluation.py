"""Evaluate raw state independently of reward and upstream rollout summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .source_adapter import JOINT_ORDER
from .workflow import Rejection


def finite_tree(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def evaluate_raw(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or rows[0].get("kind") != "header":
        raise Rejection("Raw trajectory header missing")
    header = rows[0]
    if header.get("canonical_actuator_order") != list(JOINT_ORDER):
        raise Rejection("Wrong actuator mapping")
    if [name.split("/")[-1] for name in header["actuator_order"]] != list(JOINT_ORDER):
        raise Rejection("Scene mapping does not match contract")
    if header.get("auto_reset"):
        raise Rejection("Automatic reset can conceal the true terminal")
    kinds = [row["kind"] for row in rows]
    required = ("reset_initial", "physics_step", "terminal", "pre_reset", "reset_after_terminal")
    if any(kind not in kinds for kind in required):
        raise Rejection("Missing initial, integration, terminal, or reset boundary")
    terminal = kinds.index("terminal")
    if not terminal < kinds.index("pre_reset") < kinds.index("reset_after_terminal"):
        raise Rejection("Terminal captured after reset")
    state = "closed"
    episodes = []
    current = None
    for row in rows[1:]:
        kind = row["kind"]
        if kind == "reset_initial":
            if state != "closed":
                raise Rejection("Reset initial concealed an unfinished episode")
            current = {"initial": row, "physics": []}
            state = "active"
        elif kind in {"physics_step", "post_policy", "policy_call"}:
            if state != "active":
                raise Rejection("Physics or policy step outside a live episode")
            if kind == "physics_step":
                current["physics"].append(row)
        elif kind == "terminal":
            if state != "active" or not row.get("reason"):
                raise Rejection("Missing live true pre-reset terminal state or reason")
            current["final"] = row
            state = "terminal"
        elif kind == "pre_reset":
            if state != "terminal":
                raise Rejection("Pre-reset boundary did not follow the terminal")
            state = "pre_reset"
        elif kind == "reset_after_terminal":
            if state != "pre_reset":
                raise Rejection("Reset concealed terminal or pre-reset state")
            episodes.append(current)
            state = "closed"
        else:
            raise Rejection(f"Unknown raw lifecycle event: {kind}")
    if state != "closed":
        raise Rejection("Incomplete final episode; missing terminal/reset boundary")
    physics = [row for row in rows if row["kind"] == "physics_step"]
    if [row["physics_tick"] for row in physics] != list(range(1, len(physics) + 1)):
        raise Rejection("Missing or duplicate integration frame")
    for row in rows[1:]:
        if not finite_tree(row):
            raise Rejection("Nonfinite raw state")
        if len(row["body_positions"]) != len(header["body_order"]) or len(row["body_quaternions"]) != len(header["body_order"]):
            raise Rejection("Missing full-body state")
        if header.get("schema") == "microduck_4d_raw_v2":
            integrated = row["kind"] == "physics_step"
            if integrated:
                interval = row.get("contact_interval")
                expected = [(row["physics_tick"] - 1) * header["integration_dt"], row["physics_tick"] * header["integration_dt"]]
                if interval != expected or any(contact.get("impulse_contact_frame") is None for contact in row["contacts"]):
                    raise Rejection("Executed integration interval or contact impulse missing")
            elif row.get("contact_interval") is not None or any(contact.get("impulse_contact_frame") is not None for contact in row["contacts"]):
                raise Rejection("Forward-only/reset snapshot falsely claims an integrated contact impulse")
            measurement = row.get("termination_measurement", {})
            if row["kind"] == "reset_after_terminal" and measurement.get("belongs_to_current_episode"):
                raise Rejection("Reset labels the previous episode termination cache as current")
            if row["kind"] == "terminal" and row["reason"] in {"terminated", "time_limit"} and not (
                measurement.get("belongs_to_current_episode") and measurement.get("computed_at_current_physics_tick")
            ):
                raise Rejection("Natural terminal result was not computed in the current episode/tick")
            if row["kind"] == "policy_call":
                trace = row.get("policy_input_and_output_at_call")
                if trace is None or trace.get("episode_epoch") != row["episode_epoch"]:
                    raise Rejection("Actual policy-call trace missing or from a previous episode")
            elif row.get("policy_input_and_output_at_call") is not None:
                raise Rejection("Historical inference trace mislabeled as a new policy call")
    initial = episodes[0]["initial"]
    final = episodes[-1]["final"]
    root_id = next(i for i, name in enumerate(header["body_order"]) if name and name.endswith("trunk_base"))
    def tilt(row):
        w, x, y, z = row["body_quaternions"][root_id]
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x*x + y*y)))))
    tilts = [tilt(row) for row in physics]
    positions = [row["body_positions"][root_id] for row in physics]
    displacement = math.dist(initial["body_positions"][root_id][:2], final["body_positions"][root_id][:2])
    penetrations = [max(0.0, -contact["distance"]) for row in physics for contact in row["contacts"]]
    return {"schema": "independent_state_evaluation_v1", "status": "diagnostic_only",
            "episodes_in_denominator": len(episodes), "terminated_or_truncated": final["reason"],
            "episode_reasons": [episode["final"]["reason"] for episode in episodes],
            "complete_first_source_episode": episodes[0]["final"]["reason"] in {"terminated", "time_limit"},
            "physics_frames": len(physics), "max_tilt_degrees": max(tilts),
            "minimum_root_height": min(position[2] for position in positions),
            "planar_displacement": displacement, "max_contact_penetration": max(penetrations, default=0.0),
            "source_integrations_per_policy": header["source_substeps"],
            "missing_skill_oracle": True, "skill_passed": False,
            "note": "No reward consumed. Quantitative per-skill oracle must be fixed before selection."}
