"""Validate explicit task-contact proposals separately from physics qualification.

Every source part must have an owner and a declared contact role. The validator
checks provenance and budgets; it never certifies that an internal part remains
hidden, a convex approximation is accurate, or a model can enter training.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .artifacts import DT, sha256

ROLES = frozenset(("exact_sole", "exact_grip", "external_skin", "external_detail",
                   "articulation_surface", "internal_mass_only", "inherited_mass_only"))
MASS_ONLY_ROLES = frozenset(("internal_mass_only", "inherited_mass_only"))
KNOWN_INTERFERENCE_PARTS = frozenset((
    "lower_grip_shell", "torso_mounted_shell_right_fore",
    "left_ankle_pitch_bridge", "left_hip_pitch_static_front_carrier",
    "right_ankle_pitch_bridge", "right_hip_pitch_static_front_carrier",
))


def validate_roles(profile: dict, scene: dict, manifest: dict, parent: dict) -> dict:
    """Reject incomplete or misleading role tables without loading a simulator."""
    if (profile.get("schema") != "goose_task_contact_roles_v1"
            or profile.get("status") != "PROPOSED_UNQUALIFIED"
            or profile.get("training_release") is not False
            or profile.get("source_authority_unchanged") is not True
            or profile.get("collision_exclusions_changed") is not False
            or profile.get("inertia_from_collision_geometry") is not False):
        raise ValueError("Only an explicit unqualified SI-preserving proposal is accepted")
    if (any(parent[key] != DT for key in ("physics_dt_s", "torque_dt_s", "policy_dt_s"))
            or any(parent[key] != 1 for key in ("physics_steps_per_tick", "torque_updates_per_tick", "policy_calls_per_tick"))
            or parent.get("training_release") is not False):
        raise ValueError("The parent must retain single-step 50 Hz and no training release")
    source_parts = {row["name"]: row for row in scene["parts"]}
    owners = {row["name"]: row["body"] for row in manifest["parts"]}
    parent_bodies = {body["name"] for body in parent["bodies"]}
    rows = profile["source_parts"]
    names = [row["part"] for row in rows]
    if (len(source_parts) != 460 or len(source_parts) != len(scene["parts"])
            or len(owners) != len(manifest["parts"]) or set(owners) != set(source_parts)
            or not set(owners.values()).issubset(parent_bodies)
            or len(names) != len(set(names)) or set(names) != set(source_parts)):
        raise ValueError("A one-to-one declaration for all 460 source parts is required")
    if profile.get("cad_to_assembly_translation_m") != scene["assembly_translation_m"]:
        raise ValueError("CAD-to-assembly coordinate identity mismatch")
    for row in rows:
        name, role = row["part"], row["role"]
        budget = row["piece_budget"]
        if (role not in ROLES or row["body"] != owners[name]
                or row["source_geometry_sha256"] != source_parts[name]["source_sha256"]
                or row.get("si_retained") is not True or row.get("release") is not False
                or not isinstance(row.get("rationale"), str) or not row["rationale"].strip()
                or type(budget) is not int or budget < 0
                or (role in MASS_ONLY_ROLES) != (budget == 0)):
            raise ValueError(f"Invalid contact declaration: {name}")
        if name in KNOWN_INTERFERENCE_PARTS and (role in MASS_ONLY_ROLES or row.get("known_interference_guard") is not True):
            raise ValueError(f"Known interference cannot disappear into a mass-only role: {name}")
        if name in ("left_flexible_sole", "right_flexible_sole") and (role != "exact_sole" or budget != 6):
            raise ValueError("Each foot must retain six inherited compliant contact patches")
        if name in ("upper_grip_cassette_pad", "lower_grip_cassette_pad") and (role != "exact_grip" or budget != 1):
            raise ValueError("Each gripping face must retain its exact inherited mesh")
    allocated = sum(row["piece_budget"] for row in rows)
    if (profile.get("max_robot_collision_leaves") != 512 or profile.get("max_support_vertices") != 64
            or allocated != profile.get("allocated_piece_budget") or allocated > 512):
        raise ValueError("Task proxy allocation exceeds or misreports the frozen budget")
    if not KNOWN_INTERFERENCE_PARTS.issubset(source_parts):
        raise ValueError("Known source interference identities are missing")
    return {"status": "PROPOSAL_STRUCTURE_VALID", "source_part_count": len(rows),
            "allocated_piece_budget": allocated, "role_counts": dict(Counter(row["role"] for row in rows)),
            "source_qualified": False, "target_qualified": False, "training_release": False,
            "scope": "Provenance, complete roles and declared budgets; geometry and internal exposure not evaluated"}


def read_role_profile(path: Path) -> tuple[dict, dict]:
    """Bind the proposal to its exact parent, original scene and ownership table."""
    profile = json.loads(path.read_text())
    documents = {}
    for name in ("source_scene", "source_manifest", "parent_model", "parent_contract"):
        source = Path(profile[name + "_path"])
        if sha256(source) != profile[name + "_sha256"]:
            raise ValueError(f"Role proposal source identity changed: {name}")
        if name != "parent_model":
            documents[name] = json.loads(source.read_text())
    parent = documents["parent_contract"]
    if (parent["model_sha256"] != profile["parent_model_sha256"]
            or parent["source_scene_sha256"] != profile["source_scene_sha256"]):
        raise ValueError("Parent and source identity ledger disagree")
    return profile, validate_roles(profile, documents["source_scene"], documents["source_manifest"], parent)
