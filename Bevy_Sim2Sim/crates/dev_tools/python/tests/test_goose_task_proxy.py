"""Prevent incomplete contact proposals from silently removing source geometry."""
import copy
import json
from pathlib import Path

import pytest

from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.task_proxy import KNOWN_INTERFERENCE_PARTS, read_role_profile, validate_roles


def fixture():
    names = sorted(KNOWN_INTERFERENCE_PARTS) + ["left_flexible_sole", "right_flexible_sole",
                                              "upper_grip_cassette_pad", "lower_grip_cassette_pad"]
    names += [f"source_part_{index}" for index in range(460 - len(names))]
    rows = []
    for name in names:
        role, budget = "inherited_mass_only", 0
        if name in KNOWN_INTERFERENCE_PARTS:
            role, budget = "articulation_surface", 8
        if name.endswith("flexible_sole"):
            role, budget = "exact_sole", 6
        if name.endswith("cassette_pad"):
            role, budget = "exact_grip", 1
        rows.append(dict(part=name, body="torso", role=role, piece_budget=budget,
                         source_geometry_sha256="source", si_retained=True, release=False,
                         known_interference_guard=name in KNOWN_INTERFERENCE_PARTS, rationale="Declared scope"))
    profile = dict(schema="goose_task_contact_roles_v1", status="PROPOSED_UNQUALIFIED", training_release=False,
                   source_authority_unchanged=True, collision_exclusions_changed=False,
                   inertia_from_collision_geometry=False, source_parts=rows, allocated_piece_budget=62,
                   cad_to_assembly_translation_m=[0, 0, .0037], max_robot_collision_leaves=512, max_support_vertices=64)
    scene = dict(parts=[dict(name=name, source_sha256="source") for name in names], assembly_translation_m=[0, 0, .0037])
    manifest = dict(parts=[dict(name=name, body="torso") for name in names])
    parent = dict(physics_dt_s=.02, torque_dt_s=.02, policy_dt_s=.02, physics_steps_per_tick=1,
                  torque_updates_per_tick=1, policy_calls_per_tick=1, training_release=False, bodies=[dict(name="torso")])
    return profile, scene, manifest, parent


def test_complete_profile_only_validates_proposal_structure():
    result = validate_roles(*fixture())
    assert result["source_part_count"] == 460
    assert not result["source_qualified"] and not result["training_release"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "owner", "hash", "known_guard", "sole",
                                   "grip", "rationale", "allocation", "frame", "substep", "release"])
def test_rejects_silent_scope_or_provenance_changes(change):
    profile, scene, manifest, parent = copy.deepcopy(fixture())
    if change == "missing":
        profile["source_parts"].pop()
    elif change == "duplicate":
        profile["source_parts"][-1] = profile["source_parts"][0]
    elif change == "owner":
        profile["source_parts"][0]["body"] = "other_body"
    elif change == "hash":
        profile["source_parts"][0]["source_geometry_sha256"] = "other_shape"
    elif change == "known_guard":
        profile["source_parts"][0].update(role="internal_mass_only", piece_budget=0)
    elif change in ("sole", "grip"):
        name = "left_flexible_sole" if change == "sole" else "upper_grip_cassette_pad"
        next(row for row in profile["source_parts"] if row["part"] == name).update(role="articulation_surface", piece_budget=1)
    elif change == "rationale":
        profile["source_parts"][0]["rationale"] = ""
    elif change == "allocation":
        profile["allocated_piece_budget"] = 61
    elif change == "frame":
        profile["cad_to_assembly_translation_m"] = [0, 0, 0]
    elif change == "substep":
        parent["physics_steps_per_tick"] = 2
    elif change == "release":
        profile["training_release"] = True
    with pytest.raises(ValueError):
        validate_roles(profile, scene, manifest, parent)


@pytest.mark.parametrize("changed_source", ["source_scene", "source_manifest", "parent_model", "parent_contract"])
def test_read_rejects_mutated_authority(tmp_path: Path, changed_source):
    profile, scene, manifest, parent = fixture()
    paths = {name: tmp_path / (name + ".json") for name in ("source_scene", "source_manifest", "parent_model", "parent_contract")}
    for name, document in (("source_scene", scene), ("source_manifest", manifest)):
        paths[name].write_text(json.dumps(document))
    paths["parent_model"].write_text("<mujoco/>")
    parent.update(model_sha256=sha256(paths["parent_model"]), source_scene_sha256=sha256(paths["source_scene"]))
    paths["parent_contract"].write_text(json.dumps(parent))
    for name, path in paths.items():
        profile[name + "_path"] = str(path)
        profile[name + "_sha256"] = sha256(path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    assert read_role_profile(profile_path)[1]["status"] == "PROPOSAL_STRUCTURE_VALID"
    with paths[changed_source].open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="source identity changed"):
        read_role_profile(profile_path)
