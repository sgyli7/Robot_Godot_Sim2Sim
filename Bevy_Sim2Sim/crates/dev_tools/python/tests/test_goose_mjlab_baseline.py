"""Upstream model wiring preserves effort motors, inertia and admission scope."""
import json

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab")

from bevy_microduck_tools.goose.artifacts import CANDIDATES, sha256
from bevy_microduck_tools.goose.mjlab_baseline import (
    NATIVE_PARENTS, _read_native_contract, build_reference, make_entity_cfg)
from test_goose_50hz import fixture_runtime


@pytest.fixture(params=tuple(NATIVE_PARENTS))
def parent(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    runtime, path = fixture_runtime(tmp_path)
    contract = runtime.contract
    contract.update(candidate=request.param, torque_updates_per_tick=1, policy_calls_per_tick=1)
    path.write_text(json.dumps(contract))
    return tmp_path / "robot.xml", path


@pytest.mark.parametrize("field", ["native_discrete", "numerical_metric", "ground_contact"])
def test_cpu_experiments_cannot_silently_enter_upstream(parent, field):
    model, path = parent
    contract = json.loads(path.read_text())
    contract[field] = {"revision": "private_cpu_experiment"}
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="experimental"):
        _read_native_contract(model, path)


def test_upstream_model_keeps_native_physics_and_effort_motor_order(parent, tmp_path):
    model, contract = parent
    reference, identity = build_reference(model, contract, tmp_path / "reference")
    before = mujoco.MjModel.from_xml_path(str(model))
    cfg = make_entity_cfg(reference, identity)
    wrapped = cfg.build().spec.compile()
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_axis",
                 "jnt_range", "dof_armature", "dof_damping", "actuator_trnid", "actuator_gainprm"):
        np.testing.assert_array_equal(getattr(wrapped, name), getattr(before, name))
    assert [wrapped.actuator(i).name for i in range(18)] == [before.actuator(i).name for i in range(18)]
    assert wrapped.ngeom == before.ngeom - 1
    assert cfg.articulation.actuators[0].command_field == "effort"
    assert len(cfg.articulation.actuators[0].target_names_expr) == 18
    native = json.loads(identity.read_text())
    assert native["model_sha256"] == sha256(reference)
    assert native["training_release"] is False
    parent_identity = json.loads(contract.read_text())
    assert native["candidate"] == NATIVE_PARENTS[parent_identity["candidate"]]
    assert native["upstream_baseline"]["parent_candidate"] == parent_identity["candidate"]
    assert native["upstream_baseline"]["decimation"] == 1
    assert native["upstream_baseline"]["custom_constraint_callbacks"] is False
    with pytest.raises(FileExistsError):
        build_reference(model, contract, tmp_path / "reference")


def test_identity_and_tick_drift_fail_before_loading_mjcf(parent):
    model, path = parent
    contract = json.loads(path.read_text())
    contract["torque_updates_per_tick"] = 4
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="timing"):
        _read_native_contract(model, path)
    contract["torque_updates_per_tick"] = 1
    contract["model_sha256"] = "stale_model"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="identity"):
        _read_native_contract(model, path)


def test_full_body_and_unknown_parents_cannot_enter_condensed_reference(parent, tmp_path):
    model, path = parent
    contract = json.loads(path.read_text())
    for name in (CANDIDATES[0], "goose_task_collision_v1_full50", "goose_unknown_condensed50"):
        contract["candidate"] = name
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="condensed native parent"):
            build_reference(model, path, tmp_path / "rejected_reference")
        assert not (tmp_path / "rejected_reference").exists()
