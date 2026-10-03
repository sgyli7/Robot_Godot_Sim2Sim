"""Upstream model wiring preserves effort motors, inertia and admission scope."""
import json
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab")

from bevy_microduck_tools.goose.artifacts import CANDIDATES, sha256
from bevy_microduck_tools.goose.mjlab_baseline import (
    EULER_CANDIDATE, NATIVE_PARENTS, _read_native_contract, build_reference, make_entity_cfg)
from bevy_microduck_tools.goose.runtime import GooseSourceRuntime
from test_goose_50hz import fixture_runtime


@pytest.fixture(params=tuple(NATIVE_PARENTS))
def parent(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    runtime, path = fixture_runtime(tmp_path)
    contract = runtime.contract
    contract.update(candidate=request.param, torque_updates_per_tick=1, policy_calls_per_tick=1)
    path.write_text(json.dumps(contract))
    return tmp_path / "robot.xml", path


def make_euler_reference(model, path):
    """Give the small wiring fixture the selected native numerical profile."""
    tree = ET.fromstring(model.read_text())
    option = tree.find("option")
    option.set("integrator", "Euler")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("eulerdamp", "disable")
    model.write_text(ET.tostring(tree, encoding="unicode"))
    contract = json.loads(path.read_text())
    contract.update(candidate=EULER_CANDIDATE, integrator="Euler", model_sha256=sha256(model),
        native_integration_flow={"revision": "goose_native_euler_eulerdamp_disabled_v1",
            "dt_s": .02, "decimation": 1, "integrations_per_tick": 1,
            "physical_dampers_enabled": True, "implicit_joint_damping_integration_enabled": False})
    path.write_text(json.dumps(contract))
    return model, path


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
    assert native["upstream_baseline"]["external_nonfinite_abort_required"] is True
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


@pytest.mark.parametrize("mutation", ["compiled_integrator", "implicit_damping", "physical_damping",
                                     "declared_integrator", "missing_profile", "wrong_revision"])
def test_euler_reference_rejects_mislabeled_or_changed_integration_flow(parent, tmp_path, mutation):
    model, path = build_reference(*parent, tmp_path / "reference")
    make_euler_reference(model, path)
    tree = ET.fromstring(model.read_text())
    contract = json.loads(path.read_text())
    if mutation == "compiled_integrator": tree.find("option").set("integrator", "implicitfast")
    elif mutation == "implicit_damping": tree.find("./option/flag").set("eulerdamp", "enable")
    elif mutation == "physical_damping": tree.find("./option/flag").set("damper", "disable")
    elif mutation == "declared_integrator": contract["integrator"] = "implicitfast"
    elif mutation == "missing_profile": contract.pop("native_integration_flow")
    else: contract["native_integration_flow"]["revision"] = "unknown"
    model.write_text(ET.tostring(tree, encoding="unicode"))
    contract["model_sha256"] = sha256(model)
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="reference|integration profile"):
        make_entity_cfg(model, path)


def test_c_failure_flag_does_not_block_warp_or_enable_native_recovery(parent, tmp_path):
    import warp as wp
    import mujoco_warp as mjw

    model, contract = parent
    # Also cover adoption of a parent explicitly disabling native recovery.
    tree = ET.fromstring(model.read_text())
    flag = tree.find('./option/flag')
    if flag is None:
        flag = ET.SubElement(tree.find('option'), 'flag')
    flag.set('autoreset', 'disable')
    model.write_text(ET.tostring(tree, encoding='unicode'))
    identity = json.loads(contract.read_text())
    identity['model_sha256'] = sha256(model)
    contract.write_text(json.dumps(identity))
    reference, path = build_reference(model, contract, tmp_path / 'reference')
    exported = mujoco.MjModel.from_xml_path(str(reference))
    assert not int(exported.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    with wp.ScopedDevice('cpu'):
        mjw.put_model(exported)  # Real upstream validation, no physics step.
    runtime = GooseSourceRuntime(reference, path)
    assert int(runtime.model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    runtime.data.qvel[0] = np.nan
    with pytest.raises(FloatingPointError, match='no automatic reset'):
        runtime.step(np.zeros(18))
    assert not np.isfinite(runtime.data.qvel).all()


@pytest.mark.parametrize('allowed_foot', [False, True])
def test_warp_host_contacts_use_canonical_ids_without_false_body_touch(tmp_path, allowed_foot):
    import warp as wp
    import mujoco_warp as mjw

    _, path = fixture_runtime(tmp_path)
    model_path = tmp_path / 'robot.xml'
    tree = ET.fromstring(model_path.read_text())
    tree.find('option').set('integrator', 'implicitfast')
    ET.SubElement(tree.find('./worldbody/body'), 'geom', name='test_toe', type='box', size='.1 .1 .1')
    model_path.write_text(ET.tostring(tree, encoding='unicode'))
    contract = json.loads(path.read_text())
    contract['model_sha256'] = sha256(model_path)
    contract['passive_contacts'] = [{'name': 'test_toe'}] if allowed_foot else []
    path.write_text(json.dumps(contract))
    runtime = GooseSourceRuntime(model_path, path)
    runtime.data.qpos[2] = .09
    mujoco.mj_forward(runtime.model, runtime.data)  # Actual contact solve, no integration.
    assert runtime.data.ncon == 4
    assert runtime._nonfoot_ground_contact() is (not allowed_foot)
    expected_ids = runtime.data.contact.geom.copy()
    backend = copy.copy(runtime.model)
    backend.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    host = mujoco.MjData(runtime.model)
    with wp.ScopedDevice('cpu'):
        device_data = mjw.put_data(backend, runtime.data, nworld=1, nconmax=128, njmax=512)
        mjw.get_data_into(host, backend, device_data)
    np.testing.assert_array_equal(host.contact.geom, expected_ids)
    assert all(c.geom1 == c.geom2 == 0 for c in host.contact)  # Reproduces pinned bridge boundary.
    runtime.data = host
    assert runtime._nonfoot_ground_contact() is (not allowed_foot)
    assert runtime.physics_integrations == runtime.controller_updates == 0
