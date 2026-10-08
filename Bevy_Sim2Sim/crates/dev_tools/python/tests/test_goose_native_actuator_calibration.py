"""Real saturated-step reproduction and guards for a separate gain candidate."""
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.mjlab_env import RigidGooseActionCfg, make_development_env_cfg
from bevy_microduck_tools.goose.native_actuator_calibration import (
    build_reference, make_source_runtime)
from bevy_microduck_tools.goose.upstream_discrete_mjlab import DiscreteMujocoCfg


@pytest.fixture
def experiment():
    path = os.environ.get("GOOSE_GAIN_CALIBRATION_EXPERIMENT")
    if not path:
        pytest.skip("Requires the frozen native gain calibration and actual failure trace")
    return Path(path)


def test_native_bridge_uses_the_two_declared_gains(experiment):
    model, contract = experiment / "candidate/robot.xml", experiment / "candidate/contract.json"
    cfg = make_development_env_cfg(model, contract)
    assert isinstance(cfg.sim.mujoco, DiscreteMujocoCfg)
    assert isinstance(cfg.actions["goose"], RigidGooseActionCfg)
    assert cfg.decimation == 1 and not cfg.auto_reset
    assembled = cfg.scene.entities["robot"].spec_fn().compile()
    for name in ("right_ankle_roll", "left_ankle_roll"):
        joint = assembled.joint(name).id
        index = np.flatnonzero(assembled.actuator_trnid[:, 0] == joint).item()
        assert assembled.actuator_gainprm[index, 0] == 10.0
        assert assembled.actuator_biasprm[index, 1] == -10.0
        assert assembled.actuator_biasprm[index, 2] == -.175


@pytest.mark.parametrize("change", ("extra_torque", "hidden_timestep", "larger_foot"))
def test_calibration_rejects_unrelated_changes_even_with_updated_model_hash(
        experiment, tmp_path, change):
    destination = tmp_path / change
    shutil.copytree(experiment / "candidate", destination)
    tree = ET.parse(destination / "robot.xml")
    root = tree.getroot()
    if change == "extra_torque":
        list(root.find("actuator"))[11].set("forcerange", "-8 8")
    elif change == "hidden_timestep":
        root.find("option").set("timestep", "0.01")
    else:
        root.find('.//geom[@name="right_flexible_sole"]').set("size", "0.5 0.5 0.5")
    tree.write(destination / "robot.xml", encoding="unicode")
    contract = json.loads((destination / "contract.json").read_text())
    contract["model_sha256"] = sha256(destination / "robot.xml")
    (destination / "contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Calibration changed"):
        make_source_runtime(destination / "robot.xml", destination / "contract.json")


def test_builder_preserves_parent_and_does_not_overwrite(experiment, tmp_path):
    saved = json.loads((experiment / "candidate/contract.json").read_text())
    parent = saved["native_actuator_calibration"]
    before = sha256(Path(parent["parent_model"]))
    paths = build_reference(parent["parent_model"], parent["parent_contract"], tmp_path / "candidate")
    runtime = make_source_runtime(*paths)
    assert runtime.physics_integrations == runtime.controller_updates == 0
    assert not runtime.contract["training_release"]
    assert sha256(Path(parent["parent_model"])) == before
    with pytest.raises(FileExistsError):
        build_reference(parent["parent_model"], parent["parent_contract"], tmp_path / "candidate")


def test_actual_saturated_tick_is_reproduced_and_calibration_removes_overshoot(experiment):
    receipt = json.loads((experiment.parent / "ankle_failure_mechanism_185/observed_events.json").read_text())
    event = receipt["top"][0]
    rollout = np.load(experiment.parent / "ppo_lateral_exploration_182/pilot" / event["file"])
    tick, world = event["t"], event["world"]
    calibrated = make_source_runtime(experiment / "candidate/robot.xml", experiment / "candidate/contract.json")
    parent_path = calibrated.contract["native_actuator_calibration"]["parent_model"]
    models = (mujoco.MjModel.from_xml_path(parent_path), calibrated.model)
    results = []
    for model in models:
        data = mujoco.MjData(model)
        data.qpos[:] = rollout["qpos"][tick-1, world]
        data.qvel[:] = rollout["qvel"][tick-1, world]
        data.ctrl[:] = rollout["ctrl"][tick, world]
        model.actuator_forcerange[:] = rollout["force_bounds"][tick, world]
        # Partial-state mechanism probe: GPU warmstart/history was not saved.
        # No reset assistance or change is applied during the one actual Tick.
        mujoco.mj_step(model, data)
        assert data.time == .02 and not any(w.number for w in data.warning)
        joint = model.joint("right_ankle_roll")
        q = data.qpos[joint.qposadr[0]]
        excess = max(joint.range[0]-q, q-joint.range[1], 0.0)
        results.append((q, excess, data.qvel[joint.dofadr[0]]))
    assert results[0][1] > .5
    assert abs(results[0][0]-rollout["qpos"][tick, world, models[0].joint("right_ankle_roll").qposadr[0]]) < 1e-5
    assert results[1][1] <= .0001
    assert abs(results[1][2]) < 8.0
