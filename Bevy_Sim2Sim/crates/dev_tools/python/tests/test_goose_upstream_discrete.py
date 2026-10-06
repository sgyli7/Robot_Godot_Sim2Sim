"""Real source replay and qualification boundaries for the named new engine."""
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.upstream_discrete import make_source_runtime


@pytest.fixture
def frozen_inputs():
    path = os.environ.get("GOOSE_UPSTREAM_DISCRETE_EXPERIMENT")
    if not path:
        pytest.skip("Requires the frozen native3.15 admission inputs")
    return Path(path)


def test_real_native_replay_has_one_twenty_ms_step_and_original_drive(frozen_inputs):
    root = frozen_inputs
    protocol = json.loads((root/"replay_test_inputs.json").read_text())
    saved = np.load(protocol["actions"])
    expected = np.load(root/"discrete315_fresh_legal_two_ticks.npz")
    rt = make_source_runtime(root/"candidate/robot.xml", root/"candidate/contract.json")
    rt.data.qpos[:] = np.load(protocol["births"])["qpos"][0]
    rt.data.qvel[:] = 0
    mujoco.mj_forward(rt.model, rt.data)  # Genuine cold initialization only.
    initial_integrations, initial_updates = rt.physics_integrations, rt.controller_updates
    for tick in range(2):
        observation, info = rt.step(saved["action"][tick])
        assert observation.shape == (65,) and not info["auto_reset"]
        assert rt.data.time == pytest.approx((tick+1)*.02, abs=1e-12)
        assert rt.physics_integrations-initial_integrations == tick+1
        assert rt.controller_updates-initial_updates == tick+1
        np.testing.assert_allclose(rt.data.qpos, expected["qpos"][tick], atol=1e-7, rtol=0)
        np.testing.assert_allclose(rt.data.qvel, expected["qvel"][tick], atol=1e-5, rtol=0)
        np.testing.assert_allclose(rt.last_tau, expected["applied_joint_effort"][tick], atol=1e-6, rtol=0)
    assert not rt.contract["upstream_discrete"]["source_qualified"]
    assert not rt.contract["training_release"]


def test_discrete_profile_rejects_silent_control_changes(frozen_inputs, tmp_path):
    model = frozen_inputs/"candidate/robot.xml"
    cfg = json.loads((model.parent/"contract.json").read_text())
    cfg["joints"][0]["kp_nm_rad"] += 1
    modified = tmp_path/"control.json"
    modified.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="inherited control contract"):
        make_source_runtime(model, modified)


def test_discrete_profile_rejects_changed_solver_with_a_new_model_hash(frozen_inputs, tmp_path):
    copied = tmp_path/"copied"
    shutil.copytree(frozen_inputs/"candidate", copied)
    model, contract = copied/"robot.xml", copied/"contract.json"
    tree = ET.parse(model)
    option = tree.getroot().find("option")
    option.set("iterations", str(int(option.get("iterations"))-1))
    tree.write(model, encoding="unicode")
    cfg = json.loads(contract.read_text())
    cfg["model_sha256"] = sha256(model)
    contract.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="changed option iterations"):
        make_source_runtime(model, contract)
