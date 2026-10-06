"""Real compiled-model configuration and version boundary regressions."""
import dataclasses
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.mjlab_baseline import require_upstream_stack
from bevy_microduck_tools.goose.mjlab_env import (
    GooseDevelopmentEnv, RigidGooseActionCfg, make_development_env_cfg)
from bevy_microduck_tools.goose.native_geometry315 import collision_geom_vertices
from bevy_microduck_tools.goose.upstream_discrete_mjlab import DiscreteMujocoCfg


@pytest.fixture
def candidate_paths():
    root = os.environ.get("GOOSE_DISCRETE_MJLAB_EXPERIMENT")
    if not root:
        pytest.skip("Requires frozen native3.15/mjlab1.6 candidate")
    root = Path(root)/"candidate"
    return root/"robot.xml", root/"contract.json"


def test_bridge_preserves_native_options_and_original_rigid_action(candidate_paths):
    model_path, contract_path = candidate_paths
    cfg = make_development_env_cfg(model_path, contract_path)
    assert isinstance(cfg.sim.mujoco, DiscreteMujocoCfg)
    assert isinstance(cfg.actions["goose"], RigidGooseActionCfg)
    assert cfg.decimation == 1 and not cfg.auto_reset
    native = mujoco.MjModel.from_xml_path(str(model_path))
    assembled = mujoco.MjModel.from_xml_path(str(model_path))
    cfg.sim.mujoco.apply(assembled)
    for name in dir(native.opt):
        if name.startswith("_") or callable(getattr(native.opt, name)):
            continue
        np.testing.assert_array_equal(getattr(assembled.opt, name), getattr(native.opt, name))
    assert cfg.sim.mujoco.integrator == "discrete"
    assert not json.loads(contract_path.read_text())["training_release"]
    # The old stack does not become admissible by importing the new bridge.
    with pytest.raises(ValueError, match="baseline requires mjlab==1.3.0"):
        require_upstream_stack()


def test_bridge_rejects_hidden_integrator_or_timestep_changes(candidate_paths):
    model_path, contract_path = candidate_paths
    cfg = make_development_env_cfg(model_path, contract_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    for changes in ({"integrator": "implicitfast"}, {"timestep": .01},
                    {"disableflags": ("refsafe",)}):
        bad = dataclasses.replace(cfg.sim.mujoco, **changes)
        with pytest.raises(ValueError, match="safe 50Hz"):
            bad.apply(model)


def test_discrete_environment_rejects_generic_enum_configuration_before_gpu(candidate_paths):
    from mjlab.sim import MujocoCfg
    model_path, contract_path = candidate_paths
    cfg = make_development_env_cfg(model_path, contract_path)
    cfg.sim.mujoco = MujocoCfg(**dataclasses.asdict(cfg.sim.mujoco))
    with pytest.raises(ValueError, match="explicit mjlab bridge"):
        GooseDevelopmentEnv(cfg, "cuda:0")


def test_native315_support_matches_frozen_compiled_schema(candidate_paths):
    import importlib.util
    model_path, contract_path = candidate_paths
    root = Path(json.loads(contract_path.read_text())["upstream_mjlab"]["source_model"]).parent.parent
    spec = importlib.util.spec_from_file_location("frozen_geometry315", root/"native_geometry315.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    count = 0
    for geom in range(model.ngeom):
        if int(model.geom_type[geom]) not in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH)):
            continue
        np.testing.assert_array_equal(collision_geom_vertices(model, geom),
                                      frozen.collision_geom_vertices(model, geom))
        count += 1
    assert count == 11
    from bevy_microduck_tools.goose.native_geometry import collision_geom_vertices as original
    with pytest.raises(ValueError, match="requires MuJoCo 3.10.0"):
        original(model, model.geom("right_flexible_sole").id)


def test_adopted_source_replays_original_discrete_ticks_and_rejects_qualification_edits(
        candidate_paths, tmp_path):
    from bevy_microduck_tools.goose.upstream_discrete_mjlab import make_source_runtime
    model_path, contract_path = candidate_paths
    cfg = json.loads(contract_path.read_text())
    reference_root = Path(cfg["upstream_mjlab"]["source_model"]).parent.parent
    inputs = json.loads((reference_root/"replay_test_inputs.json").read_text())
    runtime = make_source_runtime(model_path, contract_path)
    runtime.data.qpos[:] = np.load(inputs["births"])["qpos"][0]
    runtime.data.qvel[:] = 0
    mujoco.mj_forward(runtime.model, runtime.data)
    saved = np.load(inputs["actions"])["action"]
    expected = np.load(reference_root/"discrete315_fresh_legal_two_ticks.npz")
    for tick in range(2):
        observation, info = runtime.step(saved[tick])
        assert observation.shape == (65,) and not info["auto_reset"]
        assert runtime.physics_integrations == runtime.controller_updates == tick+1
        np.testing.assert_allclose(runtime.data.qpos, expected["qpos"][tick], atol=1e-7, rtol=0)
        np.testing.assert_allclose(runtime.last_tau, expected["applied_joint_effort"][tick], atol=1e-6, rtol=0)
    cfg["upstream_mjlab"]["source_qualified"] = True
    modified = tmp_path/"false_qualification.json"
    modified.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="adoption identity"):
        make_source_runtime(model_path, modified)


def test_goal_backend_keeps_all_real_tree_joint_bindings(candidate_paths):
    from bevy_microduck_tools.goose.goal_recovery import make_goal_recovery_cfg
    model_path, contract_path = candidate_paths
    root = model_path.parent.parent
    protocol = json.loads((root/"protocol.json").read_text())
    cfg = make_goal_recovery_cfg(model_path, contract_path,
        poses=np.load(protocol["inputs"]["births"])["qpos"], num_envs=2)
    assert len(cfg.events["goal_birth"].params["joint_names"]) == 20
    assert "beak_input_rotor" in cfg.events["goal_birth"].params["joint_names"]
    assert len(cfg.actions) == 1 and isinstance(cfg.sim.mujoco, DiscreteMujocoCfg)
