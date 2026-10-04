"""Real frozen model/manager-loop checks of the 004 compatibility candidate."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import mujoco

pytest.importorskip('mjlab')
from bevy_microduck_tools.goose.artifacts import sha256
from bevy_microduck_tools.goose.mjlab_baseline import (
    build_task_proxy_reference, make_entity_cfg, TASK_PROXY_CANDIDATE)
from bevy_microduck_tools.goose.mjlab_env import GooseDevelopmentEnv, make_development_env_cfg


@pytest.fixture
def frozen_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package = os.environ.get('GOOSE_FROZEN_TASK_PROXY_PACKAGE')
    if not package:
        pytest.skip('Independent frozen external 004 package required')
    package = Path(package).resolve(strict=True)
    return (package / 'robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml',
            package / 'robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json', package)


def test_named_export_preserves_all_physical_arrays(frozen_parent, tmp_path):
    model, path = build_task_proxy_reference(*frozen_parent, tmp_path / 'candidate')
    before = mujoco.MjModel.from_xml_path(str(frozen_parent[0]))
    after = mujoco.MjModel.from_xml_path(str(model))
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_axis','jnt_range',
                  'geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity',
                  'geom_friction','geom_margin','geom_gap','geom_solref','geom_solimp',
                  'dof_armature','dof_damping','dof_frictionloss','eq_type','eq_data','eq_solref',
                  'actuator_trnid','actuator_gainprm','mesh_vert','mesh_face','exclude_signature'):
        np.testing.assert_array_equal(getattr(after,field),getattr(before,field))
    assert after.ngeom == before.ngeom == 12 and after.nbody == before.nbody == 22
    assert after.opt.disableflags == before.opt.disableflags | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    c = json.loads(path.read_text())
    assert c['candidate'] == TASK_PROXY_CANDIDATE and c['model_sha256'] == sha256(model)
    assert not c['training_release']
    cfg = make_entity_cfg(model,path)
    assert cfg.articulation.actuators[0].target_names_expr[5] == 'beak_input_rotor'
    with pytest.raises(FileExistsError):
        build_task_proxy_reference(*frozen_parent,tmp_path/'candidate')


def test_real_upstream_loop_contact_history_and_selective_reset(frozen_parent,tmp_path):
    model,path = build_task_proxy_reference(*frozen_parent,tmp_path/'candidate')
    cfg = make_development_env_cfg(model,path,num_envs=2)
    env = GooseDevelopmentEnv(cfg,'cpu')
    try:
        obs,_ = env.reset()
        term = env.action_manager.get_term('goose')
        assert env.contact_adapter is not None and env.sim.wp_model.callback.control is not None
        assert obs['actor'].shape == (2,65)
        assert env.sim.mj_model.ngeom == 12 and env.sim.mj_model.nbody == 22
        native = mujoco.MjModel.from_xml_path(str(model))
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_axis','jnt_range',
                      'geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity',
                      'geom_friction','geom_margin','geom_gap','geom_solref','geom_solimp',
                      'dof_armature','dof_damping','dof_frictionloss','eq_type','eq_data','eq_solref',
                      'actuator_trnid','actuator_gainprm','mesh_vert','mesh_face','exclude_signature'):
            np.testing.assert_allclose(getattr(env.sim.mj_model,field),getattr(native,field),atol=1e-13,rtol=0)
        assert env.sim.mj_model.opt.disableflags == native.opt.disableflags
        assert env.sim.mj_model.opt.timestep == .02
        assert env.sim.mj_model.narena == native.narena
        initial = env.sim.data.qpos.clone()
        action = torch.zeros((2,18),dtype=torch.float64)
        action[1,0] = .005
        for _ in range(3):
            before = env.sim.data.time.clone()
            obs,*_ = env.step(action)
            torch.testing.assert_close(env.sim.data.time-before,torch.full((2,),.02),atol=2e-6,rtol=0)
            assert obs['actor'].shape == (2,65) and torch.isfinite(obs['actor']).all()
            assert env.contact_adapter.errors.numpy()[0] == 0
        assert term.drive.prepared_ticks.tolist() == term.drive.completed_ticks.tolist() == [3,3]
        assert not torch.equal(env.sim.data.qpos[0],env.sim.data.qpos[1])
        before = env.sim.data.qpos[1].clone()
        env.reset(env_ids=torch.tensor([0]))
        torch.testing.assert_close(env.sim.data.qpos[0],initial[0],atol=1e-7,rtol=0)
        torch.testing.assert_close(env.sim.data.qpos[1],before,atol=0,rtol=0)
        assert term.drive.completed_ticks.tolist() == [0,3]
        env.step(action)
        assert term.drive.completed_ticks.tolist() == [1,4]
    finally:
        env.close()


def test_missing_mandatory_adapter_profile_is_rejected(frozen_parent,tmp_path):
    model,path = build_task_proxy_reference(*frozen_parent,tmp_path/'candidate')
    c = json.loads(path.read_text())
    c['upstream_baseline']['contact_adapter_required'] = False
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='contact adapter'):
        make_entity_cfg(model,path)
