"""Exact cold mechanical reception and fail-closed target identity checks."""
import copy,json,os
from pathlib import Path
import mujoco
import numpy as np
import pytest
from bevy_microduck_tools.goose.artifacts import export_rapier_plant,write_json
from bevy_microduck_tools.goose.native_geometry315 import collision_geom_vertices

@pytest.fixture
def intake(tmp_path,monkeypatch):
    root=os.environ.get('GOOSE_MOVE023_TARGET_INTAKE')
    if not root:pytest.skip('Requires the frozen own Move023 target intake')
    monkeypatch.chdir(tmp_path)
    root=Path(root)
    manifest=json.loads((root/'intake_manifest.json').read_text())
    return root,manifest['bundle']

def test_cold_export_preserves_native_support_and_pose_without_integrating(intake,tmp_path):
    root,bundle=intake
    path=tmp_path/'plant.json'
    export_rapier_plant(bundle,path,initial_state_path=root/'model/cold_initial.npz')
    plant=json.loads(path.read_text())
    expected=json.loads((root/'model/plant.json').read_text())
    assert plant==expected
    m=mujoco.MjModel.from_xml_path(bundle['model_path'])
    d=mujoco.MjData(m);d.qpos[:]=np.load(root/'model/cold_initial.npz')['qpos']
    mujoco.mj_fwdPosition(m,d)
    for geom in plant['colliders']:
        gid=m.geom(geom['name']).id
        if geom['kind']=='convex_mesh':
            np.testing.assert_array_equal(geom['vertices_local_m'],collision_geom_vertices(m,gid))
        else:np.testing.assert_array_equal(geom['half_extents_m'],m.geom_size[gid])
    assert len(plant['colliders'])==11 and len(plant['bodies'])==21
    assert plant['geometry_export']['actual_integrals']==0 and d.time==0

@pytest.mark.parametrize('field,value',[
    ('phase_frequency_hz',1.2),('physics_dt_s',.005),
    ('controller_revision','sampled_pd_kp_parent_kd_quarter_v1'),('model_sha256','a'*64)])
def test_export_rejects_changed_identity_before_writing(intake,tmp_path,field,value):
    root,bundle=intake
    altered=copy.deepcopy(bundle)
    contract=json.loads(Path(bundle['contract_path']).read_text());contract[field]=value
    altered['contract_path']=str(tmp_path/'control.json');write_json(Path(altered['contract_path']),contract)
    output=tmp_path/'plant.json'
    with pytest.raises(ValueError):
        export_rapier_plant(altered,output,initial_state_path=root/'model/cold_initial.npz')
    assert not output.exists()

def test_legacy_export_does_not_silently_accept_new_engine(intake,tmp_path):
    root,bundle=intake
    altered=copy.deepcopy(bundle)
    contract=json.loads(Path(bundle['contract_path']).read_text())
    contract['candidate']='goose_task_proxy_11_rigid_braking_v1'
    altered['candidate']=contract['candidate'];altered['contract_path']=str(tmp_path/'control.json')
    write_json(Path(altered['contract_path']),contract)
    with pytest.raises(ValueError,match='MuJoCo 3.10.0'):
        export_rapier_plant(altered,tmp_path/'plant.json',initial_state_path=root/'model/cold_initial.npz')
