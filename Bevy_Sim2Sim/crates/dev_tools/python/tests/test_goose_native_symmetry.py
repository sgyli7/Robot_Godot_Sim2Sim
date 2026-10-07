"""Check native RSL mirror loss against actual Goose coordinate semantics."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.extensions import Symmetry
from rsl_rl.models import MLPModel

from bevy_microduck_tools.goose.native_symmetry import (
    augment_native_symmetry, mirror_public_axes, mirror_public_observation,
    require_bilateral_contract)


@pytest.fixture
def robot_input():
    root = os.environ.get('GOOSE_DISCRETE_MJLAB_EXPERIMENT')
    if not root:pytest.skip('Received rigid Goose contract required')
    p = Path(root)/'candidate'
    return p, json.loads((p/'contract.json').read_text())


def test_public_reflection_is_an_involution_with_axial_polar_and_phase_signs():
    for dimension in (65,69):
        values = torch.arange(3*dimension,dtype=torch.float64).reshape(3,dimension)
        before = values.clone()
        mirrored = mirror_public_observation(values)
        assert torch.equal(values,before)
        assert torch.equal(mirror_public_observation(mirrored),before)
        torch.testing.assert_close(mirrored[0,:9],values[0,:9]*torch.tensor(
            [-1,1,-1,1,-1,1,1,-1,-1]))
        torch.testing.assert_close(mirrored[:,63:65],-values[:,63:65])
        assert torch.equal(mirrored[:,9+9],-values[:,9+15])
        assert torch.equal(mirrored[:,9+15],-values[:,9+9])
        if dimension==69:
            assert torch.equal(mirrored[:,66],-values[:,66])
            assert torch.equal(mirrored[:,68],values[:,68])


def test_native_extension_uses_auxiliary_loss_without_relabeling_ppo_samples():
    torch.manual_seed(211)
    obs = TensorDict({'actor':torch.randn(12,65),'critic':torch.randn(12,69)},batch_size=[12])
    actor = MLPModel(obs,{'actor':['actor']},'actor',18,hidden_dims=(16,),
        obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution',
            'std_type':'log','init_std':.2})
    actions = torch.randn(12,18)
    batch = SimpleNamespace(observations=obs,actions=actions)
    extension = Symmetry(env=None,data_augmentation_func=augment_native_symmetry,
        use_data_augmentation=False,use_mirror_loss=True,mirror_loss_coeff=.1)
    extension.augment_batch(batch,12)
    assert batch.observations is obs and batch.actions is actions
    loss = extension.compute_loss(actor,batch,12)
    assert loss.requires_grad and torch.isfinite(loss) and loss>0
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in actor.parameters())
    assert batch.actions is actions
    assert torch.equal(batch.observations['actor'][:12],obs['actor'])
    out,_ = augment_native_symmetry(env=None,obs=obs)
    assert out.batch_size == torch.Size([24])
    _,augmented = augment_native_symmetry(env=None,actions=actions)
    assert torch.equal(augmented[:12],actions)
    assert torch.equal(mirror_public_axes(augmented[12:]),actions)


def test_map_matches_native_leg_fk_and_preserves_original_target_limits(robot_input):
    mujoco = pytest.importorskip('mujoco')
    from scipy.spatial.transform import Rotation
    folder, contract = robot_input
    require_bilateral_contract(contract)
    invalid = copy.deepcopy(contract)
    invalid['joints'][9]['range_rad'][1] -= .1
    with pytest.raises(ValueError,match='joint-range'):
        require_bilateral_contract(invalid)
    model = mujoco.MjModel.from_xml_path(str(folder/'robot.xml'))
    left,right = mujoco.MjData(model),mujoco.MjData(model)
    q = model.qpos0.copy()
    rotation = Rotation.from_euler('xyz',[.08,.12,-.3]).as_matrix()
    q[:3] = [.12,-.04,.15]
    q[3:7] = np.roll(Rotation.from_matrix(rotation).as_quat(),1)
    public = np.array([.02,.08,-.1,.05,.1,.05,-.12,.15,.2,.4,-.18,.08,
        .17,-.22,-.18,-.35,.12,-.05])
    qids = [int(model.joint(n).qposadr[0]) for n in contract['joint_order']]
    q[qids] = public
    left.qpos[:] = q
    reflected = mirror_public_axes(torch.from_numpy(public)).numpy()
    mirrored_q = q.copy();mirrored_q[qids] = reflected
    reflection = np.diag([1.,-1.,1.])
    mirrored_q[:3] = reflection@q[:3]
    mirrored_q[3:7] = np.roll(Rotation.from_matrix(reflection@rotation@reflection).as_quat(),1)
    right.qpos[:] = mirrored_q
    mujoco.mj_fwdPosition(model,left);mujoco.mj_fwdPosition(model,right)
    for original,target in [('right_flexible_sole','left_flexible_sole'),
            ('left_flexible_sole','right_flexible_sole')]:
        a,b = model.geom(original).id,model.geom(target).id
        np.testing.assert_allclose(right.geom_xpos[b],reflection@left.geom_xpos[a],atol=1e-8,rtol=0)
        np.testing.assert_allclose(right.geom_xmat[b].reshape(3,3),
            reflection@left.geom_xmat[a].reshape(3,3)@reflection,atol=1e-8,rtol=0)
    for value,n in zip(reflected,contract['joint_order']):
        lo,hi = model.joint(n).range
        assert lo<=value<=hi
