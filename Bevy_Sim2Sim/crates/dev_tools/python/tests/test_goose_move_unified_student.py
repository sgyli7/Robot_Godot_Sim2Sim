import copy

import pytest
import torch
from rsl_rl.models import MLPModel

from bevy_microduck_tools.goose.move_unified_student import (
    SharedMoveModel, initialize_shared_from_native_leaf, shared_task_gradient_summary)
from bevy_microduck_tools.goose.native_action_units import (
    PublicActionScale, PublicUnitsGaussian, native_coordinate_mse)


def mapped_actor(kind):
    actor = kind({'actor': torch.zeros((1, 65))}, {'actor': ['actor']}, 'actor', 18,
        hidden_dims=[16, 8], activation='elu', obs_normalization=True,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'log'})
    scale = torch.linspace(.05, .5, 18)
    actor.mlp.append(PublicActionScale(scale))
    actor.distribution = PublicUnitsGaussian(actor.distribution, scale)
    return actor


def test_shared_transfer_retains_parent_units_statistics_sigma_and_applied_means():
    torch.manual_seed(23)
    parent = mapped_actor(MLPModel)
    parent.obs_normalizer.update(torch.randn((64, 65)))
    with torch.no_grad():
        parent.distribution.log_std_param.fill_(-1.7)
        parent.mlp[4].bias.fill_(5.)
    student = mapped_actor(SharedMoveModel)
    initialize_shared_from_native_leaf(student, parent.state_dict())
    values = {'actor': torch.randn((12, 65))}
    assert torch.equal(student(values), parent(values).clamp(-1., 1.))
    assert isinstance(student.mlp[-1], PublicActionScale)
    assert torch.equal(student.distribution.log_std_param, parent.distribution.log_std_param)
    fixed = copy.deepcopy(student.obs_normalizer.state_dict())
    student.update_normalization({'actor': torch.randn((256, 65)) * 10})
    assert all(torch.equal(value, student.obs_normalizer.state_dict()[key]) for key, value in fixed.items())
    assert all(parameter.requires_grad for parameter in student.mlp.parameters())


def test_gradient_measurement_does_not_update_weights_or_accumulate_gradients():
    parent = mapped_actor(MLPModel)
    student = mapped_actor(SharedMoveModel)
    initialize_shared_from_native_leaf(student, parent.state_dict())
    states = torch.randn((8, 65))
    labels = torch.randn((8, 18)) * .1
    before = copy.deepcopy(student.state_dict())
    report = shared_task_gradient_summary(student, states, labels,
        {'left': torch.arange(8) < 4, 'right': torch.arange(8) >= 4})
    assert report['optimizer_updates'] == 0 and not report['causal_gradient_conflict_claim']
    assert set(report['task_gradient_norms']) == {'left', 'right'}
    assert -1.00001 <= report['pairwise_cosines']['left:right'] <= 1.00001
    assert all(parameter.grad is None for parameter in student.parameters())
    assert all(torch.equal(value, student.state_dict()[key]) for key, value in before.items())


def test_transfer_rejects_expert_checkpoint_and_wrong_action_units():
    parent = mapped_actor(MLPModel)
    state = copy.deepcopy(parent.state_dict())
    state['mlp.protected_weight_0'] = state['mlp.0.weight']
    with pytest.raises(ValueError, match='plain MLP'):
        initialize_shared_from_native_leaf(mapped_actor(SharedMoveModel), state)
    state = copy.deepcopy(parent.state_dict())
    state['mlp.5.public_action_scale'] *= 2
    with pytest.raises(ValueError, match='unit maps'):
        initialize_shared_from_native_leaf(mapped_actor(SharedMoveModel), state)


def test_gradient_report_counts_an_actor_update_carried_only_by_bias():
    parent = mapped_actor(MLPModel)
    student = mapped_actor(SharedMoveModel)
    initialize_shared_from_native_leaf(student, parent.state_dict())
    student.mlp.requires_grad_(False)
    bias = student.mlp[4].bias
    bias.requires_grad_(True)
    states = torch.zeros((4, 65))
    labels = student({'actor': states}).detach() + .01
    loss = native_coordinate_mse(student, student({'actor': states}), labels)
    expected = torch.autograd.grad(loss, bias)[0].norm()
    report = shared_task_gradient_summary(student, states, labels,
        {'all': torch.ones(4, dtype=torch.bool)})
    assert report['parameter_scope'] == 'all_trainable_actor_mean_parameters_including_bias'
    assert report['parameter_names'] == ['mlp.4.bias']
    assert report['parameter_count'] == 18
    assert report['task_gradient_norms']['all'] == pytest.approx(float(expected))
    assert expected > 0 and bias.grad is None
