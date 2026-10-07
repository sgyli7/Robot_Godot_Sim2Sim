"""Goose coordinate map for the installed RSL-RL symmetry extension.

Only training samples/means are transformed. Actual observations, actions,
drive targets, bodies and simulation state retain their public65/18 contract.
This is a soft bilateral prior; manufacturing masses and beak geometry are
not claimed to define an exactly symmetric physical MDP.
"""
import torch
from tensordict import TensorDict

from .artifacts import JOINT_ORDER

REVISION = 'goose_native_rsl_bilateral_half_cycle_v1'
PAIR_ORDER = (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11)
PAIR_SIGNS = (-1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1,
              -1, -1, -1, -1, -1, -1)


def mirror_public_axes(values):
    """Reflect neck/head signs and swap mirrored18-axis leg coordinates."""
    if values.shape[-1] != 18:
        raise ValueError('Goose symmetry requires18 public axes')
    order = torch.tensor(PAIR_ORDER, device=values.device)
    return values.index_select(-1, order)*values.new_tensor(PAIR_SIGNS)


def mirror_public_observation(values):
    """Reflect raw Actor65 or Critic69, using axial versus polar vectors.

    Advancing the exogenous phase by pi lets a straight symmetric birth start
    either foot rather than forcing both legs to execute the same motion.
    This map commutes with the original constant-rate phase advancement.
    The four Critic-only entries are world velocity3 and root height1.
    """
    if values.shape[-1] not in (65, 69):
        raise ValueError('Goose symmetry requires raw65/69 observations')
    result = values.clone()
    result[..., :3] *= values.new_tensor((-1., 1., -1.))
    result[..., 3:6] *= values.new_tensor((1., -1., 1.))
    result[..., 6:9] *= values.new_tensor((1., -1., -1.))
    for start in (9, 27, 45):
        result[..., start:start+18] = mirror_public_axes(values[..., start:start+18])
    result[..., 63:65] *= -1
    if values.shape[-1] == 69:
        result[..., 65:68] *= values.new_tensor((1., -1., 1.))
    return result


def augment_native_symmetry(*, env, obs=None, actions=None):
    """Native RSL callable: originals first, reflected samples second."""
    del env
    augmented_obs = augmented_actions = None
    if obs is not None:
        if set(obs.keys()) != {'actor', 'critic'}:
            raise ValueError('Explicit raw Actor/Critic groups required for Goose symmetry')
        if obs['actor'].shape[-1] != 65 or obs['critic'].shape[-1] != 69:
            raise ValueError('Goose symmetry retains Actor65 and Critic69')
        augmented_obs = TensorDict({name:torch.cat((values, mirror_public_observation(values)),dim=0)
            for name,values in obs.items()}, batch_size=[2*obs.batch_size[0]])
    if actions is not None:
        augmented_actions = torch.cat((actions, mirror_public_axes(actions)),dim=0)
    return augmented_obs, augmented_actions


def require_bilateral_contract(contract):
    """Reject a map inconsistent with public signs, limits or drive units."""
    if contract['joint_order'] != list(JOINT_ORDER):
        raise ValueError('Original Goose18-axis order required')
    joints = contract['joints']
    for axis, source in enumerate(PAIR_ORDER):
        a, original = joints[axis], joints[source]
        if a['q_neutral_rad'] != PAIR_SIGNS[axis]*original['q_neutral_rad']:
            raise ValueError('Bilateral neutral-coordinate mismatch')
        reflected_range = sorted(PAIR_SIGNS[axis]*v for v in original['range_rad'])
        if a['range_rad'] != reflected_range:
            raise ValueError('Bilateral joint-range mismatch')
        for field in ('action_scale_rad', 'torque_peak_limit_nm',
                'continuous_design_limit_nm', 'speed_limit_rad_s', 'kp_nm_rad', 'kd_nm_s_rad'):
            if a[field] != original[field]:
                raise ValueError(f'Bilateral actuator-unit mismatch: {field}')
