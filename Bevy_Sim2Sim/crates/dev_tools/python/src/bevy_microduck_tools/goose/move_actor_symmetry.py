"""Opt-in native RSL mirror auxiliary without relabeling terrain Critic data.

This is a soft bilateral policy prior, not an exact symmetry claim about the
manufactured robot or terrain. Actual observations/actions remain public65/18.
PPO evaluates its original rollout and Critic before this auxiliary is called.
"""
from functools import partial

import torch
from tensordict import TensorDict

from .native_action_units import native_coordinate_symmetry_loss
from .native_symmetry import mirror_public_axes, mirror_public_observation

REVISION = "goose_actor_only_native_mirror237_v1"


def augment_actor_only_symmetry(*, env, obs=None, actions=None):
    """Provide only Actor samples to the separate native auxiliary loss."""
    del env
    augmented_obs = augmented_actions = None
    if obs is not None:
        actor = obs["actor"]
        if actor.ndim != 2 or actor.shape[-1] != 65:
            raise ValueError("Actor-only mirror requires raw public65 observations")
        augmented_obs = TensorDict(
            {"actor": torch.cat((actor, mirror_public_observation(actor)), dim=0)},
            batch_size=[2 * actor.shape[0]],
        )
    if actions is not None:
        augmented_actions = torch.cat((actions, mirror_public_axes(actions)), dim=0)
    return augmented_obs, augmented_actions


def make_actor_only_mirror_cfg():
    """Use the existing Goose auxiliary coefficient; no PPO data augmentation."""
    return dict(
        data_augmentation_func=__name__ + ":augment_actor_only_symmetry",
        use_data_augmentation=False,
        use_mirror_loss=True,
        mirror_loss_coeff=0.1,
    )


def install_actor_only_native_mirror_units(algorithm):
    """Delegate the installed native loss after undoing fixed public units."""
    extension = algorithm.symmetry
    if (extension is None or extension.use_data_augmentation
            or not extension.use_mirror_loss
            or extension.data_augmentation_func is not augment_actor_only_symmetry):
        raise ValueError("Requires the separate Actor-only native mirror auxiliary")
    if getattr(extension, "_goose_native_mirror_units", False):
        raise ValueError("Native mirror unit conversion already installed")
    extension.compute_loss = partial(native_coordinate_symmetry_loss,
                                     extension.compute_loss)
    extension._goose_native_mirror_units = True
