"""Native learning coordinates with exported Goose public position actions."""

from math import isfinite
from types import SimpleNamespace

import torch
from mjlab.envs import mdp as native_mdp
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from .mature_training import native_physical_action_std

REVISION = "goose_native_physical_actor_coordinates_v1"
TRACKING_REVISION = "goose_native_tracking_physical_actor_coordinates_v1"
DISTILLATION_REVISION = "goose_native_distillation_action_coordinates_v2"
ACTION_RATE_REVISION = "goose_native_action_rate_coordinates_v1"


class PublicActionScale(torch.nn.Module):
    """Fixed unit conversion included in the native deterministic exporter."""

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("public_action_scale", scale.detach().clone())

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.public_action_scale


class PublicUnitsGaussian(GaussianDistribution):
    """Native sigma parameters/bounds; probability density in public units."""

    def __init__(self, original, scale):
        parameter_name = "std_param" if original.std_type == "scalar" else "log_std_param"
        parameter = getattr(original, parameter_name)
        super().__init__(18, init_std=1., std_range=tuple(original.std_range),
            std_type=original.std_type, learn_std=parameter.requires_grad)
        # Preserve the native Parameter object already registered with Adam.
        setattr(self, parameter_name, parameter)
        self.register_buffer("public_action_scale", scale.detach().clone())

    def update(self, mlp_output: torch.Tensor) -> None:
        # The final fixed layer has already converted the mean. Native sigma
        # clamping stays in the original coordinates before this conversion.
        super().update(mlp_output)
        self._distribution = Normal(self.mean, self.std * self.public_action_scale)


def native_coordinate_mse(actor, predicted_public, target_public):
    """Use installed RSL MSE after undoing the fixed public action unit map.

    PPO density already accounts for this coordinate map. MSE does not:
    squaring public outputs would implicitly weight native axes by scale**2.
    Both predictions and own expert labels must return to the same native
    coordinates. This does not alter exported means, Gaussian exploration,
    model parameters or the native optimizer/storage/update implementation.
    """
    if (not isinstance(actor.distribution, PublicUnitsGaussian)
            or predicted_public.shape != target_public.shape
            or predicted_public.shape[-1] != 18):
        raise ValueError("Native-coordinate MSE requires matching mapped18-axis actions")
    scale = actor.distribution.public_action_scale
    return torch.nn.functional.mse_loss(predicted_public/scale, target_public/scale)


def native_coordinate_action_rate_l2(env, public_action_scale):
    """Delegate the installed penalty in its original unscaled action units.

    The Goose manager stores public actions after PublicActionScale. Undo that
    fixed map for both action histories in this reward-only view. Actual motor
    targets, action history and exported policy values are never modified.
    """
    if (len(public_action_scale) != 18
            or any(not isfinite(x) or x <= 0 for x in public_action_scale)):
        raise ValueError("Action-rate units require positive finite18-axis scales")
    manager = env.action_manager
    if manager.action.shape != manager.prev_action.shape or manager.action.shape[-1] != 18:
        raise ValueError("Action-rate units require matching18-axis histories")
    scale = manager.action.new_tensor(public_action_scale)
    view = SimpleNamespace(action_manager=SimpleNamespace(
        action=manager.action/scale, prev_action=manager.prev_action/scale))
    return native_mdp.action_rate_l2(view)


def initialize_native_action_units(runner, contract):
    """Keep native mean/sigma/Adam coordinates, expose public65 to18 actions.

    The pinned mjlab G1 recipe uses .25*effort/stiffness radians per action
    unit. Goose retains its existing public action scales. A fixed output
    map and corresponding Gaussian density convert units throughout learning
    and ONNX inference; multiplying the initial output weights alone would
    change Adam's coordinates. Critic, normalization and trainable parameter
    objects remain untouched. This does not assert equivalent physical
    exploration after slew/contact/thermal limits. Install on a fresh runner,
    including before full resume; never reinitialize a trained Actor.
    """
    return _initialize_native_units(runner, contract, 65, "log", REVISION)


def initialize_native_tracking_action_units(runner, contract):
    """Explicit private113 teacher, retaining the native scalar Gaussian.

    The18 public motor actions retain the same units and limits. Reference
    observations remain private teacher inputs, separate from Actor65.
    """
    return _initialize_native_units(runner, contract, 113, "scalar", TRACKING_REVISION)


def _initialize_native_units(runner, contract, observation_dim, std_type, revision):
    actor = runner.alg.actor
    if runner.alg.optimizer.state or getattr(runner, "current_learning_iteration", 0):
        raise ValueError("Physical action initialization requires a fresh runner")
    normalizer = getattr(actor, "obs_normalizer", None)
    count = getattr(normalizer, "count", None)
    if count is None or count.numel() != 1 or int(count.item()) != 0:
        raise ValueError("Physical action initialization requires fresh statistics")
    if (not torch.equal(normalizer._mean, torch.zeros_like(normalizer._mean))
            or not torch.equal(normalizer._var, torch.ones_like(normalizer._var))
            or not torch.equal(normalizer._std, torch.ones_like(normalizer._std))):
        raise ValueError("Physical action initialization requires fresh statistics")
    distribution = getattr(actor, "distribution", None)
    parameter_name = "std_param" if std_type == "scalar" else "log_std_param"
    parameter = getattr(distribution, parameter_name, None)
    if (actor.obs_dim != observation_dim or type(distribution) is not GaussianDistribution
            or distribution.std_type != std_type
            or not isinstance(parameter, torch.nn.Parameter)
            or parameter.shape != (18,)):
        raise ValueError(f"Physical action initialization requires a native{observation_dim} to18 Actor with {std_type} sigma")
    expected = torch.ones_like(parameter) if std_type == "scalar" else torch.zeros_like(parameter)
    if not torch.equal(parameter.detach(), expected):
        raise ValueError("Physical action initialization requires a fresh unit Gaussian")
    output = actor.mlp[-1]
    if (not isinstance(output, torch.nn.Linear) or output.out_features != 18
            or output.bias is None):
        raise ValueError("Physical action initialization requires a native18-axis mean")
    factors = parameter.new_tensor(native_physical_action_std(contract))
    if not bool(torch.isfinite(factors).all()) or not bool((factors > 0).all()):
        raise ValueError("Physical action factors must be finite and positive")

    # No random draw, parameter replacement or optimizer group alteration.
    # Full resume constructs this topology first, then restores all state.
    scaled_distribution = PublicUnitsGaussian(distribution, factors)
    actor.mlp.append(PublicActionScale(factors))
    actor.distribution = scaled_distribution
    return revision
