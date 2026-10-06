"""Native learning coordinates with exported Goose public position actions."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from .mature_training import native_physical_action_std

REVISION = "goose_native_physical_actor_coordinates_v1"


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
        super().__init__(18, init_std=1., std_range=tuple(original.std_range),
            std_type="log", learn_std=original.log_std_param.requires_grad)
        # Preserve the native Parameter object already registered with Adam.
        self.log_std_param = original.log_std_param
        self.register_buffer("public_action_scale", scale.detach().clone())

    def update(self, mlp_output: torch.Tensor) -> None:
        # The final fixed layer has already converted the mean. Native sigma
        # clamping stays in the original coordinates before this conversion.
        super().update(mlp_output)
        self._distribution = Normal(self.mean, self.std * self.public_action_scale)


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
    log_std = getattr(distribution, "log_std_param", None)
    if (actor.obs_dim != 65 or type(distribution) is not GaussianDistribution
            or not isinstance(log_std, torch.nn.Parameter)
            or log_std.shape != (18,)):
        raise ValueError("Physical action initialization requires a native65 to18 Actor")
    if not torch.equal(log_std.detach(), torch.zeros_like(log_std)):
        raise ValueError("Physical action initialization requires a fresh unit Gaussian")
    output = actor.mlp[-1]
    if (not isinstance(output, torch.nn.Linear) or output.out_features != 18
            or output.bias is None):
        raise ValueError("Physical action initialization requires a native18-axis mean")
    factors = log_std.new_tensor(native_physical_action_std(contract))
    if not bool(torch.isfinite(factors).all()) or not bool((factors > 0).all()):
        raise ValueError("Physical action factors must be finite and positive")

    # No random draw, parameter replacement or optimizer group alteration.
    # Full resume constructs this topology first, then restores all state.
    scaled_distribution = PublicUnitsGaussian(distribution, factors)
    actor.mlp.append(PublicActionScale(factors))
    actor.distribution = scaled_distribution
    return REVISION
