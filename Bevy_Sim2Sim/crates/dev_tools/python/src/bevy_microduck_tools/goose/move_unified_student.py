"""One shared native RSL Actor; frozen experts are training labels only.

There are no expert branches or command gates in the exported student. The
public65 observations and original18 action units remain unchanged. History
adaptation, if needed, requires a separately versioned runtime interface.
"""

import torch
from torch import nn
from rsl_rl.models import MLPModel

from .native_action_units import PublicActionScale, native_coordinate_mse

REVISION = "goose_unified_move65_v1"


class SharedMoveModel(MLPModel):
    """Native shared MLP with explicitly frozen inherited normalizer statistics."""

    def update_normalization(self, obs):
        # Fixed statistics are selected before training, not updated according
        # to whichever task happened to dominate the most recent rollout.
        return None


class DriveEnvelopeActionScale(PublicActionScale):
    """Original fixed units with the drive's deterministic input envelope."""

    def forward(self, action):
        return super().forward(action).clamp(-1., 1.)


def initialize_shared_from_native_leaf(actor, state):
    """Copy a proven plain MLP into trainable weights, excluding expert gates.

    Call after fresh native action-unit initialization. This is an initial
    weight transfer, not an immutable old-policy branch or an ensemble. Clamp
    deterministic means to the original drive's public [-1,1] input envelope;
    training, ONNX and the actual drive then see the same deterministic map.
    Native RSL still owns stochastic sampling and the optimizer.
    """
    if type(actor) is not SharedMoveModel or actor.obs_dim != 65:
        raise ValueError("Shared transfer requires the plain public65 native MLP")
    if not isinstance(actor.mlp[-1], PublicActionScale):
        raise ValueError("Initialize native action units before shared transfer")
    if any("protected" in key or "frozen" in key for key in state):
        raise ValueError("Transfer requires an identified plain MLP checkpoint")
    scale_key = f"mlp.{len(actor.mlp) - 1}.public_action_scale"
    scale = state[scale_key]
    if not torch.equal(scale.to(actor.mlp[-1].public_action_scale),
                       actor.mlp[-1].public_action_scale):
        raise ValueError("Inherited leaf and student action unit maps must match")
    normalizer = {key.removeprefix("obs_normalizer."): value
                  for key, value in state.items() if key.startswith("obs_normalizer.")}
    actor.obs_normalizer.load_state_dict(normalizer, strict=True)
    distribution = {key.removeprefix("distribution."): value
                    for key, value in state.items() if key.startswith("distribution.")}
    actor.distribution.load_state_dict(distribution, strict=True)
    with torch.no_grad():
        for name, layer in actor.mlp.named_children():
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(state[f"mlp.{name}.weight"])
                layer.bias.copy_(state[f"mlp.{name}.bias"])
    actor.mlp[-1] = DriveEnvelopeActionScale(scale.to(actor.mlp[-1].public_action_scale))
    return REVISION


def shared_task_gradient_summary(actor, observations, targets, groups, *, limit=256):
    """Measure task gradients on the same actual rollout, without an update.

    Cosines are descriptive evidence. A negative value does not establish
    that conflicting gradients caused behavior to regress.
    """
    if observations.shape != (len(observations), 65) or targets.shape != (len(observations), 18):
        raise ValueError("Use matching actual65 observations and18-axis labels")
    named_parameters = tuple((name, parameter)
                             for name, parameter in actor.mlp.named_parameters()
                             if parameter.requires_grad)
    if not named_parameters:
        raise ValueError("Gradient measurement requires trainable Actor mean parameters")
    parameters = tuple(parameter for _, parameter in named_parameters)
    norms, losses, vectors = {}, {}, {}
    for name, mask in groups.items():
        indices = mask.nonzero(as_tuple=False).flatten()[:limit]
        if not len(indices):
            continue
        predictions = actor({"actor": observations[indices]})
        loss = native_coordinate_mse(actor, predictions, targets[indices])
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        vector = torch.cat([torch.zeros_like(parameter).flatten() if gradient is None
                            else gradient.detach().flatten()
                            for parameter, gradient in zip(parameters, gradients, strict=True)])
        vectors[name] = vector
        norms[name] = float(vector.norm())
        losses[name] = float(loss.detach())
    cosines = {}
    for left, a in vectors.items():
        for right, b in vectors.items():
            if left < right:
                denominator = a.norm() * b.norm()
                cosines[left + ":" + right] = (float(a.dot(b) / denominator)
                                               if float(denominator) > 0. else None)
    return {"task_gradient_norms": norms, "task_losses": losses,
            "pairwise_cosines": cosines, "optimizer_updates": 0,
            "parameter_scope": "all_trainable_actor_mean_parameters_including_bias",
            "parameter_names": ["mlp." + name for name, _ in named_parameters],
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "causal_gradient_conflict_claim": False}
