"""Explicit calibration of an existing Goose velocity conditioning encoder.

Public commands and measured/reward velocities remain metres per second.
This opt-in changes only a frozen RSL observation encoder's forward command
column. It preserves the parent's function at the corresponding old input;
it cannot establish that the robot physically attains the new command.
"""
import math

import torch

from rsl_rl.modules import EmpiricalNormalization

REVISION = "goose_frozen_forward_command_encoder_v1"


def remap_forward_command_encoder(actor, critic, *, factor):
    """Encode new vx as the parent's factor*vx without changing its weights.

    RSL divides by std+eps, so both the mean and that entire denominator
    must be rescaled. The calibrated buffers no longer denote empirical raw
    command statistics. Freeze their updates and repeat that freeze when
    loading a saved checkpoint; RSL's `until` is not a serialized buffer.
    Validate both models before mutating either of them.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Forward command factor must be finite and positive")
    pending = []
    for model, size in ((actor, 65), (critic, 69)):
        normalizer = model.obs_normalizer
        if (model.obs_dim != size or not isinstance(normalizer, EmpiricalNormalization)
                or normalizer._mean.shape != (1, size)
                or normalizer._std.shape != (1, size)
                or normalizer._var.shape != (1, size)):
            raise ValueError("Calibration requires the Goose65/69 RSL encoder")
        if hasattr(normalizer, "_goose_forward_command_factor"):
            raise ValueError("Forward command encoder was already calibrated")
        mean = normalizer._mean[:, 6] / factor
        std = (normalizer._std[:, 6] + normalizer.eps) / factor - normalizer.eps
        if (not math.isfinite(normalizer.eps) or normalizer.eps < 0
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all()):
            raise ValueError("Calibration requires a positive finite denominator")
        pending.append((normalizer, mean, std))
    with torch.no_grad():
        for normalizer, mean, std in pending:
            normalizer._mean[:, 6] = mean
            normalizer._std[:, 6] = std
            normalizer._var[:, 6] = std.square()
            normalizer.until = int(normalizer.count.item())
            normalizer._goose_forward_command_factor = factor
    return {"revision": REVISION, "forward_conditioning_factor": factor,
        "changed_observation_column": 6, "public_command_units": "m/s",
        "reward_command_rescaled": False, "physical_contract_changed": False,
        "encoder_buffers_are_empirical_raw_statistics": False,
        "freeze_required_on_checkpoint_reload": True}
