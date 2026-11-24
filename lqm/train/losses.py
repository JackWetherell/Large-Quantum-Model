"""Training losses and one-step metrics (float64).

Every model is trained on the MSE of (n_hat_{k+1} - n_{k+1}) / sigma_delta over all positions and
grid points; the reported metric is the one-step L1 error sum_i |n_hat - n| DX.
"""

from __future__ import annotations

import torch

from lqm import constants as C


def scaled_mse(n_pred: torch.Tensor, n_target: torch.Tensor, sigma_delta: float) -> torch.Tensor:
    """mean(((n_pred - n_target) / sigma_delta)^2) in float64."""
    return (((n_pred.double() - n_target.double()) / sigma_delta) ** 2).mean()


def l1_per_frame(n_pred: torch.Tensor, n_target: torch.Tensor) -> torch.Tensor:
    """[..., NX] -> [...] float64 L1 distance sum_i |n_pred - n_target| DX."""
    return (n_pred.double() - n_target.double()).abs().sum(dim=-1) * C.DX


def teacher_forced_loss(model: torch.nn.Module, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Loss with exact inputs: frames 0..T-2 predict frames 1..T-1.

    Parameters
    ----------
    model : GPT or MLP
        The model.
    n : torch.Tensor
        [B, T, NX] exact frames.
    v : torch.Tensor
        [B, NX] potentials.
    """
    out = model(n[:, :-1], v)["n_next"]
    return scaled_mse(out, n[:, 1:], model.normaliser.sigma_delta)


def pushforward_loss(model: torch.nn.Module, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Two-pass "pushforward" loss.

    Pass 1 (no grad) makes teacher-forced predictions n_hat_1..n_hat_{T-2}; pass 2 takes the inputs
    [n_0, n_hat_1, ..., n_hat_{T-2}] and is trained against the exact targets 1..T-1, so the model
    learns to correct its own one-step errors.
    """
    with torch.no_grad():
        pred = model(n[:, :-1], v)["n_next"]  # pred[:, k] ~ n_{k+1}, float64
    inputs = torch.cat([n[:, :1].double(), pred[:, :-1]], dim=1)
    out = model(inputs, v)["n_next"]
    return scaled_mse(out, n[:, 1:], model.normaliser.sigma_delta)
