"""The ``Stepper`` protocol: the single interface between models and evaluation (frozen).

Anything that advances the density by one frame implements it: the GPT with each head, the MLP
baseline (``lqm.model``) and the trivial baselines (``lqm.eval.baselines``). Rollouts and
metrics in ``lqm.eval`` only ever talk to this protocol.

Conventions
-----------
- Densities ``n`` are in physical units: shape [..., NX], sum(n) * DX = 1.
- ``v`` is the full potential for t > 0, v = v0 + E*x (``lqm.io.potential``), shape [B, NX].
- Rollout tensors are float64; implementations may compute internally in float32 but must return
  float64 and must perform their final (norm-preserving) update in float64.
- Frame index k corresponds to t_k = k * DT; ``n_hist[:, -1]`` is the most recent frame.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Stepper(Protocol):
    """Advances the density one frame (DT) at a time."""

    name: str
    """Short identifier used in result files and plots."""

    context: int | None
    """Maximum number of past frames (including the current one) the model uses; None = unlimited."""

    def predict_next(self, n_hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Predict the next frame from a history.

        Parameters
        ----------
        n_hist : torch.Tensor
            [B, k, NX] float64, frames t_0 .. t_{k-1} (k >= 1); ``n_hist[:, 0]`` is the ground state.
        v : torch.Tensor
            [B, NX] float64 potential for t > 0.

        Returns
        -------
        torch.Tensor
            [B, NX] float64 prediction of the frame at t_k.
        """
        ...

    def predict_teacher_forced(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """One-step predictions for every position of an exact sequence.

        Parameters
        ----------
        n : torch.Tensor
            [B, T, NX] exact frames t_0 .. t_{T-1}.
        v : torch.Tensor
            [B, NX] potential for t > 0.

        Returns
        -------
        torch.Tensor
            [B, T, NX] float64 where ``out[:, k]`` predicts ``n[:, k+1]`` using frames 0..k only.
        """
        ...
