"""Trivial ``Stepper`` baselines (and an oracle for testing the harness).

All three work on whatever device their inputs live on and return float64, as required by
``lqm.model.stepper.Stepper``.
"""

from __future__ import annotations

import numpy as np
import torch


class PersistenceStepper:
    """n̂_{k+1} = n_k: the density never changes (exactly norm- and positivity-preserving)."""

    name: str = "persistence"
    context: int | None = 1

    def predict_next(self, n_hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return the most recent frame."""
        return n_hist[:, -1].to(torch.float64, copy=True)

    def predict_teacher_forced(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Position k predicts frame k+1 as n_k."""
        return n.to(torch.float64, copy=True)


class LinearExtrapolationStepper:
    """n̂_{k+1} = 2 n_k − n_{k−1}, with persistence for the first step (only n_0 known).

    Preserves ∫n dx exactly in exact arithmetic (2·1 − 1 = 1); no clipping, so it can go negative.
    Note that a free rollout from n_0 alone is *identical* to persistence: the first step gives
    n̂_1 = n_0, after which 2 n_0 − n_0 = n_0 (exact in floating point, by Sterbenz's lemma). It is
    only informative in teacher-forced mode or from a history of at least two exact frames.
    """

    name: str = "linear"
    context: int | None = 2

    def predict_next(self, n_hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Extrapolate linearly from the last two frames (persistence if only one is given)."""
        last = n_hist[:, -1].to(torch.float64)
        if n_hist.shape[1] == 1:
            return last.clone()
        return 2.0 * last - n_hist[:, -2].to(torch.float64)

    def predict_teacher_forced(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Position 0 predicts n_0; position k ≥ 1 predicts 2 n_k − n_{k−1}."""
        n = n.to(torch.float64)
        out = n.clone()
        out[:, 1:] = 2.0 * n[:, 1:] - n[:, :-1]
        return out


class OracleStepper:
    """Returns the exact next frame; for testing the evaluation harness only.

    Each query row is matched to a stored trajectory by its ground state ``n_hist[:, 0]`` and its
    potential ``v`` (L∞ distance ≤ ``tol``), and the frame index by the history length, so any
    batching, ordering or subset of the stored trajectories works. It therefore needs the full
    history (``context = None``) and distinct (n_0, v) pairs across stored trajectories.

    Parameters
    ----------
    n : array or torch.Tensor
        [N, T, NX] exact densities.
    v : array or torch.Tensor
        [N, NX] potentials for t > 0.
    tol : float
        Maximum L∞ distance between a query's (n_0, v) and the matched stored trajectory.
    """

    name: str = "oracle"
    context: int | None = None

    def __init__(self, n: np.ndarray | torch.Tensor, v: np.ndarray | torch.Tensor, tol: float = 1e-12) -> None:
        self._n = torch.as_tensor(n, dtype=torch.float64)
        self._v = torch.as_tensor(v, dtype=torch.float64)
        if self._n.ndim != 3 or self._v.shape != (self._n.shape[0], self._n.shape[2]):
            raise ValueError(f"expected n [N, T, NX] and v [N, NX], got {tuple(self._n.shape)}, {tuple(self._v.shape)}")
        self.tol = tol

    def _lookup(self, n0: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._n.device != n0.device:
            self._n, self._v = self._n.to(n0.device), self._v.to(n0.device)
        query = torch.cat([n0.to(torch.float64), v.to(torch.float64)], dim=-1)
        ref = torch.cat([self._n[:, 0], self._v], dim=-1)
        dist = torch.cdist(query, ref, p=float("inf"))  # [B, N]
        best_dist, idx = dist.min(dim=1)
        if (best_dist > self.tol).any():
            raise ValueError(f"oracle: no stored trajectory within tol={self.tol} (worst {best_dist.max().item():.3e})")
        return idx

    def predict_next(self, n_hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exact frame k for a history of k frames."""
        k = n_hist.shape[1]
        if k >= self._n.shape[1]:
            raise IndexError(f"oracle has {self._n.shape[1]} frames, cannot predict frame {k}")
        idx = self._lookup(n_hist[:, 0], v)  # first: may move the stored data to the query device
        return self._n[idx, k].clone()

    def predict_teacher_forced(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exact frames 1..T; positions whose target lies beyond the stored trajectory are NaN."""
        idx = self._lookup(n[:, 0], v)
        out = torch.full(n.shape, float("nan"), dtype=torch.float64, device=n.device)
        m = min(n.shape[1], self._n.shape[1] - 1)
        out[:, :m] = self._n[idx, 1 : m + 1]
        return out
