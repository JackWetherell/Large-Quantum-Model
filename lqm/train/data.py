"""GPU-resident trajectory data and parity augmentation (no DataLoader)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from lqm.io import load_split, potential

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """Whole trajectories held on one device.

    Attributes
    ----------
    n : torch.Tensor
        [N, NT, NX] float32 densities.
    v : torch.Tensor
        [N, NX] float64 full potential for t > 0, v = v0 + E*x.
    """

    n: torch.Tensor
    v: torch.Tensor

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str) -> TrajectoryData:
        """Load ``n``, ``v0`` and ``E`` of one split onto ``device``."""
        d = load_split(path, keys=["n", "v0", "E"], device=device)
        data = cls(n=d["n"], v=potential(d["v0"], d["E"]))
        logger.info("loaded %s: n %s on %s (%.2f GB)", path, tuple(data.n.shape), device, data.n.nbytes / 1e9)
        return data

    def __len__(self) -> int:
        return self.n.shape[0]


def parity_flip(n: torch.Tensor, v: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror x -> -x independently per trajectory with probability ``p``.

    Exact symmetry of the dynamics on the symmetric grid: n(x) -> n(-x), v(x) -> v(-x)
    (equivalently v0 mirrored and E -> -E).

    Parameters
    ----------
    n : torch.Tensor
        [B, T, NX] densities.
    v : torch.Tensor
        [B, NX] potentials.
    p : float
        Flip probability.
    """
    if p <= 0:
        return n, v
    flip = torch.rand(n.shape[0], device=n.device) < p
    n = torch.where(flip[:, None, None], n.flip(-1), n)
    v = torch.where(flip[:, None], v.flip(-1), v)
    return n, v


def epoch_batches(n_traj: int, batch_size: int, device: torch.device | str) -> Iterator[tuple[int, torch.Tensor]]:
    """Endless ``(epoch, indices)`` stream: a fresh permutation each epoch, incomplete batches dropped."""
    if batch_size > n_traj:
        raise ValueError(f"batch_size {batch_size} > number of trajectories {n_traj}")
    epoch = 0
    while True:
        perm = torch.randperm(n_traj, device=device)
        for s in range(0, n_traj - batch_size + 1, batch_size):
            yield epoch, perm[s : s + batch_size]
        epoch += 1
