"""Global input/output scales for the density models, computed from the training split only.

All scales are population standard deviations (ddof = 0) over every trajectory, frame and grid
point (or face) of the training split, computed in float64 with a two-pass algorithm. Atomic units.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from lqm import constants as C

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Normaliser:
    """Scales that make model inputs and outputs O(1).

    Attributes
    ----------
    sigma_n : float
        Std of the density n [1/bohr].
    sigma_v : float
        Std of the mean-free potential v - mean_x(v) [hartree].
    sigma_delta : float
        Std of the one-frame change n_{k+1} - n_k [1/bohr].
    sigma_J : float
        Std of the frame-averaged flux on the NFACE interior faces [1/(a.u. of time)].
    """

    sigma_n: float
    sigma_v: float
    sigma_delta: float
    sigma_J: float

    def to_dict(self) -> dict[str, float]:
        """Plain-float dict (stored in checkpoints and ``config.json``)."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Normaliser:
        """Inverse of :meth:`to_dict`."""
        return cls(**{k: float(d[k]) for k in ("sigma_n", "sigma_v", "sigma_delta", "sigma_J")})


def frame_flux(n: torch.Tensor) -> torch.Tensor:
    """Frame-averaged flux that carries frame k into frame k+1 by the discrete continuity equation.

    J_{k, i+1/2} = -(DX/DT) * sum_{m <= i} (n_{k+1} - n_k)_m on the interior faces i = 0..NFACE-1,
    so that n_{k+1} = n_k - (DT/DX) * (J_{i+1/2} - J_{i-1/2}) with J = 0 at both walls.

    Parameters
    ----------
    n : torch.Tensor
        [..., T, NX] densities.

    Returns
    -------
    torch.Tensor
        [..., T-1, NFACE] flux, same dtype as ``n``.
    """
    dn = n[..., 1:, :] - n[..., :-1, :]
    return -(C.DX / C.DT) * torch.cumsum(dn, dim=-1)[..., : C.NFACE]


_QUANTITIES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "sigma_n": lambda n, v: n,
    "sigma_v": lambda n, v: v - v.mean(dim=-1, keepdim=True),
    "sigma_delta": lambda n, v: n[:, 1:] - n[:, :-1],
    "sigma_J": lambda n, v: frame_flux(n),
}


def compute_normaliser(n: torch.Tensor, v: torch.Tensor, chunk_size: int = 256) -> Normaliser:
    """Compute the normaliser from a (training) split.

    Parameters
    ----------
    n : torch.Tensor
        [N, T, NX] densities (any float dtype, any device).
    v : torch.Tensor
        [N, NX] full potential for t > 0, v = v0 + E*x.
    chunk_size : int
        Trajectories per chunk (bounds the float64 working memory).

    Returns
    -------
    Normaliser
    """
    n_traj = n.shape[0]
    chunks = [(s, min(s + chunk_size, n_traj)) for s in range(0, n_traj, chunk_size)]
    sums = dict.fromkeys(_QUANTITIES, 0.0)
    counts = dict.fromkeys(_QUANTITIES, 0)
    for s, e in chunks:
        nc, vc = n[s:e].double(), v[s:e].double()
        for name, f in _QUANTITIES.items():
            q = f(nc, vc)
            sums[name] += q.sum().item()
            counts[name] += q.numel()
    means = {k: sums[k] / counts[k] for k in _QUANTITIES}
    ssd = dict.fromkeys(_QUANTITIES, 0.0)
    for s, e in chunks:
        nc, vc = n[s:e].double(), v[s:e].double()
        for name, f in _QUANTITIES.items():
            ssd[name] += ((f(nc, vc) - means[name]) ** 2).sum().item()
    norm = Normaliser(**{k: math.sqrt(ssd[k] / counts[k]) for k in _QUANTITIES})
    logger.info("normaliser from %d trajectories: %s", n_traj, norm.to_dict())
    return norm
