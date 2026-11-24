"""Output heads: map the final hidden state to the next density frame.

Every head ends in a single zero-initialised ``Linear``, so every model starts as exact persistence
(n_hat_{k+1} = n_k; for ``softmax`` up to renormalisation of max(n_k, N_FLOOR)). The update from
n_k to n_hat is done in float64; the hidden state is float32.

Heads
-----
unconstrained           n_hat = n_k + sigma_delta * delta
unconstrained_meanfree  n_hat = n_k + sigma_delta * (delta - mean_x delta)    exact norm, no locality
softmax                 n_hat = softmax_x(log max(n_k, N_FLOOR) + delta) / DX  exact norm (= 1), n_hat > 0
flux                    J = sigma_J * g on the NFACE interior faces, J = 0 at both walls,
                        n_hat = n_k - (DT/DX) * (J_{i+1/2} - J_{i-1/2})         exact local conservation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lqm import constants as C
from lqm.model.normaliser import Normaliser

HEADS: tuple[str, ...] = ("unconstrained", "unconstrained_meanfree", "softmax", "flux")


class Head(nn.Module):
    """Density-update head.

    Parameters
    ----------
    kind : str
        One of ``HEADS``.
    d_model : int
        Width of the hidden state.
    normaliser : Normaliser
        Supplies ``sigma_delta`` (unconstrained heads) and ``sigma_J`` (flux head).
    """

    def __init__(self, kind: str, d_model: int, normaliser: Normaliser) -> None:
        super().__init__()
        if kind not in HEADS:
            raise ValueError(f"unknown head {kind!r}; expected one of {HEADS}")
        self.kind = kind
        self.sigma_delta = float(normaliser.sigma_delta)
        self.sigma_J = float(normaliser.sigma_J)
        self.proj = nn.Linear(d_model, C.NFACE if kind == "flux" else C.NX)
        self.zero_init()

    def zero_init(self) -> None:
        """Zero the final layer so the model starts as persistence."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, n_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the head.

        Parameters
        ----------
        h : torch.Tensor
            [B, T, d_model] hidden state (float32).
        n_k : torch.Tensor
            [B, T, NX] float64 current frames.

        Returns
        -------
        n_next : torch.Tensor
            [B, T, NX] float64 predicted next frames.
        J : torch.Tensor or None
            [B, T, NFACE] float64 predicted frame-averaged flux (``flux`` head only).
        """
        if n_k.dtype != torch.float64:
            raise TypeError(f"n_k must be float64, got {n_k.dtype}")
        out = self.proj(h).double()
        if self.kind == "unconstrained":
            return n_k + self.sigma_delta * out, None
        if self.kind == "unconstrained_meanfree":
            return n_k + self.sigma_delta * (out - out.mean(dim=-1, keepdim=True)), None
        if self.kind == "softmax":
            logits = torch.log(n_k.clamp_min(C.N_FLOOR)) + out
            return torch.softmax(logits, dim=-1) / C.DX, None
        J = self.sigma_J * out
        J_walls = F.pad(J, (1, 1))  # zero current through both walls
        return n_k - (C.DT / C.DX) * (J_walls[..., 1:] - J_walls[..., :-1]), J
