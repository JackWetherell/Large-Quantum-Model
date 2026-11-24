"""Residual-MLP baseline with a two-frame context.

Input at position k: [n_k / sigma_n, (n_k - n_{k-1}) / sigma_delta, (v - mean_x v) / sigma_v] with
n_{-1} := n_0; ``Linear(3*NX -> d)``, pre-LayerNorm residual blocks (Linear d -> 4d, GELU,
Linear 4d -> d), a final LayerNorm and the same heads, normaliser and loss as the GPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from lqm import constants as C
from lqm.model.gpt import init_backbone_
from lqm.model.heads import Head
from lqm.model.normaliser import Normaliser


@dataclass
class MLPConfig:
    """Hyperparameters of :class:`MLP`.

    ``d_model = 312`` gives 4.85M parameters, matched to the 4.89M GPT (d = 320 would give 5.09M).
    """

    d_model: int = 312
    n_layers: int = 6
    mlp_ratio: int = 4


class ResidualMLPBlock(nn.Module):
    """x + Linear(GELU(Linear(LayerNorm(x))))."""

    def __init__(self, d_model: int, mlp_ratio: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc", nn.Linear(d_model, mlp_ratio * d_model))
        self.mlp.add_module("act", nn.GELU())
        self.mlp.add_module("out", nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class MLP(nn.Module):
    """Per-position residual MLP on the current frame and the last one-frame change.

    Parameters
    ----------
    cfg : MLPConfig
        Hyperparameters.
    head : str
        Head kind (``lqm.model.heads.HEADS``).
    normaliser : Normaliser
        Input/output scales (from the training split).
    """

    model_type = "mlp"
    context = 2
    max_len = None

    def __init__(self, cfg: MLPConfig, head: str, normaliser: Normaliser) -> None:
        super().__init__()
        self.cfg = cfg
        self.normaliser = normaliser
        self.embed = nn.Linear(3 * C.NX, cfg.d_model)
        self.blocks = nn.ModuleList(ResidualMLPBlock(cfg.d_model, cfg.mlp_ratio) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = Head(head, cfg.d_model, normaliser)
        init_backbone_(self, cfg.n_layers)
        self.head.zero_init()

    def forward(
        self, n: torch.Tensor, v: torch.Tensor, return_attn: bool = False, pos_offset: int = 0
    ) -> dict[str, Any]:
        """Predict the next frame at every position (same signature as ``GPT.forward``).

        ``return_attn`` and ``pos_offset`` are accepted for interface compatibility and ignored
        (``attn`` is always None). ``n_{-1} := n[:, 0]`` for the first position.
        """
        nm = self.normaliser
        n_prev = torch.cat([n[:, :1], n[:, :-1]], dim=1)
        dn = n - n_prev  # in the input dtype (float64 at inference) before the cast
        v_c = (v - v.mean(dim=-1, keepdim=True)) / nm.sigma_v
        feats = torch.cat(
            [(n / nm.sigma_n).float(), (dn / nm.sigma_delta).float(), v_c.float()[:, None, :].expand_as(n)],
            dim=-1,
        )
        x = self.embed(feats)
        resid = [x]
        for block in self.blocks:
            x = block(x)
            resid.append(x)
        n_next, J = self.head(self.ln_f(x), n.double())
        return {"n_next": n_next, "J": J, "resid": resid, "attn": None}
