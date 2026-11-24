"""GPT-style causal transformer over density frames.

Token k is ``Linear(2*NX -> d)`` of [n_k / sigma_n, (v - mean_x v) / sigma_v] plus a learned absolute
positional embedding; pre-LayerNorm blocks (multi-head causal self-attention, 4x GELU MLP), a final
LayerNorm, and a density-update head (``lqm.model.heads``). No dropout. Backbone in float32, head
update in float64.

An optional attention window W restricts the output at position k to depend on frames k-W+1..k only
(W = 2 gives the model the same information as the MLP baseline). A per-layer sliding-window mask
would not do this: through L stacked layers the receptive field grows to 1 + L(W-1) frames. The
windowed model therefore runs the transformer separately on each window [k-W+1..k] (at its absolute
positions, causal inside the window) and keeps the last token, at W times the token count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from lqm import constants as C
from lqm.model.heads import Head
from lqm.model.normaliser import Normaliser


@dataclass
class GPTConfig:
    """Hyperparameters of :class:`GPT`.

    Attributes
    ----------
    d_model, n_layers, n_heads, mlp_ratio : int
        Width, depth, attention heads and MLP expansion factor.
    max_len : int
        Number of learned positions (= frames of input context).
    window : int or None
        Context window W (the output at k depends on frames k-W+1..k only); None = full causal context.
    """

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    max_len: int = C.NT - 1
    window: int | None = None


def causal_mask(T: int, device: torch.device | str) -> torch.Tensor:
    """Boolean [T, T] mask, True where query position q may attend to key position k (k <= q)."""
    q = torch.arange(T, device=device)
    return q[None, :] <= q[:, None]


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a fast (SDPA) path and an explicit-weights path."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, return_attn: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend.

        Parameters
        ----------
        x : torch.Tensor
            [B, T, d] input.
        mask : torch.Tensor or None
            Boolean mask broadcastable to [B, H, T, T] (True = may attend); None = plain causal.
        return_attn : bool
            If True, compute the softmax explicitly and also return the weights [B, H, T, T].
        """
        B, T, d = x.shape
        hd = d // self.n_heads
        q, k, v = (t.view(B, T, self.n_heads, hd).transpose(1, 2) for t in self.qkv(x).split(d, dim=-1))
        weights = None
        if return_attn:
            if mask is None:
                mask = causal_mask(T, x.device)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
            y = weights @ v
        elif mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, d)
        return self.out(y), weights


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc", nn.Linear(d_model, mlp_ratio * d_model))
        self.mlp.add_module("act", nn.GELU())
        self.mlp.add_module("out", nn.Linear(mlp_ratio * d_model, d_model))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, return_attn: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        a, weights = self.attn(self.ln1(x), mask, return_attn)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, weights


def init_backbone_(model: nn.Module, n_layers: int) -> None:
    """GPT-2 initialisation: N(0, 0.02) weights, zero biases, residual output projections scaled
    by 1/sqrt(2 * n_layers). The head is re-zeroed afterwards by the caller."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
    for name, p in model.named_parameters():
        if name.endswith(("attn.out.weight", "mlp.out.weight")):
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))


class GPT(nn.Module):
    """Causal transformer that predicts frame k+1 at every position k.

    Parameters
    ----------
    cfg : GPTConfig
        Hyperparameters.
    head : str
        Head kind (``lqm.model.heads.HEADS``).
    normaliser : Normaliser
        Input/output scales (from the training split).
    """

    model_type = "gpt"

    def __init__(self, cfg: GPTConfig, head: str, normaliser: Normaliser) -> None:
        super().__init__()
        self.cfg = cfg
        self.normaliser = normaliser
        self.max_len = cfg.max_len
        self.context = cfg.max_len if cfg.window is None else cfg.window
        self.embed = nn.Linear(2 * C.NX, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(cfg.max_len, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = Head(head, cfg.d_model, normaliser)
        init_backbone_(self, cfg.n_layers)
        nn.init.normal_(self.pos, std=0.02)
        self.head.zero_init()

    def forward(
        self, n: torch.Tensor, v: torch.Tensor, return_attn: bool = False, pos_offset: int = 0
    ) -> dict[str, Any]:
        """Predict the next frame at every position.

        Parameters
        ----------
        n : torch.Tensor
            [B, T, NX] frames in physical units (float32 or float64; a float64 input is used at
            full precision as the base of the head update).
        v : torch.Tensor
            [B, NX] potential for t > 0.
        return_attn : bool
            Also return the attention weights of every layer (slower, O(T^2) memory).
        pos_offset : int
            Absolute frame index of ``n[:, 0]`` (used by windowed inference on a suffix).

        Returns
        -------
        dict
            ``n_next`` [B, T, NX] float64; ``J`` [B, T, NFACE] float64 or None; ``resid`` list of
            n_layers + 1 tensors [B, T, d] (embedding output, then after each block); ``attn`` list
            of n_layers tensors [B, H, T, T] or None.
        """
        T = n.shape[1]
        if pos_offset + T > self.max_len:
            raise ValueError(f"positions {pos_offset}..{pos_offset + T - 1} exceed max_len {self.max_len}")
        nm = self.normaliser
        n_tok = (n / nm.sigma_n).float()
        v_tok = ((v - v.mean(dim=-1, keepdim=True)) / nm.sigma_v).float()
        tokens = torch.cat([n_tok, v_tok[:, None, :].expand(-1, T, -1)], dim=-1)
        if self.cfg.window is None:
            x = self.embed(tokens) + self.pos[pos_offset : pos_offset + T]
            resid = [x]
            attn: list[torch.Tensor] | None = [] if return_attn else None
            for block in self.blocks:
                x, weights = block(x, None, return_attn)
                resid.append(x)
                if attn is not None:
                    attn.append(weights)
        else:
            x, resid, attn = self._forward_windowed(tokens, pos_offset, return_attn)
        n_next, J = self.head(self.ln_f(x), n.double())
        return {"n_next": n_next, "J": J, "resid": resid, "attn": attn}

    def _forward_windowed(
        self, tokens: torch.Tensor, pos_offset: int, return_attn: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
        """Run every window [k-W+1..k] as its own causal sequence and keep its last token.

        Windows that would start before frame 0 are left-padded with copies of frame 0; the pads
        are masked out as keys (each pad attends only to itself, which keeps it finite). Returned
        tensors are those of the last token of each window: hidden state [B, T, d], residual stream
        [B, T, d] per layer, and the attention of query k over keys k-W+1..k scattered into a banded
        [B, H, T, T] matrix per layer.
        """
        W = self.cfg.window
        B, T, _ = tokens.shape
        dev = tokens.device
        j = torch.arange(W, device=dev)
        frame = torch.arange(T, device=dev)[:, None] - (W - 1) + j[None, :]  # [T, W] frame index
        valid = frame >= 0
        frame = frame.clamp_min(0)
        x = self.embed(tokens[:, frame]) + self.pos[pos_offset + frame]  # [B, T, W, d]
        x = x.reshape(B * T, W, -1)
        causal = j[None, :] <= j[:, None]  # [W(query), W(key)]
        mask = causal & (valid[:, None, :] | torch.eye(W, dtype=torch.bool, device=dev))  # [T, W, W]
        mask = mask.expand(B, T, W, W).reshape(B * T, 1, W, W)

        resid = [x[:, -1].reshape(B, T, -1)]
        attn: list[torch.Tensor] | None = [] if return_attn else None
        for block in self.blocks:
            # Always the explicit-softmax path: the fused SDPA kernels pad every W-token window to a
            # 64x64 tile, which made a W=2 training step ~5x slower than the full-context model.
            x, weights = block(x, mask, True)
            resid.append(x[:, -1].reshape(B, T, -1))
            if attn is not None:
                H = weights.shape[1]
                last = weights[:, :, -1, :].reshape(B, T, H, W).permute(0, 2, 1, 3)  # [B, H, T, W]
                band = last.new_zeros(B, H, T, T)
                band.scatter_add_(-1, frame.expand(B, H, T, W), last * valid)
                attn.append(band)
        return x[:, -1].reshape(B, T, -1), resid, attn
