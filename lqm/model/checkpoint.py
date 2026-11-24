"""Model construction, checkpoint I/O and the ``Stepper`` adapter used by evaluation.

Checkpoint format (``torch.save`` dict, loadable with ``weights_only=True``)::

    {model_state, model_cfg, train_cfg, head, model_type ("gpt" | "mlp"),
     normaliser {sigma_n, sigma_v, sigma_delta, sigma_J}, step, val_loss}
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lqm.model.gpt import GPT, GPTConfig
from lqm.model.mlp import MLP, MLPConfig
from lqm.model.normaliser import Normaliser

logger = logging.getLogger(__name__)

MODEL_TYPES: tuple[str, ...] = ("gpt", "mlp")


def build_model(model_type: str, head: str, model_cfg: dict[str, Any], normaliser: Normaliser) -> GPT | MLP:
    """Construct a freshly initialised model (head zero-initialised => persistence).

    Parameters
    ----------
    model_type : str
        ``"gpt"`` or ``"mlp"``.
    head : str
        Head kind (``lqm.model.heads.HEADS``).
    model_cfg : dict
        Keyword arguments of ``GPTConfig`` or ``MLPConfig``.
    normaliser : Normaliser
        Scales from the training split.
    """
    if model_type == "gpt":
        return GPT(GPTConfig(**model_cfg), head, normaliser)
    if model_type == "mlp":
        return MLP(MLPConfig(**model_cfg), head, normaliser)
    raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def save_checkpoint(path: str | Path, model: GPT | MLP, train_cfg: dict[str, Any], **extra: Any) -> None:
    """Save model weights with everything needed to rebuild it.

    Parameters
    ----------
    path : str or Path
        Output ``.pt`` file.
    model : GPT or MLP
        The model.
    train_cfg : dict
        Training configuration (JSON-serialisable).
    **extra
        Additional plain-Python metadata (e.g. ``step``, ``val_loss``).
    """
    ckpt = {
        "model_state": model.state_dict(),
        "model_cfg": asdict(model.cfg),
        "train_cfg": train_cfg,
        "head": model.head.kind,
        "model_type": model.model_type,
        "normaliser": model.normaliser.to_dict(),
        **extra,
    }
    torch.save(ckpt, path)


def load_model(ckpt_path: str | Path, device: torch.device | str) -> tuple[GPT | MLP, dict[str, Any]]:
    """Rebuild a model from a checkpoint (in eval mode). Returns ``(model, checkpoint_dict)``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_type"], ckpt["head"], ckpt["model_cfg"], Normaliser.from_dict(ckpt["normaliser"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


class ModelStepper:
    """``lqm.model.stepper.Stepper`` adapter around a trained GPT or MLP.

    Inputs may live on any device; they are moved to the model's device and the float64 result is
    returned on the input's device. Batches are processed in chunks of ``max_batch``.

    Attributes
    ----------
    name : str
        Run name (the checkpoint's directory name).
    context : int
        Frames used per prediction: ``max_len`` (full GPT), ``W`` (windowed GPT) or 2 (MLP).
    model : GPT or MLP
        The underlying model (eval mode), e.g. for residual-stream / attention analysis.
    """

    def __init__(self, model: GPT | MLP, name: str, max_batch: int = 256) -> None:
        self.model = model.eval()
        self.name = name
        self.context = model.context
        self.max_batch = max_batch
        self.device = next(model.parameters()).device
        self._warned = False

    def _chunked(self, fn: Any, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out_device = n.device
        n = n.to(self.device, torch.float64)
        v = v.to(self.device, torch.float64)
        parts = [fn(n[s : s + self.max_batch], v[s : s + self.max_batch]) for s in range(0, n.shape[0], self.max_batch)]
        return torch.cat(parts, dim=0).to(out_device)

    @torch.no_grad()
    def _next(self, n_hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k = n_hist.shape[1]
        start = max(0, k - self.context)
        span = k - start
        max_len = self.model.max_len
        pos_offset = start if max_len is None else min(start, max_len - span)
        if pos_offset != start and not self._warned:
            # Only reachable for histories longer than the trained positions (k > max_len).
            logger.warning("%s: history of %d frames exceeds max_len %s; positions are shifted", self.name, k, max_len)
            self._warned = True
        return self.model(n_hist[:, start:], v, pos_offset=pos_offset)["n_next"][:, -1]

    @torch.no_grad()
    def predict_next(self, n_hist: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """[B, k, NX] history -> [B, NX] float64 prediction of frame k (uses the last ``context`` frames)."""
        return self._chunked(self._next, n_hist, v)

    @torch.no_grad()
    def _teacher_forced(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        n_frames = n.shape[1]
        max_len = self.model.max_len
        if max_len is None or n_frames <= max_len:
            return self.model(n, v)["n_next"]
        head = self.model(n[:, :max_len], v)["n_next"]
        tail = [self._next(n[:, : k + 1], v) for k in range(max_len, n_frames)]
        return torch.cat([head, torch.stack(tail, dim=1)], dim=1)

    @torch.no_grad()
    def predict_teacher_forced(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """[B, T, NX] exact frames -> [B, T, NX] float64, ``out[:, k]`` predicts ``n[:, k+1]``."""
        return self._chunked(self._teacher_forced, n, v)


def load_stepper(ckpt_path: str | Path, device: torch.device | str) -> ModelStepper:
    """Load a checkpoint as a ``Stepper`` named after its run directory."""
    model, _ = load_model(ckpt_path, device)
    return ModelStepper(model, name=Path(ckpt_path).resolve().parent.name)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
