"""Training loop: AdamW + warmup/cosine, teacher forcing, parity augmentation, optional pushforward,
periodic validation, best/last checkpoints and a CSV log. Full float32 (no autocast/TF32).
"""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from lqm.model.checkpoint import save_checkpoint
from lqm.train.data import TrajectoryData, epoch_batches, parity_flip
from lqm.train.losses import l1_per_frame, pushforward_loss, scaled_mse, teacher_forced_loss

logger = logging.getLogger(__name__)

LOG_COLUMNS: tuple[str, ...] = (
    "step",
    "epoch",
    "lr",
    "train_loss",
    "val_loss",
    "val_l1",
    "persistence_l1",
    "wall_s",
    "ms_per_step",
)


@dataclass
class TrainConfig:
    """Optimisation and schedule settings.

    Attributes
    ----------
    batch_size : int
        Trajectories per step (each contributes 200 teacher-forced positions).
    lr, betas, weight_decay : float, tuple, float
        AdamW settings; weight decay applies to matrices (dim >= 2) only.
    warmup_steps : int
        Linear warmup length; cosine decay to 0 over the remaining steps.
    max_epochs : int
        Training length in epochs, used when ``max_steps`` is None.
    max_steps : int or None
        Training length in optimiser steps (overrides ``max_epochs``).
    grad_clip : float
        Global gradient-norm clip.
    eval_every : int
        Validation interval in steps.
    patience : int
        Early stopping after this many validations without a new best val loss.
    p_parity : float
        Per-trajectory probability of the x -> -x augmentation.
    pushforward_frac : float
        Fraction of batches trained with the two-pass pushforward loss.
    eval_batch_size : int
        Trajectories per validation forward pass.
    """

    batch_size: int = 64
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 60
    max_steps: int | None = None
    grad_clip: float = 1.0
    eval_every: int = 375
    patience: int = 10
    p_parity: float = 0.5
    pushforward_frac: float = 0.0
    eval_batch_size: int = 256

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Build from a JSON dict (unknown keys are an error)."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown train config keys: {sorted(unknown)}")
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)


def lr_at(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """Linear warmup over ``warmup_steps``, then cosine decay from ``lr`` to 0 at ``total_steps``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices (Linear weights, positional embedding) only."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: TrajectoryData, batch_size: int) -> dict[str, float]:
    """Teacher-forced validation over a whole split.

    Returns
    -------
    dict
        ``val_loss`` (training loss), ``val_l1`` (mean over trajectories and frames of the one-step
        L1 error) and ``persistence_l1`` (the same for n_hat_{k+1} = n_k).
    """
    was_training = model.training
    model.eval()
    sigma = model.normaliser.sigma_delta
    sq = l1 = pers = 0.0
    n_pos = 0
    for s in range(0, len(data), batch_size):
        n, v = data.n[s : s + batch_size], data.v[s : s + batch_size]
        out = model(n[:, :-1], v)["n_next"]
        target = n[:, 1:]
        sq += scaled_mse(out, target, sigma).item() * out.shape[0] * out.shape[1]
        l1 += l1_per_frame(out, target).sum().item()
        pers += l1_per_frame(n[:, :-1], target).sum().item()
        n_pos += out.shape[0] * out.shape[1]
    model.train(was_training)
    return {"val_loss": sq / n_pos, "val_l1": l1 / n_pos, "persistence_l1": pers / n_pos}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(
    model: torch.nn.Module,
    train_data: TrajectoryData,
    val_data: TrajectoryData,
    cfg: TrainConfig,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Train ``model`` and write ``log.csv``, ``best.pt`` and ``last.pt`` to ``run_dir``.

    Parameters
    ----------
    model : GPT or MLP
        Model on the same device as the data.
    train_data, val_data : TrajectoryData
        Splits resident on the training device.
    cfg : TrainConfig
        Settings.
    run_dir : str or Path
        Existing output directory.

    Returns
    -------
    dict
        ``best_val_loss``, ``best_step``, ``final_step``, ``stopped_early``, ``mean_ms_per_step``.
    """
    torch.set_float32_matmul_precision("highest")
    run_dir = Path(run_dir)
    device = train_data.n.device
    steps_per_epoch = len(train_data) // cfg.batch_size
    total_steps = cfg.max_steps if cfg.max_steps is not None else cfg.max_epochs * steps_per_epoch
    train_cfg = asdict(cfg)
    opt = make_optimizer(model, cfg)
    batches = epoch_batches(len(train_data), cfg.batch_size, device)
    logger.info("training %d steps (%d steps/epoch), %s", total_steps, steps_per_epoch, train_cfg)

    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)

    best_val, best_step, bad_evals, stopped_early = math.inf, 0, 0, False
    t_start = time.perf_counter()
    train_time, train_steps_total = 0.0, 0

    def validate(step: int, lr: float, train_loss: float, ms_per_step: float) -> bool:
        """Log, checkpoint, and return True if training should stop early."""
        nonlocal best_val, best_step, bad_evals
        metrics = evaluate(model, val_data, cfg.eval_batch_size)
        row = {
            "step": step,
            "epoch": step / steps_per_epoch,
            "lr": lr,
            "train_loss": train_loss,
            **metrics,
            "wall_s": time.perf_counter() - t_start,
            "ms_per_step": ms_per_step,
        }
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([row[c] for c in LOG_COLUMNS])
        logger.info(
            "step %d ep %.2f lr %.2e train %.4e val %.4e val_l1 %.3e (persistence %.3e) %.1f ms/step",
            step,
            row["epoch"],
            lr,
            train_loss,
            metrics["val_loss"],
            metrics["val_l1"],
            metrics["persistence_l1"],
            ms_per_step,
        )
        meta = {"step": step, "val_loss": metrics["val_loss"]}
        if metrics["val_loss"] < best_val:
            best_val, best_step, bad_evals = metrics["val_loss"], step, 0
            save_checkpoint(run_dir / "best.pt", model, train_cfg, **meta)
        else:
            bad_evals += 1
        save_checkpoint(run_dir / "last.pt", model, train_cfg, **meta)
        return bad_evals >= cfg.patience

    model.train()
    validate(0, 0.0, math.nan, math.nan)
    step = 0
    running_loss = torch.zeros((), device=device, dtype=torch.float64)
    n_since_eval = 0
    _sync(device)
    t_seg = time.perf_counter()
    while step < total_steps:
        _, idx = next(batches)
        lr = lr_at(step, cfg, total_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        n, v = parity_flip(train_data.n[idx], train_data.v[idx], cfg.p_parity)
        if cfg.pushforward_frac > 0 and random.random() < cfg.pushforward_frac:
            loss = pushforward_loss(model, n, v)
        else:
            loss = teacher_forced_loss(model, n, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        running_loss += loss.detach()
        n_since_eval += 1
        if step % cfg.eval_every == 0 or step == total_steps:
            _sync(device)
            seg = time.perf_counter() - t_seg
            train_time += seg
            train_steps_total += n_since_eval
            stop = validate(step, lr, running_loss.item() / n_since_eval, 1e3 * seg / n_since_eval)
            running_loss.zero_()
            n_since_eval = 0
            if stop:
                logger.info("early stopping at step %d (best val %.4e at step %d)", step, best_val, best_step)
                stopped_early = True
                break
            _sync(device)
            t_seg = time.perf_counter()

    summary = {
        "best_val_loss": best_val,
        "best_step": best_step,
        "final_step": step,
        "stopped_early": stopped_early,
        "mean_ms_per_step": 1e3 * train_time / max(1, train_steps_total),
    }
    logger.info("done: %s", summary)
    return summary
