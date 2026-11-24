"""Free autoregressive rollout and teacher-forced evaluation of a ``Stepper``.

Both are device-agnostic: they run on the device of the input tensors (the stepper must accept
tensors there) and return float64 tensors on that device.
"""

from __future__ import annotations

import torch

from lqm.model.stepper import Stepper


def _check_output(pred: torch.Tensor, shape: tuple[int, ...], who: str) -> None:
    if pred.dtype != torch.float64:
        raise TypeError(f"{who} returned {pred.dtype}; the Stepper protocol requires float64")
    if tuple(pred.shape) != shape:
        raise ValueError(f"{who} returned shape {tuple(pred.shape)}, expected {shape}")


@torch.no_grad()
def rollout(stepper: Stepper, n0: torch.Tensor, v: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Free autoregressive rollout from the ground state.

    Frame k+1 is predicted from the model's own frames 0..k (truncated to the last
    ``stepper.context`` frames when that is not None); no exact data are used after ``n0``.

    Parameters
    ----------
    stepper : Stepper
        Model to roll out.
    n0 : torch.Tensor
        [B, NX] initial density (cast to float64).
    v : torch.Tensor
        [B, NX] potential for t > 0 (cast to float64, moved to ``n0``'s device).
    n_steps : int
        Number of steps (frames predicted).

    Returns
    -------
    torch.Tensor
        [B, n_steps + 1, NX] float64 with ``out[:, 0] = n0`` and ``out[:, k]`` the prediction of t_k.

    Notes
    -----
    The history handed to ``predict_next`` is a view into the output buffer; steppers must not
    modify it in place.
    """
    n0 = torch.as_tensor(n0, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64).to(n0.device)
    if n0.ndim != 2 or v.shape != n0.shape:
        raise ValueError(f"expected n0 and v of shape [B, NX], got {tuple(n0.shape)} and {tuple(v.shape)}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    out = torch.empty((n0.shape[0], n_steps + 1, n0.shape[1]), dtype=torch.float64, device=n0.device)
    out[:, 0] = n0
    for k in range(n_steps):
        start = 0 if stepper.context is None else max(0, k + 1 - stepper.context)
        pred = stepper.predict_next(out[:, start : k + 1], v)
        _check_output(pred, tuple(n0.shape), f"{stepper.name}.predict_next")
        out[:, k + 1] = pred
    return out


@torch.no_grad()
def teacher_forced(stepper: Stepper, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """One-step predictions from exact histories.

    Parameters
    ----------
    stepper : Stepper
        Model to evaluate.
    n : torch.Tensor
        [B, T, NX] exact frames t_0 .. t_{T-1} (cast to float64).
    v : torch.Tensor
        [B, NX] potential for t > 0.

    Returns
    -------
    torch.Tensor
        [B, T, NX] float64; ``out[:, k]`` predicts ``n[:, k+1]`` from exact frames 0..k (so the last
        position predicts a frame not contained in ``n``; pass ``n[:, :-1]`` to score against ``n``).
    """
    n = torch.as_tensor(n, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64).to(n.device)
    pred = stepper.predict_teacher_forced(n, v)
    _check_output(pred, tuple(n.shape), f"{stepper.name}.predict_teacher_forced")
    return pred
