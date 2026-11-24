"""Density and trajectory metrics for LQM-1e evaluation.

All functions take NumPy arrays, compute in float64 and treat the **last axis as space** (NX
points, spacing ``dx``) unless stated otherwise; leading axes are broadcast (e.g. [B, T, NX] in,
[B, T] out). Atomic units throughout: x in Bohr, t in ħ/Hartree, ω in Hartree/ħ.

Integrals are plain Riemann sums ``sum(f) * dx`` over all grid points including the walls, the same
convention as the dataset's normalisation ``sum(n) * DX = 1``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from lqm import constants as C


def _f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def l1(pred: np.ndarray, true: np.ndarray, dx: float = C.DX) -> np.ndarray:
    """L1 density error ∫|n̂ − n| dx (dimensionless; 2 = disjoint supports, 0 = identical)."""
    return np.abs(_f64(pred) - _f64(true)).sum(axis=-1) * dx


def rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Relative L2 error ‖n̂ − n‖₂ / ‖n‖₂ (dimensionless; the grid spacing cancels)."""
    diff = _f64(pred) - _f64(true)
    return np.sqrt((diff**2).sum(axis=-1) / (_f64(true) ** 2).sum(axis=-1))


def norm_error(n: np.ndarray, dx: float = C.DX) -> np.ndarray:
    """Signed normalisation error ∫n dx − 1."""
    return _f64(n).sum(axis=-1) * dx - 1.0


def negative_mass(n: np.ndarray, dx: float = C.DX) -> np.ndarray:
    """Total negative density ∫max(−n, 0) dx (≥ 0; zero for a physical density)."""
    return np.maximum(-_f64(n), 0.0).sum(axis=-1) * dx


def dipole(n: np.ndarray, dx: float = C.DX, x: np.ndarray = C.X) -> np.ndarray:
    """Dipole moment ⟨x⟩ = ∫x n dx in Bohr (not divided by ∫n dx, so norm drift shows up here too)."""
    return (_f64(n) * x).sum(axis=-1) * dx


def dipole_spectrum(d: np.ndarray, dt: float = C.DT, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed power spectrum of a dipole time series.

    P(ω) = |dt Σ_k w_k (d_k − d̄) e^{iωt_k}|², with d̄ the time mean (removes the DC peak from the
    field-shifted equilibrium) and w the symmetric Hann window (``np.hanning``).

    Parameters
    ----------
    d : np.ndarray
        [..., T] dipole ⟨x⟩(t_k) in Bohr; **the last axis is time** here.
    dt : float
        Frame spacing in ħ/Hartree.
    n_fft : int, optional
        FFT length (zero padding, default T). Padding interpolates the spectrum but does not
        improve the resolution 2π/(T dt).

    Returns
    -------
    omega : np.ndarray
        [n_fft//2 + 1] angular frequencies in Hartree/ħ (directly comparable to level spacings).
    power : np.ndarray
        [..., n_fft//2 + 1] spectral power in Bohr² (ħ/Hartree)².
    """
    d = _f64(d)
    nt = d.shape[-1]
    n_fft = nt if n_fft is None else int(n_fft)
    w = np.hanning(nt)
    spec = np.fft.rfft((d - d.mean(axis=-1, keepdims=True)) * w, n=n_fft, axis=-1) * dt
    omega = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt)
    return omega, np.abs(spec) ** 2


def time_to_threshold(err: np.ndarray, thr: float = 0.1, dt: float = C.DT) -> np.ndarray:
    """Time at which an error trace first exceeds a threshold.

    Parameters
    ----------
    err : np.ndarray
        [..., T] error at frames t_k = k * dt (**last axis is time**), e.g. rollout L1.
    thr : float
        Threshold; "exceeds" means ``err > thr`` strictly.
    dt : float
        Frame spacing.

    Returns
    -------
    np.ndarray
        [...] first t_k with ``err[k] > thr``. Conventions: a non-finite error (NaN/inf from a
        diverged rollout) counts as exceeding the threshold at that frame; a trace that never
        exceeds it within the horizon returns ``+inf`` (right-censored at ``(T-1) * dt``).
    """
    err = _f64(err)
    exceeded = ~(err <= thr)  # True for err > thr and for NaN
    first = np.argmax(exceeded, axis=-1)
    return np.where(exceeded.any(axis=-1), first * dt, np.inf)


class CI(NamedTuple):
    """Sample mean with a bootstrap confidence interval [lo, hi] (floats, or arrays for 2-D input)."""

    mean: np.ndarray | float
    lo: np.ndarray | float
    hi: np.ndarray | float


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Percentile-bootstrap confidence interval for the mean over axis 0.

    Resamples rows (axis 0, e.g. trajectories) with replacement. A resample is represented by
    multinomial counts c [n_boot, N], so the bootstrap means are one matrix product c @ x / N; this
    makes per-time CI bands for [N, T] input cheap and uses the same resamples for every column.

    Parameters
    ----------
    x : np.ndarray
        [N] or [N, ...] samples; must be finite (NaN/inf propagate to NaN bounds).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        1 − confidence level (0.05 ⇒ 95% interval).
    seed : int
        Seed of the resampling RNG (results are bit-reproducible for a fixed seed).

    Returns
    -------
    CI
        ``mean`` = sample mean, ``lo``/``hi`` = alpha/2 and 1 − alpha/2 quantiles of the bootstrap
        means. Scalars for 1-D input, arrays of shape x.shape[1:] otherwise.
    """
    x = _f64(x)
    n = x.shape[0]
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one sample")
    flat = x.reshape(n, -1)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    boot = counts @ flat / n  # [n_boot, M]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    mean = flat.mean(axis=0)
    shape = x.shape[1:]
    if not shape:
        return CI(float(mean[0]), float(lo[0]), float(hi[0]))
    return CI(mean.reshape(shape), lo.reshape(shape), hi.reshape(shape))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Bootstrap CI for mean(a − b) where a[i] and b[i] are measured on the same trajectory i.

    Resampling the paired differences cancels the trajectory-to-trajectory variation shared by both
    models, so the interval is usually much narrower than comparing two unpaired CIs.
    """
    a, b = _f64(a), _f64(b)
    if a.shape != b.shape:
        raise ValueError(f"paired samples need equal shapes, got {a.shape} and {b.shape}")
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)
