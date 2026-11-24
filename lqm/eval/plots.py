"""Reusable matplotlib helpers for LQM-1e evaluation figures.

Axes-level functions draw into a given ``ax`` and return it; figure-level functions return the
figure. Nothing here calls ``plt.show()``. Call ``apply_style()`` once in a script for the shared
look, and use ``save_figure`` to write PNGs. Atomic units on every axis.

Colours: models get fixed categorical slots in order of first appearance (a colourblind-checked
eight-hue order); the baselines ``persistence``/``linear``/``oracle`` and the exact data always get
neutral inks with distinct line styles, so they read as references rather than competitors.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from lqm import constants as C
from lqm.eval import metrics as M

logger = logging.getLogger(__name__)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
REFERENCE_STYLES: dict[str, dict[str, object]] = {
    "exact": {"color": INK, "linestyle": "-"},
    "oracle": {"color": INK, "linestyle": "-"},
    "persistence": {"color": INK_MUTED, "linestyle": "--"},
    "linear": {"color": INK_SECONDARY, "linestyle": ":"},
}
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "lqm_density", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIFF_CMAP = LinearSegmentedColormap.from_list("lqm_diff", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1716"])

STYLE: dict[str, object] = {
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "lines.linewidth": 1.5,
    "image.cmap": "lqm_density",
}

T_LABEL = "t (a.u.)"
X_LABEL = "x (a.u.)"


def apply_style() -> None:
    """Set the shared rcParams (call once from a script; library functions do not mutate rcParams)."""
    if "lqm_density" not in mpl.colormaps:
        mpl.colormaps.register(DENSITY_CMAP)
    mpl.rcParams.update(STYLE)


def line_styles(names: Sequence[str]) -> dict[str, dict[str, object]]:
    """Fixed colour/linestyle per series name: references get neutral inks, models categorical slots.

    Raises
    ------
    ValueError
        If more than eight non-reference models are requested (facet instead of inventing hues).
    """
    models = [n for n in dict.fromkeys(names) if n not in REFERENCE_STYLES]
    if len(models) > len(CATEGORICAL):
        raise ValueError(f"{len(models)} models > {len(CATEGORICAL)} categorical slots; split into panels")
    styles = {n: {"color": c, "linestyle": "-"} for n, c in zip(models, CATEGORICAL, strict=False)}
    styles.update({n: dict(REFERENCE_STYLES[n]) for n in names if n in REFERENCE_STYLES})
    return styles


def _positive(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0, a, np.nan)


def plot_metric_vs_time(
    ax: Axes,
    t: np.ndarray,
    curves: Mapping[str, np.ndarray],
    ylabel: str,
    logy: bool = True,
    ci: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> Axes:
    """Mean over trajectories vs time for several models, with pointwise 95% bootstrap bands.

    Parameters
    ----------
    ax : Axes
        Target axes.
    t : np.ndarray
        [T] frame times.
    curves : mapping
        ``{model name: [N, T] per-trajectory values}``.
    ylabel : str
        Y-axis label (with units).
    logy : bool
        Log y-axis; non-positive means/bounds are masked (e.g. zero error at t = 0).
    ci : bool
        Draw the bootstrap band (trajectories resampled jointly across times).
    n_boot, seed : int
        Bootstrap settings.
    """
    styles = line_styles(list(curves))
    mask = _positive if logy else np.asarray
    for name, vals in curves.items():
        if ci:
            mean, lo, hi = M.bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            ax.fill_between(t, mask(lo), mask(hi), color=styles[name]["color"], alpha=0.2, linewidth=0)
        else:
            mean = np.mean(vals, axis=0)
        ax.plot(t, mask(mean), label=name, **styles[name])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(ylabel)
    if len(curves) > 1:
        ax.legend()
    return ax


def plot_error_vs_time(
    ax: Axes,
    t: np.ndarray,
    curves: Mapping[str, np.ndarray],
    threshold: float | None = 0.1,
    ylabel: str = r"rollout $L_1 = \int|\hat n - n|\,dx$",
    **kwargs: object,
) -> Axes:
    """Rollout error vs time (mean ± 95% CI) with an optional horizontal threshold line."""
    plot_metric_vs_time(ax, t, curves, ylabel=ylabel, **kwargs)
    if threshold is not None:
        ax.axhline(threshold, color=INK_MUTED, linewidth=0.8, linestyle="-.", zorder=0)
    return ax


def plot_norm_diagnostics(
    axes: Sequence[Axes],
    t: np.ndarray,
    norm_error: Mapping[str, np.ndarray],
    negative_mass: Mapping[str, np.ndarray],
    **kwargs: object,
) -> Sequence[Axes]:
    """Two panels: mean |∫n̂dx − 1| and mean negative mass ∫max(−n̂,0)dx vs time (log scale, 95% CI)."""
    abs_norm = {k: np.abs(v) for k, v in norm_error.items()}
    plot_metric_vs_time(axes[0], t, abs_norm, ylabel=r"$|\int\hat n\,dx - 1|$", **kwargs)
    plot_metric_vs_time(axes[1], t, negative_mass, ylabel=r"negative mass $\int\max(-\hat n,0)\,dx$", **kwargs)
    return axes


def plot_dipole(ax: Axes, t: np.ndarray, d_true: np.ndarray, d_pred: Mapping[str, np.ndarray]) -> Axes:
    """Dipole traces ⟨x⟩(t) of one trajectory: exact (black) and each model's rollout."""
    styles = line_styles(["exact", *d_pred])
    ax.plot(t, d_true, label="exact", **styles["exact"])
    for name, d in d_pred.items():
        ax.plot(t, d, label=name, **styles[name])
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(r"$\langle x\rangle$ (a.u.)")
    ax.legend()
    return ax


def plot_dipole_spectrum(
    ax: Axes,
    t: np.ndarray,
    d_true: np.ndarray,
    d_pred: Mapping[str, np.ndarray],
    n_fft: int = 2048,
    omega_max: float = 2.0,
) -> Axes:
    """Hann-windowed dipole power spectra (``metrics.dipole_spectrum``) of one trajectory vs ω in Hartree.

    The resolution is 2π/(T Δt) ≈ 0.16 Ha for 201 frames; ``n_fft`` zero padding only interpolates.
    The x-range stops at ``omega_max`` (the frame Nyquist frequency π/Δt = 15.7 Ha is far above the
    0.3-0.6 Ha dipole frequencies of the dataset).
    """
    dt = float(t[1] - t[0])
    styles = line_styles(["exact", *d_pred])
    for name, d in {"exact": d_true, **d_pred}.items():
        omega, p = M.dipole_spectrum(d, dt=dt, n_fft=n_fft)
        ax.plot(omega, p, label=name, **styles[name])
    ax.set_xlim(0, min(omega_max, np.pi / dt))
    ax.set_xlabel(r"$\omega$ (Ha)")
    ax.set_ylabel(r"$|\tilde d(\omega)|^2$ (a.u.)")
    ax.legend()
    return ax


def plot_heatmaps(
    n_true: np.ndarray,
    n_pred: np.ndarray,
    t: np.ndarray,
    x: np.ndarray = C.X,
    title: str | None = None,
) -> Figure:
    """Side-by-side (x, t) maps of the exact density, the prediction and their difference.

    The first two panels share one colour scale; the difference uses a symmetric diverging scale
    centred on zero.

    Parameters
    ----------
    n_true, n_pred : np.ndarray
        [T, NX] densities of one trajectory.
    t : np.ndarray
        [T] frame times.
    x : np.ndarray
        [NX] grid.
    title : str, optional
        Figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout="constrained")
    extent = (x[0], x[-1], t[0], t[-1])
    vmax = float(max(np.nanmax(n_true), np.nanmax(n_pred)))
    vmin = float(min(0.0, np.nanmin(n_pred)))
    diff = n_pred - n_true
    dmax = float(np.nanmax(np.abs(diff))) or 1.0
    common = {"origin": "lower", "aspect": "auto", "extent": extent, "interpolation": "nearest"}
    im0 = axes[0].imshow(n_true, cmap=DENSITY_CMAP, vmin=vmin, vmax=vmax, **common)
    axes[1].imshow(n_pred, cmap=DENSITY_CMAP, vmin=vmin, vmax=vmax, **common)
    im2 = axes[2].imshow(diff, cmap=DIFF_CMAP, vmin=-dmax, vmax=dmax, **common)
    for ax, label in zip(axes, ("exact $n$", r"predicted $\hat n$", r"$\hat n - n$"), strict=True):
        ax.set_title(label)
        ax.set_xlabel(X_LABEL)
        ax.grid(False)
    axes[0].set_ylabel(T_LABEL)
    fig.colorbar(im0, ax=axes[:2], label="density (a.u.)", shrink=0.9)
    fig.colorbar(im2, ax=axes[2], label="difference (a.u.)", shrink=0.9)
    if title:
        fig.suptitle(title)
    return fig


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save ``fig`` (format from the suffix, parent directories created) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)
    return path
